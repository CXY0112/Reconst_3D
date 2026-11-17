"""Isolated Open3D geometry service. Its stdout protocol is prefixed JSON lines."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
from runtime_paths import activate_geometry_packages
activate_geometry_packages()
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from loop_checks import rigid_pose,validate_graph

REG = o3d.pipelines.registration
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def cloud(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points,dtype=float))
    return p


class Frame:
    def __init__(self, points):
        points = points[np.isfinite(points).all(axis=1)]
        self.base = cloud(points).voxel_down_sample(.0075)
        if len(self.base.points) < 200:
            raise ValueError('有效点云表面不足')
        # Isolated spikes are not allowed to define global shape descriptors.
        self.base,_ = self.base.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.5)
        self.levels = {}
        for size in (.03,.015,.0075):
            p = self.base.voxel_down_sample(size)
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=size*3,max_nn=30))
            self.levels[size] = p
        coarse = self.base.voxel_down_sample(.05)
        coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.10,max_nn=30))
        coarse.orient_normals_towards_camera_location(np.zeros(3))
        self.xyz = np.asarray(coarse.points).copy()
        feat = np.asarray(REG.compute_fpfh_feature(coarse,o3d.geometry.KDTreeSearchParamHybrid(radius=.25,max_nn=100)).data).T.copy()
        self.features = feat/(np.linalg.norm(feat,axis=1,keepdims=True)+1e-6)
        self.feature_tree = cKDTree(self.features)

    def save(self,path):
        arrays={'base':np.asarray(self.base.points),'xyz':self.xyz,'features':self.features}
        for size,p in self.levels.items():
            arrays[f'p{size}']=np.asarray(p.points);arrays[f'n{size}']=np.asarray(p.normals)
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        frame=cls.__new__(cls)
        with np.load(path) as data:
            frame.base=cloud(data['base']);frame.xyz=data['xyz'].copy();frame.features=data['features'].copy()
            frame.levels={}
            for size in (.03,.015,.0075):
                p=cloud(data[f'p{size}']);p.normals=o3d.utility.Vector3dVector(data[f'n{size}'])
                frame.levels[size]=p
        frame.feature_tree=cKDTree(frame.features);return frame

    def nbytes(self):
        # Include tree data/index and point/normal arrays; allocator overhead is measured separately.
        return self.xyz.nbytes+self.features.nbytes+self.feature_tree.data.nbytes+self.feature_tree.indices.nbytes+np.asarray(self.base.points).nbytes+sum(np.asarray(p.points).nbytes+np.asarray(p.normals).nbytes for p in self.levels.values())


def geometry_validation(source,target,pose,distance=.025):
    result = REG.evaluate_registration(source.base,target.base,distance,pose)
    inverse = REG.evaluate_registration(target.base,source.base,distance,np.linalg.inv(pose))
    if result.fitness < .25 or inverse.fitness < .15 or result.inlier_rmse > .015:
        raise ValueError(f'几何校验失败：双向重叠 {result.fitness:.0%}/{inverse.fitness:.0%}，残差 {result.inlier_rmse*1000:.1f}mm')
    # A nearly single plane cannot constrain all six pose degrees of freedom.
    normals = np.asarray(target.levels[.03].normals)
    eigen = np.linalg.eigvalsh(normals.T@normals/max(len(normals),1))
    if eigen[1] < .002:
        raise ValueError('几何结构接近单一平面，位置存在滑动歧义，请增加墙角或物体结构')
    return float(result.fitness),float(result.inlier_rmse)


def refine(source,target,initial):
    pose = initial.copy()
    for size,threshold in ((.03,.08),(.015,.045),(.0075,.025)):
        result = REG.registration_icp(source.levels[size],target.levels[size],threshold,pose,
                    REG.TransformationEstimationPointToPlane(REG.TukeyLoss(k=threshold/2)),
                    REG.ICPConvergenceCriteria(relative_fitness=1e-5,relative_rmse=1e-5,max_iteration=20))
        pose = result.transformation.copy()
    overlap,rmse = geometry_validation(source,target,pose)
    correction = pose@np.linalg.inv(initial)
    angle = np.arccos(np.clip((np.trace(correction[:3,:3])-1)/2,-1,1))
    if np.linalg.norm(correction[:3,3]) > .2 or angle > np.deg2rad(20):
        raise ValueError('ICP 修正幅度过大，几何粗定位不可靠')
    info = REG.get_information_matrix_from_point_clouds(source.base,target.base,.025,pose)
    return pose,overlap,rmse,np.asarray(info)


class GeometryEngine:
    def __init__(self, folder):
        self.folder = Path(folder)
        from frame_store import FrameStore
        self.frames=FrameStore(self.folder/'registration_frames',Frame.load)
        self.poses,self.edges = [],[]
        self.pending = None
        self.revision=0;self.loop_pending=None

    def log(self,event,**values):
        with (self.folder/'loop_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event,nodes=len(self.frames),**values),ensure_ascii=False)+'\n')

    def prepare(self, path):
        if self.loop_pending is not None:raise ValueError('Loop transaction in progress')
        started = time.perf_counter()
        frame = Frame(np.load(path))
        candidates = []
        for ref_id in self.frames.candidates(frame):
            reference_xyz,reference_features=self.frames.match_data(ref_id)
            distances,nearest = cKDTree(reference_features).query(frame.features,k=1,workers=1)
            _,reverse = frame.feature_tree.query(reference_features,k=1,workers=1)
            mutual = reverse[nearest] == np.arange(len(nearest))
            score = float(mutual.sum()/max(np.sqrt(len(frame.xyz)*len(reference_xyz)),1))
            candidates.append((score,ref_id,nearest))
        candidates.sort(reverse=True,key=lambda row:row[0])
        boundary=max(0,len(self.frames)-4)
        local=[row for row in candidates if row[1]>=boundary][:2]
        historical=[row for row in candidates if row[1]<boundary][:2]
        selected=local+historical
        arrays = {}
        for _,ref_id,nearest in selected:
            # Limit network correspondence work, not the saved/fused point cloud.
            ids = np.random.default_rng(42).choice(len(frame.xyz),min(512,len(frame.xyz)),replace=False)
            arrays[f'source_{ref_id}'] = frame.xyz[ids].astype(np.float32)
            arrays[f'target_{ref_id}'] = self.frames.match_data(ref_id)[0][nearest[ids]].astype(np.float32)
        packet = self.folder/'geometry_correspondences.npz'
        np.savez(packet,**arrays)
        self.pending = {'frame':frame,'selected':selected,'started':started}
        self.log('retrieval',candidates=[dict(reference=int(i),score=float(score),historical=i<boundary) for score,i,_ in candidates],selected=[int(i) for _,i,_ in selected],memory=self.frames.stats())
        return {'candidates':[r[1] for r in selected],'packet':str(packet),'feature_points':len(frame.xyz),'retrieval_candidates':len(candidates),'source':len(self.frames),'version':self.revision,'local_candidates':[r[1] for r in local],'historical_candidates':[r[1] for r in historical]}

    def finish(self, proposals):
        frame = self.pending['frame']
        selected={row[1] for row in self.pending['selected']}
        if len(proposals)>4 or any(p['reference'] not in selected for p in proposals):raise ValueError('Invalid registration candidate')
        matches,failures = [],[]
        references={p['reference']:self.frames[p['reference']] for p in proposals}
        for proposal in proposals:
            ref_id = proposal['reference']
            try:
                if 'error' in proposal:
                    raise ValueError(proposal['error'])
                initial = rigid_pose(proposal['pose'])
                pose,overlap,rmse,info = refine(frame,references[ref_id],initial)
                matches.append({'reference':ref_id,'pose':pose,'world':self.poses[ref_id]@pose,
                                'overlap':overlap,'rmse':rmse,'info':info,'score':overlap/(rmse+.005)})
            except ValueError as exc:
                failures.append({'reference':ref_id,'reason':str(exc)})
                self.log('candidate_rejected',reference=ref_id,reason=str(exc))
        if self.frames and not matches:
            return {'accepted':False,'reason':'未找到可靠几何位置：'+'；'.join(r['reason'] for r in failures),
                    'candidates':failures}
        if not self.frames:
            world = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1.]],float)
            matches = []
            report = {'reference':None,'overlap':1.,'rmse_mm':0.}
        else:
            matches.sort(key=lambda row:row['score'],reverse=True)
            best = matches[0]
            world = best['world']
            consistent = []
            for match in matches:
                difference = np.linalg.inv(world)@match['world']
                angle = np.arccos(np.clip((np.trace(difference[:3,:3])-1)/2,-1,1))
                if np.linalg.norm(difference[:3,3]) < .08 and angle < np.deg2rad(10):
                    consistent.append(match)
                elif match['score'] > best['score']*.9:
                    return {'accepted':False,'reason':'几何匹配存在多个相近可信度的位置，请增加独特结构与重叠区域'}
            matches = consistent
            report = {'reference':best['reference'],'overlap':best['overlap'],'rmse_mm':best['rmse']*1000}
        # Add independent measurements to several prior views, then optimize ALL poses.
        poses = [p.copy() for p in self.poses]+[world.copy()]
        edges = list(self.edges)
        current = len(self.frames)
        for index,match in enumerate(matches):
            edges.append((current,match['reference'],match['pose'],match['info'],index>0 or match['reference']<max(0,current-4)))
        graph = REG.PoseGraph()
        for pose in poses:
            graph.nodes.append(REG.PoseGraphNode(pose))
        for source,target,pose,info,uncertain in edges:
            graph.edges.append(REG.PoseGraphEdge(source,target,pose,info,uncertain))
        optimize_started = time.perf_counter()
        if len(poses)>1:
            REG.global_optimization(graph,REG.GlobalOptimizationLevenbergMarquardt(),REG.GlobalOptimizationConvergenceCriteria(),
                                    REG.GlobalOptimizationOption(max_correspondence_distance=.025,edge_prune_threshold=.25,
                                                                 preference_loop_closure=1.,reference_node=0))
        poses = [np.asarray(node.pose).copy() for node in graph.nodes]
        graph_ms = (time.perf_counter()-optimize_started)*1000
        edges = [(edge.source_node_id,edge.target_node_id,np.asarray(edge.transformation).copy(),
                  np.asarray(edge.information).copy(),edge.uncertain) for edge in graph.edges
                 if not edge.uncertain or edge.confidence >= .25]
        if current and not any(s==current or t==current for s,t,*_ in edges):
            return {'accepted':False,'reason':'新增匹配未通过图一致性检查'}
        # Check previously reliable links as well as the newly measured geometry.
        try:
            validate_graph(self.poses,poses,self.edges,[edge for edge in edges if edge[0]==current])
            for match in matches:
                geometry_validation(frame,references[match['reference']],np.linalg.inv(poses[match['reference']])@poses[-1])
        except ValueError as exc:
            self.log('graph_rejected',reason=str(exc))
            local=[p for p in proposals if p['reference']>=max(0,current-4)]
            if local and len(local)<len(proposals):
                result=self.finish(local);result['deferred_loops_reason']=str(exc);return result
            raise
        self.pending.update(poses=poses,edges=edges,graph=graph)
        self.log('graph_accepted',reference=report['reference'],rmse_mm=report['rmse_mm'],graph_ms=graph_ms,loop_edges=sum(s==current and t<max(0,current-4) for s,t,*_ in edges))
        return {'accepted':True,**report,'poses':[p.tolist() for p in poses],
                'edges':[{'source':s,'target':t,'pose':p.tolist(),'information':info.tolist(),'uncertain':uncertain} for s,t,p,info,uncertain in edges],
                'geometry_edges':len(edges),'graph_ms':graph_ms,
                'loop_candidates':sum(m['reference']<max(0,current-4) for m in matches),
                'loop_edges':sum(s==current and t<max(0,current-4) for s,t,*_ in edges),
                'memory':self.frames.stats(),
                'matching_ms':(time.perf_counter()-self.pending['started'])*1000,
                'method':'PointDSC+FPFH+robust_point_to_plane_ICP','rgb_inliers':0,'candidates':failures}

    def commit(self):
        self.frames.append(self.pending['frame'])
        self.poses,self.edges = self.pending['poses'],self.pending['edges']
        o3d.io.write_pose_graph(str(self.folder/'geometry_pose_graph.json'),self.pending['graph'])
        self.pending = None
        self.revision+=1
        return {'version':self.revision,'committed':True,'memory':self.frames.stats()}


    def state(self,poses=None,edges=None):
        poses=self.poses if poses is None else poses;edges=self.edges if edges is None else edges
        return dict(version=self.revision,poses=[p.tolist() for p in poses],edges=[dict(source=s,target=t,pose=p.tolist(),information=info.tolist(),uncertain=u) for s,t,p,info,u in edges])

    def propose_loop(self,source,version,links):
        if self.pending is not None or self.loop_pending is not None:raise ValueError('Map transaction in progress')
        if version>self.revision or source<0 or source>=len(self.frames):raise ValueError('Invalid loop version/node')
        known={tuple(sorted((s,t))) for s,t,*_ in self.edges};extra=[]
        for link in links[:2]:
            target=int(link['target'])
            if target<0 or target>=max(0,source-4) or tuple(sorted((source,target))) in known:continue
            pose=rigid_pose(link['pose']);info=np.asarray(link['information'],float)
            if info.shape!=(6,6) or not np.isfinite(info).all() or not np.allclose(info,info.T,atol=1e-6) or np.linalg.eigvalsh(info).min()<0:raise ValueError('Invalid loop information')
            extra.append((source,target,pose,info,True));known.add(tuple(sorted((source,target))))
        if not extra:return dict(accepted=False,reason='No new loop constraints')
        graph=REG.PoseGraph()
        for p in self.poses:graph.nodes.append(REG.PoseGraphNode(p.copy()))
        for a,b,p,info,u in self.edges+extra:graph.edges.append(REG.PoseGraphEdge(a,b,p,info,u))
        REG.global_optimization(graph,REG.GlobalOptimizationLevenbergMarquardt(),REG.GlobalOptimizationConvergenceCriteria(),REG.GlobalOptimizationOption(max_correspondence_distance=.025,edge_prune_threshold=.25,preference_loop_closure=1.,reference_node=0))
        poses=[np.asarray(node.pose).copy() for node in graph.nodes]
        edges=[(e.source_node_id,e.target_node_id,np.asarray(e.transformation).copy(),np.asarray(e.information).copy(),e.uncertain) for e in graph.edges if not e.uncertain or e.confidence>=.25]
        retained=[e for e in extra if any(s==e[0] and t==e[1] for s,t,*_ in edges)]
        if not retained:return dict(accepted=False,reason='Loop constraints pruned')
        validate_graph(self.poses,poses,self.edges,retained)
        for a,b,*_ in retained:geometry_validation(self.frames[a],self.frames[b],np.linalg.inv(poses[b])@poses[a])
        self.loop_pending=dict(poses=poses,edges=edges,graph=graph,base_revision=self.revision)
        self.log('async_loop_proposed',source=source,created_version=version,validated_version=self.revision,retained=len(retained))
        return dict(accepted=True,**self.state(poses,edges))

    def commit_loop(self):
        candidate=self.loop_pending
        if candidate is None or candidate['base_revision']!=self.revision:raise ValueError('Stale loop transaction')
        # Persist first, then publish one complete graph state.
        path=self.folder/'geometry_pose_graph.pending.json'
        o3d.io.write_pose_graph(str(path),candidate['graph'])
        if not path.is_file():raise OSError('Cannot save loop graph')
        json.loads(path.read_text(encoding='utf-8'))
        os.replace(path,self.folder/'geometry_pose_graph.json')
        self.poses,self.edges=candidate['poses'],candidate['edges'];self.loop_pending=None;self.revision+=1
        self.log('async_loop_committed',version=self.revision)
        return dict(committed=True,**self.state())


if __name__ == '__main__':
    engine = GeometryEngine(sys.argv[1])
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request['command']
            if command == 'close':
                break
            if command == 'prepare':
                result = engine.prepare(request['path'])
            elif command == 'finish':
                result = engine.finish(request['proposals'])
            elif command == 'commit':
                result = engine.commit()
            elif command == 'propose_loop':
                result=engine.propose_loop(request['source'],request['version'],request['links'])
            elif command == 'commit_loop':
                result=engine.commit_loop()
            elif command == 'discard_capture':
                engine.pending=None;result={'discarded':True}
            elif command == 'discard_loop':
                engine.loop_pending=None;result={'discarded':True}
            else:
                raise ValueError('Unknown geometry command')
            print('@@GEOMETRY '+json.dumps(result),flush=True)
        except Exception as exc:
            print('@@GEOMETRY '+json.dumps({'error':str(exc)}),flush=True)
