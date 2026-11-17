"""Offline source-aware scalar-normal Kalman surfel fusion; camera poses are immutable."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from file_snapshots import read_shared_json,read_shared_bytes
from rgbd_geometry import recorded_coordinates

OPTICAL_FROM_WORLD=np.array([[0,-1,0],[0,0,-1],[1,0,0.]],float)


def _rigid(pose, name):
    pose=np.asarray(pose,float)
    if (pose.shape!=(4,4) or not np.isfinite(pose).all() or
        not np.allclose(pose[3],[0,0,0,1],atol=1e-5) or
        not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=.01) or
        not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=.01)):
        raise ValueError('无效刚体变换：'+name)
    return pose


def session_observations(source):
    """Resolve accepted capture identities explicitly, never zip all raw files with poses."""
    source=Path(source).resolve(); folder=source if source.is_dir() else source.parent
    required=('geometry_poses.json','sdk_calibration.json','frames.jsonl','registrations.jsonl')
    if not all((folder/name).is_file() for name in required) or not any((folder/name).is_dir() for name in ('pointcloud_raw','pointcloud_local')):
        raise ValueError('卡尔曼融合需要扫描目录：局部表面观测或旧pointcloud_raw、标定、位姿、帧清单和配准记录；单独PLY不足以恢复多次观测')
    graph=read_shared_json(folder/'geometry_poses.json')
    calibration=json.loads((folder/'sdk_calibration.json').read_text(encoding='utf-8'))
    extrinsic=np.eye(4)
    extrinsic[:3,:3]=[[calibration[f'R{i}{j}'] for j in range(3)] for i in range(3)]
    extrinsic[:3,3]=[calibration[f'T{i}']/1000 for i in range(1,4)]
    extrinsic=_rigid(extrinsic,'IR→RGB')
    frames={}
    for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            row=json.loads(line); capture=int(row['capture'])
            if capture in frames:raise ValueError('帧清单包含重复拍摄编号')
            frames[capture]=row
    accepted={}
    for line in (folder/'registrations.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            row=json.loads(line)
            if row.get('accepted'):
                node=int(row['node'])
                if node in accepted:raise ValueError('配准记录包含重复接受节点')
                accepted[node]=int(row['capture'])
    nodes=graph['node_ids']; poses=graph['poses']
    if not nodes or len(nodes)!=len(poses) or len(set(nodes))!=len(nodes):
        raise ValueError('位姿节点清单不完整或重复')
    observations=[]; captures=set()
    for node,pose in zip(nodes,poses):
        if node not in accepted:raise ValueError(f'节点{node}缺少对应的接受记录')
        capture=accepted[node]
        if capture in captures or capture not in frames:raise ValueError('拍摄来源重复或缺失')
        captures.add(capture)
        frame=frames[capture]
        kind=frame.get('cloud_source')
        if kind not in ('sdk_xyz','local_surface_v1'):
            raise ValueError('当前卡尔曼原始观测入口只接受已记录的SDK XYZ')
        path=folder/('pointcloud_local' if kind=='local_surface_v1' else 'pointcloud_raw')/(f'{float(frame["stamp"]):.6f}'+('.npz' if kind=='local_surface_v1' else '.npy'))
        if not path.is_file():raise FileNotFoundError(f'缺少原始点云：{path}')
        coordinates=recorded_coordinates(frame)
        pointcloud_frame=coordinates['pointcloud_frame'] if coordinates else 'ir'
        rgb_pose=_rigid(pose,f'节点{node}')
        world_from_points=rgb_pose@extrinsic if pointcloud_frame=='ir' else rgb_pose
        optical_from_points=np.eye(4)
        optical_from_points[:3,:3]=OPTICAL_FROM_WORLD@world_from_points[:3,:3]
        optical_from_points[:3,3]=OPTICAL_FROM_WORLD@world_from_points[:3,3]
        observations.append({'capture':capture,'node':node,'path':path,'pose':optical_from_points,'cloud_source':kind,
                             'rgb_pose':rgb_pose,'coordinates':coordinates})
    fingerprints={name:hashlib.sha256(read_shared_bytes(folder/name)).hexdigest() for name in required}
    if (folder/'local_preprocessing.json').is_file():
        fingerprints['local_preprocessing.json']=hashlib.sha256((folder/'local_preprocessing.json').read_bytes()).hexdigest()
    return folder,observations,fingerprints


def measurement_variance(depth, incidence, floor_mm=2., noise_mm=10.):
    """Conservative heuristic, not a calibrated HP60C accuracy model (variance m²)."""
    sigma=np.maximum(noise_mm/3000, .0015+.001*np.asarray(depth)**2)
    return np.maximum((floor_mm/1000)**2,sigma**2/np.maximum(np.asarray(incidence),.15))


class SurfelMap:
    def __init__(self,params):
        self.params=params; self.position=np.empty((0,3),float)
        self.normal=np.empty((0,3),float); self.anchor=np.empty((0,3),float)
        self.variance=np.empty(0,float);self.support=np.empty(0,np.int32)
        self.fusable=np.empty(0,bool);self.capture_ids=set();self.metrics=[]

    def integrate(self,points,normals,variance,fusable,capture,cancel=None):
        from preprocess import _check
        _check(cancel)
        if capture in self.capture_ids:raise ValueError('同一次拍摄不能重复增加融合置信度')
        points=np.asarray(points,float);normals=np.asarray(normals,float)
        variance=np.asarray(variance,float);fusable=np.asarray(fusable,bool)
        if (points.ndim!=2 or points.shape[1]!=3 or normals.shape!=points.shape or
            variance.shape!=(len(points),) or fusable.shape!=(len(points),) or
            not np.isfinite(points).all() or not np.isfinite(normals).all() or
            not np.isfinite(variance).all() or np.any(variance<=0)):
            raise ValueError('无效表面元观测')
        variance=np.maximum(variance,(self.params.kalman_floor_mm/1000)**2)
        n=len(points); matched=np.full(n,-1,np.int64); best_weight=np.zeros(n)
        eligible=np.flatnonzero(self.fusable)
        gate=self.params.noise_mm/1000
        tangent_gate=max(self.params.output_voxel_mm*1.5/1000,.006)
        radius=np.hypot(gate,tangent_gate)
        cosine=np.cos(np.deg2rad(self.params.normal_angle_deg))
        if len(eligible):
            tree=cKDTree(self.position[eligible]); k=min(8,len(eligible))
            for a in range(0,n,self.params.batch_size):
                _check(cancel);b=min(n,a+self.params.batch_size)
                distance,near=tree.query(points[a:b],k=k,distance_upper_bound=radius,workers=2)
                distance=distance.reshape(b-a,k); near=near.reshape(b-a,k)
                exists=near<len(eligible); targets=eligible[np.minimum(near,len(eligible)-1)]
                delta=points[a:b,None]-self.position[targets]
                height=np.einsum('nki,nki->nk',delta,self.normal[targets])
                tangent=np.sqrt(np.maximum(0,np.sum(delta*delta,axis=2)-height**2))
                agreement=np.einsum('ni,nki->nk',normals[a:b],self.normal[targets])
                innovation=height**2/(self.variance[targets]+variance[a:b,None])
                allowed=(exists&fusable[a:b,None]&(agreement>=cosine)&(abs(height)<=gate)&
                         (tangent<=tangent_gate)&(innovation<=9))
                cost=innovation+(tangent/tangent_gate)**2
                cost=np.where(allowed,cost,np.inf)
                choice=np.argmin(cost,axis=1); row=np.arange(b-a); ok=np.isfinite(cost[row,choice])
                # Competing parallel sheets are not collapsed merely because both pass a gate.
                first=self.position[targets[row,choice]]
                sheet_offset=np.abs(np.einsum('nki,ni->nk',self.position[targets]-first[:,None],self.normal[targets[row,choice]]))
                ambiguous=np.any(allowed&(sheet_offset>self.params.layer_gap_mm/1000)&
                                 (cost<cost[row,choice,None]+.5),axis=1)
                ok&=~ambiguous
                matched[a:b][ok]=targets[row[ok],choice[ok]]
                best_weight[a:b][ok]=np.exp(-.5*(tangent[row[ok],choice[ok]]/tangent_gate)**2)/variance[a:b][ok]
        good=matched>=0; updated=np.empty(0,int); innovations=np.empty(0)
        _check(cancel)
        self.capture_ids.add(capture)
        if good.any():
            target=matched[good]; unique,inverse=np.unique(target,return_inverse=True)
            w=best_weight[good]; total=np.bincount(inverse,weights=w)
            normal=self.normal[unique]
            height=np.sum((points[good]-self.position[target])*self.normal[target],axis=1)
            z=np.bincount(inverse,weights=w*height)/total
            scatter=np.bincount(inverse,weights=w*(height-z[inverse])**2)/total
            # Correlated pixels from ONE frame produce ONE measurement, not N independent updates.
            r=np.bincount(inverse,weights=w*variance[good])/total+scatter
            p=self.variance[unique];gain=p/(p+r)
            proposal=self.position[unique]+(gain*z)[:,None]*normal
            displacement=proposal-self.anchor[unique]
            length=np.linalg.norm(displacement,axis=1)
            proposal=self.anchor[unique]+displacement*np.minimum(1.,self.params.max_displacement_mm/1000/np.maximum(length,1e-12))[:,None]
            self.position[unique]=proposal
            self.variance[unique]=np.maximum((self.params.kalman_floor_mm/1000)**2,(1-gain)*p)
            self.support[unique]+=1
            updated=unique;innovations=abs(z)*1000
        new=~good
        self.position=np.vstack((self.position,points[new]));self.anchor=np.vstack((self.anchor,points[new]))
        self.normal=np.vstack((self.normal,normals[new]));self.variance=np.r_[self.variance,variance[new]]
        self.support=np.r_[self.support,np.ones(new.sum(),np.int32)]
        self.fusable=np.r_[self.fusable,fusable[new]]
        record={'capture':int(capture),'observations':n,'associated_observations':int(good.sum()),
                'updated_surfels_once':len(updated),'new_surfels':int(new.sum()),'surfels':len(self.position),
                'innovation_mm_p50_p90':np.percentile(innovations,[50,90]).tolist() if len(innovations) else []}
        self.metrics.append(record)
        return record

    def output(self,cancel=None,progress=None):
        """Final density reduction never adds same-view confidence or shrinks variance."""
        keys,inverse,count=np.unique(np.floor(self.position/(self.params.output_voxel_mm/1000)).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
        points=np.column_stack([np.bincount(inverse,weights=self.position[:,i])/count for i in range(3)])
        support=np.zeros(len(count),np.int32);np.maximum.at(support,inverse,self.support)
        sigma=np.zeros(len(count));np.maximum.at(sigma,inverse,np.sqrt(self.variance)*1000)
        order=np.argsort(inverse,kind='stable');first=np.r_[0,np.cumsum(count)[:-1]]
        normals=self.normal[order[first]]
        points=np.asarray(points,np.float32)
        attributes={'support_views':support,'sigma_normal_mm':sigma.astype(np.float32),'normals':normals.astype(np.float32)}
        self.sampling_metrics={'algorithm':'voxel centroid','input_points':len(self.position),'output_points':len(points)}
        if self.params.kalman_sampling=='surface_uniform':
            from surface_sampling import uniform_surface_indices
            ids,self.sampling_metrics=uniform_surface_indices(points,attributes['normals'],
                self.params.surface_spacing_mm/1000,support,attributes['sigma_normal_mm'],progress,cancel)
            self.sampling_metrics['pre_sampling_voxel_mm']=self.params.output_voxel_mm
            points=points[ids];attributes={key:value[ids] for key,value in attributes.items()}
        return points,attributes


def process_session(source,params=None,before_points=None,progress=None,cancel=None):
    from preprocess import Parameters,ProcessingResult,_check,_notify,_geometry,quality
    from voxel_map import voxel_groups
    from cloud_import import load_cloud
    params=params or Parameters();params.validate();started=time.perf_counter();_check(cancel)
    folder,observations,fingerprints=session_observations(source)
    if before_points is None:before_points=load_cloud(folder/'map_cloud.ply').points
    before_points=np.asarray(before_points,np.float32)
    before=quality(before_points,cancel)
    model=SurfelMap(params);removed=[];raw_count=0;invalid_count=0;source_files=[]
    isolated_count=0;precomputed=[]
    for index,obs in enumerate(observations):
        _check(cancel);base=.04+.81*index/len(observations);width=.81/len(observations)
        _notify(progress,base,f'第{index+1}/{len(observations)}次拍摄：读取局部观测')
        path=obs['path'];payload=path.read_bytes();source_files.append({'capture':obs['capture'],
            'node':obs['node'],'file':str(path.relative_to(folder)),'sha256':hashlib.sha256(payload).hexdigest()})
        import io
        from local_preprocessing import prepare_local,load_local
        if obs['cloud_source']=='local_surface_v1':
            prepared=load_local(io.BytesIO(payload))
            expected_frame=obs['coordinates']['pointcloud_frame'] if obs['coordinates'] else 'ir'
            if prepared['metadata'].get('camera_frame','ir')!=expected_frame:
                raise ValueError('Local cloud coordinates disagree with frame manifest')
            local_params=prepared['metadata']['parameters']
            if params.radius_mm!=local_params['radius_mm'] or params.neighbors!=local_params['neighbors']:
                raise ValueError('局部拟合已在采集时完成，不能重新修改拟合邻域或邻居数；请使用采集时参数')
            if (folder/'local_preprocessing.json').is_file():
                saved_profile=json.loads((folder/'local_preprocessing.json').read_text(encoding='utf-8'))
                if saved_profile.get('schema')!='local_surface_v1' or saved_profile['parameters']!=local_params:
                    raise ValueError('局部观测与采集参数记录不一致')
            precomputed.append({'capture':obs['capture'],'parameters':local_params})
        else:
            raw=np.asarray(np.load(io.BytesIO(payload),allow_pickle=False),float).reshape(-1,3)/1000
            local_progress=lambda value,text:_notify(progress,base+width*.1+width*.5*value,text)
            frame=obs['coordinates']['pointcloud_frame'] if obs['coordinates'] else 'ir'
            prepared=prepare_local(raw,params,local_progress,cancel,coordinate_frame=frame)
        local,normals=prepared['points'],prepared['normals'].copy()
        valid,boundary=prepared['valid'],prepared['boundary']
        stats=prepared['metadata'];raw_count+=stats['input_points'];invalid_count+=stats['invalid_points']
        isolated_count+=stats['isolated_points']
        pose=obs['pose']
        removed.append(prepared['removed']@pose[:3,:3].T+pose[:3,3])
        rays=local/np.maximum(np.linalg.norm(local,axis=1)[:,None],1e-12)
        incidence=np.abs(np.sum(normals*rays,axis=1))
        variance=measurement_variance(local[:,2],incidence,params.kalman_floor_mm,params.noise_mm)
        points=local@pose[:3,:3].T+pose[:3,3];normals=normals@pose[:3,:3].T
        _notify(progress,base+width*.65,f'第{index+1}/{len(observations)}次拍摄：表面关联与卡尔曼更新')
        model.integrate(points,normals,variance,valid&~boundary&(incidence>.15),obs['capture'],cancel)
        _notify(progress,base+width,f'已融合{index+1}/{len(observations)}次拍摄 · {len(model.position):,}个表面元')
    _check(cancel);_notify(progress,.87,'生成均匀表面点云')
    sampling_progress=lambda value,text:_notify(progress,.87+.06*value,text)
    points,diagnostics=model.output(cancel,sampling_progress);after=quality(points,cancel)
    removed=np.vstack(removed) if removed else np.empty((0,3))
    original_delta=np.linalg.norm(model.position-model.anchor,axis=1)*1000
    _notify(progress,.96,'验证源位姿和采集文件未变化')
    for name,digest in fingerprints.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('融合期间源标定/位姿发生变化')
    for entry in source_files:
        if hashlib.sha256((folder/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('融合期间原始点云发生变化')
    report={'schema_version':2,'backend':'kalman','parameters':vars(params),'poses_modified':False,
        'source_session':str(folder),'source_sha256':fingerprints,'accepted_raw_observations':source_files,
        'source_unchanged':True,'input_points':len(before_points),'raw_points':raw_count,'invalid_raw_points':invalid_count,
        'isolated_points':isolated_count,'exact_duplicates':0,'output_points':len(points),
        'precomputed_local_captures':len(precomputed),'local_preprocessing':precomputed,
        'removed_points_coordinates_available':not bool(precomputed),
        'local_processing_note':'新扫描复用采集时局部清洗和法线；二阶段不重复拟合。容忍尺度用于融合和方差估计，输出体素用于最终密度归并；局部清洗参数单独记录。旧扫描仍现场计算局部表面。',
        'coordinates':'camera','units':'m','captures':len(observations),'surfels_before_output':len(model.position),
        'surfels_supported_by_multiple_views':int((model.support>=2).sum()),
        'output_points_supported_by_multiple_views':int((diagnostics['support_views']>=2).sum()),
        'uniform_sampling':model.sampling_metrics,
        'surfel_anchor_movement_mm_p50_p90_p99_max':np.percentile(original_delta,[50,90,99,100]).tolist(),
        'movement_mm_p50_p90_p99_max':[0.,0.,0.,0.],
        'before':before,'after':after,'elapsed_seconds':time.perf_counter()-started,'per_capture':model.metrics,
        'measurement_model':'heuristic sigma=max(noise_mm/3,1.5+depth_m^2) mm; R=sigma²/max(incidence,0.15); not hardware calibrated',
        'filter_model':'static normal position, Q=0; one aggregate update per surfel per capture; P floor; fixed seed normal/tangent anchor',
        'association':'oriented normal, tangent support, hard normal gate, innovation and competing-sheet rejection',
        'index_note':'本后端从原始多次观测重建，无最终PLY逐点位移对应；索引/位移文件为空，表面元锚点位移另记于报告。',
        'metric_note':'方差是含假设的工程置信度，不是真值精度；厚带减少也可能误合并真实薄结构，需查看截面和边缘。'}
    _notify(progress,1.,'卡尔曼表面融合完成')
    return ProcessingResult(points,removed,np.empty(0,np.int64),np.empty(0,np.float32),report,diagnostics)
