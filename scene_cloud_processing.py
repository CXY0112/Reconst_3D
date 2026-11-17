"""Whole-scene consolidation, feature-aware simplification and measured RGB provenance."""
from contextlib import contextmanager
from pathlib import Path
import json,os,shutil,time,uuid
import cv2
import numpy as np
from scipy.spatial import cKDTree
from preprocess import Parameters,ProcessingCancelled,_check,save_result


def session_for(source):
    source=Path(source).resolve();folder=source if source.is_dir() else source.parent
    if (folder/'setp1').is_dir():folder=folder/'setp1'
    if not (folder/'geometry_poses.json').is_file():
        raise ValueError('完整场景融合需要一阶段的RGB-D、相机位姿和原始观测；逐帧点图导出缺少位姿，不能直接拼接')
    return folder


def colorize(points,session,progress=None,cancel=None,frames=None):
    """Depth-tested best measured photo, with explicit gray for unobserved points."""
    from tsdf_reconstruction import sdk_frames
    from kalman_fusion import OPTICAL_FROM_WORLD
    started=time.perf_counter();points=np.asarray(points,np.float32)
    if frames is None:frames,k,_=sdk_frames(session)
    else:frames,k=frames
    colors=np.full((len(points),3),160,np.uint8);best=np.full(len(points),np.inf,np.float32)
    captures=np.full(len(points),-1,np.int32);pixels=np.full((len(points),2),-1,np.int32)
    for index,frame in enumerate(frames):
        _check(cancel)
        rgb=cv2.imread(str(Path(session)/'rgb_raw'/frame['depth'].name),cv2.IMREAD_COLOR)
        depth=cv2.imread(str(frame['depth']),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2]!=depth.shape:raise ValueError('缺少同网格RGB-D，不能为完整场景可靠着色')
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB);depth=depth.astype(np.float32)/1000
        pose=np.asarray(frame['pose'],np.float32)
        rotation=np.asarray(OPTICAL_FROM_WORLD@pose[:3,:3],np.float32);origin=np.asarray(OPTICAL_FROM_WORLD@pose[:3,3],np.float32)
        for a in range(0,len(points),131072):
            _check(cancel);b=min(len(points),a+131072)
            local=np.einsum('ni,ij->nj',points[a:b]-origin,rotation,dtype=np.float32)
            z=local[:,2];positive=np.flatnonzero((z>=.15)&(z<=4))
            if not len(positive):continue
            q=local[positive];u=np.rint(k[0,0]*q[:,0]/q[:,2]+k[0,2]).astype(int);v=np.rint(k[1,1]*q[:,1]/q[:,2]+k[1,2]).astype(int)
            ok=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]);positive=positive[ok];u=u[ok];v=v[ok];q=q[ok]
            observed=depth[v,u];residual=abs(observed-q[:,2]);valid=(observed>=.15)&(observed<=4)&(residual<=.03)
            score=residual+.002*np.sum((q[:,:2]/q[:,2,None])**2,axis=1)
            ids=positive+a;take=valid&(score<best[ids]);ids=ids[take]
            colors[ids]=rgb[v[take],u[take]];best[ids]=score[take]
            captures[ids]=int(frame['capture']);pixels[ids]=np.c_[u[take],v[take]]
        if progress:progress((index+1)/len(frames),f'完整场景RGB可见性匹配 {index+1}/{len(frames)}')
    report=dict(method='best depth-consistent measured RGB observation',points=len(points),
        observed_points=int((captures>=0).sum()),coverage=float(np.mean(captures>=0)) if len(points) else 0.,
        unmatched_color=[160,160,160],depth_gate_mm=30.,elapsed_seconds=time.perf_counter()-started,
        note='No invented colors; per-point capture ID and original UV retained. Not a one-to-one image grid.')
    return colors,dict(source_capture=captures,source_uv=pixels,color_valid=captures>=0),report


def simplify(points,normals,support,sigma,spacing_mm=8.,cancel=None,progress=None):
    """Keep all detected feature points; reduce flat surfaces without moving points."""
    from surface_sampling import uniform_surface_indices
    _check(cancel);points=np.asarray(points);normals=np.asarray(normals)
    n=len(points);feature=np.ones(n,bool)
    if n>=12:
        tree=cKDTree(points);feature=np.zeros(n,bool)
        for a in range(0,n,16384):
            _check(cancel);b=min(n,a+16384)
            distance,near=tree.query(points[a:b],k=min(16,n),workers=2)
            patch=points[near];offset=patch-patch.mean(1,keepdims=True)
            eigen=np.linalg.eigvalsh(np.einsum('nki,nkj->nij',offset,offset)/near.shape[1])
            curvature=eigen[:,0]/np.maximum(eigen.sum(1),1e-12)
            dot=np.abs(np.einsum('ni,nki->nk',normals[a:b],normals[near]))
            # Corners/curves, opposing normals, sparse borders, and line-like patches.
            spread=np.einsum('ni,nki->nk',normals[a:b],normals[near])
            feature[a:b]=(np.min(dot,axis=1)<np.cos(np.deg2rad(25)))|(np.min(spread,axis=1)<0)|\
                (curvature>.025)|(eigen[:,1]<eigen[:,2]*.15)|(distance[:,-1]>.035)
        # Retain a buffer around the feature set so flat sampling cannot erode an edge.
        edge=np.flatnonzero(feature)
        if len(edge):feature|=cKDTree(points[edge]).query(points,workers=2)[0]<=.006
    protected=np.flatnonzero(feature);flat=np.flatnonzero(~feature)
    if len(flat):
        selected,sampling=uniform_surface_indices(points[flat],normals[flat],spacing_mm/1000,
            support[flat],sigma[flat],progress,cancel)
        ids=np.sort(np.r_[protected,flat[selected]])
    else:ids=protected;sampling={}
    return ids,dict(algorithm='feature-protected flat-surface simplification',input_points=n,output_points=len(ids),
        feature_points=int(feature.sum()),all_detected_features_retained=True,flat_spacing_mm=spacing_mm,
        feature_mask_heuristic=True,new_geometry=False,sampling=sampling),feature[ids]


def process_scene(source,points,params=None,progress=None,cancel=None):
    from kalman_fusion import process_session
    started=time.perf_counter();session=session_for(source);params=params or Parameters()
    def part(low,high):return lambda f,t:progress(low+(high-low)*f,t) if progress else None
    result=process_session(session,params,before_points=points,progress=part(0,.75),cancel=cancel)
    attributes=result.diagnostics
    ids,report,features=simplify(result.points,attributes['normals'],attributes['support_views'],attributes['sigma_normal_mm'],
                                cancel=cancel,progress=part(.75,.85))
    result.points=result.points[ids];result.diagnostics={k:v[ids] for k,v in attributes.items()}
    colors,provenance,color_report=colorize(result.points,session,part(.85,1),cancel)
    result.diagnostics.update(rgb=colors,feature_protected=features,**provenance)
    result.report.update(scene_rgb_schema='colored_scene_v1',consolidation_points=report['input_points'],
        output_points=len(result.points),simplification=report,color=color_report,
        output_points_supported_by_multiple_views=int((result.diagnostics['support_views']>=2).sum()),
        elapsed_seconds=time.perf_counter()-started,
        pipeline='local outliers -> normal Kalman surfel fusion -> voxel dedup -> uniform resampling -> feature-protected simplification')
    return result


def write_colored_ply(path,points,colors):
    packed=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,key in enumerate(('x','y','z')):packed[key]=points[:,i]
    for i,key in enumerate(('red','green','blue')):packed[key]=colors[:,i]
    with Path(path).open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\ncomment RTAB-Map world XYZ in metres\n'+
            f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'+
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        packed.tofile(stream)


@contextmanager
def _scene_destination(destination,session):
    destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    session=Path(session).resolve()
    if destination==session or session in destination.parents or destination in session.parents:raise ValueError('不得覆盖一阶段原件或其父目录')
    if destination.exists():
        try:owned=json.loads((destination/'report.json').read_text(encoding='utf-8')).get('scene_rgb_schema') in ('colored_scene_v1','colored_scene_skip_v1')
        except (OSError,ValueError):owned=False
        if not owned:raise ValueError('目标已有其他数据，请选择彩色场景分支的保存目录')
    staging=destination.parent/('.pending_scene_'+uuid.uuid4().hex);backup=destination.parent/('.previous_scene_'+uuid.uuid4().hex)
    committed=False
    try:
        yield staging
        if destination.exists():destination.rename(backup)
        try:staging.rename(destination);committed=True
        except BaseException:
            if backup.exists():backup.rename(destination)
            raise
    finally:
        for path in ((staging,backup) if committed else (staging,)):
            if path.exists() and path.resolve().parent==destination.parent and path.name.startswith(('.pending_scene_','.previous_scene_')):
                shutil.rmtree(path)


def save_scene(result,source,destination):
    session=session_for(source)
    with _scene_destination(destination,session) as staging:
        save_result(result,source,staging)
        world=np.column_stack((result.points[:,2],-result.points[:,0],-result.points[:,1]))
        write_colored_ply(staging/'processed_colored.ply',world,result.diagnostics['rgb'])
        # Standard processed.ply also carries RGB, while preserving its geometry schema.
        shutil.copyfile(staging/'processed_colored.ply',staging/'processed.ply')
    return Path(destination).resolve()


def save_scene_skip(source,destination,point_count):
    """Record the bypass choice without copying or pretending to clean raw points."""
    session=session_for(source);destination=Path(destination).resolve()
    if not (session/'map_cloud.ply').is_file():raise ValueError('跳过预处理需要保存完整的一阶段扫描')
    report=dict(complete=True,backend='skip',skipped=True,scene_rgb_schema='colored_scene_skip_v1',
        source=str(session/'map_cloud.ply'),source_session=str(session),
        source_session_relative=os.path.relpath(session,destination),
        input_points=int(point_count),output_points=int(point_count),
        poses_modified=False,points_moved=False,points_resampled=False,
        note='Stage 2 explicitly skipped. Stage 3 reads the original saved map and measured RGB-D; no cloud copied.')
    with _scene_destination(destination,session) as staging:
        staging.mkdir()
        (staging/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
