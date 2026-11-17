"""Pose-independent local cleaning and surface estimation, shared by capture and legacy replay."""
from pathlib import Path
import json
import os
import time
import numpy as np
from scipy.spatial import cKDTree

SCHEMA='local_surface_v1'
PROFILE_KEYS=('noise_mm','radius_mm','output_voxel_mm','neighbors','isolation_factor')


def profile(params):
    return {key:getattr(params,key) for key in PROFILE_KEYS}


def prepare_local(points,params,progress=None,cancel=None,*,coordinate_frame='ir'):
    from rgbd_geometry import camera_frame
    camera_frame(coordinate_frame)
    from preprocess import _geometry,_check
    from voxel_map import voxel_groups
    started=time.perf_counter();_check(cancel)
    raw=np.asarray(points,float).reshape(-1,3)
    total=len(raw)
    good=np.isfinite(raw).all(1)&(raw[:,2]>=.15)&(raw[:,2]<=4)
    invalid=int((~good).sum());raw=raw[good]
    if len(raw)<12:raise ValueError('拍摄有效点不足，不能估计局部表面')
    _,sums,count=voxel_groups(raw,params.output_voxel_mm/1000);local=sums/count[:,None]
    tree=cKDTree(local);spacing=tree.query(local,k=min(9,len(local)),workers=2)[0][:,-1]
    _,ids=tree.query(local,k=min(17,len(local)),workers=2)
    isolated=spacing>params.isolation_factor*np.maximum(np.median(spacing[ids[:,1:]],axis=1),1e-6)
    removed=local[isolated];local=local[~isolated];spacing=spacing[~isolated]
    if len(local)<12:raise ValueError('清洗后有效点不足，不能估计局部表面')
    _,normals,valid,boundary=_geometry(local,spacing,params,progress,0.,1.,cancel)
    normals[np.sum(normals*(-local),axis=1)<0]*=-1
    return dict(points=local,normals=normals,valid=valid,boundary=boundary,removed=removed,
        metadata=dict(schema=SCHEMA,units='m',coordinates=f'{coordinate_frame.upper()} optical X right Y down Z forward',camera_frame=coordinate_frame,
            parameters=profile(params),input_points=total,invalid_points=invalid,isolated_points=len(removed),
            output_points=len(local),elapsed_seconds=time.perf_counter()-started))


def save_local(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('局部观测已存在，拒绝覆盖：'+str(path))
    temporary=path.with_suffix('.npz.part')
    try:
        with temporary.open('xb') as stream:
            np.savez_compressed(stream,points=data['points'],normals=data['normals'],
                valid=data['valid'],boundary=data['boundary'],metadata=np.array(json.dumps(data['metadata'])))
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def load_local(path):
    with np.load(path,allow_pickle=False) as packet:
        data={key:packet[key].copy() for key in ('points','normals','valid','boundary')}
        meta=json.loads(str(packet['metadata'].item()))
    p,n=data['points'],data['normals']
    frame=meta.get('camera_frame','ir')
    if (meta.get('schema')!=SCHEMA or meta.get('units')!='m' or frame not in ('ir','rgb') or
        meta.get('coordinates')!=f'{frame.upper()} optical X right Y down Z forward' or
        p.ndim!=2 or p.shape[1]!=3 or len(p)<12 or n.shape!=p.shape or
        not np.isfinite(p).all() or not np.isfinite(n).all() or
        (np.linalg.norm(n,axis=1)<.5).any() or meta.get('output_points')!=len(p)):
        raise ValueError('局部表面观测无效：'+str(path))
    for key in ('valid','boundary'):
        if data[key].shape!=(len(p),) or data[key].dtype!=np.bool_:raise ValueError('局部表面标签无效')
    data.update(metadata=meta,removed=np.empty((0,3)))
    return data
