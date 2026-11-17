"""Isolated Open3D FPFH + CPU MAC, one input packet per process."""
from runtime_paths import activate_geometry_packages
activate_geometry_packages()
import json,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from mac_solver import solve
from bufferx_backend import validate_registration


def features(points,settings):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    voxel=settings.get('voxel_m',.03);cloud=cloud.voxel_down_sample(voxel)
    if len(cloud.points)<100:raise ValueError('MAC sampled points insufficient')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2,max_nn=30))
    f=np.asarray(o3d.pipelines.registration.compute_fpfh_feature(cloud,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=100)).data).T
    norms=np.linalg.norm(f,axis=1);valid=np.isfinite(f).all(1)&(norms>1e-8)
    if valid.sum()<100:raise ValueError('MAC valid FPFH features insufficient')
    return np.asarray(cloud.points)[valid],f[valid]/norms[valid,None]


def register(source,target,settings,validation):
    began=time.monotonic();s,fs=features(source,settings);t,ft=features(target,settings)
    distance,st=cKDTree(ft).query(fs);ts=cKDTree(fs).query(ft)[1]
    ids=np.flatnonzero(ts[st]==np.arange(len(s)))
    # Spatially distribute matches so one repetitive surface patch cannot
    # consume the whole graph budget with nearly duplicate hypotheses.
    selected=[];left=set();right=set();spacing=settings.get('correspondence_spacing_m',.06)
    for i in ids[np.argsort(distance[ids],kind='stable')]:
        a=tuple(np.floor(s[i]/spacing).astype(int));b=tuple(np.floor(t[st[i]]/spacing).astype(int))
        if a in left or b in right:continue
        left.add(a);right.add(b);selected.append(i)
        if len(selected)>=settings.get('max_correspondences',256):break
    ids=np.asarray(selected,dtype=int)
    result=solve(s[ids],t[st[ids]],settings)
    _,metrics=validate_registration(source,target,result,validation,'MAC')
    result.update(validation=metrics,feature='FPFH',matching='mutual_L2',points=[len(s),len(t)],
                  total_ms=(time.monotonic()-began)*1000)
    return result


if __name__=='__main__':
    packet=np.load(sys.argv[1]);meta=json.loads(str(packet['metadata']))
    try:result=dict(ok=True,result=register(packet['source'],packet['target'],meta['settings'],meta['validation']))
    except ValueError as exc:result=dict(ok=False,error=str(exc))
    Path(sys.argv[2]).write_text(json.dumps(dict(job=meta['job'],reference=meta['reference'],source_stamp=meta.get('source_stamp'),**result)),encoding='utf-8')
