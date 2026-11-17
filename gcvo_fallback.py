"""Independent CPU recovery: verified RGB-D PnP, or MAC + bounded trimmed ICP.

Every returned pose maps the CURRENT cloud to the trusted reference. The backend
still applies full-cloud geometry, confidence and temporal confirmation gates.
"""
import time
import numpy as np
from scipy.spatial import cKDTree


def sample(points,voxel,limit=6000):
    p=np.asarray(points,dtype=float);p=p[np.isfinite(p).all(1)]
    if len(p)<100:raise ValueError('恢复配准有效点不足')
    p=p[::max(1,int(np.ceil(len(p)/60000)))]
    _,ids=np.unique(np.floor(p/voxel).astype(np.int64),axis=0,return_index=True)
    p=p[ids]
    return p[::max(1,int(np.ceil(len(p)/limit)))]


def rigid(left,right):
    a,b=left.mean(0),right.mean(0)
    u,s,v=np.linalg.svd((left-a).T@(right-b))
    if s[1]<1e-9:raise ValueError('ICP 对应点退化')
    r=v.T@u.T
    if np.linalg.det(r)<0:v[-1]*=-1;r=v.T@u.T
    out=np.eye(4);out[:3,:3]=r;out[:3,3]=b-r@a
    return out


def refine_icp(source,target,initial):
    pose=np.asarray(initial,float).copy();total=0;started=time.monotonic()
    if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('ICP 初值无效')
    for voxel,gate in ((.05,.12),(.025,.08),(.015,.05)):
        left,right=sample(source,voxel),sample(target,voxel);tree=cKDTree(right);converged=False
        for _ in range(30):
            if time.monotonic()-started>1.:raise ValueError('ICP 恢复计算超时')
            warped=left@pose[:3,:3].T+pose[:3,3]
            distances,ids=tree.query(warped,workers=2)
            good=distances<gate
            if good.sum()<50 or good.mean()<.3:raise ValueError('ICP 当前帧对应不足')
            threshold=np.quantile(distances[good],.8)
            good&=distances<=threshold
            delta=rigid(warped[good],right[ids[good]])
            pose=delta@pose;total+=1
            angle=np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))
            if np.linalg.norm(delta[:3,3])<1e-4 and angle<1e-4:
                converged=True;break
        if not converged:raise ValueError('ICP 恢复尚未收敛')
    # An old MAC hypothesis cannot drag the current cloud an arbitrary distance.
    change=np.linalg.inv(initial)@pose
    if np.linalg.norm(change[:3,3])>.15 or np.trace(change[:3,:3])<1+2*np.cos(np.deg2rad(15)):
        raise ValueError('MAC 初值与当前帧变化过大')
    return pose,total


def register(source,target,initial,diagnostics,settings,verify_rgbd=None):
    from bufferx_backend import validate_registration
    start=time.perf_counter();method=diagnostics.get('method')
    if method=='rgbd_orb_pnp':
        if verify_rgbd is None:raise ValueError('缺少当前 RGB-D 独立验证')
        support=verify_rgbd(initial)
        if support is None:raise ValueError('RGB-D 对应的双帧深度不一致')
        pose=initial;iterations=0;label='RGB-D ORB/PnP'
    else:
        pose,iterations=refine_icp(source,target,initial);support={};label='MAC + ICP'
    points=sample(source,.02);fixed=sample(target,.02)
    moved=points@pose[:3,:3].T+pose[:3,3]
    inliers=int(np.sum(cKDTree(fixed).query(moved,workers=2)[0]<settings.get('validation_distance_m',.06)))
    result=dict(pose=np.asarray(pose).tolist(),inliers=inliers,points=[len(points),len(fixed)],
        registration_method=label,independent_recovery=True,converged=True,iterations=iterations,
        initialization=diagnostics,depth_support=support,matching_ms=(time.perf_counter()-start)*1000)
    validate_registration(source,target,result,settings,label)
    return result
