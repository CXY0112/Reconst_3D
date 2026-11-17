"""Conservative, metric graph checks independent of the optimizer implementation."""
import numpy as np


def rigid_pose(pose):
    p=np.asarray(pose,dtype=float)
    if p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1],atol=1e-6):raise ValueError('Invalid rigid transformation')
    if not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-3) or abs(np.linalg.det(p[:3,:3])-1)>1e-3:raise ValueError('Non-rigid transformation')
    return p


def residual(poses,source,target,measurement):
    delta=np.linalg.inv(measurement)@np.linalg.inv(poses[target])@poses[source]
    return float(np.linalg.norm(delta[:3,3])),float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))


def validate_graph(before,after,old_edges,new_edges):
    for p in after:rigid_pose(p)
    for source,target,measurement,_,uncertain in old_edges:
        if uncertain:continue
        previous=residual(before,source,target,measurement);current=residual(after,source,target,measurement)
        if current[0]>previous[0]+.020 or current[1]>previous[1]+np.deg2rad(3):
            raise ValueError('回环使已有可靠约束明显恶化，暂不接受')
    for source,target,measurement,_,_ in new_edges:
        distance,angle=residual(after,source,target,measurement)
        if distance>.030 or angle>np.deg2rad(3):raise ValueError('新增约束与优化结果不一致，暂不接受')
