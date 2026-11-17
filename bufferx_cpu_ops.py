"""Inference-only CPU equivalents of BUFFER-X's CUDA geometry operators.

Ball query keeps input order, uses strict radius, and pads with the first hit,
matching pointnet2_ops. No network layers or trained parameters are changed.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def furthest_point_sample(xyz,npoint):
    data=xyz.detach().cpu().numpy();out=[]
    for points in data:
        selected=np.empty(npoint,np.int64);dist=np.full(len(points),1e10);farthest=0
        for i in range(npoint):
            selected[i]=farthest
            delta=np.sum((points-points[farthest])**2,axis=1)
            # The CUDA kernel excludes near-origin padding coordinates.
            delta[np.sum(points**2,axis=1)<=1e-3]=-1
            dist=np.minimum(dist,delta);farthest=int(np.argmax(dist))
        out.append(selected)
    return torch.from_numpy(np.stack(out)).to(xyz.device)


def gather_operation(features,idx):
    return torch.gather(features,2,idx.long()[:,None].expand(-1,features.shape[1],-1))


def grouping_operation(features,idx):
    b,c,_=features.shape
    flat=idx.long().reshape(b,1,-1).expand(-1,c,-1)
    return torch.gather(features,2,flat).reshape(b,c,*idx.shape[1:])


def ball_query(radius,nsample,xyz,new_xyz):
    result=np.zeros((len(xyz),new_xyz.shape[1],nsample),np.int64)
    for b,(points,queries) in enumerate(zip(xyz.detach().cpu().numpy(),new_xyz.detach().cpu().numpy())):
        hits=cKDTree(points).query_ball_point(queries,float(radius),return_sorted=True)
        for i,ids in enumerate(hits):
            if ids:
                ids=np.asarray(ids,np.int64)
                ids=ids[np.sum((points[ids]-queries[i])**2,axis=1)<radius*radius][:nsample]
                if len(ids):result[b,i]=ids[0];result[b,i,:len(ids)]=ids
    return torch.from_numpy(result).to(xyz.device)


class KNN:
    def __init__(self,k,transpose_mode=False):self.k=k;self.transpose_mode=transpose_mode
    def __call__(self,ref,query):
        if not self.transpose_mode:ref=ref.transpose(1,2);query=query.transpose(1,2)
        values=[];indices=[]
        for r,q in zip(ref,query):
            # Block distances to bound peak memory at the official 1500 keypoints.
            blocks=[torch.topk(torch.cdist(part[None],r[None])[0],self.k,largest=False,dim=-1) for part in q.split(256)]
            values.append(torch.cat([v.values for v in blocks]));indices.append(torch.cat([v.indices for v in blocks]))
        return torch.stack(values),torch.stack(indices)


def install():
    """Install aliases inside the isolated worker only; CUDA path never uses them."""
    import sys,types
    ops=types.ModuleType('pointnet2_ops.pointnet2_utils')
    for name in ('furthest_point_sample','gather_operation','grouping_operation','ball_query'):setattr(ops,name,globals()[name])
    package=types.ModuleType('pointnet2_ops');package.pointnet2_utils=ops
    sys.modules['pointnet2_ops']=package;sys.modules['pointnet2_ops.pointnet2_utils']=ops
    knn=types.ModuleType('knn_cuda');knn.KNN=KNN;sys.modules['knn_cuda']=knn
    svd=types.ModuleType('torch_batch_svd');svd.svd=torch.svd;sys.modules['torch_batch_svd']=svd
