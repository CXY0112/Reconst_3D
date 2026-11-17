"""CPU adaptation of basic MAC's graph/clique/SVD stages.

Reference: zhangxy0517/3D-Registration-with-Maximal-Cliques, MIT;
see vendor/mac/source/LICENSE. No RANSAC or maximum-clique substitution.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist


def rigid(source,target,weights=None):
    w=np.ones(len(source)) if weights is None else np.asarray(weights)
    w=w/w.sum();a=w@source;b=w@target
    h=((source-a)*w[:,None]).T@(target-b)
    u,s,vt=np.linalg.svd(h)
    if s[1]<1e-10:raise ValueError('MAC degenerate collinear consensus')
    d=np.eye(3);d[2,2]=np.linalg.det(vt.T@u.T)
    r=vt.T@d@u.T;t=b-r@a
    pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=t
    return pose


def maximal_cliques(adjacency,deadline,limit):
    """Pivoted Bron-Kerbosch using bit sets; exhaustion rejects the job."""
    neighbors=[sum(1<<int(j) for j in np.flatnonzero(row)) for row in adjacency]
    cliques=[]
    def visit(r,p,x):
        if time.monotonic()>deadline:raise ValueError('MAC clique search timed out')
        if not p and not x:
            if len(r)>=3:
                if len(cliques)>=limit:raise ValueError('MAC maximal-clique budget exceeded')
                cliques.append(r)
            return
        union=p|x
        pivot=max((i for i in range(len(neighbors)) if union>>i&1),key=lambda i:(p&neighbors[i]).bit_count(),default=None)
        todo=p if pivot is None else p&~neighbors[pivot]
        while todo:
            bit=todo&-todo;v=bit.bit_length()-1
            visit(r+[v],p&neighbors[v],x&neighbors[v]);p&=~bit;x|=bit;todo&=~bit
    visit([], (1<<len(neighbors))-1,0)
    return cliques


def solve(source,target,settings):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3 or not np.isfinite([source,target]).all():
        raise ValueError('MAC requires finite paired N x 3 correspondences')
    if len(source)>settings.get('max_correspondences',256):raise ValueError('MAC correspondence budget exceeded')
    began=time.monotonic();sigma=settings.get('graph_sigma_m',.3)
    discrepancy=np.abs(cdist(source,source)-cdist(target,target))
    first=np.maximum(0,1-(discrepancy/sigma)**2);np.fill_diagonal(first,0)
    first[first<settings.get('graph_threshold',.99)]=0
    second=(first@first)*first
    cliques=maximal_cliques(second>0,began+settings.get('clique_timeout_s',2.),settings.get('max_cliques',10000))
    # Each node votes for its highest-weight maximal clique (official basic MAC).
    best_by_node={}
    for index,clique in enumerate(cliques):
        weight=float(np.triu(second[np.ix_(clique,clique)],1).sum())
        for node in clique:
            if weight>best_by_node.get(node,(-1,None))[0]:best_by_node[node]=(weight,index)
    selected=sorted(set(index for weight,index in best_by_node.values() if weight>0))
    threshold=settings.get('correspondence_distance_m',.06)
    def evaluate(pose):
        residual=np.linalg.norm(source@pose[:3,:3].T+pose[:3,3]-target,axis=1)
        return float(np.maximum(0,1-residual/threshold).sum()),residual
    best=None;score=-1
    for index in selected:
        ids=cliques[index]
        try:pose=rigid(source[ids],target[ids])
        except ValueError:continue
        value,residual=evaluate(pose)
        if value>score:score=value;best=pose
    if best is None:raise ValueError('MAC found no non-degenerate maximal-clique hypothesis')
    for _ in range(20):
        _,residual=evaluate(best);good=residual<threshold
        if good.sum()<3:break
        try:pose=rigid(source[good],target[good],1/(1+(residual[good]/threshold)**2))
        except ValueError:break
        value,_=evaluate(pose)
        if value<=score+1e-8:break
        score=value;best=pose
    _,residual=evaluate(best);inliers=int(np.sum(residual<threshold))
    if inliers<settings.get('min_correspondence_inliers',12) or inliers/len(source)<settings.get('min_inlier_ratio',.1):
        raise ValueError('MAC correspondence support insufficient')
    return dict(pose=best.tolist(),inliers=inliers,matches=len(source),cliques=len(cliques),hypotheses=len(selected),
                correspondence_score=score,solver_ms=(time.monotonic()-began)*1000,solver_status=0)
