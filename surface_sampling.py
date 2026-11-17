"""Deterministic surface-aware maximal minimum-distance sampling, no new geometry."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def _coordinate_hash(points, spacing):
    cells=np.floor(np.asarray(points,float)/(spacing/1024)).astype(np.int64).view(np.uint64)
    with np.errstate(over='ignore'):
        value=cells[:,0]*np.uint64(73856093)^cells[:,1]*np.uint64(19349663)^cells[:,2]*np.uint64(83492791)
        value^=value>>np.uint64(30);value*=np.uint64(0xbf58476d1ce4e5b9)
        value^=value>>np.uint64(27);value*=np.uint64(0x94d049bb133111eb)
        value^=value>>np.uint64(31)
    return value


def spacing_metrics(points,normals,spacing,sample_count=15000,cancel=None):
    from preprocess import _check
    _check(cancel)
    points=np.asarray(points,float);normals=np.asarray(normals,float)
    if len(points)<2:return {'samples':0}
    ids=np.random.default_rng(20261015).choice(len(points),min(sample_count,len(points)),replace=False)
    d,near=cKDTree(points).query(points[ids],k=min(17,len(points)),workers=2)
    d=d[:,1:];near=near[:,1:]
    offset=points[near]-points[ids,None]
    dot=np.einsum('nki,ni->nk',normals[near],normals[ids])
    height=np.maximum(abs(np.einsum('nki,nki->nk',offset,normals[near])),
                      abs(np.einsum('nki,ni->nk',offset,normals[ids])))
    compatible=(dot>=np.cos(np.deg2rad(40)))&(height<=spacing*.5)
    same=np.min(np.where(compatible,d,np.inf),axis=1);finite=np.isfinite(same)
    return {'samples':len(ids),'nearest_mm_p10_p50_p90':np.percentile(d[:,0]*1000,[10,50,90]).tolist(),
            'same_surface_neighbor_samples':int(finite.sum()),
            'same_surface_nearest_mm_p10_p50_p90':np.percentile(same[finite]*1000,[10,50,90]).tolist() if finite.any() else [],
            'same_surface_fraction_closer_than_target':float(np.mean(same[finite]<spacing*(1-1e-6))) if finite.any() else 0.,
            'note':'候选邻居最多16个；法线和法向带宽近似同表面，不是全地图真值表面积密度。'}


def uniform_surface_indices(points,normals,spacing,support=None,sigma_mm=None,progress=None,cancel=None):
    """Greedy maximal independent set in the compatible close-neighbor graph.

    Retain original positions/attributes; every removed point is covered by a kept
    compatible point within spacing. Different normals and resolved layers survive.
    """
    from preprocess import _check,_notify
    started=time.perf_counter();_check(cancel)
    points=np.asarray(points,float);normals=np.asarray(normals,float)
    if (points.ndim!=2 or points.shape[1]!=3 or normals.shape!=points.shape or
        not np.isfinite(points).all() or not np.isfinite(normals).all() or not np.isfinite(spacing) or spacing<=0):
        raise ValueError('表面采样需要有限XYZ、法线和正间距')
    n=len(points)
    if n==0:return np.empty(0,np.int64),{'input_points':0,'output_points':0}
    support=np.ones(n,int) if support is None else np.asarray(support)
    sigma_mm=np.ones(n,float) if sigma_mm is None else np.asarray(sigma_mm)
    if support.shape!=(n,) or sigma_mm.shape!=(n,) or not np.isfinite(sigma_mm).all() or not np.isfinite(support).all():
        raise ValueError('采样属性与点数不一致或包含无效值')
    norms=np.linalg.norm(normals,axis=1);valid=norms>1e-8
    normals=np.where(valid[:,None],normals/np.maximum(norms[:,None],1e-12),0.)
    before=spacing_metrics(points,normals,spacing,cancel=cancel)
    _notify(progress,.1,'查找跨体素的过密邻居')
    tree=cKDTree(points)
    probes=np.linspace(0,n-1,min(n,2048),dtype=int)
    estimated_neighbors=tree.query_ball_point(points[probes],spacing,return_length=True,workers=2)
    if float(np.mean(estimated_neighbors))*n/2>15000000:
        raise ValueError('目标间距相对输入过大，会产生过多邻域连接；请先增大输出体素或减小均匀间距')
    pairs=tree.query_pairs(spacing,output_type='ndarray')
    _check(cancel);keep_pair=np.zeros(len(pairs),bool)
    cosine=np.cos(np.deg2rad(40))
    for a in range(0,len(pairs),200000):
        _check(cancel);b=min(len(pairs),a+200000);left,right=pairs[a:b].T
        delta=points[right]-points[left]
        dot=np.einsum('ni,ni->n',normals[left],normals[right])
        height=np.maximum(abs(np.einsum('ni,ni->n',delta,normals[left])),
                          abs(np.einsum('ni,ni->n',delta,normals[right])))
        keep_pair[a:b]=(valid[left]&valid[right]&(dot>=cosine)&(height<=spacing*.5)&
                        (np.sum(delta*delta,axis=1)<(spacing*(1-1e-9))**2))
    pairs=pairs[keep_pair]
    # CSR avoids per-point Python neighbor lists and bounds work by local graph size.
    graph=csr_matrix((np.ones(len(pairs),bool),(pairs[:,0],pairs[:,1])),shape=(n,n))
    graph=graph+graph.T
    hashes=_coordinate_hash(points,spacing)
    # Limit confidence priority so repeated scans do not request extra output density.
    order=np.lexsort((points[:,2],points[:,1],points[:,0],hashes,np.round(sigma_mm*2),-np.minimum(support,8)))
    blocked=np.zeros(n,bool);chosen=[]
    _notify(progress,.35,'按同一目标间距选择表面点，保护法线冲突区域')
    for offset,index in enumerate(order):
        if offset%8192==0:
            _check(cancel);_notify(progress,.35+.5*offset/n,'全局表面均匀采样')
        if blocked[index]:continue
        chosen.append(index)
        blocked[graph.indices[graph.indptr[index]:graph.indptr[index+1]]]=True
    ids=np.sort(np.asarray(chosen,np.int64));_check(cancel)
    after=spacing_metrics(points[ids],normals[ids],spacing,cancel=cancel)
    sample=np.random.default_rng(42).choice(n,min(n,15000),replace=False)
    coverage=cKDTree(points[ids]).query(points[sample],workers=2)[0]*1000
    report={'algorithm':'surface-aware maximal Poisson-disk-style selection','target_spacing_mm':spacing*1000,
            'input_points':n,'output_points':len(ids),'removed_for_density':n-len(ids),
            'compatible_close_pairs':len(pairs),'before':before,'after':after,
            'sampled_input_to_output_mm_p50_p95_max':np.percentile(coverage,[50,95,100]).tolist(),
            'compatible_removed_point_coverage_bound_mm':spacing*1000,
            'position_changed':False,'new_points_generated':0,'elapsed_seconds':time.perf_counter()-started,
            'surface_gate':'oriented normals <=40 degrees, symmetric normal offset <=half target spacing',
            'limitation':'只减少过密点，不填补稀疏区域或消除错层；不同表面的近点可保留，故不承诺全图欧氏最小间距。'}
    _notify(progress,1.,'表面均匀采样完成')
    return ids,report
