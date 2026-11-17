"""Unified CPU/CUDA ABI and RGB-D preprocessing for G-CVO.

CPU: independent float64/OpenMP port of upstream's GN path.
CUDA: ABI wrapper around unchanged official release/v2 source.
Native public convention fixed <- moving; estimate(current,reference) exposes
reference <- current to match the application's other geometry frontends.
"""
import ctypes as ct
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
PARAMETER_KEYS=('ell','ell_min','amplitude','cutoff','feature_ell','feature_amplitude','max_iterations','tolerance',
    'max_neighbors','ell_decay','decay_start','window','indicator_tolerance','kernel_type','metric_gate',
    'squared_euclidean_gate','use_features','connection','symmetrize','ell2_regularization','ema','neighbor_decay','threads')


def settings_for(device=None):
    path=Path(os.environ.get('RECONST_GCVO_CONFIG',ROOT/'vendor/gcvo/runtime.json'))
    settings=json.loads(path.read_text(encoding='utf-8'))
    if device is not None:settings['device']=device
    if settings['device'] not in ('cpu','cuda'):raise ValueError('G-CVO device 必须为 cpu 或 cuda')
    extension='.dll' if os.name=='nt' else '.so'
    filename=('' if os.name=='nt' else 'lib')+'gcvo_'+settings['device']+extension
    path=Path(settings.get('library',ROOT/'vendor/gcvo/bin'/filename))
    settings['library']=str(path.resolve())
    if not path.is_file():raise RuntimeError(f'G-CVO {settings["device"]} 库未构建：{path}')
    return settings


def covariance(points,neighbors=12,minimum=1e-5,maximum=.0025):
    tree=cKDTree(points);ids=tree.query(points,k=min(neighbors,len(points)),workers=2)[1]
    patch=points[ids];delta=patch-patch.mean(axis=1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',delta,delta)/max(1,patch.shape[1]-1)
    values,vectors=np.linalg.eigh(cov)
    return np.ascontiguousarray((vectors*np.clip(values,minimum,maximum)[:,None,:])@vectors.transpose(0,2,1))


def prepare(points,settings,color=None,k=None,depth=None):
    start=time.perf_counter();points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('需要米制 N×3 点云')
    points=points[np.isfinite(points).all(1)&(points[:,2]>.1)&(points[:,2]<5)]
    if len(points)<100:raise ValueError('G-CVO 有效点数不足 100')
    # Deterministic input cap and voxel centroids, explicitly saved in runtime.
    stride=max(1,int(np.ceil(len(points)/settings.get('input_limit',60000))));points=points[::stride]
    voxel=settings.get('voxel_m',.03);budget=settings.get('max_points',3000)
    for _ in range(5):
        keys=np.floor(points/voxel).astype(np.int64)
        shifted=keys-keys.min(axis=0)
        if shifted.max()<(1<<21):
            packed=(shifted[:,0]<<42)|(shifted[:,1]<<21)|shifted[:,2]
            _,inverse=np.unique(packed,return_inverse=True)
        else:_,inverse=np.unique(keys,axis=0,return_inverse=True)
        counts=np.bincount(inverse)
        reduced=np.column_stack([np.bincount(inverse,weights=points[:,j])/counts for j in range(3)])
        if len(reduced)<=budget:break
        voxel*=max(1.1,np.sqrt(len(reduced)/budget))
    if len(reduced)>budget:reduced=reduced[np.linspace(0,len(reduced)-1,budget,dtype=int)]
    features=np.empty((len(reduced),0))
    if settings.get('use_features'):
        if color is None or k is None:raise ValueError('G-CVO RGB 核需要已对齐 RGB 与内参')
        h,w=color.shape[:2];u=np.rint(reduced[:,0]/reduced[:,2]*k[0,0]+k[0,2]).astype(int)
        v=np.rint(reduced[:,1]/reduced[:,2]*k[1,1]+k[1,2]).astype(int)
        good=(u>=0)&(u<w)&(v>=0)&(v<h)
        if depth is not None:
            if depth.shape!=(h,w):raise ValueError('Registered depth shape differs from RGB')
            z=depth[np.clip(v,0,h-1),np.clip(u,0,w-1)].astype(float)/1000
            good&=(z>0)&(np.abs(z-reduced[:,2])<=settings.get('color_depth_tolerance_m',.03))
        reduced=reduced[good]
        # Registered RGB input is OpenCV BGR, values normalized to [0,1].
        features=np.ascontiguousarray(color[v[good],u[good],::-1],dtype=float)/255
    if len(reduced)<100:raise ValueError('G-CVO 采样后有效点不足 100')
    cov=covariance(reduced,settings.get('cov_neighbors',12),settings.get('cov_eig_min',1e-5),settings.get('cov_eig_max',.0025))
    return dict(points=np.ascontiguousarray(reduced),covariances=cov,features=features,
        voxel_m=voxel,input_stride=stride,prepare_ms=(time.perf_counter()-start)*1000)


class GCVOSolver:
    def __init__(self,device=None,settings=None):
        self.settings=dict(settings or settings_for(device))
        if device:self.settings['device']=device
        self.lib=ct.CDLL(self.settings['library']);self.lib.gcvo_last_error.restype=ct.c_char_p;self.lib.gcvo_backend.restype=ct.c_char_p
        if hasattr(self.lib,'gcvo_device_available') and self.lib.gcvo_device_available()!=1:
            raise RuntimeError(self.lib.gcvo_last_error().decode())
        ptr=ct.POINTER(ct.c_double)
        self.lib.gcvo_align.argtypes=[ptr,ct.c_int,ptr,ct.c_int,ptr,ptr,ptr,ptr,ct.c_int,ptr,ptr,ptr,ptr]
        self.lib.gcvo_align.restype=ct.c_int
        self.parameters=np.array([self.settings[key] for key in PARAMETER_KEYS],dtype=float)
        if not np.isfinite(self.parameters).all() or min(self.parameters[:6])<=0:raise ValueError('Invalid G-CVO parameters')
        if self.settings['kernel_type'] not in (0,1,2):raise ValueError('Invalid G-CVO kernel')
        if (not 1<=self.settings['max_iterations']<=2000 or not 1<=self.settings['max_neighbors']<=4096
                or not 1<=self.settings['threads']<=64 or not 1<=self.settings['window']
                or not 0<self.settings['ell_min']<=self.settings['ell']
                or not 0<self.settings['cutoff']<self.settings['amplitude']**2
                or not 0<self.settings['ell_decay']<1
                or self.settings['tolerance']<=0):raise ValueError('Invalid G-CVO iteration/kernel bounds')
        self.backend=self.lib.gcvo_backend().decode()
        if self.settings['device'] not in self.backend:raise RuntimeError('G-CVO 库与请求设备不一致，拒绝静默切换后端')

    def align(self,fixed,moving,initial=None,*,max_iterations=None,ell=None):
        parameters=self.parameters.copy()
        if max_iterations is not None:
            if not 1<=max_iterations<=2000 or int(max_iterations)!=max_iterations:raise ValueError('Invalid retry iteration budget')
            parameters[6]=max_iterations
        if ell is not None:
            if not self.settings['ell_min']<=ell<=self.settings['ell']:raise ValueError('Invalid continuation kernel scale')
            parameters[0]=ell
        initial=np.ascontiguousarray(np.eye(4) if initial is None else initial,dtype=float)
        if (initial.shape!=(4,4) or not np.isfinite(initial).all() or not np.allclose(initial[3],[0,0,0,1])
                or not np.allclose(initial[:3,:3].T@initial[:3,:3],np.eye(3),atol=1e-5)
                or np.linalg.det(initial[:3,:3])<.99):raise ValueError('Invalid initial SE3')
        arrays=[]
        for item in (fixed,moving):
            points=np.ascontiguousarray(item['points'],dtype=float);cov=np.ascontiguousarray(item['covariances'],dtype=float)
            feat=np.ascontiguousarray(item['features'],dtype=float)
            if (points.ndim!=2 or points.shape[1]!=3 or cov.shape!=(len(points),3,3) or feat.ndim!=2 or len(feat)!=len(points)
                    or len(points)<3 or not all(np.isfinite(a).all() for a in (points,cov,feat))):raise ValueError('Invalid prepared cloud')
            if not np.allclose(cov,cov.transpose(0,2,1),atol=1e-8) or np.linalg.eigvalsh(cov).min()<=0:raise ValueError('Covariance must be SPD')
            arrays.append((points,cov,feat))
        (x,cx,fx),(y,cy,fy)=arrays
        if fx.shape[1]!=fy.shape[1] or fx.shape[1] not in (0,3) or (self.settings['use_features'] and fx.shape[1]!=3):raise ValueError('Feature dimensions do not agree')
        output=np.empty((4,4));stats=np.zeros(9)
        ptr=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_double))
        status=self.lib.gcvo_align(ptr(x),len(x),ptr(y),len(y),ptr(cx),ptr(cy),ptr(fx),ptr(fy),fx.shape[1],ptr(initial),ptr(parameters),ptr(output),ptr(stats))
        if status<0:raise ValueError('G-CVO: '+self.lib.gcvo_last_error().decode())
        result=dict(pose=output,solver_status=status,iterations=int(stats[0]),solver_ms=stats[1]*1000,
            correlation_count=int(stats[2]),objective=float(stats[3]),step_norm=float(stats[4]),
            hessian_eigenvalues=[float(stats[5]),float(stats[6])],capped_rows=int(stats[7]),final_ell=float(stats[8]),implementation=self.backend)
        if self.settings['device']=='cuda':
            for key in ('correlation_count','objective','step_norm','hessian_eigenvalues','capped_rows','final_ell'):result[key]=None
        return result


class GCVOConvergenceError(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def align_tracking(solver,fixed,moving,initial):
    """One bounded warm restart, still requiring the original solver tolerance.

    No failed candidate is committed to tracker state. CPU retains its final
    kernel scale; EMA/window history restarts because the ABI does not expose it.
    CUDA has no final-scale diagnostics and restarts its configured schedule.
    """
    budget=solver.settings.get('retry_iterations',80)
    if not 0<=budget<=2000 or int(budget)!=budget:raise ValueError('Invalid G-CVO retry_iterations')
    result=solver.align(fixed,moving,initial)
    attempts=[{key:value for key,value in result.items() if key!='pose'}]
    if result['solver_status']==1 and budget:
        try:
            result=solver.align(fixed,moving,result['pose'],max_iterations=budget,ell=result.get('final_ell'))
        except ValueError as exc:
            raise GCVOConvergenceError('G-CVO 续算失败：'+str(exc),dict(attempts=attempts,retried=True)) from exc
        attempts.append({key:value for key,value in result.items() if key!='pose'})
    result={**result,'attempts':attempts,'retried':len(attempts)>1,
            'iterations':sum(a['iterations'] for a in attempts),
            'solver_ms':sum(a['solver_ms'] for a in attempts)}
    if result['solver_status']!=0:
        diagnostics={key:value for key,value in result.items() if key!='pose'}
        step=result.get('step_norm')
        suffix=f'，最终步长 {step:.3g}' if step is not None else ''
        raise GCVOConvergenceError(f'G-CVO 在 {result["iterations"]} 次迭代预算内未收敛{suffix}，本帧未提交',diagnostics)
    return result


class GCVOMatcher:
    def __init__(self,folder,device='cpu'):
        self.settings=settings_for(device);self.solver=GCVOSolver(settings=self.settings)
        self.target=None;self.prepared_target=None;self.guess=np.eye(4);self.candidate=None
        self.frame_rgb=None;self.frame_k=None;self.frame_depth=None;self.reference_rgb=None;self.reference_k=None;self.reference_depth=None;self.current_pack=None
        from gcvo_rgbd_seed import RGBDSeed
        self.rgbd_seed=RGBDSeed()
        (Path(folder)/'gcvo_runtime.json').write_text(json.dumps(self.settings,indent=2),encoding='utf-8')
    def set_frame_rgb(self,color,k,depth=None):self.frame_rgb=color;self.frame_k=k;self.frame_depth=depth;self.current_pack=None
    def reference_updated(self,cloud,color,k,depth=None):
        self.rgbd_seed.reset()
        self.target=cloud;self.prepared_target=self.current_pack;self.current_pack=None
        self.reference_rgb=color.copy();self.reference_k=k.copy();self.reference_depth=None if depth is None else depth.copy();self.guess=np.eye(4)
    def estimate(self,source,target,initial=None):
        started=time.perf_counter()
        if self.prepared_target is None:self.prepared_target=prepare(target,self.settings,self.reference_rgb,self.reference_k,self.reference_depth)
        current=prepare(source,self.settings,self.frame_rgb,self.frame_k,self.frame_depth);self.current_pack=current
        self.candidate=None
        result=align_tracking(self.solver,self.prepared_target,current,self.guess if initial is None else initial)
        moved=current['points']@result['pose'][:3,:3].T+result['pose'][:3,3]
        distances=cKDTree(self.prepared_target['points']).query(moved,workers=2)[0]
        result['inliers']=int(np.sum(distances<self.settings['validation_distance_m']))
        result['inlier_definition']='sampled geometry NN distance, not discrete feature correspondences'
        self.candidate=result['pose'].copy()
        return {**result,'pose':result['pose'].tolist(), 'points':[len(current['points']),len(self.prepared_target['points'])],
            'matching_ms':(time.perf_counter()-started)*1000,'prepare_ms':current['prepare_ms'],'device':self.settings['device']}
    def commit_candidate(self):
        if self.candidate is not None:self.guess=self.candidate;self.candidate=None
    def discard_candidate(self):self.candidate=None
    def reset_prediction(self):self.guess=np.eye(4);self.candidate=None
    def recovery_seed(self):
        import cv2
        try:return self.rgbd_seed.estimate(self.reference_rgb,self.reference_depth,self.reference_k,self.frame_rgb,self.frame_k)
        except cv2.error:return None  # Degenerate image/PnP cannot stop acquisition.
    def estimate_fallback(self,source,target,initial,diagnostics):
        from gcvo_fallback import register
        self.candidate=None
        result=register(source,target,initial,diagnostics,self.settings,
                        verify_rgbd=lambda pose:self.rgbd_seed.verify_depth(pose,self.frame_depth))
        self.candidate=np.asarray(result['pose']).copy()
        return result
    def close(self):pass
