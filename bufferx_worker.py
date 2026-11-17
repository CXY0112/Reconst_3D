"""Persistent official BUFFER-X inference worker, isolated from UI/RTAB DLLs."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
PROTOCOL=sys.stdout


class Matcher:
    def __init__(self,settings):
        import numpy as np
        import torch
        self.np=np;self.torch=torch
        torch.set_num_threads(settings.get('threads',4))
        self.device=settings.get('device','cpu')
        source=Path(settings['source'])
        sys.path.insert(0,str(source))
        if self.device=='cpu':
            from bufferx_cpu_ops import install
            install()
        from config import make_cfg
        from models.BUFFERX import BufferX
        from utils.tools import sphericity_based_voxel_analysis
        import open3d as o3d
        self.o3d=o3d;self.voxel_analysis=sphericity_based_voxel_analysis
        cfg=make_cfg('3DMatch');cfg.stage='test'
        cfg.test.pose_refine=bool(settings.get('pose_refine',True));cfg.test.enable_timing=False
        # Keep all official scales and 1500 FPS points by default. Optional
        # overrides are explicitly recorded; they are not silently reduced on CPU.
        for key in ('num_fps','num_points_radius_estimate'):
            if key in settings:cfg.patch[key]=int(settings[key])
        cfg.match.enable_early_exit=False;cfg.match.pose_estimator='ransac'
        self.cfg=cfg;self.model=BufferX(cfg)
        for stage in ('Desc','Pose'):
            path=Path(settings['weights'])/stage/'best.pth'
            state=torch.load(path,map_location='cpu',weights_only=True)
            selected={key:value for key,value in state.items() if key.startswith(stage+'.')}
            expected={key for key in self.model.state_dict() if key.startswith(stage+'.')}
            if set(selected)!=expected:raise RuntimeError(f'{stage} 权重与官方网络不匹配')
            self.model.load_state_dict({**self.model.state_dict(),**selected},strict=True)
        self.model.to(self.device).eval()
        matching=self.model.mutual_matching
        def checked_matching(*args):
            source_ids,target_ids=matching(*args)
            if source_ids.numel()<3:raise ValueError('BUFFER-X 双向特征对应不足 3 对')
            return source_ids,target_ids
        self.model.mutual_matching=checked_matching
        solve=self.model.pose_estimator.estimate_pose
        def checked_solve(source,target,ids):
            if len(ids)<3:raise ValueError('BUFFER-X 共识内点不足 3 对')
            return solve(source,target,ids)
        self.model.pose_estimator.estimate_pose=checked_solve
        if self.device=='cpu':
            # Descriptor patches are independent in eval mode. Chunking keeps
            # CPU RAM bounded without changing keypoints, scales or descriptors.
            original=self.model.Desc.forward
            def chunked(pts,kpts,*args,**kwargs):
                parts=[original(pts,chunk,*args,**kwargs) for chunk in kpts.split(24,dim=1)]
                return {key:torch.cat([part[key] for part in parts],dim=0) for key in parts[0]}
            self.model.Desc.forward=chunked
        self.settings=settings
        self.parameters=dict(num_fps=cfg.patch.num_fps,num_scales=cfg.patch.num_scales,
            num_points_per_patch=cfg.patch.num_points_per_patch,early_exit=False,
            pose_refine=cfg.test.pose_refine,pose_estimator='ransac',descriptor_batch=24 if self.device=='cpu' else None)

    def estimate(self,source,target):
        np=self.np;torch=self.torch;started=time.perf_counter()
        torch.manual_seed(123);np.random.seed(123);self.o3d.utility.random.seed(123)
        clouds=[]
        for points in (source,target):
            points=np.asarray(points,np.float64)
            if points.ndim!=2 or points.shape[1]!=3:raise ValueError('需要 N×3 米制点云')
            points=points[np.isfinite(points).all(1)]
            if len(points)<100:raise ValueError('BUFFER-X 输入点数不足 100')
            cloud=self.o3d.geometry.PointCloud();cloud.points=self.o3d.utility.Vector3dVector(points);clouds.append(cloud)
        voxel,sphericity,_=self.voxel_analysis(*clouds)
        points=[np.asarray(cloud.voxel_down_sample(voxel).points) for cloud in clouds]
        max_points=30000
        points=[p[np.random.choice(len(p),max_points,replace=False)] if len(p)>max_points else p for p in points]
        minimum=max(self.cfg.patch.num_fps,self.cfg.patch.num_points_radius_estimate)
        if min(map(len,points))<minimum:raise ValueError(f'自适应降采样后不足 {minimum} 点，拒绝重复填充特征')
        data=dict(src_fds_pcd=torch.as_tensor(points[0],dtype=torch.float32,device=self.device),
                  tgt_fds_pcd=torch.as_tensor(points[1],dtype=torch.float32,device=self.device),
                  is_aligned_to_global_z=False)
        with torch.inference_mode():pose,times,inliers,matches,consensus,scales=self.model(data)
        if pose is None:raise ValueError('BUFFER-X 未找到刚体变换')
        return dict(pose=np.asarray(pose).tolist(),inliers=int(inliers),matches=int(matches),
                    consensus=int(consensus),scales=int(scales),voxel_m=float(voxel),
                    sphericity=float(sphericity),points=list(map(len,points)),
                    matching_ms=(time.perf_counter()-started)*1000,device=self.device)


def main():
    settings=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    def reply(value):PROTOCOL.write('@@BUFFERX '+json.dumps(value,ensure_ascii=False)+'\n');PROTOCOL.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):matcher=Matcher(settings)
        reply(dict(ready=True,device=matcher.device,parameters=matcher.parameters))
    except Exception as exc:
        reply(dict(error=f'{type(exc).__name__}: {exc}'));return
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request['command']=='close':break
            with contextlib.redirect_stdout(sys.stderr):
                import numpy as np
                with np.load(request['packet'],allow_pickle=False) as data:result=matcher.estimate(data['source'],data['target'])
            reply(result)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply(dict(error=f'{type(exc).__name__}: {exc}',rejected=isinstance(exc,ValueError)))


if __name__=='__main__':main()
