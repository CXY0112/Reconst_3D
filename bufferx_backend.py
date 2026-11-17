"""Geometry frontend + native visual loop closure/iSAM2 + existing submaps."""
import json
import time
from functools import partial
import numpy as np
from scipy.spatial import cKDTree
from submap_backend import SubmapBackend
from rtab_client import RtabClient
from bufferx_client import BufferXClient

OPTICAL_WORLD=np.array([[0,0,1.,0],[-1.,0,0,0],[0,-1.,0,0],[0,0,0,1.]])


def validate_registration(source,target,result,settings,name='BUFFER-X'):
    pose=np.asarray(result['pose'],float)
    if (pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1])
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-3)
            or np.linalg.det(pose[:3,:3])<.99):raise ValueError(name+' 返回无效刚体变换')
    if result.get('inliers',0)<settings.get('min_inliers',8):raise ValueError(name+' 有效内点不足')
    distance=float(np.linalg.norm(pose[:3,3]));angle=float(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1)))
    if distance>settings.get('max_translation_m',1.) or angle>np.deg2rad(settings.get('max_rotation_deg',90)):
        raise ValueError(name+' 相对参考帧运动过大，拒绝位姿跳变')
    threshold=settings.get('validation_distance_m',.06)
    # Deterministic samples bound validation cost; evaluate both directions.
    left=np.asarray(source)[::max(1,len(source)//6000)]
    right=np.asarray(target)[::max(1,len(target)//6000)]
    warped=left@pose[:3,:3].T+pose[:3,3]
    forward=cKDTree(right).query(warped,workers=2)[0];reverse=cKDTree(warped).query(right,workers=2)[0]
    overlap=float(min(np.mean(forward<threshold),np.mean(reverse<threshold)))
    good=forward[forward<threshold];rmse=float(np.sqrt(np.mean(good**2))) if len(good) else float('inf')
    if overlap<settings.get('min_overlap',.25) or rmse>settings.get('max_rmse_m',.04):
        raise ValueError(f'{name} 几何验证未通过：重叠 {overlap:.1%}，RMSE {rmse*1000:.1f} mm')
    return pose,dict(overlap=overlap,rmse_m=rmse,translation_m=distance,rotation_deg=float(np.rad2deg(angle)))


class BufferXBackend(SubmapBackend):
    frontend_name='BUFFER-X'
    backend_name='bufferx'
    def __init__(self,folder,optimizer='isam2',matcher_factory=BufferXClient,client_factory=None):
        self.matcher=matcher_factory(folder)
        try:
            super().__init__(folder,client_factory=client_factory or partial(RtabClient,bridge_dir='online_bufferx'),optimizer=optimizer)
        except Exception:
            self.matcher.close()
            raise
        if not self.client.optimizer_info.get('external_pose'):
            self.close();raise RuntimeError(self.frontend_name+' 需要支持外部位姿的原生后端')
        self.reference_node=None;self.reference_cloud=None;self.reference_odom=None
        self.settings=getattr(self.matcher,'settings',{})

    def _save_poses(self):
        self.pose_writer.update(dict(node_ids=self.node_ids,poses=[self.poses[n].tolist() for n in self.node_ids],
            edges=[],backend=self.backend_name,optimizer=self.optimizer,dense_backend='local_voxel_submaps_v1',
            loop_closure='rtabmap_visual',reference_mode='last_native_keyframe'))

    def _before_submit(self,details,metrics,stamp,initial=False):pass
    def _registration_failed(self,exc,details,metrics,stamp):pass
    def _after_submit(self,status,stamp):pass
    def _estimate(self,source,target,stamp):return self.matcher.estimate(source,target)

    def process(self,color,depth,k,stamp,geometry=None):
        started=time.perf_counter();self.frames+=1
        if geometry is None:raise ValueError(self.frontend_name+' 配准需要完整点云')
        cloud=np.asarray(geometry,float);cloud=cloud[np.isfinite(cloud).all(1)]
        reference=self.reference_node;details={};metrics={}
        if hasattr(self.matcher,'set_frame_rgb'):self.matcher.set_frame_rgb(color,k,depth)
        try:
            if len(cloud)<100:raise ValueError(self.frontend_name+' 有效点云不足')
            if reference is None:world=OPTICAL_WORLD.copy()
            else:
                details=self._estimate(cloud,self.reference_cloud,stamp)
                relative,metrics=validate_registration(cloud,self.reference_cloud,details,self.settings,self.frontend_name)
                # Pose convention: source(current) -> target(reference).
                # Native map.process expects a continuous ODOMETRY frame. Its
                # map->odom correction is applied by RTAB itself. Feeding the
                # optimized reference here would apply loop correction twice.
                world=self.reference_odom@relative
            self._before_submit(details,metrics,stamp,initial=reference is None)
        except ValueError as exc:
            self._registration_failed(exc,details,metrics,stamp)
            if hasattr(self.matcher,'discard_candidate'):self.matcher.discard_candidate()
            self.timings[:]=0;self.timings[0]=(time.perf_counter()-started)*1000
            self.report=dict(accepted=False,skipped=False,reference=reference,method=self.frontend_name,reason=str(exc),matching_ms=float(self.timings[0]))
            diagnostics=getattr(exc,'diagnostics',details)
            if diagnostics:self.report['frontend']={key:value for key,value in diagnostics.items() if key!='pose'}
            if metrics:self.report['validation']=metrics
            self._record(stamp);return 0
        # Worker infrastructure/runtime errors propagate; never silently run VO.
        variance=max(.0004,metrics.get('rmse_m',.02)**2)
        response=self.client.frame(color,depth,k,stamp,pose=world,variance=variance)
        self.timings[:]=response['timings'];self.timings[0]=(time.perf_counter()-started)*1000
        self.timings[2:]=[details.get('inliers',0),sum(details.get('points',[0,0])),details.get('matches',0)]
        with self.lock:self._poses(response['poses'])
        status,node=response['status'],response['node']
        if status>0 and hasattr(self.matcher,'commit_candidate'):self.matcher.commit_candidate()
        elif status<=0 and hasattr(self.matcher,'discard_candidate'):self.matcher.discard_candidate()
        if status==2:
            if node<=0 or node not in self.poses:raise RuntimeError(self.frontend_name+' 原生关键帧位姿缺失')
            self.reference_node=node;self.reference_cloud=cloud.copy();self.reference_odom=world.copy()
            if hasattr(self.matcher,'reference_updated'):self.matcher.reference_updated(self.reference_cloud,color,k,depth)
        self._after_submit(status,stamp)
        self.report=dict(accepted=status==2,skipped=status==1,node=node if status==2 else None,
            reference=reference,method=self.frontend_name,reason=response.get('reason') or (self.frontend_name+' 关键帧已接收' if status==2 else '已配准，尚未形成新关键帧' if status==1 else '原生后端未接受候选位姿'),
            matching_ms=float(self.timings[0]),validation=metrics,
            frontend={key:value for key,value in details.items() if key!='pose'})
        self._record(stamp);return status

    def _record(self,stamp):
        with (self.folder/(self.backend_name+'_metrics.jsonl')).open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(stamp=stamp,**self.report),ensure_ascii=False)+'\n')

    def close(self):
        try:super().close()
        finally:self.matcher.close()
