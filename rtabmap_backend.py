"""Native RGB-D F2M odometry and RTAB graph; dense geometry only on keyframes."""
import threading
import time
from pathlib import Path
import numpy as np
from rtab_client import RtabClient
from disk_map import CloudStore,DiskVoxelMap
from voxel_map import voxel_groups
from file_snapshots import AtomicJsonWriter


class RtabmapBackend:
    tracking_first=True

    def __init__(self,folder,client_factory=RtabClient,optimizer='default',dense_factory=DiskVoxelMap):
        if optimizer not in ('default','isam2'):raise ValueError('未知轨迹优化方案')
        self.optimizer=optimizer
        self.folder=Path(folder)
        self.client=client_factory(self.folder) if optimizer=='default' else client_factory(self.folder,optimizer=optimizer)
        try:
            self.clouds=CloudStore(self.folder/'map_frames')
            # Dense operations are serialized by dense_lock, on the dense worker.
            self.voxel_map=dense_factory(self.folder/'voxel_map.sqlite',.003,check_same_thread=False)
        except Exception:
            self.client.close(force=True);raise
        self.frames=0;self.timings=np.zeros(5);self.report={}
        self.node_ids=[];self.retained_ids=set();self.poses={};self.closed=False
        self.lock=threading.RLock();self.dense_lock=threading.RLock()
        self.dirty_nodes=set();self.radii={};self.cloud_cache=None;self.preview_stats={}
        self.preview_tolerance_m=self.voxel_map.size/4
        self.fusion_order={};self.fusion_sequence=0
        self.max_history_updates=1
        self.pose_writer=AtomicJsonWriter(self.folder/'geometry_poses.json')

    def _poses(self,rows):
        # Bridge poses map RGB optical coordinates into RTAB's robot world.
        changed=False
        for row in rows:
            pose=np.eye(4);pose[:3]=row[1:].reshape(3,4)
            node=int(row[0]);old=self.poses.get(node)
            changed|=old is None or not np.array_equal(old,pose)
            if node in self.retained_ids and (old is None or not np.array_equal(old,pose)):self.dirty_nodes.add(node)
            self.poses[node]=pose
        if any(node not in self.poses for node in self.node_ids):
            raise RuntimeError('RTAB-Map返回的关键帧位姿不完整')
        if not changed and (self.folder/'geometry_poses.json').exists():
            self.pose_writer.flush(timeout=0,strict=False);return
        self._save_poses()

    def _save_poses(self):
        self.pose_writer.update(dict(node_ids=self.node_ids,poses=[self.poses[n].tolist() for n in self.node_ids],edges=[],backend='rtabmap_rgbd_f2m',optimizer=self.optimizer,dense_backend=getattr(self,'dense_backend','global_voxels')))

    def process(self,color,depth,k,stamp,geometry=None):
        response=self.client.frame(color,depth,k,stamp)
        self.frames+=1;self.timings[:]=response['timings']
        status,node=response['status'],response['node']
        with self.lock:
            reference=self.node_ids[-1] if self.node_ids else None
            self._poses(response['poses'])
        if status==2:
            if node<=0 or node not in self.poses:raise RuntimeError('Invalid native keyframe id')
            if geometry is None:raise ValueError('Keyframe requires complete camera geometry')
        self.report=dict(accepted=status==2,skipped=status==1,node=node if status==2 else None,
                         reference=reference,method='RTAB-Map RGB-D F2M',matching_ms=float(self.timings[0]),
                         pose_checkpoint_pending=self.pose_writer.pending is not None,
                         reason=response['reason'] or ('跟踪丢失：请缓慢返回已观察区域' if status==0 else '已跟踪，尚未形成新关键帧' if status==1 else '原生关键帧已接收'))
        return status

    def retain(self,node,geometry):
        """Called only by the dense worker, after its bounded admission decision."""
        _,sums,counts=voxel_groups(geometry,self.voxel_map.size)
        self.clouds[node]=sums/counts[:,None]
        with self.lock:
            if node in self.retained_ids:raise ValueError('Duplicate retained keyframe')
            self.node_ids.append(node);self.retained_ids.add(node)
            self.radii[node]=float(np.sqrt(np.max(np.sum(np.asarray(geometry)**2,axis=1)))) if len(geometry) else 0.
            self.dirty_nodes.add(node);self._save_poses()

    def error(self):return self.report.get('reason','')

    def preview(self,force=False):
        with self.dense_lock:return self._preview(force)

    def final_preview(self):return self.preview(force=True)

    def pending_updates(self):
        with self.lock:return bool(self.dirty_nodes)

    def _preview(self,force=False):
        with self.lock:
            records=[(node,self.poses[node][:3].copy()) for node in self.node_ids]
            dirty=set(self.node_ids) if force else self.dirty_nodes.copy()
        nodes=[];updated=0;deferred=0;started=time.perf_counter();new=[];history=[]
        for node,pose in records:
            old=self.voxel_map.poses.get(node)
            if old is None or node in dirty:
                # Compare to the last FUSED pose, so sub-threshold changes
                # accumulate instead of being permanently forgotten.
                displacement=np.inf if old is None else np.linalg.norm(old[:,3]-pose[:,3])+self.radii[node]*np.linalg.norm(old[:,:3]-pose[:,:3],ord=2)
                if force or displacement>self.preview_tolerance_m:
                    (new if old is None else history).append((node,pose))
                else:
                    deferred+=1
                    with self.lock:
                        if np.array_equal(self.poses[node][:3],pose):self.dirty_nodes.discard(node)
            t=pose[:,3];nodes.append([-t[1],-t[2],t[0]])
        # A loop adjustment must not monopolize the dense worker with an entire
        # map rebuild. New observations take priority; older corrections drain
        # fairly between arrivals. Finalization explicitly bypasses this budget.
        history.sort(key=lambda row:self.fusion_order.get(row[0],0))
        work=new+(history if force else history[:self.max_history_updates])
        for node,pose in work:
            updated+=int(self.voxel_map.update(node,self.clouds[node],pose))
            self.fusion_sequence+=1;self.fusion_order[node]=self.fusion_sequence
            with self.lock:
                if np.array_equal(self.poses[node][:3],pose):self.dirty_nodes.discard(node)
        fusion_ms=(time.perf_counter()-started)*1000;started=time.perf_counter()
        if updated or self.cloud_cache is None:
            world=self.voxel_map.cloud()
            self.cloud_cache=np.column_stack((-world[:,1],-world[:,2],world[:,0]))
            self.cloud_cache.flags.writeable=False
        with self.lock:pending=len(self.dirty_nodes)
        self.preview_stats=dict(updated_nodes=updated,deferred_small_pose_nodes=deferred,pending_pose_nodes=pending,retained_nodes=len(records),fusion_ms=fusion_ms,cloud_read_ms=(time.perf_counter()-started)*1000,force=force,tolerance_m=self.preview_tolerance_m)
        return self.cloud_cache,np.asarray(nodes).reshape(-1,3)

    def close(self):
        if self.closed:return
        try:
            with self.lock:self.pose_writer.flush()
        finally:
            try:self.client.close()
            finally:
                try:self.voxel_map.close()
                finally:self.closed=True
