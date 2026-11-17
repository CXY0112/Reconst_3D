"""PointDSC geometric registration, joint pose optimization and voxel fusion."""
import ctypes as ct
import json
import time
import numpy as np
from online_backend import OnlineBackend
from geometry_client import GeometryClient
from voxel_map import voxel_groups


class SnapshotBackend(OnlineBackend):
    def __init__(self, folder,keyframe_translation=0.,keyframe_rotation=0.):
        super().__init__(folder,voxel_size=.003,bridge_dir='online_geometry')
        from disk_map import CloudStore,DiskVoxelMap
        self.clouds=CloudStore(self.folder/'map_frames')
        self.voxel_map=DiskVoxelMap(self.folder/'voxel_map.sqlite',self.voxel_map.size)
        self.geometry_engine = GeometryClient(self.folder,asynchronous=True)
        self.node_ids,self.geometry_poses = [],[]
        self.reason = ''
        self.keyframe_translation=keyframe_translation;self.keyframe_rotation=keyframe_rotation
        self.report = {}
        self.lib.snapshot_add.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
                                         ct.c_void_p,ct.c_double,ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.snapshot_add.restype = ct.c_int
        self.lib.geometry_sync.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int]
        self.lib.geometry_sync.restype = ct.c_int

    def error(self):
        return self.reason or super().error()

    def process(self,color,depth,k,stamp,geometry=None):
        started = time.perf_counter()
        self.frames += 1
        self.reason = ''
        try:
            full = geometry if geometry is not None else self.full_points(depth,k)
            report = self.geometry_engine.register(full)
            if not report['accepted']:
                self.report = report
                self.reason = report['reason']
                self.timings[:] = 0
                self.timings[0] = (time.perf_counter()-started)*1000
                return 0
            poses = np.asarray(report.pop('poses'))
            edges = report.pop('edges')
            if self.node_ids and (getattr(self,'keyframe_translation',0)>0 or getattr(self,'keyframe_rotation',0)>0):
                delta=np.linalg.inv(self.geometry_poses[-1])@poses[-1]
                distance=np.linalg.norm(delta[:3,3]);angle=np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))
                if distance<self.keyframe_translation and angle<self.keyframe_rotation:
                    self.geometry_engine.discard_capture()
                    self.report=dict(accepted=False,skipped=True,reason='相对上一关键帧不足3 cm / 5°',translation_m=float(distance),rotation_deg=float(np.rad2deg(angle)))
                    self.timings[:]=0;self.timings[0]=(time.perf_counter()-started)*1000
                    return 1
            color,depth = np.ascontiguousarray(color,np.uint8),np.ascontiguousarray(depth,np.uint16)
            intrinsics = np.array([k[0,0],k[1,1],k[0,2],k[1,2]],float)
            pose = np.ascontiguousarray(poses[-1,:3])
            node = ct.c_int()
            result = self.lib.snapshot_add(self.handle,color.ctypes.data,depth.ctypes.data,depth.shape[1],depth.shape[0],
                                           intrinsics.ctypes.data,stamp,pose.ctypes.data,ct.byref(node))
            if result < 0:
                raise RuntimeError(super().error())
            if result != 2:
                raise ValueError('RTAB-Map 未创建几何拍摄节点，原始数据已保留')
            ids = self.node_ids+[node.value]
            records = np.asarray([[i,*p[:3].ravel()] for i,p in zip(ids,poses)],float)
            links = np.asarray([[ids[e['source']],ids[e['target']],*np.asarray(e['pose'])[:3].ravel()] for e in edges],float).reshape(-1,14)
            if self.lib.geometry_sync(self.handle,records.ctypes.data,len(records),links.ctypes.data,len(links)):
                raise RuntimeError(super().error())
            self.geometry_engine.request('commit')
            self.node_ids,self.geometry_poses = ids,poses
            _,sums,counts = voxel_groups(full,self.voxel_map.size)
            self.clouds[node.value] = sums/counts[:,None]
            self.lib.online_times(self.handle,self.timings.ctypes.data)
            self.timings[1] = report.get('graph_ms',0.)
            self.timings[0] = (time.perf_counter()-started)*1000
            self.timings[2:5] = 0
            reference = report['reference']
            self.report = {**report,'reference':None if reference is None else ids[reference],
                           'node':node.value,'matching_ms':self.timings[0]}
            (self.folder/'geometry_poses.json').write_text(json.dumps({'node_ids':ids,'poses':poses.tolist(),'edges':edges}),encoding='utf-8')
            return 2
        except ValueError as exc:
            self.reason = str(exc)
            self.timings[:] = 0
            self.timings[0] = (time.perf_counter()-started)*1000
            self.report = {'accepted':False,'reason':self.reason}
            return 0

    def poll_background(self):
        candidate=self.geometry_engine.poll_loop()
        if candidate is None:return False
        if not candidate.get('accepted'):
            with (self.folder/'async_loop_results.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(candidate)+'\n')
            return False
        poses=np.asarray(candidate['poses']);edges=candidate['edges'];ids=self.node_ids
        if len(poses)!=len(ids):
            self.geometry_engine.request('discard_loop');raise RuntimeError('Loop graph/node mismatch')
        records=np.asarray([[node,*pose[:3].ravel()] for node,pose in zip(ids,poses)],float)
        links=np.asarray([[ids[e['source']],ids[e['target']],*np.asarray(e['pose'])[:3].ravel()] for e in edges],float).reshape(-1,14)
        if self.lib.geometry_sync(self.handle,records.ctypes.data,len(records),links.ctypes.data,len(links)):
            self.geometry_engine.request('discard_loop');raise RuntimeError(super().error())
        self.geometry_engine.request('commit_loop')
        self.geometry_poses=poses
        path=self.folder/'geometry_poses.pending.json'
        path.write_text(json.dumps(dict(node_ids=ids,poses=poses.tolist(),edges=edges)),encoding='utf-8')
        path.replace(self.folder/'geometry_poses.json')
        with (self.folder/'async_loop_results.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(accepted=True,version=candidate['version'],background=candidate['background']))+'\n')
        return True

    def finish_background(self,timeout=5.):
        deadline=time.monotonic()+timeout
        while self.geometry_engine.background_busy() and time.monotonic()<deadline:
            self.poll_background();time.sleep(.025)
        # Final map is stable before export: unfinished optional loops are cancelled.
        self.geometry_engine.stop_background()

    def preview(self):
        nodes = []
        for node,pose in zip(self.node_ids,self.geometry_poses):
            # Replaces each old node contribution when joint optimization moves it.
            old=self.voxel_map.poses.get(node)
            if old is None or not np.allclose(old,pose[:3],rtol=0,atol=1e-6):
                self.voxel_map.update(node,self.clouds[node],pose[:3])
            t = pose[:3,3]
            nodes.append([-t[1],-t[2],t[0]])
        world = self.voxel_map.cloud()
        return np.column_stack((-world[:,1],-world[:,2],world[:,0])),np.asarray(nodes).reshape(-1,3)

    def close(self):
        try:
            try:self.geometry_engine.close()
            finally:
                if hasattr(self.voxel_map,'close'):self.voxel_map.close()
        finally:
            super().close()

    @staticmethod
    def full_points(depth,k):
        v,u = np.nonzero((depth >= 150)&(depth <= 4000))
        z = depth[v,u].astype(float)/1000
        return np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z))
