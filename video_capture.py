"""Fixed-rate RGB-D sampling with one latest pending frame; storage runs in mapper."""
from dataclasses import dataclass
import json,queue,threading,time
import cv2
import numpy as np
from snapshot_window import SnapshotSession
from rgbd_geometry import capture_geometry,intrinsics,rgb_geometry,camera_frame,coordinate_metadata
from reconstruct import Calibration
from rtabmap_scan import RGBDCamera,register_rgbd,write_configuration
from rtabmap_backend import RtabmapBackend


class Cadence:
    def __init__(self,hz):
        if not np.isfinite(hz) or not .1<=hz<=10:raise ValueError('采样频率应为0.1至10 Hz')
        self.interval=1./hz;self.deadline=0.;self.last_ids=None;self.last_stamp=None
    def take(self,now,ids,stamp):
        # A small scheduler jitter around a camera's nominal period must not
        # halve sampling rate. Keep the cadence anchored; never catch up a stall.
        if now<self.deadline-self.interval*.05:return False
        if self.last_ids and (ids[0]==self.last_ids[0] or ids[1]==self.last_ids[1]):return False
        if self.last_stamp is not None and stamp<=self.last_stamp:return False
        self.deadline=self.deadline+self.interval if abs(now-self.deadline)<=self.interval*.5 else now+self.interval
        self.last_ids=ids;self.last_stamp=stamp;return True


@dataclass(frozen=True)
class VideoFrame:
    rgb:np.ndarray
    depth:np.ndarray
    points:np.ndarray
    calibration:object
    stamp:float
    ids:tuple
    sdk_xyz:bool
    depth_frame:str='ir'
    points_frame:str|None=None
    tracking_epoch:int=0

    def __post_init__(self):
        camera_frame(self.depth_frame);camera_frame(self.geometry_frame)
        if not self.sdk_xyz and self.geometry_frame!=self.depth_frame:
            raise ValueError('Raw depth and deprojected points must share a frame')

    @property
    def geometry_frame(self):
        return self.points_frame or ('rgb' if self.sdk_xyz else self.depth_frame)


class VideoSession(SnapshotSession):
    def __init__(self,events,camera_factory=RGBDCamera,hz=10.,backend_factory=None,optimizer='default',dense_mode='global_voxels',frontend='rgbd'):
        if optimizer not in ('default','isam2'):raise ValueError('未知轨迹优化方案')
        if dense_mode not in ('global_voxels','submaps'):raise ValueError('未知稠密融合方案')
        if frontend not in ('rgbd','bufferx','gcvo_cpu','gcvo_cuda'):raise ValueError('未知配准前端')
        if frontend!='rgbd' and (optimizer!='isam2' or dense_mode!='submaps'):raise ValueError('几何前端实验需要 iSAM2 与子地图融合')
        if (optimizer!='default' or dense_mode!='global_voxels' or frontend!='rgbd') and backend_factory is not None:raise ValueError('自定义后端不能同时指定轨迹优化、前端或稠密融合方案')
        self.frontend=frontend
        self.optimizer=optimizer
        self.dense_mode=dense_mode
        self.cadence=Cadence(hz);self.video_mode=True;self.source_fps=hz
        super().__init__(events,camera_factory)
        self.backend_factory=backend_factory or RtabmapBackend
        if dense_mode=='submaps':
            from submap_backend import SubmapBackend
            self.backend_factory=SubmapBackend
        if frontend=='bufferx':
            from bufferx_backend import BufferXBackend
            self.backend_factory=BufferXBackend
        if frontend.startswith('gcvo_'):
            from gcvo_backend import GCVOBackend
            from functools import partial
            self.backend_factory=partial(GCVOBackend,device=frontend.split('_')[1])
        if optimizer!='default':
            from functools import partial
            self.backend_factory=partial(self.backend_factory,optimizer=optimizer)
        self.incoming=queue.Queue(maxsize=1);self.paused=threading.Event();self.sampled=0
        self.pause_lock=threading.Lock();self.tracking_epoch=0;self.processed_epoch=0
        self.dense_pending=queue.Queue(maxsize=1);self.dense_thread=None;self.dense_error=None;self.dense_dropped=0
        self.dense_last_capture=0
        (self.folder/'frames.jsonl').touch()
        (self.folder/'video_capture.json').write_text(json.dumps(dict(mode='video',backend='rtabmap_rgbd_f2m' if frontend=='rgbd' else frontend,frontend=frontend,optimizer=optimizer,dense_mode=dense_mode,target_hz=hz,pending_frames=1,policy='latest-only',native_map_max_hz=2,keyframe_translation_m=.03,keyframe_rotation_rad=.03)),encoding='utf-8')

    def toggle_pause(self):
        with self.pause_lock:
            if self.paused.is_set():
                self.tracking_epoch+=1;self.paused.clear();return False
            self.paused.set();return True

    def request_capture(self):return False

    def _should_publish(self,result):return result==2

    def _publish(self,backend,count,final=False):
        if not getattr(backend,'tracking_first',False):super()._publish(backend,count,final)

    def _dense_worker(self,backend):
        from local_preprocessing import prepare_local,save_local
        while True:
            try:item=self.dense_pending.get(timeout=.5)
            except queue.Empty:
                try:
                    if self.dense_error is None and callable(getattr(backend,'pending_updates',None)) and backend.pending_updates():
                        SnapshotSession._publish(self,backend,self.dense_last_capture)
                except Exception as exc:
                    self.dense_error=exc;self.failed=True;self.stop.set()
                    self.events.put(('error',f'后台地图修正失败：{exc}；恢复数据：{self.folder}'))
                continue
            try:
                if item is None:return
                if self.dense_error is not None:continue
                count,packet,color,depth,geometry,report=item
                start=time.perf_counter();name=f'{packet.stamp:.6f}.png'
                prepared=prepare_local(packet.points,self.local_params,coordinate_frame=packet.geometry_frame)
                save_local(self.folder/'pointcloud_local'/f'{packet.stamp:.6f}.npz',prepared)
                for directory,image in [('rgb',color),('depth',depth),('rgb_raw',packet.rgb),('depth_raw',packet.depth)]:
                    if not cv2.imwrite(str(self.folder/directory/name),image):raise OSError('无法保存视频关键帧')
                backend.retain(report['node'],geometry);self.names.append(name)
                self.dense_last_capture=count
                SnapshotSession._publish(self,backend,count)
                with (self.folder/'dense_metrics.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(capture=count,node=report['node'],elapsed_ms=(time.perf_counter()-start)*1000,local_preprocessing=prepared['metadata']))+'\n')
            except Exception as exc:
                self.dense_error=exc;self.failed=True;self.stop.set()
                self.events.put(('error',f'后台关键帧处理失败：{exc}；恢复数据：{self.folder}'))
            finally:self.dense_pending.task_done()

    def _finish_processing(self):
        if self.dense_thread is not None:
            self.dense_pending.join();self.dense_pending.put(None);self.dense_thread.join();self.dense_thread=None
        if self.dense_error is not None:raise RuntimeError(str(self.dense_error))
        self._write_video_metrics()

    def _capture(self):
        try:
            with self.camera_factory() as camera:
                preview_at=0
                while not self.stop.is_set():
                    try:(rgb,dep),calibration=camera.read(timeout=2 if self.ready else 20)
                    except TimeoutError:
                        if self.stop.is_set():break
                        raise
                    if self.stop.is_set():break
                    if not self.ready:self.ready=True;self.events.put(('capture_ready',None))
                    now=time.monotonic()
                    with self.pause_lock:
                        take=not self.paused.is_set() and self.cadence.take(now,(rgb[2],dep[2]),dep[3]/1000)
                        epoch=self.tracking_epoch
                    if now-preview_at<.1 and not take:continue
                    sdk=getattr(camera,'pair_cloud',None);depth_frame=getattr(camera,'depth_frame','ir')
                    points,frame=capture_geometry(dep[0],calibration,sdk,depth_frame=depth_frame,
                                                  points_frame=getattr(camera,'pointcloud_frame','rgb'))
                    if now-preview_at>=.1:
                        self.live_frame=points;self.live_rgb=rgb[0].copy()
                        self.live_intrinsics=intrinsics(calibration,frame);preview_at=now
                    if take and not self.stop.is_set():
                        packet=VideoFrame(rgb[0].copy(),dep[0].copy(),points,Calibration.from_buffer_copy(bytes(calibration)),dep[3]/1000,(rgb[2],dep[2]),sdk is not None,depth_frame,frame,epoch)
                        self.sampled+=1;self.submit((self.sampled,packet,None,None,packet.stamp))
        except Exception as exc:
            self.failed=True;self.stop.set();self.events.put(('error',f'视频采集失败：{exc}；恢复数据：{self.folder}'))
        finally:self.capture_done.set()

    def _process_frame(self,backend,count,packet,_depth,_k,stamp):
        # The mapper owns tracker state; UI/capture threads only tag packets.
        # Epochs survive latest-only dropped packets and queued pre-pause work.
        resumed=packet.tracking_epoch>self.processed_epoch
        if resumed:
            self.processed_epoch=packet.tracking_epoch
            if callable(getattr(backend,'resume_tracking',None)):backend.resume_tracking(stamp)
        cloud={f'points_{packet.geometry_frame}':packet.points*1000} if packet.sdk_xyz else {}
        color,depth,k=register_rgbd(packet.rgb,packet.depth,packet.calibration,depth_frame=packet.depth_frame,**cloud)
        name=f'{stamp:.6f}.png'
        if self.k is None:
            self.k,self.shape=k,depth.shape;write_configuration(self.folder,k,depth.shape,self.source_fps)
            (self.folder/'sdk_calibration.json').write_text(json.dumps({n:getattr(packet.calibration,n) for n,_ in Calibration._fields_}),encoding='utf-8')
        if getattr(backend,'tracking_first',False):
            self.skip_native_cloud_export=isinstance(backend,RtabmapBackend)
            # No local normal fitting or PNG/NPZ I/O for ordinary tracking frames.
            geometry=rgb_geometry(packet.points,packet.calibration,packet.geometry_frame)
            result=backend.process(color,depth,k,stamp,geometry=geometry)
            if backend.report.get('accepted'):
                if self.dense_thread is None:
                    self.dense_thread=threading.Thread(target=self._dense_worker,args=(backend,),daemon=True);self.dense_thread.start()
                try:self.dense_pending.put_nowait((count,packet,color,depth,geometry,backend.report.copy()))
                except queue.Full:
                    self.dense_dropped+=1;result=1
                    backend.report.update(accepted=False,skipped=True,reason='稠密处理繁忙，本节点仅用于原生定位图，未保留稠密观测')
        else:
            # Explicit injected legacy/test backend; production always takes tracking-first.
            self.local_geometry[count]=packet.points;self.geometry[count]=rgb_geometry(packet.points,packet.calibration,packet.geometry_frame)
            self.local_frames[count]=packet.geometry_frame
            result=super()._process_frame(backend,count,color,depth,k,stamp)
        accepted=bool(backend.report.get('accepted'))
        if accepted and not getattr(backend,'tracking_first',False):
            for directory,image in [('rgb',color),('depth',depth),('rgb_raw',packet.rgb),('depth_raw',packet.depth)]:
                if not cv2.imwrite(str(self.folder/directory/name),image):raise OSError('无法保存视频关键帧')
            self.names.append(name)
        row=dict(capture=count,stamp=stamp,rgb_id=packet.ids[0],depth_id=packet.ids[1],cloud_source='local_surface_v1',geometry_origin='sdk_xyz' if packet.sdk_xyz else 'raw16_calibrated',retained=accepted,
                 tracking_epoch=packet.tracking_epoch,resumed=resumed,
                 **coordinate_metadata(packet.depth_frame,packet.geometry_frame))
        with (self.folder/'frames.jsonl').open('a',encoding='utf-8') as manifest:manifest.write(json.dumps(row)+'\n')
        self.events.put(('video_stats',dict(sampled=self.sampled,processed=self.processed+1,dropped=self.dropped,retained=len(self.names))))
        return result

    def _after_process(self,backend,count,result):
        super()._after_process(backend,count,result)
        self._write_video_metrics()

    def _write_video_metrics(self):
        (self.folder/'video_metrics.json').write_text(json.dumps(dict(sampled=self.sampled,processed=self.processed,dropped_pending=self.dropped,retained=len(self.names),pending_capacity=1,dense_pending_capacity=1,dense_keyframes_skipped=self.dense_dropped)),encoding='utf-8')

    def _finalize_saved(self,folder):
        if not getattr(self,'skip_native_cloud_export',False):return super()._finalize_saved(folder)
        from reconstruct import save_cloud
        from scipy.spatial.transform import Rotation
        save_cloud(folder/'map_cloud.ply',self.final_world_cloud,'RTAB-Map world XYZ in metres, X forward Y left Z up')
        graph=json.loads((folder/'geometry_poses.json').read_text(encoding='utf-8'))
        frames={r['capture']:r['stamp'] for r in (json.loads(line) for line in (folder/'frames.jsonl').read_text(encoding='utf-8').splitlines())}
        captures={r['node']:r['capture'] for r in (json.loads(line) for line in (folder/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted')}
        optical_rotation=np.array([[0,0,1.],[-1.,0,0],[0,-1.,0]])
        with (folder/'map_poses.txt').open('w',encoding='utf-8') as stream:
            stream.write('#timestamp x y z qx qy qz qw id\n# Retained dense nodes only; robot world in metres. Full native graph: map.db\n')
            for node,pose in zip(graph['node_ids'],graph['poses']):
                pose=np.asarray(pose);q=Rotation.from_matrix(pose[:3,:3]@optical_rotation.T).as_quat()
                values=[frames[captures[node]],*pose[:3,3],*q]
                stream.write(' '.join(f'{x:.9f}' for x in values)+f' {node}\n')
