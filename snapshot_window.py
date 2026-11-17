"""Manual independent RGB-D captures, sharing the existing viewer/save lifecycle."""
import json
import threading
import time
import cv2
import numpy as np
from reconstruct import Calibration, depth_points, save_cloud
from rtabmap_scan import RGBDCamera, register_rgbd, write_configuration
from scan_window import ScanSession, ScanWindow
from geometry_snapshot_backend import SnapshotBackend
from rgbd_geometry import capture_geometry,intrinsics,rgb_geometry,ir_to_rgb,coordinate_metadata


class SnapshotSession(ScanSession):
    def __init__(self,events,camera_factory=RGBDCamera):
        super().__init__(events,camera_factory=camera_factory,backend_factory=SnapshotBackend)
        self.request = threading.Event()
        self.ready = False
        self.capture_busy = False
        self.minimum_frames = 1
        self.export_voxel_size = .003
        self.export_decimation = 1
        self.geometry = {}
        self.local_geometry = {}
        self.local_frames = {}
        from preprocess import Parameters
        from local_preprocessing import profile
        # Capture-side cleaning keeps its established cost; stage-two output defaults are independent.
        self.local_params=Parameters(output_voxel_mm=5.)
        (self.folder/'local_preprocessing.json').write_text(json.dumps(dict(schema='local_surface_v1',parameters=profile(self.local_params)),indent=2),encoding='utf-8')
        self.final_world_cloud = None
        self.live_frame = None  # Latest-only mailbox; live frames never queue behind drawing.
        self.live_intrinsics = None
        for directory in ('depth_raw','rgb_raw','pointcloud_local'):
            (self.folder/directory).mkdir()

    def request_capture(self):
        if self.ready and not self.capture_busy and not self.stop.is_set():
            self.capture_busy = True
            self.request.set()
            return True
        return False

    def _capture(self):
        try:
            with (self.folder/'frames.jsonl').open('w',encoding='utf-8') as manifest:
                with self.camera_factory() as camera:
                    preview_at = 0
                    while not self.stop.is_set():
                        try:
                            (rgb,dep),calibration = camera.read(timeout=2 if self.ready else 20)
                        except TimeoutError:
                            if self.stop.is_set():
                                break
                            raise
                        if not self.ready:
                            self.ready = True
                            self.events.put(('capture_ready',None))
                        now = time.monotonic()
                        if now-preview_at < .1 and not self.request.is_set():
                            continue
                        sdk_cloud = getattr(camera,'pair_cloud',None)
                        depth_frame = getattr(camera,'depth_frame','ir')
                        points,frame = capture_geometry(dep[0],calibration,sdk_cloud,depth_frame=depth_frame,
                                                        points_frame=getattr(camera,'pointcloud_frame','rgb'))
                        self.live_intrinsics = intrinsics(calibration,frame)
                        if now-preview_at >= .1:
                            self.live_frame = points
                            self.live_rgb = rgb[0]
                            preview_at = now
                        if not self.request.is_set():
                            continue
                        self.request.clear()
                        if self.stop.is_set():
                            break
                        cloud = {f'points_{frame}':points*1000} if sdk_cloud is not None else {}
                        color,depth,k = register_rgbd(rgb[0],dep[0],calibration,depth_frame=depth_frame,**cloud)
                        name = f'{dep[3]/1000:.6f}.png'
                        for directory,image in [('rgb',color),('depth',depth),('depth_raw',dep[0]),('rgb_raw',rgb[0])]:
                            if not cv2.imwrite(str(self.folder/directory/name),image):
                                raise OSError('无法保存拍摄数据')
                        if self.k is None:
                            self.k,self.shape = k,depth.shape
                            write_configuration(self.folder,k,depth.shape,10)
                            (self.folder/'sdk_calibration.json').write_text(json.dumps({n:getattr(calibration,n) for n,_ in Calibration._fields_},indent=2))
                        self.names.append(name)
                        count = len(self.names)
                        self.geometry[count] = rgb_geometry(points,calibration,frame)
                        self.local_geometry[count] = points
                        self.local_frames[count] = frame
                        manifest.write(json.dumps({'stamp':dep[3]/1000,'rgb_id':rgb[2],'depth_id':dep[2],'capture':count,
                                                   'cloud_source':'local_surface_v1','geometry_origin':'sdk_xyz' if sdk_cloud is not None else 'raw16_calibrated',
                                                   **coordinate_metadata(depth_frame,frame)})+'\n')
                        manifest.flush()
                        self.submit((count,color,depth,k,dep[3]/1000))
                        self.events.put(('capture_matching',count))
        except Exception as exc:
            self.failed = True
            self.stop.set()
            self.events.put(('error',f'拍摄失败：{exc}\n恢复数据：{self.folder}'))
        finally:
            self.capture_done.set()

    def _publish(self,backend,count,final=False):
        super()._publish(backend,count,final)

    def _publish_every_capture(self):
        return True

    def _after_process(self,backend,count,result):
        self.capture_busy = False
        with (self.folder/'registrations.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'capture':count,**backend.report},ensure_ascii=False)+'\n')
        preview=getattr(backend,'reference_preview',None)
        if preview is not None and preview['revision']!=getattr(self,'_reference_revision',None):
            self._reference_revision=preview['revision']
            self.events.put(('tracking_reference',preview))
        self.events.put(('capture_result',(count,backend.report.copy())))

    def _process_frame(self,backend,count,color,depth,k,stamp):
        from local_preprocessing import prepare_local,save_local
        # Runs in the mapping thread, never in the live camera/preview loop.
        self.events.put(('local_preparing',count))
        raw=self.local_geometry.pop(count)
        prepared=prepare_local(raw,self.local_params,coordinate_frame=self.local_frames.pop(count,'ir'))
        path=self.folder/'pointcloud_local'/f'{stamp:.6f}.npz'
        save_local(path,prepared)
        self.events.put(('capture_matching',count))
        result=backend.process(color,depth,k,stamp,geometry=self.geometry.pop(count))
        if not backend.report.get('accepted'):
            path.unlink(missing_ok=True)
        else:
            backend.report['local_preprocessing']=prepared['metadata']
            backend.report['local_packet_bytes']=path.stat().st_size
        return result

    def _before_close(self,backend,points,nodes):
        # Preview optical-style world -> RTAB-Map robot world, in metres.
        self.final_world_cloud = np.column_stack((points[:,2],-points[:,0],-points[:,1]))
        np.save(self.folder/'full_geometry_world.npy',self.final_world_cloud)

    def _finalize_saved(self, folder):
        (folder/'map_cloud.ply').rename(folder/'map_cloud_rgb_registered.ply')
        save_cloud(folder/'map_cloud.ply',self.final_world_cloud,'RTAB-Map world XYZ in metres, X forward Y left Z up')



def native_geometry(raw_depth,calibration):
    points = depth_points(raw_depth,calibration,stride=1)
    return ir_to_rgb(points,calibration)


def ir_geometry(raw_depth,calibration,sdk_cloud=None):
    """Legacy explicit IR helper. New capture code uses capture_geometry with frame labels."""
    if sdk_cloud is None:
        return depth_points(raw_depth,calibration,stride=1)
    points = np.asarray(sdk_cloud,dtype=float).reshape(-1,3)/1000
    return points[np.isfinite(points).all(axis=1)&(points[:,2]>=.15)&(points[:,2]<=4)]


class SnapshotWindow(ScanWindow):
    def __init__(self,root, *, embedded=False, on_preprocess=None):
        from tkinter import ttk
        super().__init__(root, embedded=embedded, on_preprocess=on_preprocess)
        if not embedded:
            root.title('HP60C · 不同位置拍摄与拼接')
        self.start_button.configure(text='开始拍摄会话')
        self.finish_button.configure(text='完成')
        self.capture_button = ttk.Button(self.start_button.master,text='拍摄当前位置',command=self.capture,state='disabled')
        self.capture_button.pack(side='left',padx=5,before=self.finish_button)
        self._space_down=False
        tag=f'CaptureSpace_{id(self)}'
        self.root.bind_class(tag,'<KeyPress-space>',self._space_press)
        self.root.bind_class(tag,'<KeyRelease-space>',self._space_release)
        def attach(widget):
            widget.bindtags((tag,*widget.bindtags()))
            for child in widget.winfo_children():attach(child)
        attach(root)
        root.bind('<Unmap>',lambda event:self._space_release() if event.widget is root else None,add='+')
        self.status.set('就绪 · 开始会话后移动到合适位置，点击拍摄')

    def _space_press(self,event):
        if event.widget.winfo_class() in ('Entry','TEntry','Text','TCombobox','Spinbox','TSpinbox'):return
        if not self._space_down:
            self._space_down=True
            if self.session and self.busy and str(self.capture_button['state'])!='disabled':self.capture()
        return 'break'

    def _space_release(self,event=None):
        self._space_down=False
        return 'break'

    def start(self):
        if self.busy or self.importing:
            return
        if not getattr(self,'can_write',lambda:True)():return
        if self.session is not None and self.has_map and self.session.folder.exists() and self.session.folder.name.startswith('.pending_scan_'):
            self.status.set('请点击“完成”保存当前扫描，再开始新的拍摄会话');return
        self.session = self.create_session()
        self.busy,self.has_map = True,False
        self.tracking_lost = False
        self.map_panel.clear()
        self.live_panel.reset()
        self.rgb_panel.clear()
        self.start_button.configure(state='disabled')
        self.finish_button.configure(state='normal')
        self.capture_button.configure(state='disabled')
        self.status.set('正在连接相机，首批画面到达后预热 1 秒再开始采集…')
        self.session.start()

    def create_session(self):return SnapshotSession(self.events)

    def capture(self):
        if self.session.request_capture():
            self.capture_button.configure(state='disabled')
            self.status.set('正在拍摄与匹配，请保持相机稳定…')

    def finish(self):
        self.capture_button.configure(state='disabled')
        super().finish()


class VideoWindow(SnapshotWindow):
    def __init__(self,root,**kwargs):
        from tkinter import ttk
        import tkinter as tk
        super().__init__(root,**kwargs)
        if not kwargs.get('embedded',False):root.title('HP60C · 连续视频扫描')
        self.start_button.configure(text='开始视频扫描')
        self.capture_button.configure(text='暂停采集')
        self.map_frame.configure(text='连续融合地图')
        self.rate=tk.StringVar(value='10')
        self.rate_row=ttk.Frame(root,padding=(15,0,15,3))
        self.rate_row.pack(fill='x',after=self.start_button.master)
        ttk.Label(self.rate_row,text='目标采样频率（Hz）').pack(side='left')
        self.rate_box=ttk.Spinbox(self.rate_row,from_=.1,to=10,increment=.1,textvariable=self.rate,width=6)
        self.rate_box.pack(side='left')
        ttk.Label(self.rate_row,text='默认10 Hz视觉跟踪；关键帧建图最高2 Hz；繁忙时保留最新帧，请缓慢移动').pack(side='left',padx=8)
        self.status.set('就绪 · 开始后自动采样建图；空格暂停/继续采集')

    def create_session(self):
        from video_capture import VideoSession
        try:hz=10. if getattr(self,'beta',False) else float(self.rate.get())
        except ValueError:raise ValueError('采样频率必须为数字')
        return VideoSession(self.events,hz=hz)

    def start(self):
        from ui_dialogs import messagebox
        try:super().start()
        except ValueError as exc:messagebox.showerror('采样频率',str(exc),parent=self.root)
        if self.busy:
            self.rate_box.configure(state='disabled');self.capture_button.configure(text='暂停采集')

    def poll(self):
        super().poll()
        if hasattr(self,'rate_box'):self.rate_box.configure(state='disabled' if self.busy else 'normal')
        if self.busy and getattr(self.session,'video_mode',False) is True and self.session.paused.is_set():
            self.status.set('采集已暂停，已排队帧继续处理；空格或按钮继续')

    def capture(self):
        if self.session and self.busy:
            paused=self.session.toggle_pause()
            self.capture_button.configure(text='继续采集' if paused else '暂停采集')
            self.status.set('采集已暂停，已排队帧继续处理' if paused else
                '正在续接，请对照最后可信参考视角并短暂停稳' if getattr(self.session,'frontend','').startswith('gcvo_') else '自动采样中，请缓慢移动镜头')


def launch():
    import tkinter as tk
    root = tk.Tk()
    VideoWindow(root)
    root.mainloop()


if __name__ == '__main__':
    launch()
