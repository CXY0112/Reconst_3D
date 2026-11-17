"""RGBD mode: SLAM acquisition, colored consolidation and instance segmentation."""
from pathlib import Path
import queue,threading,time
import tkinter as tk
from tkinter import ttk
from ui_dialogs import messagebox,filedialog
import cv2
import numpy as np
from snapshot_window import VideoWindow
from preprocess_window import PreprocessWindow
from map_cloud_view import MapCloudPanel
from preprocess import Parameters,ProcessingCancelled,load_input
from scene_cloud_processing import session_for,colorize,process_scene,save_scene,save_scene_skip


class ScenePreprocessWindow(PreprocessWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.skip_button.configure(text='跳过预处理 → 几何重建')
        self.reconstruct_button.configure(text='进入第三阶段 · 几何重建')
        self.view.configure(values=('处理结果',));self.view.pack_forget()
        self.process_button.configure(text='场景预处理：融合 / 均匀化 / 保特征精简')
        self._colored_loaded=None;self._summarized=None;self._skip_summarized=None
        # Replace only the experiment tab's labels; retain the established full-size pair.
        self.before.label.master.configure(text='未处理 · 完整场景 RGB 点云')
        self.after.label.master.configure(text='预处理后 · 完整场景 RGB 点云')
        self.summary.set('完整场景：去噪＋跨帧表面融合＋去重＋均匀重采样＋保特征精简；颜色保留可见RGB观测。')

    def load(self,source):
        from dataset_store import route_dataset
        if route_dataset(self,source) or self.working:return
        self._begin('load','正在读取完整场景，并关联原始RGB颜色…')
        unit={'自动':'auto','米':'m','毫米':'mm'}[self.unit.get()]
        axes={'自动':'auto','世界':'world','相机':'camera'}[self.axes.get()]
        def worker():
            try:
                session=session_for(source);path,loaded=load_input(session,unit,axes)
                colors,_,report=colorize(loaded.points,session,lambda f,t:setattr(self,'latest_progress',(f,t)),self.cancel)
                loaded.scene_colors=colors;loaded.color_report=report
                self.events.put(('loaded',(path,loaded)))
            except ProcessingCancelled:self.events.put(('cancelled',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def process(self):
        if self.working or self.loaded is None:return
        if not getattr(self,'can_write',lambda:True)():return
        try:
            values={key:float(field.get()) for key,field in self.fields.items()}
            values['kalman_sampling']='surface_uniform'
            if self.local_profile:values['neighbors']=self.local_profile['neighbors']
            params=Parameters(**values);params.validate()
        except ValueError as exc:messagebox.showerror('参数',str(exc),parent=self.root);return
        source,points=self.source,self.loaded.points
        self._begin('process','正在处理完整场景：consolidation → simplification…')
        def worker():
            try:
                result=process_scene(source,points,params,lambda f,t:setattr(self,'latest_progress',(f,t)),self.cancel)
                self.events.put(('processed',result))
            except ProcessingCancelled:self.events.put(('cancelled',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _show_result(self):
        if self.result is not None:
            colors=self.result.diagnostics.get('rgb') if self.result.diagnostics else None
            self.after.set_cloud(self.result.points,colors=colors)
            self._sync_from_left()

    def skip(self,destination=None):
        if self.working or self.loaded is None:return
        if not getattr(self,'can_write',lambda:True)():return
        if destination is None:
            if getattr(self,'dataset',None):destination=Path(self.dataset)/'rgbd_scene/setp2'
            else:
                parent=filedialog.askdirectory(parent=self.root,title='选择跳过记录的保存父目录')
                if not parent:return
                destination=Path(parent)/'rgbd_scene_setp2'
        source,count=self.source,len(self.loaded.points)
        self._skip_summarized=None
        self._begin('save','正在记录跳过预处理，三阶段将直接使用一阶段融合点云…')
        def worker():
            try:self.events.put(('skipped',(Path(destination),save_scene_skip(source,destination,count))))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _poll(self):
        super()._poll()
        if self.closed:return
        if self.loaded is not None and self.loaded is not self._colored_loaded and hasattr(self.loaded,'scene_colors'):
            self._colored_loaded=self.loaded;self.before.set_cloud(self.loaded.points,colors=self.loaded.scene_colors)
        if self.result is None and self.saved_path is not None and self.loaded is not None:
            key=(self.saved_path,id(self.loaded))
            if key!=self._skip_summarized:
                self._skip_summarized=key
                self.after.label.master.configure(text='已跳过 · 一阶段完整 RGB 点云')
                self.after.set_cloud(self.loaded.points,colors=getattr(self.loaded,'scene_colors',None));self._sync_from_left()
                self.status.set('二阶段已跳过；三阶段直接使用一阶段点云进行几何重建')
                self.summary.set(f'直接使用 {len(self.loaded.points):,} 点；不执行二阶段去噪、卡尔曼融合、均匀化或精简；仅保存一阶段引用。')
        if self.result is not None and self.result is not self._summarized:
            self._summarized=self.result;r=self.result.report
            self.after.label.master.configure(text='预处理后 · 完整场景 RGB 点云')
            if 'simplification' in r:
                self.summary.set(f'完整场景 {r["input_points"]:,} → 表面融合/均匀化 {r["consolidation_points"]:,} → '
                    f'保特征精简 {r["output_points"]:,} 点（总减少 {100*(1-r["output_points"]/max(r["input_points"],1)):.1f}%） · '
                    f'保留特征点 {r["simplification"]["feature_points"]:,} · RGB实测着色 {r["color"]["coverage"]:.1%}')

    def save(self,destination=None):
        if self.working or self.result is None:return
        if not getattr(self,'can_write',lambda:True)():return
        if destination is None and getattr(self,'dataset',None):destination=Path(self.dataset)/'rgbd_scene/setp2'
        if destination is None:
            destination=filedialog.askdirectory(parent=self.root,title='选择保存父目录')
            if not destination:return
            destination=Path(destination)/'rgbd_scene_setp2'
        self._begin('save','正在保存完整彩色点云、表面属性与RGB来源…')
        result,source=self.result,self.source
        def worker():
            try:self.events.put(('saved',save_scene(result,source,destination)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def load_saved(self,folder,scan=None):
        if self.working:return
        self._begin('load','正在恢复完整场景彩色点云对照…')
        def worker():
            try:
                import json
                from preprocess import ProcessingResult
                folder_path=Path(folder);report=json.loads((folder_path/'report.json').read_text(encoding='utf-8'))
                source=session_for(scan or ((folder_path/report['source_session_relative']) if report.get('skipped') else report['source']))/'map_cloud.ply'
                _,loaded=load_input(source)
                loaded.scene_colors,_,loaded.color_report=colorize(loaded.points,source.parent,
                    lambda f,t:setattr(self,'latest_progress',(f,t)),self.cancel)
                if report.get('skipped'):
                    self.events.put(('skip_restored',(source,loaded,folder_path,report)));return
                display=lambda points:np.column_stack((-points[:,1],-points[:,2],points[:,0])).astype(np.float32)
                with np.load(folder_path/'surfel_attributes.npz',allow_pickle=False) as data:
                    attributes={key:data[key].copy() for key in data if key!='normals_world'}
                    if 'normals_world' in data:attributes['normals']=display(data['normals_world'])
                result=ProcessingResult(display(np.load(folder_path/'processed_world.npy')),np.empty((0,3)),np.empty(0,int),np.empty(0),report,attributes)
                self.events.put(('restored',(source,loaded,result,folder_path)))
            except ProcessingCancelled:self.events.put(('cancelled',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()


class SceneLivePanel(MapCloudPanel):
    def __init__(self,parent,provider):
        super().__init__(parent);self.provider=provider;self.packet=None

    def update(self,points):
        packet=self.provider()
        if packet is None or packet is self.packet:return
        self.packet=packet;xyz,rgb=packet;self.set_cloud(xyz,colors=rgb)


class SceneVideoWindow(VideoWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        parent=self.live_panel.label.master
        for child in parent.winfo_children():child.destroy()
        self.live_panel=SceneLivePanel(parent,lambda:getattr(self.session,'scene_live',None))
        self.color_jobs=queue.Queue(maxsize=1);self.color_results=queue.Queue(maxsize=1)
        self.color_cancel=threading.Event();self.color_thread=None;self.scene_id=0
        self.root.bind('<Destroy>',self._scene_destroy,add='+')
        self.color_timer=self.root.after(60,self._poll_colors)
        self.preprocess_button.configure(text='完整场景预处理')
        self.graph_optimizer=tk.StringVar(value='iSAM2（增量优化）')
        rate_hint=self.rate_row.winfo_children()[-1]
        ttk.Label(self.rate_row,text='轨迹优化').pack(side='left',padx=(12,4),before=rate_hint)
        ttk.Label(self.rate_row,textvariable=self.graph_optimizer).pack(side='left',before=rate_hint)
        self.registration_frontend=tk.StringVar(value='RGB-D（现有方案）')
        from tracking_reference_panel import TrackingReferencePanel
        self.rgb_panel.frame.pack_configure(side='left')
        self.reference_panel=TrackingReferencePanel(self.rgb_panel.frame.master)
        self.frontend_row=ttk.Frame(self.root,padding=(15,0,15,3))
        self.frontend_row.pack(fill='x',after=self.rate_row)
        ttk.Label(self.frontend_row,text='配准前端').pack(side='left',padx=(0,6))
        self.frontend_box=ttk.Combobox(self.frontend_row,textvariable=self.registration_frontend,state='readonly',width=23,
            values=('RGB-D（现有方案）','BUFFER-X（实验／CPU）','G-CVO（CPU）','G-CVO（CUDA）'))
        self.frontend_box.pack(side='left')
        self.frontend_hint=tk.StringVar(value='现有视觉配准；切换方案后开始新的扫描')
        ttk.Label(self.frontend_row,textvariable=self.frontend_hint).pack(side='left',padx=8)
        def frontend_hint(*_):
            choice=self.registration_frontend.get()
            self.reference_panel.show(choice.startswith('G-CVO'))
            text='现有视觉配准；切换方案后开始新的扫描'
            if choice.startswith('BUFFER-X'):text='CPU 单对约 1–2 分钟；建议先用录制回放对比，扫描时保持视野重叠'
            if choice=='G-CVO（CPU）':text='CPU 原生二阶配准；请保持连续重叠，实际速度取决于点数和迭代次数'
            if choice=='G-CVO（CUDA）':text='需要构建 CUDA 库及可用 NVIDIA GPU；本机暂不可用'
            self.frontend_hint.set(text)
        self.registration_frontend.trace_add('write',frontend_hint)

    def create_session(self):
        from video_capture import VideoSession
        from rtabmap_scan import RGBDCamera
        from pointmap_processing import audit_sdk_grid,xyz_from_depth
        optimizer='isam2'
        choice=self.registration_frontend.get()
        frontend={'G-CVO（CPU）':'gcvo_cpu','G-CVO（CUDA）':'gcvo_cuda'}.get(choice,'bufferx' if choice.startswith('BUFFER-X') else 'rgbd')
        if frontend=='bufferx':
            from bufferx_client import runtime_settings
            try:runtime_settings()
            except (RuntimeError,OSError,ValueError) as exc:raise ValueError(str(exc)) from exc
        if frontend.startswith('gcvo_'):
            from gcvo_matcher import settings_for
            try:settings_for(frontend.split('_')[1])
            except (RuntimeError,OSError,ValueError) as exc:raise ValueError(str(exc)) from exc
        session=VideoSession(self.events,hz=float(self.rate.get()),optimizer=optimizer,dense_mode='submaps',frontend=frontend)
        class TappedCamera:
            def __enter__(tap):tap.camera=RGBDCamera();tap.camera.__enter__();tap.checked=False;tap.last=0.;return tap
            def __exit__(tap,*args):return tap.camera.__exit__(*args)
            def __getattr__(tap,key):return getattr(tap.camera,key)
            def read(tap,timeout=20):
                deadline=time.monotonic()+timeout
                while True:
                    pair,calibration=tap.camera.read(max(.01,deadline-time.monotonic()))
                    if tap.checked or tap.camera.pair_cloud is not None:break
                    if time.monotonic()>=deadline:raise TimeoutError('等待可核验的SDK RGB-D/XYZ同步帧超时')
                if time.monotonic()-tap.last>=.1:
                    (rgb,depth)=pair;d=depth[0].astype(np.float32)/1000
                    k=np.array([[calibration.fxrgb,0,calibration.cxrgb],[0,calibration.fyrgb,calibration.cyrgb],[0,0,1.]])
                    native=tap.camera.pair_cloud
                    if not tap.checked:
                        audit_sdk_grid(d,k,native/1000);tap.checked=True
                    xyz=xyz_from_depth(d,k);valid=np.isfinite(xyz).all(2)
                    session.scene_live=(xyz[valid],np.ascontiguousarray(cv2.cvtColor(rgb[0],cv2.COLOR_BGR2RGB)[valid]))
                    tap.last=time.monotonic()
                return pair,calibration
        session.camera_factory=TappedCamera
        return session

    def draw(self,points,count,nodes=None):
        self.last_scene_snapshot=(points,count,nodes)
        if self.color_thread is None:
            self.color_thread=threading.Thread(target=self._color_worker,daemon=True);self.color_thread.start()
        source=self.session.folder if self.session else getattr(self,'imported_path',None)
        if source is None:return
        source=Path(source);source=source if source.is_dir() else source.parent
        # Live and imported snapshots already use optical display-world axes.
        MapCloudPanel._latest(self.color_jobs,(points,nodes,source,self.scene_id))
        self.draw_points=len(points)

    def _color_worker(self):
        while not self.color_cancel.is_set():
            try:points,nodes,source,revision=self.color_jobs.get(timeout=.2)
            except queue.Empty:continue
            try:
                colors,_,report=colorize(points,source,cancel=self.color_cancel)
                MapCloudPanel._latest(self.color_results,('colored',(points,nodes,colors,report,revision)))
            except ProcessingCancelled:return
            except Exception as exc:
                MapCloudPanel._latest(self.color_results,('color_error',(str(exc),revision)))

    def _poll_colors(self):
        if self.closed:return
        folder=getattr(self.session,'folder',None)
        if folder is not None and folder!=getattr(self,'last_color_folder',None):
            self.last_color_folder=folder
            if getattr(self,'last_scene_snapshot',None):self.draw(*self.last_scene_snapshot)
        try:
            kind,payload=self.color_results.get_nowait()
            if kind=='colored':
                points,nodes,colors,report,revision=payload
                if revision==self.scene_id:self.map_panel.set_cloud(points,nodes,colors=colors)
            else:
                error,revision=payload
                if revision==self.scene_id:self.status.set('完整场景着色等待原始观测就绪：'+error)
        except queue.Empty:pass
        self.color_timer=self.root.after(60,self._poll_colors)

    def start(self):
        self.scene_id+=1
        previous=self.session
        super().start()
        if self.session is not previous:self.reference_panel.clear()
        self.frontend_box.configure(state='disabled' if self.busy else 'readonly')

    def poll(self):
        super().poll()
        if hasattr(self,'frontend_box'):self.frontend_box.configure(state='disabled' if self.busy or self.importing else 'readonly')

    def import_cloud(self,path=None):
        self.scene_id+=1
        super().import_cloud(path)

    def _scene_destroy(self,event):
        if event.widget is self.root:
            self.color_cancel.set()
            if getattr(self,'color_timer',None):self.root.after_cancel(self.color_timer)
            if self.color_thread and self.color_thread is not threading.current_thread():self.color_thread.join(timeout=.5)


# Compatibility imports for callers of the former explicit rgbd entry point.
from scene_workbench import SceneWorkbench,launch
