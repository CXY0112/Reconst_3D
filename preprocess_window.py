from dataset_store import route_dataset
"""Offline before/after viewer. Geometry work and exports never run on the Tk thread."""
from pathlib import Path
from dataset_store import replace_stage, stage_folder
import queue
import threading
import tkinter as tk
from tkinter import ttk
from ui_dialogs import filedialog,messagebox
import numpy as np
from cloud_import import ROOT
from live_cloud_view import ViewCamera
from map_cloud_view import MapCloudPanel
from preprocess import Parameters, process_input, load_input, save_result, ProcessingCancelled


class PreprocessWindow:
    def __init__(self, root, source=None, *, embedded=False, on_reconstruct=None):
        self.root = root
        self.on_reconstruct = on_reconstruct; self.saved_path = None
        if not embedded:
            root.title('点云预处理 · 卡尔曼表面融合'); root.geometry('1450x900')
        self.events = queue.Queue(); self.cancel = threading.Event()
        self.working = None; self.loaded = None; self.source = None; self.result = None
        self.closed = False; self.close_pending = False; self.latest_progress = None
        self.last_views = None
        self.local_profile=None
        controls = ttk.Frame(root, padding=8); controls.pack(fill='x')
        self.open_button = ttk.Button(controls, text='导入点云', command=self.open_file); self.open_button.pack(side='left', padx=3)
        self.process_button = ttk.Button(controls, text='开始预处理', command=self.process); self.process_button.pack(side='left', padx=3)
        self.skip_button = ttk.Button(controls, text='跳过预处理', command=self.skip); self.skip_button.pack(side='left', padx=3)
        self.cancel_button = ttk.Button(controls, text='取消', command=self.cancel.set); self.cancel_button.pack(side='left', padx=3)
        self.save_button = ttk.Button(controls, text='保存 / 覆盖本阶段', command=self.save); self.save_button.pack(side='left', padx=3)
        self.reconstruct_button = ttk.Button(controls, text='进入几何重建', command=self.open_reconstruction)
        if on_reconstruct is not None: self.reconstruct_button.pack(side='left', padx=3)
        self.sync = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='同步视角', variable=self.sync, command=self._sync_from_left).pack(side='left', padx=10)
        self.view = ttk.Combobox(controls, values=('处理结果', '被剔除的孤立点', '移动超过3mm的原位置', '多次观测确认的表面'), state='readonly', width=24)
        self.view.set('处理结果'); self.view.pack(side='right'); self.view.bind('<<ComboboxSelected>>', lambda event:self._show_result())
        options = ttk.Frame(root, padding=(10, 0, 10, 5)); options.pack(fill='x')
        self.fields = {}
        for key, text, value in (('noise_mm','容忍尺度(mm)',10), ('radius_mm','拟合邻域(mm)',40),
                                 ('output_voxel_mm','输出体素(mm)',2)):
            ttk.Label(options, text=text).pack(side='left')
            entry = ttk.Entry(options, width=6); entry.insert(0, str(value)); entry.pack(side='left', padx=(3, 12)); self.fields[key] = entry
        self.unit = ttk.Combobox(options, values=('自动','米','毫米'), state='readonly', width=6); self.unit.set('自动')
        ttk.Label(options, text='源单位').pack(side='left'); self.unit.pack(side='left', padx=4)
        self.axes = ttk.Combobox(options, values=('自动','世界','相机'), state='readonly', width=6); self.axes.set('自动')
        ttk.Label(options, text='源坐标').pack(side='left'); self.axes.pack(side='left', padx=4)
        models = ttk.Frame(root,padding=(10,0,10,5)); models.pack(fill='x')
        self.backend = tk.StringVar(value='卡尔曼表面融合')
        for key,text,value in (('kalman_floor_mm','卡尔曼不确定度下限(mm)',2),('surface_spacing_mm','卡尔曼均匀间距(mm)',2)):
            ttk.Label(models,text=text).pack(side='left'); entry=ttk.Entry(models,width=5)
            entry.insert(0,str(value));entry.pack(side='left',padx=(3,10));self.fields[key]=entry
        self.sampling = tk.StringVar(value='表面均匀采样')
        ttk.Label(models,text='新扫描复用局部表面；旧扫描兼容原始点云；位姿固定').pack(side='left',padx=5)
        ttk.Label(root, text='固定拍摄位姿 · 不补洞 · 无标注文件默认米/世界坐标 · 修改单位或坐标后需重新导入', padding=(10, 0)).pack(fill='x')
        self.status = tk.StringVar(value='导入文件或扫描目录后开始；无需连接相机')
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill='x')
        self.bar = ttk.Progressbar(root, maximum=100); self.bar.pack(fill='x', padx=10)
        panels = ttk.Panedwindow(root, orient='horizontal'); panels.pack(fill='both', expand=True)
        left = ttk.Labelframe(panels, text='原始融合点云'); right = ttk.Labelframe(panels, text='处理后 / 检查视图')
        panels.add(left, weight=1); panels.add(right, weight=1)
        self.before = MapCloudPanel(left); self.after = MapCloudPanel(right)
        self.summary = tk.StringVar(value='局部平面起伏是形状指标，不代表硬件绝对精度。按住左键转向，WASD移动，QE升降，R复位。')
        ttk.Label(root, textvariable=self.summary, padding=8, wraplength=1400).pack(fill='x')
        if not embedded:
            root.protocol('WM_DELETE_WINDOW', self.close)
        self._buttons(); self.timer = root.after(40, self._poll)
        if source is not None:
            root.after(0, lambda:self.load(source))

    def _buttons(self):
        idle = self.working is None
        self.open_button.configure(state='normal' if idle else 'disabled')
        self.process_button.configure(state='normal' if idle and self.loaded is not None else 'disabled')
        self.skip_button.configure(state='normal' if idle and self.loaded is not None else 'disabled')
        self.save_button.configure(state='normal' if idle and self.result is not None else 'disabled')
        self.cancel_button.configure(state='normal' if self.working in ('load','process') else 'disabled')
        self.reconstruct_button.configure(state='normal' if idle and self.saved_path is not None else 'disabled')
        for key,field in self.fields.items():
            field.configure(state=('readonly' if key=='radius_mm' and self.local_profile else 'normal') if idle else 'disabled')
        for field in (self.unit, self.axes):
            field.configure(state='readonly' if idle else 'disabled')

    def open_file(self):
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(ROOT/'results'),
               filetypes=[('点云','*.ply *.pcd *.npy *.npz *.xyz *.txt *.csv'),('所有文件','*.*')])
        if path: self.load(path)

    def open_folder(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=str(ROOT/'results'), title='选择含 map_cloud.ply 的扫描目录')
        if path: self.load(path)

    def load(self, source):
        if route_dataset(self,source):return
        if self.working: return
        unit = {'自动':'auto','米':'m','毫米':'mm'}[self.unit.get()]
        axes = {'自动':'auto','世界':'world','相机':'camera'}[self.axes.get()]
        self._begin('load', '正在后台读取点云…')
        def worker():
            try:
                result = load_input(source, unit, axes)
                if self.cancel.is_set(): raise ProcessingCancelled()
                self.events.put(('loaded', result))
            except ProcessingCancelled: self.events.put(('cancelled', None))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def load_saved(self, folder, scan=None):
        if self.working:return
        from preprocess import ProcessingResult
        from cloud_import import load_cloud
        import json
        folder=Path(folder)
        self._begin('load','正在恢复已保存的预处理结果…')
        def worker():
            try:
                report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
                if report.get('skipped'):
                    from geometry_inputs import source_info
                    source=source_info(folder)['session']/'map_cloud.ply'
                    loaded=load_cloud(source)
                    self.events.put(('skip_restored',(source,loaded,folder,report)));return
                source=Path(scan)/'map_cloud.ply' if scan else folder/('processed.ply' if (folder/'processed.ply').is_file() else 'processed_world.npy')
                loaded=load_cloud(source)
                display=lambda points:np.column_stack((-points[:,1],-points[:,2],points[:,0])).astype(np.float32)
                def array(name, fallback):
                    return np.load(folder/name,allow_pickle=False) if (folder/name).is_file() else fallback
                with np.load(folder/'surfel_attributes.npz') as data:
                    attributes={key:data[key].copy() for key in data if key!='normals_world'}
                    if 'normals_world' in data:attributes['normals']=display(data['normals_world'])
                result=ProcessingResult(display(np.load(folder/'processed_world.npy')),
                    display(array('removed_world.npy',np.empty((0,3)))),
                    array('retained_loaded_indices.npy',np.empty(0,dtype=int)),
                    array('movement_mm.npy',np.empty(0)),report,attributes)
                self.events.put(('restored',(source,loaded,result,folder)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _begin(self, kind, text):
        self.working = kind; self.cancel.clear(); self.bar['value'] = 0; self.latest_progress = None
        self.status.set(text); self._buttons()

    def _capture_profile(self):
        import json
        self.local_profile=None
        folder=Path(self.source) if self.source else None
        if folder is None:return
        folder=folder if folder.is_dir() else folder.parent
        path=folder/'local_preprocessing.json'
        if path.is_file() and any((folder/'pointcloud_local').glob('*.npz')):
            data=json.loads(path.read_text(encoding='utf-8'))
            self.local_profile=data['parameters']
            field=self.fields['radius_mm'];field.configure(state='normal');field.delete(0,'end');field.insert(0,str(self.local_profile['radius_mm']))
            self.summary.set('局部清洗与表面拟合已在采集时完成；拟合邻域只读。容忍尺度用于融合，输出体素与均匀间距用于最终密度控制。')

    def process(self):
        if self.working or self.loaded is None: return
        if not getattr(self,'can_write',lambda:True)():return
        try:
            values = {key:float(field.get()) for key,field in self.fields.items()}
            values['kalman_sampling']='surface_uniform'
            if self.local_profile:values['neighbors']=self.local_profile['neighbors']
            if getattr(self,'beta',False):
                from dataclasses import asdict
                values=asdict(Parameters())
                if self.local_profile:
                    values.update(radius_mm=self.local_profile['radius_mm'],neighbors=self.local_profile['neighbors'])
            params = Parameters(**values); params.validate()
        except ValueError as exc:
            messagebox.showerror('参数无效', str(exc), parent=self.root); return
        points = self.loaded.points
        backend='kalman'
        source=self.source
        self._begin('process', '正在预处理；可随时取消，原始数据保持不变')
        def progress(value, text): self.latest_progress = (value, text)
        def worker():
            try: self.events.put(('processed', process_input(source, points, params, backend=backend, progress=progress, cancel=self.cancel)))
            except ProcessingCancelled: self.events.put(('cancelled', None))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def skip(self, destination=None):
        if self.working or self.loaded is None:return
        if not getattr(self,'can_write',lambda:True)():return
        if destination is None:
            if getattr(self,'dataset',None):destination=stage_folder(self.dataset,2)
            else:
                from result_paths import stage_result_path
                destination=stage_result_path(2)
        source,count=self.source,len(self.loaded.points)
        self._begin('save','正在记录跳过预处理；直接使用一阶段数据…')
        def worker():
            try:
                from geometry_inputs import save_skip
                with replace_stage(destination,2,getattr(self,'dataset',None)) as target:
                    report=save_skip(source,target,count)
                self.events.put(('skipped',(Path(destination),report)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _show_result(self):
        if self.result is None: return
        choice = self.view.get()
        if choice=='被剔除的孤立点':
            points = self.result.removed
            if self.result.report.get('removed_points_coordinates_available') is False:
                self.status.set('孤立点已在采集时剔除，仅保留计数，未保存被剔除点的坐标')
        elif choice=='移动超过3mm的原位置':
            points = self.loaded.points[self.result.retained_indices[self.result.movement_mm>3]]
            if self.result.report['backend']=='kalman':
                self.status.set('卡尔曼从原始观测重建，无最终PLY逐点移动对应；可查看“多次观测确认的表面”')
        elif choice=='多次观测确认的表面':
            attributes=self.result.diagnostics
            points=self.result.points[attributes['support_views']>=2] if attributes is not None else np.empty((0,3))
            if attributes is None:self.status.set('该检查视图仅适用于卡尔曼表面融合')
        else: points = self.result.points
        self.after.set_cloud(points, np.empty((0,3)))
        self._sync_from_left()

    def _sync_from_left(self):
        if self.sync.get():
            camera = self.before.camera
            self.after.camera = ViewCamera(camera.position, camera.yaw, camera.pitch,camera.roll)
            self.after.request_draw()
            self.last_views = self._view_states()

    def _view_states(self):
        return [tuple(p.camera.position)+(p.camera.yaw,p.camera.pitch,p.camera.roll) for p in (self.before,self.after)]

    def _sync_views(self):
        states = self._view_states()
        if self.sync.get() and self.last_views is not None:
            changed = [states[i]!=self.last_views[i] for i in (0,1)]
            if any(changed):
                index = 1 if changed[1] and not changed[0] else 0
                src,dst = (self.before,self.after) if index==0 else (self.after,self.before)
                dst.camera = ViewCamera(src.camera.position, src.camera.yaw, src.camera.pitch,src.camera.roll); dst.request_draw()
                states = self._view_states()
        self.last_views = states

    def save(self, destination=None):
        if self.working or self.result is None: return
        if not getattr(self,'can_write',lambda:True)():return
        if destination is None and getattr(self,'dataset',None):
            destination=stage_folder(self.dataset,2)
        if destination is None:
            parent = filedialog.askdirectory(parent=self.root, initialdir=str(ROOT/'results'), title='选择输出父目录，将新建 setp2_时间 文件夹')
            if not parent: return
            from result_paths import stage_result_path
            destination = stage_result_path(2, parent)
        result,source,loaded = self.result,self.source,self.loaded
        self._begin('save', '正在另存点云和参数报告…')
        def worker():
            try:
                with replace_stage(destination,2,getattr(self,'dataset',None)) as target:
                    save_result(result,source,target,{'unit':loaded.unit,'coordinates':loaded.coordinates,'invalid_removed_on_load':loaded.removed})
                self.events.put(('saved',destination))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def open_reconstruction(self):
        if self.working is None and self.saved_path is not None and self.on_reconstruct is not None:
            self.on_reconstruct(self.saved_path)

    def _poll(self):
        if self.closed: return
        if self.latest_progress is not None:
            value,text = self.latest_progress; self.latest_progress = None
            self.bar['value'] = value*100; self.status.set(f'{value:.0%} · {text}')
        while True:
            try: kind,payload = self.events.get_nowait()
            except queue.Empty: break
            if kind in ('error','cancelled') and getattr(self,'on_task_failed',None):self.on_task_failed(str(payload or '已取消'))
            self.working = None; self.latest_progress = None
            if kind in ('skip_restored','skipped'):
                if kind=='skip_restored':
                    self.source,self.loaded,self.saved_path,report=payload
                    self._capture_profile()
                    self.before.clear();self.before.set_cloud(self.loaded.points)
                else:self.saved_path,report=payload
                self.result=None;self.after.clear();self.bar['value']=100
                self.status.set('二阶段已跳过；三阶段可选择原始点云Poisson或SDK深度TSDF')
                self.summary.set('未复制、平滑或均匀采样原始点云；跳过记录只引用源扫描。')
                if kind=='skipped':
                    if getattr(self,'on_stage_saved',None):self.on_stage_saved(2,self.saved_path)
                    self.open_reconstruction()
            elif kind=='restored':
                self.source,self.loaded,self.result,self.saved_path=payload
                self._capture_profile()
                self.before.clear();self.after.clear();self.before.set_cloud(self.loaded.points)
                self.view.set('处理结果');self._show_result();self.bar['value']=100
                self.status.set(f'已恢复：{self.saved_path} · 可修改参数重新处理并覆盖保存')
            elif kind=='loaded':
                self.saved_path = None
                self.source,self.loaded = payload; self.result = None
                self._capture_profile()
                self.before.clear(); self.after.clear()
                self.before.set_cloud(self.loaded.points, np.empty((0,3)))
                self._sync_from_left()
                if not self.local_profile:self.summary.set('已载入原始融合点云；开始处理后右侧显示结果。')
                self.status.set(f'{self.source.name} · {len(self.loaded.points):,} 点 · 源 {self.loaded.unit}/{self.loaded.coordinates}')
            elif kind=='processed':
                self.saved_path = None
                self.result = payload; self.view.set('处理结果'); self._show_result(); self.bar['value'] = 100
                r = payload.report; shift = r['movement_mm_p50_p90_p99_max']
                self.status.set(f'{r["backend"]} 处理完成 · {r["input_points"]:,} → {r["output_points"]:,} 点 · {r["elapsed_seconds"]:.1f}s · 尚未保存')
                self.summary.set(f'孤立点 {r["isolated_points"]:,} · 精确重复点 {r["exact_duplicates"]:,} · '
                    f'平滑位移中位/P90/最大 {shift[0]:.2f}/{shift[1]:.2f}/{shift[3]:.2f}mm · '
                    f'局部平面起伏中位(mm) {r["before"]["planar_sigma_mm_p50_p90"][:1]} → {r["after"].get("planar_sigma_mm_p50_p90",[])[:1]} '
                    '（非绝对精度；请检查边缘和薄结构）')
                if r['backend']=='kalman':
                    self.summary.set(f'{r["captures"]} 次原始拍摄 · 多次观测确认 {r["output_points_supported_by_multiple_views"]:,} 点 · '
                        f'每次拍摄每表面元最多更新一次 · 不确定度下限 {r["parameters"]["kalman_floor_mm"]:g}mm · 位姿固定 · 方差不是硬件精度保证')
                    sampling=r.get('uniform_sampling',{})
                    if sampling.get('algorithm','').startswith('surface-aware'):
                        self.summary.set(self.summary.get()+f' · 表面间距 {sampling["target_spacing_mm"]:g}mm · 减少过密点 {sampling["removed_for_density"]:,}')
                if getattr(self,'dataset',None):self.save()
            elif kind=='saved':
                self.saved_path = Path(payload)
                if getattr(self,'on_stage_saved',None):self.on_stage_saved(2,self.saved_path)
                self.status.set(f'已另存：{payload}')
            elif kind=='cancelled': self.status.set('已取消 · 原始数据及上次结果保留')
            elif kind=='error':
                self.status.set('操作失败 · 原始数据及上次结果保留')
                if not self.close_pending: messagebox.showerror('预处理失败', payload, parent=self.root)
            self._buttons()
        if self.close_pending and self.working is None:
            self.close(); return
        self._sync_views(); self.timer = self.root.after(40, self._poll)

    def close(self):
        if self.working:
            self.close_pending = True; self.cancel.set()
            self.status.set('正在停止处理…' if self.working!='save' else '等待结果保存完成后关闭…')
            return
        self.closed = True
        self.root.after_cancel(self.timer); self.root.destroy()


def launch(source=None):
    root = tk.Tk(); PreprocessWindow(root, source); root.mainloop()


if __name__=='__main__':
    launch()
