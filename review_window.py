from dataset_store import route_dataset
from runtime_paths import geometry_python
"""Fifth-stage read-only review of saved assets; no reconstruction is run."""
from pathlib import Path
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from appearance_window import AppearanceWindow
from geometry_window import ROOT, display_points
from mesh_view import MeshPanel
from map_cloud_view import MapCloudPanel
from rgb_preview import RGBPreviewPanel
from cloud_import import load_cloud
from saved_sources import review_sources
from result_paths import RESULTS, result_reports


class ReviewWindow(AppearanceWindow):
    def __init__(self, root):
        self.root=root; self.source=None; self.output=None; self.working=None; self.closed=False
        self.events=queue.Queue(); self.cancel=threading.Event(); self.lock=threading.Lock(); self.process=None
        self.last_views=None; self.mesh=None; self.colors=None; self.covered=None
        self.filled=None; self.regions=None; self.observed_colors=None; self.assets={}; self.frames=[]
        controls=ttk.Frame(root,padding=8); controls.pack(fill='x')
        self.source_button=ttk.Button(controls,text='导入第四阶段结果',command=self.open_result); self.source_button.pack(side='left')
        self.latest_button=ttk.Button(controls,text='查看最新结果',command=self.use_latest); self.latest_button.pack(side='left',padx=5)
        self.cancel_button=ttk.Button(controls,text='取消读取',command=self.cancel_job); self.cancel_button.pack(side='left')
        self.sync=tk.BooleanVar(value=True)
        ttk.Checkbutton(controls,text='同步三维视角',variable=self.sync,command=self.sync_from_left).pack(side='left',padx=8)
        self.left_mode=tk.StringVar(value='原始几何网格')
        self.left_select=ttk.Combobox(controls,textvariable=self.left_mode,state='readonly',width=17)
        self.left_select.pack(side='left'); self.left_select.bind('<<ComboboxSelected>>',lambda _:self.show_left())
        self.mode=tk.StringVar(value='RGB颜色')
        modes=ttk.Combobox(controls,textvariable=self.mode,values=('RGB颜色','仅实测颜色','覆盖检查','表面分区','灰色几何'),state='readonly',width=12)
        modes.pack(side='left',padx=8); modes.bind('<<ComboboxSelected>>',lambda _:self.update_mode())
        self.status=tk.StringVar(value='选择第四阶段结果，自动关联已保存点云、几何和RGB；全程只读')
        self.status_label=ttk.Label(root,textvariable=self.status,padding=8,wraplength=1400)
        self.status_label.pack(fill='x')
        self.metrics=tk.StringVar(value='读取数据后显示点云、网格与颜色覆盖指标')
        metrics_box=self.metrics_box=ttk.Labelframe(root,text='数据与质量指标',padding=6);metrics_box.pack(fill='x')
        ttk.Label(metrics_box,textvariable=self.metrics,wraplength=1380,justify='left').pack(fill='x')
        self.photo_controls=ttk.Frame(root)
        ttk.Button(self.photo_controls,text='上一张',command=lambda:self.change_photo(-1)).pack(side='left')
        ttk.Button(self.photo_controls,text='下一张',command=lambda:self.change_photo(1)).pack(side='left')
        self.photo_index=tk.StringVar(); self.photo_select=ttk.Combobox(self.photo_controls,textvariable=self.photo_index,state='readonly',width=42)
        self.photo_select.pack(side='left'); self.photo_select.bind('<<ComboboxSelected>>',lambda _:self.show_photo())
        panels=ttk.Panedwindow(root,orient='horizontal'); panels.pack(fill='both',expand=True)
        left=ttk.Labelframe(panels,text='源数据查看'); right=ttk.Labelframe(panels,text='最终RGB网格 / 覆盖检查')
        panels.add(left,weight=1); panels.add(right,weight=1)
        self.left_hosts={name:ttk.Frame(left) for name in ('mesh','cloud','rgb')}
        self.left_mesh=MeshPanel(self.left_hosts['mesh']); self.left_cloud=MapCloudPanel(self.left_hosts['cloud'])
        self.rgb=RGBPreviewPanel(self.left_hosts['rgb']); self.before=self.left_mesh; self.after=MeshPanel(right)
        self._buttons(); self.timer=root.after(40,self._poll)

    def _buttons(self):
        for button in (self.source_button,self.latest_button):
            button.configure(state='disabled' if self.working else 'normal')
        self.cancel_button.configure(state='normal' if self.working else 'disabled')

    def use_latest(self):
        if getattr(self,'dataset',None):
            from dataset_store import stage_folder,inspect_dataset
            if inspect_dataset(self.dataset)[4]['complete']:self.load_result(stage_folder(self.dataset,4))
            else:messagebox.showinfo('当前数据集','本数据集尚无完整第四阶段结果',parent=self.root)
            return
        # Accept only the completed production appearance pipeline.
        folders=[p.parent for p in RESULTS.glob('*/report.json') if 'setp4_' in p.parent.name]+[p.parent for p in RESULTS.glob('*/setp4/report.json')]
        for folder in sorted(folders,reverse=True):
            try: review_sources(folder)
            except ValueError: continue
            self.load_result(folder); return
        messagebox.showinfo('成果展示','尚无完整第四阶段结果',parent=self.root)

    def set_source(self,path):
        self.load_result(path)

    def open_result(self):
        path=filedialog.askdirectory(parent=self.root,initialdir=str(RESULTS),title='选择第四阶段着色目录')
        if path:self.load_result(path)

    def load_result(self,path):
        if route_dataset(self,path):return
        if self.working:return
        folder=Path(path).resolve(); folder=folder if folder.is_dir() else folder.parent
        self.working='load'; self.cancel.clear(); self._buttons(); self.status.set('正在读取保存的数据…')
        threading.Thread(target=self._load,args=(folder,),daemon=True).start()

    def _load(self,folder):
        try:
            info=review_sources(folder); assets={}; warnings=[]
            RESULTS.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.review_preview_',dir=RESULTS) as temporary:
                def mesh(path):
                    packet=Path(temporary)/'mesh.npz'
                    self._command([str(geometry_python()),str(ROOT/'mesh_read_worker.py'),str(path),str(packet)])
                    with np.load(packet) as data:return {key:data[key].copy() for key in data}
                final=mesh(folder/'mesh_colored.ply')
                for label,path,reader in (
                    ('原始几何网格', info['geometry']/'mesh.ply' if info['geometry'] else None,mesh),
                    ('预处理点云',info['preprocessed']/'processed_world.npy' if info['preprocessed'] else None,lambda p:display_points(np.load(p))),
                    ('原始点云',info['raw'],lambda p:load_cloud(p).points)):
                    if self.cancel.is_set():raise InterruptedError()
                    if path and path.is_file():
                        try:assets[label]=reader(path)
                        except InterruptedError:raise
                        except Exception as exc:warnings.append(f'{label}不可用：{exc}')
                extras={}
                if (folder/'color_attributes.npz').is_file():
                    with np.load(folder/'color_attributes.npz') as data:extras={key:data[key].copy() for key in ('covered','filled','surface_region','observed_colors') if key in data}
            self.events.put(('loaded',(folder,info,final,assets,extras,warnings)))
        except InterruptedError:self.events.put(('cancelled',None))
        except Exception as exc:self.events.put(('error',str(exc)))

    def show_left(self):
        for host in self.left_hosts.values():host.pack_forget()
        self.photo_controls.pack_forget()
        label=self.left_mode.get()
        if label=='原始RGB图像':
            self.left_hosts['rgb'].pack(fill='both',expand=True)
            self.photo_controls.pack(fill='x',before=self.left_hosts['rgb'].master.master)
            self.show_photo();return
        asset=self.assets.get(label)
        if asset is None:return
        if isinstance(asset,dict):
            self.before=self.left_mesh; kind='mesh'
            self.before.set_mesh(*(asset[k] for k in ('vertices','triangles','normals')))
        else:
            self.before=self.left_cloud; kind='cloud'; self.before.set_cloud(asset)
        self.left_hosts[kind].pack(fill='both',expand=True); self.sync_from_left()

    def show_photo(self):
        index=self.photo_select.current()
        if index<0 or index>=len(self.frames):return
        image=cv2.imdecode(np.fromfile(self.frames[index],dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is not None:self.rgb.update(image)
        else:self.rgb.clear();self.rgb.label.configure(text='RGB文件不可读取')

    def change_photo(self,delta):
        if self.frames:
            self.photo_select.current((max(0,self.photo_select.current())+delta)%len(self.frames));self.show_photo()

    def sync_from_left(self):
        if self.left_mode.get()!='原始RGB图像':super().sync_from_left()

    def _sync_views(self):
        if self.left_mode.get()!='原始RGB图像':super()._sync_views()

    def _poll(self):
        if self.closed:return
        while True:
            try:kind,payload=self.events.get_nowait()
            except queue.Empty:break
            if kind in ('error','cancelled') and getattr(self,'on_task_failed',None):self.on_task_failed(str(payload or '已取消'))
            if kind not in ('loaded','cancelled','error'):continue
            self.working=None
            if self.cancel.is_set() or kind=='cancelled':self.status.set('已取消读取')
            elif kind=='error':
                self.status.set('读取失败；已有显示保留');messagebox.showerror('成果展示',payload,parent=self.root)
            else:
                folder,info,final,self.assets,extras,warnings=payload
                self.source=self.output=folder; self.report=info['report'];self.mesh=tuple(final[k] for k in ('vertices','triangles','normals'))
                self.colors=final.get('colors');self.covered=extras.get('covered');self.filled=extras.get('filled')
                self.regions=extras.get('surface_region');self.observed_colors=extras.get('observed_colors',self.colors)
                self.frames=info['frames']; choices=list(self.assets)
                from review_metrics import describe
                self.metrics.set(describe(info,len(self.assets['原始点云']) if '原始点云' in self.assets else None))
                if self.frames:choices.append('原始RGB图像')
                self.left_select['values']=choices;self.left_mode.set(choices[0] if choices else '')
                self.photo_select['values']=[f'{i+1}/{len(self.frames)} · {p.name}' for i,p in enumerate(self.frames)]
                if self.frames:self.photo_select.current(0)
                self.after.clear();self.update_mode();self.show_left()
                self.status.set(f'{folder.name} · 可用源数据：'+('、'.join(choices) or '无，右侧网格可单独查看')+(' · '+'；'.join(warnings) if warnings else ''))
            self._buttons()
        self._sync_views();self.timer=self.root.after(40,self._poll)
