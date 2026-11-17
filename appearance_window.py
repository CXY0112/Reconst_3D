from dataset_store import route_dataset
from runtime_paths import geometry_python
"""Fourth-stage RGB baseline, with matched gray/color/coverage mesh views."""
from pathlib import Path
from contextlib import nullcontext
from dataset_store import replace_stage, stage_folder
import json
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
from geometry_window import GeometryWindow,ROOT
from mesh_view import MeshPanel
from result_paths import RESULTS,stage_result_path,result_reports
from saved_sources import reference


class AppearanceWindow(GeometryWindow):
    def __init__(self,root):
        self.root=root;self.source=None;self.output=None;self.report=None;self.mesh=None;self.colors=None;self.covered=None
        self.filled=None;self.regions=None;self.observed_colors=None
        self.working=None;self.closed=False;self.last_views=None;self.session_override=None
        self.events=queue.Queue();self.cancel=threading.Event();self.lock=threading.Lock();self.process=None
        controls=ttk.Frame(root,padding=8);controls.pack(fill='x')
        self.source_button=ttk.Button(controls,text='导入几何结果',command=self.open_source);self.source_button.pack(side='left',padx=3)
        self.latest_button=ttk.Button(controls,text='使用最新几何结果',command=self.use_latest);self.latest_button.pack(side='left',padx=3)
        self.start_button=ttk.Button(controls,text='开始RGB着色',command=self.start);self.start_button.pack(side='left',padx=3)
        self.cancel_button=ttk.Button(controls,text='取消',command=self.cancel_job);self.cancel_button.pack(side='left',padx=3)
        self.result_button=ttk.Button(controls,text='导入着色结果',command=self.open_result);self.result_button.pack(side='left',padx=3)
        self.sync=tk.BooleanVar(value=True)
        ttk.Checkbutton(controls,text='同步视角',variable=self.sync,command=self.sync_from_left).pack(side='left',padx=8)
        self.mode=tk.StringVar(value='RGB颜色')
        modes=ttk.Combobox(controls,textvariable=self.mode,values=('RGB颜色','仅实测颜色','覆盖检查','表面分区','灰色几何'),state='readonly',width=10)
        modes.pack(side='left');modes.bind('<<ComboboxSelected>>',lambda _:self.update_mode())
        self.source_text=tk.StringVar(value='请选择第三阶段几何结果；自动追溯源扫描的RGB、深度、标定和位姿')
        ttk.Label(root,textvariable=self.source_text,padding=8,wraplength=1400).pack(fill='x')
        ttk.Label(root,text='逐顶点选最佳照片 · 保守提高RGB覆盖 · 几何表面分区内补全小块缺色 · 几何和位姿保持不变',padding=8).pack(fill='x')
        self.status=tk.StringVar(value='就绪 · 无需连接相机')
        ttk.Label(root,textvariable=self.status,padding=8,wraplength=1400).pack(fill='x')
        self.bar=ttk.Progressbar(root,maximum=100);self.bar.pack(fill='x',padx=10)
        panels=ttk.Panedwindow(root,orient='horizontal');panels.pack(fill='both',expand=True)
        left=ttk.Labelframe(panels,text='原始几何网格');right=ttk.Labelframe(panels,text='RGB着色 / 覆盖检查')
        panels.add(left,weight=1);panels.add(right,weight=1)
        self.before=MeshPanel(left);self.after=MeshPanel(right)
        self.summary=tk.StringVar(value='结果自动保存至 setp4_时间；覆盖检查：绿=实测，蓝=插值补全，红=缺色。表面分区是几何分区，尚非物体语义分割。')
        ttk.Label(root,textvariable=self.summary,padding=8,wraplength=1400).pack(fill='x')
        self._buttons();self.timer=root.after(40,self._poll)

    def _buttons(self):
        super()._buttons()

    def choose_session(self):
        path=filedialog.askdirectory(parent=self.root,initialdir=str(RESULTS),title='选择此网格对应的原始扫描；须包含RGB、深度、标定和位姿')
        if path:
            self.session_override=Path(path).resolve()
            self.source_text.set(f'几何：{self.source} · 手动指定扫描：{self.session_override}')

    def open_source(self):
        path=filedialog.askdirectory(parent=self.root,initialdir=str(RESULTS),title='选择第三阶段几何目录（setp3）')
        if path:self.set_source(path)

    def use_latest(self):
        if self.working:return
        if getattr(self,'dataset',None):
            source=stage_folder(self.dataset,3)
            if (source/'mesh.ply').is_file():self.set_source(source)
            else:messagebox.showinfo('当前数据集','请先完成本数据集的第三阶段',parent=self.root)
            return
        reports=result_reports('final')
        if reports:self.set_source(reports[-1][0].parent)
        else:messagebox.showerror('暂无几何结果','请先在第三阶段重建并保存',parent=self.root)

    def set_source(self,path):
        if route_dataset(self,path):return
        if self.working:return
        folder=Path(path).resolve();folder=folder if folder.is_dir() else folder.parent
        self._begin('load','正在读取几何及源扫描关联…')
        threading.Thread(target=self._job,args=(folder,'source'),daemon=True).start()

    def open_result(self):
        path=filedialog.askdirectory(parent=self.root,initialdir=str(RESULTS),title='选择第四阶段着色目录（setp4）')
        if path:self.load_result(path)

    def load_result(self,path):
        if route_dataset(self,path):return
        if self.working:return
        self._begin('result','正在读取着色结果…')
        threading.Thread(target=self._job,args=(Path(path).resolve(),'result'),daemon=True).start()

    def start(self):
        if self.working or self.source is None:return
        if not getattr(self,'can_write',lambda:True)():return
        self._begin('colorize','正在筛选RGB观测并着色…')
        threading.Thread(target=self._job,args=(stage_folder(self.dataset,4) if getattr(self,'dataset',None) else stage_result_path(4),'colorize'),daemon=True).start()

    def _job(self,folder,kind):
        try:
            context=replace_stage(folder,4,getattr(self,'dataset',None)) if kind=='colorize' else nullcontext(folder)
            with context as destination:
                payload=self._job_data(destination,kind)
                if self.cancel.is_set():raise InterruptedError()
            self.events.put(('done',(folder,*payload)))
        except InterruptedError:self.events.put(('cancelled',None))
        except Exception as exc:self.events.put(('error',str(exc)))

    def _job_data(self,folder,kind):
        RESULTS.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.appearance_preview_',dir=RESULTS) as temporary:
            if Path(temporary).resolve().parent!=RESULTS.resolve():raise ValueError('Unsafe preview directory')
            packet=Path(temporary)/'mesh.npz'
            if kind=='colorize':
                command=[str(geometry_python()),'-u',str(ROOT/'tools/colorize_geometry.py'),
                    '--worker','--source',str(self.source),'--output',str(folder),'--preview-packet',str(packet)]
                if self.session_override:command+=['--session',str(self.session_override)]
                self._command(command)
            else:
                report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
                from geometry_methods import is_geometry_report
                valid=is_geometry_report(report) if kind=='source' else report.get('complete') and report.get('pipeline')=='rgb_vertex_color_baseline'
                if not valid:raise ValueError('请选择对应阶段的完整结果目录')
                self._command([str(geometry_python()),'-u',str(ROOT/'mesh_read_worker.py'),
                               str(folder/('mesh.ply' if kind=='source' else 'mesh_colored.ply')),str(packet)])
            report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
            with np.load(packet) as data:
                mesh=tuple(data[key].copy() for key in ('vertices','triangles','normals'))
                colors=data['colors'].copy() if 'colors' in data else None
            covered=None;extras={}
            if kind!='source':
                with np.load(folder/'color_attributes.npz') as data:
                    covered=data['covered'].copy()
                    extras={key:data[key].copy() for key in ('filled','surface_region','observed_colors') if key in data}
                if colors is None or covered.shape!=(len(mesh[0]),):raise ValueError('着色结果或覆盖属性不完整')
            return kind,report,mesh,colors,covered,extras

    def update_mode(self):
        if self.mesh is None:return
        colors=self.colors
        if self.mode.get()=='灰色几何':colors=None
        elif self.mode.get()=='仅实测颜色':colors=self.observed_colors
        elif self.mode.get()=='表面分区':
            if self.regions is not None:
                labels=self.regions.astype(np.uint64)+1
                colors=np.column_stack([.2+.65*((labels*factor)%251)/250 for factor in (37,71,131)])
            else:colors=None
        elif self.mode.get()=='覆盖检查':
            colors=np.tile([.85,.25,.2],(len(self.mesh[0]),1))
            if self.covered is not None:colors[self.covered]=[.15,.7,.35]
            if self.filled is not None:colors[self.filled]=[.2,.45,.95]
        self.after.set_mesh(*self.mesh,colors=colors)

    def _poll(self):
        if self.closed:return
        while True:
            try:kind,payload=self.events.get_nowait()
            except queue.Empty:break
            if kind in ('error','cancelled') and getattr(self,'on_task_failed',None):self.on_task_failed(str(payload or '已取消'))
            if kind=='progress':continue
            if kind=='rgb_progress':
                current,total=payload;self.bar['value']=100*current/total
                self.status.set(f'正在筛选并融合RGB · {current}/{total}次拍摄');continue
            self.working=None
            if self.cancel.is_set():self.status.set('已停止；未完成结果不会作为正式输出使用')
            elif kind=='done':
                folder,job,self.report,self.mesh,self.colors,self.covered,extras=payload
                self.filled=extras.get('filled');self.regions=extras.get('surface_region')
                self.observed_colors=extras.get('observed_colors',self.colors)
                self.source=folder if job=='source' else reference(self.report['input'],folder)
                if job=='source':self.session_override=None
                self.output=None if job=='source' else folder
                if job=='colorize' and getattr(self,'on_stage_saved',None):self.on_stage_saved(4,folder)
                self.before.clear();self.after.clear();self.before.set_mesh(*self.mesh);self.update_mode();self.sync_from_left()
                self.source_text.set(str(self.source));self.bar['value']=100
                if job=='source':self.status.set('已载入几何；点击“开始RGB着色”检查源扫描并生成基线')
                else:
                    coverage=self.report['coverage']
                    self.status.set(f'完成 · 实测顶点 {coverage["vertex_fraction"]:.1%} · 插值补全 {coverage.get("interpolated_vertices",0):,} 点 · 着色面积 {coverage.get("display_fully_colored_area_fraction",coverage["fully_colored_area_fraction"]):.1%} · {self.report["elapsed_seconds"]:.1f}s')
                    self.summary.set(f'已保存：{folder} · 覆盖检查：绿=实测，蓝=插值，红=缺色；表面分区是几何分区，尚非物体语义分割')
            elif kind=='error':
                self.status.set('操作失败 · 原显示和已有结果保留')
                messagebox.showerror('RGB外观重建',payload,parent=self.root)
            self._buttons()
        self._sync_views();self.timer=self.root.after(40,self._poll)
