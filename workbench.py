"""Shared tabbed shell; stage views retain state until the application closes."""
import tkinter as tk
import threading
import queue
from pathlib import Path
from tkinter import ttk
from ui_dialogs import messagebox,filedialog,simpledialog
from dataset_store import create_dataset, dataset_for, inspect_dataset, stage_folder
from result_paths import RESULTS


class Workbench:
    def __init__(self, root, initial_stage='acquisition',beta=False):
        self.root = root
        self.beta=beta
        self.exporting=False;self.export_events=queue.Queue()
        self.closing = False
        self.dataset = None
        self.automation=None
        if beta:
            self.header=ttk.Frame(root);self.header.pack(fill='x')
            self.header.columnconfigure(0,weight=1)
            self.header_left=ttk.Frame(self.header);self.header_left.grid(row=0,column=0,sticky='nsew')
            self.metrics_host=ttk.Labelframe(self.header,text='数据与质量指标',padding=5)
        toolbar=ttk.Frame(self.header_left if beta else root,padding=6);toolbar.pack(fill='x')
        ttk.Button(toolbar,text='导入已有数据集',command=self.import_dataset).pack(side='left')
        ttk.Button(toolbar,text='新建数据集',command=self.new_dataset).pack(side='left',padx=6)
        if beta:
            self.export_button=ttk.Button(toolbar,text='导出数据集',command=self.export_data)
            self.export_button.pack(side='left',padx=6)
        self.dataset_status=tk.StringVar(value='未选择数据集 · 首次保存扫描时设置名称')
        ttk.Label(toolbar,textvariable=self.dataset_status,wraplength=1100).pack(side='left',padx=10)
        root.title('基于 RGB-D 的三维场景重建与交互建模')
        root.geometry('1450x950')
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='both', expand=True)
        self.frames = {}
        self.views = {}
        for name, title in (('acquisition', '一阶段 · 点云采集与查看'),
                            ('preprocessing', '二阶段 · 点云预处理'),
                            ('reconstruction', '三阶段 · 几何重建'),
                            ('appearance', '四阶段 · RGB外观'),
                            ('review', '五阶段 · 成果展示')):
            if beta:title={'acquisition':'扫描 / 导入','preprocessing':'处理','reconstruction':'处理','appearance':'处理','review':'成果展示'}[name]
            frame = ttk.Frame(self.tabs)
            self.frames[name] = frame
            self.tabs.add(frame, text=title)
            if beta and name not in ('acquisition','review'):self.tabs.hide(frame)
        if beta:
            from beta_pipeline import BetaPipeline
            self.automation=BetaPipeline(self,toolbar)
        self.select_stage(initial_stage)
        self.tabs.bind('<<NotebookTabChanged>>', self._tab_changed)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def _ensure_stage(self, name, hydrate=True):
        if name not in self.frames:
            raise ValueError(f'Unknown stage: {name}')
        if name not in self.views:
            if name == 'acquisition':
                from snapshot_window import VideoWindow as SnapshotWindow
                view = SnapshotWindow(self.frames[name], embedded=True,
                                      on_preprocess=self.open_preprocessing)
            elif name == 'preprocessing':
                from preprocess_window import PreprocessWindow
                view = PreprocessWindow(self.frames[name], embedded=True, on_reconstruct=self.open_reconstruction)
            elif name == 'reconstruction':
                from geometry_window import GeometryWindow
                view = GeometryWindow(self.frames[name])
            elif name == 'review':
                from review_window import ReviewWindow
                view = ReviewWindow(self.frames[name])
            else:
                from appearance_window import AppearanceWindow
                view = AppearanceWindow(self.frames[name])
            if self.beta:
                from beta_ui import apply_beta
                apply_beta(view,name)
            self.views[name] = view
            view.dataset=self.dataset
            view.on_stage_saved=self.stage_saved
            view.on_import_dataset=self.import_dataset
            if self.beta and name=='acquisition':view.import_button.configure(text='导入数据集',command=self.import_dataset)
            if self.beta and name=='review':
                view.source_button.configure(text='导入数据集',command=self.import_dataset)
                self._place_review_metrics(view)
            view.can_write=lambda candidate=view:self.can_write(candidate)
            if self.dataset and hydrate:self._hydrate(name,view)
        elif hydrate and getattr(self.views[name],'dataset_refresh',False) and self.dataset:
            view=self.views[name]
            if not getattr(view,'working',None) and not getattr(view,'busy',False):
                view.dataset_refresh=False;self._hydrate(name,view)
        return self.views[name]

    def _place_review_metrics(self, view):
        # Keep the notebook's expanding area for images; details stay in the header.
        view.status_label.pack_forget();view.metrics_box.pack_forget()
        for child in self.metrics_host.winfo_children():child.destroy()
        text=tk.Text(self.metrics_host,height=7,width=68,wrap='word',font='TkDefaultFont',
                     relief='flat',borderwidth=0,background=self.root.cget('background'),
                     state='disabled',takefocus=False)
        scroll=ttk.Scrollbar(self.metrics_host,orient='vertical',command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');text.pack(side='left',fill='both',expand=True)
        def update(*_):
            if text.winfo_exists():
                text.configure(state='normal');text.delete('1.0','end')
                text.insert('1.0',view.metrics.get());text.configure(state='disabled');text.yview_moveto(0)
        view.metrics.trace_add('write',update);update()

    def _busy(self):
        return bool(self.automation and self.automation.active) or self.exporting or any(getattr(v,'working',None) or getattr(v,'busy',False) or getattr(v,'importing',False) for v in self.views.values())

    def can_write(self, candidate):
        if self.automation and self.automation.active and candidate is not self.automation.view:
            candidate.status.set('自动处理正在运行，请等待完成或先停止处理');return False
        if self.exporting:
            candidate.status.set('请等待数据导出完成');return False
        if self.dataset and any(v is not candidate and (getattr(v,'working',None) in ('process','save','reconstruct','colorize') or getattr(v,'busy',False)) for v in self.views.values()):
            candidate.status.set('请等待其他阶段的任务完成，再处理或覆盖当前数据集');return False
        return True

    def export_data(self):
        if self._busy():
            self.dataset_status.set('请等待当前任务完成后导出');return
        scan=self.views.get('acquisition')
        if scan and scan.session and scan.has_map and scan.session.folder.exists() and scan.session.folder.name.startswith('.pending_scan_'):
            self.dataset_status.set('请先点击“完成”保存当前扫描，再导出');return
        if self.dataset is None:
            self.dataset_status.set('请先保存扫描或导入已有数据集');return
        path=filedialog.asksaveasfilename(parent=self.root,title='导出数据集',defaultextension='.zip',
            initialfile=self.dataset.name+'.zip',filetypes=[('数据集压缩包','*.zip')])
        if not path:return
        source=self.dataset
        self.exporting=True;self.export_button.configure(state='disabled')
        self.dataset_status.set('正在导出数据集…')
        def worker():
            try:
                from dataset_export import export_dataset
                self.export_events.put(('done',export_dataset(source,path,overwrite=True)))
            except Exception as exc:self.export_events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()
        self.root.after(80,self._poll_export)

    def _poll_export(self):
        try:kind,value=self.export_events.get_nowait()
        except queue.Empty:
            self.root.after(80,self._poll_export);return
        self.exporting=False;self.export_button.configure(state='normal')
        if kind=='done':self.dataset_status.set(f'已导出：{value}')
        else:
            self.refresh_dataset_status();messagebox.showerror('导出失败',value,parent=self.root)

    def new_dataset(self):
        if self._busy():
            self.dataset_status.set('请等待当前任务完成后切换数据集');return
        scan=self.views.get('acquisition')
        if scan and scan.session and scan.has_map and scan.session.folder.exists() and scan.session.folder.name.startswith('.pending_scan_'):
            self.dataset_status.set('请先保存当前扫描，再新建数据集');return
        name=simpledialog.askstring('新建数据集','数据名称（例如 test）',parent=self.root)
        if name is None:return
        try:self.import_dataset(create_dataset(RESULTS,name))
        except Exception as exc:messagebox.showerror('新建数据集',str(exc),parent=self.root)

    def import_dataset(self,path=None):
        if self._busy():
            self.dataset_status.set('请等待当前任务完成后切换数据集');return
        scan=self.views.get('acquisition')
        if scan is not None and scan.session is not None and scan.has_map and scan.session.folder.exists() and scan.session.folder.name.startswith('.pending_scan_'):
            self.dataset_status.set('当前有未保存扫描，请先点击“完成”再切换数据集');return
        if path is None:path=filedialog.askdirectory(parent=self.root,initialdir=str(RESULTS),title='选择数据集目录（例如 test）')
        if not path:return
        try:
            folder=Path(path).resolve();inspect_dataset(folder)
        except Exception as exc:
            messagebox.showerror('导入数据集',str(exc),parent=self.root);return
        selected=next((name for name,frame in self.frames.items() if str(frame)==self.tabs.select()),'acquisition')
        for view in self.views.values():
            if hasattr(view,'shutdown'):view.shutdown()
            else:
                view.closed=True;view.root.after_cancel(view.timer)
        # Replace frames as well as children so bindings release the old view/data.
        for name,frame in list(self.frames.items()):
            index=self.tabs.index(frame);title=self.tabs.tab(frame,'text')
            frame.destroy()
            replacement=ttk.Frame(self.tabs)
            self.tabs.insert('end' if index>=len(self.tabs.tabs()) else index,replacement,text=title)
            self.frames[name]=replacement
        self.views.clear();self.dataset=folder
        if self.beta:
            self.metrics_host.grid_remove()
            for child in self.metrics_host.winfo_children():child.destroy()
        if self.beta:
            selected='acquisition'
            for name in ('preprocessing','reconstruction','appearance'):self.tabs.hide(self.frames[name])
        self.refresh_dataset_status();self.select_stage(selected)
        if self.automation:self.automation.start()

    def refresh_dataset_status(self):
        if self.dataset:
            states=inspect_dataset(self.dataset)
            if self.beta:
                self.dataset_status.set(f'数据集：{self.dataset.name}');return
            self.dataset_status.set(f'数据集：{self.dataset.name} · '+ ' / '.join(f'{n}阶段：{states[n]["state"]}' for n in range(1,5)))

    def stage_saved(self,stage,path):
        folder=dataset_for(path)
        if folder is None:return
        self.dataset=folder
        for name,view in self.views.items():
            view.dataset=folder
            number={'acquisition':1,'preprocessing':2,'reconstruction':3,'appearance':4,'review':5}[name]
            if number>stage:view.dataset_refresh=True
        self.refresh_dataset_status()
        if self.automation:
            if stage==1 and not self.closing:self.automation.start()
            else:self.automation.saved(stage)

    def _hydrate(self,name,view):
        states=inspect_dataset(self.dataset)
        path=lambda n:states[n]['path']
        if name=='acquisition' and states[1]['complete']:view.import_cloud(path(1)/'map_cloud.ply')
        elif name=='preprocessing':
            if states[2]['complete']:view.load_saved(path(2),path(1) if states[1]['complete'] else None)
            elif states[1]['complete']:view.load(path(1))
        elif name=='reconstruction':
            if states[3]['complete']:view.load_result(path(3))
            elif states[2]['complete']:view.set_source(path(2))
        elif name=='appearance':
            if states[4]['complete']:view.load_result(path(4))
            elif states[3]['complete']:view.set_source(path(3))
        elif name=='review' and states[4]['complete']:view.load_result(path(4))

    def select_stage(self, name):
        view = self._ensure_stage(name)
        if not self.beta or name in ('acquisition','review'):self.tabs.select(self.frames[name])
        return view

    def _tab_changed(self, event=None):
        if self.closing:
            return
        selected = self.tabs.select()
        for name, frame in self.frames.items():
            if str(frame) == selected:
                view = self._ensure_stage(name)
                if self.beta:
                    if name=='review':self.metrics_host.grid(row=0,column=1,sticky='nsew',padx=(8,6),pady=5)
                    else:self.metrics_host.grid_remove()
                preprocessing = self.views.get('preprocessing')
                if name == 'reconstruction' and view.source is None and view.working is None and preprocessing is not None and preprocessing.saved_path is not None:
                    view.set_source(preprocessing.saved_path)
                geometry = self.views.get('reconstruction')
                if name == 'appearance' and view.source is None and view.working is None and geometry is not None and geometry.output is not None:
                    view.set_source(geometry.output)
                appearance = self.views.get('appearance')
                if name == 'review' and view.source is None and view.working is None and appearance is not None and appearance.output is not None:
                    view.set_source(appearance.output)
                break

    def open_reconstruction(self, source=None):
        view = self.select_stage('reconstruction')
        if source is not None and view.working is None and (view.source is None or view.source.resolve() != source.resolve()):
            view.set_source(source)

    def open_preprocessing(self, source=None, metadata=None):
        if self.automation:
            self.automation.start();return
        view = self.select_stage('preprocessing')
        if source is None:
            return
        if view.working is not None:
            view.status.set('当前任务仍在运行；完成后可导入采集结果')
            return
        if view.source is not None and view.source.resolve() == source.resolve():
            return
        if view.result is not None and not messagebox.askyesno(
                '切换预处理数据', '导入此点云将替换当前预处理视图。是否继续？', parent=self.root):
            return
        view.unit.set({'m': '米', 'mm': '毫米'}[metadata[0]] if metadata else '自动')
        view.axes.set({'world': '世界', 'camera': '相机'}[metadata[1]] if metadata else '自动')
        view.load(source)

    def close(self):
        if self.exporting:
            self.dataset_status.set('正在导出，完成后关闭…');self.root.after(100,self.close);return
        if self.closing:
            return
        self.closing = True
        if self.automation:self.automation.stop()
        # Stop camera/mapping and allow its existing save prompt to finish.
        scan = self.views.get('acquisition')
        if scan is not None and scan.busy:
            scan.finish()
        preprocessing = self.views.get('preprocessing')
        if preprocessing is not None and preprocessing.working in ('load', 'process'):
            preprocessing.cancel.set()
        geometry = self.views.get('reconstruction')
        if geometry is not None and geometry.working is not None:
            geometry.cancel_job()
        appearance = self.views.get('appearance')
        if appearance is not None and appearance.working is not None:
            appearance.cancel_job()
        review = self.views.get('review')
        if review is not None and review.working is not None: review.cancel_job()
        self._wait_close()

    def _wait_close(self):
        scan = self.views.get('acquisition')
        preprocessing = self.views.get('preprocessing')
        geometry = self.views.get('reconstruction')
        appearance = self.views.get('appearance')
        review = self.views.get('review')
        if (review is not None and review.working is not None) or (scan is not None and (scan.busy or scan.importing)) or (
                preprocessing is not None and preprocessing.working is not None) or (
                geometry is not None and geometry.working is not None) or (
                appearance is not None and appearance.working is not None):
            self.root.after(50, self._wait_close)
            return
        if preprocessing is not None:
            preprocessing.closed = True
            preprocessing.root.after_cancel(preprocessing.timer)
        if geometry is not None: geometry.shutdown()
        if appearance is not None: appearance.shutdown()
        if review is not None: review.shutdown()
        self.root.destroy()


def launch(initial_stage='acquisition',beta=False):
    root = tk.Tk()
    Workbench(root, initial_stage,beta=beta)
    root.mainloop()
