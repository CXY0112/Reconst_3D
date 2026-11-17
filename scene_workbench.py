"""Five-stage RGB-D geometry workbench and its streamlined Beta flow."""
from pathlib import Path
import queue,threading
import tkinter as tk
from tkinter import ttk
from ui_dialogs import filedialog,messagebox
from workbench import Workbench
from scene_stages import source_for,base_for,read_report

STAGES={'acquisition':1,'preprocessing':2,'scene_reconstruction':3,'segmentation':4,'review':5}
TITLES=('一阶段 · 点云采集','二阶段 · 预处理（可跳过）','三阶段 · 场景重建','四阶段 · 场景分割 / 精细建模','五阶段 · 成果展示')


class SceneWorkbench(Workbench):
    def __init__(self,root,beta=False):
        # Reuse dataset import/export utilities, without constructing the old workflow.
        self.root=root;self.beta=beta;self.dataset=None;self.closing=False;self.automation=None
        self.exporting=False;self.export_events=queue.Queue();self.auto_active=False;self.auto_phase=None;self.auto_pending=False
        self.auto_cancel=threading.Event();self.auto_events=queue.Queue();self.auto_timer=None
        root.title('基于 RGB-D 的三维场景重建与交互建模'+(' · Beta' if beta else ''))
        from desktop_display import configure_window
        configure_window(root,beta)
        from beta_theme import install,BetaHeader
        install(root);self.beta_header=BetaHeader(self)
        self.tabs=ttk.Notebook(root);self.tabs.pack(fill='both',expand=True,padx=16,pady=(0,12));self.frames={};self.views={}
        for (name,number),title in zip(STAGES.items(),TITLES):
            frame=ttk.Frame(self.tabs);self.frames[name]=frame
            self.tabs.add(frame,text={'acquisition':'拍摄 / 导入','segmentation':'场景分割 / 精细建模','review':'成果展示'}.get(name,title) if beta else title)
            if beta and name in ('preprocessing','scene_reconstruction'):self.tabs.hide(frame)
        self.select_stage('acquisition');self.tabs.bind('<<NotebookTabChanged>>',self._tab_changed)
        root.protocol('WM_DELETE_WINDOW',self.close)

    def _ensure_stage(self,name,hydrate=True):
        if name not in STAGES:raise ValueError('未知工作阶段')
        if name not in self.views:
            if name=='acquisition':
                from scene_cloud_window import SceneVideoWindow
                view=SceneVideoWindow(self.frames[name],embedded=True,on_preprocess=self.open_preprocessing)
                if self.beta:
                    view.beta=True;view.rate_row.pack_forget();view.preprocess_button.pack_forget()
                    view.import_button.configure(text='导入数据集',command=self.import_dataset)
            elif name=='preprocessing':
                from scene_cloud_window import ScenePreprocessWindow
                view=ScenePreprocessWindow(self.frames[name],embedded=True,on_reconstruct=self.open_scene_reconstruction)
            elif name=='scene_reconstruction':
                from scene_reconstruction_window import SceneReconstructionWindow
                view=SceneReconstructionWindow(self.frames[name],kind='base')
                view.on_refine=self.open_segmentation;view.on_review=self.open_review
                view.on_ready=self._base_ready;view.on_complete=self._base_completed;view.on_failed=self.stop_beta
                if self.beta:
                    view.method.set('Poisson');view.method_box.pack_forget();view.smoothing_iterations=20
            elif name=='segmentation':
                from scene_refinement_window import SceneRefinementWindow
                view=SceneRefinementWindow(self.frames[name]);view.on_review=self.open_review
            else:
                from scene_review_window import SceneReviewWindow
                view=SceneReviewWindow(self.frames[name]);self._place_review_metrics(view)
            from beta_theme import style_view
            style_view(view,name)
            self.views[name]=view;view.dataset=self.dataset;view.on_stage_saved=self.stage_saved
            view.on_import_dataset=self.import_dataset;view.can_write=lambda candidate=view:self.can_write(candidate)
            if hydrate and self.dataset:self._hydrate(name,view)
        elif hydrate and self.dataset and getattr(self.views[name],'dataset_refresh',False):
            view=self.views[name]
            if not getattr(view,'working',None) and not getattr(view,'busy',False):
                view.dataset_refresh=False;self._hydrate(name,view)
        return self.views[name]

    def _hydrate(self,name,view):
        view.dataset_refresh=False
        root=Path(self.dataset);source=source_for(root)
        if name=='acquisition':
            if (root/'setp1/map_cloud.ply').is_file():view.import_cloud(root/'setp1/map_cloud.ply')
        elif name=='preprocessing':
            if (root/'rgbd_scene/setp2/report.json').is_file():view.load_saved(root/'rgbd_scene/setp2',root/'setp1')
            elif (root/'setp1/map_cloud.ply').is_file():view.load(root/'setp1')
        elif name=='scene_reconstruction':
            if source.is_dir():view.set_source(source)
        elif name=='segmentation':
            from scene_segmentation import fingerprint
            report=read_report(base_for(root)/'report.json')
            try:valid=report.get('complete') and report.get('source_fingerprint')==fingerprint(source)
            except (ValueError,OSError):valid=False
            if valid:view.set_source(source)
            else:view.status.set('请先完成第三阶段几何重建，再选择需要精细化的区域')
        elif name=='review':
            if (base_for(root)/'report.json').is_file():view.set_source(source)

    def select_stage(self,name):
        view=self._ensure_stage(name)
        if not self.beta or name not in ('preprocessing','scene_reconstruction'):self.tabs.select(self.frames[name])
        return view

    def _tab_changed(self,event=None):
        if self.closing:return
        name=next((k for k,v in self.frames.items() if str(v)==self.tabs.select()),None)
        if not name:return
        view=self._ensure_stage(name)
        if name=='review':self.metrics_host.grid(row=0,column=1,sticky='nsew',padx=6,pady=5)
        else:self.metrics_host.grid_remove()
        if name=='scene_reconstruction' and not view.source and not view.working:
            previous=self.views.get('preprocessing')
            if previous and previous.saved_path:view.set_source(previous.saved_path)

    def _place_review_metrics(self,view):
        for child in view.root.winfo_children():
            if child.winfo_class()=='TLabel' and str(child.cget('textvariable'))==str(view.metrics):child.pack_forget()
        for child in self.metrics_host.winfo_children():child.destroy()
        from beta_theme import MetricsPanel
        MetricsPanel(self.metrics_host,view).pack(fill='both',expand=True)

    def open_preprocessing(self,source=None,metadata=None):
        if self.beta:self.start_beta();return
        view=self.select_stage('preprocessing')
        if source is not None and not view.working and view.source is None:view.load(source)
        return view

    def open_scene_reconstruction(self,source=None):
        view=self._ensure_stage('scene_reconstruction',hydrate=False)
        if not self.beta:self.tabs.select(self.frames['scene_reconstruction'])
        if not view.working:
            if self.dataset:self._hydrate('scene_reconstruction',view)
            elif source:view.set_source(source)
        return view

    def open_segmentation(self,source=None):
        view=self.select_stage('segmentation')
        if not self.dataset and not view.working:
            previous=self.views.get('scene_reconstruction')
            if previous and previous.base:view.set_source(previous.source,previous.base[0])
        return view
    def open_review(self):
        view=self.select_stage('review');stage3=self.views.get('scene_reconstruction')
        if stage3 and stage3.base:
            from scene_stages import result_kind,MESH_VIEW,SPLAT_VIEW
            view.preferred_result=result_kind(stage3.base[1])
            if not view.working and view.preferred_result in view.bases:
                view.right_mode.set(SPLAT_VIEW if view.preferred_result=='gaussian' else MESH_VIEW);view.show_right()
        return view

    def _busy(self):return self.auto_active or self.auto_pending or super()._busy()

    def can_write(self,candidate):
        if self.closing and candidate is not self.views.get('acquisition'):return False
        if self.auto_active and candidate is not self.views.get('scene_reconstruction'):
            candidate.status.set('请等待自动重建完成或停止处理');return False
        if self.exporting:return False
        if any(v is not candidate and (getattr(v,'working',None) or getattr(v,'busy',False)) for v in self.views.values()):
            candidate.status.set('请等待其他阶段的任务完成');return False
        return True

    def stage_saved(self,stage,path):
        from dataset_store import dataset_for,read_manifest
        folder=dataset_for(path)
        if folder is None:return
        self.dataset=folder
        for name,view in self.views.items():
            view.dataset=folder
            if STAGES[name]>stage:view.dataset_refresh=True
        if stage in (1,2,3,4):
            manifest=read_manifest(folder);branch=manifest.setdefault('branches',{}).setdefault('rgbd_scene',{})
            branch['workflow_version']=2;stages=branch.setdefault('stages',{})
            stages[str(stage)]=dict(folder=str(Path(path).relative_to(folder)),stale=False)
            for downstream in range(stage+1,5):
                if str(downstream) in stages:stages[str(downstream)]['stale']=True
            from scene_reconstruction import write_json
            write_json(folder/'dataset.json',manifest)
        self.refresh_dataset_status()
        if self.beta and stage==1 and not self.closing:self.root.after(100,self.start_beta)

    def refresh_dataset_status(self):
        if self.dataset:self.dataset_status.set('数据集：'+self.dataset.name+(' · 自动生成几何模型中…' if self.auto_active else ''))

    def import_dataset(self,path=None):
        if self._busy():self.dataset_status.set('请等待当前任务完成后切换数据集');return
        scan=self.views.get('acquisition')
        if scan and scan.session and scan.has_map and scan.session.folder.exists() and scan.session.folder.name.startswith('.pending_scan_'):
            self.dataset_status.set('请先点击完成，保存当前扫描');return
        if path is None:path=filedialog.askdirectory(parent=self.root,title='选择数据集目录',initialdir=str(Path(__file__).parent/'results'))
        if not path:return
        from dataset_store import inspect_dataset
        try:folder=Path(path).resolve();inspect_dataset(folder)
        except Exception as exc:messagebox.showerror('导入数据集',str(exc),parent=self.root);return
        for view in self.views.values():self._shutdown(view)
        self.views.clear()
        for name,frame in list(self.frames.items()):
            index=self.tabs.index(frame);title=self.tabs.tab(frame,'text');frame.destroy()
            replacement=ttk.Frame(self.tabs);self.tabs.insert('end' if index>=len(self.tabs.tabs()) else index,replacement,text=title);self.frames[name]=replacement
            if self.beta and name in ('preprocessing','scene_reconstruction'):self.tabs.hide(replacement)
        self.dataset=folder;self.refresh_dataset_status();self.metrics_host.grid_remove()
        cached=False
        if self.beta:
            from scene_segmentation import fingerprint
            report=read_report(base_for(folder)/'report.json')
            pre=read_report(folder/'rgbd_scene/setp2/report.json')
            geometry=read_report(base_for(folder)/'geometry/report.json')
            params=pre.get('parameters',{})
            try:cached=(report.get('complete') and report.get('source_fingerprint')==fingerprint(source_for(folder))
                and not pre.get('skipped') and params.get('output_voxel_mm')==1 and params.get('surface_spacing_mm')==1
                and geometry.get('parameters',{}).get('smoothing_iterations')==20 and report.get('method')=='poisson'
                and report.get('geometry_only') is True)
            except (OSError,ValueError):pass
        if cached:self.open_review()
        else:
            self.select_stage('acquisition')
            if self.beta and (folder/'setp1/map_cloud.ply').is_file():self.root.after(100,self._start_beta_when_idle)

    def _start_beta_when_idle(self):
        if self.closing:return
        if any(getattr(v,'working',None) or getattr(v,'importing',False) for v in self.views.values()):self.root.after(100,self._start_beta_when_idle)
        else:self.start_beta()

    def start_beta(self):
        if not self.beta or self._busy():return
        if not self.dataset or not (self.dataset/'setp1/map_cloud.ply').is_file():
            self.dataset_status.set('请先完成采集并保存，或导入已有扫描数据集');return
        self.auto_active=True;self.auto_pending=True;self.auto_phase='preprocess';self.auto_cancel.clear();self.refresh_dataset_status()
        source=self.dataset/'setp1';target=self.dataset/'rgbd_scene/setp2'
        def worker():
            try:
                from preprocess import Parameters,load_input,_check,ProcessingCancelled
                from scene_cloud_processing import process_scene,save_scene
                _,loaded=load_input(source)
                params=Parameters(output_voxel_mm=1.,surface_spacing_mm=1.)
                result=process_scene(source,loaded.points,params,cancel=self.auto_cancel)
                _check(self.auto_cancel)
                save_scene(result,source,target);self.auto_events.put(('preprocessed',target))
            except ProcessingCancelled:self.auto_events.put(('cancelled',None))
            except Exception as exc:self.auto_events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start();self.auto_timer=self.root.after(50,self._poll_beta)

    def _poll_beta(self):
        self.auto_timer=None
        try:kind,value=self.auto_events.get_nowait()
        except queue.Empty:self.auto_timer=self.root.after(50,self._poll_beta);return
        self.auto_pending=False
        if kind=='preprocessed':self.stage_saved(2,value)
        if not self.auto_active:return
        if kind=='cancelled':self.stop_beta();return
        if kind=='error':
            self.stop_beta();self.dataset_status.set('自动处理失败：'+value);return
        self.auto_phase='load';self.open_scene_reconstruction(value)

    def _base_ready(self):
        if self.auto_active and self.auto_phase=='load':
            self.auto_phase='reconstruct';view=self.views['scene_reconstruction'];view.method.set('Poisson');view.start_base()

    def _base_completed(self):
        if self.auto_active:
            self.auto_active=False;self.auto_phase=None;self.refresh_dataset_status();self.open_review()

    def stop_beta(self):
        self.auto_active=False;self.auto_phase=None;self.auto_cancel.set()
        view=self.views.get('scene_reconstruction')
        if view and view.working:view.cancel_job()
        self.refresh_dataset_status()

    @staticmethod
    def _shutdown(view):
        if hasattr(view,'shutdown'):view.shutdown()
        else:
            view.closed=True
            if getattr(view,'timer',None):view.root.after_cancel(view.timer);view.timer=None

    def close(self):
        if self.closing:return
        if self.exporting:self.root.after(100,self.close);return
        self.closing=True;self.stop_beta()
        scan=self.views.get('acquisition')
        if scan and scan.busy:scan.finish()
        for view in self.views.values():
            if getattr(view,'working',None):
                if hasattr(view,'cancel_job'):view.cancel_job()
                else:view.cancel.set()
        self._wait_close()

    def _wait_close(self):
        if self.auto_pending or any(getattr(v,'working',None) or getattr(v,'busy',False) or getattr(v,'importing',False) for v in self.views.values()):
            self.root.after(80,self._wait_close);return
        for view in self.views.values():self._shutdown(view)
        if self.auto_timer:self.root.after_cancel(self.auto_timer);self.auto_timer=None
        self.root.destroy()


def launch(beta=False,initial_stage='acquisition'):
    from desktop_display import enable_dpi_awareness
    enable_dpi_awareness()
    root=tk.Tk();app=SceneWorkbench(root,beta=beta)
    if initial_stage!='acquisition':app.select_stage(initial_stage)
    root.mainloop()
