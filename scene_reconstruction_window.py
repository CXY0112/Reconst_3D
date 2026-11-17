"""Stage-three geometry and stage-four explicitly submitted FAL objects."""
from pathlib import Path
import json,queue,shutil,threading
import tkinter as tk
from tkinter import ttk
from ui_dialogs import filedialog,messagebox
import numpy as np
from map_cloud_view import MapCloudPanel
from mesh_view import MeshPanel
from live_cloud_view import ViewCamera
from rgb_prompt_panel import RgbPromptPanel
from preprocess import ProcessingCancelled
from scene_reconstruction import load_inputs,base_reconstruction,read_mesh,segmentation_digest,result_path
from scene_segmentation import fingerprint
from fal_object_modeling import prepare_object,submit_object,fetch_object,read_state,validate_inputs
from brush_reconstruction import is_splats,LABEL as BRUSH_LABEL,METHOD as BRUSH_METHOD
from scene_stages import MESH_VIEW,SPLAT_VIEW,result_kind,result_target,preferred_kind


class SceneReconstructionWindow:
    def __init__(self,root,kind='base'):
        self.kind=kind
        self.fetch_timer=None;self.auto_fetch=False;self.fetch_failures=0
        self.root=root;self.dataset=None;self.scene=None;self.source=None;self.output=None
        self.working=None;self.closed=False;self.events=queue.Queue();self.cancel=threading.Event();self.latest_progress=None
        self.current_model=None;self.mesh=None;self.colors=None;self.run=None;self.run_state=None;self.env_file=None;self.base=None
        self.bases={}
        row=ttk.Frame(root,padding=6);row.pack(fill='x')
        self.method=tk.StringVar(value='Poisson')
        self.method_box=ttk.Combobox(row,textvariable=self.method,values=('Poisson','TSDF','NKSR',BRUSH_LABEL),state='readonly',width=14)
        self.method_box.pack(side='left')
        self.method_box.bind('<<ComboboxSelected>>',lambda _:self.method_changed())
        self.base_button=ttk.Button(row,text='生成几何模型',command=self.start_base);self.base_button.pack(side='left',padx=4)
        self.show_base_button=ttk.Button(row,text='查看基础模型',command=self.show_base);self.show_base_button.pack(side='left',padx=4)
        self.export_button=ttk.Button(row,text='导出当前模型',command=self.export);self.export_button.pack(side='left',padx=4)
        self.cancel_button=ttk.Button(row,text='停止等待 / 取消本地任务',command=self.cancel_job);self.cancel_button.pack(side='left',padx=4)
        self.next_button=ttk.Button(row,text='进入场景分割 / 精细建模',command=lambda:getattr(self,'on_refine',lambda:None)())
        self.next_button.pack(side='left',padx=4)
        self.review_button=ttk.Button(row,text='进入成果展示',command=lambda:getattr(self,'on_review',lambda:None)())
        self.review_button.pack(side='left',padx=4)
        self.colored=tk.BooleanVar(value=True)
        self.color_check=ttk.Checkbutton(row,text='显示颜色',variable=self.colored,command=self.redraw_mesh)
        if kind=='refine':self.color_check.pack(side='right')
        self.sync=tk.BooleanVar(value=True)
        self.sync_check=ttk.Checkbutton(row,text='同步视角',variable=self.sync)
        if kind=='base':self.sync_check.pack(side='right')
        row=ttk.Frame(root,padding=(6,0));row.pack(fill='x');self.fal_row=row
        self.object=tk.StringVar();self.object_box=ttk.Combobox(row,textvariable=self.object,state='readonly',width=28)
        self.object_box.pack(side='left');self.object_box.bind('<<ComboboxSelected>>',lambda _:self.choose_object())
        self.prepare_button=ttk.Button(row,text='准备建模',command=self.prepare);self.prepare_button.pack(side='left',padx=4)
        self.submit_button=ttk.Button(row,text='生成精细模型',command=self.submit);self.submit_button.pack(side='left',padx=4)
        self.fetch_button=ttk.Button(row,text='刷新 / 获取结果',command=self.fetch);self.fetch_button.pack(side='left',padx=4)
        self.config_button=ttk.Button(row,text='服务配置',command=self.configure_api);self.config_button.pack(side='left',padx=4)
        hint=ttk.Label(root,text='选择已分割的物体，生成精细三维模型。完成后可独立查看细节，并导出建模成果。',padding=6,wraplength=1400);hint.pack(fill='x')
        self.status=tk.StringVar(value='请导入数据集；精细建模使用第四阶段已保存的框选分割')
        ttk.Label(root,textvariable=self.status,padding=6,wraplength=1400).pack(fill='x')
        self.bar=ttk.Progressbar(root,maximum=100);self.bar.pack(fill='x',padx=8)
        panels=ttk.Panedwindow(root,orient='horizontal');panels.pack(fill='both',expand=True)
        left=ttk.Labelframe(panels,text='输入 · 完整场景 / 选中物体');right=ttk.Frame(panels)
        panels.add(left,weight=1);panels.add(right,weight=1);self.before=MapCloudPanel(left)
        if kind=='refine':left.configure(text='场景点云 · 选中物体高亮');self.sync.set(False)
        self.display_mode=tk.StringVar()
        self.display_box=ttk.Combobox(right,textvariable=self.display_mode,state='readonly',width=18)
        self.display_box.pack(anchor='w',padx=4,pady=3);self.display_box.bind('<<ComboboxSelected>>',lambda _:self.choose_result())
        self.result_tabs=ttk.Notebook(right);self.result_tabs.pack(fill='both',expand=True)
        self.model_frame=ttk.Frame(self.result_tabs);self.preview_frame=ttk.Frame(self.result_tabs)
        self.result_tabs.add(self.model_frame,text=MESH_VIEW);self.result_tabs.add(self.preview_frame,text='物体参考预览')
        self.after=MeshPanel(self.model_frame)
        self.splat_frame=ttk.Frame(self.result_tabs);self.splat_preview=None
        self.preview=RgbPromptPanel(self.preview_frame,lambda _:None,lambda:False,
            instruction='查看当前物体的参考图像与选中范围。')
        self.summary=tk.StringVar(value='场景几何与物体模型独立保存，可分别查看和导出。')
        ttk.Label(root,textvariable=self.summary,padding=6,wraplength=1400).pack(fill='x')
        self.last_views=None;self.timer=root.after(40,self._poll);self._buttons()
        if kind=='base':
            self.fal_row.pack_forget();hint.pack_forget();self.result_tabs.hide(self.preview_frame)
            self.show_base_button.pack_forget()
        else:
            self.display_box.pack_forget()
            self.method_box.pack_forget();self.base_button.pack_forget();self.next_button.pack_forget()
            self.fetch_button.pack_forget();self.show_base_button.pack_forget()
            self.result_tabs.tab(self.model_frame,text='三维模型 · 仅生成物体')

    def schedule_fetch(self,delay=3000):
        if self.fetch_timer:self.root.after_cancel(self.fetch_timer)
        if self.auto_fetch and not self.closed:self.fetch_timer=self.root.after(delay,self._auto_fetch)

    def _auto_fetch(self):
        self.fetch_timer=None
        if not self.auto_fetch or self.closed:return
        if self.working:self.schedule_fetch(1000)
        else:self.fetch()

    def stage_root(self):
        stage='setp3' if self.kind=='base' else 'setp4'
        if self.dataset:return Path(self.dataset)/'rgbd_scene'/stage
        if self.scene is None:raise ValueError('请先导入完整场景')
        source=self.scene.source
        if source.parent.name=='rgbd_scene':return source.parent/stage
        if source.name in ('setp1','setp2'):return source.parent/'rgbd_scene'/stage
        return source.parent/(stage+'_'+source.name)

    def method_changed(self):
        if self.kind!='base':return
        splats=self.method.get()==BRUSH_LABEL
        self.base_button.configure(text='训练 3DGS（8,000 步）' if splats else '生成几何模型')
        if self.working is None and splats:
            self.status.set('Brush 3DGS · 使用一阶段 RGB 图像与已保存位姿 · 8,000 步 · 最长边 320 px')
        kind='gaussian' if splats else 'mesh'
        if self.working is None and kind in self.bases:
            self.base=self.bases[kind];self.show_base();self._buttons()

    def choose_result(self):
        kind='gaussian' if self.display_mode.get()==SPLAT_VIEW else 'mesh'
        if self.working is None and kind in self.bases:
            self.base=self.bases[kind];self.show_base();self._buttons()

    def refresh_results(self):
        values=[label for kind,label in (('mesh',MESH_VIEW),('gaussian',SPLAT_VIEW)) if kind in self.bases]
        self.display_box.configure(values=values,state='readonly' if self.working is None and values else 'disabled')
        if self.base:self.display_mode.set(SPLAT_VIEW if is_splats(self.base[1]) else MESH_VIEW)
        else:self.display_mode.set('尚无已保存结果')

    def _buttons(self):
        idle=self.working is None;has=self.scene is not None;selected=bool(self.object.get())
        status=(self.run_state or {}).get('status')
        current=has and self.scene.instance is not None and (self.run_state or {}).get('segmentation_fingerprint')==segmentation_digest(self.scene)
        for button,enabled in ((self.base_button,idle and has),(self.show_base_button,idle and self.base is not None),
            (self.prepare_button,idle and has and selected),(self.submit_button,idle and self.run is not None and status=='prepared' and current),
            (self.fetch_button,idle and bool((self.run_state or {}).get('request_id') or status=='downloaded')),
            (self.export_button,idle and self.current_model is not None),(self.cancel_button,not idle or self.auto_fetch),(self.config_button,idle)):
            button.configure(state='normal' if enabled else 'disabled')
        for box in (self.method_box,self.object_box):box.configure(state='readonly' if idle else 'disabled')
        self.display_box.configure(state='readonly' if idle and self.bases else 'disabled')
        self.next_button.configure(state='normal' if idle and self.base else 'disabled')
        self.review_button.configure(state='normal' if idle and self.base else 'disabled')
        self.sync_check.configure(state='normal' if idle and self.mesh is not None else 'disabled')

    def _run(self,kind,text,fn):
        if self.working:return
        self.working=kind;self.job_kind=kind;self.cancel.clear();self.status.set(text);self.bar['value']=0;self._buttons()
        def worker():
            try:self.events.put((kind,fn()))
            except ProcessingCancelled:self.events.put(('cancelled',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def set_source(self,source,segmentation=None):
        if self.working:return
        self.auto_fetch=False
        if self.fetch_timer:self.root.after_cancel(self.fetch_timer);self.fetch_timer=None
        def load():
            scene,warning=load_inputs(source,segmentation,self.cancel);bases={}
            from scene_stages import bases_for,results_for_stage
            candidates=bases_for(self.dataset) if self.dataset else results_for_stage(self.stage_root()) if self.scene else {}
            override=getattr(self,'base_override',None)
            if not self.dataset and override:candidates[result_kind(json.loads((override/'report.json').read_text(encoding='utf-8')))]=override
            for kind,target in candidates.items():
                report=json.loads((target/'report.json').read_text(encoding='utf-8'))
                if report.get('complete') and report.get('source_fingerprint')==scene.report['source_fingerprint']:
                    from scene_stages import base_model
                    path=base_model(target,report)
                    mesh=read_mesh(path,self.cancel)[0] if not is_splats(report) else None
                    bases[kind]=(target,report,mesh,None)
                else:warning+=' 基础模型对应旧输入，需要重新生成。'
            return scene,warning,bases
        self._run('load','正在读取完整场景及已保存物体…',load)

    def start_base(self):
        if self.scene is None or self.working:return
        if not getattr(self,'can_write',lambda:True)():return
        scene=self.scene;method=BRUSH_METHOD if self.method.get()==BRUSH_LABEL else self.method.get().lower()
        target=result_target(self.stage_root(),'gaussian' if method==BRUSH_METHOD else 'mesh')
        options={} if not hasattr(self,'smoothing_iterations') else dict(smoothing_iterations=self.smoothing_iterations)
        self._run('reconstruct','正在训练 Brush 3DGS…' if method==BRUSH_METHOD else '正在生成场景几何模型…',
            lambda:base_reconstruction(scene,target,method,self.cancel,lambda s:setattr(self,'latest_progress',s),**options))

    def base_path(self):
        from scene_stages import base_for
        if self.dataset:return base_for(self.dataset)
        root=self.stage_root()
        return root/'base' if self.kind=='base' else root.parent/'setp3/base'

    def _set_mesh(self,path,mesh,colors,independent=False):
        self.result_tabs.configure(style='TNotebook')
        self.result_tabs.tab(self.model_frame,state='normal')
        if str(self.splat_frame) in self.result_tabs.tabs():self.result_tabs.hide(self.splat_frame)
        if independent:
            # glTF uses Y-up. Adapt only the viewer axes; original GLB bytes,
            # node transforms and unverified real-world scale are preserved.
            rotation=np.array([[0,0,-1],[-1,0,0],[0,1,0]],np.float32)
            mesh=(mesh[0]@rotation.T,mesh[1],mesh[2]@rotation.T)
        self.current_model=Path(path);self.mesh=mesh;self.colors=colors
        self.sync.set(not independent);self.after.clear();self.redraw_mesh();self.result_tabs.select(self.model_frame)
        if independent:
            # An elevated initial camera also reveals thin horizontal objects.
            low,high=self.after.points.min(axis=0),self.after.points.max(axis=0)
            center=(low+high)/2
            camera=ViewCamera(yaw=np.deg2rad(20),pitch=np.deg2rad(-25))
            aspect=max(self.after.label.winfo_width(),1)/max(self.after.label.winfo_height(),1)
            radius=max(float(np.linalg.norm(high-low))/2,.02)
            distance=radius/max(np.sin(np.arctan(np.tan(np.deg2rad(30))*min(aspect,1))),.1)*1.12
            camera.position=center-camera.basis()[:,2]*distance
            self.after.camera=camera;self.after.request_draw()
        else:self.copy_view(self.before,self.after)

    def clear_result(self):
        self.result_tabs.configure(style='TNotebook')
        self.after.clear();self.current_model=self.mesh=self.colors=None
        self.result_tabs.tab(self.model_frame,state='normal')
        if str(self.splat_frame) in self.result_tabs.tabs():self.result_tabs.hide(self.splat_frame)
        if self.splat_preview is not None:
            self.splat_preview.clear()

    def show_scene_cloud(self):
        if self.scene is None:return
        from scene_segmentation import region_colors
        ident=getattr(self,'objects',{}).get(self.object.get())
        self.before.set_cloud(self.scene.points,colors=region_colors(self.scene,selected=[ident] if ident is not None else [],assigned=False))

    def show_base(self):
        if self.base is None:return
        target,report,mesh,_=self.base
        if self.kind=='refine':self.show_scene_cloud();return
        self.before.clear();self.before.set_cloud(self.scene.points,colors=self.scene.rgb)
        self.display_mode.set(SPLAT_VIEW if is_splats(report) else MESH_VIEW)
        if is_splats(report):
            self.show_splats(target,report);return
        from scene_stages import geometry_model
        self._set_mesh(geometry_model(target,report),mesh,None)
        self.result_tabs.tab(self.model_frame,text=MESH_VIEW)
        self.status.set('场景几何模型 · '+report.get('method','poisson'))
        self.summary.set(f'{len(mesh[0]):,} 顶点 · {len(mesh[1]):,} 三角面 · 几何重建，不执行 RGB 着色')

    def show_splats(self,target,report):
        from scene_stages import base_model
        from brush_preview import BrushPreview
        self.current_model=base_model(target,report);self.mesh=self.colors=None;self.sync.set(False)
        self.after.clear()
        if self.splat_preview is None:self.splat_preview=BrushPreview(self.splat_frame)
        ttk.Style(self.root).layout('SceneOnly.TNotebook',[('Notebook.client',{'sticky':'nswe'})])
        ttk.Style(self.root).layout('SceneOnly.TNotebook.Tab',[])
        self.result_tabs.configure(style='SceneOnly.TNotebook')
        self.result_tabs.add(self.splat_frame,text=SPLAT_VIEW)
        self.result_tabs.hide(self.model_frame);self.result_tabs.select(self.splat_frame)
        self.splat_preview.set_result(target,report)
        self.status.set('3DGS 三维效果 · 在右侧视口直接旋转、移动与缩放 · 可导出高斯 PLY')
        self.summary.set(f'{report["gaussians"]:,} 个高斯 · {report["parameters"]["steps"]:,} 步 · '
                         f'留出视角平均 PSNR {report["mean_psnr_db"]:.2f} dB · 输入位姿未细化')

    def redraw_mesh(self):
        if self.mesh is not None:self.after.set_mesh(*self.mesh,colors=self.colors if self.colored.get() else None)

    def choose_object(self,update_cloud=True):
        if not self.scene or self.working:return
        self.run=None;self.run_state=None
        self.auto_fetch=False
        if self.fetch_timer:self.root.after_cancel(self.fetch_timer);self.fetch_timer=None
        ident=self.objects.get(self.object.get())
        if ident is None:self._buttons();return
        selected=self.scene.instance==ident
        if self.kind=='refine':self.show_scene_cloud()
        if update_cloud:
            if self.kind=='refine':self.show_base()
            else:self.before.clear();self.before.set_cloud(self.scene.points[selected],colors=self.scene.rgb[selected])
            self.after.clear();self.mesh=self.colors=self.current_model=None
            self.preview.set_image(None)
        latest=self.stage_root()/'fal'/f'object_{ident:06d}'/'latest.json'
        if latest.is_file():
            try:
                name=json.loads(latest.read_text(encoding='utf-8'))['run'];candidate=(latest.parent/name).resolve()
                if candidate.parent!=latest.parent.resolve():raise ValueError('无效任务目录')
                state=read_state(candidate)
                if state.get('schema')=='fal_object_run_v1':
                    self.run=candidate;self.run_state=state;self.preview.set_image(candidate/'input_preview.png')
                    note='将自动获取结果' if state.get('segmentation_fingerprint')==segmentation_digest(self.scene) else '对应旧分割，可取回已有任务；新建模需重新准备输入'
                    self.status.set(f"物体 #{ident} · 已有建模任务：{state['status']}；{note}")
            except (ValueError,OSError,KeyError):pass
        self._buttons()
        if self.run_state and self.run_state.get('status') in ('submitted','downloaded'):
            self.auto_fetch=True;self.schedule_fetch(100)

    def prepare(self):
        if self.working or self.scene is None:return
        if not getattr(self,'can_write',lambda:True)():return
        ident=self.objects.get(self.object.get())
        if ident is None:return
        self.auto_fetch=False
        if self.fetch_timer:self.root.after_cancel(self.fetch_timer);self.fetch_timer=None
        scene=self.scene;target=self.stage_root()/'fal'
        self._run('prepare','正在准备所选物体的建模资料…',
            lambda:prepare_object(scene,ident,target,self.cancel))

    def configure_api(self):
        path=filedialog.askopenfilename(parent=self.root,title='选择建模服务配置文件',filetypes=[('配置文件','*.env'),('所有文件','*')])
        if path:self.env_file=Path(path);self.status.set('建模服务配置已选择')

    def submit(self):
        if self.working or self.run is None:return
        if not getattr(self,'can_write',lambda:True)():return
        run=self.run;scene=self.scene;env_file=self.env_file
        self._run('submit','正在提交精细建模任务，请稍候…',
            lambda:submit_object(run,scene,env_file,self.cancel))

    def fetch(self):
        if self.working or self.run is None:return
        run=self.run;env_file=self.env_file
        def fetch():
            state=fetch_object(run,env_file,self.cancel);result=None
            if state.get('status')=='downloaded':
                mesh,colors=read_mesh(run/'generated.glb',self.cancel);result=(mesh,colors)
            return state,result
        self._run('fetch','正在获取物体建模结果…',fetch)

    def export(self):
        if self.current_model is None or self.working:return
        suffix=self.current_model.suffix
        name=filedialog.asksaveasfilename(parent=self.root,title='导出当前模型',defaultextension=suffix,
            initialfile=self.current_model.name,filetypes=[('模型文件','*'+suffix)])
        if name:
            destination=Path(name)
            if destination.suffix.lower()!=suffix.lower():destination=destination.with_suffix(suffix)
            if destination.resolve()!=self.current_model.resolve():shutil.copy2(self.current_model,destination)
            self.status.set(f'已导出：{destination}')

    def copy_view(self,source,target):
        c=source.camera;target.camera=ViewCamera(c.position,c.yaw,c.pitch,c.roll);target.fitted=True;target.request_draw()
        self.last_views=self.view_states()

    def view_states(self):return [tuple(p.camera.position)+(p.camera.yaw,p.camera.pitch,p.camera.roll) for p in (self.before,self.after)]

    def _poll(self):
        if self.closed:return
        if self.latest_progress:
            progress=self.latest_progress;self.latest_progress=None
            if isinstance(progress,dict):self.status.set(progress['text']);self.bar['value']=progress['percent']
            else:self.status.set(progress)
        try:
            kind,value=self.events.get_nowait();self.working=None
            if kind=='load':
                self.scene,warning,self.bases=value;self.source=self.scene.source;self.run=self.run_state=None
                active=preferred_kind(self.stage_root(),self.bases)
                self.base=self.bases.get(active);self.refresh_results()
                if self.base:
                    self.method.set(BRUSH_LABEL if active=='gaussian' else {'poisson':'Poisson','tsdf':'TSDF','nksr':'NKSR'}.get(self.base[1].get('method'),'Poisson'))
                    self.base_button.configure(text='训练 3DGS（8,000 步）' if active=='gaussian' else '生成几何模型')
                self.before.clear()
                if self.kind!='refine':self.before.set_cloud(self.scene.points,colors=self.scene.rgb)
                self.clear_result()
                self.objects={f"#{o['id']} · {o['name']} · {o['points']:,} 点":o['id'] for o in self.scene.objects()}
                self.object_box.configure(values=list(self.objects));self.object.set(next(iter(self.objects),'') )
                self.status.set(warning or ('已读取完整点云；选择方法后生成场景几何模型' if self.kind=='base' else '已载入分割物体，请选择对象开始精细建模'))
                if self.objects:self.choose_object(update_cloud=False)
                elif self.kind=='refine':self.show_scene_cloud()
                if self.base:self.show_base()
            elif kind=='reconstruct':
                self.base=value;self.bases[result_kind(value[1])]=value;self.output=value[0];self.refresh_results()
                self.show_base();self.bar['value']=100;self._saved()
                if getattr(self,'on_complete',None):self.on_complete()
            elif kind=='prepare':
                self.run,self.run_state=value;self.preview.set_image(self.run/'input_preview.png');self.result_tabs.select(self.preview_frame)
                self.status.set("物体资料已准备，可查看参考预览并生成精细模型")
            elif kind=='submit':
                self.run_state=value;self.auto_fetch=True;self.fetch_failures=0;self.schedule_fetch(500)
                self.status.set("精细建模已开始，完成后将自动显示结果")
            elif kind=='fetch':
                self.run_state,result=value;self.fetch_failures=0
                if result:
                    self.auto_fetch=False
                    self._set_mesh(self.run/'generated.glb',*result,independent=True);self.bar['value']=100;self._saved()
                    self.status.set('精细模型已就绪，可旋转查看细节或导出成果')
                    self.summary.set(f'{len(result[0][0]):,} 顶点 · {len(result[0][1]):,} 三角面 · 可独立查看与导出')
                else:
                    self.status.set('建模状态：'+value[0].get('queue_status','未知')+'；自动等待结果')
                    if value[0].get('queue_status') in ('FAILED','CANCELLED'):self.auto_fetch=False
                    self.schedule_fetch()
            elif kind=='cancelled':
                self.latest_progress=None;self.bar['value']=0
                self.status.set('已取消本地重建，原有结果已保留' if self.job_kind=='reconstruct' else '已停止等待，建模任务记录已保留')
            elif kind=='error':
                self.latest_progress=None
                if self.job_kind=='reconstruct':self.bar['value']=0
                if self.run and (self.run/'run_state.json').is_file():self.run_state=read_state(self.run)
                self.status.set('操作失败，已保存结果保留'+('；请先确认已有任务状态，再尝试提交' if self.job_kind in ('submit','fetch') else ''))
                if self.job_kind=='fetch' and self.auto_fetch:
                    self.fetch_failures+=1
                    if self.fetch_failures<=5:
                        self.status.set('查询暂时失败，将自动重试已有任务');self.schedule_fetch(min(30000,3000*2**self.fetch_failures))
                    else:self.auto_fetch=False;messagebox.showerror('自动查询已暂停',value,parent=self.root)
                else:messagebox.showerror('几何重建 / 物体建模',value,parent=self.root)
                if getattr(self,'on_failed',None):self.on_failed()
            self._buttons()
            if kind=='load' and getattr(self,'on_ready',None):self.on_ready()
        except queue.Empty:pass
        states=self.view_states()
        if self.sync.get() and self.last_views:
            changed=[states[i]!=self.last_views[i] for i in (0,1)]
            if any(changed):self.copy_view(*( (self.after,self.before) if changed[1] and not changed[0] else (self.before,self.after)))
        self.last_views=self.view_states();self.timer=self.root.after(40,self._poll)

    def _saved(self):
        if self.kind=='base' and self.base:
            from scene_reconstruction import write_json
            write_json(self.stage_root()/'active_result.json',dict(kind=result_kind(self.base[1])))
        if getattr(self,'on_stage_saved',None):
            try:self.on_stage_saved(3 if self.kind=='base' else 4,self.stage_root())
            except Exception as exc:
                self.status.set('模型已保存，但数据集索引更新失败')
                messagebox.showerror('数据集索引',str(exc),parent=self.root)

    def cancel_job(self):
        self.auto_fetch=False
        if self.fetch_timer:self.root.after_cancel(self.fetch_timer);self.fetch_timer=None
        self.cancel.set();self.status.set('正在取消本地重建…' if self.working=='reconstruct' else '正在停止等待，已提交任务的记录将保留')

    def shutdown(self):
        self.cancel.set();self.closed=True
        if self.splat_preview is not None:self.splat_preview.shutdown()
        for timer in (self.fetch_timer,):
            if timer:self.root.after_cancel(timer)
        if self.timer:self.root.after_cancel(self.timer);self.timer=None
