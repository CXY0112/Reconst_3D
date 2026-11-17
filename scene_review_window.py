"""Read-only point cloud / RGB and geometry / independent object showcase."""
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk
from scene_reconstruction_window import SceneReconstructionWindow
from rgb_prompt_panel import RgbPromptPanel
from rgbd_instance_segmentation import source_frames
from scene_stages import (read_report,bases_for,segmentation_for,base_model,object_models,ply_count,
                          preferred_kind,MESH_VIEW,SPLAT_VIEW)
from scene_reconstruction import read_mesh,load_inputs,digest
from brush_reconstruction import is_splats


class SceneReviewWindow(SceneReconstructionWindow):
    def __init__(self,root):
        super().__init__(root,kind='base')
        for name in ('method_box','base_button','next_button','show_base_button','review_button'):
            getattr(self,name).pack_forget()
        self.display_box.pack_forget()
        row=self.export_button.master
        self.left_mode=tk.StringVar(value='点云')
        self.left_select=ttk.Combobox(row,textvariable=self.left_mode,state='readonly',width=14,values=('点云','原始RGB图像'))
        self.left_select.pack(side='left',padx=6);self.left_select.bind('<<ComboboxSelected>>',lambda _:self.show_left())
        self.photo_index=tk.StringVar()
        self.photo_box=ttk.Combobox(row,textvariable=self.photo_index,state='readonly',width=14)
        self.photo_box.bind('<<ComboboxSelected>>',lambda _:self.show_photo())
        self.right_mode=tk.StringVar(value=MESH_VIEW)
        self.right_select=ttk.Combobox(row,textvariable=self.right_mode,state='readonly',width=16,values=(MESH_VIEW,'精细物体模型'))
        self.right_select.pack(side='left',padx=6);self.right_select.bind('<<ComboboxSelected>>',lambda _:self.show_right())
        self.model_choice=tk.StringVar()
        self.model_select=ttk.Combobox(row,textvariable=self.model_choice,state='readonly',width=24)
        self.model_select.bind('<<ComboboxSelected>>',lambda _:self.show_right())
        left=self.before.label.master;left.configure(text='场景来源 · 点云 / 原始 RGB')
        self.cloud=self.before;self.rgb_host=ttk.Frame(left)
        self.rgb=RgbPromptPanel(self.rgb_host,lambda _:None,lambda:False,instruction='原始 RGB 图像 · 滚轮切换照片')
        self.rgb.canvas.bind('<MouseWheel>',self.photo_wheel)
        self.rgb.canvas.bind('<Button-4>',lambda e:self.photo_wheel(e,1))
        self.rgb.canvas.bind('<Button-5>',lambda e:self.photo_wheel(e,-1))
        self.frames=[];self.final=None;self.models=[];self.metrics_values={}
        self.metrics=self.summary
        self.result_tabs.tab(self.model_frame,text=MESH_VIEW)
        self.status.set('左侧查看点云或原始 RGB，右侧切换几何结构、3DGS 三维效果或独立物体')
        self._buttons()

    def set_source(self,source,segmentation=None):
        if self.working:return
        dataset=self.dataset
        def load():
            scene,_=load_inputs(source,segmentation_for(dataset),self.cancel)
            bases={}
            for kind,target in bases_for(dataset).items():
                report=read_report(target/'report.json')
                if not report.get('complete') or report.get('source_fingerprint')!=scene.report['source_fingerprint']:continue
                path=base_model(target,report);mesh=read_mesh(path,self.cancel)[0] if not is_splats(report) else None
                bases[kind]=(target,report,mesh,None)
            if not bases:raise ValueError('尚无与当前点云对应的重建结果，请先完成第三阶段')
            try:frames,_=source_frames(scene)
            except (ValueError,OSError):frames=[]
            return scene,bases,frames,object_models(dataset,scene)
        self._run('review_load','正在读取场景几何、点云与原始照片…',load)

    def _poll(self):
        if self.closed:return
        try:
            kind,value=self.events.get_nowait()
            if kind=='review_load':
                self.working=None;self.scene,self.bases,self.frames,self.models=value
                active=getattr(self,'preferred_result',None)
                if active not in self.bases:active=preferred_kind(self.stage_root(),self.bases)
                self.base=self.bases[active]
                self.final=(base_model(self.base[0],self.base[1]),self.base[2],self.base[3])
                self.source=self.scene.source;self.before=self.cloud;self.before.clear()
                self.photo_box.configure(values=[f"采集 {f['capture']}" for f in self.frames])
                if self.frames:self.photo_box.current(0)
                else:self.photo_index.set('')
                self.left_select.configure(values=('点云','原始RGB图像') if self.frames else ('点云',))
                self.left_mode.set('点云')
                mode=SPLAT_VIEW if active=='gaussian' else MESH_VIEW
                modes=[label for kind,label in (('mesh',MESH_VIEW),('gaussian',SPLAT_VIEW)) if kind in self.bases]
                self.right_select.configure(values=(*modes,'精细物体模型'));self.right_mode.set(mode)
                self.model_select.configure(values=[m['name'] for m in self.models])
                if self.models:self.model_select.current(0)
                else:self.model_choice.set('暂无物体模型')
                self.show_left();self.show_right();self.update_metrics()
                self.status.set('成果已读取 · '+mode+'与精细物体独立展示')
                self._buttons()
            elif kind=='review_object':
                self.working=None;path,mesh,colors=value
                self._set_mesh(path,mesh,colors,independent=True)
                self.result_tabs.tab(self.model_frame,text='精细物体 · 独立模型')
                self.status.set('正在查看独立物体模型；模型没有放回或替换原场景')
                self._buttons()
            else:self.events.put((kind,value))
        except queue.Empty:pass
        super()._poll()

    def update_metrics(self):
        report=read_report(self.scene.source/'report.json')
        raw=ply_count(Path(self.dataset)/'setp1/map_cloud.ply') or report.get('input_points')
        count=len(self.scene.points_world);geo=self.base[1].get('geometry',{})
        coverage=geo.get('input_coverage',{});ten=coverage.get('10',coverage.get('10.0'))
        splats=is_splats(self.base[1]);mesh=self.base[2]
        self.metrics_values=dict(points=count,raw=raw,reduction=100*(1-count/max(raw,1)) if raw else None,
            vertices=None if splats else len(mesh[0]),faces=None if splats else len(mesh[1]),method='Brush 3DGS' if splats else self.base[1].get('method','poisson').upper(),
            gaussians=self.base[1].get('gaussians'),psnr=self.base[1].get('mean_psnr_db'),
            coverage=ten,p95=geo.get('input_to_mesh_mm_p95'),objects=len(self.models))
        m=self.metrics_values
        reduction=f"（减少 {m['reduction']:.1f}%）" if m['reduction'] is not None else ''
        original=f"原始 {raw:,} 点；" if raw else ''
        if splats:
            self.metrics.set(f'点云：{original}当前 {count:,} 点{reduction}。\n'
                f'Brush 3DGS：{m["gaussians"]:,} 个高斯 · 留出视角平均 PSNR {m["psnr"]:.2f} dB。\n'
                f'精细模型 {len(self.models)} 份 · 右侧直接显示 3DGS 三维效果')
            return
        self.metrics.set(f"点云：{original}当前 {count:,} 点{reduction}。\n"
            f"几何：{m['vertices']:,} 顶点 · {m['faces']:,} 三角面 · {m['method']}。\n"
            f"{self._coverage(geo)} · 精细模型 {len(self.models)} 份")

    @staticmethod
    def _coverage(geometry,color=None):
        parts=[];coverage=geometry.get('input_coverage',{});ten=coverage.get('10',coverage.get('10.0'))
        if ten is not None:parts.append(f'10mm内贴合覆盖 {ten:.1%}')
        if geometry.get('input_to_mesh_mm_p95') is not None:parts.append(f"点到网格P95 {geometry['input_to_mesh_mm_p95']:.2f}mm")
        return '；'.join(parts) or '未记录几何覆盖指标'

    def show_photo(self):
        i=self.photo_box.current()
        if 0<=i<len(self.frames):self.rgb.set_image(self.frames[i]['rgb'])

    def photo_wheel(self,event,direction=None):
        if not self.frames:return 'break'
        direction=direction if direction is not None else (1 if event.delta>0 else -1 if event.delta<0 else 0)
        self.photo_box.current(max(0,min(len(self.frames)-1,self.photo_box.current()-direction)));self.show_photo()
        return 'break'

    def show_left(self):
        if not self.scene:return
        for child in self.cloud.label.master.winfo_children():child.pack_forget()
        photo=self.left_mode.get()=='原始RGB图像'
        self.photo_box.pack_forget()
        if photo:
            self.photo_box.pack(side='left',after=self.left_select,padx=4)
            self.rgb_host.pack(fill='both',expand=True);self.show_photo()
        else:
            self.before=self.cloud
            # Restore the viewport and its two navigation/status captions.
            for child in self.cloud.label.master.winfo_children():
                if child is not self.rgb_host:child.pack(fill='both' if child is self.cloud.label else 'x',expand=child is self.cloud.label)
            self.cloud.set_cloud(self.scene.points,colors=self.scene.rgb)
        self.sync.set(not photo and self.right_mode.get()==MESH_VIEW)
        self._buttons()

    def show_right(self):
        if not self.base or self.working:return
        self.model_select.pack_forget()
        if self.right_mode.get()==SPLAT_VIEW:
            self.base=self.bases['gaussian']
            self.show_splats(self.base[0],self.base[1]);self._buttons()
            self.update_metrics()
        elif self.right_mode.get()==MESH_VIEW:
            self.base=self.bases['mesh'];self.final=(base_model(self.base[0],self.base[1]),self.base[2],self.base[3])
            self._set_mesh(*self.final);self.result_tabs.tab(self.model_frame,text=MESH_VIEW)
            self.sync.set(self.left_mode.get()=='点云')
            self.status.set('场景几何网格 · '+self.base[1].get('method','poisson').upper())
            self._buttons()
            self.update_metrics()
        else:
            self.sync.set(False)
            self.model_select.pack(side='left',after=self.right_select,padx=4)
            i=self.model_select.current()
            if not 0<=i<len(self.models):
                self.clear_result()
                self.status.set('当前数据集还没有已下载的精细物体模型');self._buttons();return
            record=self.models[i]
            def load():
                path=record['path'];expected=record['state'].get('sha256')
                if expected and digest(path)!=expected:raise ValueError('物体模型文件校验失败，请检查文件')
                mesh,colors=read_mesh(path,self.cancel);return path,mesh,colors
            self._run('review_object','正在读取独立物体模型…',load)

    def _buttons(self):
        super()._buttons()
        if not hasattr(self,'right_select'):return
        for box in (self.left_select,self.photo_box,self.right_select,self.model_select):
            box.configure(state='disabled' if self.working else 'readonly')
        scene_pair=self.left_mode.get()=='点云' and self.right_mode.get()==MESH_VIEW
        self.sync_check.configure(state='normal' if scene_pair and not self.working else 'disabled')
