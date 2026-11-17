"""Fourth RGB-D stage: prompted RGB-box instance segmentation only."""
from pathlib import Path
import copy,queue, threading
import tkinter as tk
from tkinter import ttk
from ui_dialogs import filedialog,messagebox,simpledialog
import numpy as np
from map_cloud_view import MapCloudPanel
from live_cloud_view import ViewCamera
from preprocess import ProcessingCancelled
from scene_segmentation import (load_scene, load_segmentation,
    save_segmentation, instance_colors, export_object)
from rgbd_instance_segmentation import MobileSamClient,source_frames,prompt_rgbd,apply_prompt
from rgb_prompt_panel import RgbPromptPanel


class SegmentationWindow:
    def __init__(self,root):
        self.root=root;self.dataset=None;self.source=None;self.output=None;self.scene=None
        self.working=None;self.closed=False;self.events=queue.Queue();self.cancel=threading.Event()
        self.latest_progress=None;self.history=[];self.selected=[];self.dirty=False
        self.sam=MobileSamClient();self.frames=[];self.intrinsics=None;self.pending_prompt=None
        controls=ttk.Frame(root,padding=8);controls.pack(fill='x')
        self.open_button=ttk.Button(controls,text='导入扫描 / 预处理结果',command=self.open_source);self.open_button.pack(side='left',padx=3)
        self.reset_button=ttk.Button(controls,text='清空分割',command=self.reset_labels);self.reset_button.pack(side='left',padx=3)
        self.cancel_button=ttk.Button(controls,text='取消',command=self.cancel_job);self.cancel_button.pack(side='left',padx=3)
        self.save_button=ttk.Button(controls,text='保存 / 覆盖本阶段',command=self.save);self.save_button.pack(side='left',padx=3)
        self.export_button=ttk.Button(controls,text='导出选中物体 PLY',command=self.export);self.export_button.pack(side='left',padx=3)
        self.next_button=ttk.Button(controls,text='保存分割并精细建模',command=self.next_stage);self.next_button.pack(side='left',padx=3)
        self.next_after_save=False
        self.mode=tk.StringVar(value='分割区域颜色')
        box=ttk.Combobox(controls,textvariable=self.mode,values=('分割区域颜色','原始 RGB'),state='readonly',width=16)
        box.pack(side='right');box.bind('<<ComboboxSelected>>',lambda _:self.show_scene())
        self.status=tk.StringVar(value='导入完整点云后，在右上原始 RGB 照片中拖框进行提示分割。')
        ttk.Label(root,textvariable=self.status,padding=(8,2),wraplength=1400).pack(fill='x')
        self.bar=ttk.Progressbar(root,maximum=100);self.bar.pack(fill='x',padx=10)
        objects=ttk.Frame(root,padding=(8,4));objects.pack(fill='x')
        self.tree=ttk.Treeview(objects,columns=('name','points'),show='tree headings',height=4,selectmode='extended')
        self.tree.heading('#0',text='物体编号');self.tree.column('#0',width=90,stretch=False)
        for key,title,width in (('name','名称',180),('points','点数',110)):
            self.tree.heading(key,text=title);self.tree.column(key,width=width,stretch=True)
        self.tree.pack(side='left',fill='x',expand=True)
        scroll=ttk.Scrollbar(objects,orient='vertical',command=self.tree.yview);scroll.pack(side='left',fill='y')
        self.tree.configure(yscrollcommand=scroll.set);self.tree.bind('<<TreeviewSelect>>',self.select)
        editing=ttk.Frame(objects);editing.pack(side='left',fill='y',padx=8)
        self.merge_button=ttk.Button(editing,text='合并选中区域',command=self.merge);self.merge_button.pack(fill='x',pady=2)
        self.undo_button=ttk.Button(editing,text='撤销修改',command=self.undo);self.undo_button.pack(fill='x',pady=2)
        panels=ttk.Panedwindow(root,orient='horizontal');panels.pack(fill='both',expand=True)
        left=ttk.Labelframe(panels,text='场景点云 · 物体区域高亮');right=ttk.Panedwindow(panels,orient='vertical')
        panels.add(left,weight=1);panels.add(right,weight=1)
        photos=ttk.Labelframe(right,text='原始 RGB · 框选提示分割');objects_view=ttk.Labelframe(right,text='选中物体 · 原始 RGB 点云')
        right.add(photos,weight=1);right.add(objects_view,weight=2)
        row=ttk.Frame(photos);row.pack(fill='x')
        self.photo_index=tk.StringVar()
        self.photo_box=ttk.Combobox(row,textvariable=self.photo_index,state='readonly',width=22)
        self.photo_box.pack(side='left',padx=3);self.photo_box.bind('<<ComboboxSelected>>',lambda _:self.change_photo())
        self.prompt_action=tk.StringVar(value='单独分出物体')
        self.action_box=ttk.Combobox(row,textvariable=self.prompt_action,state='readonly',width=18,
            values=('单独分出物体','补入选中物体','从选中物体移除'))
        self.action_box.pack(side='left',padx=3)
        self.apply_button=ttk.Button(row,text='应用框选结果',command=self.apply_box);self.apply_button.pack(side='left',padx=3)
        self.rgb_panel=RgbPromptPanel(photos,self.request_box,lambda:self.working is None and bool(self.frames),self.clear_prompt)
        self.rgb_panel.canvas.bind('<MouseWheel>',self.photo_wheel)
        self.rgb_panel.canvas.bind('<Button-4>',lambda e:self.photo_wheel(e,1))
        self.rgb_panel.canvas.bind('<Button-5>',lambda e:self.photo_wheel(e,-1))
        self.before=MapCloudPanel(left);self.after=MapCloudPanel(objects_view)
        self.last_views=None;self.sync_timer=root.after(16,self._sync_views)
        self.before.label.bind('<Control-Button-1>',self.pick)
        self.summary=tk.StringVar(value='选择 RGB 照片并拖框 → 预览 → 应用；换视角后可补入同一物体。Ctrl＋左键选择已有物体。')
        ttk.Label(root,textvariable=self.summary,padding=8,wraplength=1400).pack(fill='x')
        self.timer=root.after(50,self._poll);self._buttons()

    def _buttons(self):
        idle=self.working is None;has=self.scene is not None;segmented=has and self.scene.instance is not None
        for button,enabled in ((self.open_button,idle),(self.reset_button,idle and segmented),
            (self.cancel_button,not idle and self.working!='save'),(self.save_button,idle and segmented),
            (self.export_button,idle and segmented and len(self.selected)==1),
            (self.merge_button,idle and segmented and len(self.selected)>1),
            (self.undo_button,idle and bool(self.history)),
            (self.apply_button,idle and self.pending_prompt is not None)):
            button.configure(state='normal' if enabled else 'disabled')
        self.tree.configure(selectmode='extended' if idle else 'none')
        for box in (self.photo_box,self.action_box):box.configure(state='readonly' if idle else 'disabled')
        self.next_button.configure(state='normal' if idle and segmented and getattr(self,'on_reconstruct',None) else 'disabled')

    def next_stage(self):
        if self.working or not getattr(self,'on_reconstruct',None):return
        self.next_after_save=True;self.save()
        if self.working!='save':self.next_after_save=False

    def _run(self,kind,text,fn):
        if self.working:return
        self.working=kind;self.cancel.clear();self.latest_progress=None;self.status.set(text);self.bar['value']=0;self._buttons()
        def worker():
            try:self.events.put((kind,fn()))
            except ProcessingCancelled:self.events.put(('cancelled',None))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def open_source(self):
        path=filedialog.askdirectory(parent=self.root,title='选择数据集、一阶段扫描或二阶段目录')
        if path:
            from dataset_store import route_dataset
            if route_dataset(self,path):return
            self.set_source(path)

    def set_source(self,source):
        if self.working:return
        self._run('load','正在读取完整彩色场景…',lambda:load_scene(source,cancel=self.cancel,
            progress=lambda f,t:setattr(self,'latest_progress',(f,t))))

    def load_result(self,folder,source=None):
        self._run('restore','正在恢复分割标签并校验来源数据…',lambda:(load_segmentation(folder,source,
            progress=lambda f,t:setattr(self,'latest_progress',(f,t)),cancel=self.cancel),Path(folder)))

    def show_scene(self):
        if self.scene is None:return
        from scene_segmentation import region_colors
        pending=self.pending_prompt['indices'] if self.pending_prompt is not None else None
        colors=self.scene.rgb if self.mode.get()=='原始 RGB' else region_colors(self.scene,self.selected,pending)
        self.before.set_cloud(self.scene.points,colors=colors)

    def _load_photos(self):
        self.pending_prompt=None;self.frames=[];self.photo_error=None
        try:self.frames,self.intrinsics=source_frames(self.scene)
        except (OSError,ValueError,KeyError) as exc:self.photo_error=str(exc)
        values=[f"照片 {i+1} · 采集 {f['capture']}" for i,f in enumerate(self.frames)]
        self.photo_box.configure(values=values)
        self.photo_index.set(values[0] if values else '无关联 RGB-D')
        self.change_photo()

    def change_photo(self):
        self.pending_prompt=None
        index=self.photo_box.current()
        self.rgb_panel.set_image(self.frames[index]['rgb'] if 0<=index<len(self.frames) else None)
        self.show_scene();self._buttons()

    def photo_wheel(self,event,direction=None):
        if self.working or not self.frames:return 'break'
        direction=direction if direction is not None else (1 if event.delta>0 else -1 if event.delta<0 else 0)
        index=max(0,self.photo_box.current())
        self.photo_box.current(max(0,min(len(self.frames)-1,index-direction)));self.change_photo()
        return 'break'

    def clear_prompt(self):
        self.pending_prompt=None;self.show_scene();self._buttons()

    def request_box(self,box):
        if self.working or not self.frames or self.scene is None:return
        index=self.photo_box.current()
        if index<0:return
        self.pending_prompt=None
        scene=self.scene;frame=self.frames[index]
        self._run('prompt','正在计算框选掩码与深度一致的三维点…',
            lambda:prompt_rgbd(scene,self.sam,frame,self.intrinsics,box,self.cancel))

    def _remember(self,indices):
        indices=np.unique(indices)
        self.history.append((indices,None if self.scene.instance is None else self.scene.instance[indices].copy(),
            None if self.scene.semantic is None else self.scene.semantic[indices].copy(),copy.deepcopy(self.scene.report)))
        def size(item):return sum(value.nbytes for value in item[:3] if isinstance(value,np.ndarray))
        while len(self.history)>1 and (len(self.history)>12 or sum(map(size,self.history))>64*1024**2):self.history.pop(0)

    def reset_labels(self):
        if self.working or self.scene is None or self.scene.instance is None:return
        if not getattr(self,'can_write',lambda:True)():return
        self._remember(np.arange(len(self.scene.rgb)))
        self.scene.instance=np.zeros(len(self.scene.rgb),np.int32)
        self.scene.semantic=np.full(len(self.scene.rgb),19,np.int32)
        self.scene.report={key:self.scene.report[key] for key in ('source_fingerprint','input_kind','preprocessing_skipped') if key in self.scene.report}
        self.scene.report.update(backend='rgbd_box_only',unassigned_instance_id=0,complete=True,prompt_edits=[])
        self.pending_prompt=None;self.rgb_panel.mask=None;self.rgb_panel.box=None;self.rgb_panel.draw();self.dirty=True
        self.show_scene();self._objects();self._buttons();self.status.set('已清空实例标签，可重新拖框；原始点云保留，支持撤销，尚未保存')

    def apply_box(self):
        if self.working or self.pending_prompt is None:return
        if not getattr(self,'can_write',lambda:True)():return
        actions={'单独分出物体':'new','补入选中物体':'add','从选中物体移除':'remove'}
        action=actions[self.prompt_action.get()]
        target=self.selected[0] if len(self.selected)==1 else None
        try:result,ident=apply_prompt(self.scene,self.pending_prompt,action,target)
        except ValueError as exc:messagebox.showerror('框选优化',str(exc),parent=self.root);return
        name=None
        if action!='remove':
            name=simpledialog.askstring('物体名称','为此物体设置名称：',initialvalue=result.report.get('object_names',{}).get(str(ident),f'物体 {ident}'),parent=self.root)
        self._remember(self.pending_prompt['indices']);self.scene=result;self.pending_prompt=None;self.dirty=True
        if name and name.strip():self.scene.report.setdefault('object_names',{})[str(ident)]=name.strip()
        self.show_scene();self._objects()
        if str(ident) in self.tree.get_children():self.tree.selection_set(str(ident));self.select()
        self._buttons();self.status.set('已应用框选结果，尚未保存；支持撤销')

    def _view_states(self):
        return [tuple(p.camera.position)+(p.camera.yaw,p.camera.pitch,p.camera.roll) for p in (self.before,self.after)]

    def _copy_view(self,source,target):
        camera=source.camera
        target.camera=ViewCamera(camera.position,camera.yaw,camera.pitch,camera.roll)
        target.fitted=True;target.request_draw()
        self.last_views=self._view_states()

    def _sync_views(self):
        if self.closed:return
        states=self._view_states()
        if self.last_views is not None:
            changed=[states[i]!=self.last_views[i] for i in (0,1)]
            if any(changed):
                index=1 if changed[1] and (not changed[0] or self.after.label.focus_get() is self.after.label) else 0
                self._copy_view(*( (self.after,self.before) if index else (self.before,self.after)))
                states=self._view_states()
        self.last_views=states
        self.sync_timer=self.root.after(16,self._sync_views)

    def _objects(self):
        self.tree.delete(*self.tree.get_children());self.selected=[]
        if self.scene is None or self.scene.instance is None:
            self.after.clear();self._copy_view(self.before,self.after)
            if self.scene is not None:self.summary.set(f'{len(self.scene.rgb):,} 点 · 尚未创建物体；在右上原始 RGB 中拖框并应用')
            return
        objects=self.scene.objects()
        for obj in objects:
            name=self.scene.report.get('object_names',{}).get(str(obj['id']),obj['name'])
            self.tree.insert('', 'end',iid=str(obj['id']),text=f"#{obj['id']}",
                values=(name,f"{obj['points']:,}"))
        if objects:self.tree.selection_set(str(objects[0]['id']))
        source_name='一阶段原始点云' if self.scene.report.get('input_kind')=='stage1' else '二阶段预处理点云'
        unknown=int((self.scene.instance==self.scene.report.get('unassigned_instance_id',-1)).sum())
        self.summary.set(f"{source_name} · {len(self.scene.rgb):,} 点 · {len(objects)} 个区域 · 未分配 {unknown:,} 点（灰色） · "
            f"{self.scene.report.get('device','未知设备')} · {self.scene.report.get('elapsed_seconds',0):.1f}s · "
            'RGB 拖框创建或调整物体；换视角后可补入同一物体；支持合并与撤销。')

    def select(self,event=None):
        self.selected=[int(i) for i in self.tree.selection()];self.show_scene()
        if self.scene is not None and self.scene.instance is not None and self.selected:
            selected=np.isin(self.scene.instance,self.selected)
            # Keep world coordinates and the shared camera; changing selection
            # must not fit the object into a new independent coordinate/view.
            self.after.fitted=True
            self.after.set_cloud(self.scene.points[selected],colors=self.scene.rgb[selected])
            self._copy_view(self.before,self.after)
        else:
            self.after.clear();self._copy_view(self.before,self.after)
        self._buttons()

    def pick(self,event):
        if self.working or self.scene is None or self.scene.instance is None:return 'break'
        width=max(self.before.label.winfo_width(),1);height=max(self.before.label.winfo_height(),1)
        p=self.before.camera.project_coordinates(self.scene.points);valid=p[:,2]>.02
        ids=np.flatnonzero(valid);q=p[valid]
        focal=height/(2*np.tan(np.deg2rad(30)))
        uv=q[:,:2]/q[:,2,None]*focal+[width/2,height/2]
        distance=np.sum((uv-[event.x,event.y])**2,axis=1);near=np.flatnonzero(distance<8**2)
        if len(near):
            # Choose visible frontmost points in the small cursor neighborhood.
            ix=near[np.argmin(q[near,2])];ident=int(self.scene.instance[ids[ix]])
            if str(ident) not in self.tree.get_children():return 'break'
            self.tree.selection_set(str(ident));self.tree.see(str(ident));self.select()
        return 'break'

    def merge(self):
        if self.working or len(self.selected)<2:return
        ids=self.selected.copy();mask=np.isin(self.scene.instance,ids);indices=np.flatnonzero(mask)
        self._remember(indices)
        self.scene.instance[mask]=min(ids);self.dirty=True
        self.show_scene();self._objects();self.status.set('区域已合并，尚未保存；可以撤销')

    def undo(self):
        if self.working or not self.history:return
        indices,labels,semantic,report=self.history.pop()
        if labels is None:self.scene.instance=None;self.scene.semantic=None
        else:self.scene.instance[indices]=labels;self.scene.semantic[indices]=semantic
        self.scene.report=report;self.dirty=True
        self.pending_prompt=None
        self.show_scene();self._objects();self._buttons();self.status.set('已撤销修改，尚未保存')

    def save(self,destination=None):
        if self.working or self.scene is None or self.scene.instance is None:return
        if not getattr(self,'can_write',lambda:True)():return
        if destination is None and self.dataset:destination=Path(self.dataset)/'rgbd_scene/setp4/segmentation'
        if destination is None:
            parent=filedialog.askdirectory(parent=self.root,title='选择保存父目录')
            if not parent:return
            destination=Path(parent)/'rgbd_scene_setp4/segmentation'
        scene=self.scene
        self._run('save','正在保存分割标签及物体清单…',lambda:save_segmentation(scene,destination))

    def export(self):
        if self.working or len(self.selected)!=1:return
        path=filedialog.asksaveasfilename(parent=self.root,title='导出物体彩色点云',defaultextension='.ply',
            initialfile=f'object_{self.selected[0]}.ply',filetypes=[('PLY点云','*.ply')])
        if not path:return
        scene,ident=self.scene,self.selected[0]
        self._run('export','正在导出物体…',lambda:(Path(path),export_object(scene,ident,path)))

    def _poll(self):
        if self.closed:return
        if self.latest_progress:
            f,text=self.latest_progress;self.latest_progress=None;self.bar['value']=f*100;self.status.set(text)
        try:
            kind,value=self.events.get_nowait();self.working=None
            if kind in ('load','restore'):
                self.scene,self.output=value if kind=='restore' else (value,None)
                self.source=self.scene.source;self.history=[];self.dirty=False
                self.before.clear();self.show_scene();self._copy_view(self.before,self.after);self._objects();self.bar['value']=100
                self._load_photos()
                source_name='一阶段原始融合点云（跳过二阶段）' if self.scene.report.get('input_kind')=='stage1' else '二阶段预处理点云'
                self.status.set(('已读取完整彩色点云，请在右上 RGB 照片中拖框' if kind=='load' else f'已恢复：{self.output}')+' · 来源：'+source_name)
            elif kind=='prompt':
                self.pending_prompt=value;self.rgb_panel.preview(value);self.show_scene();self.bar['value']=100
                indices=value['indices'];self.after.fitted=True
                self.after.set_cloud(self.scene.points[indices],colors=self.scene.rgb[indices]);self._copy_view(self.before,self.after)
                self.status.set(f"框选预览：{len(value['indices']):,} 个深度一致点；选择修改方式后点击“应用框选结果”")
            elif kind=='save':
                self.output=Path(value);self.dirty=False;self.status.set(f'已保存：{value}')
                if getattr(self,'on_stage_saved',None):
                    try:self.on_stage_saved(4,self.output.parent)
                    except Exception as exc:
                        self.status.set(f'分割文件已保存；数据集索引更新失败：{exc}')
                        messagebox.showerror('数据集索引',str(exc),parent=self.root)
                if self.next_after_save and getattr(self,'on_reconstruct',None):
                    self.next_after_save=False;self.on_reconstruct(self.output)
            elif kind=='export':self.status.set(f'已导出 {value[1]:,} 点：{value[0]}')
            elif kind=='cancelled':self.status.set('已取消 · 原点云和上次分割结果保留')
            elif kind=='error':
                self.next_after_save=False
                self.status.set('操作失败 · 原数据和上次分割结果保留')
                messagebox.showerror('点云分割',value,parent=self.root)
            self._buttons()
        except queue.Empty:pass
        self.timer=self.root.after(50,self._poll)

    def cancel_job(self):
        if self.working:self.cancel.set();self.status.set('正在停止分割操作…')

    def shutdown(self):
        self.cancel.set();self.closed=True
        self.sam.close()
        if self.timer:self.root.after_cancel(self.timer);self.timer=None
        if self.sync_timer:self.root.after_cancel(self.sync_timer);self.sync_timer=None

