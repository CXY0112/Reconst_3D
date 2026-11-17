"""Shared desktop art direction. No changes to reconstruction or stored colors."""
import math
import tkinter as tk
from tkinter import ttk, font

BG='#101214'
SURFACE='#191d20'
VIEW='#0b1013'
LINE='#30383d'
TEXT='#edf2ee'
MUTED='#a0afaf'
ACCENT='#b8eed5'


def install(root):
    """Main and Beta each install the shared theme in their own Tk interpreter."""
    root.beta_visual=True
    root.viewport_background=(11,16,19)
    root.viewport_focus=(184/255,238/255,213/255,1.)
    root.viewport_border=(48/255,56/255,61/255,1.)
    root.configure(background=BG)
    families=set(font.families(root))
    body=next((f for f in ('Microsoft YaHei UI','Microsoft YaHei','Noto Sans CJK SC') if f in families),'TkDefaultFont')
    display='Geist' if 'Geist' in families else 'Segoe UI'
    style=ttk.Style(root)
    if 'spatial-beta' not in style.theme_names():style.theme_create('spatial-beta',parent='clam')
    style.theme_use('spatial-beta')
    style.configure('.',background=BG,foreground=TEXT,font=(body,10),bordercolor=LINE,
                    lightcolor=LINE,darkcolor=LINE,troughcolor=VIEW,selectbackground='#304d46',selectforeground=TEXT)
    style.configure('TFrame',background=BG)
    style.configure('TLabel',background=BG,foreground=MUTED)
    style.configure('Title.TLabel',foreground=TEXT,font=(body,20,'bold'))
    style.configure('Wordmark.TLabel',foreground=ACCENT,font=(display,10,'bold'))
    style.configure('Status.TLabel',foreground=ACCENT,font=(body,10))
    style.configure('Metric.TLabel',background=SURFACE,foreground=TEXT,font=(display,19,'bold'))
    style.configure('MetricCaption.TLabel',background=SURFACE,foreground=MUTED,font=(body,9))
    style.configure('Metric.TFrame',background=SURFACE)
    style.configure('TLabelframe',background=BG,borderwidth=1,relief='solid')
    style.configure('TLabelframe.Label',background=BG,foreground=TEXT,font=(body,10,'bold'),padding=(8,5))
    style.configure('TButton',padding=(10,5),background=SURFACE,foreground=TEXT,borderwidth=1,focusthickness=1,focuscolor=ACCENT)
    style.map('TButton',background=[('disabled','#171b1e'),('pressed','#384844'),('active','#2b3538')],
              foreground=[('disabled','#778386')],bordercolor=[('focus',ACCENT),('active','#617875')])
    style.configure('Primary.TButton',background=ACCENT,foreground='#12241d',font=(body,10,'bold'))
    style.map('Primary.TButton',background=[('disabled','#293a33'),('pressed','#8cc7ad'),('active','#d6ffe9')],
              foreground=[('disabled','#82968d'),('!disabled','#12241d')])
    style.configure('TNotebook',background=BG,borderwidth=0,tabmargins=(0,4,0,6))
    style.configure('TNotebook.Tab',padding=(18,7),background=BG,foreground=MUTED,borderwidth=0)
    style.map('TNotebook.Tab',background=[('selected','#293a34'),('active',SURFACE)],foreground=[('selected',ACCENT),('active',TEXT)])
    for name in ('TEntry','TCombobox','TSpinbox'):
        style.configure(name,fieldbackground=SURFACE,background=SURFACE,foreground=TEXT,
                        insertcolor=TEXT,arrowcolor=ACCENT,padding=5,borderwidth=1)
        style.map(name,fieldbackground=[('readonly',SURFACE),('disabled','#171b1e')],
                  foreground=[('disabled','#778386'),('readonly',TEXT)],bordercolor=[('focus',ACCENT)])
    for name in ('TCheckbutton','TRadiobutton'):
        style.configure(name,background=BG,foreground=MUTED,indicatorbackground=SURFACE,indicatormargin=5)
        style.map(name,background=[('active',BG)],foreground=[('active',TEXT),('disabled','#778386')],
                  indicatorbackground=[('selected','#527a68')])
    row_height=font.Font(root,font=(body,10)).metrics('linespace')+10
    style.configure('Treeview',background=VIEW,fieldbackground=VIEW,foreground=TEXT,rowheight=row_height,borderwidth=0)
    style.map('Treeview',background=[('selected','#304d46')],foreground=[('selected',TEXT)])
    style.configure('Treeview.Heading',background=SURFACE,foreground=MUTED,padding=(8,7),relief='flat')
    style.map('Treeview.Heading',background=[('active','#2b3538')])
    style.configure('TPanedwindow',background=BG,sashwidth=10)
    style.configure('Horizontal.TProgressbar',background=ACCENT,troughcolor=SURFACE,borderwidth=0,arrowsize=1)
    style.layout('Horizontal.TProgressbar',[('Horizontal.Progressbar.trough',{'sticky':'nswe','children':[
        ('Horizontal.Progressbar.pbar',{'side':'left','sticky':'ns'})]})])
    for axis in ('Horizontal','Vertical'):
        style.configure(axis+'.TScrollbar',background=LINE,troughcolor=BG,arrowcolor=MUTED,borderwidth=0)
    root.option_add('*TCombobox*Listbox.background',SURFACE)
    root.option_add('*TCombobox*Listbox.foreground',TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground','#304d46')
    root.option_add('*TCombobox*Listbox.selectForeground',TEXT)
    return style


class BetaHeader:
    def __init__(self,app):
        self.app=app;self.timer=None;self.running=False
        root=app.root
        app.header=ttk.Frame(root,padding=(22,12,22,6));app.header.pack(fill='x')
        app.header.columnconfigure(0,weight=1)
        app.header_left=ttk.Frame(app.header);app.header_left.grid(row=0,column=0,sticky='nsew')
        brand=ttk.Frame(app.header_left);brand.pack(fill='x',pady=(0,5))
        ttk.Label(brand,text='让空间，成为作品。',style='Title.TLabel').pack(side='left')
        ttk.Label(brand,text='SPATIAL STUDIO / BETA' if app.beta else 'SPATIAL STUDIO',style='Wordmark.TLabel').pack(side='left',padx=(16,0))
        ttk.Label(app.header_left,text='采集真实空间   /   自动几何重建   /   物体精细建模' if app.beta else '采集空间   /   精炼点云   /   重建几何   /   物体建模').pack(anchor='w')
        app.metrics_host=ttk.Labelframe(app.header,text='场景数据',padding=(12,6))
        toolbar=ttk.Frame(root,padding=(22,6,22,8));toolbar.pack(fill='x')
        ttk.Button(toolbar,text='导入数据集',command=app.import_dataset).pack(side='left')
        ttk.Button(toolbar,text='新建数据集',command=app.new_dataset).pack(side='left',padx=7)
        app.export_button=ttk.Button(toolbar,text='导出数据集',command=app.export_data)
        app.export_button.pack(side='left')
        self.stop_button=ttk.Button(toolbar,text='停止处理',command=app.stop_beta)
        if app.beta:self.stop_button.pack(side='right')
        self.generate_button=ttk.Button(toolbar,text='生成 / 重新处理',command=app.start_beta)
        if app.beta:self.generate_button.pack(side='right',padx=7)
        app.dataset_status=tk.StringVar(root,value='未选择数据集 · 开始扫描或导入已有数据')
        row=ttk.Frame(root,padding=(22,0,22,8));row.pack(fill='x')
        row.columnconfigure(0,weight=1)
        self.dataset_label=ttk.Label(row,textvariable=app.dataset_status,wraplength=800)
        self.dataset_label.grid(row=0,column=0,sticky='w')
        self.phase=tk.StringVar(root,value='准备就绪')
        ttk.Label(row,textvariable=self.phase,style='Status.TLabel').grid(row=0,column=1,sticky='e',padx=(12,0))
        track=ttk.Frame(root,height=3);track.pack(fill='x',padx=22);track.pack_propagate(False)
        self.bar=ttk.Progressbar(track,mode='indeterminate')
        # One Python timer owns status and progress; no orphan Tcl animation timer.
        self.bar.bind('<Destroy>',self.close,add='+')
        self.timer=root.after(200,self.update)

    def update(self):
        self.timer=None
        app=self.app
        if app.closing:return
        busy=app._busy();scan=app.views.get('acquisition')
        phase={'preprocess':'表面融合与点云清洗…','load':'准备几何重建…','reconstruct':'构建几何表面…'}.get(app.auto_phase)
        if app.auto_pending and not app.auto_active:phase='正在停止处理…'
        if not phase:
            phase='实时采集中' if scan and scan.busy else '后台处理中…' if busy else '准备就绪'
        self.phase.set(phase)
        self.generate_button.state(['disabled'] if busy or not app.dataset else ['!disabled'])
        self.stop_button.state(['!disabled'] if app.auto_active else ['disabled'])
        if busy!=self.running:
            self.running=busy
            if busy:self.bar.pack(fill='both',expand=True)
            else:self.bar.pack_forget();self.bar['value']=0
        if busy:self.bar.step(6)
        self.timer=app.root.after(250,self.update)

    def close(self,event):
        if self.timer:self.app.root.after_cancel(self.timer);self.timer=None


class MetricsPanel(ttk.Frame):
    """Three readable KPIs, with provenance/quality on a subordinate line."""
    def __init__(self,parent,view):
        super().__init__(parent,style='Metric.TFrame',padding=(12,8))
        self.view=view;self.values={};self.captions={}
        for col,(key,label) in enumerate((('points','当前点云'),('reduction','点数精简'),('faces','几何三角面'))):
            cell=ttk.Frame(self,style='Metric.TFrame',padding=(8,0));cell.grid(row=0,column=col,sticky='nsew')
            self.columnconfigure(col,weight=1)
            caption=ttk.Label(cell,text=label,style='MetricCaption.TLabel');caption.pack(anchor='w');self.captions[key]=caption
            value=tk.StringVar(self,value='—');self.values[key]=value
            ttk.Label(cell,textvariable=value,style='Metric.TLabel').pack(anchor='w',pady=(2,5))
        self.detail=tk.StringVar(self,value='载入结果后显示实际指标')
        ttk.Label(self,textvariable=self.detail,style='MetricCaption.TLabel',wraplength=510,justify='left').grid(row=1,column=0,columnspan=3,sticky='w',padx=8)
        self.trace=view.metrics.trace_add('write',self.update_values)
        self.bind('<Destroy>',self.dispose);self.update_values()

    def update_values(self,*args):
        data=self.view.metrics_values
        gaussian=data.get('gaussians')
        self.captions['faces'].configure(text='高斯数量' if gaussian is not None else '几何三角面')
        for key,var in self.values.items():
            value=gaussian if key=='faces' and gaussian is not None else data.get(key)
            var.set('—' if value is None else f'{value:.1f}%' if key=='reduction' else f'{value:,}')
        if data:
            coverage=data.get('coverage');p95=data.get('p95');raw=data.get('raw')
            quality=f'10 mm 覆盖 {coverage:.1%}' if coverage is not None else '10 mm 覆盖未记录'
            if p95 is not None:quality+=f'  ·  P95 {p95:.2f} mm'
            if gaussian is not None:quality=f'留出视角 PSNR {data["psnr"]:.2f} dB · 图像拟合指标'
            self.detail.set(f"{data['method']}  ·  精细模型 {data['objects']} 份"+(f"  ·  原始 {raw:,} 点" if raw else '')+'\n'+quality)

    def dispose(self,event):
        if event.widget is self:self.view.metrics.trace_remove('write',self.trace)


class EmptyViewport:
    """Event-driven illustration, never mixed with measured geometry or exported data."""
    def __init__(self,parent):
        self.canvas=tk.Canvas(parent,background=VIEW,highlightthickness=0,takefocus=False)
        self.title='等待场景进入视野'
        self.subtitle='开始扫描，或从上方导入数据集'
        self.canvas.bind('<Configure>',self.draw)

    def show(self):
        self.canvas.place(relx=0,rely=0,relwidth=1,relheight=1)
        tk.Misc.lift(self.canvas)

    def hide(self):self.canvas.place_forget()

    def draw(self,event=None):
        c=self.canvas;w=c.winfo_width();h=c.winfo_height()
        if w<20 or h<20:return
        c.delete('all');cx=w*.5;cy=h*.43;r=min(w*.23,h*.24,150)
        # A quiet point-surface study: perspective rings and a folded scan volume.
        for i in range(1,6):
            radius=r*(.7+i*.25)
            c.create_oval(cx-radius,cy+r*.7-radius*.19,cx+radius,cy+r*.7+radius*.19,outline='#1a2a2c')
        for row in range(19):
            v=-1+row/9
            for col in range(33):
                a=col/32*math.pi*2
                radius=r*math.sqrt(max(0,1-v*v))*(.87+.13*math.sin(a*3+v*4))
                x=radius*math.cos(a);z=radius*math.sin(a)
                px=cx+x*.9+v*r*.23;py=cy+v*r*.88-z*.25
                luminance=(z/max(r,1)+1)/2
                color='#%02x%02x%02x'%(int(45+120*luminance),int(70+140*luminance),int(72+111*luminance))
                size=.65+luminance*.6
                c.create_oval(px-size,py-size,px+size,py+size,fill=color,outline='')
        c.create_text(cx,cy+r+35,text=self.title,fill=TEXT,font=('Microsoft YaHei UI',13,'bold'))
        if h>300:c.create_text(cx,cy+r+65,text=self.subtitle,fill=MUTED,font=('Microsoft YaHei UI',10),width=max(w-40,100))
        c.create_text(18,h-20,text='空间示意 · 非采集数据',fill='#718480',anchor='w',font=('Microsoft YaHei UI',9))


def style_view(view,name):
    """Shared presentation edits after each lazily constructed page."""
    def visit(widget):
        if widget.winfo_class()=='Canvas':widget.configure(background=VIEW)
        for child in widget.winfo_children():visit(child)
    visit(view.root)
    if name=='acquisition':
        view.start_button.configure(style='Primary.TButton',text='开始扫描')
        if getattr(view,'beta',False):view.finish_button.configure(text='完成并生成')
        if getattr(view,'beta',False):
            view.import_button.pack_forget()
            view.preprocess_button.master.pack_forget()
        view.map_frame.configure(text='空间画布  /  已融合场景')
        view.live_panel.label.master.configure(text='实时彩色点云')
        view.rgb_panel.frame.configure(background=VIEW)
        view.rgb_panel.label.configure(background=VIEW,foreground=MUTED,font=('Microsoft YaHei UI',11))
        view.map_panel.empty_view.title='从真实空间开始'
        view.live_panel.empty_view.title='等待深度相机'
        view.live_panel.empty_view.subtitle='开始扫描后显示实时 RGB 点云'
        # Clear hierarchy: large accumulated scene, narrower live RGB + cloud column.
        panes=view.map_frame.master
        panes.pane(view.map_frame,weight=3)
        panes.pane(panes.panes()[1],weight=2)
        initial_split(panes,.6)
        initial_split(view.rgb_panel.frame.master.master,.4,vertical=True)
        for child in view.start_button.master.winfo_children():
            if child.winfo_class()=='TLabel':child.configure(wraplength=580)
    elif name=='segmentation':
        view.segmentation.tree.configure(height=3)
        view.segmentation.apply_button.configure(style='Primary.TButton')
        initial_split(view.segmentation.rgb_panel.canvas.master.master,.55,vertical=True)
        view.segmentation.open_button.configure(text='导入分割来源')
        view.segmentation.save_button.configure(text='保存分割')
        view.segmentation.export_button.configure(text='导出物体')
        view.segmentation.next_button.configure(text='保存并精细建模')
        view.refinement.cancel_button.configure(text='停止任务')
        view.refinement.prepare_button.configure(text='准备建模')
        view.refinement.submit_button.configure(text='生成精细模型')
        view.refinement.config_button.configure(text='服务配置')
    elif name=='review':
        view.export_button.configure(style='Primary.TButton')
        view.cancel_button.configure(text='停止读取')
        view.cloud.label.master.configure(text='场景来源  /  点云 · RGB')
        view.result_tabs.tab(view.model_frame,text='重建成果')


def initial_split(panes,fraction,vertical=False):
    """Wait for real layout dimensions; no delayed callback outliving destroyed pages."""
    def sized(event):
        extent=event.height if vertical else event.width
        if extent>300:
            panes.sashpos(0,int(extent*fraction))
            panes.unbind('<Configure>',binding)
    binding=panes.bind('<Configure>',sized,add='+')
