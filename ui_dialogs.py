"""Parent-aware dialogs: styled Beta modals, native dialogs in the regular UI."""
from pathlib import Path
from types import SimpleNamespace
import fnmatch
import tkinter as tk
from tkinter import ttk, filedialog as native_files, messagebox as native_messages, simpledialog as native_simple


def themed(parent):
    while parent is not None:
        if getattr(parent,'beta_visual',False):return True
        parent=getattr(parent,'master',None)
    return False


class Modal:
    def __init__(self,parent,title):
        from beta_theme import BG,TEXT
        self.parent=parent;self.result=None;self.previous=parent.grab_current()
        self.window=tk.Toplevel(parent);self.window.withdraw();self.window.title(title)
        self.window.configure(background=BG);self.window.transient(parent.winfo_toplevel())
        self.body=ttk.Frame(self.window,padding=22);self.body.pack(fill='both',expand=True)
        ttk.Label(self.body,text=title,foreground=TEXT,font=('Microsoft YaHei UI',15,'bold')).pack(anchor='w',pady=(0,16))
        self.window.protocol('WM_DELETE_WINDOW',self.cancel)
        self.window.bind('<Escape>',lambda _:self.cancel())

    def finish(self,value):self.result=value;self.window.destroy()
    def cancel(self):self.finish(None)

    def show(self,focus=None):
        w=self.window;w.update_idletasks();host=self.parent.winfo_toplevel()
        x=max(0,min(host.winfo_rootx()+(host.winfo_width()-w.winfo_reqwidth())//2,w.winfo_screenwidth()-w.winfo_reqwidth()))
        y=max(0,min(host.winfo_rooty()+(host.winfo_height()-w.winfo_reqheight())//2,w.winfo_screenheight()-w.winfo_reqheight()-60))
        w.geometry(f'+{x}+{y}');w.deiconify();w.wait_visibility();w.grab_set()
        (focus or w).focus_set()
        try:self.parent.wait_window(w)
        finally:
            if self.previous is not None and self.previous.winfo_exists():self.previous.grab_set()
        return self.result


def askstring(title,prompt,parent=None,initialvalue='',**kwargs):
    if not themed(parent):return native_simple.askstring(title,prompt,parent=parent,initialvalue=initialvalue,**kwargs)
    d=Modal(parent,title);ttk.Label(d.body,text=prompt,wraplength=480,justify='left').pack(fill='x')
    value=tk.StringVar(d.window,value=initialvalue or '')
    field=ttk.Entry(d.body,textvariable=value,width=42,show=kwargs.get('show',''));field.pack(fill='x',pady=(12,18))
    field.selection_range(0,'end')
    row=ttk.Frame(d.body);row.pack(fill='x')
    ttk.Button(row,text='保存',style='Primary.TButton',command=lambda:d.finish(value.get())).pack(side='right')
    ttk.Button(row,text='取消',command=d.cancel).pack(side='right',padx=8)
    d.window.bind('<Return>',lambda _:d.finish(value.get()))
    return d.show(field)


def message(kind,title,message,parent=None,**kwargs):
    if not themed(parent):return getattr(native_messages,kind)(title,message,parent=parent,**kwargs)
    d=Modal(parent,title);ttk.Label(d.body,text=message,wraplength=540,justify='left').pack(fill='x')
    row=ttk.Frame(d.body);row.pack(fill='x',pady=(20,0))
    question=kind.startswith('ask')
    choices=[('确定',True if question else 'ok')]
    if question:choices=[('是',True),('否',False)] if kind in ('askyesno','askyesnocancel') else [('确定',True),('取消',False)]
    for i,(text,value) in enumerate(choices):
        button=ttk.Button(row,text=text,style='Primary.TButton' if i==0 else 'TButton',command=lambda v=value:d.finish(v))
        button.pack(side='right' if i==0 else 'left')
    if kind=='askyesnocancel':ttk.Button(row,text='取消',command=d.cancel).pack(side='left',padx=8)
    result=d.show()
    return bool(result) if kind in ('askyesno','askokcancel','askretrycancel') else result


class FilePicker(Modal):
    def __init__(self,parent,mode,**options):
        super().__init__(parent,options.get('title') or '选择路径')
        self.mode=mode;self.options=options
        initial=Path(options.get('initialdir') or Path(__file__).parent/'results').expanduser()
        self.folder=initial if initial.is_dir() else Path.home()
        row=ttk.Frame(self.body);row.pack(fill='x')
        ttk.Button(row,text='上一级',command=lambda:self.navigate(self.folder.parent)).pack(side='left',padx=(0,8))
        self.path=tk.StringVar(self.window,value=str(self.folder));entry=ttk.Entry(row,textvariable=self.path,width=65);entry.pack(side='left',fill='x',expand=True)
        entry.bind('<Return>',lambda _:self.navigate(Path(self.path.get()).expanduser()))
        listing=ttk.Frame(self.body);listing.pack(fill='both',expand=True,pady=12)
        self.tree=ttk.Treeview(listing,columns=('kind',),show='tree headings',height=12,selectmode='browse')
        self.tree.heading('#0',text='名称');self.tree.heading('kind',text='类型');self.tree.column('#0',width=490);self.tree.column('kind',width=80,stretch=False)
        self.tree.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(listing,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind('<<TreeviewSelect>>',self.select);self.tree.bind('<Double-1>',self.open_selected)
        self.filename=tk.StringVar(self.window,value=options.get('initialfile',''))
        if mode!='directory':
            ttk.Label(self.body,text='文件名').pack(anchor='w')
            ttk.Entry(self.body,textvariable=self.filename).pack(fill='x',pady=(4,8))
        self.error=tk.StringVar(self.window)
        ttk.Label(self.body,textvariable=self.error,wraplength=600,foreground='#f4b6a0').pack(fill='x')
        row=ttk.Frame(self.body);row.pack(fill='x',pady=(12,0))
        ttk.Button(row,text='选择此文件夹' if mode=='directory' else '保存' if mode=='save' else '打开',style='Primary.TButton',command=self.accept).pack(side='right')
        ttk.Button(row,text='取消',command=self.cancel).pack(side='right',padx=8)
        self.navigate(self.folder)

    def navigate(self,path):
        try:
            path=path.resolve()
            if not path.is_dir():raise ValueError('请输入有效文件夹路径')
            entries=sorted(path.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))
        except (OSError,ValueError) as exc:self.error.set(str(exc));return
        self.folder=path;self.path.set(str(path));self.tree.delete(*self.tree.get_children());self.entries={};self.error.set('')
        patterns=[p for _,pattern in self.options.get('filetypes',[]) for p in (pattern if isinstance(pattern,(tuple,list)) else pattern.split())]
        for item in entries:
            directory=item.is_dir()
            if not directory and (self.mode=='directory' or patterns and not any(fnmatch.fnmatch(item.name.lower(),p.lower()) for p in patterns)):continue
            ident=self.tree.insert('','end',text=item.name,values=('文件夹' if directory else item.suffix,));self.entries[ident]=item

    def select(self,event=None):
        selected=self.tree.selection()
        if selected and not self.entries[selected[0]].is_dir():self.filename.set(self.entries[selected[0]].name)

    def open_selected(self,event=None):
        selected=self.tree.selection()
        if selected:
            path=self.entries[selected[0]]
            if path.is_dir():self.navigate(path)
            elif self.mode=='open':self.accept()

    def accept(self):
        try:
            if self.mode=='directory':
                selected=self.tree.selection();path=self.entries[selected[0]] if selected else Path(self.path.get()).expanduser()
                if not path.is_dir():raise ValueError('请选择有效文件夹')
            else:
                name=self.filename.get().strip()
                if not name:raise ValueError('请输入文件名')
                path=self.folder/Path(name).expanduser()
                if self.mode=='save' and not path.suffix:path=path.with_suffix(self.options.get('defaultextension',''))
                if self.mode=='open' and not path.is_file():raise ValueError('文件不存在')
                if self.mode=='save':
                    if path.is_dir() or not path.parent.is_dir():raise ValueError('保存位置或文件名无效')
                    if path.exists() and not message('askyesno','覆盖文件',f'已有同名文件：{path.name}\n是否覆盖？',parent=self.window):return
            self.finish(str(path.resolve()))
        except (OSError,ValueError) as exc:self.error.set(str(exc))


def file_dialog(mode,**options):
    parent=options.pop('parent',None)
    if not themed(parent):return getattr(native_files,{'open':'askopenfilename','save':'asksaveasfilename','directory':'askdirectory'}[mode])(parent=parent,**options)
    dialog=FilePicker(parent,mode,**options)
    return dialog.show(dialog.tree) or ''


simpledialog=SimpleNamespace(askstring=askstring)
messagebox=SimpleNamespace(**{kind:(lambda title,message,k=kind,**kw:globals()['message'](k,title,message,**kw))
    for kind in ('showinfo','showwarning','showerror','askyesno','askyesnocancel','askokcancel','askretrycancel')})
filedialog=SimpleNamespace(askdirectory=lambda **kw:file_dialog('directory',**kw),
    askopenfilename=lambda **kw:file_dialog('open',**kw),asksaveasfilename=lambda **kw:file_dialog('save',**kw))
