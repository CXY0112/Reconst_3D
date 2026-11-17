"""Sequential demo orchestration over production stage controllers."""
import tkinter as tk
from tkinter import ttk
from dataset_store import inspect_dataset


class BetaPipeline:
    def __init__(self, app, toolbar):
        self.app=app;self.active=False;self.view=None;self.phase=None;self.number=None;self.timer=None
        self.status=tk.StringVar(value='扫描后点击完成并保存，或导入数据集，即可自动生成模型')
        self.button=ttk.Button(toolbar,text='继续处理',command=self.start)
        self.button.pack(side='right')
        self.status_label=ttk.Label(app.header_left,textvariable=self.status,wraplength=700,padding=6)
        self.status_label.pack(fill='x')
        app.header_left.bind('<Configure>',lambda event:self.status_label.configure(wraplength=max(180,event.width-16)))

    def schedule(self):
        if self.active and not self.app.closing:self.timer=self.app.root.after(100,self.tick)

    def start(self):
        if self.active or self.app.closing or not self.app.dataset:return
        self.active=True;self.phase='next';self.view=None
        self.button.configure(text='停止处理',command=self.stop)
        self.status.set('正在检查数据集…');self.schedule()

    def saved(self, number):
        if self.active and self.phase=='run' and number==self.number:
            self.phase='next';self.view=None

    def fail(self, reason):
        self.stop()
        self.status.set('自动处理已停止：'+reason+'。修正后可点击“继续处理”；已保存的结果保留。')

    def stop(self):
        self.active=False
        if self.timer is not None:
            self.app.root.after_cancel(self.timer);self.timer=None
        if self.view:
            if hasattr(self.view,'cancel_job'):self.view.cancel_job()
            elif hasattr(self.view,'cancel'):self.view.cancel.set()
        self.button.configure(text='继续处理',command=self.start)
        self.status.set('处理已停止；已完整保存的结果保留')

    @staticmethod
    def busy(view):
        return getattr(view,'working',None) or getattr(view,'busy',False) or getattr(view,'importing',False)

    def tick(self):
        self.timer=None
        if not self.active or self.app.closing:return
        try:self.advance()
        except Exception as exc:self.fail(str(exc))
        self.schedule()

    def advance(self):
        app=self.app
        if self.phase=='next':
            if any(self.busy(v) for v in app.views.values()):return
            states=inspect_dataset(app.dataset)
            pending=next((n for n in (2,3,4) if not states[n]['complete'] or states[n]['stale']),None)
            if pending is None:
                self.view=app.select_stage('review');self.view.on_task_failed=self.fail
                self.phase='review';self.status.set('正在加载展示模型…');return
            if not states[1]['complete']:
                self.active=False;self.button.configure(text='继续处理',command=self.start)
                self.status.set('请先扫描并保存，或导入包含完整扫描数据的数据集');return
            self.number=pending
            name={2:'preprocessing',3:'reconstruction',4:'appearance'}[pending]
            self.view=app._ensure_stage(name,hydrate=False)
            self.view.dataset_refresh=False;self.view.on_task_failed=self.fail
            self.status.set({2:'正在清洗、融合并均匀化点云…',3:'正在使用 Poisson 生成几何模型…',4:'正在投影 RGB 并补全小范围缺色…'}[pending])
            source=states[pending-1]['path']
            if pending==2:self.view.load(source)
            else:self.view.set_source(source)
            self.phase='load';return
        if self.busy(self.view):
            # Forward useful native progress without exposing parameter controls.
            if self.phase=='run':self.status.set(self.view.status.get())
            return
        if self.phase=='load':
            self.phase='run'
            if self.number==2:self.view.process()
            else:self.view.start()
            if not self.busy(self.view) and self.phase=='run':self.fail('任务未能启动')
        elif self.phase=='run':
            self.fail('任务未完成保存，请检查上方错误提示')
        elif self.phase=='review':
            self.active=False;self.view=None
            self.button.configure(text='继续处理',command=self.start)
            self.status.set('模型已完成，可查看源点云、原始 RGB、最终模型及质量指标，或导出数据集')
