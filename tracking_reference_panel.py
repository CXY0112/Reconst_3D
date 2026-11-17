"""A fixed, trusted keyframe thumbnail beside the live RGB view (Tk thread only)."""
from datetime import datetime,timezone,timedelta
import tkinter as tk
from tkinter import ttk
from rgb_preview import RGBPreviewPanel


class TrackingReferencePanel:
    def __init__(self,parent):
        self.frame=ttk.LabelFrame(parent,text='G-CVO 可信参考关键帧',width=260,height=225)
        self.frame.pack_propagate(False)
        self.caption=tk.StringVar(value='等待建立可信参考')
        self.hint=tk.StringVar(value='恢复时，请对照此图返回相近视角')
        ttk.Label(self.frame,textvariable=self.hint,wraplength=245).pack(side='bottom',fill='x',padx=5,pady=4)
        ttk.Label(self.frame,textvariable=self.caption,wraplength=245).pack(side='bottom',fill='x',padx=5)
        self.preview=RGBPreviewPanel(self.frame)
        self.preview.frame.configure(height=135)
        self.key=None
        self.clear()

    def show(self,visible):
        self.clear()
        if visible:self.frame.pack(side='right',fill='both')
        else:self.frame.pack_forget()

    def clear(self):
        self.key=None;self.preview.clear()
        self.preview.label.configure(text='等待可信参考图像')
        self.caption.set('等待建立可信参考')
        self.hint.set('恢复时，请对照此图返回相近视角')

    def set_reference(self,packet):
        key=(packet['node'],packet['stamp'],packet['revision'])
        if key==self.key:return
        self.preview.update(packet['image']);self.key=key
        t=datetime.fromtimestamp(packet['stamp'],timezone(timedelta(hours=8)))
        self.caption.set(f'参考 #{packet["node"]} · {t:%m-%d %H:%M:%S}.{t.microsecond//1000:03d}（北京时间）')

    def set_tracking(self,tracking):
        if tracking.get('state')=='RECOVERY':
            self.hint.set(f'RECOVERY · 参考保持不变\n对照实时 RGB 返回此视角 · 确认 {tracking.get("recovery_good_count",0)}/{tracking.get("recovery_good_required",3)}')
            if tracking.get('mac_enabled'):
                label={'running':'MAC 后台搜索','seed_ready':'MAC 候选确认中','failed':'MAC 本轮未通过，稍后重试',
                       'rejected':'MAC 初值未通过，稍后重试','expired':'MAC 候选已过期','idle':'MAC 等待有效点云'}
                self.hint.set(self.hint.get()+'\n'+label.get(tracking.get('mac_status'),'MAC 等待有效点云'))
        elif self.key is not None:
            self.hint.set('NORMAL · 参考随可信关键帧更新')
