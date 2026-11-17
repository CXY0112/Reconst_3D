"""Scene-only Gaussian viewport, using saved cameras and GPU splats."""
from pathlib import Path
import time
import tkinter as tk
import numpy as np
from scipy.spatial.transform import Rotation
from gaussian_scene import GaussianScene,saved_cameras


class EmbeddedBrush:
    def __init__(self,parent):
        self.host=tk.Frame(parent,background='black',takefocus=True);self.host.pack(fill='both',expand=True)
        self.message=tk.Label(self.host,text='等待 3DGS 模型',foreground='#b7d8ca',background='black',wraplength=500)
        self.message.place(relx=.5,rely=.5,anchor='center')
        self.model=None;self.identity=None;self.scene=None;self.cameras=[];self.camera_index=0
        self.gpu=None;self.job=None;self.present_job=None;self.closed=False;self.error=None;self.drag=None;self.draw_count=0
        self.render_ms=0.;self.pose=None;self.initial_pose=None;self.focus_distance=1.
        self.host.bind('<Map>',lambda _:self.request_draw())
        self.host.bind('<Unmap>',lambda _:self.stop())
        self.host.bind('<Configure>',lambda _:self.request_draw())
        self.host.bind('<Expose>',lambda _:self.request_draw())
        for button in (1,2,3):
            self.host.bind(f'<ButtonPress-{button}>',self._press)
            self.host.bind(f'<B{button}-Motion>',lambda e,b=button:self._motion(e,b))
            self.host.bind(f'<ButtonRelease-{button}>',lambda _:setattr(self,'drag',None))
        self.host.bind('<MouseWheel>',self._wheel)
        self.host.bind('<Double-Button-1>',lambda _:self.reset())
        self.host.bind('<KeyPress>',self._key)

    def set_model(self,model,identity=None):
        model=Path(model).resolve();key=(str(model),identity)
        if key!=self.identity:
            self.clear();self.model=model;self.identity=key
            try:
                self.scene=GaussianScene(model);self.cameras=saved_cameras(model)
                self.camera_index=0;self.reset()
            except Exception as exc:self._failed(str(exc))
        self.request_draw()

    def reset(self):
        if not self.cameras:return
        camera=self.cameras[self.camera_index]
        self.pose=camera['pose'].copy();self.initial_pose=self.pose.copy()
        depth=((self.scene.means-self.pose[:3,3])@self.pose[:3,:3])[:,2]
        depth=depth[(depth>.1)&(self.scene.opacity>.1)]
        self.focus_distance=float(np.median(depth)) if len(depth) else 1.
        self.request_draw()

    def request_draw(self):
        if not self.closed and self.scene is not None and self.error is None and self.host.winfo_ismapped() and self.job is None:
            self.job=self.host.after(16,self._draw)

    def _draw(self):
        self.job=None
        if self.closed or self.scene is None or not self.host.winfo_ismapped():return
        try:
            if self.gpu is None:
                from gaussian_renderer import GaussianRenderer
                self.gpu=GaussianRenderer(self.host)
            width,height=self.host.winfo_width(),self.host.winfo_height()
            if min(width,height)<2:return
            camera=self.cameras[self.camera_index];w,h=camera['size']
            scale=min(width/w,height/h)
            fx,fy,cx,cy=camera['intrinsics']*scale
            intr=[fx,fy,cx+(width-w*scale)/2,cy+(height-h*scale)/2]
            start=time.perf_counter()
            self.gpu.draw_scene(self.scene,self.pose,intr,width,height)
            self.render_ms=(time.perf_counter()-start)*1000;self.draw_count+=1
            self.message.place_forget()
            if self.present_job is None:self.present_job=self.host.after(250,self._present)
        except Exception as exc:self._failed(str(exc))

    def _present(self):
        self.present_job=None
        if self.closed or self.scene is None or not self.host.winfo_ismapped():return
        try:
            # Tk may erase a native GL child after async page layout has settled.
            # Reuse the last GPU image; do not reproject or sort while idle.
            if self.gpu:self.gpu.present()
            self.present_job=self.host.after(250,self._present)
        except Exception as exc:self._failed(str(exc))

    def _press(self,event):
        self.host.focus_set();self.drag=(event.x,event.y)

    def _motion(self,event,button):
        if self.pose is None or self.drag is None:return
        dx,dy=event.x-self.drag[0],event.y-self.drag[1];self.drag=(event.x,event.y)
        if button==2:
            self.pose[:3,3]+=self.pose[:3,:3]@np.array([-dx,-dy,0])*self.focus_distance*.002
        else:
            center=self.pose[:3,3]+self.pose[:3,2]*self.focus_distance
            turn=Rotation.from_rotvec(np.array([-dy,dx,0])*.004).as_matrix()
            self.pose[:3,:3]=self.pose[:3,:3]@turn
            if button==1:self.pose[:3,3]=center-self.pose[:3,2]*self.focus_distance
        self.request_draw()

    def _wheel(self,event):
        if self.pose is not None:
            self.pose[:3,3]+=self.pose[:3,2]*(event.delta/120)*max(.02,self.focus_distance*.08)
            self.request_draw()
        return 'break'

    def _key(self,event):
        key=event.keysym.lower()
        if key=='r':self.reset()
        elif key in ('bracketleft','bracketright') and self.cameras:
            self.camera_index=(self.camera_index+(1 if key=='bracketright' else -1))%len(self.cameras);self.reset()
        elif self.pose is not None and key in ('w','s','a','d','q','e'):
            direction={'w':[0,0,1],'s':[0,0,-1],'a':[-1,0,0],'d':[1,0,0],'q':[0,-1,0],'e':[0,1,0]}[key]
            self.pose[:3,3]+=self.pose[:3,:3]@np.array(direction)*max(.02,self.focus_distance*.03)
            self.request_draw()
        return 'break'

    def _failed(self,message):
        self.stop();self.error=message
        self.message.configure(text='3DGS 载入失败：'+message);self.message.place(relx=.5,rely=.5,anchor='center')

    def stop(self):
        if self.job:self.host.after_cancel(self.job);self.job=None
        if self.present_job:self.host.after_cancel(self.present_job);self.present_job=None

    def clear(self):
        self.stop();self.model=self.identity=self.scene=self.pose=None;self.cameras=[];self.error=None
        self.message.configure(text='等待 3DGS 模型');self.message.place(relx=.5,rely=.5,anchor='center')

    def shutdown(self):
        self.closed=True;self.stop()
        if self.gpu:self.gpu.close();self.gpu=None
