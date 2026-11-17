"""Letterboxed original RGB with image-coordinate box prompts and mask preview."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image,ImageTk


class RgbPromptPanel:
    def __init__(self,parent,on_box,can_edit,on_clear=None,instruction=None):
        self.on_box=on_box;self.can_edit=can_edit;self.image=None;self.mask=None;self.box=None;self.drag=None
        self.on_clear=on_clear
        self.canvas=tk.Canvas(parent,height=220,background='#333333',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda _:self.draw())
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.move)
        self.canvas.bind('<ButtonRelease-1>',self.release)
        ttk.Label(parent,text=instruction or '在原始 RGB 照片上拖框 → 预览掩码 → 应用到三维；框选不会改变原始点坐标。').pack(fill='x')

    def set_image(self,path):
        self.image=Image.open(path).convert('RGB') if path else None
        self.mask=None;self.box=None;self.drag=None;self.draw()

    def geometry(self):
        if self.image is None:return 1.,0.,0.
        w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height())
        scale=min(w/self.image.width,h/self.image.height)
        return scale,(w-self.image.width*scale)/2,(h-self.image.height*scale)/2

    def image_point(self,x,y,clamp=False):
        if self.image is None:return None
        scale,ox,oy=self.geometry();u=(x-ox)/scale;v=(y-oy)/scale
        if not clamp and not (0<=u<=self.image.width and 0<=v<=self.image.height):return None
        return float(np.clip(u,0,self.image.width)),float(np.clip(v,0,self.image.height))

    def draw(self):
        self.canvas.delete('all')
        if self.image is None:
            self.canvas.create_text(10,10,anchor='nw',text='尚未载入原始 RGB 图像',fill='white');return
        rgb=np.asarray(self.image).copy()
        if self.mask is not None:
            rgb[self.mask]=(rgb[self.mask]*.5+np.array([35,220,110])*.5).astype(np.uint8)
        scale,ox,oy=self.geometry()
        image=Image.fromarray(rgb).resize((max(1,round(self.image.width*scale)),max(1,round(self.image.height*scale))),Image.Resampling.BILINEAR)
        self.photo=ImageTk.PhotoImage(image);self.canvas.create_image(round(ox),round(oy),image=self.photo,anchor='nw')
        if self.box:
            x0,y0,x1,y1=self.box
            self.canvas.create_rectangle(ox+x0*scale,oy+y0*scale,ox+x1*scale,oy+y1*scale,outline='#ffb933',width=2)

    def press(self,event):
        if not self.can_edit():return
        point=self.image_point(event.x,event.y)
        if point is None:return
        if self.on_clear:self.on_clear()
        self.drag=point;self.mask=None;self.box=(*point,*point);self.draw()

    def move(self,event):
        if self.drag is None:return
        point=self.image_point(event.x,event.y,True)
        self.box=(min(self.drag[0],point[0]),min(self.drag[1],point[1]),max(self.drag[0],point[0]),max(self.drag[1],point[1]))
        self.draw()

    def release(self,event):
        if self.drag is None:return
        self.move(event);self.drag=None
        if self.box[2]-self.box[0]>=5 and self.box[3]-self.box[1]>=5:self.on_box(self.box)
        else:self.box=None;self.draw()

    def preview(self,proposal):
        self.mask=proposal['mask'];self.box=proposal['box'];self.draw()
