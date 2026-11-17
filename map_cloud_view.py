"""Full-map first-person view: native WGL/VBO, with an asynchronous CPU fallback."""
import ctypes as ct
import os
import queue
import threading
import time
import weakref
import numpy as np
import cv2
from live_cloud_view import LiveCloudPanel,ViewCamera


def view_matrix(camera):
    matrix = np.eye(4,dtype=np.float32)
    matrix[:3,:3] = np.diag([1.,-1.,-1.])@camera.basis().T
    matrix[:3,3] = -matrix[:3,:3]@camera.position
    return matrix


def cpu_map_image(points,nodes,camera,width,height,colors=None,background=(255,255,255)):
    """Project every map point; no depth-range filter or point-count sampling."""
    image = np.empty((height,width,3),np.uint8); image[:] = background
    basis = camera.basis()
    shifted = points-camera.position.astype(np.float32)
    x = np.sum(shifted*basis[:,0],axis=1)
    y = np.sum(shifted*basis[:,1],axis=1)
    z = np.sum(shifted*basis[:,2],axis=1)
    focal = height/(2*np.tan(np.deg2rad(60)/2))
    positive = z>.02
    x,y,z = x[positive],y[positive],z[positive]
    u = np.rint(focal*x/z+width/2).astype(np.int32)
    v = np.rint(focal*y/z+height/2).astype(np.int32)
    inside = (u>=0)&(u<width)&(v>=0)&(v<height)
    # Flat blue points only need occupied pixels; all projected points share colour.
    if colors is None:
        image[v[inside],u[inside]] = [66,133,223]
    else:
        selected=np.flatnonzero(positive)[inside]
        pixels=v[inside]*width+u[inside]
        buffer=np.full(width*height,np.inf,np.float32)
        np.minimum.at(buffer,pixels,z[inside])
        visible=z[inside]<=buffer[pixels]+1e-7
        image[v[inside][visible],u[inside][visible]]=np.asarray(colors)[selected[visible]]
    if len(nodes):
        p = camera.project_coordinates(nodes)
        visible = p[:,2]>.02
        pixels = np.rint(p[:,:2]/np.maximum(p[:,2,None],.02)*focal+[width/2,height/2]).astype(np.int32)
        for i in range(1,len(pixels)):
            if visible[i-1] and visible[i]:
                cv2.line(image,tuple(pixels[i-1]),tuple(pixels[i]),(232,154,36),1,cv2.LINE_AA)
        for xy in pixels[visible]:
            cv2.circle(image,tuple(xy),3,(232,154,36),-1)
    return image


class _WGLRenderer:
    def __init__(self,widget):
        if os.name != 'nt':
            raise RuntimeError('WGL requires Windows')
        from ctypes import wintypes as w
        class PFD(ct.Structure):
            _fields_ = [('nSize',w.WORD),('nVersion',w.WORD),('dwFlags',w.DWORD),
                        *[(n,ct.c_ubyte) for n in ('iPixelType','cColorBits','cRedBits','cRedShift','cGreenBits','cGreenShift',
                         'cBlueBits','cBlueShift','cAlphaBits','cAlphaShift','cAccumBits','cAccumRedBits','cAccumGreenBits',
                         'cAccumBlueBits','cAccumAlphaBits','cDepthBits','cStencilBits','cAuxBuffers','iLayerType','bReserved')],
                        ('dwLayerMask',w.DWORD),('dwVisibleMask',w.DWORD),('dwDamageMask',w.DWORD)]
        self.widget = widget
        self.window = widget.winfo_id()
        self.user,self.gdi,self.gl = ct.WinDLL('user32'),ct.WinDLL('gdi32'),ct.WinDLL('opengl32')
        self.user.GetDC.argtypes=[ct.c_void_p]; self.user.GetDC.restype=ct.c_void_p
        self.user.ReleaseDC.argtypes=[ct.c_void_p,ct.c_void_p]
        self.gdi.ChoosePixelFormat.argtypes=[ct.c_void_p,ct.POINTER(PFD)]; self.gdi.ChoosePixelFormat.restype=ct.c_int
        self.gdi.SetPixelFormat.argtypes=[ct.c_void_p,ct.c_int,ct.POINTER(PFD)]
        self.gdi.SwapBuffers.argtypes=[ct.c_void_p]
        self.gl.wglCreateContext.argtypes=[ct.c_void_p]; self.gl.wglCreateContext.restype=ct.c_void_p
        self.gl.wglMakeCurrent.argtypes=[ct.c_void_p,ct.c_void_p]
        self.gl.wglDeleteContext.argtypes=[ct.c_void_p]
        self.gl.wglGetProcAddress.argtypes=[ct.c_char_p]; self.gl.wglGetProcAddress.restype=ct.c_void_p
        self.gl.glGetString.argtypes=[ct.c_uint]; self.gl.glGetString.restype=ct.c_char_p
        self.dc = self.user.GetDC(self.window)
        self.context = None
        self.buffer = ct.c_uint(0)
        try:
            pfd=PFD(); pfd.nSize=ct.sizeof(PFD); pfd.nVersion=1; pfd.dwFlags=0x25
            pfd.cColorBits=24; pfd.cDepthBits=24; pfd.cStencilBits=8
            pixel=self.gdi.ChoosePixelFormat(self.dc,ct.byref(pfd))
            if not pixel or not self.gdi.SetPixelFormat(self.dc,pixel,ct.byref(pfd)):
                raise RuntimeError('Cannot create OpenGL pixel format')
            self.context=self.gl.wglCreateContext(self.dc)
            if not self.context or not self.gl.wglMakeCurrent(self.dc,self.context):
                raise RuntimeError('Cannot create OpenGL context')
            self.name=(self.gl.glGetString(0x1F01) or b'OpenGL').decode(errors='replace')
            if 'GDI Generic' in self.name:
                raise RuntimeError('OpenGL software driver; use background rasterizer')
            definitions={'glClearColor':([ct.c_float]*4,None),'glClear':([ct.c_uint],None),
                'glEnable':([ct.c_uint],None),'glDisable':([ct.c_uint],None),
                'glViewport':([ct.c_int]*4,None),'glMatrixMode':([ct.c_uint],None),
                'glScissor':([ct.c_int]*4,None),
                'glLoadIdentity':([],None),'glFrustum':([ct.c_double]*6,None),
                'glLoadMatrixf':([ct.POINTER(ct.c_float)],None),'glPointSize':([ct.c_float],None),
                'glColor3f':([ct.c_float]*3,None),'glEnableClientState':([ct.c_uint],None),
                'glVertexPointer':([ct.c_int,ct.c_uint,ct.c_int,ct.c_void_p],None),
                'glColorPointer':([ct.c_int,ct.c_uint,ct.c_int,ct.c_void_p],None),
                'glDisableClientState':([ct.c_uint],None),
                'glDrawArrays':([ct.c_uint,ct.c_int,ct.c_int],None),'glBegin':([ct.c_uint],None),
                'glVertex3f':([ct.c_float]*3,None),'glEnd':([],None),'glFlush':([],None),'glFinish':([],None)}
            for name,(args,result) in definitions.items():
                fn=getattr(self.gl,name); fn.argtypes=args; fn.restype=result
            self.gen=self._extension('glGenBuffers',None,ct.c_int,ct.POINTER(ct.c_uint))
            self.bind=self._extension('glBindBuffer',None,ct.c_uint,ct.c_uint)
            self.upload=self._extension('glBufferData',None,ct.c_uint,ct.c_ssize_t,ct.c_void_p,ct.c_uint)
            self.delete=self._extension('glDeleteBuffers',None,ct.c_int,ct.POINTER(ct.c_uint))
            if self.gen and self.bind and self.upload:
                self.gen(1,ct.byref(self.buffer))
            self.gl.glEnable(0x0B71)  # GL_DEPTH_TEST
            self.gl.glEnableClientState(0x8074)  # GL_VERTEX_ARRAY
        except Exception:
            self.close()
            raise

    def _extension(self,name,result,*args):
        address=self.gl.wglGetProcAddress(name.encode())
        if address in (None,0,1,2,3,ct.c_void_p(-1).value):
            return None
        return ct.WINFUNCTYPE(result,*args)(address)

    def draw(self,points,nodes,camera,width,height,dirty,colors=None):
        gl=self.gl
        gl.wglMakeCurrent(self.dc,self.context)
        focus=self.widget.focus_get() is self.widget
        theme=self.widget.winfo_toplevel()
        gl.glClearColor(*(getattr(theme,'viewport_focus',(66/255,133/255,223/255,1)) if focus else getattr(theme,'viewport_border',(.75,.75,.75,1))))
        gl.glClear(0x4100)
        gl.glEnable(0x0C11); gl.glScissor(2,2,max(width-4,1),max(height-4,1))
        gl.glClearColor(*(v/255 for v in getattr(theme,'viewport_background',(255,255,255))),1); gl.glClear(0x4100)
        gl.glDisable(0x0C11); gl.glViewport(2,2,max(width-4,1),max(height-4,1))
        gl.glMatrixMode(0x1701); gl.glLoadIdentity()
        near=.02; top=near*np.tan(np.deg2rad(60)/2); right=top*width/max(height,1)
        gl.glFrustum(-right,right,-top,top,near,1000.)
        gl.glMatrixMode(0x1700)
        matrix=np.ascontiguousarray(view_matrix(camera).T)
        gl.glLoadMatrixf(matrix.ctypes.data_as(ct.POINTER(ct.c_float)))
        gl.glColor3f(66/255,133/255,223/255); gl.glPointSize(1.4)
        colored=colors is not None
        if colored:
            gl.glEnableClientState(0x8076)
            if dirty or not hasattr(self,'packed'):
                self.packed=np.ascontiguousarray(np.column_stack((points,np.asarray(colors,np.float32)/255)),dtype=np.float32)
            data=self.packed;stride=24
        else:
            gl.glDisableClientState(0x8076);data=points;stride=0
        if self.buffer.value:
            self.bind(0x8892,self.buffer.value)
            if dirty:
                self.upload(0x8892,data.nbytes,data.ctypes.data,0x88E4)
            gl.glVertexPointer(3,0x1406,stride,None)
            if colored:gl.glColorPointer(3,0x1406,stride,ct.c_void_p(12))
        else:
            gl.glVertexPointer(3,0x1406,stride,data.ctypes.data)
            if colored:gl.glColorPointer(3,0x1406,stride,ct.c_void_p(data.ctypes.data+12))
        gl.glDrawArrays(0,0,len(points))
        if self.buffer.value:
            self.bind(0x8892,0)
        gl.glDisableClientState(0x8076)
        gl.glDisable(0x0B71); gl.glColor3f(232/255,154/255,36/255)
        gl.glBegin(3)
        for node in nodes:
            gl.glVertex3f(*map(float,node))
        gl.glEnd(); gl.glPointSize(5); gl.glBegin(0)
        for node in nodes:
            gl.glVertex3f(*map(float,node))
        gl.glEnd(); gl.glEnable(0x0B71)
        gl.glFlush(); self.gdi.SwapBuffers(self.dc)
        # Bound queued camera-view latency and measure completed GPU work.
        gl.glFinish()

    def close(self):
        if self.context:
            self.gl.wglMakeCurrent(self.dc,self.context)
            if self.buffer.value and getattr(self,'delete',None):
                self.delete(1,ct.byref(self.buffer))
            self.gl.wglMakeCurrent(None,None); self.gl.wglDeleteContext(self.context); self.context=None
        if self.dc:
            self.user.ReleaseDC(self.window,self.dc); self.dc=None


class MapCloudPanel(LiveCloudPanel):
    """Shares FPS camera controls with the right view; never rotates the map."""
    def __init__(self,parent,on_render=None):
        import tkinter as tk
        from tkinter import ttk
        self.points=np.empty((0,3),np.float32); self.nodes=np.empty((0,3),np.float32)
        self.camera=ViewCamera(); self.keys=set(); self.navigation_timer=None; self.navigation_at=None
        self.last_drag=None; self.previous_at=None; self.navigation_interval=16
        self.label=tk.Frame(parent,background='white',takefocus=True,highlightthickness=2,
                            highlightbackground='#bbbbbb',highlightcolor='#4285df')
        self.label.pack(fill='both',expand=True)
        self.info=tk.StringVar(value='等待融合地图 · 点击选中后使用 WASD/QE 移动视点')
        ttk.Label(parent,textvariable=self.info,wraplength=650).pack(fill='x')
        ttk.Label(parent,text='左键拖动：原地转头 · W/S前后 A/D左右 Q上 E下 · Z逆/C顺时针滚转 · 滚轮前后 · R/双击居中',wraplength=650).pack(fill='x')
        for event,handler in [('<ButtonPress-1>',self._press),('<B1-Motion>',self._drag),
            ('<MouseWheel>',self._dolly),('<Double-Button-1>',self.reset),('<KeyPress>',self._key_press),
            ('<KeyRelease>',self._key_release)]:
            self.label.bind(event,handler)
        self.label.bind('<FocusIn>',lambda event:self.request_draw())
        self.label.bind('<FocusOut>',self._focus_out)
        self.label.bind('<Configure>',lambda event:self.request_draw())
        self.label.bind('<Expose>',lambda event:self.request_draw())
        self.label.bind('<Destroy>',self._destroy)
        self.on_render=on_render; self.renderer=None; self.tried_gl=False; self.dirty=True
        self.fitted=False; self.closed=False; self.draw_timer=None; self.poll_timer=None
        self.requests=queue.Queue(maxsize=1); self.results=queue.Queue(maxsize=1)
        self.cpu_thread=None; self.photo=None; self.image_label=None
        self.scene_revision=0; self.last_draw=0.
        self.cloud_source=None
        self.colors=None
        self.empty_view=None
        if getattr(parent.winfo_toplevel(),'beta_visual',False):
            from beta_theme import EmptyViewport,VIEW,LINE,ACCENT
            self.label.configure(background=VIEW,highlightbackground=LINE,highlightcolor=ACCENT)
            self.empty_view=EmptyViewport(self.label)
            self.empty_view.show()

    def _show_empty(self):
        if self.empty_view is None:return False
        if not len(self.points):
            self.empty_view.show();return True
        self.empty_view.hide();return False

    def _focus_out(self,event):
        self._stop_navigation(event)
        self.request_draw()

    def clear(self):
        self._stop_navigation(); self.camera=ViewCamera(); self.fitted=False
        self.points=np.empty((0,3),np.float32); self.nodes=np.empty((0,3),np.float32)
        self.scene_revision+=1
        self.cloud_source=None
        self.colors=None
        self.dirty=True; self.request_draw()

    def reset(self,event=None):
        self._stop_navigation(); self.camera=ViewCamera()
        if len(self.points):
            low,high=self.points.min(axis=0),self.points.max(axis=0)
            center=(low+high)/2
            aspect=max(self.label.winfo_width(),1)/max(self.label.winfo_height(),1)
            distance=max((high[1]-low[1])/2,(high[0]-low[0])/max(2*aspect,.1),.1)/np.tan(np.deg2rad(30))*1.12
            self.camera.position=np.array([center[0],center[1],low[2]-distance])
            self.fitted=True
        if event is not None:
            self.label.focus_set()
        self.request_draw()

    def set_cloud(self,points,nodes=None,colors=None):
        # Backend snapshots are immutable. Repeated pose-only publications should
        # update the trajectory without copying/re-uploading millions of points.
        immutable=isinstance(points,np.ndarray) and not points.flags.writeable
        same=immutable and self.cloud_source is not None and self.cloud_source() is points
        if colors is not None:
            points=np.asarray(points,dtype=np.float32)
            rgb=np.asarray(colors)
            if rgb.shape!=points.shape or rgb.dtype!=np.uint8:raise ValueError('Point colors must be matching N×3 RGB uint8')
            valid=np.isfinite(points).all(axis=1)
            self.colors=np.ascontiguousarray(rgb[valid]);same=False
        elif self.colors is not None:
            self.colors=None;self.dirty=True
        if not same:
            try:self.cloud_source=weakref.ref(points)
            except TypeError:self.cloud_source=None
            points=np.asarray(points,dtype=np.float32)
            valid=np.isfinite(points).all(axis=1)
            self.points=np.ascontiguousarray(points if immutable and valid.all() else points[valid])
            self.dirty=True
        self.scene_revision+=1
        if nodes is not None:
            self.nodes=np.ascontiguousarray(nodes,dtype=np.float32).reshape(-1,3)
        if not self.fitted and len(self.points):
            self.reset()
        self.request_draw()

    def update(self,points):
        # Navigation reuses the existing GPU buffer, with no point-array rebuild.
        if points is not self.points:
            self.set_cloud(points)
        else:
            self.request_draw()

    def request_draw(self):
        if not self.closed and self.draw_timer is None:
            delay=max(1,int((.016-(time.perf_counter()-self.last_draw))*1000))
            self.draw_timer=self.label.after(delay,self._draw)

    def _draw(self):
        self.draw_timer=None
        if self.closed:
            return
        if self._show_empty():return
        width,height=max(self.label.winfo_width(),2),max(self.label.winfo_height(),2)
        if not self.tried_gl:
            self.tried_gl=True
            try:
                self.renderer=_WGLRenderer(self.label)
            except Exception as exc:
                self.fallback_reason=str(exc)
                self._start_cpu()
        if self.renderer:
            before=time.perf_counter()
            if self.colors is None:self.renderer.draw(self.points,self.nodes,self.camera,width,height,self.dirty)
            else:self.renderer.draw(self.points,self.nodes,self.camera,width,height,self.dirty,self.colors)
            self.dirty=False; self._rendered((time.perf_counter()-before)*1000,self.renderer.name)
        else:
            task=(self.points,self.nodes,ViewCamera(self.camera.position,self.camera.yaw,self.camera.pitch,self.camera.roll),width,height,self.scene_revision,self.colors,getattr(self.label.winfo_toplevel(),'viewport_background',(255,255,255)))
            self._latest(self.requests,task)

    @staticmethod
    def _latest(mailbox,value):
        try:
            mailbox.put_nowait(value)
        except queue.Full:
            try: mailbox.get_nowait()
            except queue.Empty: pass
            mailbox.put_nowait(value)

    def _start_cpu(self):
        import tkinter as tk
        # A child label presents only completed frames; projection stays off the UI thread.
        self.image_label=tk.Label(self.label,background='white',takefocus=False)
        self.image_label.place(relx=0,rely=0,relwidth=1,relheight=1)
        for sequence in ('<ButtonPress-1>','<B1-Motion>','<MouseWheel>','<Double-Button-1>'):
            self.image_label.bind(sequence,lambda event,s=sequence:self.label.event_generate(s,x=event.x,y=event.y,delta=event.delta) if s=='<MouseWheel>' else self.label.event_generate(s,x=event.x,y=event.y))
        def worker():
            while not self.closed:
                try: task=self.requests.get(timeout=.2)
                except queue.Empty: continue
                points,nodes,camera,width,height,revision,colors,background=task
                before=time.perf_counter(); result=cpu_map_image(points,nodes,camera,width,height,colors,background)
                self._latest(self.results,(result,(time.perf_counter()-before)*1000,revision))
        self.cpu_thread=threading.Thread(target=worker,daemon=True); self.cpu_thread.start()
        self.poll_timer=self.label.after(15,self._poll_cpu)

    def _poll_cpu(self):
        import tkinter as tk
        if self.closed:
            return
        try:
            result,elapsed,revision=self.results.get_nowait()
            if revision == self.scene_revision:
                h,w=result.shape[:2]
                self.photo=tk.PhotoImage(data=f'P6\n{w} {h}\n255\n'.encode()+result.tobytes(),format='PPM',master=self.label)
                self.image_label.configure(image=self.photo)
                self._rendered(elapsed,'CPU后台绘图')
        except queue.Empty:
            pass
        self.poll_timer=self.label.after(15,self._poll_cpu)

    def _rendered(self,elapsed,name):
        self.last_draw=time.perf_counter()
        x,y,z=self.camera.position
        self.info.set(f'{len(self.points):,} 点 · 绘图 {elapsed:.0f}ms · {name} | 视点 {x:+.2f}, {y:+.2f}, {z:+.2f}m')
        if self.on_render:
            self.on_render(elapsed,len(self.points))

    def _destroy(self,event):
        if event.widget is not self.label:
            return
        self.closed=True; self._stop_navigation()
        for timer in (self.draw_timer,self.poll_timer):
            if timer is not None:
                self.label.after_cancel(timer)
        if self.renderer:
            self.renderer.close()
        if self.cpu_thread and self.cpu_thread is not threading.current_thread():
            # Finish the bounded latest-only worker before Tk variables are released.
            self.cpu_thread.join(timeout=.5)
