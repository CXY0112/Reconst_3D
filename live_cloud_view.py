"""Independent raster preview of full camera XYZ, without Matplotlib redraws."""
import time
import cv2
import numpy as np

NEAR_METRES, FAR_METRES = .45, 4.
DISTANCE_PALETTE = cv2.applyColorMap(np.arange(255,-1,-1,dtype=np.uint8)[:,None],cv2.COLORMAP_JET)[:,0,::-1].copy()


def distance_colors(depth_metres):
    """Fixed camera-Z colour scale, unchanged by camera rotation or frame bounds."""
    level = (np.clip((np.asarray(depth_metres)-NEAR_METRES)/(FAR_METRES-NEAR_METRES),0,1)*255).astype(np.uint8)
    return DISTANCE_PALETTE[level]


class ViewCamera:
    """Virtual camera pose in IR metres (X right, Y down, Z forward)."""
    def __init__(self, position=None, yaw=0., pitch=0., roll=0.):
        self.position = np.zeros(3) if position is None else np.asarray(position,dtype=float).copy()
        self.yaw, self.pitch = yaw, pitch
        self.roll=roll

    def basis(self):
        cy,sy,cp,sp = np.cos(self.yaw),np.sin(self.yaw),np.cos(self.pitch),np.sin(self.pitch)
        # Columns are the camera's right, down and forward vectors in the fixed cloud.
        base=np.array([[cy,sy*sp,sy*cp],[0,cp,-sp],[-sy,cy*sp,cy*cp]],np.float32)
        cr,sr=np.cos(self.roll),np.sin(self.roll)
        return base@np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]],np.float32)

    def project_coordinates(self, points):
        return (points-self.position.astype(np.float32))@self.basis()

    def look(self, dx, dy):
        self.yaw = (self.yaw+dx*.006+np.pi)%(2*np.pi)-np.pi
        self.pitch = float(np.clip(self.pitch-dy*.006,-np.deg2rad(89),np.deg2rad(89)))

    def move(self, keys, seconds, speed=.75):
        turn=int('c' in keys)-int('z' in keys)
        if turn:self.roll=(self.roll+turn*np.deg2rad(60)*seconds+np.pi)%(2*np.pi)-np.pi
        basis = self.basis()
        direction = (int('d' in keys)-int('a' in keys))*basis[:,0]
        direction = direction+(int('w' in keys)-int('s' in keys))*basis[:,2]
        direction = direction+np.array([0,int('e' in keys)-int('q' in keys),0])
        length = np.linalg.norm(direction)
        if length:
            self.position += direction/length*speed*seconds
        return bool(length or turn)


def raster_cloud(points, yaw=0., pitch=0., scale=1., width=640, height=480, intrinsics=None, enlarge=True, camera_position=None, roll=0.):
    """Same pinhole, reversed JET, black background and 2px dots as HP60C test."""
    image = np.zeros((height, width, 3), np.uint8)
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)&(points[:,2]>=NEAR_METRES)&(points[:,2]<=FAR_METRES)]
    if not len(points):
        return image
    projected = points
    if yaw or pitch or roll or camera_position is not None:
        projected = ViewCamera(camera_position,yaw,pitch,roll).project_coordinates(points)
    positive = projected[:,2] > .01
    points, projected = points[positive], projected[positive]
    fx,fy,cx,cy = intrinsics if intrinsics is not None else (444.6015625,444.7115783691406,322.6365661621094,239.99423217773438)
    view_depth = projected[:,2]
    u = np.rint(fx*scale*projected[:,0]/view_depth+cx).astype(np.int32)
    v = np.rint(fy*scale*projected[:,1]/view_depth+cy).astype(np.int32)
    valid = (u >= 0)&(u < width)&(v >= 0)&(v < height)
    # Per-pixel z buffer preserves the closest surface without sorting all points.
    indices = v[valid]*width+u[valid]
    z_buffer = np.full(height*width, np.inf, np.float32)
    np.minimum.at(z_buffer, indices, view_depth[valid])
    visible = view_depth[valid] <= z_buffer[indices]+1e-6
    colors = distance_colors(points[:,2])
    image[v[valid][visible], u[valid][visible]] = colors[valid][visible]
    if enlarge:
        image = cv2.dilate(image,np.ones((2,2),np.uint8))
    return image


class LiveCloudPanel:
    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.points = np.empty((0, 3))
        self.camera = ViewCamera()
        self.keys = set()
        self.navigation_timer = None
        self.navigation_at = None
        self.last_drag = None
        self.previous_at = None
        self.intrinsics = None
        self.label = tk.Label(parent, background='black',takefocus=True,
                              highlightthickness=2,highlightbackground='#444444',highlightcolor='#4285df')
        self.label.pack(fill='both', expand=True)
        legend = ttk.Frame(parent)
        legend.pack(fill='x',padx=5,pady=5)
        ttk.Label(legend,text='深度 Z：近 0.45m').pack(side='left')
        gradient = np.tile(DISTANCE_PALETTE[None,:,:],(12,1,1))
        self.legend_photo = tk.PhotoImage(data=b'P6\n256 12\n255\n'+gradient.tobytes(),format='PPM',master=parent)
        tk.Label(legend,image=self.legend_photo).pack(side='left',padx=5)
        ttk.Label(legend,text='远 4.00m · 红→黄→绿→蓝').pack(side='left')
        self.info = tk.StringVar(value='等待相机 · 正视图，X向右、Y向下、Z向前')
        ttk.Label(parent, textvariable=self.info).pack(fill='x')
        ttk.Label(parent,text='点击启用视角控制 · 拖动转头 · WASD移动 QE上下 · Z逆/C顺时针滚转 · 滚轮前后 · 双击/R复位',
                  wraplength=650).pack(fill='x')
        self.label.bind('<ButtonPress-1>', self._press)
        self.label.bind('<B1-Motion>', self._drag)
        self.label.bind('<MouseWheel>', self._dolly)
        self.label.bind('<Double-Button-1>', self.reset)
        self.label.bind('<KeyPress>', self._key_press)
        self.label.bind('<KeyRelease>', self._key_release)
        self.label.bind('<FocusOut>', self._stop_navigation)
        self.label.bind('<Destroy>', self._stop_navigation)

    def reset(self, event=None):
        self._stop_navigation()
        self.camera = ViewCamera()
        self.previous_at = None
        if event is not None:
            self.label.focus_set()
        self.update(self.points if event is not None else np.empty((0, 3)))

    def _press(self, event):
        self.label.focus_set()
        self.last_drag = (event.x, event.y)

    def _drag(self, event):
        if self.last_drag:
            self.camera.look(event.x-self.last_drag[0],event.y-self.last_drag[1])
        self.last_drag = (event.x, event.y)
        self.update(self.points)

    def _dolly(self, event):
        self.label.focus_set()
        self.camera.move({'w' if event.delta > 0 else 's'},.12)
        self.update(self.points)

    def _key_press(self, event):
        key = event.keysym.lower()
        if key == 'r':
            self.reset(event)
            return 'break'
        if key not in 'wasdqezc' or len(key) != 1:
            return
        self.keys.add(key)
        if self.navigation_timer is None:
            self.navigation_at = time.perf_counter()-.033
            self._navigation_tick()
        return 'break'

    def _key_release(self, event):
        self.keys.discard(event.keysym.lower())
        if not self.keys:
            self._stop_navigation()
        return 'break'

    def _stop_navigation(self, event=None):
        self.keys.clear()
        self.last_drag = None
        if self.navigation_timer is not None:
            self.label.after_cancel(self.navigation_timer)
            self.navigation_timer = None

    def _navigation_tick(self):
        self.navigation_timer = None
        if not self.keys:
            return
        now = time.perf_counter()
        dt = min(now-self.navigation_at,.1)
        self.navigation_at = now
        if self.camera.move(self.keys,dt):
            self.update(self.points)
        self.navigation_timer = self.label.after(getattr(self,'navigation_interval',33),self._navigation_tick)

    def update(self, points):
        import tkinter as tk
        before = time.perf_counter()
        self.points = points
        image = raster_cloud(points,self.camera.yaw,self.camera.pitch,intrinsics=self.intrinsics,
                             camera_position=self.camera.position,roll=self.camera.roll)
        cx,cy = self.intrinsics[2:] if self.intrinsics is not None else (322.636566,239.994232)
        cv2.drawMarker(image,(int(cx),int(cy)),(255,255,255),cv2.MARKER_CROSS,20,2)
        # Tk consumes PPM directly, so Pillow/PNG compression aren't on the live path.
        width, height = self.label.winfo_width(), self.label.winfo_height()
        if width > 1 and height > 1:
            scale = min(width/640, height/480, 1.)
            if scale < 1:
                image = cv2.resize(image, (max(1, int(640*scale)), max(1, int(480*scale))), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        data = f'P6\n{w} {h}\n255\n'.encode('ascii')+image.tobytes()
        self.photo = tk.PhotoImage(data=data, format='PPM', master=self.label)
        self.label.configure(image=self.photo)
        now = time.perf_counter()
        fps = 0 if self.previous_at is None else 1/max(now-self.previous_at, 1e-6)
        self.previous_at = now
        x,y,z = self.camera.position
        self.info.set(f'{len(points):,} 点 · {fps:.1f} FPS · 绘图 {(now-before)*1000:.0f}ms | '
                      f'虚拟视点 X {x:+.2f} Y {y:+.2f} Z {z:+.2f} m')
