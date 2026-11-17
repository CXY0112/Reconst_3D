"""Aspect-preserving SDK BGR preview; all Tk work stays on the UI thread."""
import tkinter as tk
import cv2
import numpy as np


class RGBPreviewPanel:
    def __init__(self, parent):
        self.frame = tk.Frame(parent, background='black', height=240)
        self.frame.pack(fill='both', expand=True)
        self.label = tk.Label(self.frame, background='black', foreground='white',
                              text='等待实时 RGB', borderwidth=0)
        self.label.place(relx=.5, rely=.5, anchor='center')
        self.image = None
        self.photo = None
        self.timer = None
        self.frame.bind('<Configure>', self._resize)
        self.frame.bind('<Destroy>', self._destroy)

    def update(self, bgr):
        image = np.asarray(bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('RGB preview requires an H×W×3 uint8 BGR image')
        self.image = image
        self._draw()

    def clear(self):
        self.image = self.photo = None
        self.label.configure(image='', text='等待实时 RGB')

    def _resize(self, event=None):
        if self.timer is None:
            self.timer = self.frame.after_idle(self._draw)

    def _draw(self):
        if self.timer is not None:
            self.frame.after_cancel(self.timer)
            self.timer = None
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        width, height = self.frame.winfo_width(), self.frame.winfo_height()
        if width <= 1 or height <= 1:
            return
        scale = min(width/w, height/h, 1.)
        size = max(1, int(w*scale)), max(1, int(h*scale))
        image = cv2.resize(self.image, size, interpolation=cv2.INTER_AREA) if size != (w, h) else self.image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = f'P6\n{size[0]} {size[1]}\n255\n'.encode('ascii') + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.label, data=data, format='PPM')
        self.label.configure(image=self.photo, text='')

    def _destroy(self, event):
        if event.widget is self.frame and self.timer is not None:
            self.frame.after_cancel(self.timer)
            self.timer = None
