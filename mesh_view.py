"""Embedded indexed triangle rendering with the existing first-person camera."""
import ctypes as ct
import time
import numpy as np
from map_cloud_view import MapCloudPanel, _WGLRenderer, view_matrix


def mesh_buffers(vertices, triangles, normals, colors=None):
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles)
    normals = np.asarray(normals, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape or not np.isfinite(vertices).all() or not np.isfinite(normals).all():
        raise ValueError('Invalid mesh vertices/normals')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu' or not len(triangles) or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Invalid triangle indices')
    display = np.ascontiguousarray(np.column_stack((-vertices[:, 1], -vertices[:, 2], vertices[:, 0])))
    light = np.array([-1., -.3, .7]); light /= np.linalg.norm(light)
    normalized = normals/np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    shade = .28+.58*np.abs(normalized @ light)
    if colors is None:
        colors = np.repeat(shade[:, None], 3, axis=1)
    else:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != vertices.shape or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError('Invalid vertex colors')
    packed = np.ascontiguousarray(np.column_stack((display, colors)), dtype=np.float32)
    return display, np.ascontiguousarray(triangles, dtype=np.uint32), packed


class _MeshRenderer(_WGLRenderer):
    def __init__(self, widget):
        self.index_buffer = ct.c_uint(0)
        super().__init__(widget)
        definitions = {'glColorPointer': ([ct.c_int, ct.c_uint, ct.c_int, ct.c_void_p], None),
                       'glDrawElements': ([ct.c_uint, ct.c_int, ct.c_uint, ct.c_void_p], None),
                       'glPolygonMode': ([ct.c_uint, ct.c_uint], None)}
        for name, (args, restype) in definitions.items():
            fn = getattr(self.gl, name); fn.argtypes = args; fn.restype = restype
        self.gl.glEnableClientState(0x8076)  # GL_COLOR_ARRAY
        if self.buffer.value: self.gen(1, ct.byref(self.index_buffer))

    def draw_mesh(self, packed, triangles, camera, width, height, dirty, wireframe):
        gl = self.gl; gl.wglMakeCurrent(self.dc, self.context)
        focus = self.widget.focus_get() is self.widget
        theme=self.widget.winfo_toplevel()
        gl.glClearColor(*(getattr(theme,'viewport_focus',(.26,.52,.87,1)) if focus else getattr(theme,'viewport_border',(.75,.75,.75,1))))
        gl.glClear(0x4100); gl.glEnable(0x0C11)
        gl.glScissor(2, 2, max(width-4, 1), max(height-4, 1))
        gl.glClearColor(*(v/255 for v in getattr(theme,'viewport_background',(255,255,255))),1); gl.glClear(0x4100); gl.glDisable(0x0C11)
        gl.glViewport(2, 2, max(width-4, 1), max(height-4, 1))
        gl.glMatrixMode(0x1701); gl.glLoadIdentity()
        near = .02; top = near*np.tan(np.deg2rad(60)/2); right = top*width/max(height, 1)
        gl.glFrustum(-right, right, -top, top, near, 1000.)
        gl.glMatrixMode(0x1700)
        matrix = np.ascontiguousarray(view_matrix(camera).T)
        gl.glLoadMatrixf(matrix.ctypes.data_as(ct.POINTER(ct.c_float)))
        if self.buffer.value:
            self.bind(0x8892, self.buffer.value)
            self.bind(0x8893, self.index_buffer.value)
            if dirty:
                self.upload(0x8892, packed.nbytes, packed.ctypes.data, 0x88E4)
                self.upload(0x8893, triangles.nbytes, triangles.ctypes.data, 0x88E4)
            vertex_pointer, color_pointer, index_pointer = None, 12, None
        else:
            vertex_pointer, color_pointer, index_pointer = packed.ctypes.data, packed.ctypes.data+12, triangles.ctypes.data
        gl.glVertexPointer(3, 0x1406, 24, vertex_pointer)
        gl.glColorPointer(3, 0x1406, 24, color_pointer)
        gl.glPolygonMode(0x0408, 0x1B01 if wireframe else 0x1B02)
        gl.glDrawElements(0x0004, int(triangles.size), 0x1405, index_pointer)
        gl.glPolygonMode(0x0408, 0x1B02)
        if self.buffer.value:
            self.bind(0x8892, 0); self.bind(0x8893, 0)
        gl.glFlush(); self.gdi.SwapBuffers(self.dc); gl.glFinish()

    def close(self):
        if self.context and self.index_buffer.value and getattr(self, 'delete', None):
            self.gl.wglMakeCurrent(self.dc, self.context)
            self.delete(1, ct.byref(self.index_buffer)); self.index_buffer.value = 0
        super().close()


class MeshPanel(MapCloudPanel):
    def __init__(self, parent):
        super().__init__(parent)
        self.triangles = np.empty((0, 3), np.uint32)
        self.packed = np.empty((0, 6), np.float32)
        self.wireframe = False
        self.info.set('等待几何重建结果 · 显示真实三角面')

    def set_mesh(self, vertices, triangles, normals, colors=None):
        self.points, self.triangles, self.packed = mesh_buffers(vertices, triangles, normals, colors)
        self.scene_revision += 1; self.dirty = True
        if not self.fitted: self.reset()
        self.request_draw()

    def clear(self):
        self.triangles = np.empty((0, 3), np.uint32); self.packed = np.empty((0, 6), np.float32)
        super().clear()

    def set_wireframe(self, value):
        self.wireframe = bool(value); self.request_draw()

    def _draw(self):
        self.draw_timer = None
        if self.closed or not self.label.winfo_ismapped(): return
        if self._show_empty():return
        width, height = self.label.winfo_width(), self.label.winfo_height()
        if width < 3 or height < 3: return
        started = time.perf_counter()
        try:
            if self.renderer is None: self.renderer = _MeshRenderer(self.label)
            self.renderer.draw_mesh(self.packed, self.triangles, self.camera, width, height, self.dirty, self.wireframe)
            self.dirty = False
            self.last_draw = time.perf_counter()
            self.info.set(f'{len(self.points):,} 顶点 · {len(self.triangles):,} 三角面 · 绘图 {(time.perf_counter()-started)*1000:.0f}ms · {self.renderer.name}')
        except Exception as exc:
            self.info.set(f'网格显示失败：{exc}；已保存的文件仍可用独立网格查看器打开')
