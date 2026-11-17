"""Shared HP60C RGB-D capture, registration and calibration output."""
import ctypes as ct
import os
from pathlib import Path
import time

import cv2
import numpy as np
from reconstruct import Calibration, DepthCamera, ROOT
from rgbd_geometry import camera_frame,ir_to_rgb


class RGBDCamera(DepthCamera):
    # HP60C registered SDK mode: XYZ and depth pixels already use the RGB camera.
    depth_frame = 'rgb'
    pointcloud_frame = 'rgb'

    def __init__(self):
        super().__init__()
        self.rgb = self.depth = None
        self.last_pair = None
        self.capture_native_cloud = True
        self.native_cloud = self.pair_cloud = None
        self._warmup_until = None

    def _open(self, attr):
        with self.condition:
            self._warmup_until = None
            self.rgb = self.depth = self.last_pair = None
            self.native_cloud = self.pair_cloud = None
        super()._open(attr)

    def _on_frame(self, handle, data, private):
        if self.quit.is_set() or not data:
            return
        try:
            now = time.monotonic()
            with self.condition:
                for name, dtype, channels in [('rgb', np.uint8, 3), ('depth', np.uint16, 1)]:
                    f = getattr(data.contents, name)
                    if not f.data or not f.width or not f.height:
                        continue
                    expected = f.width * f.height * channels * np.dtype(dtype).itemsize
                    if f.size != expected:
                        continue
                    shape = (f.height, f.width, 3) if channels == 3 else (f.height, f.width)
                    image = np.frombuffer(ct.string_at(f.data, f.size), dtype).reshape(shape).copy()
                    setattr(self, name, (image, now, int(f.frame_id), int(f.timestamp)))
                if self.rgb is not None and self.depth is not None:
                    if getattr(self, 'capture_native_cloud', False):
                        f = data.contents.pointcloud
                        if f.data and f.size > 0 and f.size % 12 == 0:
                            points = np.frombuffer(ct.string_at(f.data,f.size),np.float32).reshape(-1,3).copy()
                            self.native_cloud = (points,int(f.frame_id),int(f.timestamp))
                    pair = (self.rgb[2:], self.depth[2:])
                    # SDK streams arrive independently. Bound callback arrival skew;
                    # retain raw SDK stamps so hardware synchronization can be audited.
                    if (pair != self.last_pair and abs(self.rgb[1] - self.depth[1]) <= .035
                            and abs(self.rgb[3] - self.depth[3]) <= 20):
                        # Start from actual synchronized stream arrival, not the
                        # asynchronous device-open request. Discard startup pairs
                        # so neither mapping nor its first reference sees them.
                        if self._warmup_until is None:self._warmup_until = now + 1.0
                        self.last_pair = pair
                        if now < self._warmup_until:return
                        self.frame = (self.rgb, self.depth)
                        self.sequence += 1
                        self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def read(self, timeout=20):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (self.frame is None or self.sequence == self.consumed
                   or getattr(self, 'calibration', None) is None):
                if self.error:
                    raise RuntimeError(self.error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No HP60C RGB-D pair/calibration within 20s')
                self.condition.wait(min(remaining, .2))
            self.consumed = self.sequence
            cloud = self.native_cloud
            self.pair_cloud = cloud[0] if cloud is not None and cloud[1:] == self.frame[1][2:] else None
            return self.frame, self.calibration


def matrices(c, suffix):
    k = np.array([[getattr(c, 'fx'+suffix), 0, getattr(c, 'cx'+suffix)],
                  [0, getattr(c, 'fy'+suffix), getattr(c, 'cy'+suffix)], [0, 0, 1.]])
    d = np.array([getattr(c, n+suffix) for n in ('K1', 'K2', 'P1', 'P2', 'K3')])
    if not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid SDK calibration')
    return k, d


def register_rgbd(color, depth, c, points_ir=None, *, points_rgb=None, depth_frame='ir'):
    """Rectified RGB plus RGB-frame depth, mm. Apply IR extrinsic only to IR input."""
    camera_frame(depth_frame)
    if points_ir is not None and points_rgb is not None:
        raise ValueError('Supply only one point-cloud coordinate frame')
    kr, dr = matrices(c, 'rgb')
    h, w = color.shape[:2]
    rgb = cv2.undistort(color, kr, dr)
    points = points_ir if points_ir is not None else points_rgb
    frame = ('ir' if points_ir is not None else 'rgb') if points is not None else depth_frame
    if points is None:
        kd, dd = matrices(c, frame)
        v, u = np.nonzero(depth)
        # OpenCV returns None for an empty input. A covered depth camera is a
        # valid missing observation; pass an empty cloud/zero registered depth
        # downstream so the selected frontend can handle tracking loss.
        if len(v):
            rays = cv2.undistortPoints(np.column_stack((u, v)).astype(float)[:, None], kd, dd).reshape(-1, 2)
            z = depth[v, u].astype(float)
            xyz = np.column_stack((rays * z[:, None], z))
        else:xyz = np.empty((0, 3))
    else:
        xyz = np.asarray(points,dtype=float).reshape(-1,3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)&(xyz[:,2] >= 150)&(xyz[:,2] <= 4000)]
    if frame == 'ir':xyz = ir_to_rgb(xyz/1000,c)*1000
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz[:, 2] <= 65535)
    xyz = xyz[valid]
    pixels = xyz @ kr.T
    pixels = np.rint(pixels[:, :2] / pixels[:, 2, None]).astype(int)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    pixels, xyz = pixels[valid], xyz[valid]
    # Keep the closest surface when several depth pixels project to one RGB pixel.
    target = np.full(h * w, np.inf)
    np.minimum.at(target, pixels[:, 1] * w + pixels[:, 0], xyz[:, 2])
    target[~np.isfinite(target)] = 0
    return rgb, np.rint(target.reshape(h, w)).astype(np.uint16), kr


def write_configuration(folder, k, shape, fps):
    h, w = shape
    def matrix(name, a):
        return f'{name}:\n  rows: {a.shape[0]}\n  cols: {a.shape[1]}\n  data: [{", ".join(str(float(x)) for x in a.flat)}]\n'
    yaml = '%YAML:1.0\n---\ncamera_name: hp60c\n' + f'image_width: {w}\nimage_height: {h}\n'
    yaml += matrix('camera_matrix', k) + matrix('distortion_coefficients', np.zeros((1, 5)))
    yaml += matrix('rectification_matrix', np.eye(3)) + matrix('projection_matrix', np.column_stack((k, np.zeros(3))))
    (folder / 'hp60c.yaml').write_text(yaml, encoding='utf-8')
    path = folder.as_posix()
    config = f'''[Core]
Rtabmap\\WorkingDirectory={path}
Rtabmap\\DetectionRate=2
RGBD\\Enabled=true
RGBD\\LinearUpdate=0
RGBD\\AngularUpdate=0
Mem\\IncrementalMemory=true

[Camera]
type=0
imgRate=0
calibrationName={path}/hp60c.yaml
localTransform=0 0 0 0 0 0
rgbd\\driver=7
RGBDImages\\path_rgb={path}/rgb
RGBDImages\\path_depth={path}/depth
RGBDImages\\scale=1
Images\\filenames_as_stamps=true
Images\\sync_stamps=false
'''
    (folder / 'source.ini').write_text(config, encoding='utf-8')




def tool_environment():
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join((str(ROOT/'vendor/online_geometry'),str(ROOT/'vendor/rtabmap/bin'),env.get('PATH','')))
    # Official Windows package ships qwindows only; -hide suppresses the viewer.
    env['QT_QPA_PLATFORM'] = 'windows'
    env['OMP_WAIT_POLICY'] = 'passive'
    env['OMP_NUM_THREADS'] = '2'
    env['OPENBLAS_NUM_THREADS'] = '1'
    return env
