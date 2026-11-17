"""Shared HP60C calibration, metric depth projection and PLY writing."""
from __future__ import annotations

import ctypes as ct
from pathlib import Path
import re
import sys
import time
from functools import lru_cache

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
from camera_sdk import HP60CRGB


class Calibration(ct.Structure):
    _fields_ = [(n, ct.c_float) for n in (
        'fxir fyir cxir cyir fxrgb fyrgb cxrgb cyrgb '
        'R00 R01 R02 R10 R11 R12 R20 R21 R22 T1 T2 T3 '
        'K1ir K2ir K3ir P1ir P2ir K1rgb K2rgb K3rgb P1rgb P2rgb').split()]


class DepthCamera(HP60CRGB):
    def _bind(self):
        super()._bind()
        symbols = re.findall(rb'[\x20-\x7e]{8,}', (self.dll_dir / 'AngstrongCameraSdk.dll').read_bytes())
        matches = [s.decode('ascii') for s in symbols if s.startswith(b'?AS_SDK_GetCamParameter@@')]
        if len(matches) != 1:
            raise RuntimeError('Cannot resolve SDK calibration export')
        fn = getattr(self.lib, matches[0])
        fn.argtypes = [ct.c_void_p, ct.POINTER(Calibration)]
        fn.restype = ct.c_int
        self.api['GetCamParameter'] = fn

    def _open(self, attr):
        # Never reuse calibration or a cached frame after device reattachment.
        with self.condition:
            self.calibration = None
            self.frame = None
        super()._open(attr)
        # Vendor Camera::backgroundThread also retries this asynchronous query.
        # A -1 immediately after StartStream does not imply stream failure.
        deadline = time.monotonic() + 15
        result = None
        while not self.quit.is_set():
            calibration = Calibration()
            result = self.api['GetCamParameter'](self.handle, ct.byref(calibration))
            values = [getattr(calibration, n) for n, _ in Calibration._fields_]
            if result == 0 and np.isfinite(values).all() and calibration.fxir > 0 and calibration.fyir > 0:
                with self.condition:
                    self.calibration = calibration
                    self.condition.notify_all()
                print('HP60C depth calibration ready.', flush=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f'HP60C calibration not ready after 15s (GetCamParameter={result}). '
                                   'No valid device intrinsics received; point cloud generation stopped.')
            self.quit.wait(.2)

    def _on_frame(self, handle, data, private):
        if self.quit.is_set() or not data:
            return
        try:
            f = data.contents.depth
            if not f.data or not f.width or not f.height or f.size != f.width * f.height * 2:
                return
            depth = np.frombuffer(ct.string_at(f.data, f.size), np.uint16).reshape(f.height, f.width).copy()
            with self.condition:
                self.frame = depth
                self.sequence += 1
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def read_depth(self, timeout=15):
        try:
            depth = self.read(timeout)
        except TimeoutError as exc:
            raise TimeoutError('No new HP60C depth frame. Connect camera and close other camera viewers.') from exc
        deadline = time.monotonic() + timeout
        while getattr(self, 'calibration', None) is None:
            if self.error:
                raise RuntimeError(self.error)
            if time.monotonic() >= deadline:
                raise TimeoutError('No camera calibration')
            time.sleep(.01)
        return depth, self.calibration


@lru_cache(maxsize=8)
def depth_rays(shape, stride, intrinsics, distortion):
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
    pixels = np.column_stack((u.ravel(), v.ravel())).astype(np.float64)
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    return cv2.undistortPoints(pixels[:, None], matrix, np.array(distortion)).reshape(-1, 2)


def depth_points(depth, calibration, scale_mm=1., near=.15, far=4., stride=2, camera='ir'):
    """Return XYZ in metres: X right, Y down, Z forward; reject invalid depths."""
    if camera not in ('ir','rgb'):raise ValueError('Unknown depth camera frame')
    fx, fy, cx, cy = [getattr(calibration, n+camera) for n in ('fx', 'fy', 'cx', 'cy')]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError('Invalid depth intrinsics')
    z = depth[::stride, ::stride].astype(np.float64) * scale_mm / 1000
    valid = np.isfinite(z) & (z > 0) & (z >= near) & (z <= far)
    if not valid.any():
        return np.empty((0, 3))
    distortion = np.array([getattr(calibration, n+camera) for n in ('K1', 'K2', 'P1', 'P2', 'K3')])
    if not np.isfinite(distortion).all():
        raise ValueError('Invalid depth distortion coefficients')
    rays = depth_rays(depth.shape, stride, (fx,fy,cx,cy), tuple(distortion))[valid.ravel()]
    return np.column_stack((rays * z[valid, None], z[valid]))


def save_cloud(path, points, coordinate_comment='XYZ in metres, X right Y down Z forward'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\ncomment {coordinate_comment}\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, points, fmt='%.6f')


