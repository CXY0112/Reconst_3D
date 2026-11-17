"""Persistent native RTAB-Map session. Every submitted RGB-D frame is processed once."""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import numpy as np
from reconstruct import ROOT
from rtabmap_scan import tool_environment
from voxel_map import VoxelMap, voxel_groups


class OnlineBackend:
    def __init__(self, folder, voxel_size=.01, bridge_dir='online_geometry'):
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
        self.folder = Path(folder)
        self.cookies = []
        for path in (ROOT/'vendor/rtabmap/bin', ROOT/'vendor'/bridge_dir,
                     Path('C:/Program Files (x86)/Windows Kits/10/App Certification Kit')):
            if path.is_dir():
                self.cookies.append(os.add_dll_directory(str(path)))
        self.lib = ct.CDLL(str(ROOT/'vendor'/bridge_dir/'online_bridge.dll'))
        self.lib.online_error.restype = ct.c_char_p
        self.lib.online_create.argtypes = [ct.c_char_p]
        self.lib.online_create.restype = ct.c_void_p
        self.lib.online_process.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                           ct.c_int, ct.c_int, ct.c_void_p, ct.c_double, ct.POINTER(ct.c_int)]
        self.lib.online_process.restype = ct.c_int
        self.lib.online_poses.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
        self.lib.online_poses.restype = ct.c_int
        self.lib.online_close.argtypes = [ct.c_void_p]
        self.lib.online_close.restype = ct.c_int
        self.lib.online_times.argtypes = [ct.c_void_p, ct.c_void_p]
        self.lib.online_times.restype = None
        self.handle = self.lib.online_create(str(self.folder/'map.db').encode('utf-8'))
        if not self.handle:
            raise RuntimeError(self.error())
        self.clouds = {}
        self.frames = 0
        self.voxel_map = VoxelMap(voxel_size)
        self.timings = np.zeros(5)

    def error(self):
        return self.lib.online_error().decode('utf-8', errors='replace')

    def process(self, color, depth, k, stamp):
        if not self.handle:
            raise RuntimeError('Online session is closed')
        if depth.ndim != 2 or color.shape != (*depth.shape, 3):
            raise ValueError('RGB and depth must have matching dimensions')
        if (k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0
                or not np.isfinite(stamp)):
            raise ValueError('Invalid intrinsics or timestamp')
        color = np.ascontiguousarray(color, dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.uint16)
        intrinsics = np.array([k[0,0], k[1,1], k[0,2], k[1,2]], dtype=np.float64)
        node = ct.c_int()
        result = self.lib.online_process(self.handle, color.ctypes.data, depth.ctypes.data,
                                         depth.shape[1], depth.shape[0], intrinsics.ctypes.data, stamp, ct.byref(node))
        self.frames += 1
        self.lib.online_times(self.handle, self.timings.ctypes.data)
        if result < 0:
            raise RuntimeError(self.error())
        if result == 2:
            v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
            z = depth[::4, ::4].astype(float)/1000
            valid = (z >= .15) & (z <= 4)
            points = np.column_stack(((u[valid]-k[0,2])*z[valid]/k[0,0],
                                      (v[valid]-k[1,2])*z[valid]/k[1,1], z[valid]))
            _, sums, counts = voxel_groups(points, self.voxel_map.size)
            self.clouds[node.value] = sums/counts[:,None]
        return result

    def preview(self):
        capacity = max(1024, len(self.clouds)+100)
        poses = np.empty((capacity, 13), dtype=np.float64)
        count = self.lib.online_poses(self.handle, poses.ctypes.data, capacity)
        if count < 0:
            poses = np.empty((-count, 13), dtype=np.float64)
            count = self.lib.online_poses(self.handle, poses.ctypes.data, len(poses))
        if count < 0:
            raise RuntimeError('Cannot retrieve optimized map poses')
        poses = poses[:count]
        present = [row for row in poses if int(row[0]) in self.clouds]
        if not hasattr(self, 'voxel_map'):
            self.voxel_map = VoxelMap(.01)
        present_ids = {int(row[0]) for row in present}
        for node in list(self.voxel_map.poses):
            if node not in present_ids:
                self.voxel_map.remove(node)
        nodes = []
        for row in present:
            transform = row[1:].reshape(3,4)
            points = self.clouds[int(row[0])]
            self.voxel_map.update(int(row[0]), points, transform)
            translation = transform[:,3]
            nodes.append([-translation[1], -translation[2], translation[0]])
        world = self.voxel_map.cloud()
        cloud = np.column_stack((-world[:,1], -world[:,2], world[:,0]))
        return cloud, np.asarray(nodes).reshape(-1,3)

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            result = self.lib.online_close(handle)
            if result:
                raise RuntimeError(self.error())
        for cookie in self.cookies:
            cookie.close()
        self.cookies.clear()


def export_map(folder, voxel_size=.01, decimation=4):
    folder = Path(folder)
    with (folder/'rtabmap_export.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(ROOT/'vendor/rtabmap/bin/rtabmap-export.exe'), '--cloud', '--poses', '--ascii', '--opt', '2',
                        '--voxel', str(voxel_size), '--decimation', str(decimation), '--output', 'map', '--output_dir', str(folder), str(folder/'map.db')],
                       env=tool_environment(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    if not (folder/'map_cloud.ply').exists():
        raise RuntimeError('No exported map cloud')
