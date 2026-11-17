"""Spatial fusion with one centroid per voxel and reversible node contributions."""
import numpy as np


def voxel_groups(points, size):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.empty((0,3),np.int64), np.empty((0,3)), np.empty(0,np.int64)
    keys, inverse, counts = np.unique(np.floor(points/size).astype(np.int64), axis=0,
                                       return_inverse=True, return_counts=True)
    sums = np.column_stack([np.bincount(inverse, weights=points[:,i], minlength=len(keys)) for i in range(3)])
    return keys, sums, counts


class VoxelMap:
    def __init__(self, size=.01):
        if not np.isfinite(size) or size <= 0:
            raise ValueError('voxel size must be positive')
        self.size = size
        self.cells = {}
        self.contributions = {}
        self.poses = {}
        self.cache = np.empty((0,3))
        self.dirty = False

    def remove(self, node):
        contribution = self.contributions.pop(node, None)
        self.poses.pop(node, None)
        if contribution is None:
            return
        keys, sums, counts = contribution
        for key, total, count in zip(keys, sums, counts):
            key = tuple(key)
            previous_sum, previous_count = self.cells[key]
            if previous_count == count:
                del self.cells[key]
            else:
                self.cells[key] = (previous_sum-total, previous_count-int(count))
        self.dirty = True

    def update(self, node, points, pose):
        old_pose = self.poses.get(node)
        if old_pose is not None and np.allclose(old_pose, pose, rtol=0, atol=1e-6):
            return False
        self.remove(node)
        world = points @ pose[:,:3].T + pose[:,3]
        contribution = voxel_groups(world, self.size)
        keys, sums, counts = contribution
        for key, total, count in zip(keys, sums, counts):
            key = tuple(key)
            previous_sum, previous_count = self.cells.get(key, (np.zeros(3), 0))
            self.cells[key] = (previous_sum+total, previous_count+int(count))
        self.contributions[node] = contribution
        self.poses[node] = pose.copy()
        self.dirty = True
        return True

    def cloud(self):
        if self.dirty:
            self.cache = np.asarray([total/count for total,count in self.cells.values()]).reshape(-1,3)
            self.dirty = False
        return self.cache
