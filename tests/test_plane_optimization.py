import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from plane_optimization import fit_patch, optimize_vertices
from reconstruct import ROOT


class PlaneOptimizationTests(unittest.TestCase):
    def grid(self):
        n = 41
        x, y = np.meshgrid(np.linspace(-.2, .2, n), np.linspace(-.2, .2, n))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        normals = np.tile([0., 0., 1.], (len(points), 1))
        faces = []
        for row in range(n-1):
            for col in range(n-1):
                i = row*n+col
                faces.extend(((i, i+1, i+n), (i+1, i+n+1, i+n)))
        return points, normals, np.array(faces)

    def fit(self, points, normals):
        return fit_patch(points, normals, np.zeros(3), np.eye(3), np.array([.4, .4, .04]))

    def test_accepts_wavy_plane_and_reduces_residual_without_topology_changes(self):
        points, normals, faces = self.grid()
        points[:, 2] = .004*np.sin(points[:, 0]*np.pi*30)*np.cos(points[:, 1]*np.pi*20)
        original = points.copy(); original_faces = faces.copy()
        patch, reason = self.fit(points, normals)
        self.assertIsNone(reason)
        result, labels, metrics = optimize_vertices(points, faces, normals, [patch], original)
        interior = (np.abs(points[:, :2]) < .14).all(axis=1)
        self.assertLess(np.std(result[interior, 2]), np.std(points[interior, 2])*.4)
        self.assertLessEqual(metrics['movement_mm_p50_p90_p99_max'][-1], 8.000001)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(faces, original_faces)
        self.assertEqual(metrics['flipped_faces'], 0)

    def test_coherent_curvature_is_rejected(self):
        points, normals, _ = self.grid()
        points[:, 2] = .5*points[:, 0]**2
        patch, reason = self.fit(points, normals)
        self.assertIsNone(patch); self.assertEqual(reason, 'coherent curvature')

    def test_parallel_layers_are_rejected(self):
        points, normals, _ = self.grid()
        above = points.copy(); below = points.copy()
        above[:, 2] = .006; below[:, 2] = -.006
        patch, reason = self.fit(np.vstack((above, below)), np.vstack((normals, normals)))
        self.assertIsNone(patch); self.assertEqual(reason, 'parallel layers')

    def test_nearby_nonmember_object_is_not_flattened(self):
        points, normals, faces = self.grid()
        patch, reason = self.fit(points, normals)
        patch['global_ids'] = np.arange(len(points))
        clutter = np.array([[0., 0., .018], [.01, 0., .018], [0., .01, .018]])
        vertices = np.vstack((points, clutter))
        mesh_normals = np.vstack((normals, np.tile([0., 0., 1.], (3, 1))))
        faces = np.vstack((faces, np.array([[0, 1, 2]])+len(points)))
        result, labels, metrics = optimize_vertices(vertices, faces, mesh_normals, [patch], vertices, observations=vertices)
        np.testing.assert_array_equal(result[-3:], clutter)
        np.testing.assert_array_equal(labels[-3:], [-1, -1, -1])

    def test_cumulative_budget_is_enforced(self):
        points, normals, faces = self.grid()
        target = points.copy(); target[:, 2] = .025
        patch, reason = fit_patch(target, normals, np.array([0., 0., .025]), np.eye(3), np.array([.4, .4, .04]))
        vertices = points.copy(); vertices[:, 2] = .012
        result, labels, metrics = optimize_vertices(vertices, faces, normals, [patch], points)
        self.assertLessEqual(np.linalg.norm(result-points, axis=1).max(), .015000001)
        self.assertGreater(metrics['moved_vertices'], 0)
        self.assertLessEqual(metrics['newly_moved_total_anchor_distance_max_mm'], 15.000001)

    def test_no_patch_leaves_geometry_unchanged(self):
        points, normals, faces = self.grid()
        result, labels, metrics = optimize_vertices(points, faces, normals, [], points)
        np.testing.assert_array_equal(result, points)
        self.assertTrue((labels == -1).all())
        self.assertEqual(metrics['moved_vertices'], 0)

