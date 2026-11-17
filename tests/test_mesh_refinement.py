import unittest
import numpy as np
from mesh_refinement import constrained_taubin, topology


class MeshRefinementTests(unittest.TestCase):
    def grid(self):
        n = 31
        x, y = np.meshgrid(np.arange(n)*.01, np.arange(n)*.01)
        z = np.random.default_rng(4).normal(0, .0015, (n, n))
        vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        faces = []
        for row in range(n-1):
            for col in range(n-1):
                i = row*n+col
                faces.extend(((i, i+1, i+n), (i+1, i+n+1, i+n)))
        return vertices, np.asarray(faces)

    def test_smooths_noisy_plane_with_exact_boundaries_and_motion_bound(self):
        v, f = self.grid(); original = v.copy()
        edges, boundary, feature, angles = topology(v, f)
        result, metrics = constrained_taubin(v, f)
        np.testing.assert_array_equal(v, original)
        np.testing.assert_array_equal(result[boundary], original[boundary])
        self.assertLess(np.std(result[~boundary, 2]), np.std(original[~boundary, 2])*.5)
        self.assertLessEqual(np.linalg.norm(result-original, axis=1).max(), .008000001)
        self.assertLess(metrics['adjacent_face_angle_deg_after_p50_p90'][0], metrics['adjacent_face_angle_deg_before_p50_p90'][0])
        self.assertEqual(metrics['flipped_faces'], 0)

    def test_sharp_edge_is_detected_and_retained(self):
        v = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
        f = np.array([[0, 1, 2], [0, 2, 3]])
        _, _, features, angles = topology(v, f)
        self.assertTrue(features[0] and features[2])
        self.assertAlmostEqual(angles[0], 90.)
        result, metrics = constrained_taubin(v, f)
        np.testing.assert_array_equal(result, v)

    def test_tight_displacement_bound_and_invalid_settings(self):
        v, f = self.grid()
        result, metrics = constrained_taubin(v, f, max_move_mm=.2)
        self.assertLessEqual(np.linalg.norm(result-v, axis=1).max(), .000200001)
        with self.assertRaises(ValueError): constrained_taubin(v, f, iterations=0)
        with self.assertRaises(ValueError): constrained_taubin(v, f, max_move_mm=-1)
