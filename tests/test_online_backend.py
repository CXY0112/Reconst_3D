import ctypes as ct
import unittest
import numpy as np
from online_backend import OnlineBackend


class PreviewTests(unittest.TestCase):
    def backend(self, translation=0):
        backend = OnlineBackend.__new__(OnlineBackend)
        backend.handle = 1
        backend.clouds = {1: np.array([[0., 0., 1.], [.1, .2, 1.]])}
        class FakeLibrary:
            def online_poses(self, handle, pointer, capacity):
                output = np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(ct.c_double)), shape=(capacity*13,))
                # camera optical -> robot world, then world translation along X.
                output[:13] = [1, 0,0,1,translation, -1,0,0,0, 0,-1,0,0]
                return 1
        backend.lib = FakeLibrary()
        return backend

    def test_camera_to_world_transform_preserves_points(self):
        backend = self.backend()
        cloud, nodes = backend.preview()
        self.assertEqual(len(cloud), 2)
        np.testing.assert_allclose(cloud[np.argsort(cloud[:,0])], [[0,0,1],[.1,.2,1]])
        np.testing.assert_allclose(nodes, [[0,0,0]])

    def test_duplicate_points_are_spatially_fused(self):
        backend = self.backend()
        backend.clouds[1] = np.tile([[0.,0.,1.]], (15001,1))
        cloud, _ = backend.preview()
        self.assertEqual(len(cloud), 1)

    def test_distinct_voxels_are_not_globally_capped(self):
        backend = self.backend()
        backend.clouds[1] = np.column_stack((np.arange(15001)*.02, np.zeros(15001), np.ones(15001)))
        cloud, _ = backend.preview()
        self.assertEqual(len(cloud), 15001)

    def test_optimized_pose_changes_cached_cloud_without_reprocessing(self):
        backend = self.backend(2)
        cloud, nodes = backend.preview()
        np.testing.assert_allclose(cloud[np.argsort(cloud[:,0])], [[0,0,3], [.1,.2,3]])
        np.testing.assert_allclose(nodes, [[0,0,2]])

    def test_bad_image_shape_is_rejected_before_native_call(self):
        with self.assertRaises(ValueError):
            self.backend().process(np.zeros((3,3,3),np.uint8), np.zeros((2,2),np.uint16),np.eye(3),1.)


if __name__ == '__main__':
    unittest.main()
