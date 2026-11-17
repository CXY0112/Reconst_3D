import unittest
import numpy as np
from reconstruct import Calibration
from snapshot_window import native_geometry, ir_geometry, ir_to_rgb
from geometry_snapshot_backend import SnapshotBackend


class GeometryTests(unittest.TestCase):
    def test_sdk_xyz_is_used_instead_of_raw16_geometry(self):
        c = Calibration(fxir=2,fyir=2,R00=1,R11=1,R22=1,T1=100)
        sdk = np.array([[200.,100.,1000.],[0,0,0],[0,0,np.nan]])
        points = ir_geometry(np.zeros((2,2),np.uint16),c,sdk)
        np.testing.assert_allclose(points,[[.2,.1,1.]])
        np.testing.assert_allclose(ir_to_rgb(points,c),[[.3,.1,1.]])

    def test_live_preview_does_not_use_rgb_crop_or_stride_four(self):
        c = Calibration(fxir=2,fyir=2,R00=1,R11=1,R22=1)
        points = ir_geometry(np.full((4,4),1000,np.uint16),c)
        self.assertEqual(len(points),16)
        self.assertGreater(points[:,0].max(),1.)

    def test_full_native_depth_keeps_all_valid_pixels(self):
        c = Calibration(fxir=2,fyir=2,R00=1,R11=1,R22=1)
        depth = np.full((4,4),1000,np.uint16)
        points = native_geometry(depth,c)
        self.assertEqual(len(points),16)
        np.testing.assert_allclose(points[-1],[1.5,1.5,1.])

    def test_native_geometry_is_not_cropped_to_rgb_view(self):
        c = Calibration(fxir=2,fyir=2,R00=1,R11=1,R22=1,T1=100)
        points = native_geometry(np.full((4,4),1000,np.uint16),c)
        self.assertEqual(len(points),16)
        np.testing.assert_allclose(points[0],[.1,0,1])

    def test_registered_fallback_has_no_pixel_stride(self):
        depth = np.full((4,4),1000,np.uint16)
        k = np.array([[2.,0,0],[0,2.,0],[0,0,1.]])
        points = SnapshotBackend.full_points(depth,k)
        self.assertEqual(len(points),16)


if __name__ == '__main__':
    unittest.main()
