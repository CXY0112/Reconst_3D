import tempfile
from pathlib import Path
import unittest
import numpy as np
from reconstruct import Calibration
from rtabmap_scan import register_rgbd, write_configuration


class RGBDTests(unittest.TestCase):
    def test_zero_depth_and_empty_sdk_cloud_preserve_rgb_and_return_zero_depth(self):
        c=Calibration(fxir=2,fyir=2,fxrgb=2,fyrgb=2,R00=1,R11=1,R22=1)
        color=np.full((2,2,3),127,np.uint8)
        for points in (None,np.empty((0,3)),np.array([[0,0,0],[np.nan,0,1000]])):
            with self.subTest(points=points):
                rgb,depth,k=register_rgbd(color,np.zeros((2,2),np.uint16),c,points_ir=points)
                np.testing.assert_array_equal(rgb,color)
                np.testing.assert_array_equal(depth,np.zeros((2,2),np.uint16))
                self.assertEqual(depth.dtype,np.uint16);self.assertEqual(k.shape,(3,3))

    def test_sdk_cloud_supplies_rgb_feature_depth_with_nearest_surface(self):
        c = Calibration(fxir=2,fyir=2,fxrgb=2,fyrgb=2,R00=1,R11=1,R22=1)
        sdk = np.array([[0.,0.,1000.],[0.,0.,500.],[500.,0.,1000.]])
        _,depth,_ = register_rgbd(np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint16),c,points_ir=sdk)
        np.testing.assert_array_equal(depth,np.array([[500,1000],[0,0]],np.uint16))

    def test_identity_registration_preserves_depth(self):
        c = Calibration(fxir=2, fyir=2, fxrgb=2, fyrgb=2, R00=1, R11=1, R22=1)
        depth = np.array([[1000, 2000], [0, 500]], np.uint16)
        color = np.full((2, 2, 3), 127, np.uint8)
        actual_color, actual_depth, _ = register_rgbd(color, depth, c)
        np.testing.assert_array_equal(actual_color, color)
        np.testing.assert_array_equal(actual_depth, depth)

    def test_translation_changes_registered_depth(self):
        c = Calibration(fxir=2, fyir=2, fxrgb=2, fyrgb=2, R00=1, R11=1, R22=1, T3=100)
        _, depth, _ = register_rgbd(np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 1000, np.uint16), c)
        np.testing.assert_array_equal(depth, np.full((2, 2), 1100, np.uint16))

    def test_configuration_uses_custom_calibration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp)
            write_configuration(path, np.eye(3), (480, 640), 10)
            self.assertIn('rgbd\\driver=7', (path/'source.ini').read_text())
            self.assertIn('image_width: 640', (path/'hp60c.yaml').read_text())


if __name__ == '__main__':
    unittest.main()
