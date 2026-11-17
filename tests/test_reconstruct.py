import tempfile
import unittest
from unittest.mock import patch
import ctypes as ct
from pathlib import Path
import numpy as np
from reconstruct import Calibration, DepthCamera, HP60CRGB, depth_points, save_cloud


class ReconstructionTests(unittest.TestCase):
    def test_calibration_retries_until_valid(self):
        camera = DepthCamera()
        calls = []
        def query(handle, pointer):
            calls.append(1)
            if len(calls) == 1:
                return -1
            calibration = ct.cast(pointer, ct.POINTER(Calibration)).contents
            calibration.fxir = calibration.fyir = 580
            return 0
        camera.api = {'GetCamParameter': query}
        with patch.object(HP60CRGB, '_open'), patch.object(camera.quit, 'wait'):
            camera._open(None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(camera.calibration.fxir, 580)

    def test_calibration_rejects_zero_intrinsics(self):
        camera = DepthCamera()
        camera.api = {'GetCamParameter': lambda *args: 0}
        with patch.object(HP60CRGB, '_open'), patch('reconstruct.time.monotonic', side_effect=[0, 16]):
            with self.assertRaisesRegex(TimeoutError, 'No valid device intrinsics'):
                camera._open(None)
        self.assertIsNone(camera.calibration)

    def test_metric_projection_and_invalid_depth(self):
        c = Calibration(fxir=2, fyir=2, cxir=0, cyir=0)
        depth = np.array([[1000, 2000], [0, 5000]], dtype=np.uint16)
        points = depth_points(depth, c, stride=1)
        np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2]])

    def test_scale_and_stride(self):
        c = Calibration(fxir=2, fyir=2, cxir=0, cyir=0)
        points = depth_points(np.full((3, 3), 500, np.uint16), c, scale_mm=2, stride=2)
        np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    def test_ply_count_and_coordinates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'cloud.ply'
            save_cloud(path, np.array([[1., 2., 3.]]))
            text = path.read_text()
            self.assertIn('element vertex 1\n', text)
            self.assertTrue(text.endswith('1.000000 2.000000 3.000000\n'))


if __name__ == '__main__':
    unittest.main()
