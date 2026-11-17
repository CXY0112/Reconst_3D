import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cloud_import import load_cloud, ROOT


class CloudImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'results', prefix='.test_import_')
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_sdk_units_and_axes_and_invalid_points(self):
        folder = self.folder/'pointcloud_raw'; folder.mkdir()
        path = folder/'raw.npy'
        np.save(path, [[100, 200, 1000], [float('nan'), 0, 1]])
        cloud = load_cloud(path)
        np.testing.assert_allclose(cloud.points, [[.1, .2, 1]])
        self.assertEqual((cloud.unit, cloud.coordinates, cloud.removed), ('mm', 'camera', 1))

    def test_world_transform_and_manual_override(self):
        path = self.folder/'cloud.npy'; np.save(path, [[1, 2, 3]])
        np.testing.assert_allclose(load_cloud(path).points, [[-2, -3, 1]])
        np.testing.assert_allclose(load_cloud(path, 'mm', 'camera').points, [[.001, .002, .003]])

    def test_npz_metadata_and_csv_header(self):
        path = self.folder/'cloud.npz'
        np.savez(path, points_mm=np.array([[100, 200, 300]]))
        np.testing.assert_allclose(load_cloud(path).points, [[.1, .2, .3]])
        path = self.folder/'cloud.csv'
        path.write_text('x,y,z,r\n1,2,3,255\n', encoding='utf-8')
        np.testing.assert_allclose(load_cloud(path).points, [[-2, -3, 1]])

    def test_binary_ply_and_pcd_and_camera_header(self):
        import struct
        path = self.folder/'相机.ply'
        path.write_bytes(b'ply\nformat binary_little_endian 1.0\ncomment XYZ in metres, X right Y down Z forward\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+struct.pack('<ffffff',1,2,3,4,5,6))
        np.testing.assert_allclose(load_cloud(path).points, [[1,2,3], [4,5,6]])
        path = self.folder/'cloud.pcd'
        path.write_text('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n1 2 3\n',encoding='ascii')
        np.testing.assert_allclose(load_cloud(path).points, [[-2,-3,1]])

    def test_rejects_empty_and_object_arrays(self):
        for points in (np.empty((0,3)), np.array([[np.nan,0,1]]), np.array([{'x':1}], dtype=object)):
            path = self.folder/'bad.npy'; np.save(path, points)
            with self.assertRaises(ValueError):
                load_cloud(path)

    def test_gui_background_import_failure_keeps_map_and_busy_guard(self):
        import tkinter as tk
        from snapshot_window import SnapshotWindow
        root = tk.Tk(); root.withdraw()
        window = SnapshotWindow(root)
        points = np.array([[0,0,1],[1,1,2]],np.float32)
        path = self.folder/'points.npy'; np.save(path, points)
        window.import_coordinates.set('相机：X右 Y下 Z前')
        def wait():
            until = time.monotonic()+5
            while window.importing and time.monotonic()<until:
                root.update(); time.sleep(.01)
            self.assertFalse(window.importing)
        try:
            with patch('snapshot_window.SnapshotSession') as session:
                window.import_cloud(path)
                window.start()
                session.assert_not_called()
                wait()
                session.assert_not_called()
            self.assertIsNone(window.session)
            np.testing.assert_allclose(window.map_panel.points, points)
            cached = window.map_panel.points
            with patch('tkinter.messagebox.showerror') as error:
                window.import_cloud(self.folder/'missing.ply'); wait()
                error.assert_called_once()
            self.assertIs(window.map_panel.points, cached)
            window.busy = True
            window.import_cloud(path)
            self.assertFalse(window.importing)
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
