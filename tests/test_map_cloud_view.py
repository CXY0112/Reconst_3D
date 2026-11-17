import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from live_cloud_view import ViewCamera
from map_cloud_view import view_matrix,cpu_map_image,MapCloudPanel


class MapViewTests(unittest.TestCase):
    def test_immutable_same_snapshot_does_not_reupload_gpu_points(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw();panel=MapCloudPanel(root)
        points=np.array([[0.,0,1.],[1,0,2]])
        points.flags.writeable=False
        try:
            panel.set_cloud(points);cached=panel.points;panel.dirty=False
            panel.set_cloud(points,np.array([[0.,1,2]]))
            self.assertIs(panel.points,cached);self.assertFalse(panel.dirty)
            np.testing.assert_array_equal(panel.nodes,[[0,1,2]])
        finally:root.destroy()

    def test_opengl_matrix_matches_camera_pose_without_scene_pivot(self):
        camera=ViewCamera([.4,-.2,.3],yaw=.7,pitch=-.3)
        points=np.array([[0,0,2],[.3,.1,1.5]],np.float32)
        expected=camera.project_coordinates(points)*[1,-1,-1]
        actual=np.column_stack((points,np.ones(len(points))))@view_matrix(camera).T
        np.testing.assert_allclose(actual[:,:3],expected,atol=1e-6)

    def test_map_rasterizer_keeps_points_beyond_live_camera_depth_limit(self):
        points=np.array([[0,0,8.]],np.float32)
        image=cpu_map_image(points,np.empty((0,3)),ViewCamera(),100,100)
        np.testing.assert_array_equal(image[50,50],[66,133,223])

    def test_map_navigation_reuses_full_buffer_and_keeps_cloud_fixed(self):
        import tkinter as tk
        root=tk.Tk(); root.withdraw()
        panel=MapCloudPanel(root)
        panel.tried_gl=True
        panel.renderer=Mock(name='renderer')
        panel.renderer.name='test GPU'
        points=np.column_stack((np.linspace(-1,1,15001),np.zeros(15001),np.ones(15001))).astype(np.float32)
        panel.set_cloud(points)
        try:
            panel._draw()
            self.assertTrue(panel.renderer.draw.call_args.args[-1])
            cached=panel.points
            position=panel.camera.position.copy()
            panel._press(SimpleNamespace(x=100,y=100))
            panel._drag(SimpleNamespace(x=140,y=120))
            np.testing.assert_array_equal(panel.camera.position,position)
            panel._draw()
            self.assertFalse(panel.renderer.draw.call_args.args[-1])
            self.assertIs(panel.points,cached)
            self.assertEqual(len(panel.points),15001)
            panel._key_press(SimpleNamespace(keysym='w'))
            self.assertGreater(np.linalg.norm(panel.camera.position-position),0)
            panel._key_release(SimpleNamespace(keysym='w'))
            self.assertIsNone(panel.navigation_timer)
            np.testing.assert_array_equal(panel.points,points)
        finally:
            root.destroy()

    def test_new_map_preserves_selected_camera_pose(self):
        import tkinter as tk
        root=tk.Tk(); root.withdraw()
        panel=MapCloudPanel(root)
        try:
            panel.set_cloud(np.array([[0,0,1],[1,1,2]],np.float32))
            panel.camera.look(30,-20); panel.camera.move({'w','q'},.5)
            position=panel.camera.position.copy(); yaw,pitch=panel.camera.yaw,panel.camera.pitch
            panel.set_cloud(np.array([[0,0,1],[1,1,2],[2,0,3]],np.float32))
            np.testing.assert_array_equal(position,panel.camera.position)
            self.assertEqual((yaw,pitch),(panel.camera.yaw,panel.camera.pitch))
        finally:
            root.destroy()
