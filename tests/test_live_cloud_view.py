import queue
import unittest
import numpy as np
from live_cloud_view import raster_cloud, distance_colors, ViewCamera, LiveCloudPanel
from reconstruct import Calibration, depth_points
from scan_window import ScanSession, ScanWindow


class LiveViewTests(unittest.TestCase):
    def test_camera_translation_moves_view_without_modifying_points(self):
        points = np.array([[0.,0.,2.]],np.float32)
        original = points.copy()
        first = raster_cloud(points,width=100,height=100,intrinsics=(20,20,50,50),enlarge=False)
        second = raster_cloud(points,width=100,height=100,intrinsics=(20,20,50,50),enlarge=False,
                              camera_position=np.array([.5,0,0]))
        self.assertEqual(np.nonzero(first.any(axis=2))[1][0],50)
        self.assertEqual(np.nonzero(second.any(axis=2))[1][0],45)
        np.testing.assert_array_equal(points,original)

    def test_navigation_follows_camera_heading_and_vertical_is_world_up(self):
        camera = ViewCamera(yaw=np.pi/2)
        camera.move({'w'},1,speed=1)
        np.testing.assert_allclose(camera.position,[1,0,0],atol=1e-6)
        camera.move({'d'},1,speed=1)
        camera.move({'q'},1,speed=1)
        np.testing.assert_allclose(camera.position,[1,-1,-1],atol=1e-6)
        camera.move({'e'},1,speed=1)
        np.testing.assert_allclose(camera.position,[1,0,-1],atol=1e-6)

    def test_camera_turn_rotates_about_viewpoint_instead_of_cloud_center(self):
        camera = ViewCamera(yaw=np.pi/2)
        projected = camera.project_coordinates(np.array([[2.,0.,0.],[0.,0.,2.]]))
        np.testing.assert_allclose(projected,[[0,0,2],[-2,0,0]],atol=1e-6)

    def test_diagonal_speed_and_pitch_limit(self):
        camera = ViewCamera()
        camera.move({'w','d'},1,speed=1)
        self.assertAlmostEqual(np.linalg.norm(camera.position),1)
        camera.look(0,-10000)
        self.assertLess(camera.pitch,np.pi/2)

    def test_keyboard_release_and_focus_loss_stop_navigation(self):
        import tkinter as tk
        from types import SimpleNamespace
        root = tk.Tk(); root.withdraw()
        panel = LiveCloudPanel(root)
        points = np.array([[0.,0.,2.]])
        panel.update(points)
        try:
            panel._key_press(SimpleNamespace(keysym='w'))
            self.assertGreater(panel.camera.position[2],0)
            self.assertIsNotNone(panel.navigation_timer)
            panel._key_release(SimpleNamespace(keysym='w'))
            self.assertIsNone(panel.navigation_timer)
            panel._key_press(SimpleNamespace(keysym='a'))
            panel._stop_navigation()
            self.assertFalse(panel.keys)
            self.assertIsNone(panel.navigation_timer)
            np.testing.assert_array_equal(panel.points,points)
        finally:
            root.destroy()

    def test_distance_scale_is_fixed_and_near_red_far_blue(self):
        colors = distance_colors(np.array([.45,1.,4.]))
        self.assertGreater(colors[0,0],colors[0,2])
        self.assertGreater(colors[-1,2],colors[-1,0])
        np.testing.assert_array_equal(colors[1],distance_colors(np.array([1.]))[0])

    def test_projection_keeps_right_and_down_directions(self):
        points = np.array([[-1.,-1.,1.], [1.,1.,2.]])
        image = raster_cloud(points, width=100, height=100,intrinsics=(20,20,50,50),enlarge=False)
        ys, xs = np.nonzero(np.any(image != 0, axis=2))
        self.assertEqual(len(xs), 2)
        self.assertLess(xs[0], xs[1])
        self.assertLess(ys[0], ys[1])
        self.assertGreater(image[ys[0],xs[0],0], image[ys[1],xs[1],0])

    def test_occluded_surface_is_independent_of_input_order(self):
        points = np.array([[0.,0.,1.], [0.,0.,2.]])
        np.testing.assert_array_equal(raster_cloud(points), raster_cloud(points[::-1]))

    def test_cached_rays_follow_changed_calibration(self):
        depth = np.full((2,2),1000,np.uint16)
        first = depth_points(depth,Calibration(fxir=2,fyir=2),stride=1)
        second = depth_points(depth,Calibration(fxir=4,fyir=4),stride=1)
        np.testing.assert_allclose(second[:,:2], first[:,:2]/2)
        np.testing.assert_array_equal(second[:,2], first[:,2])

    def test_live_updates_after_map_without_redrawing_map(self):
        import tkinter as tk
        from unittest.mock import patch
        root = tk.Tk()
        root.withdraw()
        window = ScanWindow(root)
        session = ScanSession(window.events)
        window.session = session
        window.has_map = True
        points = np.array([[0.,0.,1.]])
        session.live_frame = points
        rgb = np.full((2, 2, 3), 100, np.uint8)
        session.live_rgb = rgb
        try:
            with patch.object(window, 'draw') as map_draw, patch.object(window.live_panel,'update') as live_draw, patch.object(window.rgb_panel,'update') as rgb_draw:
                window.poll()
                live_draw.assert_called_once_with(points)
                rgb_draw.assert_called_once_with(rgb)
                map_draw.assert_not_called()
            self.assertIsNone(session.live_frame)
            self.assertIsNone(session.live_rgb)
        finally:
            session.discard()
            root.destroy()


if __name__ == '__main__':
    unittest.main()
