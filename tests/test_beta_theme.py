"""Presentation isolation, measured-color preservation and viewport lifecycle."""
import tkinter as tk
from tkinter import ttk
import unittest
import numpy as np
from scene_workbench import SceneWorkbench
from map_cloud_view import cpu_map_image
from live_cloud_view import ViewCamera


class BetaThemeTests(unittest.TestCase):
    def test_both_launches_share_theme_but_keep_stage_controls(self):
        for beta in (True,False):
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=beta)
            try:
                panel=app.views['acquisition'].map_panel
                self.assertEqual(ttk.Style(root).theme_use(),'spatial-beta')
                self.assertEqual(root.viewport_background,(11,16,19))
                self.assertIsNotNone(panel.empty_view)
                self.assertEqual(app.views['acquisition'].preprocess_button.master.winfo_manager(),'' if beta else 'pack')
                self.assertEqual(app.tabs.tab(app.frames['preprocessing'],'state'),'hidden' if beta else 'normal')
                for stage in ('segmentation','review'):app.select_stage(stage)
            finally:app.close()

    def test_dark_cpu_background_does_not_recolor_measurements(self):
        points=np.array([[0.,0.,1.],[0.,0.,2.]],np.float32)
        colors=np.array([[231,42,17],[3,222,98]],np.uint8)
        args=(points,np.empty((0,3)),ViewCamera(),100,100,colors)
        light=cpu_map_image(*args);dark=cpu_map_image(*args,background=(11,16,19))
        np.testing.assert_array_equal(light[50,50],colors[0])
        np.testing.assert_array_equal(dark[50,50],colors[0])
        np.testing.assert_array_equal(light[0,0],[255,255,255])
        np.testing.assert_array_equal(dark[0,0],[11,16,19])

    def test_empty_art_leaves_viewport_when_real_points_arrive(self):
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=True)
        try:
            panel=app.views['acquisition'].map_panel
            self.assertTrue(panel._show_empty())
            data=np.array([[0.,0.,1.]],np.float32)
            rgb=np.array([[21,32,43]],np.uint8)
            panel.set_cloud(data,colors=rgb)
            self.assertFalse(panel._show_empty())
            self.assertEqual(panel.empty_view.canvas.winfo_manager(),'')
            np.testing.assert_array_equal(panel.points,data)
            np.testing.assert_array_equal(panel.colors,rgb)
            panel.clear();self.assertTrue(panel._show_empty())
        finally:app.close()


if __name__=='__main__':unittest.main()
