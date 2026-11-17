"""Saved camera convention, scene-only UI and GPU lifecycle regression tests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import json,tempfile,unittest
import tkinter as tk
import numpy as np
from reconstruct import ROOT
from gaussian_scene import GaussianScene,saved_cameras
from embedded_brush import EmbeddedBrush
from test_brush_stage import gaussian


def model_fixture(folder):
    path=folder/'gaussians.ply';gaussian(path)
    (folder/'dataset').mkdir()
    pose=np.eye(4);pose[:3,:3]=[[0,0,-1],[-1,0,0],[0,1,0]];pose[:3,3]=[2,3,4]
    (folder/'dataset/transforms.json').write_text(json.dumps(dict(w=640,h=480,fl_x=600,fl_y=590,cx=330,cy=235,
        frames=[dict(file_path='images/one.png',transform_matrix=pose.tolist())])))
    return path,pose


class EmbeddingTests(unittest.TestCase):
    def test_saved_camera_basis_exactly_once_and_off_center_intrinsics(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            path,gl=model_fixture(Path(tmp));c=saved_cameras(path)[0]
            np.testing.assert_allclose(c['pose'],gl@np.diag([1,-1,-1,1]))
            np.testing.assert_allclose(c['intrinsics'],[600,590,330,235])
            point=c['pose'][:3,3]+c['pose'][:3,2]*2
            np.testing.assert_allclose((point-c['pose'][:3,3])@c['pose'][:3,:3],[0,0,2])

    def test_hidden_load_reset_navigation_and_gpu_cleanup(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            path,_=model_fixture(Path(tmp));root=tk.Tk();root.withdraw();view=EmbeddedBrush(root)
            try:
                view.set_model(path,'first');self.assertIsNone(view.error);self.assertIsNone(view.gpu)
                initial=view.pose.copy();view._key(SimpleNamespace(keysym='w'))
                self.assertFalse(np.allclose(initial,view.pose));view.reset();np.testing.assert_allclose(initial,view.pose)
                view._press(SimpleNamespace(x=10,y=10));view._motion(SimpleNamespace(x=30,y=25),1)
                self.assertFalse(np.allclose(initial,view.pose));view.reset();np.testing.assert_allclose(initial,view.pose)
                view.gpu=MagicMock();gpu=view.gpu;view.shutdown();gpu.close.assert_called_once()
                view.clear();self.assertIsNone(view.scene);self.assertIsNone(view.job)
            finally:view.shutdown();root.destroy()

    def test_missing_camera_is_explicit_error_not_arbitrary_viewpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            path=Path(tmp)/'gaussians.ply';gaussian(path);root=tk.Tk();root.withdraw();view=EmbeddedBrush(root)
            try:
                view.set_model(path);self.assertIn('transforms.json',view.error);self.assertIsNone(view.gpu)
            finally:view.shutdown();root.destroy()

    def test_projection_covariance_opacity_sh_and_depth_order(self):
        scene=GaussianScene.__new__(GaussianScene)
        scene.means=np.array([[0,0,2],[0,0,4],[0,0,-1]],np.float32)
        scene.covariance=np.tile(np.diag([.04,.01,.0025]),(3,1,1))
        scene.opacity=np.array([.5,.8,.9]);scene.dc=np.zeros((3,3));scene.rest=None
        xy,conic,op,color,extent=scene.project(np.eye(4),[100,100,40,30],80,60)
        np.testing.assert_allclose(xy,[[40,30],[40,30]])
        np.testing.assert_allclose(op,[.8,.5]);np.testing.assert_allclose(color,.5)
        np.testing.assert_allclose(conic[1],[1/100.3,0,1/25.3],rtol=1e-6)
        scene.rest=np.ones((3,3,3));self.assertTrue(np.all(scene.project(np.eye(4),[100,100,40,30],80,60)[3]>.5))
