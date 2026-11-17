"""Whole-scene scope, visibility/provenance, simplification and original UI lifecycle."""
from pathlib import Path
import json,tempfile,time,unittest,threading,gc
from unittest.mock import patch
import cv2,numpy as np
from cloud_import import ROOT,load_cloud
from reconstruct import save_cloud
from scene_cloud_processing import colorize,simplify,process_scene,save_scene


def fixture(folder):
    from test_kalman_fusion import write_session
    write_session(folder)
    cal=json.loads((folder/'sdk_calibration.json').read_text());cal.update(fxrgb=100.,fyrgb=100.,cxrgb=32.,cyrgb=24.)
    (folder/'sdk_calibration.json').write_text(json.dumps(cal))
    (folder/'rgb_raw').mkdir();(folder/'depth_raw').mkdir()
    for index in (1,2):
        image=np.full((48,64,3),[20,80,210],np.uint8)
        cv2.imwrite(str(folder/'rgb_raw'/f'{index:.6f}.png'),image)
        cv2.imwrite(str(folder/'depth_raw'/f'{index:.6f}.png'),np.full((48,64),1000+(index-1)*6,np.uint16))
    world=np.vstack([np.load(path)/1000 for path in sorted((folder/'pointcloud_raw').glob('*.npy'))])
    save_cloud(folder/'map_cloud.ply',world,'RTAB-Map world XYZ in metres')
    return load_cloud(folder/'map_cloud.ply')


class SceneCloudTests(unittest.TestCase):
    def setUp(self):gc.collect()

    def test_visibility_uses_known_pixels_and_rejects_occluded_surface(self):
        from kalman_fusion import OPTICAL_FROM_WORLD
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);fixture(folder)
            sdk=np.array([[0,0,1],[0,0,1.1],[2,0,1]],np.float32)
            colors,attr,report=colorize(sdk@OPTICAL_FROM_WORLD.T,folder)
            np.testing.assert_array_equal(colors[0],[210,80,20])
            np.testing.assert_array_equal(colors[1:],[ [160,160,160],[160,160,160]])
            self.assertEqual(attr['source_capture'].tolist(),[1,-1,-1])
            self.assertEqual(attr['source_uv'][0].tolist(),[32,24])

    def test_simplification_reduces_flat_region_and_keeps_normal_edge(self):
        x,y=np.meshgrid(np.linspace(-.08,.08,65),np.linspace(-.08,.08,65))
        points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        normals=np.tile([0.,0,1],(len(points),1));normals[points[:,0]>=0]=[1,0,0]
        ids,report,features=simplify(points,normals,np.ones(len(points)),np.ones(len(points)))
        self.assertLess(len(ids),len(points)*.7)
        self.assertTrue(np.isin(np.flatnonzero(abs(points[:,0])<.006),ids).all())
        np.testing.assert_array_equal(points[ids],points[ids].copy())
        self.assertEqual(int(features.sum()),report['feature_points'])

    def test_all_observations_consolidated_colored_saved_and_source_immutable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            parent=Path(tmp);folder=parent/'scan';folder.mkdir();loaded=fixture(folder)
            before=(folder/'geometry_poses.json').read_bytes()
            result=process_scene(folder,loaded.points)
            self.assertEqual(result.report['captures'],2)
            self.assertLess(len(result.points),len(loaded.points))
            self.assertGreater(result.report['output_points_supported_by_multiple_views'],0)
            self.assertEqual(result.diagnostics['rgb'].shape,result.points.shape)
            saved=save_scene(result,folder/'map_cloud.ply',parent/'out')
            np.testing.assert_allclose(load_cloud(saved/'processed_colored.ply').points,result.points,atol=1e-6)
            with np.load(saved/'surfel_attributes.npz') as attr:
                self.assertEqual(attr['source_uv'].shape,(len(result.points),2))
                np.testing.assert_array_equal(attr['rgb'],result.diagnostics['rgb'])
            self.assertEqual((folder/'geometry_poses.json').read_bytes(),before)

    def test_save_failure_restores_previous_colored_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            parent=Path(tmp);folder=parent/'scan';folder.mkdir();loaded=fixture(folder)
            result=process_scene(folder,loaded.points);target=save_scene(result,folder,parent/'out')
            original=(target/'processed_colored.ply').read_bytes();rename=Path.rename
            def fail_commit(path,new):
                if path.name.startswith('.pending_scene_') and Path(new)==target:raise OSError('injected commit failure')
                return rename(path,new)
            with patch.object(Path,'rename',fail_commit):
                with self.assertRaises(OSError):save_scene(result,folder,target)
            self.assertEqual((target/'processed_colored.ply').read_bytes(),original)
            self.assertFalse(list(parent.glob('.previous_scene_*')))
            with self.assertRaises(ValueError):save_scene(result,folder,parent)

    def test_video_camera_tap_produces_rgb_colors_after_sdk_frame_is_ready(self):
        import tkinter as tk
        from types import SimpleNamespace
        from scene_cloud_window import SceneWorkbench
        from pointmap_processing import xyz_from_depth
        k=np.array([[100.,0,16],[0,100,16],[0,0,1]])
        depth=np.full((32,32),1000,np.uint16);image=np.full((32,32,3),[10,80,220],np.uint8)
        class Camera:
            pair_cloud=None
            count=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,timeout):
                self.count+=1
                if self.count>=2:self.pair_cloud=xyz_from_depth(depth/1000,k).reshape(-1,3)*1000
                cal=SimpleNamespace(fxrgb=100.,fyrgb=100.,cxrgb=16.,cyrgb=16.)
                return ((image,0,1,1000),(depth,0,1,1000)),cal
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        try:
            session=SimpleNamespace()
            with patch('video_capture.VideoSession',return_value=session),patch('rtabmap_scan.RGBDCamera',Camera):
                app.views['acquisition'].create_session()
                with session.camera_factory() as camera:camera.read(2)
            points,colors=session.scene_live
            self.assertEqual(len(points),1024)
            np.testing.assert_array_equal(colors,np.tile([220,80,10],(1024,1)))
        finally:app.close()

    def test_saved_colored_scene_restores_full_pair_and_provenance(self):
        import tkinter as tk
        from scene_cloud_window import SceneWorkbench
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            parent=Path(tmp);folder=parent/'scan';folder.mkdir();loaded=fixture(folder)
            result=process_scene(folder,loaded.points);target=save_scene(result,folder,parent/'out')
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
            try:
                view=app.select_stage('preprocessing');view.load_saved(target,folder)
                deadline=time.monotonic()+12
                while view.working:
                    root.update();time.sleep(.01)
                    if time.monotonic()>deadline:self.fail('restore timeout')
                self.assertEqual(len(view.before.points),len(loaded.points))
                self.assertEqual(len(view.after.points),len(result.points))
                np.testing.assert_array_equal(view.after.colors,result.diagnostics['rgb'])
                np.testing.assert_array_equal(view.result.diagnostics['source_uv'],result.diagnostics['source_uv'])
            finally:app.close()

    def test_ui_has_two_full_scene_panels_and_reuses_video_scanner(self):
        import tkinter as tk
        from scene_cloud_window import SceneWorkbench,SceneVideoWindow
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        try:
            self.assertEqual(len(app.tabs.tabs()),5)
            self.assertIsInstance(app.views['acquisition'],SceneVideoWindow)
            view=app.select_stage('preprocessing')
            self.assertFalse(hasattr(view,'selector'))
            self.assertIn('完整场景',view.before.label.master.cget('text'))
            self.assertIn('完整场景',view.after.label.master.cget('text'))
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
                folder=Path(tmp);loaded=fixture(folder);view.load(folder)
                deadline=time.monotonic()+12
                while view.working:
                    root.update();time.sleep(.01)
                    if time.monotonic()>deadline:self.fail('UI load timeout')
                self.assertEqual(len(view.before.points),len(loaded.points))
                self.assertIsNotNone(view.before.colors)
                view.process()
                while view.working:
                    root.update();time.sleep(.01)
                    if time.monotonic()>deadline:self.fail('UI process timeout')
                self.assertEqual(view.result.report['captures'],2)
                self.assertEqual(len(view.after.points),len(view.result.points))
                self.assertIsNotNone(view.after.colors)
        finally:app.close()


if __name__=='__main__':unittest.main(verbosity=2)
