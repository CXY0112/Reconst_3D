"""Photo browsing, names, live placement and non-submitting FAL polling."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,time,unittest,tkinter as tk
import numpy as np
from scene_segmentation import ROOT,save_segmentation,load_segmentation
from test_scene_segmentation import fixture
from test_rgbd_instances import proposal


class Interactions(unittest.TestCase):
    def test_removed_buttons_and_beta_smoothing_profile(self):
        from scene_workbench import SceneWorkbench
        for beta in (False,True):
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=beta)
            try:
                base=app.select_stage('scene_reconstruction')
                self.assertEqual(base.show_base_button.winfo_manager(),'')
                self.assertEqual(getattr(base,'smoothing_iterations',30),20 if beta else 30)
                refine=app.select_stage('segmentation').refinement
                self.assertFalse(hasattr(refine,'align_button'));self.assertFalse(hasattr(refine,'confirm'))
                self.assertEqual(refine.fetch_button.winfo_manager(),'')
            finally:app.close()

    def test_beta_smoothing_parameter_reaches_geometry_worker(self):
        from scene_reconstruction import base_reconstruction,write_json
        from test_scene_reconstruction import fixture as full_fixture
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);scene=full_fixture(folder);commands=[]
            def worker(command,*args):
                commands.append(list(map(str,command)));out=Path(command[command.index('--output')+1]);(out/'geometry').mkdir()
                (out/'geometry/mesh.ply').write_bytes(b'preview fixture')
                write_json(out/'report.json',dict(schema='rgbd_scene_base_v1',complete=True))
            mesh=(np.zeros((3,3)),np.array([[0,1,2]]),np.ones((3,3)))
            with patch('scene_reconstruction.run_command',side_effect=worker),patch('scene_reconstruction.read_mesh',return_value=(mesh,None)):
                base_reconstruction(scene,folder/'output',smoothing_iterations=20)
            self.assertEqual(commands[0][-2:],['--smoothing-iterations','20'])

    def test_names_persist_and_photo_wheel_is_bounded(self):
        from segmentation_window import SegmentationWindow
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=tk.Tk();root.withdraw();view=SegmentationWindow(root)
            try:
                view.scene=fixture(Path(tmp)/'setp2');view.pending_prompt=proposal([0,1])
                with patch('segmentation_window.simpledialog.askstring',return_value='键盘'):view.apply_box()
                self.assertEqual(view.scene.objects()[0]['name'],'键盘')
                saved=save_segmentation(view.scene,Path(tmp)/'seg')
                self.assertEqual(load_segmentation(saved).objects()[0]['name'],'键盘')
                self.assertEqual(view.tree['columns'],('name','points'))
                view.frames=[{},{}];view.photo_box['values']=['one','two'];view.photo_box.current(0)
                with patch.object(view,'change_photo') as change:
                    view.photo_wheel(SimpleNamespace(delta=-120));self.assertEqual(view.photo_box.current(),1)
                    view.photo_wheel(SimpleNamespace(delta=-120));self.assertEqual(view.photo_box.current(),1)
                    view.photo_wheel(SimpleNamespace(delta=120));self.assertEqual(view.photo_box.current(),0)
                    self.assertEqual(change.call_count,3)
            finally:view.shutdown();root.destroy()

    def test_object_modeling_keeps_neutral_scene_and_highlights_selection(self):
        from scene_reconstruction_window import SceneReconstructionWindow
        from map_cloud_view import MapCloudPanel
        from test_scene_reconstruction import fixture as full_fixture
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=tk.Tk();root.withdraw();view=SceneReconstructionWindow(root,kind='refine')
            try:
                view.scene=full_fixture(Path(tmp));original=view.scene.rgb.copy()
                view.objects={'selected':1};view.object.set('selected');view.show_scene_cloud()
                self.assertIsInstance(view.before,MapCloudPanel)
                selected=view.scene.instance==1
                self.assertTrue((view.before.colors[selected]==[184,238,213]).all())
                self.assertTrue((view.before.colors[~selected]==160).all())
                np.testing.assert_array_equal(view.scene.rgb,original)
                self.assertFalse(hasattr(view,'confirm'));self.assertFalse(hasattr(view,'placement_row'))
            finally:view.shutdown();root.destroy()

    def test_submit_event_queries_and_download_remains_independent(self):
        from scene_reconstruction_window import SceneReconstructionWindow
        from test_scene_reconstruction import fixture as full_fixture
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=tk.Tk();root.withdraw();view=SceneReconstructionWindow(root,kind='refine');view.scene=full_fixture(Path(tmp))
            from scene_reconstruction import segmentation_digest
            try:
                view.run=Path(tmp);view.events.put(('submit',dict(request_id='existing-id',status='submitted')))
                with patch.object(view,'schedule_fetch') as schedule:
                    root.after_cancel(view.timer);view.timer=None
                    view._poll();self.assertTrue(view.auto_fetch);schedule.assert_called()
                vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]],np.float32);faces=np.array([[0,1,2]],np.uint32);normals=np.tile([0,0,1.],(3,1))
                state=dict(status='downloaded',segmentation_fingerprint=segmentation_digest(view.scene))
                view.events.put(('fetch',(state,((vertices,faces,normals),None))))
                root.after_cancel(view.timer);view.timer=None
                view._poll();self.assertFalse(view.auto_fetch)
                self.assertFalse(hasattr(view,'confirm'))
                self.assertEqual(view.current_model,Path(tmp)/'generated.glb')
                self.assertEqual(view.mesh[0].shape,(3,3))
            finally:view.shutdown();root.destroy()


if __name__=='__main__':unittest.main()
