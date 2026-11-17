"""Visibility, conservative association, prompted edits, provenance and UI."""
from pathlib import Path
import copy,json,tempfile,threading,time,unittest
from unittest.mock import patch
import cv2
import numpy as np
from scene_segmentation import ROOT,save_segmentation,load_segmentation
from rgbd_instance_segmentation import project_visible,lift_mask,apply_prompt,prompt_rgbd
from test_scene_segmentation import fixture


def proposal(indices):
    return dict(indices=np.array(indices),capture=1,stamp='1.000000',box=[1.,1.,12.,12.],
                mask=np.ones((16,24),bool),timing=dict(device='cpu'))


class RgbdInstancesTests(unittest.TestCase):
    def test_depth_occlusion_and_mask_shape(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            depth=Path(tmp)/'depth.png';cv2.imwrite(str(depth),np.full((33,33),1000,np.uint16))
            frame=dict(depth=depth,pose=np.eye(4));k=np.array([[100,0,16],[0,100,16],[0,0,1.]])
            points=np.array([[0,0,1],[.01,0,1],[0,0,2],[0,0,-1],[3,0,1]])
            ids,uv,shape=project_visible(points,frame,k)
            np.testing.assert_array_equal(ids,[0,1])
            np.testing.assert_array_equal(lift_mask(np.ones(shape,bool),ids,uv,shape),[0,1])
            with self.assertRaises(ValueError):lift_mask(np.ones((10,10),bool),ids,uv,shape)

    def test_prompt_split_add_remove_and_geometry_immutable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=fixture(root/'setp2')
            scene.instance=np.array([0,1,1,2],np.int32);scene.semantic=np.full(4,19,np.int32)
            scene.report['unassigned_instance_id']=0
            original=scene.instance.copy();before=scene.points_world.copy()
            split,ident=apply_prompt(scene,proposal([1]),'new')
            self.assertEqual(ident,3);np.testing.assert_array_equal(split.instance,[0,3,1,2])
            np.testing.assert_array_equal(scene.instance,original)
            added,_=apply_prompt(split,proposal([2]),'add',3)
            np.testing.assert_array_equal(added.instance,[0,3,3,2])
            removed,_=apply_prompt(added,proposal([1,3]),'remove',3)
            np.testing.assert_array_equal(removed.instance,[0,0,3,2])
            np.testing.assert_array_equal(removed.points_world,before)
            np.testing.assert_array_equal(removed.rgb,scene.rgb)
            self.assertEqual(len(removed.objects()),2)
            target=save_segmentation(removed,root/'setp3')
            restored=load_segmentation(target)
            self.assertEqual(len(restored.report['prompt_edits']),3)
            np.testing.assert_array_equal(restored.instance,removed.instance)
            with self.assertRaises(ValueError):apply_prompt(scene,proposal([1]),'add',0)
            with self.assertRaises(ValueError):apply_prompt(scene,proposal([99]))
            bad=proposal([1]);bad['observation_fingerprint']='new'
            scene.report['rgbd_observation_fingerprint']='old'
            with self.assertRaisesRegex(ValueError,'更新'):apply_prompt(scene,bad)

    def test_first_box_can_create_instances_and_reserve_background(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            scene=fixture(Path(tmp)/'setp2');result,ident=apply_prompt(scene,proposal([1,2]))
            self.assertEqual(ident,1);self.assertEqual(result.report['unassigned_instance_id'],0)
            self.assertEqual(len(result.objects()),1);self.assertIsNone(scene.instance)

    def test_changed_rgbd_invalidates_stage2_based_saved_labels(self):
        from test_scene_cloud import fixture as raw_fixture
        from rgbd_instance_segmentation import observation_fingerprint
        from scene_segmentation import fingerprint,load_scene
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);raw=root/'setp1';raw.mkdir();raw_fixture(raw)
            scene=fixture(root/'rgbd_scene/setp2');p=scene.source/'report.json'
            report=json.loads(p.read_text());report['source_session_relative']='../../setp1'
            p.write_text(json.dumps(report));scene=load_scene(scene.source)
            scene.instance=np.ones(4,np.int32);scene.semantic=np.full(4,19,np.int32)
            scene.report['rgbd_observation_fingerprint']=observation_fingerprint(scene)
            target=save_segmentation(scene,root/'rgbd_scene/setp3')
            self.assertEqual(len(load_segmentation(target).rgb),4)
            before=fingerprint(scene.source)
            cv2.imwrite(str(raw/'rgb_raw/1.000000.png'),np.full((48,64,3),99,np.uint8))
            self.assertEqual(fingerprint(scene.source),before)
            with self.assertRaisesRegex(ValueError,'RGB-D'):load_segmentation(target)
            with self.assertRaisesRegex(ValueError,'RGB-D'):save_segmentation(scene,target)

    def test_prompt_cancel_preserves_previous_prediction(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            scene=fixture(Path(tmp)/'setp2');scene.instance=np.ones(4,np.int32)
            before=scene.instance.copy();cancel=threading.Event();cancel.set()
            from preprocess import ProcessingCancelled
            with self.assertRaises(ProcessingCancelled):prompt_rgbd(scene,None,{},np.eye(3),[1,1,12,12],cancel=cancel)
            np.testing.assert_array_equal(scene.instance,before)

    def test_rgb_letterbox_and_preview_does_not_apply_until_requested(self):
        import tkinter as tk
        from segmentation_window import SegmentationWindow
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);scene=fixture(folder/'setp2');scene.instance=np.array([0,1,1,2],np.int32)
            scene.semantic=np.full(4,19,np.int32);scene.report['unassigned_instance_id']=0
            photo=folder/'rgb.png';cv2.imwrite(str(photo),np.full((16,24,3),128,np.uint8))
            frame=dict(rgb=photo,capture=1);root=tk.Tk();root.withdraw();view=SegmentationWindow(root)
            errors=[];root.report_callback_exception=lambda *a:errors.append(a)
            try:
                self.assertFalse(hasattr(view,'process'))
                self.assertFalse(hasattr(view,'backend_box'))
                view.scene=scene;view.show_scene();view._objects()
                with patch('segmentation_window.source_frames',return_value=([frame],np.eye(3))):view._load_photos()
                self.assertIsNone(view.sam.process)
                # 24x16 image in a 100x100 canvas has top/bottom letterboxing.
                with patch.object(view.rgb_panel.canvas,'winfo_width',return_value=100),patch.object(view.rgb_panel.canvas,'winfo_height',return_value=100):
                    np.testing.assert_allclose(view.rgb_panel.image_point(0,100/6),(0,0),atol=1e-5)
                    self.assertIsNone(view.rgb_panel.image_point(5,0))
                before=view.scene.instance.copy();p=proposal([1])
                with patch('segmentation_window.prompt_rgbd',return_value=p):
                    view.request_box([1,1,12,12])
                    deadline=time.monotonic()+5
                    while view.working:
                        root.update();time.sleep(.01)
                        self.assertLess(time.monotonic(),deadline)
                np.testing.assert_array_equal(view.scene.instance,before)
                self.assertIsNotNone(view.pending_prompt)
                
                with patch('segmentation_window.simpledialog.askstring',return_value='测试物体'):view.apply_box()
                self.assertEqual(view.scene.instance[1],3)
                view.undo();np.testing.assert_array_equal(view.scene.instance,before)
                self.assertNotIn('prompt_edits',view.scene.report)
                view.reset_labels();self.assertEqual(len(view.scene.objects()),0)
                self.assertFalse(view.scene.instance.any())
                saved=save_segmentation(view.scene,folder/'cleared_setp3')
                self.assertEqual(len(load_segmentation(saved).objects()),0)
                view.undo();np.testing.assert_array_equal(view.scene.instance,before)
                view.scene.instance=None;view.scene.semantic=None;view._objects()
                view.pending_prompt=p
                with patch('segmentation_window.simpledialog.askstring',return_value='物体'):view.apply_box()
                self.assertEqual(len(view.scene.objects()),1)
                view.undo();self.assertIsNone(view.scene.instance)
                self.assertEqual(len(view.after.points),0)
                self.assertEqual(errors,[])
            finally:view.shutdown();root.destroy()


if __name__=='__main__':unittest.main(verbosity=2)
