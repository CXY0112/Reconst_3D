"""Label provenance, immutable geometry, transactional saves and stage-3 UI."""
from pathlib import Path
import gc,json,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from cloud_import import ROOT,load_cloud
from scene_segmentation import (load_scene,load_segmentation,save_segmentation,
    SegmentedScene,instance_colors,export_object)


def fixture(folder):
    folder.mkdir(parents=True)
    xyz=np.array([[1,0,0],[1,.01,0],[1,.1,.1],[1,.11,.1]],np.float32)
    rgb=np.array([[220,0,0],[220,0,0],[0,220,0],[0,220,0]],np.uint8)
    np.save(folder/'processed_world.npy',xyz)
    np.savez(folder/'surfel_attributes.npz',rgb=rgb)
    from scene_cloud_processing import write_colored_ply
    write_colored_ply(folder/'processed_colored.ply',xyz,rgb)
    (folder/'report.json').write_text(json.dumps(dict(complete=True,scene_rgb_schema='colored_scene_v1')))
    return load_scene(folder)


def prediction(scene):
    return SegmentedScene(scene.source,scene.points_world,scene.rgb,
        np.array([0,0,1,1],np.int32),np.array([6,6,19,19],np.int32),
        dict(scene.report,backend='official_supercluster_scannet',device='cpu',complete=True))


class SegmentationTests(unittest.TestCase):
    def setUp(self):gc.collect()

    def test_roundtrip_relative_source_and_colored_object_preserve_geometry(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=prediction(fixture(root/'rgbd_scene/setp2'))
            before=(scene.source/'processed_world.npy').read_bytes()
            saved=save_segmentation(scene,root/'rgbd_scene/setp3')
            restored=load_segmentation(saved)
            np.testing.assert_array_equal(restored.points_world,scene.points_world)
            np.testing.assert_array_equal(restored.instance,scene.instance)
            self.assertEqual(len(restored.objects()),2)
            self.assertEqual(export_object(restored,1,root/'object.ply'),2)
            np.testing.assert_allclose(load_cloud(root/'object.ply').points,scene.points[2:])
            self.assertFalse((saved/'processed_world.npy').exists())
            self.assertEqual(before,(scene.source/'processed_world.npy').read_bytes())
            self.assertTrue(json.loads((saved/'report.json').read_text(encoding='utf-8'))['source_relative'].startswith('..'))

    def test_updated_stage2_rejects_old_labels_and_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=prediction(fixture(root/'setp2'))
            target=save_segmentation(scene,root/'setp3')
            np.save(scene.source/'processed_world.npy',scene.points_world+.001)
            with self.assertRaisesRegex(ValueError,'二阶段'):load_segmentation(target)
            with self.assertRaisesRegex(ValueError,'二阶段'):save_segmentation(scene,target)

    def test_transaction_failure_preserves_previous_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=prediction(fixture(root/'setp2'));target=save_segmentation(scene,root/'setp3')
            before=(target/'labels.npz').read_bytes();rename=Path.rename
            def fail(path,new):
                if path.name.startswith('.pending_segmentation_') and Path(new)==target:raise OSError('injected')
                return rename(path,new)
            scene.instance[:]=0
            with patch.object(Path,'rename',fail),self.assertRaises(OSError):save_segmentation(scene,target)
            self.assertEqual((target/'labels.npz').read_bytes(),before)
            with self.assertRaises(ValueError):save_segmentation(scene,scene.source.parent)

    def test_tampered_count_rejected_and_palette_stable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=prediction(fixture(root/'setp2'));target=save_segmentation(scene,root/'setp3')
            np.savez(target/'labels.npz',instance=np.array([0]),semantic=np.array([19]))
            with self.assertRaises(ValueError):load_segmentation(target)
        np.testing.assert_array_equal(instance_colors([7,2,7])[0],instance_colors([7])[0])
        self.assertFalse(np.array_equal(instance_colors([7])[0],instance_colors([2])[0]))


    def test_ui_third_stage_merge_undo_save_restore(self):
        import tkinter as tk
        from scene_cloud_window import SceneWorkbench
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root_dir=Path(tmp);scene=fixture(root_dir/'rgbd_scene/setp2')
            (root_dir/'dataset.json').write_text(json.dumps(dict(schema_version=1,name='test',stages={})))
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
            callbacks=[];root.report_callback_exception=lambda *exc:callbacks.append(exc)
            def wait(view):
                deadline=time.monotonic()+8
                while view.working:
                    root.update();time.sleep(.01)
                    if time.monotonic()>deadline:self.fail('UI job timeout')
                root.update()
            try:
                self.assertEqual(len(app.tabs.tabs()),5)
                app.dataset=root_dir;view=app.select_stage('segmentation').segmentation;view.dataset=root_dir;view.set_source(root_dir/'rgbd_scene/setp2');wait(view)
                self.assertEqual(len(view.scene.rgb),4)
                legacy=save_segmentation(prediction(view.scene),root_dir/'legacy_setp3')
                view.load_result(legacy);wait(view)
                self.assertIsNone(view.sam.process)
                self.assertFalse(hasattr(view,'process_button'))
                self.assertEqual(len(view.tree.get_children()),2)
                view.tree.selection_set(('0','1'));view.select();view.merge()
                self.assertEqual(len(view.scene.objects()),1)
                view.undo();self.assertEqual(len(view.scene.objects()),2)
                view.save();wait(view)
                self.assertTrue((root_dir/'rgbd_scene/setp4/segmentation/labels.npz').is_file())
                view.load_result(root_dir/'rgbd_scene/setp4/segmentation');wait(view)
                self.assertEqual(len(view.scene.objects()),2)
                self.assertIn('原始 RGB',view.after.label.master.cget('text'))
                self.assertEqual(callbacks,[])
            finally:app.close()

    def test_raw_scan_skip_and_source_changes_are_tracked(self):
        from test_scene_cloud import fixture as raw_fixture
        from scene_cloud_processing import save_scene_skip
        from scene_segmentation import fingerprint
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scan=root/'setp1';scan.mkdir();loaded=raw_fixture(scan)
            before=(scan/'map_cloud.ply').read_bytes();skip=root/'rgbd_scene/setp2'
            save_scene_skip(scan,skip,len(loaded.points))
            self.assertEqual([p.name for p in skip.iterdir()],['report.json'])
            raw=load_scene(skip)
            self.assertEqual(raw.report['input_kind'],'stage1')
            self.assertEqual(raw.source,scan.resolve())
            np.testing.assert_allclose(raw.points,loaded.points)
            self.assertEqual(raw.rgb.shape,loaded.points.shape)
            raw.instance=np.zeros(len(raw.rgb),np.int32);raw.semantic=np.full(len(raw.rgb),19,np.int32)
            saved=save_segmentation(raw,root/'rgbd_scene/setp3');restored=load_segmentation(saved)
            np.testing.assert_array_equal(restored.rgb,raw.rgb)
            self.assertEqual(before,(scan/'map_cloud.ply').read_bytes())
            initial=fingerprint(scan)
            import cv2
            cv2.imwrite(str(scan/'rgb_raw/1.000000.png'),np.full((48,64,3),80,np.uint8))
            self.assertNotEqual(initial,fingerprint(scan))
            with self.assertRaises(ValueError):load_segmentation(saved)

    def test_ui_skip_switches_source_and_both_cameras_follow_without_refitting(self):
        import tkinter as tk
        from scene_cloud_window import SceneWorkbench
        from test_scene_cloud import fixture as raw_fixture
        from scene_cloud_processing import process_scene,save_scene
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            parent=Path(tmp);scan=parent/'setp1';scan.mkdir();loaded=raw_fixture(scan)
            saved=save_scene(process_scene(scan,loaded.points),scan,parent/'rgbd_scene/setp2')
            (parent/'dataset.json').write_text(json.dumps(dict(schema_version=1,name='test',stages={})))
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root);app.dataset=parent
            errors=[];root.report_callback_exception=lambda *a:errors.append(a)
            def wait(view):
                deadline=time.monotonic()+15
                while view.working:
                    root.update();time.sleep(.01)
                    if time.monotonic()>deadline:self.fail('skip/restore timeout')
                root.update()
            try:
                segmentation=app.select_stage('segmentation').segmentation;segmentation.set_source(saved);wait(segmentation)
                self.assertEqual(segmentation.scene.report['input_kind'],'stage2')
                preprocessing=app.select_stage('preprocessing');wait(preprocessing)
                self.assertEqual(preprocessing.skip_button.winfo_manager(),'pack')
                preprocessing.skip();wait(preprocessing);wait(app.views['scene_reconstruction']);segmentation.set_source(saved);wait(segmentation)
                self.assertEqual(segmentation.scene.report['input_kind'],'stage1')
                self.assertEqual(len(segmentation.scene.rgb),len(loaded.points))
                self.assertFalse((saved/'processed_world.npy').exists())
                segmentation.scene.instance=np.zeros(len(loaded.points),np.int32)
                segmentation.scene.semantic=np.full(len(loaded.points),19,np.int32)
                segmentation._objects();root.update()
                original=segmentation.before.camera.position.copy()
                segmentation.select()
                np.testing.assert_allclose(segmentation.after.camera.position,original)
                segmentation.after.camera.position+=np.array([.1,.2,.3]);segmentation.after.camera.look(20,10)
                segmentation.after.camera.roll=.3
                time.sleep(.025);root.update()
                self.assertEqual(segmentation._view_states()[0],segmentation._view_states()[1])
                segmentation.before.camera.position+=np.array([-.2,0,0])
                time.sleep(.025);root.update()
                self.assertEqual(segmentation._view_states()[0],segmentation._view_states()[1])
                preprocessing.load_saved(saved,scan);wait(preprocessing)
                self.assertIsNone(preprocessing.result)
                self.assertEqual(len(preprocessing.after.points),len(loaded.points))
                self.assertIn('跳过',preprocessing.status.get())
                preprocessing.process();wait(preprocessing)
                self.assertTrue((saved/'processed_world.npy').is_file())
                wait(app.views['scene_reconstruction']);segmentation.set_source(saved);wait(segmentation)
                self.assertEqual(segmentation.scene.report['input_kind'],'stage2')
                self.assertEqual(errors,[])
            finally:app.close()


if __name__=='__main__':unittest.main(verbosity=2)
