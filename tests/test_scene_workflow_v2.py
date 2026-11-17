"""New default/Beta routing, stage ordering, saved results and dataset export."""
from pathlib import Path
import json,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from scene_reconstruction import ROOT,write_json,read_mesh,digest,run_command
from test_scene_reconstruction import fixture


def ply(path,vertices,faces):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p in vertices:f.write(' '.join(map(str,p))+' 100 140 180\n')
        for face in faces:f.write('3 '+' '.join(map(str,face))+'\n')


def base_fixture(root,scene,legacy=False):
    base=root/('rgbd_scene/setp4/base' if legacy else 'rgbd_scene/setp3/base');out=base/'color/mesh_colored.ply'
    faces=[]
    if len(scene.points_world)==480:
        for y in range(19):
            for x in range(23):
                a=y*24+x;faces.extend([[a,a+1,a+24],[a+1,a+25,a+24]])
    else:faces=[[0,1,2]]
    ply(out,scene.points_world,np.array(faces))
    report=dict(schema='rgbd_scene_base_v1',complete=True,stage=3,method='poisson',
        source_fingerprint=scene.report['source_fingerprint'],model_relative='color/mesh_colored.ply',
        geometry=dict(input_coverage={'10':.9},input_to_mesh_mm_p95=12),appearance=dict(vertex_fraction=.7,display_vertex_fraction=.8))
    write_json(base/'report.json',report)
    return base,report


def wait(root,view,seconds=35):
    until=time.monotonic()+seconds
    while getattr(view,'working',None):
        root.update();time.sleep(.01)
        if time.monotonic()>until:raise AssertionError('UI worker timeout')
    root.update()


class NewWorkflowTests(unittest.TestCase):
    def test_main_default_rgbd_alias_and_beta_use_same_launcher(self):
        import main
        with patch.object(main,'launch') as launch:
            main.main([]);launch.assert_called_with(beta=False)
            main.main(['rgbd']);launch.assert_called_with(beta=False)
            main.main(['beta']);launch.assert_called_with(beta=True)

    def test_defaults_and_acquisition_always_use_isam2_submaps(self):
        import tkinter as tk
        from types import SimpleNamespace
        from preprocess import Parameters
        from scene_workbench import SceneWorkbench
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        try:
            self.assertEqual(list(app.frames),['acquisition','preprocessing','scene_reconstruction','segmentation','review'])
            pre=app.select_stage('preprocessing')
            self.assertEqual(float(pre.fields['output_voxel_mm'].get()),2)
            self.assertEqual(float(pre.fields['surface_spacing_mm'].get()),2)
            self.assertEqual(Parameters().output_voxel_mm,2)
            with patch('video_capture.VideoSession',return_value=SimpleNamespace()) as factory:
                app.views['acquisition'].create_session()
            self.assertEqual(factory.call_args.kwargs['optimizer'],'isam2')
            self.assertEqual(factory.call_args.kwargs['dense_mode'],'submaps')
        finally:app.close()

    def test_legacy_read_five_pages_new_stage_paths_and_showcase_modes(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);scene=fixture(folder);base_fixture(folder,scene,legacy=True)
            root=tk.Tk();root.withdraw();errors=[];root.report_callback_exception=lambda *a:errors.append(a)
            app=SceneWorkbench(root);app.dataset=folder
            try:
                base=app.select_stage('scene_reconstruction');wait(root,base)
                self.assertIsNotNone(base.base);self.assertEqual(base.kind,'base')
                self.assertEqual(base.stage_root(),folder/'rgbd_scene/setp3')
                refinement=app.open_segmentation();wait(root,refinement)
                seg=refinement.segmentation
                self.assertEqual(len(seg.scene.objects()),2);self.assertIsNone(seg.sam.process)
                old=(folder/'rgbd_scene/setp3/labels.npz').read_bytes()
                seg.save();wait(root,seg)
                self.assertTrue((folder/'rgbd_scene/setp4/segmentation/labels.npz').is_file())
                self.assertEqual(old,(folder/'rgbd_scene/setp3/labels.npz').read_bytes())
                seg.next_stage();wait(root,seg);wait(root,refinement.refinement)
                self.assertEqual(len(refinement.refinement.objects),2)
                self.assertEqual(refinement.refinement.kind,'refine')
                review=app.open_review();wait(root,review)
                self.assertIsNotNone(review.final);self.assertIn('10mm',review.metrics.get())
                self.assertEqual(review.left_mode.get(),'点云');self.assertEqual(review.right_mode.get(),'几何结构')
                self.assertEqual(review.left_select['values'],('点云','原始RGB图像'))
                for mode in ('点云','原始RGB图像'):
                    review.left_mode.set(mode);review.show_left();wait(root,review)
                self.assertTrue(review.rgb.image is not None)
                self.assertEqual(errors,[])
            finally:app.close()

    def test_beta_preprocesses_at_1mm_and_finishes_at_review(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        from scene_reconstruction import load_inputs
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);fixture(folder);root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=True);app.dataset=folder
            errors=[];root.report_callback_exception=lambda *a:errors.append(a)
            calls=[]
            def reconstruct(scene,target,method,cancel,progress,**options):
                calls.append((scene.report['input_kind'],method,options));b,r=base_fixture(folder,scene)
                m,c=read_mesh(b/r['model_relative']);return b,r,m,c
            try:
                with patch('scene_reconstruction_window.base_reconstruction',side_effect=reconstruct):
                    app.start_beta();until=time.monotonic()+50
                    while app.auto_active:
                        root.update();time.sleep(.01)
                        self.assertLess(time.monotonic(),until)
                    review=app.views['review'];wait(root,review)
                self.assertEqual(calls,[('stage2','poisson',{'smoothing_iterations':20})])
                report=json.loads((folder/'rgbd_scene/setp2/report.json').read_text(encoding='utf-8'))
                self.assertFalse(report.get('skipped',False))
                self.assertEqual(report['parameters']['output_voxel_mm'],1)
                self.assertEqual(report['parameters']['surface_spacing_mm'],1)
                self.assertEqual(app.tabs.select(),str(app.frames['review']))
                self.assertEqual(app.tabs.tab(app.frames['segmentation'],'state'),'normal')
                self.assertEqual(app.tabs.tab(app.frames['preprocessing'],'state'),'hidden')
                self.assertEqual(errors,[])
            finally:app.close()


    def test_dataset_export_includes_new_branch_and_references(self):
        import zipfile
        from dataset_export import export_dataset
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);dataset=folder/'demo';dataset.mkdir();scene=fixture(dataset);base_fixture(dataset,scene)
            target=export_dataset(dataset,folder/'export.zip')
            with zipfile.ZipFile(target) as archive:
                paths=archive.namelist()
                self.assertIn('demo/rgbd_scene/setp2/processed_world.npy',paths)
                self.assertIn('demo/rgbd_scene/setp3/base/color/mesh_colored.ply',paths)
                self.assertIn('demo/setp1/map_cloud.ply',paths)

    def test_beta_stop_does_not_start_second_writer_and_close_waits(self):
        import tkinter as tk
        import threading
        from scene_workbench import SceneWorkbench
        from scene_cloud_processing import process_scene
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);fixture(folder);root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=True);app.dataset=folder
            entered=threading.Event();release=threading.Event();destroyed=threading.Event();errors=[]
            root.bind('<Destroy>',lambda e:destroyed.set() if e.widget is root else None)
            root.report_callback_exception=lambda *a:errors.append(a)
            def blocked(*args,**kwargs):entered.set();release.wait(10);return process_scene(*args,**kwargs)
            try:
                with patch('scene_cloud_processing.process_scene',side_effect=blocked) as save:
                    app.start_beta();until=time.monotonic()+20
                    while not entered.is_set():
                        root.update();time.sleep(.01);self.assertLess(time.monotonic(),until)
                    app.stop_beta();app.start_beta()
                    self.assertEqual(save.call_count,1);self.assertTrue(app.auto_pending)
                    app.close();self.assertFalse(destroyed.is_set());release.set()
                    until=time.monotonic()+10
                    while not destroyed.is_set():
                        root.update();time.sleep(.01);self.assertLess(time.monotonic(),until)
                self.assertEqual(errors,[])
            finally:
                release.set()
                if not destroyed.is_set():
                    app.close();until=time.monotonic()+20
                    while not destroyed.is_set() and time.monotonic()<until:root.update();time.sleep(.01)


if __name__=='__main__':unittest.main(verbosity=2)
