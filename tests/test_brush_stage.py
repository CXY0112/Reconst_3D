"""Gaussian result lifecycle, native-process cancellation and stage UI routing."""
from pathlib import Path
from types import SimpleNamespace
import json,shutil,sys,tempfile,threading,time,unittest
from unittest.mock import patch
import cv2,numpy as np
from reconstruct import ROOT,Calibration,save_cloud
from scene_reconstruction import base_reconstruction,write_json,digest,load_inputs
from brush_reconstruction import DEFAULTS,LABEL,METHOD,splat_count,validate_result,train
from preprocess import ProcessingCancelled,Parameters
from scene_stages import MESH_VIEW,SPLAT_VIEW,result_target,bases_for


def gaussian(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    props=['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    header='ply\nformat binary_little_endian 1.0\nelement vertex 1\n'+''.join(f'property float {n}\n' for n in props)+'end_header\n'
    path.write_bytes(header.encode()+np.zeros(len(props),dtype='<f4').tobytes())


def fixture(parent):
    from dataset_store import create_dataset
    from local_preprocessing import prepare_local,save_local
    from rgbd_geometry import coordinate_metadata
    from kalman_fusion import OPTICAL_FROM_WORLD
    folder=create_dataset(parent,'brush_case');raw=folder/'setp1';raw.mkdir()
    for name in ('rgb','rgb_raw','depth','depth_raw','map_frames','pointcloud_local'):(raw/name).mkdir()
    c=Calibration(fxir=100,fyir=100,cxir=32,cyir=24,fxrgb=100,fyrgb=100,cxrgb=32,cyrgb=24,R00=1,R11=1,R22=1)
    write_json(raw/'sdk_calibration.json',{n:getattr(c,n) for n,_ in c._fields_})
    v,u=np.mgrid[:48,:64];points=np.c_[(u.ravel()-32)/100,(v.ravel()-24)/100,np.ones(u.size)]
    pose=np.eye(4);pose[:3,:3]=OPTICAL_FROM_WORLD.T
    prepared=prepare_local(points,Parameters(),coordinate_frame='rgb');frames=[];regs=[]
    rgb=np.dstack((u*3,v*4,np.full_like(u,90))).astype(np.uint8);depth=np.full((48,64),1000,np.uint16)
    for i in (1,2,3):
        name=f'{i:.6f}.png'
        for directory,image in (('rgb',rgb),('rgb_raw',rgb),('depth',depth),('depth_raw',depth)):
            cv2.imwrite(str(raw/directory/name),image)
        np.save(raw/'map_frames'/f'{i}.npy',points)
        save_local(raw/'pointcloud_local'/f'{i:.6f}.npz',prepared)
        frames.append(dict(capture=i,stamp=float(i),cloud_source='local_surface_v1',**coordinate_metadata('rgb','rgb')))
        regs.append(dict(capture=i,node=i,accepted=True))
    (raw/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
    (raw/'registrations.jsonl').write_text('\n'.join(map(json.dumps,regs)))
    write_json(raw/'geometry_poses.json',dict(node_ids=[1,2,3],poses=[pose.tolist()]*3))
    world=points@pose[:3,:3].T;save_cloud(raw/'map_cloud.ply',world,'RTAB-Map world XYZ in metres')
    source=folder/'rgbd_scene/setp2';source.mkdir(parents=True)
    np.save(source/'processed_world.npy',world.astype(np.float32))
    np.savez(source/'surfel_attributes.npz',rgb=rgb.reshape(-1,3)[:,::-1],normals_world=np.tile([1.,0,0],(len(points),1)))
    write_json(source/'report.json',dict(complete=True,scene_rgb_schema='colored_scene_v1',source_session_relative='../../setp1',source_session='Z:/old_machine/setp1'))
    return folder,load_inputs(source)[0]


def fake_train(dataset,output,cancel=None,progress=None,settings=None):
    output=Path(output);output.mkdir();settings={**DEFAULTS,**(settings or {})}
    gaussian(output/f'export_{settings["steps"]}.ply')
    evaluation=output/f'eval_{settings["steps"]}';evaluation.mkdir()
    transforms=json.loads((Path(dataset)/'transforms.json').read_text())
    for i,frame in enumerate(transforms['frames']):
        if i%settings['eval_split_every']==0:shutil.copy2(Path(dataset)/frame['file_path'],evaluation/Path(frame['file_path']).name)
    if progress:progress(dict(text='test checkpoint',percent=85))
    return dict(parameters=settings,gaussians=1,elapsed_seconds=.01)


def wait(root,view):
    deadline=time.monotonic()+20
    while view.working:
        root.update();time.sleep(.01)
        if time.monotonic()>deadline:raise AssertionError('UI task did not finish')
    root.update()


class BrushStageTests(unittest.TestCase):
    def test_transactional_result_and_portable_relative_session(self):
        from scene_stages import base_model,geometry_model
        from dataset_export import export_dataset
        import zipfile
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder,scene=fixture(Path(tmp));target=folder/'rgbd_scene/setp3/base'
            with patch('brush_reconstruction.train',side_effect=fake_train):
                path,report,mesh,colors=base_reconstruction(scene,target,METHOD)
            self.assertIsNone(mesh);self.assertIsNone(colors);self.assertEqual(report['eval_views'],1)
            self.assertEqual(report['parameters']['steps'],8000);self.assertEqual(splat_count(base_model(path,report)),1)
            with self.assertRaises(ValueError):geometry_model(path,report)
            zip_path=export_dataset(folder,Path(tmp)/'dataset.zip')
            with zipfile.ZipFile(zip_path) as archive:self.assertIn(folder.name+'/rgbd_scene/setp3/base/gaussians.ply',archive.namelist())
            folder.rename(Path(tmp)/'moved');moved=Path(tmp)/'moved'
            from brush_reconstruction import session_for
            self.assertEqual(session_for(moved/'rgbd_scene/setp2'),moved/'setp1')
            validate_result(moved/'rgbd_scene/setp3/base',report)

    def test_cancel_or_changed_input_preserves_previous_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder,scene=fixture(Path(tmp));target=folder/'rgbd_scene/setp3/base';target.mkdir(parents=True)
            (target/'old.ply').write_text('existing mesh')
            write_json(target/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,representation='gaussian_splats',model_relative='old.ply'))
            original=digest(target/'report.json')
            def cancelled(*args,**kwargs):raise ProcessingCancelled()
            with patch('brush_reconstruction.train',side_effect=cancelled):
                with self.assertRaises(ProcessingCancelled):base_reconstruction(scene,target,METHOD)
            self.assertEqual(digest(target/'report.json'),original)
            def changed(*args,**kwargs):
                result=fake_train(*args,**kwargs)
                with (folder/'setp1/rgb/1.000000.png').open('ab') as stream:stream.write(b'changed input')
                return result
            with patch('brush_reconstruction.train',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'发生变化'):base_reconstruction(scene,target,METHOD)
            self.assertEqual(digest(target/'report.json'),original)
            self.assertFalse(list(target.parent.glob('.pending_base_*')))

    def test_cancellation_terminates_only_owned_native_process(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);dataset=root/'dataset';dataset.mkdir()
            (dataset/'transforms.json').write_text(json.dumps(dict(frames=[{}, {}, {}])))
            (dataset/'__main__.py').write_text('import time\ntime.sleep(60)\n')
            event=threading.Event();timer=threading.Timer(.5,event.set);created=[];original=subprocess.Popen
            def launch(*args,**kwargs):created.append(original(*args,**kwargs));return created[-1]
            try:
                timer.start()
                with patch('brush_reconstruction.executable',return_value=Path(sys.executable)),patch('brush_reconstruction.subprocess.Popen',side_effect=launch):
                    with self.assertRaises(ProcessingCancelled):train(dataset,root/'output',event)
                self.assertEqual(len(created),1);self.assertIsNotNone(created[0].poll())
            finally:timer.cancel()

    def test_model_hash_path_and_truncation_guards(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);model=root/'gaussians.ply';gaussian(model)
            self.assertEqual(splat_count(model),1)
            model.write_bytes(model.read_bytes()[:-4])
            with self.assertRaises(ValueError):splat_count(model)

    def test_cross_representation_overwrite_is_rejected_and_legacy_stays_readable(self):
        from scene_stages import base_for
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder,scene=fixture(Path(tmp));legacy=folder/'rgbd_scene/setp3/base';legacy.mkdir(parents=True)
            (legacy/'mesh.ply').write_text('old geometry')
            write_json(legacy/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,method='poisson',model_relative='mesh.ply'))
            before=digest(legacy/'report.json')
            with self.assertRaisesRegex(ValueError,'不同目录'):base_reconstruction(scene,legacy,METHOD)
            target=result_target(legacy.parent,'gaussian')
            with patch('brush_reconstruction.train',side_effect=fake_train):base_reconstruction(scene,target,METHOD)
            self.assertEqual(digest(legacy/'report.json'),before)
            self.assertEqual(bases_for(folder),dict(mesh=legacy,gaussian=target))
            self.assertEqual(base_for(folder),legacy)

    def test_real_ui_training_reload_export_review_and_object_tab(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        from scene_refinement_window import SceneRefinementWindow
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder,scene=fixture(Path(tmp));root=tk.Tk();root.withdraw();app=SceneWorkbench(root);app.dataset=folder
            errors=[];root.report_callback_exception=lambda *args:errors.append(args)
            try:
                with patch('scene_reconstruction_window.messagebox.showerror') as dialogs:
                    view=app.select_stage('scene_reconstruction');wait(root,view)
                    self.assertIn(LABEL,view.method_box['values']);view.method.set(LABEL);view.method_changed()
                    with patch('brush_reconstruction.train',side_effect=fake_train):view.start_base();wait(root,view)
                    dialogs.assert_not_called();self.assertIsNone(view.mesh)
                    self.assertEqual(view.current_model.name,'gaussians.ply');self.assertEqual(int(view.bar['value']),100)
                    self.assertEqual(str(view.sync_check['state']),'disabled')
                    self.assertIsNotNone(view.splat_preview.renderer.scene)
                    self.assertIsNone(view.splat_preview.renderer.error)
                    self.assertFalse(hasattr(view.splat_preview,"tabs"))
                    destination=Path(tmp)/'exported.ply'
                    with patch('scene_reconstruction_window.filedialog.asksaveasfilename',return_value=str(destination)):view.export()
                    self.assertEqual(digest(destination),digest(view.current_model))
                    self.assertFalse(hasattr(view.splat_preview,'viewer_button'))
                    self.assertEqual(view.current_model.parent,result_target(folder/'rgbd_scene/setp3','gaussian'))
                    self.assertIsNone(view.splat_preview.renderer.gpu)  # withdrawn test root starts no desktop window
                    from test_scene_workflow_v2 import ply
                    mesh_target=result_target(folder/'rgbd_scene/setp3','mesh');mesh_target.mkdir()
                    ply(mesh_target/'geometry/mesh.ply',np.array([[0,0,0],[1,0,0],[0,1,0]]),[[0,1,2]])
                    write_json(mesh_target/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,method='poisson',
                        source_fingerprint=scene.report['source_fingerprint'],model_relative='geometry/mesh.ply'))
                    view.set_source(scene.source);wait(root,view);self.assertEqual(view.method.get(),LABEL)
                    self.assertEqual(view.display_box['values'],(MESH_VIEW,SPLAT_VIEW))
                    view.display_mode.set(MESH_VIEW);view.choose_result();self.assertIsNotNone(view.mesh)
                    view.display_mode.set(SPLAT_VIEW);view.choose_result();self.assertIsNone(view.mesh)
                    review=app.open_review();wait(root,review)
                    self.assertEqual(review.right_mode.get(),SPLAT_VIEW);self.assertEqual(review.metrics_values['gaussians'],1)
                    self.assertEqual(review.right_select['values'],(MESH_VIEW,SPLAT_VIEW,'精细物体模型'))
                    self.assertIsNone(review.metrics_values['faces']);self.assertIn('PSNR',review.metrics.get())
                    review.right_mode.set(MESH_VIEW);review.show_right();self.assertEqual(review.metrics_values['faces'],1)
                    review.right_mode.set('精细物体模型');review.show_right();self.assertIsNone(review.current_model)
                    review.right_mode.set(SPLAT_VIEW);review.show_right();self.assertEqual(review.current_model.name,'gaussians.ply')
                    view.display_mode.set(MESH_VIEW);view.choose_result();app.open_review()
                    self.assertEqual(review.right_mode.get(),MESH_VIEW)
                    view.display_mode.set(SPLAT_VIEW);view.choose_result();app.open_review()
                    self.assertEqual(review.right_mode.get(),SPLAT_VIEW)
                    stage4=app.open_segmentation();wait(root,stage4)
                    self.assertIsInstance(stage4,SceneRefinementWindow)
                    dialogs.assert_not_called();self.assertFalse(errors)
            finally:app.close()


if __name__=='__main__':unittest.main(verbosity=2)
