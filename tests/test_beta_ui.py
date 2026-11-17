from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import unittest,tempfile,runpy,tkinter as tk,zipfile
import numpy as np
from reconstruct import ROOT
from workbench import Workbench
from live_cloud_view import ViewCamera,raster_cloud


class BetaUITests(unittest.TestCase):
    def test_beta_launcher(self):
        with patch('sys.argv',['main.py','beta']),patch('scene_workbench.launch') as launch:
            runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
            launch.assert_called_once_with(beta=True)

    def test_roll_changes_camera_only_and_opposite_keys_cancel(self):
        points=np.array([[.5,0,2]],np.float32);original=points.copy()
        camera=ViewCamera([1,2,3],yaw=.3,pitch=.2)
        position=camera.position.copy();forward=camera.basis()[:,2].copy()
        self.assertTrue(camera.move({'z'},.5));self.assertLess(camera.roll,0)
        np.testing.assert_allclose(camera.basis()[:,2],forward,atol=1e-7)
        np.testing.assert_array_equal(camera.position,position)
        camera.move({'c'},.5);self.assertAlmostEqual(camera.roll,0)
        first=raster_cloud(points,width=100,height=100,intrinsics=(20,20,50,50),enlarge=False)
        rotated=raster_cloud(points,width=100,height=100,intrinsics=(20,20,50,50),enlarge=False,roll=np.pi/2)
        self.assertFalse(np.array_equal(first,rotated));np.testing.assert_array_equal(points,original)

    def test_hidden_defaults_and_poisson_enforcement(self):
        root=tk.Tk();root.withdraw();app=Workbench(root,beta=True)
        try:
            p=app.select_stage('preprocessing');g=app.select_stage('reconstruction')
            self.assertEqual(p.fields['noise_mm'].master.winfo_manager(),'')
            self.assertEqual(p.fields['kalman_floor_mm'].master.winfo_manager(),'')
            self.assertEqual(g.method_box.winfo_manager(),'')
            g.source=ROOT;g.method.set('NKSR')
            with patch('geometry_window.threading.Thread'):
                g.start();self.assertEqual(g.job_method,'poisson')
            g.working=None
            p.fields['noise_mm'].delete(0,'end');p.fields['noise_mm'].insert(0,'99')
            p.source=ROOT;p.loaded=SimpleNamespace(points=np.zeros((20,3)))
            with patch('preprocess_window.process_input') as process,patch('preprocess_window.threading.Thread') as thread:
                thread.side_effect=lambda target,**kw:SimpleNamespace(start=target)
                p.process()
                params=process.call_args.args[2]
                self.assertEqual((params.noise_mm,params.kalman_floor_mm,params.surface_spacing_mm),(10,2,2))
            p.working=None
            self.assertTrue(hasattr(app,'export_button'))
        finally:
            app.close()

    def test_space_capture_is_scoped_and_does_not_repeat(self):
        root=tk.Tk();root.withdraw();app=Workbench(root)
        view=app.views['acquisition']
        try:
            view.session=Mock();view.session.toggle_pause.return_value=True;view.busy=True
            view.capture_button.configure(state='normal')
            event=SimpleNamespace(widget=view.finish_button)
            self.assertTrue(view.finish_button.bindtags()[0].startswith('CaptureSpace_'))
            self.assertEqual(view._space_press(event),'break')
            view.capture_button.configure(state='normal')
            view._space_press(event);self.assertEqual(view.session.toggle_pause.call_count,1)
            view._space_release();view._space_press(event)
            self.assertEqual(view.session.toggle_pause.call_count,2)
            other=app.select_stage('preprocessing')
            self.assertFalse(other.process_button.bindtags()[0].startswith('CaptureSpace_'))
        finally:
            view.busy=False;view.session=None;app.close()

    def test_beta_automatic_pipeline_offline_and_resume(self):
        import json,time,cv2
        from dataset_store import create_dataset,inspect_dataset
        from snapshot_window import SnapshotSession
        from reconstruct import save_cloud
        from test_kalman_fusion import write_session
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary);session=SnapshotSession(__import__('queue').Queue())
            (session.folder/'pointcloud_local').rmdir();write_session(session.folder)
            cal=json.loads((session.folder/'sdk_calibration.json').read_text())
            cal.update(fxrgb=150,fyrgb=150,cxrgb=32,cyrgb=32)
            (session.folder/'sdk_calibration.json').write_text(json.dumps(cal))
            for i in (1,2):
                cv2.imwrite(str(session.folder/'rgb'/f'{i:.6f}.png'),np.full((64,64,3),128,np.uint8))
                cv2.imwrite(str(session.folder/'depth'/f'{i:.6f}.png'),np.full((64,64),1000,np.uint16))
            session.final_world_cloud=np.load(session.folder/'pointcloud_raw/1.000000.npy')/1000
            np.save(session.folder/'full_geometry_world.npy',session.final_world_cloud)
            save_cloud(session.folder/'map_cloud.ply',session.final_world_cloud,'RTAB-Map world XYZ in metres')
            session.k=np.array([[150.,0,32],[0,150,32],[0,0,1]]);session.shape=(64,64)
            root=tk.Tk();root.withdraw();app=Workbench(root,beta=True)
            def wait():
                deadline=time.monotonic()+120
                while app._busy() and time.monotonic()<deadline:root.update();time.sleep(.01)
                root.update();self.assertFalse(app._busy(),app.automation.status.get())
            try:
                scan=app.views['acquisition'];session.events=scan.events;scan.session=session;scan.has_map=True
                with patch('tkinter.simpledialog.askstring',return_value='demo'),patch('scan_window.create_dataset',side_effect=lambda _,name:create_dataset(parent,name)),patch('tkinter.messagebox.showinfo'),patch('tkinter.messagebox.showerror') as errors:
                    scan.busy=True
                    scan.events.put(('final',(2,session.final_world_cloud,[])))
                    with patch('tkinter.messagebox.askyesno') as confirm:
                        wait();confirm.assert_not_called()
                    errors.assert_not_called()
                    self.assertTrue(all(v['complete'] and not v['stale'] for v in inspect_dataset(parent/'demo').values()),app.automation.status.get())
                    self.assertEqual(app.tabs.select(),str(app.frames['review']))
                    self.assertEqual([n for n,f in app.frames.items() if app.tabs.tab(f,'state')!='hidden'],['acquisition','review'])
                    metrics=app.views['review'].metrics.get()
                    self.assertIn('10 mm',metrics);self.assertIn('RGB',metrics)
                    self.assertNotIn('10 mm 内覆盖率 未记录',metrics)
                    files={p:p.stat().st_mtime_ns for p in (parent/'demo').rglob('*') if p.is_file()}
                    app.import_dataset(parent/'demo');wait();errors.assert_not_called()
                    self.assertEqual(app.tabs.select(),str(app.frames['review']))
                    self.assertEqual(files,{p:p.stat().st_mtime_ns for p in (parent/'demo').rglob('*') if p.is_file()})
                    # A rerun invalidates downstream output; failure must stop, not display old output.
                    app.stage_saved(1,parent/'demo/setp1')
                    with patch('preprocess_window.process_input',side_effect=RuntimeError('forced failure')):
                        # Explicitly force reprocessing via manifest stale state.
                        app.automation.stop()
                        from dataset_store import replace_stage
                        import shutil
                        with replace_stage(parent/'demo/setp2',2,parent/'demo') as target:
                            shutil.copytree(parent/'demo/setp2',target)
                        app.automation.start()
                        with patch('geometry_window.GeometryWindow._command',side_effect=RuntimeError('forced failure')):
                            wait()
                    self.assertIn('自动处理已停止',app.automation.status.get())
                    self.assertTrue(inspect_dataset(parent/'demo')[3]['stale'])
                    errors.reset_mock();app.automation.start();wait();errors.assert_not_called()
                    self.assertFalse(inspect_dataset(parent/'demo')[3]['stale'])
                    self.assertEqual(app.tabs.select(),str(app.frames['review']))
            finally:
                app.close()
                if session.folder.name.startswith('.pending_scan_') and session.folder.exists():session.discard()

    def test_export_saved_stages_and_manifest_only(self):
        from dataset_store import create_dataset
        from dataset_export import export_dataset
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary);source=create_dataset(parent,'演示')
            (source/'setp1').mkdir();(source/'setp1/data.bin').write_bytes(b'capture')
            (source/'experiment.txt').write_text('not a pipeline stage')
            target=export_dataset(source,parent/'demo.zip')
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()),{'演示/dataset.json','演示/setp1/data.bin'})
            with self.assertRaises(ValueError):export_dataset(source,source/'wrong.zip')
            with self.assertRaises(FileExistsError):export_dataset(source,target)


if __name__=='__main__':unittest.main()
