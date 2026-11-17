import hashlib
import json
from pathlib import Path
import queue
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from dataset_store import create_dataset, inspect_dataset, stage_folder, replace_stage, read_manifest, route_dataset, REQUIRED
from reconstruct import ROOT, save_cloud
from snapshot_window import SnapshotSession
from test_kalman_fusion import write_session
from workbench import Workbench


def fake_stage(path, stage, value='new'):
    path.mkdir()
    for name in REQUIRED[stage]:
        (path/name).write_text(json.dumps({'complete':True,'value':value}) if name=='report.json' else value,encoding='utf-8')


class DatasetStoreTests(unittest.TestCase):
    def test_names_and_existing_dataset_are_protected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            for name in ('','../bad','bad/name','C:\\bad','CON','name.','..'):
                with self.assertRaises(ValueError):create_dataset(temp,name)
            folder=create_dataset(temp,'测试 test')
            with self.assertRaises(FileExistsError):create_dataset(temp,'测试 test')
            self.assertTrue(all(s['state']=='未生成' for s in inspect_dataset(folder).values()))

    def test_success_replaces_one_stage_and_marks_downstream_stale(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            folder=create_dataset(temp,'test')
            for stage in (2,3,4):
                with replace_stage(stage_folder(folder,stage),stage,folder) as target:fake_stage(target,stage)
            mesh=(folder/'setp3/mesh.ply').read_bytes()
            with replace_stage(folder/'setp2',2,folder) as target:fake_stage(target,2,'replacement')
            self.assertEqual(json.loads((folder/'setp2/report.json').read_text())['value'],'replacement')
            self.assertEqual((folder/'setp3/mesh.ply').read_bytes(),mesh)
            states=inspect_dataset(folder)
            self.assertEqual(states[2]['state'],'已完成');self.assertEqual(states[3]['state'],'需更新');self.assertEqual(states[4]['state'],'需更新')
            self.assertFalse(list(folder.glob('.pending*')));self.assertFalse(list(folder.glob('.previous*')))
            with replace_stage(folder/'setp4',4,folder) as target:fake_stage(target,4)
            self.assertEqual(inspect_dataset(folder)[4]['state'],'需更新')

    def test_partial_state_and_folder_import_routing(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            folder=create_dataset(temp,'test');(folder/'setp2').mkdir()
            self.assertEqual(inspect_dataset(folder)[2]['state'],'未完成')
            view=SimpleNamespace(dataset=None,on_import_dataset=Mock())
            self.assertTrue(route_dataset(view,folder/'setp2'))
            view.on_import_dataset.assert_called_once_with(folder)
            view.dataset=folder
            self.assertFalse(route_dataset(view,folder/'setp2'))

    def test_failure_cancel_incomplete_and_manifest_failure_preserve_previous(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            folder=create_dataset(temp,'test');target=folder/'setp3'
            with replace_stage(target,3,folder) as working:fake_stage(working,3,'original')
            original=(target/'report.json').read_bytes();manifest=(folder/'dataset.json').read_bytes()
            for error in (RuntimeError,InterruptedError):
                with self.assertRaises(error):
                    with replace_stage(target,3,folder) as working:
                        fake_stage(working,3);raise error('stop')
            with self.assertRaises(ValueError):
                with replace_stage(target,3,folder) as working:working.mkdir()
            with patch('dataset_store.write_manifest',side_effect=OSError('cannot update index')):
                with self.assertRaises(OSError):
                    with replace_stage(target,3,folder) as working:fake_stage(working,3)
            self.assertEqual((target/'report.json').read_bytes(),original)
            self.assertEqual((folder/'dataset.json').read_bytes(),manifest)
            self.assertFalse(list(folder.glob('.pending*')));self.assertFalse(list(folder.glob('.previous*')))
            with self.assertRaises(ValueError):
                with replace_stage(folder/'wrong',3,folder):pass

    def test_named_scan_save_all_stages_auto_restore_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            parent=Path(temp)
            session=SnapshotSession(queue.Queue())
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
            root=tk.Tk();root.withdraw();app=Workbench(root)
            def wait(view):
                deadline=time.monotonic()+90
                while (getattr(view,'working',None) or getattr(view,'busy',False) or getattr(view,'importing',False)) and time.monotonic()<deadline:
                    root.update();time.sleep(.01)
                self.assertFalse(getattr(view,'working',None));self.assertFalse(getattr(view,'busy',False));root.update()
            try:
                scan=app.views['acquisition'];session.events=scan.events;scan.session=session;scan.has_map=True
                with patch('tkinter.simpledialog.askstring',return_value='test') as naming,patch('scan_window.create_dataset',side_effect=lambda _,name:create_dataset(parent,name)),patch('tkinter.messagebox.showinfo'),patch('tkinter.messagebox.showerror') as errors:
                    scan.request_save();wait(scan);naming.assert_called_once();errors.assert_not_called()
                    folder=parent/'test';self.assertEqual(app.dataset,folder)
                    self.assertIn((folder/'setp1').as_posix(),(folder/'setp1/source.ini').read_text())
                    self.assertNotIn('.pending',(folder/'setp1/source.ini').read_text())
                    self.assertTrue(inspect_dataset(folder)[1]['complete'])
                    pre=app.select_stage('preprocessing');wait(pre);pre.process();wait(pre)
                    errors.assert_not_called();self.assertEqual(pre.saved_path,folder/'setp2')
                    geometry=app.select_stage('reconstruction');wait(geometry);geometry.start();wait(geometry)
                    errors.assert_not_called();self.assertEqual(geometry.output,folder/'setp3')
                    old_mesh=hashlib.sha256((geometry.output/'mesh.ply').read_bytes()).hexdigest()
                    with patch.object(geometry,'_command',side_effect=RuntimeError('forced worker failure')):
                        geometry.start();wait(geometry)
                    self.assertEqual(hashlib.sha256((geometry.output/'mesh.ply').read_bytes()).hexdigest(),old_mesh)
                    errors.reset_mock()
                    appearance=app.select_stage('appearance');wait(appearance);appearance.start();wait(appearance)
                    errors.assert_not_called();self.assertEqual(appearance.output,folder/'setp4')
                    first_time=read_manifest(folder)['stages']['4']['updated_at']
                    appearance.start();wait(appearance);errors.assert_not_called()
                    self.assertNotEqual(read_manifest(folder)['stages']['4']['updated_at'],first_time)
                    review=app.select_stage('review');wait(review);self.assertIsNotNone(review.mesh)
                    # Root folder import alone restores each saved stage on demand.
                    app.import_dataset(folder);wait(app.views['review'])
                    pre=app.select_stage('preprocessing');wait(pre)
                    self.assertIsNotNone(pre.result);self.assertEqual(pre.saved_path,folder/'setp2')
                    geometry=app.select_stage('reconstruction');wait(geometry);self.assertEqual(geometry.output,folder/'setp3')
                    appearance=app.select_stage('appearance');wait(appearance);self.assertEqual(appearance.output,folder/'setp4')
                    pre=app.select_stage('preprocessing');pre.fields['noise_mm'].delete(0,'end');pre.fields['noise_mm'].insert(0,'11')
                    pre.process();wait(pre);errors.assert_not_called()
                    self.assertEqual(json.loads((folder/'setp2/report.json').read_text(encoding='utf-8'))['parameters']['noise_mm'],11)
                    self.assertEqual(inspect_dataset(folder)[3]['state'],'需更新')
                    self.assertEqual(inspect_dataset(folder)[4]['state'],'需更新')
                    self.assertEqual({p.name for p in folder.iterdir()},{'dataset.json','setp1','setp2','setp3','setp4'})
                    app.close()
            finally:
                if session.folder.name.startswith('.pending_scan_') and session.folder.exists():session.discard()
                try:root.destroy()
                except tk.TclError:pass
