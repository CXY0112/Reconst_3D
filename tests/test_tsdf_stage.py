"""Skipped-stage persistence, native TSDF geometry and downstream RGB provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from reconstruct import ROOT, save_cloud
from dataset_store import create_dataset, replace_stage, inspect_dataset
from geometry_inputs import save_skip, source_info
from appearance_pipeline import resolve_inputs
from kalman_fusion import OPTICAL_FROM_WORLD
from workbench import Workbench


def session_fixture(parent):
    dataset=create_dataset(parent,'fixture');session=dataset/'setp1';session.mkdir()
    for name in ('rgb','depth','depth_raw','pointcloud_raw'):(session/name).mkdir()
    cal={f'R{i}{j}':float(i==j) for i in range(3) for j in range(3)}
    cal.update(T1=-12.,T2=0.,T3=0.,fxrgb=100.,fyrgb=100.,cxrgb=20.,cyrgb=20.)
    poses=[];frames=[];regs=[];world=[]
    v,u=np.mgrid[:40,:40];depth=(1000+2*(u-20)).astype(np.uint16);z=depth.ravel()/1000.
    local=np.c_[(u.ravel()-20)/100*z,(v.ravel()-20)/100*z,z]
    e=np.eye(4);e[0,3]=-.012
    for i in (1,2):
        pose=np.eye(4);pose[:3,:3]=OPTICAL_FROM_WORLD.T;pose[1,3]=(i-1)*.02
        poses.append(pose.tolist());frames.append(dict(capture=i,stamp=float(i),cloud_source='sdk_xyz'))
        regs.append(dict(capture=i,node=i,accepted=True))
        equivalent=pose@e;world.append(local@equivalent[:3,:3].T+equivalent[:3,3])
        np.save(session/'pointcloud_raw'/f'{i:.6f}.npy',local*1000)
        for directory in ('depth','depth_raw'):cv2.imwrite(str(session/directory/f'{i:.6f}.png'),depth)
        cv2.imwrite(str(session/'rgb'/f'{i:.6f}.png'),np.full((40,40,3),[20,80,160],np.uint8))
    (session/'sdk_calibration.json').write_text(json.dumps(cal),encoding='utf-8')
    (session/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)),encoding='utf-8')
    (session/'registrations.jsonl').write_text('\n'.join(map(json.dumps,regs)),encoding='utf-8')
    (session/'geometry_poses.json').write_text(json.dumps(dict(node_ids=[1,2],poses=poses)),encoding='utf-8')
    points=np.vstack(world);np.save(session/'full_geometry_world.npy',points)
    save_cloud(session/'map_cloud.ply',points,'RTAB-Map world XYZ in metres')
    return dataset,session,points


class TSDFStageTests(unittest.TestCase):
    def wait(self,root,view,seconds=50):
        deadline=time.monotonic()+seconds
        while view.working is not None and time.monotonic()<deadline:root.update();time.sleep(.01)
        self.assertIsNone(view.working);root.update()

    def test_skip_record_moves_with_dataset_and_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary);dataset,session,points=session_fixture(parent)
            with replace_stage(dataset/'setp2',2,dataset) as target:save_skip(session,target,len(points))
            self.assertEqual({p.name for p in (dataset/'setp2').iterdir()},{'report.json'})
            self.assertEqual(inspect_dataset(dataset)[2]['state'],'已跳过')
            original=(dataset/'setp2/report.json').read_bytes()
            with self.assertRaises(RuntimeError):
                with replace_stage(dataset/'setp2',2,dataset) as target:
                    save_skip(session,target,len(points));raise RuntimeError('cancel')
            self.assertEqual((dataset/'setp2/report.json').read_bytes(),original)
            moved=parent/'moved';dataset.rename(moved)
            self.assertEqual(source_info(moved/'setp2')['session'],moved/'setp1')
            self.assertTrue(inspect_dataset(moved)[2]['complete'])

    def test_skip_button_handoff_and_saved_restore_without_processed_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            dataset,session,_=session_fixture(Path(temporary))
            root=tk.Tk();root.withdraw();app=Workbench(root,'preprocessing')
            try:
                with patch('tkinter.messagebox.showerror') as errors:
                    app.import_dataset(dataset);pre=app.select_stage('preprocessing');self.wait(root,pre)
                    self.assertEqual(str(pre.skip_button['state']),'normal')
                    pre.skip();self.wait(root,pre)
                    geometry=app.views['reconstruction'];self.wait(root,geometry)
                    errors.assert_not_called()
                    self.assertEqual(geometry.source,dataset/'setp2')
                    self.assertTrue(inspect_dataset(dataset)[2]['skipped'])
                    app.import_dataset(dataset);pre=app.select_stage('preprocessing');self.wait(root,pre)
                    self.assertEqual(pre.saved_path,dataset/'setp2');self.assertIsNone(pre.result)
                    self.assertIn('已跳过',pre.status.get())
                    errors.assert_not_called()
            finally:
                app.close()
                try:root.destroy()
                except tk.TclError:pass

    def test_tsdf_worker_rgb_provenance_and_invalid_grid_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary);dataset,session,points=session_fixture(parent)
            with replace_stage(dataset/'setp2',2,dataset) as target:save_skip(session,target,len(points))
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.rglob('*') if p.is_file()}
            output=parent/'tsdf'
            command=[sys.executable,str(ROOT/'tools/reconstruct_geometry.py'),'--source',str(dataset/'setp2'),'--method','tsdf']
            run=subprocess.run(command+['--output',str(output)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            report=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['complete']);self.assertEqual(report['method'],'tsdf');self.assertFalse(report['preprocessing_used'])
            self.assertGreater(report['final']['triangles'],100)
            _,_,frames,_,_=resolve_inputs(output)
            expected=np.eye(4);expected[:3,:3]=OPTICAL_FROM_WORLD.T;expected[:3,3]=OPTICAL_FROM_WORLD.T@np.array([-.012,0,0])
            np.testing.assert_allclose(frames[0]['pose'],expected,atol=1e-10)
            self.assertEqual(frames[0]['depth'].parent.name,'depth_raw')
            # Stage-four worker and stage-five source resolver accept the new route.
            colored=parent/'colored'
            run=subprocess.run([sys.executable,str(ROOT/'tools/colorize_geometry.py'),'--source',str(output),'--output',str(colored)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            from saved_sources import review_sources
            info=review_sources(colored)
            self.assertEqual(info['session'],session);self.assertEqual(info['preprocessed'],dataset/'setp2')
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
            cal=json.loads((session/'sdk_calibration.json').read_text(encoding='utf-8'));cal['fxrgb']=10.
            (session/'sdk_calibration.json').write_text(json.dumps(cal),encoding='utf-8')
            bad=parent/'bad'
            run=subprocess.run(command+['--output',str(bad)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            self.assertNotEqual(run.returncode,0)
            failed=json.loads((bad/'report.json').read_text(encoding='utf-8'))
            self.assertFalse(failed['complete']);self.assertIn('核验失败',failed['error'])
            self.assertFalse((bad/'mesh.ply').exists())
