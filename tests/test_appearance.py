import json
from pathlib import Path
import subprocess
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from appearance_pipeline import prepare_depth,observe,combine_observations,resolve_inputs
from kalman_fusion import OPTICAL_FROM_WORLD
from mesh_view import mesh_buffers
from reconstruct import ROOT
from workbench import Workbench


def fixture(parent):
    session=parent/'setp1_fixture';session.mkdir()
    for name in ('rgb','depth','pointcloud_raw'):(session/name).mkdir()
    calibration={f'R{i}{j}':float(i==j) for i in range(3) for j in range(3)}
    calibration.update(T1=0,T2=0,T3=0,fxrgb=50,fyrgb=50,cxrgb=32,cyrgb=32)
    (session/'sdk_calibration.json').write_text(json.dumps(calibration))
    pose=np.eye(4);pose[:3,:3]=OPTICAL_FROM_WORLD.T
    (session/'geometry_poses.json').write_text(json.dumps({'node_ids':[1,2],'poses':[pose.tolist(),pose.tolist()]}))
    (session/'frames.jsonl').write_text('\n'.join(json.dumps({'capture':i,'stamp':float(i),'cloud_source':'sdk_xyz'}) for i in (1,2)))
    (session/'registrations.jsonl').write_text('\n'.join(json.dumps({'capture':i,'node':i,'accepted':True}) for i in (1,2)))
    for i,bgr in ((1,[0,0,255]),(2,[255,0,0])):
        cv2.imwrite(str(session/'rgb'/f'{i:.6f}.png'),np.tile(np.uint8(bgr),(64,64,1)))
        cv2.imwrite(str(session/'depth'/f'{i:.6f}.png'),np.full((64,64),1000,np.uint16))
        np.save(session/'pointcloud_raw'/f'{i:.6f}.npy',np.array([[0,0,1000.]]))
    pre=parent/'setp2_fixture';pre.mkdir();(pre/'report.json').write_text(json.dumps({'source_session':str(session)}))
    source=parent/'setp3_fixture';source.mkdir()
    (source/'report.json').write_text(json.dumps({'complete':True,'pipeline':'poisson_supported_smooth_qem_planar','input':str(pre)}))
    optical=np.array([[x,y,1.] for y in np.linspace(-.3,.3,5) for x in np.linspace(-.3,.3,5)])
    vertices=optical@OPTICAL_FROM_WORLD
    faces=[]
    for y in range(4):
        for x in range(4):
            a=y*5+x;faces.extend([[a,a+1,a+5],[a+1,a+6,a+5]])
    body='\n'.join(' '.join(map(str,v)) for v in vertices)+'\n'
    body+='\n'.join('3 '+' '.join(map(str,f)) for f in faces)+'\n'
    (source/'mesh.ply').write_text(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'+body)
    return source,session


class AppearanceTests(unittest.TestCase):
    def test_neural_geometry_retains_rgb_source_resolution(self):
        from geometry_methods import PIPELINES
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            source,session=fixture(Path(temporary))
            metadata=json.loads((source/'report.json').read_text())
            for method in ('nksr','noksr'):
                metadata.update(pipeline=PIPELINES[method],method=method)
                (source/'report.json').write_text(json.dumps(metadata))
                resolved=resolve_inputs(source)
                self.assertIsNotNone(resolved)

    def test_depth_holes_edges_and_unknown_are_conservative(self):
        depth=np.ones((15,15));depth[5:10,5:10]=0
        filled,stable=prepare_depth(depth)
        self.assertFalse(stable[7,7]);self.assertTrue(stable[5,5]);self.assertEqual(filled[5,5],1)
        depth[:,8:]=2
        _,stable=prepare_depth(depth);self.assertFalse(stable[3,7]);self.assertFalse(stable[3,8])
        _,stable=prepare_depth(np.zeros((10,10)));self.assertFalse(stable.any())

    def test_projection_channels_pose_depth_and_incidence(self):
        vertices=np.array([[0,0,1],[0,0,1.1],[0,0,-1],[2,0,1],[.1,0,1.]])
        normals=np.tile([0,0,1.],(5,1));normals[4]=[1,0,0]
        k=np.array([[50,0,32],[0,50,32],[0,0,1.]])
        image=np.tile([255.,0,0],(64,64,1));depth=np.ones((64,64))
        ids,colors,_=observe(vertices,normals,{'pose':np.eye(4)},k,image,depth)
        np.testing.assert_array_equal(ids,[0]);np.testing.assert_array_equal(colors,[[1,0,0]])
        pose=np.eye(4);pose[:3,3]=[3,2,0]
        ids,colors,_=observe(vertices+[3,2,0],normals,{'pose':pose},k,image,depth)
        np.testing.assert_array_equal(ids,[0])

    def test_conflicts_best_view_linear_fusion_and_uncovered(self):
        samples=[(1,np.array([0,1]),np.array([[1,0,0],[.3,.3,.3]]),np.array([2.,1.])),
                 (2,np.array([0,1]),np.array([[0,0,1],[.4,.4,.4]]),np.array([1.,1.]))]
        colors,attrs=combine_observations(3,samples)
        np.testing.assert_allclose(colors[0],[1,0,0],atol=1e-12);self.assertEqual(attrs['conflict_count'][0],1)
        self.assertGreater(colors[1,0],.35);self.assertLess(colors[1,0],.4)
        self.assertFalse(attrs['covered'][2]);self.assertEqual(attrs['best_capture'][2],-1)
        _,_,packed=mesh_buffers(np.eye(3),np.array([[0,1,2]]),np.eye(3),colors)
        np.testing.assert_allclose(packed[:,3:],colors)

    def test_session_pose_uses_rgb_not_ir_and_rejected_frames_are_ignored(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            source,session=fixture(Path(temporary))
            calibration=json.loads((session/'sdk_calibration.json').read_text());calibration['T1']=12
            (session/'sdk_calibration.json').write_text(json.dumps(calibration))
            _,_,frames,_,_=resolve_inputs(source)
            expected=np.eye(4);expected[:3,:3]=OPTICAL_FROM_WORLD.T
            np.testing.assert_allclose(frames[0]['pose'],expected,atol=1e-12)
            with (session/'frames.jsonl').open('a') as stream:stream.write('\n'+json.dumps({'capture':99,'stamp':99}))
            self.assertEqual(len(resolve_inputs(source)[2]),2)

    def test_fourth_tab_real_worker_preserves_geometry_and_reloads_colors(self):
        with tempfile.TemporaryDirectory(prefix='appearance_test_',dir=ROOT/'results') as temporary:
            folder=Path(temporary).resolve();source,session=fixture(folder)
            original=(source/'mesh.ply').read_bytes()
            root=tk.Tk();root.withdraw();app=Workbench(root,'appearance');view=app.views['appearance']
            def wait():
                deadline=time.monotonic()+45
                while view.working and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertIsNone(view.working)
            try:
                with patch('tkinter.messagebox.showerror') as errors:
                    view.set_source(source);wait();self.assertEqual(len(app.tabs.tabs()),5)
                    vertices,triangles=[a.copy() for a in view.mesh[:2]]
                    output=folder/'setp4_result'
                    with patch('appearance_window.stage_result_path',return_value=output):view.start();wait()
                    errors.assert_not_called()
                    self.assertTrue(view.report['complete']);self.assertEqual(view.report['stage'],4)
                    self.assertGreater(view.report['coverage']['colored_vertices'],0)
                    self.assertTrue(view.report['source_unchanged']);self.assertFalse(view.report['geometry_modified'])
                    np.testing.assert_array_equal(view.mesh[0],vertices);np.testing.assert_array_equal(view.mesh[1],triangles)
                    self.assertTrue(np.all(view.colors[view.covered,0]>.99));self.assertTrue(np.all(view.colors[view.covered,2]<.01))
                    self.assertGreater(view.report['coverage']['conflicting_multi_view_fraction'],0)
                    self.assertEqual((source/'mesh.ply').read_bytes(),original)
                    colors=view.colors.copy();view.mode.set('覆盖检查');view.update_mode()
                    np.testing.assert_allclose(view.after.packed[view.covered,3:],np.tile([.15,.7,.35],(view.covered.sum(),1)),atol=1e-6)
                    view.load_result(output);wait();errors.assert_not_called()
                    np.testing.assert_allclose(view.colors,colors,atol=1/255+.0001)
                    view.mode.set('RGB颜色');view.update_mode()
                    np.testing.assert_allclose(view.after.packed[:,3:],view.colors)
                    app.close();self.assertTrue(view.closed)
            finally:
                try:root.destroy()
                except tk.TclError:pass

    def test_missing_rgb_fails_without_creating_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);source,session=fixture(folder)
            (session/'rgb'/'1.000000.png').unlink()
            output=folder/'absent'
            result=subprocess.run([str(ROOT/'vendor/rtabmap/bin/python.exe'),str(ROOT/'tools/colorize_geometry.py'),
                '--worker','--source',str(source),'--output',str(output)],capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse(output.exists())
