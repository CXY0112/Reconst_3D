import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from cloud_import import ROOT,load_cloud
from kalman_fusion import SurfelMap,session_observations,process_session,measurement_variance
from preprocess import Parameters,ProcessingCancelled,save_result


def write_session(folder):
    (folder/'pointcloud_raw').mkdir()
    x,y=np.meshgrid(np.linspace(-.12,.12,31),np.linspace(-.12,.12,31))
    rng=np.random.default_rng(4);frames=[];regs=[]
    for i,offset in enumerate((0,.006),1):
        points=np.column_stack([x.ravel(),y.ravel(),1+offset+rng.normal(0,.001,x.size)])
        np.save(folder/'pointcloud_raw'/f'{i:.6f}.npy',points*1000)
        frames.append({'capture':i,'stamp':i,'cloud_source':'sdk_xyz'})
        regs.append({'capture':i,'node':10+i,'accepted':True})
    (folder/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
    (folder/'registrations.jsonl').write_text('\n'.join(map(json.dumps,regs)))
    (folder/'geometry_poses.json').write_text(json.dumps({'node_ids':[11,12],'poses':[np.eye(4).tolist()]*2}))
    cal={f'R{i}{j}':int(i==j) for i in range(3) for j in range(3)}
    cal.update(T1=0,T2=0,T3=0)
    (folder/'sdk_calibration.json').write_text(json.dumps(cal))


class KalmanTests(unittest.TestCase):
    def test_scalar_normal_kalman_update(self):
        model=SurfelMap(Parameters())
        model.integrate([[0,0,1]],[[0,0,-1]],[.0001],[True],1)
        model.integrate([[0,0,1.008]],[[0,0,-1]],[.0001],[True],2)
        np.testing.assert_allclose(model.position,[[0,0,1.004]],atol=1e-9)
        np.testing.assert_allclose(model.variance,[.00005])
        self.assertEqual(model.support.tolist(),[2])

    def test_correlated_same_frame_pixels_do_not_increase_confidence(self):
        results=[]
        for count in (1,100):
            model=SurfelMap(Parameters())
            model.integrate([[0,0,1]],[[0,0,-1]],[.0001],[True],1)
            model.integrate(np.tile([0,0,1.008],(count,1)),np.tile([0,0,-1],(count,1)),np.full(count,.0001),np.ones(count,bool),2)
            results.append(model)
        np.testing.assert_allclose(results[0].position,results[1].position)
        np.testing.assert_allclose(results[0].variance,results[1].variance)
        self.assertEqual(results[1].support.tolist(),[2])
        with self.assertRaises(ValueError):
            results[1].integrate([[0,0,1]],[[0,0,-1]],[.0001],[True],2)

    def test_opposite_faces_and_out_of_gate_layers_stay_separate(self):
        model=SurfelMap(Parameters())
        model.integrate([[0,0,1]],[[0,0,-1]],[.0001],[True],1)
        model.integrate([[0,0,1.008]],[[0,0,1]],[.0001],[True],2)
        self.assertEqual(len(model.position),2)
        model.integrate([[0,0,1.04]],[[0,0,-1]],[.0001],[True],3)
        self.assertEqual(len(model.position),3)
        self.assertTrue(np.all(model.support==1))

    def test_variance_floor_and_anchor_limit(self):
        model=SurfelMap(Parameters(max_displacement_mm=2))
        model.integrate([[0,0,1]],[[0,0,-1]],[.00001],[True],1)
        for capture in range(2,30):
            model.integrate([[0,0,1.008]],[[0,0,-1]],[.00001],[True],capture)
        self.assertGreaterEqual(model.variance.min(),.002**2)
        self.assertLessEqual(np.linalg.norm(model.position[0]-model.anchor[0]),.002000001)

    def test_only_accepted_capture_files_are_mapped_to_poses(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            folder=Path(temp);write_session(folder)
            np.save(folder/'pointcloud_raw/0.500000.npy',np.zeros((20,3)))
            _,observations,_=session_observations(folder)
            self.assertEqual([o['path'].name for o in observations],['1.000000.npy','2.000000.npy'])
            graph=json.loads((folder/'geometry_poses.json').read_text());graph['poses'][0][0][0]=2
            (folder/'geometry_poses.json').write_text(json.dumps(graph))
            with self.assertRaises(ValueError):session_observations(folder)

    def test_session_pipeline_export_and_point_attributes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            parent=Path(temp);folder=parent/'scan';folder.mkdir();write_session(folder)
            reference=np.column_stack([np.zeros(100),np.zeros(100),np.linspace(.9,1.1,100)])
            original=(folder/'geometry_poses.json').read_bytes()
            result=process_session(folder,Parameters(output_voxel_mm=8),before_points=reference)
            self.assertEqual(result.report['captures'],2)
            self.assertGreater(result.report['output_points_supported_by_multiple_views'],0)
            self.assertTrue(result.report['source_unchanged'])
            self.assertEqual((folder/'geometry_poses.json').read_bytes(),original)
            self.assertEqual(len(result.retained_indices),0)
            self.assertEqual(len(result.diagnostics['support_views']),len(result.points))
            output=save_result(result,folder/'map_cloud.ply',parent/'export')
            np.testing.assert_allclose(load_cloud(output/'processed.ply').points,result.points,atol=1e-6)
            with np.load(output/'surfel_attributes.npz') as attributes:
                self.assertEqual(len(attributes['support_views']),len(result.points))
            saved=json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertIn('无最终PLY',saved['index_note'])

    def test_missing_raw_session_and_cancellation_fail_explicitly(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            with self.assertRaisesRegex(ValueError,'单独PLY'):session_observations(temp)
        stop=threading.Event();stop.set()
        with self.assertRaises(ProcessingCancelled):process_session('missing',cancel=stop)
        self.assertGreater(measurement_variance(3,1),measurement_variance(1,1))
        self.assertGreater(measurement_variance(1,.2),measurement_variance(1,1))

    def test_gui_kalman_session_support_view_and_export(self):
        import tkinter as tk
        from preprocess_window import PreprocessWindow
        from reconstruct import save_cloud
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            parent=Path(temp);folder=parent/'scan';folder.mkdir();write_session(folder)
            save_cloud(folder/'map_cloud.ply',np.load(folder/'pointcloud_raw/1.000000.npy')/1000,
                       'RTAB-Map world XYZ in metres, X forward Y left Z up')
            root=tk.Tk();root.withdraw();window=PreprocessWindow(root)
            def wait():
                deadline=time.monotonic()+20
                while window.working and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertIsNone(window.working)
            try:
                with patch('tkinter.messagebox.showerror') as error:
                    window.load(folder);wait();window.backend.set('卡尔曼表面融合');window.process();wait()
                    error.assert_not_called()
                self.assertEqual(window.result.report['backend'],'kalman')
                window.view.set('多次观测确认的表面');window._show_result()
                self.assertGreater(len(window.after.points),0)
                window.view.set('移动超过3mm的原位置');window._show_result()
                self.assertEqual(len(window.after.points),0)
                with patch('tkinter.messagebox.showerror') as error:
                    window.save(parent/'export');wait();error.assert_not_called()
                self.assertTrue((parent/'export/surfel_attributes.npz').is_file())
            finally:window.close()


if __name__=='__main__':unittest.main()
