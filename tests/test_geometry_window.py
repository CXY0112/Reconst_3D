import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import numpy as np
from reconstruct import ROOT
from mesh_view import mesh_buffers
from workbench import Workbench


class GeometryStageTests(unittest.TestCase):
    def test_two_method_selector_and_historical_neural_result_reload(self):
        from geometry_methods import PIPELINES,NOKSR_DEFAULTS
        self.assertEqual(NOKSR_DEFAULTS,dict(k_neighbors=8,last_n_layers=4,mask_threshold_mm=30.,voxel_mm=20.,meshing_voxel_mm=15.))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);source=folder/'source';self.source(source)
            root=tk.Tk();root.withdraw();app=Workbench(root,'reconstruction')
            try:
                view=app.views['reconstruction']
                self.assertEqual(tuple(view.method_box['values']),('Poisson','NKSR','TSDF'))
                for method in ('nksr','noksr'):
                    output=folder/method;output.mkdir()
                    (output/'mesh.ply').write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n1 0 0\n1 .1 0\n1 0 .1\n3 0 1 2\n')
                    (output/'report.json').write_text(json.dumps(dict(complete=True,pipeline=PIPELINES[method],method=method,
                        input=str(source),final=dict(vertices=3,triangles=1),elapsed_seconds=1.)))
                    with patch('tkinter.messagebox.showerror') as errors:
                        view.load_result(output);self.wait(root,lambda:view.working is None)
                        errors.assert_not_called()
                    self.assertEqual(view.method.get(),{'nksr':'NKSR','noksr':'Poisson'}[method])
                    self.assertEqual(len(view.after.triangles),1)
            finally:
                app.close()
                try:root.destroy()
                except tk.TclError:pass

    def wait(self, root, condition, seconds=30):
        deadline=time.monotonic()+seconds
        while not condition() and time.monotonic()<deadline:
            root.update();time.sleep(.01)
        self.assertTrue(condition())

    def source(self, folder):
        folder.mkdir()
        n=2000;z=1-2*(np.arange(n)+.5)/n;angle=np.arange(n)*np.pi*(3-np.sqrt(5))
        normals=np.column_stack((np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z))
        np.save(folder/'processed_world.npy', normals*.1)
        np.savez(folder/'surfel_attributes.npz', normals_world=normals)
        (folder/'report.json').write_text(json.dumps({'complete':True,'backend':'kalman'}), encoding='utf-8')

    def test_mesh_buffer_coordinates_indices_and_shading(self):
        world=np.array([[1.,2.,3.],[2.,2.,3.],[1.,3.,3.]])
        normals=np.tile([0.,0.,1.],(3,1));faces=np.array([[0,1,2]])
        points, triangles, packed=mesh_buffers(world,faces,normals)
        np.testing.assert_allclose(points,[[-2,-3,1],[-2,-3,2],[-3,-3,1]])
        np.testing.assert_array_equal(triangles,faces)
        self.assertEqual(packed.shape,(3,6));self.assertEqual(packed.dtype,np.float32)
        self.assertTrue(np.all((packed[:,3:]>=0)&(packed[:,3:]<=1)))
        with self.assertRaises(ValueError):mesh_buffers(world,np.array([[0,1,3]]),normals)

    def test_third_tab_handoff_real_worker_preview_and_saved_reload(self):
        with tempfile.TemporaryDirectory(prefix='geometry_ui_test_',dir=ROOT/'results') as temporary:
            folder=Path(temporary).resolve();self.assertEqual(folder.parent,(ROOT/'results').resolve())
            source=folder/'setp2_input';self.source(source)
            root=tk.Tk();root.withdraw();app=Workbench(root,'preprocessing')
            try:
                preprocessing=app.views['preprocessing'];preprocessing.saved_path=source
                preprocessing.open_reconstruction()
                geometry=app.views['reconstruction']
                self.assertEqual(len(app.tabs.tabs()),5)
                self.wait(root,lambda:geometry.working is None)
                self.assertEqual(geometry.source,source)
                self.assertEqual(app.tabs.select(),str(app.frames['reconstruction']))
                output=folder/'setp3_output'
                with patch('geometry_window.stage_result_path',return_value=output), patch('tkinter.messagebox.showerror') as errors:
                    geometry.start();self.wait(root,lambda:geometry.working is None,60)
                    errors.assert_not_called()
                self.assertTrue(geometry.report['complete']);self.assertEqual(geometry.report['stage'],3)
                self.assertEqual(len(geometry.after.triangles),geometry.report['final']['triangles'])
                self.assertEqual({p.name for p in output.iterdir()},{'mesh.ply','report.json'})
                app.select_stage('acquisition');app.select_stage('reconstruction')
                self.assertEqual(geometry.output,output)
                with patch('tkinter.messagebox.showerror') as errors:
                    geometry.load_result(output);self.wait(root,lambda:geometry.working is None)
                    errors.assert_not_called()
                self.assertEqual(geometry.output,output)
                app.close();self.assertTrue(geometry.closed)
            finally:
                try: root.destroy()
                except tk.TclError:pass

    def test_close_cancels_direct_worker_and_waits_for_exit(self):
        root=tk.Tk();root.withdraw();app=Workbench(root,'reconstruction')
        geometry=app.views['reconstruction'];geometry.source=ROOT
        real_command=geometry._command
        def waiting_command(_):
            return real_command([sys.executable,'-u','-c',"import time;print('1/6 waiting',flush=True);time.sleep(30)"])
        try:
            with patch.object(geometry,'_command',side_effect=waiting_command):
                geometry.start();self.wait(root,lambda:geometry.process is not None)
                process=geometry.process;app.close()
                self.wait(root,lambda:geometry.closed)
                self.assertIsNotNone(process.poll())
                self.assertIsNone(geometry.working)
        finally:
            try:root.destroy()
            except tk.TclError:pass
