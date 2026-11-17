import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import numpy as np
from test_appearance import fixture
from reconstruct import ROOT
from saved_sources import reference, review_sources
from runtime_paths import geometry_python
from workbench import Workbench


class ReviewTests(unittest.TestCase):
    def test_saved_review_all_assets_missing_optional_and_readonly(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary); source,session=fixture(folder)
            pre=folder/'setp2_fixture'
            points=np.array([[1.,0,0],[1.,.1,.1]])
            np.save(pre/'processed_world.npy',points)
            np.save(session/'full_geometry_world.npy',points)
            output=folder/'setp4_fixture'
            run=subprocess.run([str(geometry_python()),str(ROOT/'tools/colorize_geometry.py'),'--worker',
                '--source',str(source),'--output',str(output)],capture_output=True,timeout=60)
            self.assertEqual(run.returncode,0,run.stderr)
            original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
            root=tk.Tk();root.withdraw()
            app=Workbench(root,'review');view=app.views['review']
            def wait():
                deadline=time.monotonic()+30
                while view.working and time.monotonic()<deadline:root.update();time.sleep(.01)
                self.assertIsNone(view.working)
            try:
                with patch('tkinter.messagebox.showerror') as errors:
                    view.load_result(output);wait();errors.assert_not_called()
                    self.assertEqual(set(view.left_select['values']),{'原始几何网格','预处理点云','原始点云','原始RGB图像'})
                    self.assertEqual(len(view.frames),2)
                    for mode in view.left_select['values']:
                        view.left_mode.set(mode);view.show_left();root.update()
                    self.assertIsNotNone(view.rgb.image)
                    view.change_photo(1);self.assertEqual(view.photo_select.current(),1)
                    view.mode.set('覆盖检查');view.update_mode()
                    self.assertEqual(len(app.tabs.tabs()),5)
                    for p,digest in original.items():self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
                    # A fourth-stage result remains usable if all optional sources are absent.
                    isolated=folder/'isolated';isolated.mkdir()
                    for name in ('mesh_colored.ply','color_attributes.npz'):
                        (isolated/name).write_bytes((output/name).read_bytes())
                    report=json.loads((output/'report.json').read_text(encoding='utf-8'));report.update(input='missing_geometry',source_session='missing_session')
                    (isolated/'report.json').write_text(json.dumps(report),encoding='utf-8')
                    view.load_result(isolated);wait();errors.assert_not_called()
                    self.assertEqual(view.left_select['values'],'')
                    self.assertIsNotNone(view.mesh)
                app.close();self.assertTrue(view.closed)
            finally:
                try:root.destroy()
                except tk.TclError:pass

    def test_windows_saved_reference_after_move(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            folder=Path(temp);(folder/'setp3_fixture').mkdir();(folder/'setp4_fixture').mkdir()
            self.assertEqual(reference(r'C:\old\results\setp3_fixture',folder/'setp4_fixture'),(folder/'setp3_fixture').resolve())
