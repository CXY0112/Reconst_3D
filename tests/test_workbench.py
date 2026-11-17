"""Tabbed stage lifecycle and launcher integration, without a camera."""
from pathlib import Path
import runpy
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
from reconstruct import ROOT
from workbench import Workbench


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        # Destroyed Tk cycles must be collected on the UI thread, before a new worker.
        import gc
        gc.collect()

    def wait_for(self, root, condition):
        deadline = time.monotonic() + 15
        while not condition() and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        self.assertTrue(condition())

    def test_both_main_files_choose_initial_tab_without_arguments(self):
        for relative, stage in (('main.py', 'acquisition'),
                                ('preprocessing/main.py', 'preprocessing')):
            with patch('scene_workbench.launch') as launch,patch('sys.argv',[relative]):
                runpy.run_path(str(ROOT/relative), run_name='__main__')
                if stage=='acquisition':launch.assert_called_once_with(beta=False)
                else:launch.assert_called_once_with(initial_stage='preprocessing')

    def test_tab_switch_retains_data_and_processing_result(self):
        root = tk.Tk(); root.withdraw()
        try:
            with patch('snapshot_window.SnapshotSession') as camera_session:
                app = Workbench(root, 'preprocessing')
                preprocessing = app.views['preprocessing']
                self.assertNotIn('acquisition', app.views)
                with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
                    from test_kalman_fusion import write_session
                    from reconstruct import save_cloud
                    folder=Path(temporary); write_session(folder)
                    source=folder/'map_cloud.ply'
                    points=np.load(folder/'pointcloud_raw/1.000000.npy')/1000
                    save_cloud(source,points,'X right Y down Z forward, units: m')
                    app.open_preprocessing(source, ('m', 'camera'))
                    self.wait_for(root, lambda: preprocessing.working is None)
                    np.testing.assert_allclose(preprocessing.loaded.points, points,atol=1e-6)
                    preprocessing.process()
                    scan = app.select_stage('acquisition')
                    self.assertIsNone(scan.session)
                    self.wait_for(root, lambda: preprocessing.working is None)
                    self.assertIsNotNone(preprocessing.result)
                    result = preprocessing.result
                    self.assertIs(app.select_stage('preprocessing'), preprocessing)
                    self.assertIs(preprocessing.result, result)
                    app.select_stage('acquisition')
                    scan.imported_path = source
                    scan.imported_metadata = ('m', 'camera')
                    scan.open_preprocessing()
                    self.assertEqual(app.tabs.select(), str(app.frames['preprocessing']))
                    self.assertIs(preprocessing.result, result)
                camera_session.assert_not_called()
                app.close()
                self.assertTrue(preprocessing.closed)
        finally:
            try: root.destroy()
            except tk.TclError: pass

    def test_close_cancels_processing_and_waits_for_worker(self):
        import threading
        from preprocess import ProcessingCancelled
        root = tk.Tk(); root.withdraw()
        started = threading.Event(); stopped = threading.Event()
        def processing(source, points, params, **kwargs):
            started.set()
            kwargs['cancel'].wait(5)
            stopped.set()
            raise ProcessingCancelled()
        try:
            app = Workbench(root, 'preprocessing')
            view = app.views['preprocessing']
            from types import SimpleNamespace
            view.loaded = SimpleNamespace(points=np.zeros((10, 3)))
            view.source = Path('unused.ply')
            with patch('preprocess_window.process_input', side_effect=processing):
                view.process()
                self.assertTrue(started.wait(3))
                app.close()
                self.assertTrue(view.cancel.is_set())
                self.wait_for(root, lambda: view.closed)
                self.assertTrue(stopped.is_set())
                self.assertIsNone(view.working)
        finally:
            try: root.destroy()
            except tk.TclError: pass

    def test_close_finishes_capture_and_keeps_save_prompt(self):
        from snapshot_window import SnapshotSession
        root = tk.Tk(); root.withdraw()
        session = None
        try:
            app = Workbench(root)
            scan = app.views['acquisition']
            session = SnapshotSession(scan.events)
            scan.session = session
            scan.busy = True
            with patch('tkinter.messagebox.askyesno', return_value=False) as save_prompt:
                app.close()
                self.assertTrue(session.stop.is_set())
                self.assertFalse(scan.closed)
                scan.events.put(('final', (1, np.array([[0., 0., 1.]]), np.zeros((1, 3)))))
                self.wait_for(root, lambda: scan.closed)
                save_prompt.assert_called_once()
                self.assertFalse(session.folder.exists())
        finally:
            if session is not None and session.folder.exists(): session.discard()
            try: root.destroy()
            except tk.TclError: pass


if __name__ == '__main__':
    unittest.main()
