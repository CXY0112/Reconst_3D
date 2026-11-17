import queue
import unittest
from unittest.mock import patch
import numpy as np
from scan_window import ScanSession, ScanWindow, read_cloud
from reconstruct import ROOT


class WindowBackendTests(unittest.TestCase):
    def test_finish_waits_for_map_and_declining_save_cleans_session(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        window = ScanWindow(root)
        session = ScanSession(window.events)
        window.session, window.busy = session, True
        temporary = session.folder
        try:
            window.finish()
            self.assertTrue(session.stop.is_set())
            self.assertTrue(temporary.exists())
            points = np.array([[0., 0., 1.], [.1, .1, 1.]])
            window.events.put(('final', (2, points, np.zeros((1, 3)))))
            with patch('tkinter.messagebox.askyesno', return_value=False) as prompt:
                window.poll()
                prompt.assert_called_once()
            self.assertFalse(window.busy)
            self.assertFalse(temporary.exists())
        finally:
            if temporary.exists():
                session.discard()
            root.destroy()

    def test_latest_frame_queue_is_bounded(self):
        session = ScanSession(queue.Queue())
        try:
            for number in range(10):
                session.submit(number)
            self.assertEqual(session.incoming.qsize(), 2)
            self.assertEqual(session.dropped, 8)
            self.assertEqual(session.incoming.get_nowait(), 8)
            self.assertEqual(session.incoming.get_nowait(), 9)
        finally:
            session.discard()

    def test_incremental_backend_is_created_once_and_each_frame_processed_once(self):
        class FakeBackend:
            def __init__(self, folder):
                self.calls = []
                self.closed = 0
            def process(self, color, depth, k, stamp):
                self.calls.append(stamp)
                return 1
            def preview(self):
                return np.array([[0., 0., 1.]]), np.zeros((1,3))
            def close(self):
                self.closed += 1
        events = queue.Queue()
        backend = FakeBackend(None)
        factory = unittest.mock.Mock(return_value=backend)
        session = ScanSession(events, backend_factory=factory)
        try:
            session.names = ['1.png', '2.png']
            for i in (1,2):
                session.submit((i, None, None, None, float(i)))
            session.capture_done.set()
            session._mapping()
            factory.assert_called_once()
            self.assertEqual(backend.calls, [1.,2.])
            self.assertEqual(backend.closed, 1)
            self.assertFalse(session.failed)
            self.assertEqual([event[0] for event in list(events.queue)].count('final'), 1)
        finally:
            session.discard()

    def test_save_rewrites_paths_and_discard_refuses_saved_scan(self):
        session = ScanSession(queue.Queue())
        session.k, session.shape = np.eye(3), (2, 2)
        saved = session.save()
        try:
            self.assertTrue(saved.name.startswith('setp1_'))
            self.assertIn(saved.as_posix(), (saved/'source.ini').read_text())
            with self.assertRaises(ValueError):
                session.discard()
        finally:
            # Remove only the generated test files and empty directories.
            (saved/'source.ini').unlink()
            (saved/'hp60c.yaml').unlink()
            (saved/'rgb').rmdir()
            (saved/'depth').rmdir()
            saved.rmdir()

    def test_existing_export_loads_world_coordinates(self):
        maps = list((ROOT/'results').glob('scan_*/map_cloud.ply'))
        if not maps:
            self.skipTest('No real export available')
        points = read_cloud(maps[0])
        self.assertEqual(points.shape[1], 3)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.isfinite(points).all())


if __name__ == '__main__':
    unittest.main()
