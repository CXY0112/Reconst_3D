import ctypes as ct
from contextlib import contextmanager
import errno,json,os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from reconstruct import ROOT
from file_snapshots import AtomicJsonWriter,open_shared_binary,read_shared_json,_windows_files


@contextmanager
def deny_delete(path):
    kernel=_windows_files()
    handle=kernel.CreateFileW(str(Path(path).resolve()),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ct.c_void_p(-1).value:raise ct.WinError(ct.get_last_error())
    try:yield
    finally:kernel.CloseHandle(handle)


class SnapshotFileTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_reproduces_old_replace_failure_then_defers_and_commits_latest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            path=Path(temp)/'geometry_poses.json';path.write_text('{"version":0}')
            writer=AtomicJsonWriter(path)
            with deny_delete(path):
                old=Path(temp)/'old.pending';old.write_text('{"version":-1}')
                with self.assertRaises(PermissionError):old.replace(path)
                old.unlink()
                self.assertFalse(writer.update(dict(version=1)))
                self.assertFalse(writer.update(dict(version=2)))
                self.assertEqual(read_shared_json(path),dict(version=0))
                self.assertEqual(read_shared_json(writer.pending),dict(version=2))
            self.assertTrue(writer.flush());self.assertEqual(read_shared_json(path),dict(version=2))
            self.assertIsNone(writer.pending);self.assertFalse(list(Path(temp).glob('*.pending')))

    def test_shared_reader_keeps_old_snapshot_while_replacement_succeeds(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            path=Path(temp)/'poses.json';writer=AtomicJsonWriter(path)
            writer.update(dict(version=0,label='原始位姿'))
            with open_shared_binary(path) as reader:
                self.assertTrue(writer.update(dict(version=1,label='最新位姿')))
                self.assertEqual(json.loads(reader.read()),dict(version=0,label='原始位姿'))
            self.assertEqual(read_shared_json(path),dict(version=1,label='最新位姿'))

    def test_concurrent_readers_never_observe_partial_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            path=Path(temp)/'poses.json';writer=AtomicJsonWriter(path);writer.update(dict(version=0,poses=[0]*300))
            stop=threading.Event();errors=[];reads=[]
            def reader():
                try:
                    while not stop.is_set():
                        value=read_shared_json(path)
                        if value['poses']!=[value['version']]*300:raise AssertionError('Mixed snapshot')
                        reads.append(1)
                except Exception as exc:errors.append(exc)
            threads=[threading.Thread(target=reader) for _ in range(3)]
            for thread in threads:thread.start()
            try:
                for version in range(1,151):writer.update(dict(version=version,poses=[version]*300))
                writer.flush()
            finally:
                stop.set()
                for thread in threads:thread.join(3)
            self.assertFalse(errors,errors);self.assertTrue(reads)
            self.assertEqual(read_shared_json(path)['version'],150)

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_final_flush_waits_for_short_external_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            path=Path(temp)/'poses.json';path.write_text('{"version":0}')
            writer=AtomicJsonWriter(path);opened=threading.Event()
            def blocker():
                with deny_delete(path):opened.set();time.sleep(.15)
            thread=threading.Thread(target=blocker);thread.start();self.assertTrue(opened.wait(2))
            try:
                self.assertFalse(writer.update(dict(version=1)))
                self.assertTrue(writer.flush(timeout=1))
                self.assertEqual(read_shared_json(path)['version'],1)
            finally:thread.join(2)

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_persistent_lock_raises_at_finalization_and_preserves_pending_data(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            path=Path(temp)/'poses.json';path.write_text('{"version":0}');writer=AtomicJsonWriter(path)
            with deny_delete(path):
                self.assertFalse(writer.update(dict(version=1)))
                with self.assertRaises(PermissionError):writer.flush(timeout=.03)
                self.assertEqual(read_shared_json(path)['version'],0)
                self.assertEqual(read_shared_json(writer.pending)['version'],1)
            writer.flush()

    def test_unexpected_io_errors_are_not_hidden(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            writer=AtomicJsonWriter(Path(temp)/'poses.json')
            with patch.object(Path,'replace',side_effect=OSError(errno.EIO,'injected I/O failure')):
                with self.assertRaises(OSError):writer.update(dict(version=1))
            self.assertIsNotNone(writer.pending);writer.flush()

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_live_submap_tracking_survives_target_lock_and_close_flushes_latest_pose(self):
        from submap_backend import SubmapBackend
        class Client:
            def __init__(self,folder):self.index=0;self.closed=False
            def frame(self,*args):
                pose=np.eye(4);pose[0,3]=self.index*.001;self.index+=1
                return dict(status=2 if self.index==1 else 1,node=1 if self.index==1 else 0,
                    poses=np.array([[1,*pose[:3].ravel()]]),timings=np.zeros(5),reason='')
            def close(self,**kwargs):self.closed=True
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            backend=SubmapBackend(temp,client_factory=Client);cloud=np.array([[0.,0,1.]])
            try:
                backend.process(None,None,None,1,cloud);backend.retain(1,cloud)
                with deny_delete(Path(temp)/'geometry_poses.json'):
                    for i in range(20):self.assertEqual(backend.process(None,None,None,i+2,cloud),1)
                    self.assertTrue(backend.report['pose_checkpoint_pending'])
                    self.assertEqual(backend.poses[1][0,3],.020)
                backend.close()
                value=read_shared_json(Path(temp)/'geometry_poses.json')
                self.assertEqual(value['node_ids'],[1]);self.assertEqual(value['poses'][0][0][3],.020)
                self.assertTrue(backend.client.closed)
            finally:backend.close()


if __name__=='__main__':unittest.main()
