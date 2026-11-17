import json,queue,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import numpy as np
from reconstruct import ROOT
from video_capture import VideoSession
from rtabmap_backend import RtabmapBackend
from rtab_client import RtabClient


class Isam2Tests(unittest.TestCase):
    def test_selected_optimizer_reaches_native_client_and_saved_geometry(self):
        calls=[]
        class Client:
            def __init__(self,folder,**kwargs):calls.append(kwargs)
            def close(self,**kwargs):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=RtabmapBackend(folder,client_factory=Client,optimizer='isam2')
            try:
                backend._poses(np.array([[1,*np.eye(4)[:3].ravel()]]))
                self.assertEqual(calls,[{'optimizer':'isam2'}])
                self.assertEqual(json.loads((Path(folder)/'geometry_poses.json').read_text())['optimizer'],'isam2')
            finally:backend.close()

    def test_session_records_selection_and_keeps_default_backend_unchanged(self):
        for optimizer in ('default','isam2'):
            session=VideoSession(queue.Queue(),optimizer=optimizer)
            try:
                self.assertEqual(json.loads((session.folder/'video_capture.json').read_text())['optimizer'],optimizer)
                if optimizer=='default':self.assertIs(session.backend_factory,RtabmapBackend)
                else:
                    with patch('rtabmap_backend.RtabClient'):
                        # Factory arguments are fixed at construction, before any worker starts.
                        self.assertEqual(session.backend_factory.keywords,{'optimizer':'isam2'})
            finally:session.discard()

    def test_native_handshake_must_confirm_isam2_no_silent_fallback(self):
        for incremental in (True,False):
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
                process=MagicMock();process.poll.return_value=0
                info=dict(requested='isam2',strategy=2,incremental=incremental,verified=True)
                response=dict(status=1,reason=json.dumps(info))
                with patch('rtab_client.subprocess.Popen',return_value=process) as launch, \
                     patch('rtab_client.threading.Thread'),patch.object(RtabClient,'_receive',return_value=response):
                    if incremental:
                        client=RtabClient(folder,optimizer='isam2');client.close(force=True)
                        self.assertEqual(json.loads((Path(folder)/'optimizer.json').read_text()),info)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'未确认启用'):RtabClient(folder,optimizer='isam2')
                        self.assertFalse((Path(folder)/'optimizer.json').exists())
                    self.assertEqual(launch.call_args.args[0][-1],'isam2')
                    self.assertEqual(launch.call_count,1)

    def test_rgbd_ui_selects_optimizer_and_locks_during_capture(self):
        import tkinter as tk
        from scene_cloud_window import SceneWorkbench
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        try:
            view=app.views['acquisition']
            self.assertEqual(view.graph_optimizer.get(),'iSAM2（增量优化）')
            with patch('video_capture.VideoSession') as create:
                view.create_session()
                self.assertEqual(create.call_args.kwargs['optimizer'],'isam2')
                self.assertEqual(create.call_args.kwargs['dense_mode'],'submaps')
            view.busy=True
            view.poll()
            self.assertFalse(hasattr(view,'optimizer_box'))
            view.busy=False;view.poll()
            self.assertEqual(view.graph_optimizer.get(),'iSAM2（增量优化）')
        finally:
            view.busy=False;app.close()


if __name__=='__main__':unittest.main()
