import unittest,queue,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from reconstruct import Calibration,ROOT
from video_capture import VideoSession,VideoFrame,Cadence

class VideoCaptureTests(unittest.TestCase):
    def test_raw_zero_depth_reaches_gcvo_recovery_then_resumes_through_video_path(self):
        from rgbd_geometry import capture_geometry
        from gcvo_backend import GCVOBackend
        from test_bufferx import Native
        session=VideoSession(queue.Queue(),frontend='gcvo_cpu',optimizer='isam2',dense_mode='submaps')
        backend=None
        c=Calibration(fxir=70,fyir=80,fxrgb=50,fyrgb=50,cxir=5,cyir=6,cxrgb=16,cyrgb=16,R01=-1,R10=1,R22=1,T1=50,T3=20)
        try:
            # Real preprocessing + CPU matcher + state machine; only native IPC
            # is replaced. Unlike direct-backend tests this exercises zero depth.
            backend=GCVOBackend(session.folder,client_factory=Native)
            statuses=[]
            for i in range(7):
                depth=np.full((32,32),0 if 1<=i<=3 else 1000,np.uint16)
                points,_=capture_geometry(depth,c,depth_frame='rgb')
                # Alternate SDK-XYZ availability and RGB-grid raw fallback;
                # neither may apply the deliberately nonidentity IR extrinsic.
                packet=VideoFrame(np.zeros((32,32,3),np.uint8),depth,points,c,float(i+1),(i+1,i+1),i in (0,6),'rgb')
                status=session._process_frame(backend,i+1,packet,None,None,packet.stamp)
                session._after_process(backend,i+1,status);statuses.append(status)
                if session.dense_thread is not None:session.dense_pending.join()
                if i==3:self.assertEqual(backend.report['tracking']['state'],'RECOVERY')
                if 1<=i<=5:self.assertEqual(len(backend.client.sent),1)
            self.assertEqual(statuses,[2,0,0,0,0,0,2])
            self.assertEqual(backend.report['tracking']['state'],'NORMAL')
            self.assertFalse(session.failed);self.assertFalse(session.stop.is_set())
            self.assertIsNone(session.dense_error)
            self.assertEqual(len(backend.client.sent),2)
            np.testing.assert_allclose(backend.client.sent[0],backend.client.sent[1],atol=1e-5)
            self.assertEqual(len((session.folder/'frames.jsonl').read_text().splitlines()),7)
        finally:
            session._finish_processing()
            if backend is not None:backend.close()
            session.discard()

    def test_cadence_rejects_duplicates_and_does_not_burst_after_slow_processing(self):
        clock=Cadence(2)
        self.assertTrue(clock.take(0,(1,1),1))
        self.assertFalse(clock.take(.1,(2,2),2))
        self.assertFalse(clock.take(1,(1,3),3))
        self.assertFalse(clock.take(1,(3,3),1))
        self.assertTrue(clock.take(5,(5,5),5))
        self.assertFalse(clock.take(5.1,(6,6),6))
        for hz in (0,-1,float('nan'),11):
            with self.assertRaises(ValueError):Cadence(hz)

    def test_nominal_camera_period_tolerates_small_scheduler_jitter(self):
        clock=Cadence(10)
        for i in range(20):
            now=i*.1+(0.001 if i%2 else 0)
            self.assertTrue(clock.take(now,(i+1,i+1),now))

    def test_capture_keeps_only_latest_pending_without_writing_dropped_frames(self):
        session=VideoSession(queue.Queue(),hz=2)
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,cxir=16,cyir=16,cxrgb=16,cyrgb=16,R00=1,R11=1,R22=1)
        class Camera:
            pair_cloud=None
            def __enter__(self):self.index=0;return self
            def __exit__(self,*args):pass
            def read(self,**kwargs):
                self.index+=1
                if self.index==8:session.stop.set();raise TimeoutError()
                return ((np.full((32,32,3),self.index,np.uint8),None,self.index,self.index*1000),(np.full((32,32),1000,np.uint16),None,self.index,self.index*1000)),c
        session.camera_factory=Camera
        moments=iter([0,.2,.4,.6,.8,1.,1.2])
        try:
            with patch('video_capture.time',SimpleNamespace(monotonic=lambda:next(moments))):session._capture()
            self.assertFalse(session.failed)
            self.assertEqual(session.sampled,3);self.assertEqual(session.dropped,2)
            self.assertEqual(session.incoming.qsize(),1)
            count,packet,*_=session.incoming.get_nowait()
            self.assertEqual(count,3);self.assertEqual(packet.ids,(6,6))
            self.assertEqual(len(session.names),0)
            self.assertFalse(list((session.folder/'rgb').glob('*')))
            self.assertEqual(session.geometry,{})
            self.assertEqual(session.local_geometry,{})
            self.assertEqual(int(session.live_rgb[0,0,0]),7)
        finally:session.discard()

    def test_mapper_stores_only_retained_observation_and_explicit_ids(self):
        session=VideoSession(queue.Queue())
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,cxir=16,cyir=16,cxrgb=16,cyrgb=16,R00=1,R11=1,R22=1)
        from snapshot_window import ir_geometry
        depth=np.full((32,32),1000,np.uint16)
        packet=VideoFrame(np.zeros((32,32,3),np.uint8),depth,ir_geometry(depth,c),c,1.,(1,1),False)
        class Backend:
            report={}
            def process(self,*args,**kwargs):self.report=dict(accepted=True,node=1);return 2
        backend=Backend()
        try:
            self.assertEqual(session._process_frame(backend,5,packet,None,None,1.),2)
            self.assertEqual(session.names,['1.000000.png'])
            row=json.loads((session.folder/'frames.jsonl').read_text());self.assertEqual(row['capture'],5)
            self.assertTrue((session.folder/'pointcloud_local/1.000000.npz').exists())
            def skip(*args,**kw):backend.report=dict(accepted=False,skipped=True);return 1
            backend.process=skip
            self.assertEqual(session._process_frame(backend,9,packet,None,None,2.),1)
            self.assertFalse((session.folder/'rgb/2.000000.png').exists())
            self.assertFalse((session.folder/'pointcloud_local/2.000000.npz').exists())
            self.assertEqual(len(session.names),1)
        finally:session.discard()

    def test_threaded_video_drains_latest_frame_and_preserves_stage_two_sources(self):
        import tempfile
        from dataset_store import create_dataset
        from reconstruct import save_cloud
        from kalman_fusion import session_observations
        session=VideoSession(queue.Queue(),hz=10)
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,cxir=16,cyir=16,cxrgb=16,cyrgb=16,R00=1,R11=1,R22=1)
        class Camera:
            pair_cloud=None
            def __enter__(self):self.index=0;return self
            def __exit__(self,*args):pass
            def read(self,**kwargs):
                time.sleep(.02);self.index+=1
                if self.index==51:session.stop.set();raise TimeoutError()
                return ((np.full((32,32,3),self.index,np.uint8),None,self.index,self.index*1000),(np.full((32,32),1000,np.uint16),None,self.index,self.index*1000)),c
        class Backend:
            def __init__(self,folder):self.folder=folder;self.nodes=[];self.report={};self.timings=np.zeros(5)
            def process(self,*args,**kwargs):
                time.sleep(.3);self.nodes.append(len(self.nodes)+1)
                self.report=dict(accepted=True,node=self.nodes[-1],reference=None,overlap=1,rmse_mm=0)
                (self.folder/'geometry_poses.json').write_text(json.dumps(dict(node_ids=self.nodes,poses=[np.eye(4).tolist() for _ in self.nodes],edges=[])))
                return 2
            def preview(self):return np.array([[0,0,1.]]),np.zeros((len(self.nodes),3))
            def close(self):save_cloud(self.folder/'map_cloud.ply',np.array([[1.,0,0]]),'world metres')
        session.camera_factory=Camera;session.backend_factory=Backend
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            try:
                session.start();session.capture_thread.join(5);session.map_thread.join(10)
                self.assertFalse(session.capture_thread.is_alive());self.assertFalse(session.map_thread.is_alive())
                self.assertFalse(session.failed)
                self.assertGreater(session.dropped,0)
                self.assertEqual(session.sampled,session.processed+session.dropped)
                self.assertFalse(session.geometry);self.assertFalse(session.local_geometry)
                self.assertEqual(len(list((session.folder/'rgb').glob('*.png'))),session.processed)
                folder=create_dataset(temp,'video');session.save(dataset=folder)
                _,loaded,_=session_observations(folder/'setp1')
                self.assertEqual(len(loaded),session.processed)
                ids=[row['capture'] for row in loaded]
                self.assertEqual(len(set(ids)),len(ids));self.assertGreater(ids[-1],len(ids))
            finally:
                session.stop.set()
                if session.folder.name.startswith('.pending_scan_'):session.discard()

    def test_pause_keeps_preview_running_without_sampling(self):
        session=VideoSession(queue.Queue());session.toggle_pause()
        c=Calibration(fxir=50,fyir=50,R00=1,R11=1,R22=1)
        class Camera:
            pair_cloud=None
            def __enter__(self):self.i=0;return self
            def __exit__(self,*args):pass
            def read(self,**kwargs):
                self.i+=1
                if self.i==4:session.stop.set();raise TimeoutError()
                return ((np.full((8,8,3),self.i,np.uint8),None,self.i,self.i*1000),(np.full((8,8),1000,np.uint16),None,self.i,self.i*1000)),c
        session.camera_factory=Camera
        moments=iter([0,.2,.4])
        try:
            with patch('video_capture.time',SimpleNamespace(monotonic=lambda:next(moments))):session._capture()
            self.assertEqual(session.sampled,0);self.assertTrue(session.incoming.empty())
            self.assertEqual(int(session.live_rgb[0,0,0]),3)
            self.assertFalse(session.toggle_pause());self.assertFalse(session.paused.is_set())
        finally:session.discard()
