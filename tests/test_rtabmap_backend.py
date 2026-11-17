import json,tempfile,unittest,queue,threading,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rtabmap_backend import RtabmapBackend
from video_capture import VideoSession,VideoFrame
from reconstruct import Calibration,ROOT


class RtabBackendTests(unittest.TestCase):
    def test_large_loop_corrections_are_budgeted_but_final_map_is_complete(self):
        class Client:
            def __init__(self,folder):pass
            def close(self,**kwargs):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=RtabmapBackend(folder,client_factory=Client)
            try:
                for node in range(1,4):
                    with backend.lock:backend._poses(np.array([[node,*np.eye(4)[:3].ravel()]]))
                    backend.retain(node,np.array([[node*.1,0,1.]]))
                backend.preview()
                rows=[]
                for node in range(1,4):
                    pose=np.eye(4);pose[0,3]=.02;rows.append([node,*pose[:3].ravel()])
                with backend.lock:backend._poses(np.array(rows))
                backend.preview()
                self.assertEqual(backend.preview_stats['updated_nodes'],1)
                self.assertEqual(backend.preview_stats['pending_pose_nodes'],2)
                final,_=backend.final_preview()
                self.assertEqual(backend.preview_stats['updated_nodes'],2)
                self.assertEqual(backend.preview_stats['pending_pose_nodes'],0)
                np.testing.assert_allclose(np.sort(final[:,2]),[.12,.22,.32])
            finally:backend.close()

    def test_native_save_reuses_fused_geometry_and_writes_retained_trajectory(self):
        from dataset_store import create_dataset
        session=VideoSession(queue.Queue())
        try:
            session.k=np.eye(3);session.shape=(2,2);session.skip_native_cloud_export=True
            session.final_world_cloud=np.array([[1.,0,0]])
            (session.folder/'map.db').write_bytes(b'reserved database fixture')
            (session.folder/'sdk_calibration.json').write_text('{}')
            optical=np.array([[0,0,1.,0],[-1.,0,0,0],[0,-1.,0,0],[0,0,0,1.]])
            (session.folder/'geometry_poses.json').write_text(json.dumps(dict(node_ids=[7],poses=[optical.tolist()])))
            (session.folder/'frames.jsonl').write_text(json.dumps(dict(capture=11,stamp=2.,cloud_source='local_surface_v1'))+'\n')
            (session.folder/'registrations.jsonl').write_text(json.dumps(dict(capture=11,node=7,accepted=True))+'\n')
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
                dataset=create_dataset(temp,'native')
                with patch('scan_window.export_map',side_effect=AssertionError('redundant native cloud export')):
                    saved=session.save(dataset=dataset)
                self.assertTrue((saved/'map_cloud.ply').is_file())
                self.assertFalse((saved/'map_cloud_rgb_registered.ply').exists())
                rows=np.loadtxt(saved/'map_poses.txt');np.testing.assert_allclose(rows,[2.,0,0,0,0,0,0,1,7],atol=1e-8)
        finally:
            if session.folder.name.startswith('.pending_scan_'):session.discard()

    def test_small_pose_changes_accumulate_and_final_flush_uses_latest_pose(self):
        class Client:
            def __init__(self,folder):self.i=0
            def frame(self,*args):
                p=np.eye(4);p[0,3]=self.i*.0002;self.i+=1
                return dict(status=2 if self.i==1 else 1,node=7 if self.i==1 else 0,poses=np.array([[7,*p[:3].ravel()]]),timings=np.zeros(5),reason='')
            def close(self,**kwargs):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=RtabmapBackend(folder,client_factory=Client)
            cloud=np.array([[0.,0,1.]])
            try:
                backend.process(None,None,None,1,cloud);backend.retain(7,cloud)
                first,_=backend.preview()
                for _ in range(3):
                    backend.process(None,None,None,1,cloud);p,_=backend.preview()
                    self.assertIs(p,first);self.assertEqual(backend.preview_stats['updated_nodes'],0)
                backend.process(None,None,None,1,cloud);p,_=backend.preview()
                self.assertEqual(backend.preview_stats['updated_nodes'],1)
                self.assertAlmostEqual(p[0,2],.0008)
                backend.process(None,None,None,1,cloud);same,_=backend.preview();self.assertIs(same,p)
                final,_=backend.preview(force=True)
                self.assertAlmostEqual(final[0,2],.001)
                self.assertFalse(final.flags.writeable)
            finally:backend.close()

    def test_rotation_bound_accounts_for_points_far_from_camera(self):
        from scipy.spatial.transform import Rotation
        class Client:
            def __init__(self,folder):pass
            def close(self,**kwargs):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=RtabmapBackend(folder,client_factory=Client)
            try:
                with backend.lock:backend._poses(np.array([[1,*np.eye(4)[:3].ravel()]]))
                backend.retain(1,np.array([[0.,0,4.]]));backend.preview()
                p=np.eye(4);p[:3,:3]=Rotation.from_rotvec([0,.0004,0]).as_matrix()
                with backend.lock:backend._poses(np.array([[1,*p[:3].ravel()]]))
                backend.preview();self.assertEqual(backend.preview_stats['updated_nodes'],1)
            finally:backend.close()

    def test_native_pose_correction_replaces_old_voxel_contributions(self):
        class Client:
            def __init__(self,folder):self.i=0
            def frame(self,*args):
                self.i+=1
                p=np.eye(4);p[0,3]=.1 if self.i>1 else 0
                return dict(status=2 if self.i==1 else 1,node=7 if self.i==1 else 0,
                    poses=np.array([[7,*p[:3].ravel()]]),timings=np.zeros(5),reason='')
            def close(self,**kwargs):pass
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=RtabmapBackend(folder,client_factory=Client)
            try:
                rgb=np.zeros((2,2,3),np.uint8);dep=np.ones((2,2),np.uint16);k=np.eye(3)
                cloud=np.array([[0.,0.,1.],[.001,0,1.]])
                self.assertEqual(backend.process(rgb,dep,k,1.,cloud),2)
                backend.retain(7,cloud)
                a,_=backend.preview()
                self.assertEqual(backend.process(rgb,dep,k,2.,cloud),1)
                b,_=backend.preview()
                self.assertEqual(len(b),len(a));np.testing.assert_allclose(b-a,[[0,0,.1]])
                graph=json.loads((Path(folder)/'geometry_poses.json').read_text())
                self.assertEqual(graph['node_ids'],[7]);self.assertAlmostEqual(graph['poses'][0][0][3],.1)
            finally:backend.close()

    def test_tracking_only_does_not_run_local_surface_fit_or_save_images(self):
        session=VideoSession(queue.Queue())
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,R00=1,R11=1,R22=1)
        packet=VideoFrame(np.zeros((16,16,3),np.uint8),np.full((16,16),1000,np.uint16),np.array([[0.,0.,1.]]),c,1.,(1,1),False)
        class Backend:
            tracking_first=True
            report=dict(accepted=False,skipped=True)
            def process(self,*a,**kw):return 1
        try:
            with patch('local_preprocessing.prepare_local',side_effect=AssertionError('fit on tracking frame')):
                self.assertEqual(session._process_frame(Backend(),1,packet,None,None,1.),1)
            self.assertFalse(list((session.folder/'rgb').glob('*')))
            self.assertFalse(list((session.folder/'pointcloud_local').glob('*')))
            self.assertIs(session.backend_factory,RtabmapBackend)
        finally:session.discard()

    def test_dense_queue_is_bounded_and_does_not_block_tracking(self):
        from snapshot_window import ir_geometry
        from local_preprocessing import prepare_local
        session=VideoSession(queue.Queue());entered=threading.Event();release=threading.Event()
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,cxir=16,cyir=16,cxrgb=16,cyrgb=16,R00=1,R11=1,R22=1)
        dep=np.full((32,32),1000,np.uint16);points=ir_geometry(dep,c)
        class Backend:
            tracking_first=True
            def __init__(self):self.i=0;self.nodes=[]
            def process(self,*a,**kw):
                self.i+=1;self.report=dict(accepted=True,node=self.i,reference=None);return 2
            def retain(self,node,geometry):self.nodes.append(node)
            def preview(self):return np.array([[0.,0,1.]]),np.zeros((len(self.nodes),3))
        backend=Backend()
        def slow(*a,**kw):entered.set();release.wait(5);return prepare_local(*a,**kw)
        try:
            with patch('local_preprocessing.prepare_local',side_effect=slow):
                for i in range(3):
                    packet=VideoFrame(np.zeros((32,32,3),np.uint8),dep,points,c,float(i+1),(i+1,i+1),False)
                    t=time.monotonic();status=session._process_frame(backend,i+1,packet,None,None,packet.stamp)
                    self.assertLess(time.monotonic()-t,1)
                    if i==0:self.assertTrue(entered.wait(2))
                    self.assertEqual(status,1 if i==2 else 2)
                    session._after_process(backend,i+1,status)
                self.assertEqual(session.dense_pending.qsize(),1)
                self.assertEqual(session.dense_dropped,1)
                release.set();session._finish_processing()
            self.assertEqual(backend.nodes,[1,2]);self.assertEqual(len(session.names),2)
        finally:
            release.set();session._finish_processing();session.discard()
