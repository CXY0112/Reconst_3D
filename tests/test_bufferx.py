import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import queue
import io
import numpy as np
from bufferx_backend import BufferXBackend,OPTICAL_WORLD,validate_registration
from reconstruct import ROOT


class Matcher:
    settings={}
    def __init__(self,folder):self.calls=[];self.pose=np.eye(4);self.closed=False
    def estimate(self,source,target):
        self.calls.append((source.copy(),target.copy()))
        return dict(pose=self.pose.tolist(),inliers=50,matches=60,points=[len(source),len(target)])
    def close(self):self.closed=True


class Native:
    def __init__(self,folder,**kwargs):
        self.optimizer_info=dict(external_pose=True);self.rows=[];self.sent=[];self.status=2;self.correction=0
    def frame(self,color,depth,k,stamp,pose=None,variance=None):
        assert pose is not None,'Visual odometry must never be called'
        self.sent.append(pose.copy());node=len(self.rows)+1
        if self.status==2:self.rows.append([node,*pose[:3].ravel()])
        rows=np.array(self.rows);rows[:,4]+=self.correction
        return dict(status=self.status,node=node,poses=rows,timings=np.zeros(5),reason='')
    def close(self,**kwargs):pass


class BufferXTests(unittest.TestCase):
    def setUp(self):self.cloud=np.random.default_rng(1).uniform([-.4,-.3,.8],[.4,.3,1.6],(4000,3))

    def test_source_to_target_transform_and_loop_correction_not_double_applied(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=BufferXBackend(folder,matcher_factory=Matcher,client_factory=Native)
            try:
                self.assertEqual(backend.process(None,None,None,1,self.cloud),2)
                backend.retain(1,self.cloud)
                relative=np.eye(4);relative[0,3]=.04;backend.matcher.pose=relative
                backend.client.correction=.3
                self.assertEqual(backend.process(None,None,None,2,self.cloud-[.04,0,0]),2)
                np.testing.assert_allclose(backend.client.sent[-1],OPTICAL_WORLD@relative)
                self.assertEqual(backend.report['reference'],1)
                backend.retain(2,self.cloud-[.04,0,0])
                backend.process(None,None,None,3,self.cloud-[.08,0,0])
                np.testing.assert_allclose(backend.client.sent[-1],OPTICAL_WORLD@relative@relative)
                backend.final_preview();backend.pose_writer.flush()
                info=json.loads((Path(folder)/'geometry_poses.json').read_text())
                self.assertEqual(info['backend'],'bufferx');self.assertEqual(info['loop_closure'],'rtabmap_visual')
            finally:backend.close()

    def test_rejection_does_not_advance_reference_or_touch_native_graph(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=BufferXBackend(folder,matcher_factory=Matcher,client_factory=Native)
            try:
                backend.process(None,None,None,1,self.cloud)
                backend.matcher.pose[0,3]=2
                self.assertEqual(backend.process(None,None,None,2,self.cloud),0)
                self.assertEqual(len(backend.client.sent),1);self.assertEqual(backend.reference_node,1)
                self.assertFalse(backend.report['accepted'])
            finally:backend.close()

    def test_low_motion_keeps_last_native_reference(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=BufferXBackend(folder,matcher_factory=Matcher,client_factory=Native)
            try:
                backend.process(None,None,None,1,self.cloud)
                backend.client.status=1
                self.assertEqual(backend.process(None,None,None,1.1,self.cloud),1)
                self.assertEqual(backend.reference_node,1)
            finally:backend.close()

    def test_invalid_transform_and_nonoverlap_are_rejected(self):
        with self.assertRaises(ValueError):validate_registration(self.cloud,self.cloud+10,dict(pose=np.eye(4),inliers=30),{})
        bad=np.eye(4);bad[0,0]=-1
        with self.assertRaises(ValueError):validate_registration(self.cloud,self.cloud,dict(pose=bad,inliers=30),{})

    def test_frontend_selection_and_saved_metadata(self):
        from video_capture import VideoSession
        session=VideoSession(queue.Queue(),optimizer='isam2',dense_mode='submaps',frontend='bufferx')
        try:
            self.assertIs(session.backend_factory.func,BufferXBackend)
            info=json.loads((session.folder/'video_capture.json').read_text());self.assertEqual(info['frontend'],'bufferx')
        finally:session.discard()

    def test_external_pose_protocol_and_invalid_pose_before_write(self):
        from rtab_client import RtabClient
        from types import SimpleNamespace
        import struct
        client=RtabClient.__new__(RtabClient)
        client.optimizer_info=dict(external_pose=True);client.process=SimpleNamespace(stdin=io.BytesIO())
        client._receive=lambda:dict(status=2)
        color=np.zeros((2,3,3),np.uint8);depth=np.ones((2,3),np.uint16)
        client.frame(color,depth,np.eye(3),1.,pose=OPTICAL_WORLD)
        data=client.process.stdin.getvalue()
        self.assertEqual(struct.unpack('<i',data[:4])[0],4)
        np.testing.assert_array_equal(np.frombuffer(data[-104:-8],dtype='<f8').reshape(3,4),OPTICAL_WORLD[:3])
        before=len(data)
        with self.assertRaises(ValueError):client.frame(color,depth,np.eye(3),2.,pose=np.zeros((4,4)))
        self.assertEqual(len(client.process.stdin.getvalue()),before)

    def test_frontend_selector_is_available_in_beta_and_locked_during_scan(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        for beta in (False,True):
            root=tk.Tk();root.withdraw();app=SceneWorkbench(root,beta=beta)
            try:
                view=app.views['acquisition'];root.update_idletasks()
                self.assertEqual(view.frontend_row.winfo_manager(),'pack')
                view.registration_frontend.set('BUFFER-X（实验／CPU）')
                with patch('bufferx_client.runtime_settings',return_value={}),patch('video_capture.VideoSession') as factory:
                    view.create_session();self.assertEqual(factory.call_args.kwargs['frontend'],'bufferx')
                view.busy=True;view.poll();self.assertEqual(str(view.frontend_box.cget('state')),'disabled')
                view.busy=False;view.poll();self.assertEqual(str(view.frontend_box.cget('state')),'readonly')
            finally:
                view.busy=False;app.close()


if __name__=='__main__':unittest.main()
