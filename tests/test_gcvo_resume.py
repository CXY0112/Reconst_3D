"""Pause boundary delivery, RGB-D seed direction and strict recovery submission."""
import queue,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import cv2,numpy as np
from reconstruct import ROOT,Calibration
from video_capture import VideoSession,VideoFrame
from gcvo_rgbd_seed import RGBDSeed
from gcvo_backend import GCVOBackend
from test_bufferx import Matcher,Native


class ResumeTests(unittest.TestCase):
    def test_rgbd_seed_estimates_reference_from_current_and_rejects_blank_or_zero_depth(self):
        image=np.random.default_rng(44).integers(0,256,(240,320,3),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),0)
        k=np.array([[250.,0,160],[0,250.,120],[0,0,1]])
        depth=np.full((240,320),1000,np.uint16)
        moved=cv2.warpAffine(image,np.float32([[1,0,-20],[0,1,0]]),(320,240))
        seed=RGBDSeed();pose,diagnostics=seed.estimate(image,depth,k,moved,k)
        np.testing.assert_allclose(pose[:3,3],[.08,0,0],atol=.002)
        np.testing.assert_allclose(pose[:3,:3],np.eye(3),atol=.002)
        self.assertGreaterEqual(diagnostics['inliers'],20)
        self.assertIsNone(RGBDSeed().estimate(image,depth*0,k,moved,k))
        self.assertIsNone(RGBDSeed().estimate(image*0,depth,k,moved*0,k))

    def test_resume_epoch_survives_drops_and_notifies_only_on_mapper_packet(self):
        session=VideoSession(queue.Queue());backend=Mock()
        backend.tracking_first=False;backend.report={'accepted':False};backend.process.return_value=1
        c=Calibration(fxir=50,fyir=50,fxrgb=50,fyrgb=50,cxir=16,cyir=16,cxrgb=16,cyrgb=16,R00=1,R11=1,R22=1)
        depth=np.full((32,32),1000,np.uint16)
        from snapshot_window import ir_geometry
        def packet(epoch):return VideoFrame(np.zeros((32,32,3),np.uint8),depth,ir_geometry(depth,c),c,1.,(1,1),False,tracking_epoch=epoch)
        try:
            session.toggle_pause();session.toggle_pause();session.toggle_pause();session.toggle_pause()
            backend.resume_tracking.assert_not_called()
            session._process_frame(backend,1,packet(0),None,None,1.)
            backend.resume_tracking.assert_not_called()
            session._process_frame(backend,2,packet(2),None,None,2.)
            session._process_frame(backend,3,packet(2),None,None,3.)
            backend.resume_tracking.assert_called_once_with(2.)
        finally:session.discard()

    def test_resume_resets_prediction_and_requires_three_verified_frames(self):
        class SeedMatcher(Matcher):
            def __init__(self,folder):super().__init__(folder);self.resets=0;self.seed_calls=0
            def reset_prediction(self):self.resets+=1
            def recovery_seed(self):return np.eye(4),{'method':'rgbd_orb_pnp'}
            def estimate(self,source,target,initial=None):
                if initial is not None:self.seed_calls+=1
                return super().estimate(source,target)
        cloud=np.random.default_rng(1).uniform([-.2,-.2,.8],[.2,.2,1.2],(1200,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=SeedMatcher,client_factory=Native)
            try:
                b.process(None,None,None,0,cloud);odom=b.reference_odom.copy()
                b.resume_tracking(10);self.assertEqual(b.tracking.state,'RECOVERY');self.assertEqual(b.matcher.resets,1)
                for stamp in (10,11):
                    self.assertEqual(b.process(None,None,None,stamp,cloud),0)
                    self.assertEqual(len(b.client.sent),1);np.testing.assert_array_equal(b.reference_odom,odom)
                self.assertEqual(b.process(None,None,None,12,cloud),2)
                self.assertEqual(b.tracking.state,'NORMAL');self.assertEqual(b.matcher.seed_calls,3)
                self.assertEqual(b.report['frontend']['initialization']['method'],'rgbd_orb_pnp')
                # A good feature seed cannot override incompatible geometry.
                b.resume_tracking(20)
                self.assertEqual(b.process(None,None,None,20,cloud+10),0)
                self.assertEqual(len(b.client.sent),2)
            finally:b.close()
