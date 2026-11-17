"""Alternative recovery can submit while every G-CVO call is failing."""
import tempfile,unittest
from unittest.mock import patch
import cv2,numpy as np
from reconstruct import ROOT
from gcvo_backend import GCVOBackend
from gcvo_matcher import GCVOMatcher
from gcvo_fallback import refine_icp
from test_bufferx import Native,Matcher
from scan_window import capture_result_text


class FallbackTests(unittest.TestCase):
    def test_real_rgbd_recovers_without_gcvo_then_returns_to_gcvo(self):
        rgb=cv2.GaussianBlur(np.random.default_rng(44).integers(0,256,(240,320,3),dtype=np.uint8),(3,3),0)
        k=np.array([[250.,0,160],[0,250.,120],[0,0,1.]])
        depth=np.full((240,320),1000,np.uint16);moving_depth=depth.copy();moving_depth[:,-30:]=0
        current=cv2.warpAffine(rgb,np.float32([[1,0,-30],[0,1,0]]),(320,240))
        v,u=np.mgrid[:240,:320]
        xyz=np.stack(((u-160)/250,(v-120)/250,np.ones_like(u)),axis=-1)
        fixed=xyz.reshape(-1,3);moving=xyz[moving_depth>0]
        def factory(folder):
            m=GCVOMatcher(folder);m.settings={**m.settings,'mac':{'enabled':False}};return m
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=factory,client_factory=Native)
            try:
                b.process(rgb,depth,k,0,fixed);b.resume_tracking(1)
                with patch.object(b.matcher,'estimate',side_effect=ValueError('forced G-CVO nonconvergence')):
                    for stamp in (1,2):
                        self.assertEqual(b.process(current,moving_depth,k,stamp,moving),0)
                        self.assertEqual(len(b.client.sent),1)
                        self.assertEqual(b.report['method'],'RGB-D ORB/PnP')
                    self.assertEqual(b.process(current,moving_depth,k,3,moving),2)
                self.assertEqual(b.tracking.state,'NORMAL');self.assertEqual(len(b.client.sent),2)
                self.assertTrue(b.report['frontend']['independent_recovery'])
                self.assertNotIn('solver_status',b.report['frontend'])
                # The next NORMAL frame really runs the unmodified CPU solver.
                self.assertEqual(b.process(current,moving_depth,k,4,moving),2)
                self.assertEqual(b.report['method'],'G-CVO CPU');self.assertEqual(b.report['frontend']['solver_status'],0)
            finally:b.close()

    def test_mac_icp_refines_current_cloud_not_old_worker_pose(self):
        from gcvo_fallback import register
        rng=np.random.default_rng(8);target=rng.uniform([-.4,-.3,.8],[.4,.3,1.5],(5000,3))
        source=target-[.035,0,0];initial=np.eye(4);initial[0,3]=.02
        result=register(source,target,initial,{'method':'mac'},{})
        self.assertEqual(result['registration_method'],'MAC + ICP')
        np.testing.assert_allclose(np.asarray(result['pose'])[:3,3],[.035,0,0],atol=.001)
        with self.assertRaises(ValueError):refine_icp(source+10,target,initial)

    def test_mac_independently_recovers_when_every_gcvo_call_fails(self):
        from gcvo_fallback import register
        class Hooks(Matcher):
            settings={'mac':{'enabled':True}}
            def estimate(self,*args,**kwargs):raise ValueError('G-CVO always fails')
            def estimate_fallback(self,source,target,initial,diagnostics):
                return register(source,target,initial,diagnostics,self.settings)
        class MAC:
            def __init__(self,*args):pass
            def poll(self,*args):
                pose=np.eye(4);pose[0,3]=.02;return pose
            def reset(self):pass
            def reject_seed(self,error):raise AssertionError(error)
            def snapshot(self):return {'status':'seed_ready'}
            def close(self):pass
        cloud=np.random.default_rng(8).uniform([-.4,-.3,.8],[.4,.3,1.5],(5000,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native,mac_factory=MAC)
            try:
                b.process(None,None,None,0,cloud);b.resume_tracking(1)
                for stamp in (1,2):self.assertEqual(b.process(None,None,None,stamp,cloud-[.035,0,0]),0)
                self.assertEqual(len(b.client.sent),1)
                self.assertEqual(b.process(None,None,None,3,cloud-[.035,0,0]),2)
                self.assertEqual(b.report['method'],'MAC + ICP');self.assertEqual(b.tracking.state,'NORMAL')
            finally:b.close()

    def test_rgbd_color_match_cannot_override_conflicting_depth(self):
        from gcvo_rgbd_seed import RGBDSeed
        image=cv2.GaussianBlur(np.random.default_rng(44).integers(0,256,(240,320,3),dtype=np.uint8),(3,3),0)
        depth=np.full((240,320),1000,np.uint16);k=np.array([[250.,0,160],[0,250.,120],[0,0,1.]])
        seed=RGBDSeed();pose,_=seed.estimate(image,depth,k,image,k)
        self.assertIsNotNone(seed.verify_depth(pose,depth))
        self.assertIsNone(seed.verify_depth(pose,depth*2))

    def test_inconsistent_fallbacks_do_not_accumulate_confirmation(self):
        class Hooks(Matcher):pass
        cloud=np.random.default_rng(2).uniform([-.2,-.2,.8],[.2,.2,1.2],(1000,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native)
            try:
                b.process(None,None,None,0,cloud);b.resume_tracking(1)
                def candidate(source,target,stamp):
                    pose=np.eye(4);pose[0,3]=.1 if stamp%2 else 0
                    return dict(pose=pose.tolist(),inliers=100,independent_recovery=True,registration_method='MAC + ICP')
                b._estimate=candidate
                for stamp in (1,2,3,4):
                    moved=cloud-[.1 if stamp%2 else 0,0,0]
                    self.assertEqual(b.process(None,None,None,stamp,moved),0)
                self.assertEqual(len(b.client.sent),1);self.assertLessEqual(b.tracking.good_count,1)
                self.assertIn('MAC + ICP',capture_result_text(4,b.report))
            finally:b.close()
