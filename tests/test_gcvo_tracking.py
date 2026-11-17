import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gcvo_tracking import GCVOTracking
from gcvo_backend import GCVOBackend
from reconstruct import ROOT
from test_bufferx import Matcher,Native
from scan_window import capture_result_text


GOOD=dict(overlap=.95,rmse_m=.012,translation_m=.02,rotation_deg=2.)
LOW=dict(overlap=.40,rmse_m=.033,translation_m=.2,rotation_deg=20.)


class TrackingTests(unittest.TestCase):
    def tracker(self):
        t=GCVOTracking({});t.assess({}, {},0,initial=True);t.submitted(2,0);return t

    def test_retry_is_not_itself_a_failure_and_low_streak_has_hysteresis(self):
        t=self.tracker()
        self.assertTrue(t.assess(dict(solver_status=0,retried=True),GOOD,1))
        self.assertGreater(t.confidence,.75)
        self.assertFalse(t.assess({},LOW,2));self.assertEqual(t.state,'NORMAL')
        t.assess({},GOOD,3);t.submitted(1,3);self.assertEqual(t.low_count,0)
        for i in range(3):self.assertFalse(t.assess({},LOW,4+i))
        self.assertEqual(t.state,'RECOVERY');self.assertEqual(len(t.events),1)
        for i in range(2):self.assertFalse(t.assess({},GOOD,7+i))
        self.assertTrue(t.assess({},GOOD,9));self.assertEqual(t.state,'RECOVERY')
        t.submitted(1,9);self.assertEqual(t.state,'NORMAL');self.assertEqual(len(t.events),2)

    def test_recovery_confirmation_resets_on_bad_input_or_native_rejection(self):
        t=self.tracker()
        for i in range(3):t.failure(i,'not converged')
        t.assess({},GOOD,4);t.failure(5,'occluded');self.assertEqual(t.good_count,0)
        for i in range(3):t.assess({},GOOD,6+i)
        t.submitted(0,9);self.assertEqual(t.state,'RECOVERY');self.assertEqual(t.good_count,0)
        self.assertFalse(t.assess({},GOOD,10))
        bad={**GOOD,'overlap':float('nan')}
        self.assertFalse(t.assess({},bad,11));self.assertEqual(t.confidence,0.)

    def test_startup_cannot_enter_recovery_without_a_trusted_reference(self):
        t=GCVOTracking({})
        for i in range(5):t.failure(i,'no points')
        self.assertEqual(t.state,'NORMAL');self.assertFalse(t.initialized)
        t.assess({}, {},6,initial=True);self.assertIsNone(t.confidence)
        t.submitted(2,6);self.assertTrue(t.initialized)

    def test_repeated_native_rejection_counts_even_when_candidates_score_high(self):
        t=self.tracker()
        for i in range(3):
            self.assertTrue(t.assess({},GOOD,i+1))
            t.submitted(0,i+1)
        self.assertEqual(t.low_count,3);self.assertEqual(t.state,'RECOVERY')

    def test_disabled_policy_retains_geometry_valid_candidates_and_validates_config(self):
        t=GCVOTracking({'tracking':{'enabled':False}})
        self.assertTrue(t.assess({},LOW,1))
        for i in range(5):t.failure(i,'failure')
        self.assertEqual(t.state,'NORMAL')
        for config in ({'low_frames':0},{'recovery_good_frames':True},{'good_overlap':.1},
                       {'recovery_threshold':.4},{'good_rmse_m':.04}):
            with self.assertRaises(ValueError):GCVOTracking({'tracking':config})

    def test_backend_recovery_freezes_trusted_pose_and_only_submits_confirmed_probe(self):
        class Hooks(Matcher):
            settings={'tracking':{'low_frames':3,'recovery_good_frames':3}}
            def __init__(self,folder):
                super().__init__(folder);self.fail=False;self.commits=0;self.discards=0;self.resets=0
            def estimate(self,source,target):
                if self.fail:raise ValueError('not converged')
                return super().estimate(source,target)
            def commit_candidate(self):self.commits+=1
            def discard_candidate(self):self.discards+=1
            def reference_updated(self,*args):self.resets+=1
        cloud=np.random.default_rng(1).uniform([-.2,-.2,.8],[.2,.2,1.2],(1200,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native)
            try:
                self.assertEqual(b.process(None,None,None,0,cloud),2)
                original=b.reference_odom.copy();b.matcher.fail=True
                for stamp in (1,2,3):self.assertEqual(b.process(None,None,None,stamp,cloud),0)
                self.assertEqual(b.report['tracking']['state'],'RECOVERY')
                b.matcher.fail=False;b.matcher.pose[0,3]=.01
                for stamp in (4,5):
                    self.assertEqual(b.process(None,None,None,stamp,cloud-[.01,0,0]),0)
                    self.assertEqual(len(b.client.sent),1);self.assertEqual(b.matcher.commits,1)
                    self.assertEqual(b.matcher.resets,1);np.testing.assert_array_equal(b.reference_odom,original)
                self.assertEqual(b.process(None,None,None,6,cloud-[.01,0,0]),2)
                self.assertEqual(b.report['tracking']['state'],'NORMAL')
                self.assertEqual(len(b.client.sent),2);self.assertEqual(b.matcher.resets,2)
                np.testing.assert_allclose(b.client.sent[-1],original@b.matcher.pose)
                events=[json.loads(s) for s in (Path(folder)/'gcvo_tracking_events.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual([(e['previous'],e['state']) for e in events],[('NORMAL','RECOVERY'),('RECOVERY','NORMAL')])
                self.assertNotIn('pose',b.report['tracking'])
            finally:b.close()

    def test_display_tracks_state_on_rejected_skipped_and_accepted_frames(self):
        t=self.tracker()
        for i in range(3):t.failure(i,'not converged')
        for accepted,skipped in [(False,False),(False,True),(True,False)]:
            text=capture_result_text(1,dict(accepted=accepted,skipped=skipped,method='G-CVO CPU',reason='test',tracking=t.snapshot()))
            self.assertIn('RECOVERY',text);self.assertIn('暂停建图',text);self.assertIn('0/3',text)
            self.assertIn('质量评分',text)


if __name__=='__main__':unittest.main()
