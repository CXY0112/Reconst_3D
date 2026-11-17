"""Bounded continuation must converge and pass geometry before touching tracking."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from gcvo_matcher import GCVOSolver,GCVOConvergenceError,align_tracking,settings_for
from gcvo_backend import GCVOBackend
from reconstruct import ROOT
from test_bufferx import Matcher,Native
from test_gcvo import pack


def outcome(status,iterations,step=.001,ell=.014,pose=None):
    return dict(pose=np.eye(4) if pose is None else pose,solver_status=status,
                iterations=iterations,solver_ms=10.,step_norm=step,final_ell=ell)


class GCVOConvergenceTests(unittest.TestCase):
    def test_native_capped_candidate_converges_without_relaxing_tolerance(self):
        settings={**settings_for('cpu'),'use_features':False,'max_iterations':1,'retry_iterations':80}
        solver=GCVOSolver(settings=settings)
        x=np.random.default_rng(5).uniform([-.5,-.4,.6],[.5,.4,1.5],(1200,3))
        translation=np.array([.015,-.01,.008])
        result=align_tracking(solver,pack(x),pack(x-translation),np.eye(4))
        self.assertTrue(result['retried']);self.assertEqual(result['attempts'][0]['solver_status'],1)
        self.assertEqual(result['solver_status'],0)
        self.assertLess(result['step_norm'],settings['tolerance'])
        np.testing.assert_allclose(result['pose'][:3,3],translation,atol=.0001)
        self.assertEqual(solver.parameters[6],1)  # Per-call overrides do not leak.
        self.assertEqual(solver.parameters[0],settings['ell'])

    def test_normal_result_unchanged_and_retry_uses_candidate_scale(self):
        solver=Mock(settings={'retry_iterations':80})
        first=outcome(0,20);solver.align.return_value=first
        result=align_tracking(solver,None,None,np.eye(4))
        self.assertFalse(result['retried']);self.assertEqual(solver.align.call_count,1)
        np.testing.assert_array_equal(result['pose'],first['pose'])
        first=outcome(1,80);solver.align.reset_mock();solver.align.side_effect=[first,outcome(0,3)]
        result=align_tracking(solver,None,None,np.eye(4))
        self.assertEqual(result['iterations'],83);self.assertEqual(result['solver_ms'],20.)
        self.assertIs(solver.align.call_args.args[2],first['pose'])
        self.assertEqual(solver.align.call_args.kwargs,dict(max_iterations=80,ell=.014))

    def test_exhausted_budget_or_numerical_failure_cannot_pass_as_converged(self):
        solver=Mock(settings={'retry_iterations':80})
        solver.align.side_effect=[outcome(1,80),outcome(1,80)]
        with self.assertRaises(GCVOConvergenceError) as caught:align_tracking(solver,None,None,np.eye(4))
        self.assertEqual(caught.exception.diagnostics['iterations'],160)
        self.assertNotIn('pose',caught.exception.diagnostics)
        solver.align.side_effect=[outcome(1,80),ValueError('GN solve failed')]
        with self.assertRaisesRegex(GCVOConvergenceError,'GN solve failed'):align_tracking(solver,None,None,np.eye(4))

    def test_retry_disabled_and_cuda_unknown_scale_supported(self):
        solver=Mock(settings={'retry_iterations':0});solver.align.return_value=outcome(1,80)
        with self.assertRaises(GCVOConvergenceError):align_tracking(solver,None,None,np.eye(4))
        self.assertEqual(solver.align.call_count,1)
        solver.settings['retry_iterations']=80;solver.align.side_effect=[outcome(1,80,step=None,ell=None),outcome(0,5,step=None,ell=None)]
        result=align_tracking(solver,None,None,np.eye(4))
        self.assertTrue(result['retried']);self.assertIsNone(solver.align.call_args.kwargs['ell'])

    def test_failed_retry_logged_and_converged_bad_geometry_still_rejected(self):
        class TrackingMatcher(Matcher):
            def __init__(self,folder):
                super().__init__(folder);self.solver=Mock(settings={'retry_iterations':80});self.commits=0;self.discards=0
            def estimate(self,source,target):
                result=align_tracking(self.solver,None,None,np.eye(4))
                return {**result,'pose':result['pose'].tolist(),'inliers':100}
            def commit_candidate(self):self.commits+=1
            def discard_candidate(self):self.discards+=1
        cloud=np.random.default_rng(2).uniform([-.2,-.2,.8],[.2,.2,1.2],(1000,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=GCVOBackend(folder,matcher_factory=TrackingMatcher,client_factory=Native)
            try:
                backend.process(None,None,None,1,cloud)
                backend.matcher.solver.align.side_effect=[outcome(1,80),outcome(1,80)]
                self.assertEqual(backend.process(None,None,None,2,cloud),0)
                self.assertEqual(backend.report['frontend']['iterations'],160)
                pose=np.eye(4);pose[0,3]=2
                backend.matcher.solver.align.side_effect=[outcome(1,80),outcome(0,3,pose=pose)]
                self.assertEqual(backend.process(None,None,None,3,cloud),0)
                self.assertEqual(backend.report['frontend']['iterations'],83)
                self.assertEqual(len(backend.client.sent),1);self.assertEqual(backend.reference_node,1)
                self.assertEqual(backend.matcher.commits,1);self.assertEqual(backend.matcher.discards,2)
                logs=[json.loads(line) for line in (Path(folder)/'gcvo_cpu_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(logs[1]['frontend']['attempts'][1]['solver_status'],1)
            finally:backend.close()


if __name__=='__main__':unittest.main()
