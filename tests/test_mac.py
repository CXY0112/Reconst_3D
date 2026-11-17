import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from mac_solver import solve,maximal_cliques
from gcvo_backend import GCVOBackend
from test_bufferx import Matcher,Native
from reconstruct import ROOT


class MACTests(unittest.TestCase):
    def test_maximal_cliques_match_independent_networkx_enumeration(self):
        import networkx as nx
        rng=np.random.default_rng(5)
        for _ in range(5):
            a=rng.random((12,12))<.4;a=np.triu(a,1);a=a|a.T
            actual={tuple(sorted(c)) for c in maximal_cliques(a,time.monotonic()+2,1000)}
            expected={tuple(sorted(c)) for c in nx.find_cliques(nx.from_numpy_array(a)) if len(c)>=3}
            self.assertEqual(actual,expected)

    def test_outlier_correspondences_known_pose_and_transform_direction(self):
        rng=np.random.default_rng(10);source=rng.uniform(-1,1,(120,3));r=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();t=np.array([.2,-.1,.05])
        target=source@r.T+t;target[30:]=rng.uniform(-1,1,(90,3))
        result=solve(source,target,{})
        pose=np.array(result['pose']);np.testing.assert_allclose(pose[:3,:3],r,atol=1e-8);np.testing.assert_allclose(pose[:3,3],t,atol=1e-8)
        self.assertEqual(result['inliers'],30)

    def test_degenerate_cliques_and_exhausted_budget_rejected(self):
        line=np.column_stack((np.linspace(0,1,20),np.zeros((20,2))))
        with self.assertRaisesRegex(ValueError,'degenerate'):solve(line,line,{})
        graph=np.zeros((6,6),bool)
        graph[:3,:3]=True;graph[3:,3:]=True;np.fill_diagonal(graph,False)
        with self.assertRaisesRegex(ValueError,'budget'):maximal_cliques(graph,time.monotonic()+1,1)

    def test_mac_seed_never_submitted_without_current_frame_confirmation(self):
        class Hooks(Matcher):
            settings={'mac':{'enabled':True}}
            def __init__(self,folder):super().__init__(folder);self.lost=False;self.initials=[]
            def estimate(self,source,target,initial=None):
                self.initials.append(initial)
                if self.lost and initial is None:raise ValueError('local tracking lost')
                # Current pose differs from the worker's older candidate.
                result=super().estimate(source,target);result['pose']=np.eye(4).tolist();return result
        class MAC:
            def __init__(self,*args):self.ready=False;self.calls=0;self.resets=0
            def poll(self,*args):
                self.calls+=1
                if self.ready:
                    seed=np.eye(4);seed[0,3]=.1;return seed
            def reset(self):self.resets+=1
            def snapshot(self):return dict(status='seed_ready' if self.ready else 'running')
            def close(self):pass
        cloud=np.random.default_rng(4).uniform([-.2,-.2,.8],[.2,.2,1.2],(1000,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            b=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native,mac_factory=MAC)
            try:
                b.process(None,None,None,1,cloud);self.assertEqual(b.mac.calls,0)
                b.matcher.lost=True
                for i in range(3):b.process(None,None,None,2+i,cloud)
                self.assertEqual(b.tracking.state,'RECOVERY');self.assertEqual(len(b.client.sent),1)
                b.mac.ready=True
                for i in range(2):b.process(None,None,None,5+i,cloud)
                self.assertEqual(len(b.client.sent),1)
                b.process(None,None,None,7,cloud)
                self.assertEqual(b.tracking.state,'NORMAL');self.assertEqual(len(b.client.sent),2)
                np.testing.assert_allclose(b.client.sent[0],b.client.sent[1])
                self.assertTrue(b.report['mac']['used_seed']);self.assertTrue(b.report['tracking']['mac_enabled'])
            finally:b.close()

    def test_timeout_stale_job_and_expired_seed_cannot_reach_tracker(self):
        from mac_recovery import MACRecovery
        class Process:
            returncode=0
            def __init__(self,done=False):self.done=done;self.killed=False
            def poll(self):return 0 if self.done else None
            def wait(self):return 0
            def kill(self):self.killed=True;self.done=True
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            m=MACRecovery(folder,{'mac':{'enabled':True}})
            try:
                process=Process();m.process=process;m.reference=1;m.started=time.monotonic()-20
                self.assertIsNone(m.poll(1,None,None,0));self.assertTrue(process.killed)
                self.assertEqual(m.status,'failed')
                m.process=Process(done=True);m.job=2;m.started=time.monotonic()
                (Path(folder)/'mac_output.json').write_text(json.dumps(dict(job=1,reference=1,ok=True,result=dict(pose=np.eye(4).tolist()))))
                self.assertIsNone(m.poll(1,None,None,0));self.assertIsNone(m.seed)
                m.seed=dict(pose=np.eye(4).tolist());m.seed_time=time.monotonic()-20
                self.assertIsNone(m.poll(1,None,None,0));self.assertEqual(m.status,'expired')
            finally:m.close()


if __name__=='__main__':unittest.main()
