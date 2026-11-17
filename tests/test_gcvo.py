"""Mathematical/ABI tests; the CPU library is built by tools/build_gcvo_cpu.cmd."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from gcvo_matcher import GCVOSolver,settings_for,prepare,covariance
from reconstruct import ROOT


def skew(v):
    a,b,c=v;return np.array([[0,-c,b],[c,0,-a],[-b,a,0.]])


def pack(points,cov=None,features=None):
    return dict(points=points,covariances=np.repeat(np.eye(3)[None]*1e-5,len(points),axis=0) if cov is None else cov,
                features=np.empty((len(points),0)) if features is None else features)


class GCVOMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.settings=settings_for()
        except RuntimeError as exc:raise unittest.SkipTest(str(exc))

    def solver(self,**overrides):
        settings={**self.settings,'use_features':False,**overrides}
        return GCVOSolver(settings=settings)

    def test_known_rigid_transform_direction_and_rotation(self):
        x=np.random.default_rng(5).uniform([-.5,-.4,.6],[.5,.4,1.5],(1200,3))
        expected=np.eye(4);expected[:3,:3]=Rotation.from_rotvec([.02,-.015,.025]).as_matrix();expected[:3,3]=[.015,-.01,.008]
        y=(x-expected[:3,3])@expected[:3,:3]
        result=self.solver().align(pack(x),pack(y))
        self.assertEqual(result['solver_status'],0)
        self.assertLess(np.linalg.norm(result['pose'][:3,3]-expected[:3,3]),.0001)
        self.assertLess(Rotation.from_matrix(result['pose'][:3,:3]@expected[:3,:3].T).magnitude(),.0001)

    def test_one_gn_step_matches_independent_numpy_equations(self):
        rng=np.random.default_rng(3);x=rng.uniform([-.02,-.02,.88],[.02,.02,.92],(120,3));y=x+[.004,-.002,.001]
        cov=np.repeat(np.diag([.0001,.0002,.0003])[None],len(x),axis=0)
        solver=self.solver(max_iterations=1,max_neighbors=128,ell=.02,ell_min=.02)
        h=np.zeros((6,6));g=np.zeros(6);objective=0.;count=0
        s=np.linalg.inv(2*cov[0]+np.eye(3)*.02**2)
        for xi in x:
            for yi in y:
                residual=xi-yi;d=residual@s@residual;weight=.01*np.exp(-d/2)
                if d>100 or weight<=.0001:continue
                jac=np.column_stack((-skew(yi),np.eye(3)))
                h+=weight*jac.T@s@jac;g+=weight*jac.T@s@residual;objective+=weight;count+=1
        grad=-g;gamma=np.zeros((6,6));gamma[:3,:3]=-.5*skew(grad[:3]);gamma[3:,:3]=-skew(grad[3:])
        curvature=h-gamma.T;curvature=(curvature+curvature.T)/2
        dx=np.linalg.solve(curvature,grad)
        generator=np.zeros((4,4));generator[:3,:3]=skew(-dx[:3]);generator[:3,3]=-dx[3:]
        expected=expm(generator)
        result=solver.align(pack(x,cov),pack(y,cov))
        self.assertEqual(result['solver_status'],1)
        self.assertEqual(result['correlation_count'],count)
        self.assertAlmostEqual(result['objective'],objective,places=9)
        np.testing.assert_allclose(result['pose'],expected,atol=1e-9)

    def test_scalar_distance_gate_uses_squared_metres_not_normalized_distance(self):
        x=np.random.default_rng(9).uniform([-.02,-.02,.8],[.02,.02,.84],(120,3))
        gate=.0002;solver=self.solver(kernel_type=0,metric_gate=gate,ell=.1,ell_min=.1,max_iterations=1)
        result=solver.align(pack(x),pack(x))
        distance=np.sum((x[:,None]-x[None])**2,axis=-1)
        self.assertEqual(result['correlation_count'],int(np.sum(distance<=gate)))
        np.testing.assert_allclose(result['pose'],np.eye(4),atol=1e-10)

    def test_index_order_cap_not_nearest_neighbor_selection(self):
        x=np.random.default_rng(6).uniform([0,0,1],[.01,.01,1.01],(120,3))
        cov=np.repeat(np.eye(3)[None]*.001,len(x),axis=0)
        result=self.solver(max_neighbors=5,max_iterations=1).align(pack(x,cov),pack(x,cov))
        self.assertEqual(result['correlation_count'],600);self.assertEqual(result['capped_rows'],120)
        s=np.linalg.inv(2*cov[0]+np.eye(3)*self.settings['ell']**2)
        expected=sum(.01*np.exp(-.5*(a-b)@s@(a-b)) for a in x for b in x[:5])
        self.assertAlmostEqual(result['objective'],expected,places=9)

    def test_parallel_reduction_is_deterministic(self):
        x=np.random.default_rng(7).uniform([-.2,-.2,.8],[.2,.2,1.2],(500,3))
        first=self.solver(threads=1).align(pack(x),pack(x+[.003,0,0]))
        second=self.solver(threads=4).align(pack(x),pack(x+[.003,0,0]))
        np.testing.assert_array_equal(first['pose'],second['pose'])

    def test_sparse_match_invalid_covariance_and_reflection_fail(self):
        x=np.random.default_rng(8).uniform([-.2,-.2,.8],[.2,.2,1.2],(200,3));solver=self.solver()
        with self.assertRaisesRegex(ValueError,'kernel entries'):solver.align(pack(x),pack(x+10))
        with self.assertRaisesRegex(ValueError,'SPD'):solver.align(pack(x,np.zeros((len(x),3,3))),pack(x))
        bad=np.eye(4);bad[0,0]=-1
        with self.assertRaises(ValueError):solver.align(pack(x),pack(x),bad)

    def test_preparation_point_budget_color_and_positive_covariances(self):
        x=np.random.default_rng(4).uniform([-.2,-.2,.8],[.2,.2,1.2],(10000,3))
        color=np.full((100,100,3),[10,20,30],np.uint8);k=np.array([[100.,0,50],[0,100,50],[0,0,1]])
        result=prepare(x,{**self.settings,'max_points':500},color,k)
        self.assertLessEqual(len(result['points']),500)
        np.testing.assert_allclose(result['features'][0],[30/255,20/255,10/255])
        self.assertGreaterEqual(np.linalg.eigvalsh(result['covariances']).min(),self.settings['cov_eig_min']-1e-12)

    def test_missing_cuda_is_explicit_and_cpu_selection_does_not_fallback(self):
        with patch.dict('os.environ',{},clear=False):
            solver=self.solver();self.assertIn('cpu',solver.backend)
        if not (ROOT/'vendor/gcvo/bin/gcvo_cuda.dll').exists():
            with self.assertRaisesRegex(RuntimeError,'cuda'):settings_for('cuda')
        with self.assertRaisesRegex(RuntimeError,'不一致'):
            GCVOSolver(settings={**self.settings,'device':'cuda'})

    def test_depth_visibility_removes_occluded_color_samples(self):
        rng=np.random.default_rng(1)
        x=rng.uniform([-.3,-.3,.995],[.3,.3,1.005],(2000,3));hidden=x.copy();hidden[:,2]+=1
        color=np.full((100,100,3),[10,20,30],np.uint8);depth=np.full((100,100),1000,np.uint16)
        k=np.array([[100.,0,50],[0,100,50],[0,0,1]])
        result=prepare(np.r_[x,hidden],{**self.settings,'max_points':3000},color,k,depth)
        self.assertLess(result['points'][:,2].max(),1.03)

    def test_reference_cache_reuses_successfully_prepared_current_frame(self):
        from gcvo_matcher import GCVOMatcher
        matcher=GCVOMatcher.__new__(GCVOMatcher)
        prepared=object();matcher.current_pack=prepared
        matcher.reference_updated(np.zeros((100,3)),np.zeros((2,2,3)),np.eye(3),np.zeros((2,2)))
        self.assertIs(matcher.prepared_target,prepared);self.assertIsNone(matcher.current_pack)


class GCVOIntegrationTests(unittest.TestCase):
    def test_ui_and_session_select_both_devices_without_changing_default(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        from video_capture import VideoSession
        from gcvo_backend import GCVOBackend
        import queue
        root=tk.Tk();root.withdraw();app=SceneWorkbench(root)
        try:
            view=app.views['acquisition'];self.assertEqual(view.registration_frontend.get(),'RGB-D（现有方案）')
            for label,frontend in [('G-CVO（CPU）','gcvo_cpu'),('G-CVO（CUDA）','gcvo_cuda')]:
                view.registration_frontend.set(label)
                with patch('gcvo_matcher.settings_for',return_value={}),patch('video_capture.VideoSession') as factory:
                    view.create_session();self.assertEqual(factory.call_args.kwargs['frontend'],frontend)
                session=VideoSession(queue.Queue(),optimizer='isam2',dense_mode='submaps',frontend=frontend)
                try:
                    self.assertIs(session.backend_factory.func,GCVOBackend)
                    self.assertEqual(session.backend_factory.keywords['device'],frontend.split('_')[1])
                finally:session.discard()
        finally:app.close()

    def test_backend_metadata_and_reference_reset_only_after_accepted_node(self):
        from gcvo_backend import GCVOBackend
        from test_bufferx import Matcher,Native
        class Hooks(Matcher):
            def __init__(self,folder):super().__init__(folder);self.resets=0;self.commits=0;self.discards=0
            def set_frame_rgb(self,*args):pass
            def reference_updated(self,*args):self.resets+=1
            def commit_candidate(self):self.commits+=1
            def discard_candidate(self):self.discards+=1
        x=np.random.default_rng(3).uniform([-.2,-.2,.8],[.2,.2,1.2],(1000,3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            backend=GCVOBackend(folder,matcher_factory=Hooks,client_factory=Native)
            try:
                backend.process(None,None,None,1,x);self.assertEqual(backend.matcher.resets,1)
                backend.retain(1,x);backend.client.status=1
                backend.process(None,None,None,1.1,x);self.assertEqual(backend.matcher.resets,1)
                backend.matcher.pose[0,3]=2;backend.process(None,None,None,1.2,x)
                self.assertEqual(backend.matcher.discards,1);self.assertEqual(backend.matcher.commits,2)
                backend.pose_writer.flush();info=json.loads((Path(folder)/'geometry_poses.json').read_text())
                self.assertEqual(info['backend'],'gcvo_cpu');self.assertEqual(backend.report['method'],'G-CVO CPU')
            finally:backend.close()


if __name__=='__main__':unittest.main()
