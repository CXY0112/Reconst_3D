import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
"""Run explicitly with vendor/rtabmap/bin/python.exe (Open3D Python 3.12 ABI)."""
from geometry_worker import Frame,geometry_validation,refine,GeometryEngine,ROOT
import tempfile
import unittest
import numpy as np


class GeometryEngineTests(unittest.TestCase):
    def test_unrelated_clouds_are_rejected(self):
        rng = np.random.default_rng(20)
        first = Frame(rng.uniform(-.3,.3,(1500,3))+[0,0,1.5])
        second = Frame(rng.uniform(-.3,.3,(1500,3))+[3,0,1.5])
        with self.assertRaises(ValueError):
            geometry_validation(first,second,np.eye(4))

    def test_single_plane_reports_geometric_ambiguity(self):
        u,v = np.meshgrid(np.linspace(-.5,.5,40),np.linspace(-.5,.5,40))
        frame = Frame(np.column_stack((u.ravel(),v.ravel(),np.ones(u.size))))
        with self.assertRaisesRegex(ValueError,'单一平面'):
            geometry_validation(frame,frame,np.eye(4))

    def test_robust_icp_recovers_pose_and_joint_graph_moves_old_nodes(self):
        rng = np.random.default_rng(21)
        base = rng.uniform(-.4,.4,(2000,3))+[0,0,1.5]
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            engine = GeometryEngine(folder)
            path = ROOT/'results/geometry_synthetic_input.npy'
            try:
                np.save(path,base); engine.prepare(path)
                self.assertTrue(engine.finish([])['accepted']); engine.commit()
                second = base-[.08,0,0]
                np.save(path,second); engine.prepare(path)
                relative = np.eye(4); relative[0,3] = .08
                report = engine.finish([{'reference':0,'pose':relative.tolist()}])
                self.assertTrue(report['accepted']); engine.commit()
                anchor = engine.poses[0].copy()
                # Simulate an old noisy estimate and close a triangle with a new view.
                engine.poses[1][1,3] += .02
                np.save(path,base-[.04,.03,0]); engine.prepare(path)
                first_to_new = np.eye(4); first_to_new[:3,3] = [.04,.03,0]
                second_to_new = np.eye(4); second_to_new[:3,3] = [-.04,.03,0]
                report = engine.finish([{'reference':0,'pose':first_to_new.tolist()},
                                        {'reference':1,'pose':second_to_new.tolist()}])
                self.assertTrue(report['accepted'])
                updated = np.asarray(report['poses'])
                np.testing.assert_allclose(updated[0],anchor,atol=1e-7)
                self.assertGreater(np.linalg.norm(updated[1]-engine.poses[1]),.001)
                self.assertEqual(report['geometry_edges'],3)
            finally:
                path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
