import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import importlib.util
from runtime_paths import activate_geometry_packages
activate_geometry_packages()
if importlib.util.find_spec('open3d') is None:raise unittest.SkipTest('Requires the geometry Python runtime')
from geometry_worker import GeometryEngine

class WorkerGuardTests(unittest.TestCase):
    def test_rejected_historical_graph_falls_back_to_local_without_losing_capture(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);engine=GeometryEngine(folder)
            rng=np.random.default_rng(9);a=rng.uniform(-.6,.6,(700,2))
            points=np.concatenate([np.c_[a,np.ones(len(a))],np.c_[np.full(len(a),-.6),a[:,0],a[:,1]+1.5],np.c_[a[:,0],np.full(len(a),.6),a[:,1]+1.5]])
            path=folder/'input.npy';np.save(path,points)
            for i in range(6):
                prepared=engine.prepare(path)
                proposals=[dict(reference=ref,pose=np.eye(4).tolist()) for ref in prepared['candidates']]
                if i==5:
                    previous=[p.copy() for p in engine.poses]
                    with patch('geometry_worker.validate_graph',side_effect=[ValueError('forced inconsistent loop'),None]):
                        result=engine.finish(proposals)
                    self.assertTrue(result['accepted']);self.assertEqual(result['loop_edges'],0)
                    self.assertIn('forced inconsistent loop',result['deferred_loops_reason'])
                    for before,after in zip(previous,engine.poses):np.testing.assert_array_equal(before,after)
                else:result=engine.finish(proposals);self.assertTrue(result['accepted'])
                engine.commit()
            self.assertEqual(len(engine.frames),6)

    def test_delayed_loop_is_revalidated_and_only_commit_publishes_graph(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);engine=GeometryEngine(folder)
            rng=np.random.default_rng(29);a=rng.uniform(-.6,.6,(700,2))
            points=np.concatenate([np.c_[a,np.ones(len(a))],np.c_[np.full(len(a),-.6),a[:,0],a[:,1]+1.5],np.c_[a[:,0],np.full(len(a),.6),a[:,1]+1.5]])
            path=folder/'input.npy';np.save(path,points)
            for i in range(6):
                prepared=engine.prepare(path)
                result=engine.finish([dict(reference=ref,pose=np.eye(4).tolist()) for ref in prepared['local_candidates']])
                self.assertTrue(result['accepted']);engine.commit()
            previous=engine.state();revision=engine.revision
            link=dict(target=0,pose=np.eye(4).tolist(),information=(np.eye(6)*1000).tolist())
            with self.assertRaises(ValueError):engine.propose_loop(5,revision+10,[link])
            proposal=engine.propose_loop(5,0,[link]);self.assertTrue(proposal['accepted'])
            self.assertEqual(engine.state(),previous)
            with self.assertRaises(ValueError):engine.prepare(path)
            engine.commit_loop();self.assertEqual(engine.revision,revision+1)
            self.assertEqual(len(engine.frames),6)
            self.assertFalse(engine.propose_loop(5,0,[link])['accepted'])
            previous=engine.state()
            bad=dict(target=1,pose=np.eye(4).tolist(),information=(np.eye(6)*1000).tolist())
            # Source 5 only permits targets before node 1, so this is not historical.
            self.assertFalse(engine.propose_loop(5,0,[bad])['accepted'])
            self.assertEqual(engine.state(),previous)
