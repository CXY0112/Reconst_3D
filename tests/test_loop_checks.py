import unittest
import numpy as np
from loop_checks import validate_graph,rigid_pose

class LoopCheckTests(unittest.TestCase):
    def test_rejects_nonrigid_and_nonfinite_transform(self):
        for value in (np.full((4,4),np.nan),np.diag([2.,1,1,1])):
            with self.assertRaises(ValueError):rigid_pose(value)

    def test_preserves_hardware_tolerance_but_rejects_map_deformation(self):
        before=[np.eye(4),np.eye(4)];edge=(1,0,np.eye(4),np.eye(6),False)
        after=[p.copy() for p in before];after[1][0,3]=.009
        validate_graph(before,after,[edge],[])
        after[1][0,3]=.06
        with self.assertRaises(ValueError):validate_graph(before,after,[edge],[])

    def test_inconsistent_new_loop_is_rejected_even_without_old_edges(self):
        poses=[np.eye(4),np.eye(4)];measurement=np.eye(4);measurement[0,3]=.2
        with self.assertRaises(ValueError):validate_graph([],poses,[],[(1,0,measurement,np.eye(6),True)])
