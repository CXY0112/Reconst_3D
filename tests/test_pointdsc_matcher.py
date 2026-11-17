import unittest
import numpy as np
from pointdsc_matcher import PointDSCMatcher


class PretrainedNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = PointDSCMatcher()

    def test_pretrained_cpu_network_recovers_rotation_with_false_correspondences(self):
        rng = np.random.default_rng(123)
        source = rng.uniform(-1,1,(160,3)).astype(np.float32)
        angle = np.deg2rad(65)
        expected = np.array([[np.cos(angle),0,np.sin(angle),.5],[0,1,0,-.2],[-np.sin(angle),0,np.cos(angle),.1],[0,0,0,1.]])
        target = (source@expected[:3,:3].T+expected[:3,3])+rng.normal(0,.001,source.shape)
        target[:60] = rng.uniform(-2,2,(60,3))
        actual = self.matcher.estimate(source,target)
        np.testing.assert_allclose(actual,expected,atol=.01)

    def test_insufficient_correspondences_are_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.estimate(np.zeros((10,3)),np.zeros((10,3)))
