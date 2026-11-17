import threading
import unittest
import numpy as np
from scipy.spatial import cKDTree
from surface_sampling import uniform_surface_indices
from preprocess import Parameters,ProcessingCancelled


class SurfaceSamplingTests(unittest.TestCase):
    def test_dense_and_sparse_regions_share_spacing_and_keep_coverage(self):
        x,y=np.meshgrid(np.arange(-.06,.001,.0015),np.arange(-.03,.031,.0015))
        left=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
        x,y=np.meshgrid(np.arange(.003,.061,.0045),np.arange(-.03,.031,.0045))
        right=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
        points=np.vstack((left,right));normals=np.tile([0,0,-1.],(len(points),1))
        original=points.copy();ids,report=uniform_surface_indices(points,normals,.005)
        output=points[ids];d=cKDTree(output).query(output,k=2)[0][:,1]
        self.assertGreaterEqual(d.min(),.005-1e-9)
        self.assertLessEqual(cKDTree(output).query(points)[0].max(),.005+1e-9)
        self.assertTrue(np.any(output[:,0]<0));self.assertTrue(np.any(output[:,0]>.04))
        np.testing.assert_array_equal(points,original)
        self.assertEqual(report['new_points_generated'],0)

    def test_no_batch_input_order_seam(self):
        points=np.random.default_rng(1).uniform(-.05,.05,(1800,3));points[:,2]=1
        normals=np.tile([0,0,-1.],(len(points),1))
        ids,_=uniform_surface_indices(points,normals,.005)
        permutation=np.random.default_rng(6).permutation(len(points))
        shuffled,_=uniform_surface_indices(points[permutation],normals[permutation],.005)
        np.testing.assert_array_equal(np.sort(permutation[shuffled]),ids)

    def test_opposite_faces_and_orthogonal_edges_survive(self):
        points=np.array([[0,0,1],[0,0,1.001],[.001,0,1],[.02,0,1]])
        normals=np.array([[0,0,-1],[0,0,1],[1,0,0],[0,0,-1.]])
        ids,_=uniform_surface_indices(points,normals,.005)
        self.assertEqual(len(ids),4)

    def test_resolved_layer_is_not_merged(self):
        points=np.array([[0,0,1],[0,0,1.004]])
        ids,_=uniform_surface_indices(points,np.tile([0,0,-1.],(2,1)),.005)
        self.assertEqual(len(ids),2)

    def test_confidence_priority_and_attributes_remain_indexed(self):
        points=np.array([[0,0,1],[.001,0,1],[.02,0,1]])
        support=np.array([1,4,1]);sigma=np.array([3,2,3.])
        ids,_=uniform_surface_indices(points,np.tile([0,0,-1.],(3,1)),.005,support,sigma)
        np.testing.assert_array_equal(ids,[1,2])
        np.testing.assert_array_equal(support[ids],[4,1])

    def test_invalid_input_and_cancel(self):
        stop=threading.Event();stop.set()
        with self.assertRaises(ProcessingCancelled):uniform_surface_indices(np.zeros((3,3)),np.zeros((3,3)),.005,cancel=stop)
        with self.assertRaises(ValueError):uniform_surface_indices(np.ones((3,3)),np.ones((3,3)),0)
        with self.assertRaises(ValueError):Parameters(kalman_sampling='unknown').validate()


if __name__=='__main__':unittest.main()
