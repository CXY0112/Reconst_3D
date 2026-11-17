import unittest
import numpy as np
import torch
from bufferx_cpu_ops import ball_query,furthest_point_sample,gather_operation,grouping_operation,KNN


class BufferXCPUOpsTests(unittest.TestCase):
    def test_ball_query_input_order_strict_boundary_and_first_hit_padding(self):
        points=torch.tensor([[[0.,0,0],[.5,0,0],[-.5,0,0],[1.,0,0]]])
        queries=torch.tensor([[[0.,0,0],[10.,0,0]]])
        actual=ball_query(1.,5,points,queries)
        np.testing.assert_array_equal(actual.numpy(),[[[0,1,2,0,0],[0,0,0,0,0]]])

    def test_gather_and_group_preserve_batch_and_channel(self):
        features=torch.arange(24).reshape(2,3,4)
        ids=torch.tensor([[3,1],[0,2]])
        gathered=gather_operation(features,ids)
        np.testing.assert_array_equal(gathered[0].numpy(),features[0][:,[3,1]].numpy())
        grouped=grouping_operation(features,ids[:,None])
        np.testing.assert_array_equal(grouped[:,:,0].numpy(),gathered.numpy())

    def test_fps_spreads_samples_and_does_not_duplicate_nonpadding_points(self):
        points=torch.tensor([[[1.,0,1],[2.,0,1],[3.,0,1],[4.,0,1]]])
        actual=furthest_point_sample(points,4).numpy()[0]
        self.assertEqual(list(actual[:2]),[0,3]);self.assertEqual(len(set(actual)),4)

    def test_knn_matches_numpy_distances_in_both_layouts(self):
        ref=torch.tensor([[[0.,0],[2.,0],[5.,0]],[[0.,1],[2.,1],[5.,1]]])
        query=torch.tensor([[[1.8,0],[4.9,0]],[[1.8,1],[4.9,1]]])
        distance,index=KNN(1,True)(ref,query)
        np.testing.assert_array_equal(index.numpy().squeeze(-1),[[1,2],[1,2]])
        np.testing.assert_allclose(distance.numpy().squeeze(-1),[[.2,.1],[.2,.1]],atol=1e-5)
        d2,i2=KNN(1,False)(ref.transpose(1,2),query.transpose(1,2))
        torch.testing.assert_close(distance,d2);torch.testing.assert_close(index,i2)


if __name__=='__main__':unittest.main()
