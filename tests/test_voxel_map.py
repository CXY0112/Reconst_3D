import unittest
import numpy as np
from voxel_map import VoxelMap


class VoxelTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.column_stack((np.eye(3), np.zeros(3)))

    def test_repeated_observations_keep_one_centroid_per_cell(self):
        voxel = VoxelMap(.01)
        for node in range(100):
            voxel.update(node, np.array([[.001,.002,.003],[.003,.004,.005]]), self.pose)
        np.testing.assert_allclose(voxel.cloud(), [[.002,.003,.004]])

    def test_pose_correction_removes_old_contribution(self):
        voxel = VoxelMap(.01)
        points = np.array([[.001,0,1.]])
        voxel.update(1, points, self.pose)
        changed = self.pose.copy()
        changed[0,3] = 1
        voxel.update(1, points, changed)
        np.testing.assert_allclose(voxel.cloud(), [[1.001,0,1.]])
        self.assertEqual(len(voxel.cells), 1)

    def test_removing_node_preserves_other_observations(self):
        voxel = VoxelMap(.01)
        voxel.update(1, np.array([[.001,0,1.]]), self.pose)
        voxel.update(2, np.array([[.003,0,1.]]), self.pose)
        voxel.remove(1)
        np.testing.assert_allclose(voxel.cloud(), [[.003,0,1.]])
        voxel.remove(2)
        self.assertEqual(voxel.cloud().shape, (0,3))

    def test_negative_and_positive_cells_stay_distinct(self):
        voxel = VoxelMap(.01)
        voxel.update(1, np.array([[-.001,0,1.],[.001,0,1.]]), self.pose)
        self.assertEqual(len(voxel.cloud()), 2)

    def test_identical_pose_does_not_rebuild_map(self):
        voxel = VoxelMap(.01)
        points = np.array([[0,0,1.]])
        voxel.update(1, points, self.pose)
        cached = voxel.cloud()
        self.assertFalse(voxel.update(1, points, self.pose))
        self.assertIs(cached, voxel.cloud())

    def test_new_and_existing_measurements_are_averaged_not_stacked(self):
        voxel = VoxelMap(.01)
        voxel.update(1,np.array([[.001,.002,.003]]),self.pose)
        voxel.update(2,np.array([[.005,.006,.007]]),self.pose)
        np.testing.assert_allclose(voxel.cloud(),[[.003,.004,.005]])

    def test_bounded_volume_has_bounded_voxel_count_under_repeat_scans(self):
        voxel = VoxelMap(.01)
        rng = np.random.default_rng(30)
        for node in range(10):
            voxel.update(node,rng.uniform(.0001,.0999,(2000,3)),self.pose)
        self.assertLessEqual(len(voxel.cloud()),1000)  # 10x10x10 fixed spatial cells.


if __name__ == '__main__':
    unittest.main()
