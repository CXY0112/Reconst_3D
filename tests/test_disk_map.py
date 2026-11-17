import tempfile,unittest,sqlite3
from pathlib import Path
import numpy as np
from disk_map import DiskVoxelMap,CloudStore
from voxel_map import VoxelMap

class DiskMapTests(unittest.TestCase):
    def test_matches_memory_map_after_repeated_scans_corrections_and_removal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);disk=DiskVoxelMap(folder/'map.db',.01);memory=VoxelMap(.01);store=CloudStore(folder/'frames')
            rng=np.random.default_rng(15);pose=np.c_[np.eye(3),np.zeros(3)]
            try:
                for i in range(12):
                    points=rng.uniform(-.1,.1,(1500,3));store[i]=points
                    self.assertIsInstance(store[i],np.memmap)
                    memory.update(i,points,pose);disk.update(i,store[i],pose)
                moved=pose.copy();moved[0,3]=.02
                memory.update(0,store[0],moved);disk.update(0,store[0],moved)
                memory.remove(3);disk.remove(3)
                expected=memory.cloud();expected=expected[np.lexsort(expected.T[::-1])]
                actual=disk.cloud();actual=actual[np.lexsort(actual.T[::-1])]
                np.testing.assert_allclose(actual,expected,atol=1e-12)
                self.assertFalse(disk.update(0,store[0],moved))
                before=disk.cloud().copy();disk.db.execute('PRAGMA query_only=ON')
                with self.assertRaises(sqlite3.OperationalError):disk.update(0,store[0],pose)
                np.testing.assert_array_equal(disk.cloud(),before)
                disk.db.execute('PRAGMA query_only=OFF')
                disk.close();disk=DiskVoxelMap(folder/'map.db',.01)
                np.testing.assert_array_equal(disk.cloud(),before)
                np.testing.assert_array_equal(disk.poses[0],moved)
            finally:disk.close()

    def test_empty_negative_cells_and_mismatched_resolution(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/'map.db';disk=DiskVoxelMap(path,.01)
            self.assertEqual(disk.cloud().shape,(0,3))
            disk.update(1,np.array([[-.001,0,1],[.001,0,1]]),np.c_[np.eye(3),np.zeros(3)])
            self.assertEqual(len(disk.cloud()),2);disk.remove(1)
            self.assertEqual(len(disk.cloud()),0);disk.close()
            with self.assertRaises(ValueError):DiskVoxelMap(path,.02)
