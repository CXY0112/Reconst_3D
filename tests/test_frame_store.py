import unittest,tempfile
from pathlib import Path
import numpy as np
from frame_store import FrameStore

class Packet:
    def __init__(self,value):self.xyz=np.full((10,3),value,dtype=float);self.features=np.full((10,33),value+1,dtype=float)
    def nbytes(self):return self.xyz.nbytes+self.features.nbytes
    def save(self,path):np.savez_compressed(path,xyz=self.xyz,features=self.features)
    @classmethod
    def load(cls,path):
        p=cls(0)
        with np.load(path) as d:p.xyz=d['xyz'].copy();p.features=d['features'].copy()
        return p

class FrameStoreTests(unittest.TestCase):
    def test_evicted_history_can_be_reloaded_exactly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            store=FrameStore(Path(temp),Packet.load,budget=6000,max_frames=2)
            for i in range(20):store.append(Packet(i))
            self.assertLessEqual(store.stats()['resident_array_bytes'],6000)
            self.assertLessEqual(len(store.cache),2)
            np.testing.assert_array_equal(store[0].xyz,Packet(0).xyz)
            np.testing.assert_array_equal(store.match_data(15)[1],Packet(15).features)
            self.assertGreater(store.loads,0)
            self.assertLessEqual(len(store.candidates(Packet(0))),12)
            self.assertTrue(set(range(16,20)).issubset(store.candidates(Packet(0))))

    def test_oversized_frame_is_not_retained(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            store=FrameStore(Path(temp),Packet.load,budget=1)
            store.append(Packet(1));self.assertEqual(len(store.cache),0)
            self.assertEqual(store[0].xyz[0,0],1);self.assertEqual(len(store.cache),0)
