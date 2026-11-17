import hashlib,json,queue,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruct import ROOT
from rtabmap_backend import RtabmapBackend
from submap_backend import SubmapBackend,SubmapStore,cells_for,reduce_cells
from video_capture import VideoSession


class NoNative:
    def __init__(self,folder,**kwargs):pass
    def close(self,**kwargs):pass


def poses(backend,values):
    with backend.lock:backend._poses(np.array([[node,*p[:3].ravel()] for node,p in values.items()]))


def populate(backend,n=10):
    rng=np.random.default_rng(51);cloud=rng.uniform([-.15,-.1,.8],[.15,.1,1.2],(600,3))
    values={}
    for node in range(1,n+1):
        p=np.eye(4);p[0,3]=node*.01;values[node]=p
    poses(backend,values)
    for node in values:backend.retain(node,cloud)
    backend.preview();return values,cloud


class SubmapTests(unittest.TestCase):
    def test_rigid_loop_moves_submaps_without_rebuilding_frames(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            backend=SubmapBackend(temp,client_factory=NoNative)
            try:
                values,_=populate(backend);before=backend.preview()[0].copy()
                move=np.eye(4);move[:3,:3]=Rotation.from_euler('z',4,degrees=True).as_matrix();move[:3,3]=[.12,-.03,.02]
                poses(backend,{n:move@p for n,p in values.items()})
                with patch.object(backend.voxel_map,'_rebuild',side_effect=AssertionError('rigid correction rebuilt local geometry')):
                    after,_=backend.preview()
                self.assertEqual(backend.preview_stats['rigid_submap_updates'],2)
                self.assertEqual(backend.preview_stats['pending_pose_nodes'],0)
                self.assertFalse(after.flags.writeable);self.assertFalse(np.array_equal(before,after))
                world=np.c_[after[:,2],-after[:,0],-after[:,1]]
                self.assertEqual(len(np.unique(np.floor(world/.003).astype(int),axis=0)),len(world))
            finally:backend.close()

    def test_only_internally_changed_submaps_rebuild_with_fair_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            backend=SubmapBackend(temp,client_factory=NoNative)
            try:
                values,_=populate(backend,18)
                file=Path(temp)/'submaps/000003.npz';unchanged=hashlib.sha256(file.read_bytes()).hexdigest()
                values[2][1,3]+=.01;values[10][1,3]-=.01;poses(backend,values)
                backend.preview()
                self.assertEqual(backend.preview_stats['rebuilt_submaps'],1)
                self.assertTrue(backend.pending_updates())
                backend.preview();self.assertFalse(backend.pending_updates())
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),unchanged)
            finally:backend.close()

    def test_final_exact_fusion_matches_global_baseline_after_nonrigid_loop(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            a=Path(temp)/'a';b=Path(temp)/'b';a.mkdir();b.mkdir()
            baseline=RtabmapBackend(a,client_factory=NoNative);backend=SubmapBackend(b,client_factory=NoNative)
            try:
                values,_=populate(baseline);populate(backend)
                for node,p in values.items():
                    p[0,3]+=.02;p[1,3]+=.001*node
                    p[:3,:3]=Rotation.from_euler('y',node*.05,degrees=True).as_matrix()
                poses(baseline,values);poses(backend,values)
                expected,_=baseline.final_preview();actual,_=backend.final_preview()
                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                self.assertFalse(backend.pending_updates());self.assertTrue(backend.preview_stats['exact'])
                again,_=backend.final_preview();self.assertIs(again,actual)
                info=json.loads((b/'submaps/manifest.json').read_text());self.assertTrue(info['finalized'])
            finally:baseline.close();backend.close()

    def test_small_pose_changes_accumulate_against_last_fused_submap(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            backend=SubmapBackend(temp,client_factory=NoNative)
            try:
                values,_=populate(backend,1);first,_=backend.preview();old=first.copy()
                for step in range(1,4):
                    p=values[1].copy();p[0,3]+=.0002*step;poses(backend,{1:p})
                    same,_=backend.preview();self.assertIs(same,first)
                p[0,3]+=.0002;poses(backend,{1:p});new,_=backend.preview()
                self.assertEqual(backend.preview_stats['rigid_submap_updates'],1)
                np.testing.assert_array_equal(first,old);self.assertIsNot(new,first)
                p[0,3]+=.0001;poses(backend,{1:p});exact,_=backend.final_preview()
                self.assertTrue(backend.preview_stats['exact']);self.assertIsNot(exact,new)
            finally:backend.close()

    def test_snapshot_update_does_not_clear_a_newer_tracking_correction(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            backend=SubmapBackend(temp,client_factory=NoNative)
            try:
                values,_=populate(backend,3);values[2][0,3]+=.01;poses(backend,values)
                rebuild=backend.voxel_map._rebuild
                def newer(*args):
                    rebuild(*args)
                    next_values={n:p.copy() for n,p in values.items()};next_values[2][0,3]+=.01
                    poses(backend,next_values)
                with patch.object(backend.voxel_map,'_rebuild',side_effect=newer):backend.preview()
                self.assertTrue(backend.pending_updates());backend.preview();self.assertFalse(backend.pending_updates())
            finally:backend.close()

    def test_cache_budget_is_enforced_and_cells_do_not_alias_at_large_coordinates(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            store=SubmapStore(Path(temp)/'unused.sqlite',cache_mib=.001)
            store._cache(1,(np.zeros((100,3)),np.ones(100,dtype=np.int64)))
            self.assertLessEqual(store.cache_bytes,store.cache_limit)
            points=np.array([[4000.,0,1],[-4000.,0,1],[4000.001,0,1]])
            key,total,count=cells_for(points,.003)
            self.assertEqual(len(key),2);self.assertEqual(sorted(count.tolist()),[1,2]);store.close()

    def test_submaps_follow_translation_and_rotation_size_limits(self):
        for translation,angle in ((.41,0),(0,16)):
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
                backend=SubmapBackend(temp,client_factory=NoNative)
                try:
                    values,cloud=populate(backend,1);p=values[1].copy();p[0,3]+=translation
                    p[:3,:3]=Rotation.from_euler('y',angle,degrees=True).as_matrix()
                    values[2]=p;poses(backend,values);backend.retain(2,cloud);backend.preview()
                    self.assertEqual(len(backend.voxel_map.groups),2)
                finally:backend.close()

    def test_rgbd_session_preserves_optimizer_selection_and_stage_two_sources(self):
        for optimizer in ('default','isam2'):
            session=VideoSession(queue.Queue(),optimizer=optimizer,dense_mode='submaps')
            try:
                factory=session.backend_factory
                self.assertIs(factory,SubmapBackend) if optimizer=='default' else self.assertIs(factory.func,SubmapBackend)
                info=json.loads((session.folder/'video_capture.json').read_text())
                self.assertEqual(info['optimizer'],optimizer);self.assertEqual(info['dense_mode'],'submaps')
                self.assertTrue((session.folder/'rgb_raw').is_dir());self.assertTrue((session.folder/'pointcloud_local').is_dir())
            finally:session.discard()


if __name__=='__main__':unittest.main()
