import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import ctypes as ct
import numpy as np
from geometry_snapshot_backend import SnapshotBackend
from voxel_map import VoxelMap


class GeometryBackendTests(unittest.TestCase):
    def backend(self, folder):
        backend = SnapshotBackend.__new__(SnapshotBackend)
        backend.folder = Path(folder)
        backend.frames,backend.reason = 0,''
        backend.timings = np.zeros(5)
        backend.clouds,backend.node_ids,backend.geometry_poses = {},[],[]
        backend.voxel_map = VoxelMap(.003)
        backend.handle = 1
        backend.geometry_engine = Mock()
        return backend

    def test_geometry_is_accepted_with_completely_blank_rgb_and_depth(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            backend = self.backend(folder)
            poses = np.eye(4)[None]
            backend.geometry_engine.register.return_value = {'accepted':True,'poses':poses.tolist(),'edges':[],
                'reference':None,'overlap':1.,'rmse_mm':0.,'rgb_inliers':0,'geometry_edges':0}
            class Library:
                def snapshot_add(self,*args):
                    ct.cast(args[-1],ct.POINTER(ct.c_int))[0] = 7
                    return 2
                def geometry_sync(self,*args):
                    return 0
                def online_times(self,*args):
                    pass
            backend.lib = Library()
            points = np.array([[.001,0,1.],[.002,0,1.]])
            result = backend.process(np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint16),np.eye(3),1.,geometry=points)
            self.assertEqual(result,2)
            self.assertEqual(backend.node_ids,[7])
            backend.geometry_engine.request.assert_called_once_with('commit')
            self.assertEqual(backend.report['rgb_inliers'],0)
            self.assertTrue((Path(folder)/'geometry_poses.json').exists())

    def test_rejected_geometry_never_commits_or_touches_native_map(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            backend = self.backend(folder)
            backend.lib = Mock()
            backend.geometry_engine.register.return_value = {'accepted':False,'reason':'几何歧义'}
            result = backend.process(None,None,None,1.,geometry=np.ones((100,3)))
            self.assertEqual(result,0)
            backend.lib.snapshot_add.assert_not_called()
            backend.geometry_engine.request.assert_not_called()

    def test_joint_pose_changes_reintegrate_old_measurements(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            backend = self.backend(folder)
            backend.node_ids,backend.geometry_poses = [1,2],np.array([np.eye(4),np.eye(4)])
            backend.clouds = {1:np.array([[.001,0,1.]]),2:np.array([[.002,0,1.]])}
            backend.preview()
            self.assertEqual(len(backend.voxel_map.cloud()),1)
            backend.geometry_poses[0,0,3] = 1.
            backend.preview()
            points = backend.voxel_map.cloud()
            self.assertEqual(len(points),2)
            np.testing.assert_allclose(sorted(points[:,0]),[.002,1.001])

    def test_disk_backed_preview_updates_only_changed_nodes(self):
        from disk_map import CloudStore,DiskVoxelMap
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            backend=self.backend(folder)
            backend.clouds=CloudStore(Path(folder)/'frames')
            backend.voxel_map=DiskVoxelMap(Path(folder)/'map.sqlite',.003)
            backend.node_ids=[1,2];backend.geometry_poses=np.array([np.eye(4),np.eye(4)])
            backend.clouds[1]=np.array([[.001,0,1.]])
            backend.clouds[2]=np.array([[.002,0,1.]])
            try:
                cloud,_=backend.preview();self.assertEqual(len(cloud),1)
                backend.geometry_poses[0,0,3]=1.
                cloud,_=backend.preview();self.assertEqual(len(cloud),2)
                changes=backend.voxel_map.db.total_changes
                backend.preview();self.assertEqual(backend.voxel_map.db.total_changes,changes)
            finally:backend.voxel_map.close()

    def test_async_commit_has_one_writer_and_native_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            backend=self.backend(folder);backend.node_ids=[1,2]
            backend.geometry_poses=np.array([np.eye(4),np.eye(4)])
            changed=backend.geometry_poses.copy();changed[1,0,3]=.02
            candidate=dict(accepted=True,poses=changed.tolist(),edges=[],version=2,background=dict(source=1))
            backend.geometry_engine.poll_loop.return_value=candidate
            backend.lib=Mock();backend.lib.geometry_sync.return_value=1;backend.lib.online_error.return_value=b'native failure'
            with self.assertRaises(RuntimeError):backend.poll_background()
            self.assertEqual(backend.geometry_poses[1,0,3],0)
            backend.geometry_engine.request.assert_called_once_with('discard_loop')
            backend.geometry_engine.request.reset_mock();backend.lib.geometry_sync.return_value=0
            self.assertTrue(backend.poll_background())
            backend.geometry_engine.request.assert_called_once_with('commit_loop')
            self.assertEqual(backend.geometry_poses[1,0,3],.02)
            saved=json.loads((Path(folder)/'geometry_poses.json').read_text())
            self.assertEqual(saved['poses'][1][0][3],.02)

    def test_video_duplicate_keyframe_does_not_add_native_node(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            backend=self.backend(folder);backend.lib=Mock()
            backend.node_ids=[1];backend.geometry_poses=np.array([np.eye(4)])
            backend.keyframe_translation=.03;backend.keyframe_rotation=np.deg2rad(5)
            next_pose=np.eye(4);next_pose[0,3]=.01
            backend.geometry_engine.register.return_value=dict(accepted=True,poses=[np.eye(4).tolist(),next_pose.tolist()],edges=[])
            result=backend.process(None,None,None,1.,geometry=np.ones((10,3)))
            self.assertEqual(result,1);self.assertTrue(backend.report['skipped'])
            backend.lib.snapshot_add.assert_not_called();backend.geometry_engine.discard_capture.assert_called_once()
