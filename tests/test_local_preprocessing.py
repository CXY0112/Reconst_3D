import json,queue,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from reconstruct import ROOT
from preprocess import Parameters
from local_preprocessing import prepare_local,save_local,load_local,profile
from kalman_fusion import process_session
from test_kalman_fusion import write_session


class LocalPreprocessingTests(unittest.TestCase):
    def test_rgb_stage_resolves_new_sessions_without_raw_xyz(self):
        from test_appearance import fixture
        from appearance_pipeline import resolve_inputs
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            source,session=fixture(Path(temporary))
            x,y=np.meshgrid(np.linspace(-.1,.1,20),np.linspace(-.1,.1,20))
            data=prepare_local(np.column_stack((x.ravel(),y.ravel(),np.ones(x.size))),Parameters())
            frames=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
            for frame in frames:
                frame['cloud_source']='local_surface_v1'
                save_local(session/'pointcloud_local'/f'{frame["stamp"]:.6f}.npz',data)
            (session/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
            for path in (session/'pointcloud_raw').glob('*.npy'):path.unlink()
            (session/'pointcloud_raw').rmdir()
            _,resolved,observations,_,_=resolve_inputs(source)
            self.assertEqual(resolved,session)
            self.assertEqual(len(observations),2)
            self.assertTrue(all(row['rgb'].is_file() and row['depth'].is_file() for row in observations))

    def test_precomputed_replay_matches_legacy_with_final_pose_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);write_session(folder);(folder/'pointcloud_local').mkdir()
            params=Parameters();frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            for frame in frames:
                stamp=f'{frame["stamp"]:.6f}'
                data=prepare_local(np.load(folder/'pointcloud_raw'/(stamp+'.npy'))/1000,params)
                save_local(folder/'pointcloud_local'/(stamp+'.npz'),data)
                np.testing.assert_array_equal(data['points'],load_local(folder/'pointcloud_local'/(stamp+'.npz'))['points'])
            graph=json.loads((folder/'geometry_poses.json').read_text())
            graph['poses'][1][0][3]=.004
            (folder/'geometry_poses.json').write_text(json.dumps(graph))
            reference=np.array([[0,0,1.]]*100)
            old=process_session(folder,params,before_points=reference)
            for frame in frames:frame['cloud_source']='local_surface_v1'
            (folder/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
            (folder/'local_preprocessing.json').write_text(json.dumps(dict(schema='local_surface_v1',parameters=profile(params))))
            for path in (folder/'pointcloud_raw').glob('*.npy'):path.unlink()
            (folder/'pointcloud_raw').rmdir()
            with patch('preprocess._geometry',side_effect=AssertionError('Must not recompute local surfaces')):
                new=process_session(folder,params,before_points=reference)
            np.testing.assert_allclose(new.points,old.points,atol=1e-12)
            for key in ('normals','support_views','sigma_normal_mm'):
                np.testing.assert_allclose(new.diagnostics[key],old.diagnostics[key],atol=1e-12)
            self.assertEqual(new.report['precomputed_local_captures'],2)
            self.assertTrue(new.report['source_unchanged'])
            with self.assertRaisesRegex(ValueError,'拟合已在采集'):
                process_session(folder,Parameters(radius_mm=50),before_points=reference)

    def test_capture_saves_local_packets_only_and_discards_rejected_packets(self):
        from snapshot_window import SnapshotSession
        session=SnapshotSession(queue.Queue())
        class Backend:
            report={'accepted':True}
            def process(self,*args,**kwargs):return 2 if self.report['accepted'] else 0
        x,y=np.meshgrid(np.linspace(-.12,.12,31),np.linspace(-.12,.12,31))
        raw=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
        try:
            backend=Backend()
            for capture in (1,2):
                backend.report={'accepted':capture==1}
                session.geometry[capture]=raw.copy();session.local_geometry[capture]=raw.copy()
                session._process_frame(backend,capture,None,None,None,float(capture))
            self.assertFalse((session.folder/'pointcloud_raw').exists())
            self.assertTrue((session.folder/'pointcloud_local/1.000000.npz').exists())
            self.assertFalse((session.folder/'pointcloud_local/2.000000.npz').exists())
            self.assertEqual(session.local_geometry,{})
            self.assertEqual(session.geometry,{})
        finally:session.discard()

    def test_local_packet_cannot_silently_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            x,y=np.meshgrid(np.linspace(-.1,.1,20),np.linspace(-.1,.1,20))
            data=prepare_local(np.column_stack((x.ravel(),y.ravel(),np.ones(x.size))),Parameters())
            path=Path(temporary)/'frame.npz';save_local(path,data)
            with self.assertRaises(FileExistsError):save_local(path,data)


if __name__=='__main__':unittest.main()
