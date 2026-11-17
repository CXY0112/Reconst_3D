import json
from pathlib import Path
import queue
import tempfile
import unittest
import numpy as np
from reconstruct import Calibration,ROOT
from rgbd_geometry import capture_geometry,coordinate_metadata,rgb_geometry
from rtabmap_scan import RGBDCamera,register_rgbd
from video_capture import VideoFrame,VideoSession


def calibration():
    # Deliberately different intrinsics and a 90-degree extrinsic expose any
    # accidental assumption that IR and RGB are interchangeable.
    return Calibration(fxir=2,fyir=2,cxir=1,cyir=1,fxrgb=11,fyrgb=11,cxrgb=2,cyrgb=2,
                       R01=-1,R10=1,R22=1,T1=100,T3=100)


class CoordinateTests(unittest.TestCase):
    def test_rgb_sdk_bypasses_extrinsic_but_explicit_ir_sdk_uses_it(self):
        c=calibration();color=np.zeros((8,8,3),np.uint8);raw=np.zeros((8,8),np.uint16)
        sdk=np.array([[100.,0.,1000.]])
        points,frame=capture_geometry(raw,c,sdk,depth_frame='rgb')
        self.assertEqual(frame,'rgb')
        np.testing.assert_allclose(rgb_geometry(points,c,frame),[[.1,0,1.]])
        _,depth,_=register_rgbd(color,raw,c,points_rgb=sdk)
        self.assertEqual(depth[2,3],1000);self.assertEqual(np.count_nonzero(depth),1)
        _,depth,_=register_rgbd(color,raw,c,points_ir=sdk)
        self.assertEqual(depth[3,3],1100);self.assertEqual(np.count_nonzero(depth),1)
        points,frame=capture_geometry(raw,c,sdk,points_frame='ir')
        np.testing.assert_allclose(rgb_geometry(points,c,frame),[[.1,.1,1.1]])

    def test_rgb_raw_fallback_uses_rgb_rays_and_preserves_rgb_grid(self):
        c=calibration();raw=np.arange(64,dtype=np.uint16).reshape(8,8)+1000
        points,frame=capture_geometry(raw,c,depth_frame=RGBDCamera.depth_frame)
        self.assertEqual(frame,'rgb')
        np.testing.assert_allclose(points[0],[-2/11,-2/11,1.])
        _,depth,_=register_rgbd(np.zeros((8,8,3),np.uint8),raw,c,depth_frame='rgb')
        np.testing.assert_array_equal(depth,raw)
        # Raw IR still projects through IR intrinsics and one extrinsic.
        raw[:]=0;raw[1,1]=1000
        _,depth,_=register_rgbd(np.zeros((8,8,3),np.uint8),raw,c)
        self.assertEqual(depth[2,3],1100);self.assertEqual(np.count_nonzero(depth),1)

    def test_ambiguous_invalid_frames_and_empty_rgb_observation(self):
        c=calibration();color=np.zeros((8,8,3),np.uint8);raw=np.zeros((8,8),np.uint16)
        with self.assertRaises(ValueError):register_rgbd(color,raw,c,points_ir=[],points_rgb=[])
        with self.assertRaises(ValueError):capture_geometry(raw,c,depth_frame='unknown')
        with self.assertRaises(ValueError):VideoFrame(color,raw,np.empty((0,3)),c,1.,(1,1),False,'rgb','ir')
        for cloud in (None,np.empty((0,3)),np.array([[np.nan,0,1000],[0,0,0]])):
            _,depth,_=register_rgbd(color,raw,c,points_rgb=cloud,depth_frame='rgb')
            self.assertFalse(depth.any())

    def test_video_tracker_receives_same_rgb_geometry_from_sdk_and_raw(self):
        c=calibration();raw=np.full((8,8),1000,np.uint16);color=np.zeros((8,8,3),np.uint8)
        points,_=capture_geometry(raw,c,depth_frame='rgb')
        class Backend:
            tracking_first=True
            report=dict(accepted=False,skipped=True)
            def process(self,color,depth,k,stamp,geometry):
                np.testing.assert_allclose(geometry,points)
                np.testing.assert_array_equal(depth,raw)
                return 1
        for sdk in (False,True):
            session=VideoSession(queue.Queue())
            try:
                packet=VideoFrame(color,raw,points,c,1.,(1,1),sdk,'rgb')
                session._process_frame(Backend(),1,packet,None,None,1.)
                row=json.loads((session.folder/'frames.jsonl').read_text())
                self.assertEqual(row['pointcloud_frame'],'rgb');self.assertEqual(row['raw_depth_frame'],'rgb')
            finally:session.discard()

    def test_capture_uses_camera_declaration_even_without_sdk_cloud(self):
        c=calibration();session=VideoSession(queue.Queue());raw=np.full((8,8),1000,np.uint16)
        class Camera:
            depth_frame=pointcloud_frame='rgb'
            pair_cloud=None
            def __enter__(self):self.i=0;return self
            def __exit__(self,*args):pass
            def read(self,**kwargs):
                self.i+=1
                if self.i>1:session.stop.set();raise TimeoutError()
                return ((np.zeros((8,8,3),np.uint8),None,1,1000),(raw,None,1,1000)),c
        session.camera_factory=Camera
        try:
            session._capture();self.assertFalse(session.failed)
            _,packet,*_=session.incoming.get_nowait()
            self.assertEqual(packet.geometry_frame,'rgb');self.assertEqual(packet.depth_frame,'rgb')
            np.testing.assert_allclose(packet.points[0],[-2/11,-2/11,1.])
            self.assertEqual(session.live_intrinsics,(11,11,2,2))
        finally:session.discard()

    def test_manual_snapshot_records_sdk_rgb_frame_and_rgb_preview(self):
        from snapshot_window import SnapshotSession
        c=calibration();session=SnapshotSession(queue.Queue());raw=np.full((8,8),1000,np.uint16)
        points,_=capture_geometry(raw,c,depth_frame='rgb')
        class Camera:
            depth_frame=pointcloud_frame='rgb'
            pair_cloud=points*1000
            def __enter__(self):self.i=0;return self
            def __exit__(self,*args):pass
            def read(self,**kwargs):
                self.i+=1
                if self.i>1:session.stop.set();raise TimeoutError()
                return ((np.zeros((8,8,3),np.uint8),None,1,1000),(raw,None,1,1000)),c
        session.camera_factory=Camera;session.request.set()
        try:
            session._capture();self.assertFalse(session.failed)
            np.testing.assert_allclose(session.geometry[1],points)
            np.testing.assert_allclose(session.local_geometry[1],points)
            self.assertEqual(session.local_frames[1],'rgb')
            self.assertEqual(session.live_intrinsics,(11,11,2,2))
            _,_,depth,_,_=session.incoming.get_nowait()
            np.testing.assert_array_equal(depth,raw)
            row=json.loads((session.folder/'frames.jsonl').read_text())
            self.assertEqual(row['pointcloud_frame'],'rgb')
        finally:session.discard()

    def test_downstream_pose_selects_explicit_frame_and_preserves_legacy(self):
        from test_kalman_fusion import write_session
        from kalman_fusion import session_observations,OPTICAL_FROM_WORLD
        from tsdf_reconstruction import sdk_frames
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);write_session(folder)
            c=json.loads((folder/'sdk_calibration.json').read_text());c.update(T1=100,fxrgb=11,fyrgb=11,cxrgb=2,cyrgb=2)
            (folder/'sdk_calibration.json').write_text(json.dumps(c))
            frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            frames[0].update(coordinate_metadata('rgb','rgb'))
            (folder/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
            (folder/'depth_raw').mkdir()
            for i in (1,2):(folder/'depth_raw'/f'{i:.6f}.png').touch()
            _,observations,_=session_observations(folder)
            np.testing.assert_allclose(observations[0]['pose'][:3,3],[0,0,0])
            np.testing.assert_allclose(observations[1]['pose'][:3,3],OPTICAL_FROM_WORLD@[.1,0,0])
            resolved,_,_=sdk_frames(folder)
            np.testing.assert_allclose(resolved[0]['pose'],np.eye(4))
            self.assertEqual(resolved[1]['pose'][0,3],.1)
            frames[0]['coordinate_schema']='unknown'
            (folder/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
            with self.assertRaises(ValueError):session_observations(folder)

    def test_local_packet_roundtrip_records_rgb_coordinates(self):
        from local_preprocessing import prepare_local,save_local,load_local
        from preprocess import Parameters
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            x,y=np.meshgrid(np.linspace(-.1,.1,20),np.linspace(-.1,.1,20))
            points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
            data=prepare_local(points,Parameters(),coordinate_frame='rgb')
            path=Path(temporary)/'local.npz';save_local(path,data)
            loaded=load_local(path)
            self.assertEqual(loaded['metadata']['camera_frame'],'rgb')
            np.testing.assert_array_equal(loaded['points'],data['points'])

    def test_appearance_uses_rgb_pose_without_undoing_an_unused_ir_extrinsic(self):
        from test_appearance import fixture
        from appearance_pipeline import resolve_inputs
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            source,session=fixture(Path(temporary))
            rows=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
            for row in rows:row.update(coordinate_metadata('rgb','rgb'))
            (session/'frames.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            c=json.loads((session/'sdk_calibration.json').read_text());c.update(T1=100,T2=200,T3=300)
            (session/'sdk_calibration.json').write_text(json.dumps(c))
            _,_,observations,_,_=resolve_inputs(source)
            graph=json.loads((session/'geometry_poses.json').read_text())
            np.testing.assert_allclose(observations[0]['pose'],graph['poses'][0])

    def test_rgb_stage_two_result_is_independent_of_ir_extrinsic(self):
        from test_kalman_fusion import write_session
        from local_preprocessing import prepare_local,save_local
        from kalman_fusion import process_session
        from preprocess import Parameters
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);write_session(folder)
            rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            for row in rows:
                stamp=f'{row["stamp"]:.6f}'
                points=np.load(folder/'pointcloud_raw'/(stamp+'.npy'))/1000
                save_local(folder/'pointcloud_local'/(stamp+'.npz'),prepare_local(points,Parameters(),coordinate_frame='rgb'))
                row.update(cloud_source='local_surface_v1',**coordinate_metadata('rgb','rgb'))
            (folder/'frames.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            reference=np.tile([0,0,1.],(100,1))
            original=process_session(folder,Parameters(),before_points=reference)
            c=json.loads((folder/'sdk_calibration.json').read_text())
            c.update(R00=0,R01=-1,R10=1,R11=0,T1=100,T2=200,T3=300)
            (folder/'sdk_calibration.json').write_text(json.dumps(c))
            actual=process_session(folder,Parameters(),before_points=reference)
            np.testing.assert_allclose(actual.points,original.points,atol=1e-12)
            np.testing.assert_allclose(actual.diagnostics['normals'],original.diagnostics['normals'],atol=1e-12)

    def test_replay_requires_verified_frame_for_unlabelled_recordings(self):
        from tools.replay_registration import replay_depth_frames
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary);names=['1.000000.png']
            with self.assertRaises(ValueError):replay_depth_frames(folder,names)
            self.assertEqual(replay_depth_frames(folder,names,'rgb'),{names[0]:'rgb'})
            row=dict(stamp=1.,**coordinate_metadata('rgb','rgb'))
            (folder/'frames.jsonl').write_text(json.dumps(row))
            self.assertEqual(replay_depth_frames(folder,names),{names[0]:'rgb'})


if __name__=='__main__':unittest.main()
