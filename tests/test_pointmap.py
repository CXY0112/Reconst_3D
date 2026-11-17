"""Pixel identity, edge preservation, recoverable exports and shared grid helpers."""
from pathlib import Path
import tempfile,threading,time,unittest
from unittest.mock import Mock
import numpy as np
import pointmap_processing as p
from map_cloud_view import cpu_map_image
from live_cloud_view import ViewCamera

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    k=np.array([[120.,0,32],[0,120,24],[0,0,1]])
    rgb=np.zeros((48,64,3),np.uint8);rgb[:,:32]=[255,20,30];rgb[:,32:]=[5,100,200]
    rng=np.random.default_rng(7);depth=(.8+rng.normal(0,.003,(48,64))).astype(np.float32);depth[10:13,10:13]=np.nan
    p.save_frame(folder/'setp1/frame_000001',rgb,depth,dict(K=k.tolist(),model='test_pixel_grid'))
    p.write_json(folder/'pointmap_dataset.json',dict(schema=p.SCHEMA,frames=[dict(capture=7,folder='setp1/frame_000001')]))
    return rgb,depth,k


class PointmapTests(unittest.TestCase):
    def test_rgbd_launcher_does_not_start_normal_or_beta_pipeline(self):
        from unittest.mock import patch
        import runpy
        with patch('sys.argv',['main.py','rgbd']),patch('scene_workbench.launch') as launch,patch('workbench.launch') as normal:
            runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
            launch.assert_called_once_with(beta=False);normal.assert_not_called()

    def test_clean_preserves_rgb_rays_holes_and_reduces_plane_noise(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            rgb,depth,k=fixture(Path(tmp));original=depth.copy();result=p.clean(rgb,depth,k)
            np.testing.assert_array_equal(result['rgb'],rgb)
            np.testing.assert_array_equal(np.isnan(original),np.isnan(depth))
            self.assertTrue(np.isnan(result['xyz'][10:13,10:13]).all())
            core=np.s_[5:-5,5:-5]
            self.assertLess(np.nanstd(result['depth'][core]),np.nanstd(depth[core])*.65)
            self.assertLess(result['report']['pixel_roundtrip_max_px'],.001)
            self.assertLessEqual(result['report']['max_movement_mm'],10.001)
            np.testing.assert_allclose(result['xyz'][...,2],result['depth'],equal_nan=True)

    def test_depth_step_not_averaged_and_isolated_speckle_rejected(self):
        rgb=np.full((32,32,3),127,np.uint8);depth=np.ones((32,32),np.float32);depth[:,16:]=1.1
        depth[10,10]=1.25;k=np.array([[100,0,16],[0,100,16],[0,0,1]])
        result=p.clean(rgb,depth,k)
        self.assertTrue(np.isnan(result['depth'][10,10]))
        np.testing.assert_allclose(result['depth'][5:-5,15],1,atol=1e-6)
        np.testing.assert_allclose(result['depth'][5:-5,16],1.1,atol=1e-6)

    def test_export_roundtrip_and_create_only_protects_original(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);source=folder/'input';source.mkdir();rgb,_,_=fixture(source)
            rows=p.sources(source);hashes={r:p.sha(rows[0][r]) for r in ('rgb','depth')}
            target=p.export_sequence(rows,folder/'output');restored=p.sources(target)
            np.testing.assert_array_equal(p.load(restored[0])[0],rgb)
            xyz=np.load(target/'setp2/frame_000001/xyz_opencv.npy')
            flipped=np.load(target/'setp2/frame_000001/xyz_pytorch3d.npy')
            np.testing.assert_allclose(flipped,xyz*[-1,-1,1],equal_nan=True)
            self.assertEqual(restored[0]['capture'],7)
            for r,h in hashes.items():self.assertEqual(p.sha(rows[0][r]),h)
            with self.assertRaises(FileExistsError):p.export_sequence(rows,target)

    def test_cancel_removes_only_incomplete_owned_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);source=folder/'input';source.mkdir();fixture(source)
            cancel=threading.Event();cancel.set()
            with self.assertRaises(InterruptedError):p.export_sequence(p.sources(source),folder/'output',cancel=cancel)
            self.assertFalse((folder/'output').exists());self.assertTrue((source/'pointmap_dataset.json').exists())
            self.assertFalse(list(folder.glob('.pending_pointmap_*')))

    def test_native_reprocessing_replaces_only_stage2_and_cancel_retains_old(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);source=folder/'input';source.mkdir();fixture(source)
            target=p.export_sequence(p.sources(source),folder/'output');rows=p.sources(target)
            raw_hash={path:p.sha(path) for path in (target/'setp1').rglob('*') if path.is_file()}
            report=target/'setp2/frame_000001/report.json';old=p.sha(report)
            cancel=threading.Event();cancel.set()
            with self.assertRaises(InterruptedError):p.export_sequence(rows,target,cancel=cancel,overwrite=True)
            self.assertEqual(p.sha(report),old)
            p.export_sequence(rows,target,params=p.Parameters(depth_sigma_mm=4),overwrite=True)
            self.assertEqual(p.read_json(report)['parameters']['depth_sigma_mm'],4)
            for path,h in raw_hash.items():self.assertEqual(p.sha(path),h)

    def test_colored_raster_nearest_surface_wins(self):
        points=np.array([[0,0,1],[0,0,2]],np.float32);colors=np.array([[255,0,0],[0,0,255]],np.uint8)
        image=cpu_map_image(points,np.empty((0,3)),ViewCamera(),100,100,colors)
        np.testing.assert_array_equal(image[50,50],[255,0,0])

    def test_malformed_input_and_manifest_escape_rejected(self):
        with self.assertRaises(ValueError):p.clean(np.zeros((3,3,3),np.uint8),np.ones((4,4)),np.eye(3))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);p.write_json(folder/'pointmap_dataset.json',dict(schema=p.SCHEMA,frames=[dict(capture=1,folder='../outside')]))
            with self.assertRaises(ValueError):p.sources(folder)


if __name__=='__main__':unittest.main(verbosity=2)
