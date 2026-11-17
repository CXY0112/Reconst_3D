import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from tools.export_brush import export,BASIS


class BrushExportTests(unittest.TestCase):
    def test_timestamp_join_camera_basis_and_training_only_rgb_initialization(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'results') as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            for name in ['rgb','depth','map_frames']:(source/name).mkdir()
            pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',70,degrees=True).as_matrix();pose[:3,3]=[2,1,.5]
            (source/'geometry_poses.json').write_text(json.dumps(dict(node_ids=[8,20],poses=[np.eye(4).tolist(),pose.tolist()])))
            (source/'frames.jsonl').write_text('\n'.join(json.dumps(dict(capture=i,stamp=float(i))) for i in [1,2]))
            (source/'registrations.jsonl').write_text('\n'.join(json.dumps(dict(capture=i,node=n,accepted=True)) for i,n in [(2,20),(1,8)]))
            (source/'sdk_calibration.json').write_text(json.dumps(dict(fxrgb=20,fyrgb=20,cxrgb=16,cyrgb=16)))
            for i,node in [(1,8),(2,20)]:
                cv2.imwrite(str(source/'rgb'/f'{i:.6f}.png'),np.full((32,32,3),[10,20,30] if i==2 else [200,150,100],np.uint8))
                cv2.imwrite(str(source/'depth'/f'{i:.6f}.png'),np.full((32,32),1000,np.uint16))
                np.save(source/'map_frames'/f'{node}.npy',np.array([[0.,0,1],[.1,0,1],[0,.1,1]]))
            out=root/'export';report=export(source,out)
            data=json.loads((out/'transforms.json').read_text())
            self.assertTrue(data['frames'][1]['file_path'].endswith('2.000000.png'))
            np.testing.assert_allclose(np.array(data['frames'][1]['transform_matrix'])@BASIS,pose)
            self.assertLess(report['max_coordinate_roundtrip_error_m'],1e-12)
            self.assertEqual((report['train_views'],report['eval_views']),(1,1))
            lines=(out/'initial.ply').read_text().split('end_header\n')[1].splitlines()
            rows=np.array([list(map(float,line.split())) for line in lines])
            np.testing.assert_array_equal(rows[:,3:],np.tile([30,20,10],(3,1)))
            with self.assertRaises(FileExistsError):export(source,out)


if __name__=='__main__':unittest.main()
