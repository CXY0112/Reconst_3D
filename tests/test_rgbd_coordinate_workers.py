"""Exercise the actual bundled geometry runtime with newly labelled RGB clouds."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from reconstruct import ROOT,save_cloud
from rgbd_geometry import coordinate_metadata


class CoordinateWorkerTests(unittest.TestCase):
    def test_rgb_local_cloud_tsdf_and_appearance_keep_original_rgb_pose(self):
        from test_tsdf_stage import session_fixture
        from local_preprocessing import prepare_local,save_local
        from preprocess import Parameters
        from appearance_pipeline import resolve_inputs
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            parent=Path(temporary);_,session,_=session_fixture(parent)
            rows=[json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
            graph=json.loads((session/'geometry_poses.json').read_text());world=[]
            for row,matrix in zip(rows,graph['poses']):
                name=f'{row["stamp"]:.6f}'
                points=np.load(session/'pointcloud_raw'/(name+'.npy'))/1000
                save_local(session/'pointcloud_local'/(name+'.npz'),prepare_local(points,Parameters(),coordinate_frame='rgb'))
                row.update(cloud_source='local_surface_v1',**coordinate_metadata('rgb','rgb'))
                pose=np.array(matrix);world.append(points@pose[:3,:3].T+pose[:3,3])
            (session/'frames.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            points=np.vstack(world);np.save(session/'full_geometry_world.npy',points)
            save_cloud(session/'map_cloud.ply',points,'RTAB-Map world XYZ in metres')
            output=parent/'tsdf'
            run=subprocess.run([sys.executable,str(ROOT/'tools/reconstruct_geometry.py'),'--source',str(session),
                                '--method','tsdf','--output',str(output)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertTrue(json.loads((output/'report.json').read_text(encoding='utf-8'))['complete'])
            _,_,resolved,_,_=resolve_inputs(output)
            for row,matrix in zip(resolved,graph['poses']):np.testing.assert_allclose(row['pose'],matrix,atol=1e-12)
            colored=parent/'colored'
            run=subprocess.run([sys.executable,str(ROOT/'tools/colorize_geometry.py'),'--source',str(output),
                                '--output',str(colored)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertTrue(json.loads((colored/'report.json').read_text(encoding='utf-8'))['complete'])
