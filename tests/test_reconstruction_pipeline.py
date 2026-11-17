import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from reconstruct import ROOT
from mesh_processing import retention_schedule, select_decimation


class FakeMesh:
    def __init__(self, count=1000, area=1.):
        self.triangles = range(len(count.triangles) if isinstance(count, FakeMesh) else count)
        self.area = area
    def get_surface_area(self): return self.area
    def remove_degenerate_triangles(self): pass
    def remove_duplicated_triangles(self): pass
    def remove_unreferenced_vertices(self): pass
    def compute_vertex_normals(self): pass


class ProductionReconstructionTests(unittest.TestCase):
    def exercise_schedule(self, errors, areas=None):
        targets = []
        def decimate(fixed, large, target):
            targets.append(target)
            return FakeMesh(target, areas[len(targets)-1] if areas else 1.)
        with patch('mesh_processing.component_meshes', return_value=(FakeMesh(0), [FakeMesh()])), \
             patch('mesh_processing.decimate_to_total', side_effect=decimate), \
             patch('mesh_processing.scene_for', return_value=object()), \
             patch('mesh_processing.sampled_surface', return_value=np.zeros((2, 3))), \
             patch('mesh_processing.distances', side_effect=[np.array([error]) for error in errors for _ in range(2)]):
            result, attempts, _ = select_decimation(FakeMesh(), FakeMesh(), object(), np.zeros((2, 3)))
        return result, attempts, targets

    def test_starts_at_40_and_increases_by_10_until_first_success(self):
        self.assertEqual(retention_schedule(), (.4, .5, .6, .7, .8, .9, 1.))
        mesh, attempts, targets = self.exercise_schedule([20., 16., 14.])
        self.assertEqual(targets, [400, 500, 600])
        self.assertEqual([a['target_ratio'] for a in attempts], [.4, .5, .6])
        self.assertEqual(len(mesh.triangles), 600)
        self.assertEqual([a['accepted'] for a in attempts], [False, False, True])

    def test_area_check_can_reject_an_otherwise_close_candidate(self):
        _, attempts, targets = self.exercise_schedule([1., 1.], areas=[.9, .99])
        self.assertEqual(targets, [400, 500])
        self.assertFalse(attempts[0]['accepted']); self.assertTrue(attempts[1]['accepted'])

    def test_100_percent_is_last_fallback_and_failure_is_explicit(self):
        mesh, attempts, targets = self.exercise_schedule([20.]*6+[0.])
        self.assertEqual([a['target_ratio'] for a in attempts], list(retention_schedule()))
        self.assertEqual(len(mesh.triangles), 1000)
        with self.assertRaises(RuntimeError): self.exercise_schedule([20.]*7)

    def test_single_worker_final_export_and_input_protection(self):
        with tempfile.TemporaryDirectory(prefix='pipeline_test_', dir=ROOT/'results') as temporary:
            folder = Path(temporary).resolve()
            self.assertEqual(folder.parent, (ROOT/'results').resolve())
            source = folder/'input'; source.mkdir()
            n = 2000
            z = 1-2*(np.arange(n)+.5)/n
            angle = np.arange(n)*np.pi*(3-np.sqrt(5))
            normals = np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
            np.save(source/'processed_world.npy', normals*.1)
            np.savez(source/'surfel_attributes.npz', normals_world=normals)
            (source/'report.json').write_text('{}', encoding='utf-8')
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            output = folder/'output'
            command = [sys.executable, str(ROOT/'tools/reconstruct_geometry.py'), '--source', str(source)]
            result = subprocess.run(command+['--output', str(output)], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            self.assertIn('1/6 Screened Poisson', result.stdout)
            self.assertEqual({p.name for p in output.iterdir()}, {'mesh.ply', 'report.json'})
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['complete']); self.assertTrue(report['source_unchanged'])
            self.assertEqual(report['decimation_attempts'][0]['target_ratio'], .4)
            self.assertTrue(report['decimation_attempts'][-1]['accepted'])
            self.assertTrue(report['final']['planar_topology_unchanged'])
            self.assertLessEqual(report['planar_acceptance'][-1]['sampled_two_way_max_mm'], 15.)
            self.assertGreater(report['final']['triangles'], 0)
            self.assertGreaterEqual(report['wall_seconds_including_worker_startup'], report['elapsed_seconds'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            blocked = subprocess.run(command+['--output', str(source/'bad_output')], capture_output=True,
                                     text=True, encoding='utf-8', errors='replace', timeout=30)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertFalse((source/'bad_output').exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
