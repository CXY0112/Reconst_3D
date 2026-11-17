import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from geometry_comparison import load_source, nksr_scale_options

ROOT = Path(__file__).resolve().parents[1]


class GeometryComparisonTests(unittest.TestCase):

    def test_detail_zero_is_forwarded_without_a_voxel_override(self):
        self.assertEqual(nksr_scale_options(detail_level=0.), {'detail_level': 0.})
        self.assertEqual(nksr_scale_options(detail_level=.8), {'detail_level': .8})
        self.assertEqual(nksr_scale_options(), {'detail_level': .4})
        self.assertEqual(nksr_scale_options(20.), {'voxel_size': .02})
        for voxel, detail in ((10.,0.), (None,-.1), (None,1.1), (None,float('nan'))):
            with self.assertRaises(ValueError): nksr_scale_options(voxel,detail)

    def source(self, parent, invalid=False):
        folder = Path(parent)/'source'; folder.mkdir()
        n = 2000
        z = 1-2*(np.arange(n)+.5)/n
        angle = np.arange(n)*np.pi*(3-np.sqrt(5))
        normals = np.column_stack((np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z))
        points = normals*.1
        if invalid: points[0, 0] = np.nan
        np.save(folder/'processed_world.npy', points)
        np.savez(folder/'surfel_attributes.npz', normals_world=normals*2)
        (folder/'report.json').write_text('{}', encoding='utf-8')
        return folder

    def test_normalization_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory(prefix='comparison_input_test_', dir=ROOT/'results') as temp:
            source = self.source(temp)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            _, normals, hashes = load_source(source)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)
            self.assertEqual(set(hashes), set(before))
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            np.save(source/'processed_world.npy', np.full((2000, 3), np.nan))
            with self.assertRaises(ValueError): load_source(source)

    def test_unavailable_nksr_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory(prefix='comparison_test_', dir=ROOT/'results') as temp:
            source = self.source(temp); output = Path(temp)/'comparison'
            # A controlled independent process failure, independent of installed NKSR.
            fake = Path(temp)/'unavailable.py'
            fake.write_text('import sys\nprint("controlled unavailable runtime")\nsys.exit(2)\n', encoding='utf-8')
            # The independent launcher fails regardless of installed NKSR.
            import os
            if os.name != 'nt': self.skipTest('Windows launcher integration')
            wrapper = Path(temp)/'no_nksr.cmd'
            wrapper.write_text(f'@"{sys.executable}" "{fake}"\n', encoding='utf-8')
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            result = subprocess.run([sys.executable, str(ROOT/'tools/compare_geometry.py'),
                '--source', str(source), '--output', str(output), '--nksr-python', str(wrapper)],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output/'comparison.json').read_text(encoding='utf-8'))
            self.assertFalse(report['complete']); self.assertTrue(report['input_unchanged'])
            self.assertEqual(report['methods']['poisson']['status'], 'complete')
            self.assertEqual(report['methods']['nksr']['status'], 'unavailable')
            self.assertFalse((output/'nksr').exists())
            for variant in ('raw', 'supported'):
                self.assertTrue((output/'poisson'/f'{variant}.ply').is_file())
                metrics = report['methods']['poisson']['variants'][variant]
                self.assertGreater(metrics['input_coverage']['10'], .99)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})



if __name__ == '__main__': unittest.main()
