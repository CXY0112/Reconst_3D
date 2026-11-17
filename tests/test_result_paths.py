from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from result_paths import stage_result_path, result_reports, scan_clouds
from reconstruct import ROOT


class ResultPathTests(unittest.TestCase):
    def test_stage_names_and_experiment_tags(self):
        instant = datetime(2026, 10, 15, 16, 20, 30, 123456)
        for stage in (1, 2, 3, 4):
            self.assertEqual(stage_result_path(stage, now=instant).name, f'setp{stage}_20261015_162030_123456')
        self.assertEqual(stage_result_path(3, now=instant, tag='planar').name, 'setp3_20261015_162030_123456_planar')
        with self.assertRaises(ValueError): stage_result_path(5)
        with self.assertRaises(ValueError): stage_result_path(3, tag='../bad')

    def test_discovery_accepts_old_and_new_names_but_ignores_incomplete(self):
        with tempfile.TemporaryDirectory(prefix='paths_test_', dir=ROOT/'results') as temporary:
            root = Path(temporary).resolve(); self.assertEqual(root.parent, (ROOT/'results').resolve())
            for name in ('geometry_final_old', 'setp3_20261015_1', 'setp3_incomplete'):
                folder = root/name; folder.mkdir()
                (folder/'report.json').write_text(json.dumps({'complete': name != 'setp3_incomplete',
                    'pipeline':'poisson_supported_smooth_qem_planar'}), encoding='utf-8')
            self.assertEqual({p.parent.name for p, _ in result_reports('final', root)}, {'geometry_final_old','setp3_20261015_1'})
            for name in ('scan_old','setp1_new'):
                folder=root/name;folder.mkdir();(folder/'map_cloud.ply').touch()
            self.assertEqual({p.parent.name for p in scan_clouds(root)}, {'scan_old','setp1_new'})
