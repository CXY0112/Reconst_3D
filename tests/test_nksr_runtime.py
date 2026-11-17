from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from nksr_runtime import default_nksr_python, inference_command, wsl_path


class NksrRuntimeTests(unittest.TestCase):
    def test_only_selected_and_inference_verified_runtime_becomes_default(self):
        with tempfile.TemporaryDirectory(prefix='nksr_selection_', dir=Path(__file__).resolve().parents[1]/'results') as temp:
            root = Path(temp); marker = root/'vendor/nksr_linux/nksr_ready.json'
            marker.parent.mkdir(parents=True)
            info = dict(extension_imported=True, real_inference_verified=False, selected_for_comparison=True,
                        distribution='Ubuntu-22.04')
            with patch('nksr_runtime.ROOT', root), patch.dict(os.environ, RECONST_NKSR_PYTHON=''):
                marker.write_text(json.dumps(info))
                self.assertEqual(default_nksr_python(), sys.executable)
                info['real_inference_verified'] = True; marker.write_text(json.dumps(info))
                self.assertEqual(default_nksr_python(), 'wsl:Ubuntu-22.04' if os.name == 'nt' else sys.executable)
                with patch.dict(os.environ, RECONST_NKSR_PYTHON='D:/native/python.exe'):
                    self.assertEqual(default_nksr_python(), 'D:/native/python.exe')
                info['distribution'] = 'invalid;command'; marker.write_text(json.dumps(info))
                self.assertEqual(default_nksr_python(), sys.executable)

    def test_wsl_translates_paths_with_spaces_without_shell_interpolation(self):
        with patch('nksr_runtime.ROOT', Path('C:/project test/Reconst_3D')):
            command = inference_command('wsl:Ubuntu-22.04', 'C:/project test/Reconst_3D/tools/compare_geometry.py',
                ['--infer', '--source', 'D:/my scans/test/setp2', '--packet', 'C:/result with spaces.npz', '--device', 'cpu'])
        self.assertEqual(command[:5], ['wsl.exe', '--distribution', 'Ubuntu-22.04', '--exec', 'env'])
        self.assertIn('/mnt/d/my scans/test/setp2', command)
        self.assertIn('/mnt/c/result with spaces.npz', command)
        self.assertIn('/opt/reconst_3d/nksr/env/bin/python', command)
        self.assertEqual(command[-2:], ['--device', 'cpu'])

    def test_native_interpreter_is_unchanged(self):
        self.assertEqual(inference_command('D:/env/python.exe', 'C:/test.py', ['--probe']),
                         ['D:/env/python.exe', 'C:/test.py', '--probe'])

    def test_relative_unc_and_invalid_distribution_are_rejected(self):
        for value in ('relative/file', '\\\\server\\share\\data'):
            with self.assertRaises(ValueError): wsl_path(value)
        with self.assertRaises(ValueError): inference_command('wsl:Ubuntu;bad', '/tmp/test.py', [])


if __name__ == '__main__': unittest.main()
