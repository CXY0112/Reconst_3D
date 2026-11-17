"""Use an explicit interpreter or a selected, inference-verified WSL runtime."""
from pathlib import Path, PureWindowsPath
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parent


def default_nksr_python():
    if os.environ.get('RECONST_NKSR_PYTHON'): return os.environ['RECONST_NKSR_PYTHON']
    marker = ROOT/'vendor/nksr_linux/nksr_ready.json'
    if os.name == 'nt' and marker.is_file():
        try:
            info = json.loads(marker.read_text(encoding='utf-8'))
            distribution = info.get('distribution', '')
            if (info.get('extension_imported') and info.get('real_inference_verified')
                    and info.get('selected_for_comparison') and re.fullmatch(r'[A-Za-z0-9_.-]+', distribution)):
                return 'wsl:'+distribution
        except (OSError, ValueError): pass
    return sys.executable


def wsl_path(path):
    text = str(path)
    if text.startswith('/'): return text
    value = PureWindowsPath(text)
    if not re.fullmatch('[A-Za-z]:', value.drive) or not value.is_absolute():
        raise ValueError('WSL bridge requires an absolute Windows drive path')
    return '/mnt/'+value.drive[0].lower()+'/'+ '/'.join(value.parts[1:])


def inference_command(python, script, arguments):
    python = str(python)
    if not python.startswith('wsl:'): return [python, str(script), *map(str, arguments)]
    distribution = python[4:]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', distribution): raise ValueError('Invalid WSL distribution name')
    marker = ROOT/'vendor/nksr_linux/nksr_ready.json'
    prefix = '/opt/reconst_3d/nksr/env'
    if marker.is_file():
        info = json.loads(marker.read_text(encoding='utf-8'))
        prefix = info.get('prefix', prefix)
    if not isinstance(prefix, str) or not prefix.startswith('/') or any(c in prefix for c in '\n\r\0'):
        raise ValueError('Invalid Linux NKSR environment prefix')
    converted = []
    for index, argument in enumerate(arguments):
        converted.append(wsl_path(argument) if index and arguments[index-1] in ('--source', '--packet') else str(argument))
    return ['wsl.exe', '--distribution', distribution, '--exec', 'env',
            'CUDA_HOME='+prefix, 'LD_LIBRARY_PATH='+prefix+'/lib:'+prefix+'/targets/x86_64-linux/lib',
            'TORCH_HOME='+prefix.rstrip('/').rsplit('/', 1)[0]+'/torch_cache',
            'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1', prefix+'/bin/python', wsl_path(script), *converted]
