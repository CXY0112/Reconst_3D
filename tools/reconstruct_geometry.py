"""Production geometry reconstruction; Poisson, NKSR or SDK TSDF."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import geometry_python
from result_paths import stage_result_path, latest_preprocessed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Preprocessed folder with world points and surfel normals')
    parser.add_argument('--method', choices=('poisson','nksr','tsdf'),default='poisson')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--debug', action='store_true', help='Also save intermediate meshes')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--preview-packet', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONIOENCODING='utf-8')
        output = args.output or stage_result_path(3)
        started = time.perf_counter()
        process = subprocess.Popen([str(geometry_python()), '-u',
            str(Path(__file__).resolve()), '--worker', *sys.argv[1:], '--output', str(output)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in process.stdout: print(line, end='', flush=True)
        status = process.wait()
        if status == 0:
            path = output/'report.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            report['wall_seconds_including_worker_startup'] = time.perf_counter()-started
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        raise SystemExit(status)
    from runtime_paths import activate_geometry_packages
    activate_geometry_packages()
    sys.path.insert(0, str(ROOT))
    from reconstruction_pipeline import run_pipeline
    if args.source:
        source = args.source.resolve()
    else:
        source = latest_preprocessed()
    output = args.output or stage_result_path(3)
    run_pipeline(source, output, debug=args.debug, preview_packet=args.preview_packet,method=args.method)


if __name__ == '__main__': main()
