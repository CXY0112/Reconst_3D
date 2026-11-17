"""Optional geometry comparison. No arguments: compare latest saved Kalman result."""
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
from nksr_runtime import default_nksr_python


def inference(args):
    import numpy as np
    import torch
    import nksr
    from geometry_comparison import load_source, nksr_scale_options
    points, normals, _ = load_source(args.source)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    stamp = time.perf_counter()
    # NKSR may enable gradients internally to evaluate spatial derivatives.
    with torch.no_grad():
        model = nksr.Reconstructor(device)
        field = model.reconstruct(torch.tensor(points, dtype=torch.float32, device=device),
                                  torch.tensor(normals, dtype=torch.float32, device=device),
                                  **nksr_scale_options(args.voxel_mm, args.detail_level))
        if field is None: raise RuntimeError('NKSR produced no field')
        mesh = field.extract_dual_mesh(mise_iter=1)
        vertices, faces = mesh.v.detach().cpu().numpy(), mesh.f.detach().cpu().numpy()
    np.savez(args.packet, vertices=vertices, triangles=faces, seconds=time.perf_counter()-stamp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--nksr-python', default=default_nksr_python())
    scale = p.add_mutually_exclusive_group()
    scale.add_argument('--voxel-mm', type=float, help='Explicit fixed finest voxel size in mm')
    scale.add_argument('--detail-level', type=float, help='Automatic density scale, from 0 to 1 (default: 0.4)')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda:0'), default='auto')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--infer', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--packet', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.voxel_mm is None and args.detail_level is None: args.detail_level = .4
    if args.probe:
        import platform
        info = {'python': sys.executable, 'version': sys.version, 'platform': platform.platform()}
        try:
            import torch
            info.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
            import nksr
            from nksr import _C
            info.update(nksr=getattr(nksr, '__version__', 'unknown'), extension=str(_C.__file__))
        except Exception as exc:
            info['error'] = f'{type(exc).__name__}: {exc}'
            print(json.dumps(info, indent=2)); return 2
        print(json.dumps(info, indent=2)); return 0
    if args.infer:
        inference(args); return 0
    from geometry_comparison import nksr_scale_options
    try: nksr_scale_options(args.voxel_mm, args.detail_level)
    except ValueError as exc: p.error(str(exc))
    from result_paths import latest_preprocessed
    from runtime_paths import geometry_python, activate_geometry_packages
    source = args.source or latest_preprocessed()
    source = source.resolve()
    output = args.output or source.parent/('geometry_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not args.worker:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONIOENCODING='utf-8')
        return subprocess.call([str(geometry_python()), '-u', str(Path(__file__).resolve()), '--worker',
            '--source', str(source), '--output', str(output), '--nksr-python', str(args.nksr_python),
            *(['--detail-level', str(args.detail_level)] if args.detail_level is not None
              else ['--voxel-mm', str(args.voxel_mm)]),
            '--device', args.device], env=env)
    activate_geometry_packages()
    from geometry_comparison import run_comparison
    report = run_comparison(source, output, args.nksr_python, args.voxel_mm, args.device, args.detail_level)
    return 0 if report['complete'] else 2


if __name__ == '__main__': sys.exit(main())
