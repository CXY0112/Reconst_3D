"""RGB appearance worker using the existing isolated geometry runtime."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime_paths import geometry_python


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--session',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preview-packet',type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if not args.worker:
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
        raise SystemExit(subprocess.call([str(geometry_python()),str(Path(__file__).resolve()),'--worker',*sys.argv[1:]],env=env))
    from runtime_paths import activate_geometry_packages
    activate_geometry_packages()
    from result_paths import stage_result_path,result_reports
    from appearance_pipeline import run_pipeline
    source=args.source
    if source is None:
        reports=result_reports('final')
        if not reports: raise FileNotFoundError('请先完成第三阶段几何重建')
        source=reports[-1][0].parent
    run_pipeline(source,args.output or stage_result_path(4),args.session,args.preview_packet,enhanced=True)


if __name__=='__main__': main()
