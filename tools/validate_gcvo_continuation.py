"""Compare capped CPU solves and bounded continuation on saved keyframe pairs.

These are retained frames, not the unsaved rejected live inputs or ground truth.
Identity initialization isolates the continuation effect from tracker history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gcvo_matcher import GCVOSolver,GCVOConvergenceError,align_tracking,settings_for,prepare
from bufferx_backend import validate_registration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=45)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    d=args.source.resolve()
    graph=json.loads((d/'geometry_poses.json').read_text())
    rows=[json.loads(line) for line in (d/'registrations.jsonl').read_text(encoding='utf-8').splitlines()]
    captures={r['node']:r['capture'] for r in rows if r.get('accepted')}
    stamps={r['capture']:r['stamp'] for r in map(json.loads,(d/'frames.jsonl').read_text().splitlines())}
    calibration=json.loads((d/'sdk_calibration.json').read_text())
    k=np.array([[calibration['fxrgb'],0,calibration['cxrgb']],[0,calibration['fyrgb'],calibration['cyrgb']],[0,0,1.]])
    settings=settings_for('cpu');solver=GCVOSolver(settings=settings)
    report=dict(source=str(d),settings=settings,library_sha256=hashlib.sha256(Path(settings['library']).read_bytes()).hexdigest(),
                limitations=__doc__,pairs=[])
    previous=None
    for node in graph['node_ids'][:args.limit]:
        stamp=stamps[captures[node]]
        cloud=np.load(d/'map_frames'/f'{node}.npy')
        rgb=cv2.imread(str(d/'rgb'/f'{stamp:.6f}.png'))
        depth=cv2.imread(str(d/'depth'/f'{stamp:.6f}.png'),-1)
        current=prepare(cloud,settings,rgb,k,depth)
        if previous is not None:
            old_node,old_cloud,old_pack=previous
            row=dict(pair=[old_node,node])
            try:
                result=align_tracking(solver,old_pack,current,np.eye(4))
                row.update({key:value for key,value in result.items() if key!='pose'})
                row['converged']=True
                moved=current['points']@result['pose'][:3,:3].T+result['pose'][:3,3]
                from scipy.spatial import cKDTree
                result['inliers']=int(np.sum(cKDTree(old_pack['points']).query(moved)[0]<settings['validation_distance_m']))
                try:
                    _,row['geometry']=validate_registration(cloud,old_cloud,result,settings,'G-CVO CPU')
                    row['geometry_passed']=True
                except ValueError as exc:row.update(geometry_passed=False,geometry_error=str(exc))
            except GCVOConvergenceError as exc:row.update(exc.diagnostics,converged=False,error=str(exc))
            except ValueError as exc:row.update(converged=False,error=str(exc))
            report['pairs'].append(row)
        previous=(node,cloud,current)
    pairs=report['pairs'];capped=[r for r in pairs if r.get('attempts',[{}])[0].get('solver_status')==1]
    report['summary']=dict(pairs=len(pairs),initially_capped=len(capped),
        rescued_convergence=sum(r['converged'] for r in capped),
        rescued_and_geometry_passed=sum(r.get('geometry_passed',False) for r in capped),
        still_rejected=sum(not r['converged'] for r in pairs),
        solver_ms_p50_p95=np.percentile([r['solver_ms'] for r in pairs if 'solver_ms' in r],[50,95]).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':main()
