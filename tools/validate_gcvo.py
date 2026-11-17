"""Recorded-cloud and known-transform tests of either G-CVO backend.

No camera, no ground-truth claims on real adjacent frames. The same prepared
inputs/settings feed both backends when rerun on a CUDA workstation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gcvo_matcher import GCVOSolver,settings_for,prepare


def metrics(fixed,moving,pose):
    warped=moving@pose[:3,:3].T+pose[:3,3]
    distances=np.r_[cKDTree(fixed).query(warped,workers=2)[0],cKDTree(warped).query(fixed,workers=2)[0]]
    near=distances[distances<.03]
    return dict(coverage_10mm=float(np.mean(distances<.01)),coverage_30mm=float(np.mean(distances<.03)),
        truncated_rmse_mm=float(np.sqrt(np.mean(near**2))*1000) if len(near) else None,
        median_mm=float(np.median(distances)*1000))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='A saved baseline setp1 with raw frame centroids and optimized poses')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();graph=json.loads((source/'geometry_poses.json').read_text())
    accepted=[json.loads(line) for line in (source/'registrations.jsonl').read_text(encoding='utf-8').splitlines() if line]
    captures={r['node']:r['capture'] for r in accepted if r.get('accepted')}
    frames={r['capture']:r['stamp'] for r in map(json.loads,(source/'frames.jsonl').read_text().splitlines())}
    nodes=graph['node_ids'][:args.limit];poses=list(map(np.array,graph['poses'][:args.limit]))
    c=json.loads((source/'sdk_calibration.json').read_text());k=np.array([[c['fxrgb'],0,c['cxrgb']],[0,c['fyrgb'],c['cyrgb']],[0,0,1.]])
    settings=settings_for(args.device);solver=GCVOSolver(settings=settings);packs=[];clouds=[];prep=[]
    for node in nodes:
        cloud=np.load(source/'map_frames'/f'{node}.npy');clouds.append(cloud)
        color=cv2.imread(str(source/'rgb'/f'{frames[captures[node]]:.6f}.png'))
        depth=cv2.imread(str(source/'depth'/f'{frames[captures[node]]:.6f}.png'),-1)
        packs.append(prepare(cloud,settings,color,k,depth));prep.append(packs[-1]['prepare_ms'])
    report=dict(device=args.device,implementation=solver.backend,settings=settings,source=str(source),
        input_points=[len(p['points']) for p in packs],
        library_sha256=hashlib.sha256(Path(settings['library']).read_bytes()).hexdigest(),
        known_transforms=[],pairs=[],prepare_ms=prep,accuracy_on_real_data='No independent ground truth; consistency only',
        cuda_equivalence_verified=False)
    for rotation,translation in [([.025,-.035,.02],[.035,-.02,.015]),([-.03,.01,.04],[-.02,.01,.005]),([.01,.01,-.02],[.01,.005,-.01])]:
        expected=np.eye(4);expected[:3,:3]=Rotation.from_rotvec(rotation).as_matrix();expected[:3,3]=translation
        moving=dict(packs[0]);moving['points']=(moving['points']-expected[:3,3])@expected[:3,:3]
        moving['covariances']=expected[:3,:3].T@moving['covariances']@expected[:3,:3]
        result=solver.align(packs[0],moving)
        te=float(np.linalg.norm(result['pose'][:3,3]-expected[:3,3])*1000)
        re=float(Rotation.from_matrix(result['pose'][:3,:3]@expected[:3,:3].T).magnitude()*180/np.pi)
        report['known_transforms'].append(dict(expected=expected.tolist(),estimated=result['pose'].tolist(),translation_error_mm=te,
            rotation_error_deg=re,solver_ms=float(result['solver_ms']),passed=result['solver_status']==0 and te<1 and re<.1))
    # Pack a controlled pair for CPU/CUDA differential tests on a future host.
    np.savez_compressed(args.output/'known_pair.npz',fixed=packs[0]['points'],moving=moving['points'],
        fixed_cov=packs[0]['covariances'],moving_cov=moving['covariances'],features=packs[0]['features'],expected=expected)
    for j in range(1,len(nodes)):
        started=time.perf_counter()
        try:
            result=solver.align(packs[j-1],packs[j])
            cpu_metrics=metrics(clouds[j-1],clouds[j],result['pose'])
            baseline=metrics(clouds[j-1],clouds[j],np.linalg.inv(poses[j-1])@poses[j])
            row=dict(pair=[nodes[j-1],nodes[j]],pose=result['pose'].tolist(),
                solver={key:float(value) if isinstance(value,np.floating) else value for key,value in result.items() if key!='pose'},
                gcvo=cpu_metrics,rgbd=baseline)
        except ValueError as exc:row=dict(pair=[nodes[j-1],nodes[j]],error=str(exc))
        row['test_wall_ms']=(time.perf_counter()-started)*1000;report['pairs'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    valid=[r for r in report['pairs'] if 'solver' in r and r['solver']['solver_status']==0]
    report['summary']=dict(converged_pairs=len(valid),total_pairs=len(report['pairs']),
        solver_ms_p50_p95=np.percentile([r['solver']['solver_ms'] for r in valid],[50,95]).tolist() if valid else None,
        prep_ms_p50_p95=np.percentile(prep,[50,95]).tolist(),known_passed=all(r['passed'] for r in report['known_transforms']))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
