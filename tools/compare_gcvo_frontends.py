"""Compare actual A/B runs, matching retained frames by timestamp, never node ID."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def quantiles(values):
    return np.percentile(values,[50,95]).tolist() if values else None


def read_poses(folder):
    graph=json.loads((folder/'geometry_poses.json').read_text())
    frames={r['capture']:r['stamp'] for r in map(json.loads,(folder/'frames.jsonl').read_text().splitlines())}
    captures={r['node']:r['capture'] for r in map(json.loads,(folder/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted')}
    return {frames[captures[node]]:np.array(pose) for node,pose in zip(graph['node_ids'],graph['poses'])}


def compare(run,input_folder):
    output=dict(run=str(run.resolve()),input=str(input_folder.resolve()),results={},pairs=[],
        scope='Same RGB-D replay; final optimized poses evaluated on identical source SDK centroid clouds at common retained timestamps.',
        limitations=['No independent ground truth; residuals measure consistency, not true accuracy.',
                     'Retained-frame recordings omit rejected/live intermediate inputs and cannot establish recovery rate or live FPS.',
                     'Timing excludes initialization. backend_process_ms uses the same Python process() boundary for both backends; reported frontend timers have different scopes. Offline total includes dense I/O and waiting.',
                     'Latest baseline was retained by G-CVO; room2 was retained by the default frontend. Both have selection bias.'])
    poses={};input_names=None
    for name in ('rgbd','gcvo_cpu'):
        rows=[json.loads(line) for line in (run/f'{name}_frames.jsonl').read_text(encoding='utf-8').splitlines()]
        names=[r['frame'] for r in rows]
        if input_names is not None and names!=input_names:raise ValueError('Different input frame sequences; A/B comparison refused')
        input_names=names
        moving=rows[1:];passed=[r for r in moving if r['status']>0]
        output['results'][name]=dict(inputs=len(rows),tracking_trials=len(moving),tracked=len(passed),
            rejected=len(moving)-len(passed),tracking_pass_rate=len(passed)/len(moving) if moving else None,
            retained=sum(r['status']==2 for r in rows),
            frontend_ms_p50_p95=quantiles([r['report']['matching_ms'] for r in moving]),
            frontend_success_ms_p50_p95=quantiles([r['report']['matching_ms'] for r in passed]),
            backend_process_ms_p50_p95=quantiles([r['backend_process_ms'] for r in moving if 'backend_process_ms' in r]),
            backend_process_success_ms_p50_p95=quantiles([r['backend_process_ms'] for r in passed if 'backend_process_ms' in r]),
            offline_total_ms_p50_p95=quantiles([r['processing_ms'] for r in moving]),
            confidence_p50_p95=quantiles([r['report']['tracking']['confidence'] for r in passed if r['report'].get('tracking',{}).get('confidence') is not None]),
            recovery_frames=sum(r['report'].get('tracking',{}).get('state')=='RECOVERY' for r in rows),
            rejection_reasons=dict(Counter(r['report'].get('reason','unknown') for r in moving if r['status']==0)))
        poses[name]=read_poses(run/name/'setp1')
    source_graph=json.loads((input_folder/'geometry_poses.json').read_text())
    source_poses=read_poses(input_folder)
    # Source timestamps and graph nodes are paired in insertion order by read_poses.
    clouds={stamp:input_folder/'map_frames'/f'{node}.npy' for stamp,node in zip(source_poses,source_graph['node_ids'])}
    common=sorted(set(poses['rgbd'])&set(poses['gcvo_cpu'])&set(clouds))
    output['common_retained_frames']=len(common)
    for a,b in zip(common,common[1:]):
        fixed=np.load(clouds[a]);moving=np.load(clouds[b]);row=dict(stamps=[a,b],points=[len(fixed),len(moving)])
        for name in ('rgbd','gcvo_cpu'):
            pose=np.linalg.inv(poses[name][a])@poses[name][b]
            moved=moving@pose[:3,:3].T+pose[:3,3]
            distances=np.r_[cKDTree(fixed).query(moved,workers=2)[0],cKDTree(moved).query(fixed,workers=2)[0]]
            near=distances[distances<.03]
            row[name]=dict(coverage_10mm=float(np.mean(distances<.01)),coverage_30mm=float(np.mean(distances<.03)),
                           rmse_30mm_mm=float(np.sqrt(np.mean(near**2))*1000) if len(near) else None,
                           median_mm=float(np.median(distances)*1000))
        output['pairs'].append(row)
    if output['pairs']:
        output['geometry_summary']={name:dict(mean_pair_rmse_30mm_mm=float(np.mean([r[name]['rmse_30mm_mm'] for r in output['pairs']])),
            mean_pair_coverage_10mm=float(np.mean([r[name]['coverage_10mm'] for r in output['pairs']]))) for name in poses}
        output['gcvo_lower_rmse_pairs']=sum(r['gcvo_cpu']['rmse_30mm_mm']<r['rgbd']['rmse_30mm_mm'] for r in output['pairs'])
    else:output['geometry_unavailable']='Fewer than two common retained frames; no comparable relative pose.'
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();report=compare(args.run,args.input)
    (args.run/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','limitations')},ensure_ascii=False,indent=2))
