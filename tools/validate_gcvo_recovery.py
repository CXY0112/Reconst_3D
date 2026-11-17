"""Offline real CPU/native-backend recovery check with controlled input loss.

Three empty inputs simulate missing geometry, followed by the exact trusted
view. This verifies state/commit behavior, not camera recovery performance.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gcvo_backend import GCVOBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();d=args.source.resolve();args.output.mkdir(parents=True,exist_ok=False)
    node=json.loads((d/'geometry_poses.json').read_text())['node_ids'][0]
    registration=next(r for r in map(json.loads,(d/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted') and r['node']==node)
    frame=next(r for r in map(json.loads,(d/'frames.jsonl').read_text().splitlines()) if r['capture']==registration['capture'])
    stamp=frame['stamp'];name=f'{stamp:.6f}.png'
    cloud=np.load(d/'map_frames'/f'{node}.npy');color=cv2.imread(str(d/'rgb'/name));depth=cv2.imread(str(d/'depth'/name),-1)
    c=json.loads((d/'sdk_calibration.json').read_text());k=np.array([[c['fxrgb'],0,c['cxrgb']],[0,c['fyrgb'],c['cyrgb']],[0,0,1.]])
    b=GCVOBackend(args.output);calls=[];rows=[];original=b.client.frame
    def counted(*positional,**keywords):
        calls.append(np.array(keywords['pose']).copy());return original(*positional,**keywords)
    b.client.frame=counted
    try:
        for i in range(7):
            geometry=np.empty((0,3)) if 1<=i<=3 else cloud
            status=b.process(color,depth,k,stamp+i,geometry)
            rows.append(dict(index=i,status=status,reference=b.reference_node,tracking=b.report['tracking']))
        checks=dict(initialized=rows[0]['status']==2,
            entered_recovery=rows[3]['tracking']['state']=='RECOVERY',
            probes_withheld=all(rows[i]['status']==0 for i in (4,5)),
            returned_normal=rows[6]['status']>0 and rows[6]['tracking']['state']=='NORMAL',
            only_trusted_submissions=len(calls)==2,
            world_continuity=len(calls)==2 and bool(np.allclose(calls[0],calls[1],atol=1e-5)))
        report=dict(scope=__doc__,source=str(d),checks=checks,passed=all(checks.values()),rows=rows,
                    submitted_poses=[p.tolist() for p in calls])
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'],checks=checks),indent=2))
        if not report['passed']:raise RuntimeError('Recovery validation failed; see report.json')
    finally:b.close()


if __name__=='__main__':main()
