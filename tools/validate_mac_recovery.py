"""Offline real MAC + G-CVO + native-graph recovery with a bad-initial-guess fault.

One retained RGB-D view is repeated. No camera is opened. This proves wiring
and continuity, not live recovery recall under occlusion or changed viewpoints.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gcvo_backend import GCVOBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);d=args.source
    node=json.loads((d/'geometry_poses.json').read_text())['node_ids'][0]
    row=next(r for r in map(json.loads,(d/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted') and r['node']==node)
    frame=next(r for r in map(json.loads,(d/'frames.jsonl').read_text().splitlines()) if r['capture']==row['capture'])
    stamp=frame['stamp'];name=f'{stamp:.6f}.png'
    cloud=np.load(d/'map_frames'/f'{node}.npy');rgb=cv2.imread(str(d/'rgb'/name));depth=cv2.imread(str(d/'depth'/name),-1)
    c=json.loads((d/'sdk_calibration.json').read_text());k=np.array([[c['fxrgb'],0,c['cxrgb']],[0,c['fyrgb'],c['cyrgb']],[0,0,1.]])
    backend=GCVOBackend(args.output);used=False;entered=False;history=[];calls=[];original=backend.client.frame
    def counted(*args,**kwargs):calls.append(np.array(kwargs['pose']).copy());return original(*args,**kwargs)
    backend.client.frame=counted
    try:
        backend.process(rgb,depth,k,stamp,cloud)
        # Deliberate test-only fault: put local GN far outside its capture basin.
        backend.matcher.guess[:3,3]=[3,3,3]
        began=time.monotonic()
        while time.monotonic()-began<20:
            backend.process(rgb,depth,k,stamp+time.monotonic()-began+.1,cloud)
            state=backend.report['tracking']['state'];entered|=state=='RECOVERY'
            used|=backend.report.get('mac',{}).get('used_seed',False)
            history.append(dict(elapsed_s=time.monotonic()-began,matching_ms=backend.report.get('matching_ms'),state=state,mac=backend.report.get('mac'),status=backend.report['reason']))
            if entered and state=='NORMAL':break
            time.sleep(.05)
        checks=dict(entered_recovery=entered,used_real_mac_seed=used,returned_normal=entered and backend.tracking.state=='NORMAL',
                    only_two_submissions=len(calls)==2,world_continuous=len(calls)==2 and bool(np.allclose(calls[0],calls[1],atol=1e-4)),
                    tracker_reinitialized=bool(np.allclose(backend.matcher.guess,np.eye(4),atol=1e-4)))
        report=dict(scope=__doc__,checks=checks,passed=all(checks.values()),history=history,wall_s=time.monotonic()-began,
                    submitted_poses=[p.tolist() for p in calls])
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'],checks=checks,wall_s=report['wall_s']),indent=2))
        if not report['passed']:raise RuntimeError('Real MAC recovery validation failed')
    finally:backend.close()


if __name__=='__main__':main()
