"""Run real FPFH/CPU-MAC workers on controlled transforms of a saved cloud."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mac_recovery import MACRecovery
from gcvo_matcher import settings_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    fixed=np.load(args.cloud);settings=settings_for('cpu');results=[]
    for i,(rotation,translation) in enumerate([([.15,-.2,.25],[.18,-.08,.05]),([-.2,.12,-.15],[-.12,.06,.03]),([0,0,0],[0,0,0])]):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(rotation).as_matrix();pose[:3,3]=translation
        moving=(fixed-pose[:3,3])@pose[:3,:3]
        folder=args.output/str(i);folder.mkdir();m=MACRecovery(folder,settings);seed=None;began=time.monotonic()
        try:
            while time.monotonic()-began<settings['mac']['timeout_s']+2:
                seed=m.poll(1,moving,fixed,float(i))
                if seed is not None or m.status=='failed':break
                time.sleep(.05)
            row=dict(case=i,snapshot=m.snapshot(),passed=False,wall_ms=(time.monotonic()-began)*1000)
            if seed is not None:
                te=float(np.linalg.norm(seed[:3,3]-pose[:3,3])*1000)
                re=float(Rotation.from_matrix(seed[:3,:3]@pose[:3,:3].T).magnitude()*180/np.pi)
                row.update(translation_error_mm=te,rotation_error_deg=re,passed=te<30 and re<3)
            results.append(row);print(json.dumps(row),flush=True)
        finally:m.close()
    report=dict(source=str(args.cloud.resolve()),settings=settings,tests=results,passed=all(r['passed'] for r in results),
                caveat='Controlled transforms of one retained real cloud, not live recovery recall or real-scene ground-truth accuracy.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise RuntimeError('MAC validation failed; see report')


if __name__=='__main__':main()
