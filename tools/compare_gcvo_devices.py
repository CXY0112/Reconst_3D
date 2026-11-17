"""On a CUDA host, compare CPU and official CUDA with identical prepared inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gcvo_matcher import GCVOSolver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True,help='known_pair.npz from validate_gcvo.py')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.packet,allow_pickle=False) as p:
        fixed=dict(points=p['fixed'],covariances=p['fixed_cov'],features=p['features'])
        moving=dict(points=p['moving'],covariances=p['moving_cov'],features=p['features'])
        expected=p['expected']
    outputs={}
    for device in ('cpu','cuda'):
        solver=GCVOSolver(device)
        result=solver.align(fixed,moving)
        outputs[device]={**result,'pose':result['pose'].tolist(),
            'expected_translation_error_mm':float(np.linalg.norm(result['pose'][:3,3]-expected[:3,3])*1000),
            'expected_rotation_error_deg':float(Rotation.from_matrix(result['pose'][:3,:3]@expected[:3,:3].T).magnitude()*180/np.pi)}
    a=np.array(outputs['cpu']['pose']);b=np.array(outputs['cuda']['pose'])
    report=dict(packet=str(args.packet.resolve()),results=outputs,
        device_translation_difference_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
        device_rotation_difference_deg=float(Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).magnitude()*180/np.pi),
        scope='One controlled pair; float64 CPU versus float32 CUDA, not proof of all-input equivalence')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
