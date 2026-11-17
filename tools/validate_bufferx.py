"""Real pretrained inference on a recorded cloud with a known rigid transform.

This measures a controlled pair, not sequence drift or camera accuracy.
"""
from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='pointcloud_local NPZ')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from bufferx_client import BufferXClient
    from bufferx_backend import validate_registration
    from scipy.spatial.transform import Rotation
    with np.load(args.source,allow_pickle=False) as packet:target=packet['points'].copy()
    expected=np.eye(4);expected[:3,:3]=Rotation.from_rotvec([.025,-.035,.02]).as_matrix();expected[:3,3]=[.035,-.02,.015]
    source=(target-expected[:3,3])@expected[:3,:3]
    client=BufferXClient(args.output)
    try:
        result=client.estimate(source,target);pose,metrics=validate_registration(source,target,result,client.settings)
        rotation_error=float(Rotation.from_matrix(pose[:3,:3]@expected[:3,:3].T).magnitude()*180/np.pi)
        translation_error=float(np.linalg.norm(pose[:3,3]-expected[:3,3]))
        report=dict(test='recorded local cloud under known SE3; identical support, no real temporal change',source=str(args.source.resolve()),
            expected=expected.tolist(),result=result,validation=metrics,translation_error_m=translation_error,
            rotation_error_deg=rotation_error,passed=translation_error<.03 and rotation_error<3.)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'],translation_error_m=translation_error,rotation_error_deg=rotation_error,matching_ms=result['matching_ms'])))
        if not report['passed']:raise SystemExit(1)
    finally:client.close()


if __name__=='__main__':main()
