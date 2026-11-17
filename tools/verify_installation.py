"""Check dependencies and bundled assets without connecting a camera or service."""
from pathlib import Path
import argparse, importlib, struct, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows-sdk',action='store_true');args=parser.parse_args()
    if struct.calcsize('P')!=8:raise RuntimeError('64-bit Python required')
    for name in ('tkinter','numpy','scipy','cv2','PIL','torch','torchvision','timm','trimesh'):
        importlib.import_module(name);print('OK',name,flush=True)
    from tools.mobilesam_worker import COMMIT,WEIGHT_SHA256
    import hashlib
    model=ROOT/'vendor/mobilesam/source'/('MobileSAM-'+COMMIT)
    if not model.exists():model=ROOT/'vendor/mobilesam/source/MobileSAM-master'
    if hashlib.sha256((model/'weights/mobile_sam.pt').read_bytes()).hexdigest()!=WEIGHT_SHA256:
        raise RuntimeError('MobileSAM weight checksum mismatch')
    sys.path.insert(0,str(model));importlib.import_module('mobile_sam');print('OK prompt segmentation')
    if args.windows_sdk:
        import os
        if os.name!='nt':raise RuntimeError('Windows runtime requires Windows x64')
        for item in ('vendor/camera_win64','vendor/online_geometry/rtab_host.exe','vendor/rtabmap/bin/rtabmap_core.dll'):
            if not (ROOT/item).exists():raise FileNotFoundError(item)
        from runtime_paths import geometry_python
        subprocess.run([str(geometry_python()),'-c',
            "import sys;sys.path.insert(0,sys.argv[1]);from runtime_paths import activate_geometry_packages;activate_geometry_packages();import open3d,numpy,scipy;print('OK bundled geometry')",str(ROOT)],check=True)
        print('OK Windows SDK assets present (camera not opened)')
    else:
        importlib.import_module('open3d');print('OK geometry dependencies; native capture/rendering still need target adaptation')

if __name__=='__main__':main()
