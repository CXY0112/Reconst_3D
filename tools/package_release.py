"""Build a clean ZIP: sources by default, Windows runtimes only when requested."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TOOLS={'colorize_geometry.py','reconstruct_geometry.py','compare_geometry.py',
       'setup_nksr_linux.py','setup_nksr_windows.py','online_bridge.cpp','CMakeLists.txt',
       'package_release.py','rtab_host.cpp','build_rtab_host.cmd',
       'mobilesam_worker.py','setup_mobilesam.py','scene_reconstruction_worker.py',
       'verify_installation.py','run_windows.cmd'}



def release_files(windows_runtime=False):
    def candidates():
        for directory, folders, names in os.walk(ROOT):
            rel=Path(directory).relative_to(ROOT)
            folders[:]=sorted(name for name in folders if name not in ('__pycache__','.pytest_cache','.git'))
            if not rel.parts:
                folders[:]=[name for name in folders if name not in ('results','tests')]
            elif rel.parts==('vendor',):
                folders[:]=[name for name in folders if name in (('camera_win64','rtabmap','geometry_packages','online_geometry','mobilesam') if windows_runtime else ('mobilesam',))]
            for name in sorted(names):yield Path(directory)/name
    for path in candidates():
        rel=path.relative_to(ROOT)
        if any(part in ('__pycache__','.pytest_cache','.git') for part in rel.parts):continue
        if rel.parts[0] in ('results','tests'):continue
        if rel.parts[0]=='tools' and rel.name not in TOOLS:continue
        if rel.parts[0]=='vendor':
            if not windows_runtime and rel.parts[1]!='mobilesam':continue
            if len(rel.parts)<3 or rel.parts[1] not in ('camera_win64','rtabmap','geometry_packages','online_geometry','mobilesam'):continue
            if rel.parts[1]=='online_geometry' and rel.suffix.lower() not in ('.dll','.exe','.json'):continue
        if rel.parts[0]=='vendor' and rel.parts[1]=='mobilesam':
            # Carry portable source and checked weights, never this PC's WSL marker.
            tail=rel.parts[2:]
            if not (len(tail)>2 and tail[0]=='source' and
                    (tail[2]=='mobile_sam' or tail[2:] == ('weights','mobile_sam.pt') or tail[2] in ('LICENSE','README.md'))):continue
        if rel.name=='performance_ui.py' or rel.suffix.lower()=='.pdf':continue
        if not windows_runtime and (rel.suffix.lower() in ('.dll','.exe','.pyd','.lib','.obj','.pdb','.cmd','.bat','.ps1') or rel.name=='setup_nksr_windows.py'):continue
        if rel.name.startswith('.env') or (rel.suffix.lower() in ('.env','.key','.pem') and not (rel.parts[0]=='vendor' and rel.name=='cacert.pem')) or rel.name in ('runtime.json','nksr_ready.json'):continue
        yield path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows-runtime',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if ROOT==output or ROOT in output.parents:raise ValueError('输出ZIP必须放在Reconst_3D目录之外')
    files=list(release_files(args.windows_runtime));manifest=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in files:
            rel=path.relative_to(ROOT).as_posix();data=path.read_bytes()
            archive.writestr('Reconst_3D/'+rel,data)
            manifest.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr('Reconst_3D/RELEASE_MANIFEST.json',json.dumps(dict(profile='windows-sdk' if args.windows_runtime else 'portable-source',files=manifest),indent=2))
    print(f'{output}: {len(files)} files')


if __name__=='__main__':main()
