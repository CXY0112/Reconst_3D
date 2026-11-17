"""Install the CPU experiment in an isolated venv; leave the base environment intact."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVISION='07dd9be5a58c9c02213584535592766c9023cebc'
DEPENDENCIES=['einops==0.8.2','kornia==0.8.2','easydict==1.13','scikit-learn==1.7.2','nibabel==5.4.2','huggingface_hub==2.3.0']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python',type=Path,default=Path(sys.executable),help='Existing Python with PyTorch, Open3D, SciPy, Matplotlib')
    args=parser.parse_args();base=args.base_python.resolve()
    subprocess.run([str(base),'-c','import torch,open3d,scipy,matplotlib'],check=True)
    folder=ROOT/'vendor/bufferx';folder.mkdir(exist_ok=True)
    source=folder/'source';env=folder/'env';python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not source.exists():
        subprocess.run(['git','clone','https://github.com/MIT-SPARK/BUFFER-X.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',REVISION],check=True)
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() if (source/'.git').is_dir() else REVISION
    if revision!=REVISION:raise RuntimeError('BUFFER-X 源码版本不匹配；请使用独立目录，避免覆盖现有源码')
    patch=source/'models/patch_embedder.py';text=patch.read_text(encoding='utf-8')
    old='S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()'
    new='S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).to(delta_x.device)'
    if old in text:patch.write_text(text.replace(old,new),encoding='utf-8')
    elif new not in text:raise RuntimeError('Unknown patch embedder; refusing to patch')
    subprocess.run([str(base),'-m','venv','--without-pip','--system-site-packages',str(env)],check=True)
    subprocess.run([str(base),'-m','pip','--python',str(python),'install',*DEPENDENCIES],check=True)
    # Use a script with explicit paths rather than shell interpolation.
    download=folder/'download_weights.py'
    download.write_text('from huggingface_hub import hf_hub_download\n'
        'from pathlib import Path\n'
        'root=Path(__file__).resolve().parent/"source"\n'
        'for stage in ("Desc","Pose"):\n'
        '    hf_hub_download("Hyungtae-Lim/BUFFER-X",f"snapshot/threedmatch/{stage}/best.pth",local_dir=str(root))\n',encoding='utf-8')
    subprocess.run([str(python),str(download)],check=True)
    runtime=folder/'runtime.json'
    if not runtime.exists():runtime.write_text(json.dumps(dict(python=str(python),source=str(source),weights=str(source/'snapshot/threedmatch'),device='cpu',threads=4,timeout_seconds=600),indent=2),encoding='utf-8')
    files=[patch,*[source/f'snapshot/threedmatch/{stage}/best.pth' for stage in ('Desc','Pose')],ROOT/'bufferx_cpu_ops.py']
    provenance=dict(repository='https://github.com/MIT-SPARK/BUFFER-X',commit=revision,
        cpu_adaptation='Pointnet2 FPS/ball-query/gather/group; torch KNN/SVD; device-neutral grid; 24-patch eval chunks',
        official_parameters=dict(num_fps=1500,num_scales=3,num_points_per_patch=512,early_exit=False,pose_refine=True,pose_estimator='ransac'),
        base_python=str(base),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (folder/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('BUFFER-X CPU environment ready. Build tools/build_bufferx_bridge.cmd before scanning.')


if __name__=='__main__':main()
