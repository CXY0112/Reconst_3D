"""Build CPU and optionally the unchanged official CUDA G-CVO behind one ABI.

Requires CMake/C++/Eigen; CUDA additionally needs NVCC, PCL, yaml-cpp and the
official checkout. Does not install drivers or dependencies.
"""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--eigen',type=Path,help='Directory containing Eigen/Core')
    parser.add_argument('--prefix',action='append',default=[],help='CMake package prefix (repeatable)')
    parser.add_argument('--cuda-architectures',help='For example native, or an architecture supported by your NVCC')
    parser.add_argument('--build',type=Path)
    args=parser.parse_args();build=args.build or ROOT/'vendor/gcvo'/('build_'+args.device)
    command=['cmake','-S',str(ROOT/'gcvo_native'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DRECONST_GCVO_CUDA='+('ON' if args.device=='cuda' else 'OFF')]
    eigen=args.eigen or ROOT/'vendor/nksr_windows/env/Library/include/eigen3'
    if eigen.is_dir():command.append('-DEIGEN3_INCLUDE_DIR='+str(eigen.resolve()))
    if args.prefix:command.append('-DCMAKE_PREFIX_PATH='+';'.join(args.prefix))
    if args.cuda_architectures:command.append('-DCMAKE_CUDA_ARCHITECTURES='+args.cuda_architectures)
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(build),'--config','Release','--parallel','4'],check=True)
    # Install only our ABI libraries. Do not change/install upstream globally.
    names=['gcvo_cpu']+(['gcvo_cuda'] if args.device=='cuda' else [])
    import os,shutil
    out=ROOT/'vendor/gcvo/bin';out.mkdir(parents=True,exist_ok=True)
    for name in names:
        filename=name+'.dll' if os.name=='nt' else 'lib'+name+'.so'
        candidates=[build/'Release'/filename,build/filename]
        file=next((p for p in candidates if p.is_file()),None)
        if file is None:raise FileNotFoundError(filename)
        shutil.copy2(file,out/filename)
    print('Built: '+', '.join(names))


if __name__=='__main__':main()
