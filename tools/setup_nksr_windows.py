"""Native Windows NKSR installation trial; isolated from production runtimes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'vendor/nksr_windows'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('probe', 'environment', 'install', 'sources', 'compiler', 'build'), required=True)
    args = parser.parse_args()
    if os.name != 'nt': raise RuntimeError('This trial requires native Windows')
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE/'temp').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', MAMBA_ROOT_PREFIX=str(BASE/'mamba'),
               CUDA_HOME=str(BASE/'env/Library'), CUDA_PATH=str(BASE/'env/Library'), MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='8.6',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               TEMP=str(BASE/'temp'), TMP=str(BASE/'temp'))
    env['PATH'] = os.pathsep.join([str(BASE/'env'), str(BASE/'env/Library/bin'), str(BASE/'env/Scripts'), env.get('PATH', '')])
    python = BASE/'env/python.exe'
    def run(label, command, cwd=None, check=True, custom_env=None):
        print(label, flush=True)
        log_path = BASE/(label+'.log')
        command = list(map(str, command))
        selected_env = custom_env or env
        command[0] = shutil.which(command[0], path=selected_env.get('PATH')) or command[0]
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=cwd, env=selected_env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        print(f'{label}: exit {result.returncode}', flush=True)
        if result.returncode:
            print(log_path.read_text(encoding='utf-8', errors='replace')[-5000:], flush=True)
            if check: raise RuntimeError(f'{label} failed; see {log_path}')
        return result.returncode
    def unpack(archive, folder):
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as bundle:
            for item in bundle:
                target = (folder/Path(*Path(item.name).parts[1:])).resolve()
                if not target.is_relative_to(folder.resolve()): raise ValueError('Unsafe archive member')
                if item.isdir(): target.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(item) as stream, target.open('wb') as output: shutil.copyfileobj(stream, output)
                elif item.issym() or item.islnk(): raise ValueError('Unexpected archive link')
    source = BASE/'source/NKSR'
    if not (source/'package/setup.py').is_file():
        archive = ROOT/'vendor/nksr_linux/NKSR.tar.gz'
        if not archive.is_file():
            archive = BASE/'NKSR.tar.gz'
            urllib.request.urlretrieve('https://codeload.github.com/nv-tlabs/NKSR/tar.gz/refs/heads/public', archive)
        unpack(archive, source)
        (BASE/'source_archive.json').write_text(json.dumps({'archive': str(archive),
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))
    if args.phase == 'probe':
        host_env = os.environ.copy(); host_env.update(TEMP=str(BASE/'temp'), TMP=str(BASE/'temp'), PYTHONIOENCODING='utf-8')
        run('00_host_environment', [sys.executable, '-c', 'import sys,torch; from torch.utils.cpp_extension import CUDA_HOME; '
            'print(sys.version); print("torch",torch.__version__); print("device",torch.cuda.is_available()); print("CUDA_HOME",CUDA_HOME)'],
            custom_env=host_env)
        code = run('01_official_native_install', [sys.executable, '-m', 'pip', 'install', '--dry-run', '--no-deps',
            '--no-build-isolation', str(source/'package')], custom_env=host_env, check=False)
        print('Official unmodified native source install attempt:', code, flush=True)
        return
    if args.phase == 'sources':
        for name, url, ref in [('openvdb', 'https://github.com/AcademySoftwareFoundation/openvdb.git', '7edd8cd86f105a01a41ad7b2bf59a81034cd79fb'),
                               ('eigen', 'https://gitlab.com/libeigen/eigen.git', '3.4')]:
            folder = source/'package/external'/name
            if not (folder/'.git').is_dir():
                folder.mkdir(parents=True, exist_ok=True)
                run('source_'+name+'_init', ['git', 'init', folder])
                run('source_'+name+'_remote', ['git', 'remote', 'add', 'origin', url], cwd=folder)
            run('source_'+name+'_fetch', ['git', '-c', 'http.version=HTTP/1.1', '-c', 'http.lowSpeedLimit=100',
                '-c', 'http.lowSpeedTime=30', 'fetch', '--depth', '1', 'origin', ref], cwd=folder)
            run('source_'+name+'_checkout', ['git', 'checkout', '-B', ref, 'FETCH_HEAD'] if ref == '3.4' else ['git', 'checkout', 'FETCH_HEAD'], cwd=folder)
        return
    if args.phase == 'environment':
        mamba = BASE/'micromamba.exe'
        if not mamba.is_file():
            print('Downloading native Windows micromamba...', flush=True)
            archive = BASE/'micromamba.tar.bz2'
            urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/win-64/latest', archive)
            with tarfile.open(archive) as bundle:
                item = next(x for x in bundle if x.name.endswith('/micromamba.exe'))
                with bundle.extractfile(item) as stream, mamba.open('wb') as output: shutil.copyfileobj(stream, output)
        run('02_windows_cuda_environment', [mamba, 'create', '-y', '--prefix', BASE/'env', '--override-channels',
            '-c', 'nvidia/label/cuda-12.8.0', '-c', 'conda-forge', 'python=3.10', 'pip', 'ninja',
            'cuda-nvcc=12.8', 'libcusparse-dev', 'libcublas-dev', 'libcusolver-dev', 'eigen=3.4', 'zlib'])
        run('03_nvcc_version', [BASE/'env/Library/bin/nvcc.exe', '--version'])
        return
    if not python.is_file(): raise RuntimeError('Run --phase environment first')
    if args.phase == 'install':
        run('04_windows_pytorch', [python, '-m', 'pip', 'install', 'torch==2.7.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
        run('05_windows_dependencies', [python, '-m', 'pip', 'install', 'numpy<2', 'scipy', 'open3d', 'GitPython',
            'setuptools<81', 'wheel', 'pykdtree', 'omegaconf', 'python-pycg>=1.0.1'])
        run('06_windows_torch_scatter', [python, '-m', 'pip', 'install', 'torch-scatter', '--only-binary=:all:',
            '-f', 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'])
        return
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not vs: raise RuntimeError('MSVC build tools not found')
    batch = BASE/'load_msvc.cmd'
    batch.write_text(f'@call "{Path(vs)/"Common7/Tools/VsDevCmd.bat"}" -arch=amd64 -host_arch=amd64 >nul\n@if errorlevel 1 exit /b 1\n@set\n', encoding='utf-8')
    data = subprocess.check_output(['cmd.exe', '/d', '/c', str(batch)], env=env, text=True, errors='replace')
    for line in data.splitlines():
        key, separator, value = line.partition('=')
        if separator and key: env[key.upper()] = value
    env['DISTUTILS_USE_SDK'] = '1'
    env['LIB'] = str(BASE/'env/Library/lib')+os.pathsep+env.get('LIB', '')
    if args.phase == 'compiler':
        run('compiler_msvc', ['cl.exe', '/Bv'], check=False)
        probe = BASE/'native_cuda_probe.cu'
        probe.write_text('#include <cuda_runtime.h>\n__global__ void probe(int* p){p[0]=7;}\nint main(){return 0;}\n')
        run('compiler_cuda_probe', [BASE/'env/Library/bin/nvcc.exe', probe, '-o', BASE/'native_cuda_probe.exe', '-arch=sm_86'])
        return
    run('07_native_source_build', [python, '-m', 'pip', 'install', '--no-build-isolation', str(source/'package')])
    run('08_native_import', [python, ROOT/'tools/compare_geometry.py', '--probe'])
    (BASE/'ready.json').write_text(json.dumps({'python': str(python), 'extension_imported': True,
                                              'real_inference_verified': False}, indent=2))


if __name__ == '__main__': main()
