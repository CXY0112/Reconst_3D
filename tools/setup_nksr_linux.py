"""Install an isolated official NKSR build environment on Linux x86-64 (including WSL)."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'vendor/nksr_linux'
BASE = Path(os.environ.get('RECONST_NKSR_HOME', '/opt/reconst_3d/nksr'))


def fix_cpu_compatibility():
    """Fix CPU device/grid initialization and CPU loading of CUDA-tagged weights."""
    records = []
    patches = [
        ('utils.py', '    device_index = device.index if device.index is not None else torch.cuda.current_device()',
         '    device_index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None',
         'Do not initialize CUDA when selecting CPU'),
        ('configs.py', 'return torch.hub.load_state_dict_from_url(url)',
         'return torch.hub.load_state_dict_from_url(url, map_location="cpu")', 'Load CUDA-tagged official weights onto CPU before network assignment'),
        ('svh.py', 'self._cuda_grid = _CudaIndexGrid(voxel_size, voxel_origin, device_idx)',
         'self._cuda_grid = _CudaIndexGrid(voxel_size, voxel_origin, device_idx) if self._device.type == "cuda" else None',
         'Do not construct a GPU grid placeholder for CPU-only execution')]
    for filename, old, new, reason in patches:
        for path in [BASE/'NKSR/package/nksr'/filename, *list((BASE/'env/lib').glob('python*/site-packages/nksr/'+filename))]:
            original = path.read_text()
            if new in original: continue
            if old not in original: raise RuntimeError('Review changed upstream CPU support before applying fix')
            backup = path.with_suffix('.py.cpu_upstream')
            if not backup.exists(): backup.write_text(original)
            path.write_text(original.replace(old, new, 1))
            records.append(dict(path=str(path), reason=reason,
                before_sha256=hashlib.sha256(original.encode()).hexdigest(), after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    header = BASE/'NKSR/package/csrc/vdbops/SparseFeatureIndexGrid.h'
    old = '        if (deviceIndex < 0) {\n            mDeviceIndex = c10::cuda::current_device();'
    new = ('        if (deviceIndex < 0 && std::is_same<BufferType, nanovdb::HostBuffer>::value) {\n'
           '            mDeviceIndex = -1;\n'
           '        } else if (deviceIndex < 0) {\n            mDeviceIndex = c10::cuda::current_device();')
    original = header.read_text()
    if new not in original:
        if old not in original: raise RuntimeError('Review changed upstream grid constructor before CPU fix')
        header.with_suffix('.h.cpu_upstream').write_text(original)
        changed = original.replace(old, new, 1).replace('#include <vector>', '#include <vector>\n#include <type_traits>', 1)
        header.write_text(changed)
        records.append(dict(path=str(header), reason='CPU host grids must not query a CUDA device',
            before_sha256=hashlib.sha256(original.encode()).hexdigest(), after_sha256=hashlib.sha256(header.read_bytes()).hexdigest()))
    if records:
        manifest = CONTROL/'cpu_compat_patches.json'
        previous = json.loads(manifest.read_text()) if manifest.exists() else []
        manifest.write_text(json.dumps(previous+records, indent=2))


def fetch_sources():
    """Use upstream HTTPS archives when Git's smart HTTP endpoint is unavailable."""
    source = BASE/'NKSR'
    if (source/'archive_sources.json').is_file(): return
    # Stop only the incomplete clone launched for this exact workspace.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
            if argv and Path(argv[0]).name == 'git' and 'clone' in argv and str(source) in argv:
                os.kill(int(proc.name), signal.SIGTERM)
        except (OSError, UnicodeError): pass
    if source.exists() and not (source/'package/setup.py').is_file():
        if source.resolve().parent != BASE.resolve(): raise ValueError('Unsafe partial clone path')
        shutil.rmtree(source)
    with urllib.request.urlopen('https://api.github.com/repos/nv-tlabs/NKSR/commits/public', timeout=60) as response:
        commit = json.load(response)['sha']
    provenance = []
    def unpack(name, url, folder):
        print('Fetching upstream archive: '+name, flush=True)
        archive = BASE/(name+'.tar.gz')
        with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as bundle:
            for member in bundle:
                relative = Path(*Path(member.name).parts[1:])
                target = (folder/relative).resolve()
                if not target.is_relative_to(folder.resolve()): raise ValueError('Unsafe archive path')
                if member.isdir(): target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as stream, target.open('wb') as output: shutil.copyfileobj(stream, output)
                elif member.issym() or member.islnk(): raise ValueError('Unexpected source archive link: '+member.name)
        item = {'name': name, 'url': url, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
        provenance.append(item)
        return item
    unpack('NKSR', 'https://codeload.github.com/nv-tlabs/NKSR/tar.gz/'+commit, source)
    for name, url in [('openvdb', 'https://codeload.github.com/AcademySoftwareFoundation/openvdb/tar.gz/7edd8cd86f105a01a41ad7b2bf59a81034cd79fb'),
                      ('eigen', 'https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.tar.gz')]:
        folder = source/'package/external'/name
        item = unpack(name, url, folder)
        # Valid local Git metadata lets older installer versions skip fetching.
        subprocess.run(['git', 'init', '--quiet', str(folder)], check=True)
        (folder/'.nksr_archive.json').write_text(json.dumps(item, indent=2))
    setup = source/'package/setup.py'
    original = setup.read_text()
    (source/'package/setup.py.upstream').write_text(original)
    anchor = '    repo_path = os.path.join(external_path, name)\n'
    if anchor not in original: raise RuntimeError('Upstream build file changed; review archive adaptation')
    setup.write_text(original.replace(anchor, anchor+
        '    # Dependencies provided as recorded official source archives.\n'
        '    if os.path.isfile(os.path.join(repo_path, ".nksr_archive.json")):\n'
        '        return\n', 1))
    (source/'archive_sources.json').write_text(json.dumps({'nksr_commit': commit, 'archives': provenance,
        'build_patch': 'Skip automatic Git checkout only for dependencies with recorded archive manifests; algorithm sources unchanged',
        'upstream_setup_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'patched_setup_sha256': hashlib.sha256(setup.read_bytes()).hexdigest()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('toolkit', 'verify', 'sources', 'nksr', 'build', 'smoke'), default='toolkit')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Run this installer under Linux x86-64 / WSL2')
    BASE.mkdir(parents=True, exist_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    if args.phase == 'sources': fetch_sources(); return
    env = os.environ.copy()
    env.update(MAMBA_ROOT_PREFIX=str(BASE/'mamba'), CUDA_HOME=str(BASE/'env'),
               TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST','8.6'), MAX_JOBS=os.environ.get('RECONST_NKSR_BUILD_JOBS', '4'), OMP_NUM_THREADS='4',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONUNBUFFERED='1', TORCH_HOME=str(BASE/'torch_cache'))
    env['PATH'] = str(BASE/'env/bin')+os.pathsep+env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(BASE/'env/lib'), str(BASE/'env/targets/x86_64-linux/lib'), env.get('LD_LIBRARY_PATH', '')])
    # Conda's CUDA packages place headers/libraries under targets/. Both g++
    # (CPU translation units include CUDA headers) and the linker need them.
    env['CPATH'] = os.pathsep.join([str(BASE/'env/targets/x86_64-linux/include'), str(BASE/'env/include'), env.get('CPATH', '')])
    env['LIBRARY_PATH'] = os.pathsep.join([str(BASE/'env/targets/x86_64-linux/lib'), str(BASE/'env/lib'), env.get('LIBRARY_PATH', '')])
    def run(label, command, cwd=None):
        print(label, flush=True)
        with (CONTROL/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((CONTROL/(label+'.log')).read_text(errors='replace')[-7000:], flush=True)
            raise RuntimeError(f'{label} failed ({result.returncode}); see {BASE}')
    mamba = BASE/'micromamba'
    if not mamba.exists():
        print('Downloading isolated micromamba...', flush=True)
        archive = BASE/'micromamba.tar.bz2'
        with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest', timeout=120) as response:
            with archive.open('wb') as output:
                while chunk := response.read(1024*1024): output.write(chunk)
        with tarfile.open(archive) as bundle:
            with bundle.extractfile('bin/micromamba') as input_stream, mamba.open('wb') as output:
                output.write(input_stream.read())
        mamba.chmod(0o755)
        (BASE/'bootstrap.json').write_text(json.dumps({'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'source': 'https://micro.mamba.pm/api/micromamba/linux-64/latest'}, indent=2))
    python = BASE/'env/bin/python'
    if args.phase == 'toolkit':
        run('01_cuda_environment', [mamba, 'create', '-y', '--prefix', BASE/'env', '--override-channels',
            '-c', 'nvidia/label/cuda-12.8.0', '-c', 'conda-forge', 'python=3.10', 'pip',
            'cuda-nvcc=12.8', 'libcusparse-dev', 'libcublas-dev', 'libcusolver-dev', 'eigen', 'zlib', 'ninja'])
    if args.phase in ('toolkit', 'verify'):
        run('02_nvcc_version', [BASE/'env/bin/nvcc', '--version'])
        probe = BASE/'cuda_probe.cu'
        probe.write_text('#include <cuda_runtime.h>\n#include <cstdio>\n'
            '__global__ void probe_kernel(int* p){p[0]=7;}\n'
            'int main(){int n=0; cudaError_t e=cudaGetDeviceCount(&n); '
            'std::printf("CUDA runtime=%d devices=%d status=%d\\n", CUDART_VERSION,n,(int)e); '
            'if(e==cudaSuccess && n>0){int* p=nullptr; int v=0; '
            'if(cudaMalloc(&p,sizeof(int))!=cudaSuccess)return 2; probe_kernel<<<1,1>>>(p); '
            'e=cudaMemcpy(&v,p,sizeof(int),cudaMemcpyDeviceToHost); cudaFree(p); '
            'std::printf("kernel result=%d status=%d\\n",v,(int)e); return e!=cudaSuccess || v!=7;} return 0;}\n')
        run('03_cuda_compile', [BASE/'env/bin/nvcc', probe, '-o', BASE/'cuda_probe', '-arch=sm_86'])
        run('04_cuda_runtime_probe', [BASE/'cuda_probe'])
        print((CONTROL/'02_nvcc_version.log').read_text(), flush=True)
        print((CONTROL/'04_cuda_runtime_probe.log').read_text(), flush=True)
        run('05_environment_manifest', [mamba, 'list', '--prefix', BASE/'env', '--json'])
        (CONTROL/'toolkit_ready.json').write_text(json.dumps({'compiler_verified': True, 'prefix': str(BASE/'env'),
            'nvcc': (CONTROL/'02_nvcc_version.log').read_text(),
            'runtime_probe': (CONTROL/'04_cuda_runtime_probe.log').read_text(),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2))
        print('CUDA compiler and host linking verified. Device availability is recorded separately.', flush=True)
        return
    if not python.is_file(): raise RuntimeError('Run --phase toolkit first')
    if args.phase == 'smoke':
        fix_cpu_compatibility()
        header = BASE/'NKSR/package/csrc/vdbops/SparseFeatureIndexGrid.h'
        ready = json.loads((CONTROL/'nksr_ready.json').read_text())
        if ready.get('cpu_grid_patch_sha256') != hashlib.sha256(header.read_bytes()).hexdigest():
            raise RuntimeError('CPU grid patch requires --phase build before inference')
        checkpoint = CONTROL/'ks.pth'
        if checkpoint.is_file():
            metadata = json.loads((CONTROL/'checkpoint.json').read_text())
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('Checkpoint checksum does not match the verified download')
            cache = Path(env.get('TORCH_HOME', str(Path.home()/'.cache/torch')))/'hub/checkpoints/ks.pth'
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix('.partial')
            shutil.copyfile(checkpoint, temporary); temporary.replace(cache)
        smoke = BASE/'cpu_smoke_input'; packet = BASE/'cpu_smoke_mesh.npz'
        run('12_smoke_input', [python, '-c',
            'from pathlib import Path; import numpy as np; '
            f'p=Path({str(smoke)!r}); p.mkdir(exist_ok=True); '
            'n=4096; z=1-2*(np.arange(n)+.5)/n; a=np.arange(n)*np.pi*(3-np.sqrt(5)); '
            'v=np.column_stack((np.sqrt(1-z*z)*np.cos(a),np.sqrt(1-z*z)*np.sin(a),z)); '
            'np.save(p/"processed_world.npy",v*.1); np.savez(p/"surfel_attributes.npz",normals_world=v); '
            '(p/"report.json").write_text("{}")'])
        run('13_cpu_smoke', [python, ROOT/'tools/compare_geometry.py', '--infer', '--device', 'cpu',
                            '--source', smoke, '--packet', packet, '--voxel-mm', '10'])
        run('14_smoke_validation', [python, '-c',
            'import numpy as np,json; from pathlib import Path; '
            f'd=np.load({str(packet)!r}); v=d["vertices"]; f=d["triangles"]; '
            'assert v.ndim==2 and v.shape[1]==3 and len(f)>10 and np.isfinite(v).all(); '
            'e=np.abs(np.linalg.norm(v,axis=1)-.1); assert np.median(e)<.03; '
            'report=dict(device="cpu",input_points=4096,vertices=len(v),triangles=len(f),'
            'elapsed_seconds=float(d["seconds"]),radial_error_mm_p50_p95=np.percentile(e*1000,[50,95]).tolist()); '
            f'Path({str(CONTROL/"cpu_smoke.json")!r}).write_text(json.dumps(report,indent=2)); '
            f'p=Path({str(CONTROL/"nksr_ready.json")!r}); m=json.loads(p.read_text()); '
            'm["real_inference_verified"]=True; m["selected_for_comparison"]=True; '
            'p.write_text(json.dumps(m,indent=2)); print(json.dumps(report,indent=2))'])
        print((CONTROL/'14_smoke_validation.log').read_text(), flush=True)
        return
    if args.phase == 'nksr':
        run('06_pytorch', [python, '-m', 'pip', 'install', 'torch==2.7.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
        run('07_python_dependencies', [python, '-m', 'pip', 'install', 'numpy<2', 'scipy', 'open3d',
            'GitPython', 'setuptools<81', 'wheel', 'pykdtree', 'omegaconf', 'python-pycg>=1.0.1'])
        run('07_torch_scatter', [python, '-m', 'pip', 'install', 'torch-scatter', '--only-binary=:all:',
            '-f', 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'])
    source = BASE/'NKSR'
    fetch_sources()
    fix_cpu_compatibility()
    run('10_nksr_build', [python, '-m', 'pip', 'install', '--no-build-isolation', str(source/'package')])
    run('11_nksr_probe', [python, ROOT/'tools/compare_geometry.py', '--probe'])
    (CONTROL/'nksr_ready.json').write_text(json.dumps({'extension_imported': True, 'prefix': str(BASE/'env'),
        'distribution': os.environ.get('WSL_DISTRO_NAME', 'Ubuntu-22.04'), 'python': str(python),
        'cpu_grid_patch_sha256': hashlib.sha256((BASE/'NKSR/package/csrc/vdbops/SparseFeatureIndexGrid.h').read_bytes()).hexdigest(),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'real_inference_verified': False}, indent=2))
    print('NKSR extension imported; next run a real CPU reconstruction before claiming inference support.', flush=True)


if __name__ == '__main__': main()
