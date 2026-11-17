"""Probe whether the official CUDA G-CVO solver can run on this workstation.

This performs no registration and does not substitute a different algorithm.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def cuda_probe(runtime):
    result=dict(runtime_path=str(runtime.resolve()),runtime_exists=runtime.is_file(),device_count=None)
    if not runtime.is_file():return result
    cookie=os.add_dll_directory(str(runtime.parent.resolve()))
    try:
        lib=ct.WinDLL(str(runtime.resolve()))
        lib.cudaGetErrorString.argtypes=[ct.c_int];lib.cudaGetErrorString.restype=ct.c_char_p
        for symbol in ('cudaRuntimeGetVersion','cudaDriverGetVersion','cudaGetDeviceCount'):
            func=getattr(lib,symbol);func.argtypes=[ct.POINTER(ct.c_int)];func.restype=ct.c_int
            value=ct.c_int(-1);code=func(ct.byref(value))
            result[symbol]=dict(code=code,value=value.value,error=lib.cudaGetErrorString(code).decode('utf-8'))
        probe=result['cudaGetDeviceCount']
        if probe['code']==0:result['device_count']=probe['value']
        result['usable_cuda_device']=probe['code']==0 and probe['value']>0
    except OSError as exc:result.update(load_error=str(exc),usable_cuda_device=False)
    finally:cookie.close()
    return result


def source_audit(source):
    cmake=source/'CMakeLists.txt'
    solver=source/'gcvo/include/gcvo/impl/GCvoGPU_impl.cuh'
    covariance=source/'gcvo/include/gcvo/impl/GCvoPointCloud_covariance_impl.cuh'
    files=(cmake,source/'gcvo/include/gcvo/GCvoGPU.hpp',solver,covariance)
    result=dict(path=str(source.resolve()),commit=subprocess.check_output(
        ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),sha256={},evidence=[])
    for path in files:
        result['sha256'][str(path.relative_to(source))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for index,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if (path==cmake and any(term in line for term in ('LANGUAGES CXX CUDA','find_package(CUDAToolkit','CUDA::cudart','gcvo_add_point_type(gcvo_s'))
                    or path==solver and ('cudaMalloc((void**)&params_gpu_' in line or 'thrust::reduce(thrust::device, state.H_rows' in line)):
                result['evidence'].append(dict(file=str(path.relative_to(source)),line=index,text=line.strip()))
    # These are observed source/backend facts, not a benchmark of a hypothetical port.
    result['reviewed_solver_backend']='CUDA device allocation and CUDA/Thrust optimization kernels'
    result['cpu_or_amd_solver_backend_found']=False
    result['note']='Public CUDA-free C++ headers still link the CUDA solver; no official CPU/HIP/OpenCL solver target found in this revision.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'vendor/gcvo/source')
    parser.add_argument('--runtime',type=Path,default=ROOT/'vendor/nksr_windows/env/Library/bin/cudart64_12.dll')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    hardware_command=['powershell.exe','-NoProfile','-Command',
        'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress']
    hardware=subprocess.run(hardware_command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:adapters=json.loads(hardware.stdout)
    except ValueError:adapters=dict(error=hardware.stderr.strip(),returncode=hardware.returncode)
    cuda=cuda_probe(args.runtime)
    report=dict(platform=platform.platform(),python=sys.executable,display_adapters=adapters,
        cuda=cuda,source=source_audit(args.source),registration_executed=False,
        accuracy_measured=False,latency_measured=False,production_frontend_changed=False,
        status='cuda_runtime_available' if cuda.get('usable_cuda_device') else 'official_solver_unavailable_on_this_machine',
        conclusion='Official G-CVO requires a usable NVIDIA CUDA device; toolkit installation alone is insufficient. No CPU/AMD performance or accuracy claim is made.')
    (args.output/'environment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
