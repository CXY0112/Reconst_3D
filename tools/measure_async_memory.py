"""Windows working-set probe for geometry worker process groups, not the full UI."""
import sys,os,json,tempfile,time,subprocess,ctypes as ct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['PYTHONPATH']=str(ROOT)
OUT=ROOT/'results/async_loop_v3';OUT.mkdir(exist_ok=True)

def rss(pid):
    class Counters(ct.Structure):
        _fields_=[('cb',ct.c_ulong),('faults',ct.c_ulong)]+[(n,ct.c_size_t) for n in ('peak','rss','pp','p','pn','n','pf','ppf')]
    kernel=ct.WinDLL('kernel32',use_last_error=True);psapi=ct.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[ct.c_ulong,ct.c_int,ct.c_ulong];kernel.OpenProcess.restype=ct.c_void_p
    kernel.CloseHandle.argtypes=[ct.c_void_p];kernel.CloseHandle.restype=ct.c_int
    psapi.GetProcessMemoryInfo.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_ulong];psapi.GetProcessMemoryInfo.restype=ct.c_int
    handle=kernel.OpenProcess(0x410,False,pid)
    if not handle:return None
    try:
        counters=Counters();counters.cb=ct.sizeof(counters)
        return counters.rss if psapi.GetProcessMemoryInfo(handle,ct.byref(counters),counters.cb) else None
    finally:kernel.CloseHandle(handle)

if len(sys.argv)==1:
    results={}
    for mode in ('synchronous','asynchronous'):
        run=subprocess.run([sys.executable,__file__,mode],capture_output=True,text=True,check=True)
        results[mode]=json.loads(run.stdout.strip().splitlines()[-1])
    (OUT/'worker_memory.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2));sys.exit(0)

import numpy as np
from geometry_client import GeometryClient
from snapshot_window import ir_to_rgb
from reconstruct import Calibration
mode=sys.argv[1];source=ROOT/'results/scan_20261015_102420_653218'
cal=Calibration(**json.loads((source/'sdk_calibration.json').read_text()))
points=ir_to_rgb(np.load(sorted((source/'pointcloud_raw').glob('*.npy'))[0])/1000,cal)
with tempfile.TemporaryDirectory(dir=OUT) as temp:
    client=GeometryClient(Path(temp),asynchronous=mode=='asynchronous')
    try:
        for _ in range(6):
            result=client.register(points);assert result['accepted'],result;client.request('commit')
        deadline=time.monotonic()+30
        while client.background_busy() and time.monotonic()<deadline:
            result=client.poll_loop()
            if result and result.get('accepted'):client.request('commit_loop')
            time.sleep(.025)
        pids=dict(host=os.getpid(),geometry=client.process.pid)
        if client.background:
            pids['background_matcher']=client.background.process.pid
            if client.background.refiner_pid:pids['background_icp']=client.background.refiner_pid
        values={name:rss(pid) for name,pid in pids.items()}
        print(json.dumps(dict(mode=mode,captures=6,rss_bytes=values,total_rss_bytes=sum(v for v in values.values() if v is not None),all_available=all(v is not None for v in values.values()))),flush=True)
    finally:client.close()
