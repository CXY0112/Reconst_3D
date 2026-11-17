"""Offline long-sequence checks and isolated map-storage RSS comparison."""
import sys,os,time,json,tempfile,subprocess,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
OUT=ROOT/'results/loop_memory_v2';OUT.mkdir(exist_ok=True)
if len(sys.argv)>1:
    import ctypes as ct
    from types import SimpleNamespace
    class Counters(ct.Structure):
        _fields_=[('cb',ct.c_ulong),('faults',ct.c_ulong)]+[(name,ct.c_size_t) for name in ('peak_rss','rss','peak_pool','pool','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    def memory_info():
        ct.windll.kernel32.GetCurrentProcess.restype=ct.c_void_p
        handle=ct.windll.kernel32.GetCurrentProcess()
        ct.windll.psapi.GetProcessMemoryInfo.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_ulong]
        counters=Counters();counters.cb=ct.sizeof(counters)
        if not ct.windll.psapi.GetProcessMemoryInfo(handle,ct.byref(counters),counters.cb):raise OSError(ct.get_last_error())
        return SimpleNamespace(rss=counters.rss)

    from voxel_map import VoxelMap
    from disk_map import DiskVoxelMap,CloudStore
    mode=sys.argv[1];process=SimpleNamespace(memory_info=memory_info);rng=np.random.default_rng(8)
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        folder=Path(temp);gc.collect();initial=process.memory_info().rss
        voxel=VoxelMap(.003) if mode=='memory' else DiskVoxelMap(folder/'map.sqlite',.003)
        clouds={} if mode=='memory' else CloudStore(folder/'clouds')
        samples=[];times=[];pose=np.c_[np.eye(3),np.zeros(3)]
        points=rng.uniform(-.3,.3,(5000,3))
        for i in range(100):
            start=time.perf_counter();clouds[i]=points.copy();voxel.update(i,clouds[i],pose)
            times.append(time.perf_counter()-start)
            if (i+1)%20==0:samples.append(dict(nodes=i+1,rss_bytes=process.memory_info().rss))
        result=voxel.cloud();summary=dict(mode=mode,nodes=100,points=len(result),centroid_sum=result.sum(axis=0).tolist(),initial_rss_bytes=initial,samples=samples,total_seconds=sum(times),p95_seconds=float(np.percentile(times,95)))
        if mode=='disk':summary['storage']=voxel.stats();voxel.close()
        del clouds;gc.collect()
        print(json.dumps(summary));sys.exit(0)

results={}
for mode in ('memory','disk'):
    completed=subprocess.run([sys.executable,__file__,mode],capture_output=True,text=True,check=True)
    results[mode]=json.loads(completed.stdout.strip().splitlines()[-1])
np.testing.assert_allclose(results['memory']['centroid_sum'],results['disk']['centroid_sum'],atol=1e-8)
assert results['memory']['points']==results['disk']['points']
(OUT/'map_storage.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
