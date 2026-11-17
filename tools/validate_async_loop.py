"""Saved-scan comparison with real geometry/async workers; native bridge is simulated."""
import sys,os,time,json,tempfile,ctypes as ct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['PYTHONPATH']=str(ROOT)
import numpy as np
import cv2
from geometry_client import GeometryClient
from geometry_snapshot_backend import SnapshotBackend
from disk_map import DiskVoxelMap,CloudStore
from snapshot_window import ir_to_rgb
from reconstruct import Calibration

class Native:
    def __init__(self):self.count=0;self.syncs=0
    def snapshot_add(self,*args):
        self.count+=1;ct.cast(args[-1],ct.POINTER(ct.c_int))[0]=self.count;return 2
    def geometry_sync(self,*args):self.syncs+=1;return 0
    def online_times(self,*args):pass
    def online_close(self,*args):return 0

source=ROOT/'results/scan_20261015_102420_653218'
cal=Calibration(**json.loads((source/'sdk_calibration.json').read_text()))
files=sorted((source/'pointcloud_raw').glob('*.npy'))
output=ROOT/'results/async_loop_v3';output.mkdir(exist_ok=True)
results={}
for mode in ('synchronous','asynchronous'):
    with tempfile.TemporaryDirectory(dir=output) as temp:
        folder=Path(temp);backend=SnapshotBackend.__new__(SnapshotBackend)
        backend.folder=folder;backend.clouds=CloudStore(folder/'map_frames');backend.voxel_map=DiskVoxelMap(folder/'voxel_map.sqlite')
        backend.geometry_engine=GeometryClient(folder,asynchronous=mode=='asynchronous')
        backend.node_ids=[];backend.geometry_poses=[];backend.reason='';backend.report={};backend.timings=np.zeros(5);backend.frames=0
        backend.lib=Native();backend.handle=1;backend.cookies=[]
        rows=[];started=time.perf_counter()
        try:
            for path in files:
                stamp=path.stem;color=cv2.imread(str(source/'rgb'/(stamp+'.png')));depth=cv2.imread(str(source/'depth'/(stamp+'.png')),cv2.IMREAD_UNCHANGED)
                k=np.array([[cal.fxrgb,0,cal.cxrgb],[0,cal.fyrgb,cal.cyrgb],[0,0,1.]])
                # Sole coordinator applies completed results between captures.
                backend.poll_background()
                start=time.perf_counter();status=backend.process(color,depth,k,float(stamp),geometry=ir_to_rgb(np.load(path)/1000,cal))
                foreground=time.perf_counter()-start
                backend.preview()
                rows.append(dict(file=path.name,status=status,foreground_seconds=foreground,rmse_mm=backend.report.get('rmse_mm'),relocalization=backend.report.get('synchronous_relocalization',False)))
                print(mode,len(rows),status,round(foreground,2),flush=True)
            backend.finish_background(timeout=30.)
            final,nodes=backend.preview()
            background=backend.geometry_engine.background
            stats=background.stats() if background else dict(running=0,pending=0,replaced=0)
            applied=[]
            if (folder/'async_loop_results.jsonl').exists():applied=[json.loads(line) for line in (folder/'async_loop_results.jsonl').read_text().splitlines()]
            result=dict(captures=rows,total_seconds=time.perf_counter()-started,foreground_total_seconds=sum(r['foreground_seconds'] for r in rows),foreground_p95_seconds=float(np.percentile([r['foreground_seconds'] for r in rows],95)),accepted=sum(r['status']==2 for r in rows),final_nodes=len(nodes),final_points=len(final),native_bridge_simulated=True,background=stats,loop_results=applied,sync_calls=backend.lib.syncs)
            assert result['accepted']==len(files)
            assert len(nodes)==len(files)
            assert stats['running']==stats['pending']==0
            results[mode]=result
            if (folder/'loop_diagnostics.jsonl').exists():(output/(mode+'_diagnostics.jsonl')).write_bytes((folder/'loop_diagnostics.jsonl').read_bytes())
        finally:backend.close()
    (output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
