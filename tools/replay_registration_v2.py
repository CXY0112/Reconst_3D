"""Replay real saved scans through both PointDSC registration backends without a camera."""
import sys,json,time,tempfile,zipfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['PYTHONPATH']=str(ROOT)
import numpy as np
import geometry_client
from reconstruct import Calibration
from snapshot_window import ir_to_rgb
session=ROOT/'results/scan_20261015_102420_653218'
calibration=Calibration(**json.loads((session/'sdk_calibration.json').read_text()))
files=sorted((session/'pointcloud_raw').glob('*.npy'))
output=ROOT/'results/loop_memory_v2';output.mkdir(exist_ok=True)
results={}
for name in ('baseline','optimized'):
 with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
  folder=Path(temp)
  if name=='baseline':
   with zipfile.ZipFile(ROOT.parent/'Reconst_3D_archive_before_loop_memory_v2_20261019.zip') as z:
    (folder/'geometry_worker.py').write_bytes(z.read('Reconst_3D/geometry_worker.py'))
   geometry_client.ROOT=folder
  else:geometry_client.ROOT=ROOT
  client=geometry_client.GeometryClient(folder);records=[]
  try:
   for path in files:
    points=ir_to_rgb(np.load(path)/1000,calibration)
    start=time.perf_counter();result=client.register(points)
    elapsed=time.perf_counter()-start
    if result['accepted']:
     committed=client.request('commit');result['memory']=committed.get('memory')
    records.append(dict(file=path.name,seconds=elapsed,accepted=result['accepted'],rmse_mm=result.get('rmse_mm'),loop_edges=result.get('loop_edges'),reason=result.get('reason'),memory=result.get('memory')))
    print(name,len(records),result['accepted'],round(elapsed,2),flush=True)
   results[name]=records
  finally:client.close()
 (output/'real_replay.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
