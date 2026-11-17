"""Exercise native map synchronization and disk fusion using saved RGB-D only."""
import sys,json,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from geometry_snapshot_backend import SnapshotBackend
from snapshot_window import ir_to_rgb
from reconstruct import Calibration
source=ROOT/'results/scan_20261015_102420_653218';output=ROOT/'results/loop_memory_v2';output.mkdir(exist_ok=True)
cal=Calibration(**json.loads((source/'sdk_calibration.json').read_text()))
import re
k=np.array(json.loads(re.search(r'camera_matrix:.*?data: (\[.*?\])',(source/'hp60c.yaml').read_text(),re.S).group(1))).reshape(3,3)
records=[]
with tempfile.TemporaryDirectory(dir=output) as temp:
    backend=SnapshotBackend(Path(temp))
    try:
        for path in sorted((source/'pointcloud_raw').glob('*.npy')):
            name=path.stem+'.png';color=cv2.imread(str(source/'rgb'/name));depth=cv2.imread(str(source/'depth'/name),cv2.IMREAD_UNCHANGED)
            start=time.perf_counter();status=backend.process(color,depth,k,float(path.stem),geometry=ir_to_rgb(np.load(path)/1000,cal))
            points,nodes=backend.preview()
            records.append(dict(file=path.name,status=status,points=len(points),nodes=len(nodes),seconds=time.perf_counter()-start,report=backend.report))
            print(len(records),status,len(points),flush=True)
        assert all(r['status']==2 for r in records),records
        storage=backend.voxel_map.stats()
    finally:backend.close()
    assert (Path(temp)/'map.db').stat().st_size>0
    result=dict(captures=records,storage=storage,native_database_saved=True)
    (output/'native_replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
