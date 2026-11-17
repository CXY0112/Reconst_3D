"""Controlled registration benchmark: exact coarse poses isolate retrieval/cache/ICP/graph."""
import sys,os,time,json,tempfile,zipfile,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import activate_geometry_packages
activate_geometry_packages()
import numpy as np
from geometry_worker import GeometryEngine,Frame
from frame_store import FrameStore

archive=ROOT.parent/'Reconst_3D_archive_before_loop_memory_20261019.zip'
with zipfile.ZipFile(archive) as z:old=z.read('Reconst_3D/geometry_worker.py').decode('utf-8')
baseline=types.ModuleType('baseline');baseline.__file__=str(ROOT/'geometry_worker.py')
exec(compile(old,baseline.__file__,'exec'),baseline.__dict__)
rng=np.random.default_rng(7)
a=rng.uniform(-.6,.6,(1800,2))
points=np.concatenate([np.c_[a,np.ones(len(a))],np.c_[np.full(len(a),-.6),a[:,0],a[:,1]+1.5],np.c_[a[:,0],np.full(len(a),.6),a[:,1]+1.5]])
results={}
for name,klass in [('baseline',baseline.GeometryEngine),('optimized',GeometryEngine)]:
 with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
  folder=Path(temporary);engine=klass(folder)
  if name=='optimized':engine.frames.max_frames=2;engine.frames.budget=2*1024**2
  times=[];errors=[];loops=0;retrieved=0
  for i in range(24):
   path=folder/'input.npy';np.save(path,points)
   start=time.perf_counter();prepared=engine.prepare(path)
   # Same physical camera location: identity is a known coarse transform, not an estimated result.
   result=engine.finish([dict(reference=ref,pose=np.eye(4).tolist()) for ref in prepared['candidates']])
   assert result['accepted'],result
   committed=engine.commit();times.append(time.perf_counter()-start)
   errors.append(result['rmse_mm']);loops+=result.get('loop_edges',0)
   retrieved+=prepared.get('retrieval_candidates',i)
  if name=='optimized':memory=engine.frames.stats()
  else:memory=dict(resident_frames=len(engine.frames),resident_array_bytes=sum(f.nbytes() if hasattr(f,'nbytes') else np.asarray(f.base.points).nbytes+f.xyz.nbytes+f.features.nbytes+f.feature_tree.data.nbytes+f.feature_tree.indices.nbytes+sum(np.asarray(p.points).nbytes+np.asarray(p.normals).nbytes for p in f.levels.values()) for f in engine.frames))
  results[name]=dict(frames=24,total_seconds=sum(times),p95_seconds=float(np.percentile(times,95)),rmse_mm_max=max(errors),loop_edges=loops if name=='optimized' else None,reference_comparisons=retrieved,memory=memory)
out=ROOT/'results/loop_memory_comparison';out.mkdir(exist_ok=True)
(out/'controlled.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
