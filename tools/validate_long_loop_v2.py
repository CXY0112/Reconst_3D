"""Known-transform synthetic orbit; coarse proposals are supplied, not PointDSC outputs."""
import sys,time,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from geometry_worker import GeometryEngine
import numpy as np
OUT=ROOT/'results/loop_memory_v2';OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(19);a=rng.uniform(-.6,.6,(1000,2))
world=np.concatenate([np.c_[a,np.ones(len(a))],np.c_[np.full(len(a),-.6),a[:,0],a[:,1]+1.5],np.c_[a[:,0],np.full(len(a),.6),a[:,1]+1.5]])
with tempfile.TemporaryDirectory(dir=OUT) as temporary:
    folder=Path(temporary);engine=GeometryEngine(folder);engine.frames.max_frames=3
    camera=[];times=[];errors=[];loops=0;injected=0;rejected_false=0
    for i in range(100):
        pose=np.eye(4);phase=2*np.pi*i/24
        pose[:3,3]=[.05*np.sin(phase),.05*(np.cos(phase)-1),0]
        camera.append(pose);path=folder/'input.npy'
        np.save(path,(world-pose[:3,3])+rng.normal(0,.0005,world.shape))
        start=time.perf_counter();prepared=engine.prepare(path);proposals=[]
        for ref in prepared['candidates']:
            relative=np.linalg.inv(camera[ref])@pose
            if ref<max(0,i-4) and i%10==0:
                relative=relative.copy();relative[0,3]+=1.;injected+=1
            proposals.append(dict(reference=ref,pose=relative.tolist()))
        result=engine.finish(proposals)
        assert result['accepted'],result
        rejected_false+=sum('reference' in entry and entry['reference']<max(0,i-4) for entry in result['candidates'])
        engine.commit();times.append(time.perf_counter()-start);loops+=result['loop_edges']
        expected=engine.poses[0]@pose
        errors.append(float(np.linalg.norm(expected[:3,3]-engine.poses[-1][:3,3])*1000))
    summary=dict(captures=100,accepted=100,known_transform_coarse_proposals=True,total_seconds=sum(times),p95_seconds=float(np.percentile(times,95)),translation_error_mm_p50_p95_max=np.percentile(errors,[50,95,100]).tolist(),retained_historical_edges=loops,injected_bad_historical_proposals=injected,rejected_historical_candidates=rejected_false,memory=engine.frames.stats())
    assert rejected_false>=injected
    (OUT/'long_sequence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Retain diagnostics, not large temporary frame archives.
    (OUT/'long_sequence_diagnostics.jsonl').write_bytes((folder/'loop_diagnostics.jsonl').read_bytes())
    print(json.dumps(summary,indent=2))
