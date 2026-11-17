"""Replay identical retained frame clouds/poses; do not access a camera or change input."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from submap_backend import SubmapBackend
from disk_map import CloudStore


class NoNative:
    def __init__(self,folder,**kwargs):pass
    def close(self,**kwargs):pass


def set_poses(backend,poses):
    with backend.lock:backend._poses(np.array([[n,*p[:3].ravel()] for n,p in poses.items()]))


def summary(values):
    return dict(total_ms=float(sum(values)),p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),max_ms=float(max(values)),calls=len(values))


def settle(backend):
    timings=[];stats=[]
    for _ in range(1024):
        start=time.perf_counter();cloud,_=backend.preview();timings.append((time.perf_counter()-start)*1000)
        stats.append(backend.preview_stats.copy())
        if not backend.pending_updates():return cloud,dict(**summary(timings),stats=stats)
    raise RuntimeError('Pending corrections did not drain')


def distance(a,b):
    ids=np.linspace(0,len(a)-1,min(len(a),50000),dtype=int)
    d=cKDTree(b).query(a[ids],workers=2)[0]*1000
    return dict(samples=len(d),p50_mm=float(np.median(d)),p95_mm=float(np.percentile(d,95)),max_mm=float(d.max()),within3mm=float(np.mean(d<=3)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/test_new/setp1')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/experiments/submaps_20261021/before/rtabmap_backend.py')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--nodes',type=int,default=38)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if source==output or source in output.parents:raise ValueError('Output must be outside input')
    graph=json.loads((source/'geometry_poses.json').read_text(encoding='utf-8'))
    poses={n:np.array(p) for n,p in list(zip(graph['node_ids'],graph['poses']))[:args.nodes]}
    store=CloudStore(source/'map_frames')
    paths=[source/'geometry_poses.json',*[source/'map_frames'/f'{n}.npy' for n in poses]]
    hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=hashes();spec=importlib.util.spec_from_file_location('submap_baseline',args.baseline)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    runs={}
    for name,klass in (('baseline',baseline.RtabmapBackend),('submaps',SubmapBackend)):
        folder=output/name;folder.mkdir();backend=klass(folder,client_factory=NoNative);visible={};times=[];stats=[]
        try:
            for i,(node,pose) in enumerate(poses.items()):
                visible[node]=pose;start=time.perf_counter();set_poses(backend,visible)
                backend.retain(node,store[node]);backend.preview();times.append((time.perf_counter()-start)*1000)
                stats.append(backend.preview_stats.copy())
                if (i+1)%4==0:print(name,'append',i+1,round(sum(times)/1000,2),'seconds',flush=True)
            result=dict(append=dict(**summary(times),stats=stats))
            move=np.eye(4);move[:3,:3]=Rotation.from_euler('z',3,degrees=True).as_matrix();move[:3,3]=[.024,-.012,.006]
            corrected={n:move@p for n,p in poses.items()};set_poses(backend,corrected)
            preview,result['rigid_correction']=settle(backend)
            np.save(folder/'rigid_preview.npy',preview)
            print(name,'rigid',round(result['rigid_correction']['total_ms']/1000,3),'seconds',flush=True)
            start=time.perf_counter();exact,_=backend.final_preview();result['rigid_final_ms']=(time.perf_counter()-start)*1000
            np.save(folder/'rigid_final.npy',exact)
            for i,p in enumerate(corrected.values()):p[1,3]+=.004*np.sin(i*.7)
            set_poses(backend,corrected);preview,result['nonrigid_correction']=settle(backend)
            print(name,'nonrigid',round(result['nonrigid_correction']['total_ms']/1000,3),'seconds',flush=True)
            start=time.perf_counter();exact,_=backend.final_preview();result['nonrigid_final_ms']=(time.perf_counter()-start)*1000
            np.save(folder/'nonrigid_final.npy',exact)
            result['points']=len(exact);result['submaps']=len(getattr(backend.voxel_map,'groups',[]))
            result['node_count']=len(backend.node_ids)
            runs[name]=result;(folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        finally:backend.close()
    validation={}
    for phase in ('rigid','nonrigid'):
        expected=np.load(output/'baseline'/f'{phase}_final.npy',mmap_mode='r')
        actual=np.load(output/'submaps'/f'{phase}_final.npy',mmap_mode='r')
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)
        validation[phase+'_final_max_abs_m']=float(np.max(abs(actual-expected))) if len(actual) else 0.
        if phase=='rigid':
            preview=np.load(output/'submaps/rigid_preview.npy',mmap_mode='r')
            validation['preview_to_exact']=distance(preview,expected)
            validation['exact_to_preview']=distance(expected,preview)
    validation['source_unchanged']=before==hashes();assert validation['source_unchanged']
    report=dict(source=str(source),nodes=len(poses),runs=runs,validation=validation,
        scope='Same saved dense frame centroids and poses; prescribed coherent and internal corrections. Not live SLAM, not RGB coloring or local surface fitting.')
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(validation=validation,summary={k:{p:v[p]['total_ms'] for p in ('append','rigid_correction','nonrigid_correction')} for k,v in runs.items()}),indent=2),flush=True)


if __name__=='__main__':main()
