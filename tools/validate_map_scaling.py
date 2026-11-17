"""Replay saved dense voxel corrections on disposable database copies, without camera/native RTAB."""
import argparse,importlib.util,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from disk_map import DiskVoxelMap,CloudStore
from rtabmap_backend import RtabmapBackend


class NoNative:
    def __init__(self,folder):pass
    def close(self,**kwargs):pass


class OverlayClouds:
    """Existing clouds are read from source; EVERY write is confined to output."""
    def __init__(self,folder,source):self.local=CloudStore(folder);self.source=source
    def __getitem__(self,node):
        try:return self.local[node]
        except KeyError:return self.source[node]
    def __setitem__(self,node,points):self.local[node]=points


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if source==output or source in output.parents:raise ValueError('Output must be outside source')
    spec=importlib.util.spec_from_file_location('baseline_disk',args.baseline);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    store=CloudStore(source/'map_frames');results={}
    for label,klass in [('baseline',module.DiskVoxelMap),('optimized_sql',DiskVoxelMap)]:
        folder=output/label;folder.mkdir();shutil.copy2(source/'voxel_map.sqlite',folder/'voxel_map.sqlite')
        disk=klass(folder/'voxel_map.sqlite',.003)
        try:
            ids=list(disk.poses);times=[];t=time.perf_counter()
            for node in ids:
                pose=disk.poses[node].copy();pose[0,3]+=.00002
                a=time.perf_counter();disk.update(node,store[node],pose);times.append((time.perf_counter()-a)*1000)
            updates=(time.perf_counter()-t)*1000;t=time.perf_counter();cloud=disk.cloud();reading=(time.perf_counter()-t)*1000
            results[label]=dict(nodes=len(ids),points=len(cloud),update_ms=updates,cloud_read_ms=reading,total_ms=updates+reading,per_node_ms=times)
            if label=='baseline':expected=cloud
            else:np.testing.assert_allclose(cloud,expected,atol=1e-10)
        finally:disk.close()
        print(label,results[label]['total_ms'],flush=True)
    folder=output/'live_preview';folder.mkdir();shutil.copy2(source/'voxel_map.sqlite',folder/'voxel_map.sqlite')
    backend=RtabmapBackend(folder,client_factory=NoNative);backend.clouds=OverlayClouds(folder/'map_frames',store)
    try:
        backend.node_ids=list(backend.voxel_map.poses)
        for node,pose in backend.voxel_map.poses.items():
            p=np.eye(4);p[:3]=pose;backend.poses[node]=p
            backend.radii[node]=float(np.linalg.norm(store[node],axis=1).max())
        backend.preview()
        for p in backend.poses.values():p[0,3]+=.00002
        backend.dirty_nodes=set(backend.node_ids);a=time.perf_counter();cloud,_=backend.preview()
        results['live_small_change']=dict(total_ms=(time.perf_counter()-a)*1000,**backend.preview_stats)
        a=time.perf_counter();cloud,_=backend.preview(force=True)
        results['final_exact']=dict(total_ms=(time.perf_counter()-a)*1000,**backend.preview_stats)
        actual=np.column_stack((cloud[:,2],-cloud[:,0],-cloud[:,1]))
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        results['final_matches_baseline']=True
        # A single new frame should not rebuild all historical contributions.
        node=max(backend.node_ids)+1
        with backend.lock:backend.poses[node]=backend.poses[backend.node_ids[-1]].copy()
        points=store[backend.node_ids[-1]];a=time.perf_counter();backend.retain(node,points);backend.preview()
        results['append_one']=dict(total_ms=(time.perf_counter()-a)*1000,**backend.preview_stats)
    finally:backend.close()
    (output/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))


if __name__=='__main__':main()
