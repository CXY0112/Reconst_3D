"""Offline performance workloads using unchanged production pipeline APIs."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def mark(folder,event,**details):
    with (folder/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'time':time.monotonic(),'event':event,**details},ensure_ascii=False)+'\n')
    print(event,flush=True)


def replay(source,folder,core_only=False):
    import numpy as np
    import cv2
    cv2.setNumThreads(1)  # ScanWindow.__init__ sets this before live backend startup.
    from reconstruct import Calibration,save_cloud
    from snapshot_window import ir_geometry,ir_to_rgb
    from rtabmap_scan import register_rgbd
    from geometry_snapshot_backend import SnapshotBackend
    from online_backend import export_map
    calibration=Calibration()
    for key,value in json.loads((source/'sdk_calibration.json').read_text(encoding='utf-8')).items():setattr(calibration,key,value)
    frames=[json.loads(line) for line in (source/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    mark(folder,'stage1_backend_init_start')
    if core_only:
        from geometry_client import GeometryClient
        from voxel_map import VoxelMap,voxel_groups
        class CoreBackend:
            def __init__(self):
                self.geometry_engine=GeometryClient(folder);self.voxel_map=VoxelMap(.003)
                self.clouds={};self.node_ids=[];self.geometry_poses=[];self.report={}
            def process(self,color,depth,k,stamp,geometry):
                self.report=self.geometry_engine.register(geometry)
                if not self.report['accepted']:return 0
                self.geometry_poses=np.asarray(self.report['poses']);self.geometry_engine.request('commit')
                node=len(self.node_ids)+1;self.node_ids.append(node)
                _,sums,counts=voxel_groups(geometry,self.voxel_map.size)
                self.clouds[node]=sums/counts[:,None]
                return 2
            def preview(self):return SnapshotBackend.preview(self)
            def close(self):self.geometry_engine.close()
        backend=CoreBackend()
    else:backend=SnapshotBackend(folder)
    mark(folder,'stage1_backend_init_end')
    records=[];final=None
    try:
        for frame in frames:
            capture=frame['capture'];name=f'{frame["stamp"]:.6f}'
            mark(folder,'stage1_capture_start',capture=capture)
            started=time.perf_counter()
            raw=cv2.imread(str(source/'rgb_raw'/(name+'.png')))
            depth=cv2.imread(str(source/'depth_raw'/(name+'.png')),cv2.IMREAD_UNCHANGED)
            sdk=np.load(source/'pointcloud_raw'/(name+'.npy'))
            read_seconds=time.perf_counter()-started
            mark(folder,'stage1_input_loaded',capture=capture)
            start=time.perf_counter();points=ir_geometry(depth,calibration,sdk)
            color,registered,k=register_rgbd(raw,depth,calibration,points_ir=points*1000)
            geometry=ir_to_rgb(points,calibration)
            registration_seconds=time.perf_counter()-start
            mark(folder,'stage1_prepared',capture=capture)
            start=time.perf_counter();status=backend.process(color,registered,k,frame['stamp'],geometry=geometry)
            match_seconds=time.perf_counter()-start
            start=time.perf_counter()
            if status>0:final,nodes=backend.preview()
            fusion_seconds=time.perf_counter()-start
            row={'capture':capture,'accepted':status==2,'read_seconds':read_seconds,
                'rgbd_preparation_seconds':registration_seconds,'matching_seconds':match_seconds,
                'map_fusion_seconds':fusion_seconds,'total_seconds':time.perf_counter()-started,
                'raw_points':len(points),'map_points':0 if final is None else len(final),
                'matching_report':backend.report.copy()}
            records.append(row);mark(folder,'stage1_capture_end',**{k:v for k,v in row.items() if k!='matching_report'})
        mark(folder,'stage1_close_start');backend.close();backend=None
        mark(folder,'stage1_export_start');start=time.perf_counter()
        if not core_only:export_map(folder,voxel_size=.003,decimation=1)
        world=np.column_stack((final[:,2],-final[:,0],-final[:,1]))
        np.save(folder/'full_geometry_world.npy',world)
        if not core_only:(folder/'map_cloud.ply').rename(folder/'map_cloud_rgb_registered.ply')
        save_cloud(folder/'map_cloud.ply',world,'RTAB-Map world XYZ in metres, X forward Y left Z up')
        export_seconds=time.perf_counter()-start;mark(folder,'stage1_export_end')
        return {'captures':records,'accepted':sum(row['accepted'] for row in records),'map_points':len(final),
                'export_seconds':export_seconds,'camera_io_measured':False,'rendering_measured':False,
                'rtabmap_image_database_measured':not core_only,'workload':'production geometric registration/commit, voxel_groups and SnapshotBackend.preview; no native RTAB-Map image database' if core_only else 'native backend'}
    finally:
        if backend is not None:backend.close()


def preprocess(source,folder):
    from cloud_import import load_cloud
    from preprocess import process_input,Parameters,save_result
    mark(folder,'stage2_load_start');start=time.perf_counter();loaded=load_cloud(source/'map_cloud.ply' if source.is_dir() else source)
    load_seconds=time.perf_counter()-start;mark(folder,'stage2_load_end',points=len(loaded.points))
    mark(folder,'stage2_process_start');start=time.perf_counter()
    def progress(fraction,text):
        if fraction>=.87 or '第' in text:mark(folder,'stage2_progress',fraction=fraction,text=text)
    result=process_input(source,loaded.points,Parameters(),backend='kalman',progress=progress)
    process_seconds=time.perf_counter()-start;mark(folder,'stage2_process_end')
    mark(folder,'stage2_save_start');start=time.perf_counter()
    output=folder/'setp2_output';save_result(result,source=source/'map_cloud.ply' if source.is_dir() else source,
        destination=output,metadata={'unit':loaded.unit,'coordinates':loaded.coordinates})
    save_seconds=time.perf_counter()-start;mark(folder,'stage2_save_end')
    return {'load_seconds':load_seconds,'process_seconds':process_seconds,'save_seconds':save_seconds,
            'input_points':len(loaded.points),'output_points':len(result.points),'output':str(output)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['stage1','stage2','stage3','stage4','gui'])
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--core-only',action='store_true')
    args=parser.parse_args();folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
    mark(folder,'worker_start',stage=args.stage);started=time.perf_counter()
    if args.stage=='stage1':result=replay(args.source,folder,args.core_only)
    elif args.stage=='stage2':result=preprocess(args.source,folder)
    elif args.stage=='stage3':
        sys.path.insert(0,str(ROOT/'vendor/geometry_packages'))
        from reconstruction_pipeline import run_pipeline
        mark(folder,'stage3_process_start');result=run_pipeline(args.source,folder/'setp3_output')
        mark(folder,'stage3_process_end')
    elif args.stage=='stage4':
        sys.path.insert(0,str(ROOT/'vendor/geometry_packages'))
        from appearance_pipeline import run_pipeline
        mark(folder,'stage4_process_start');result=run_pipeline(args.source,folder/'setp4_output')
        mark(folder,'stage4_process_end')
    else:
        from performance_ui import run_gui
        result=run_gui(args.source,folder,mark)
    result['worker_wall_seconds']=time.perf_counter()-started
    (folder/'workload.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    mark(folder,'worker_done');time.sleep(.3)


if __name__=='__main__':main()
