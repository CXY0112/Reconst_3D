"""Replay the SAME saved RGB-D observations through either frontend, without camera.

Uses raw RGB/depth and factory calibration. Geometry is reconstructed from raw
depth (not original SDK XYZ); source optimized poses are never used as guesses.
Only stored frames can be replayed: this is not the original 10 Hz stream.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import time
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def replay_depth_frames(source,names,override='auto'):
    from rgbd_geometry import recorded_coordinates,camera_frame
    if override!='auto':return {name:camera_frame(override) for name in names}
    declarations={}
    path=source/'frames.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line);coordinates=recorded_coordinates(row)
            if coordinates:
                name=f'{float(row["stamp"]):.6f}.png';frame=coordinates['raw_depth_frame']
                if name in declarations and declarations[name]!=frame:raise ValueError('Conflicting raw-depth coordinate declarations')
                declarations[name]=frame
    if any(name not in declarations for name in names):
        raise ValueError('Recording lacks explicit raw-depth coordinates; verify camera mode and pass --depth-frame ir or --depth-frame rgb')
    return {name:declarations[name] for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New comparison directory')
    parser.add_argument('--frontend',choices=('rgbd','bufferx','gcvo_cpu','gcvo_cuda','both','rgbd_gcvo_cpu'),default='both')
    parser.add_argument('--limit',type=int,default=0,help='0 replays all stored frames')
    parser.add_argument('--depth-frame',choices=('auto','ir','rgb'),default='auto',help='New recordings use per-frame declarations; old recordings require a verified override')
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be >= 0')
    source=args.source.resolve()
    if (source/'setp1').is_dir():source=source/'setp1'
    names=sorted(p.name for p in (source/'rgb_raw').glob('*.png') if (source/'depth_raw'/p.name).is_file())
    if args.limit:names=names[:args.limit]
    if not names:parser.error('Source must contain matching rgb_raw/depth_raw PNGs')
    try:depth_frames=replay_depth_frames(source,names,args.depth_frame)
    except ValueError as exc:parser.error(str(exc))
    args.output.mkdir(parents=True,exist_ok=False)
    from reconstruct import Calibration
    from rgbd_geometry import capture_geometry
    from video_capture import VideoFrame,VideoSession
    from dataset_store import create_dataset
    c=Calibration()
    for key,value in json.loads((source/'sdk_calibration.json').read_text()).items():setattr(c,key,value)
    summary=dict(source=str(source),input_names=names,geometry='calibrated raw depth, same for both frontends',
                 depth_frame_override=args.depth_frame,raw_depth_frames=depth_frames,
                 caveat='Stored observations only; no ground truth accuracy measurement',runs={})
    frontends={'both':('rgbd','bufferx'),'rgbd_gcvo_cpu':('rgbd','gcvo_cpu')}.get(args.frontend,(args.frontend,))
    for frontend in frontends:
        session=VideoSession(queue.Queue(),optimizer='isam2',dense_mode='submaps',frontend=frontend)
        backend=None;rows=[]
        try:
            backend=session.backend_factory(session.folder)
            process_measurement={};original_process=backend.process
            def timed_process(*positional,**keywords):
                began=time.perf_counter()
                try:return original_process(*positional,**keywords)
                finally:process_measurement['ms']=(time.perf_counter()-began)*1000
            backend.process=timed_process
            for count,name in enumerate(names,1):
                rgb=cv2.imread(str(source/'rgb_raw'/name));depth=cv2.imread(str(source/'depth_raw'/name),-1)
                if rgb is None or depth is None:raise ValueError('Cannot read '+name)
                stamp=float(Path(name).stem)
                frame=depth_frames[name];points,_=capture_geometry(depth,c,depth_frame=frame)
                packet=VideoFrame(rgb,depth,points,c,stamp,(count,count),False,frame,frame)
                start=time.perf_counter();status=session._process_frame(backend,count,packet,None,None,stamp)
                session.processed+=1;session.sampled+=1
                session._after_process(backend,count,status)
                # Offline comparison is deterministic: retain every accepted
                # dense observation rather than reproducing live queue overload.
                if session.dense_thread is not None:session.dense_pending.join()
                row=dict(frame=name,status=status,processing_ms=(time.perf_counter()-start)*1000,
                         backend_process_ms=process_measurement['ms'],report=backend.report.copy())
                rows.append(row);print(f'{frontend}: {count}/{len(names)} status={status} {row["processing_ms"]:.1f} ms',flush=True)
                with (args.output/f'{frontend}_frames.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                while not session.events.empty():session.events.get_nowait()
            session._finish_processing()
            if not session.names:raise RuntimeError('No dense keyframes retained')
            points,nodes=backend.final_preview();session._before_close(backend,points,nodes);backend.close();backend=None
            saved=session.save(dataset=create_dataset(args.output,frontend))
            summary['runs'][frontend]=dict(saved=str(saved),frames=len(rows),rejected=sum(r['status']==0 for r in rows),
                retained=len(session.names),points=len(points),processing_ms_p50_p95=np.percentile([r['processing_ms'] for r in rows],[50,95]).tolist())
        finally:
            session._finish_processing()
            if backend is not None:backend.close()
            # Preserve failed pending sessions for diagnosis, never delete input.
            if session.folder.name.startswith('.pending_scan_'):summary['runs'].setdefault(frontend,dict(recovery=str(session.folder)))
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
