"""Offline native integration/latency probe. Repeats and synthetic motion are NOT a real video test."""
import argparse,csv,json,queue,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rtab_client import RtabClient
from reconstruct import Calibration
from video_capture import VideoSession,VideoFrame
from rtabmap_backend import RtabmapBackend
from snapshot_window import ir_geometry
from kalman_fusion import session_observations
from dataset_store import create_dataset


def render(color,depth,k,shift):
    v,u=np.nonzero(depth);z=depth[v,u]/1000
    x=(u-k[0,2])*z/k[0,0]-shift
    outu=np.rint(x*k[0,0]/z+k[0,2]).astype(int)
    good=(outu>=0)&(outu<depth.shape[1]);v,u,outu,z=v[good],u[good],outu[good],z[good]
    # Sort far to near so each pixel retains the nearest source surface.
    order=np.argsort(z)[::-1];v,u,outu,z=v[order],u[order],outu[order],z[order]
    rgb=np.zeros_like(color);dep=np.zeros_like(depth)
    rgb[v,outu]=color[v,u];dep[v,outu]=np.rint(z*1000).astype(np.uint16)
    return rgb,dep


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=60);parser.add_argument('--pipeline',action='store_true')
    parser.add_argument('--threaded',action='store_true')
    parser.add_argument('--optimizer',choices=('default','isam2'),default='default')
    parser.add_argument('--source',type=Path,default=ROOT/'results/test_new')
    parser.add_argument('--dense-mode',choices=('global_voxels','submaps'),default='global_voxels')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve()
    if (source/'setp1/rgb').is_dir():source=source/'setp1'
    name=sorted((source/'rgb').glob('*.png'))[0].name
    color=cv2.imread(str(source/'rgb'/name));depth=cv2.imread(str(source/'depth'/name),-1)
    calibration=json.loads((source/'sdk_calibration.json').read_text())
    k=np.array([[calibration['fxrgb'],0,calibration['cxrgb']],[0,calibration['fyrgb'],calibration['cyrgb']],[0,0,1.]])
    if args.threaded:
        threaded(args,color,depth,k);return
    record=[]
    if args.pipeline:
        session=VideoSession(queue.Queue(),optimizer=args.optimizer,dense_mode=args.dense_mode);backend=session.backend_factory(session.folder)
        c=Calibration(fxir=k[0,0],fyir=k[1,1],cxir=k[0,2],cyir=k[1,2],fxrgb=k[0,0],fyrgb=k[1,1],cxrgb=k[0,2],cyrgb=k[1,2],R00=1,R11=1,R22=1)
    else:client=RtabClient(args.output,optimizer=args.optimizer)
    try:
        for i in range(args.frames):
            shift=0. if i<10 else .005*min(i-10,args.frames-1-i)
            rgb,dep=render(color,depth,k,shift)
            t=time.perf_counter()
            if args.pipeline:
                packet=VideoFrame(rgb,dep,ir_geometry(dep,c),c,1+i*.1,(i+1,i+1),False)
                status=session._process_frame(backend,i+1,packet,None,None,packet.stamp)
                session.processed+=1;session._after_process(backend,i+1,status)
                processing=(time.perf_counter()-t)*1000
                pt=time.perf_counter()
                fusion=(time.perf_counter()-pt)*1000
                times=backend.timings;node=backend.report.get('node') or 0
            else:
                result=client.frame(rgb,dep,k,1+i*.1);processing=(time.perf_counter()-t)*1000
                status=result['status'];node=result['node'];times=result['timings'];fusion=0
            pose_error=None
            if not args.pipeline and status==2:
                pose=result['poses'][result['poses'][:,0]==node][0,1:].reshape(3,4)
                pose_error=float(np.linalg.norm(pose[:,3]-[0,-shift,0])*1000)
            record.append(dict(frame=i+1,status=status,node=node,processing_ms=processing,fusion_ms=fusion,vo_ms=times[0],graph_ms=times[1],inliers=times[2],shift_m=shift,synthetic_position_error_mm=pose_error))
        if args.pipeline:
            session._finish_processing()
            points,nodes=backend.preview(force=True);session._before_close(backend,points,nodes);backend.close()
            saved=session.save(dataset=create_dataset(args.output,'pipeline_dataset'));_,observations,_=session_observations(saved)
            summary_extra=dict(saved_folder=str(saved),retained=len(session.names),observations=len(observations),map_points=len(points))
        else:client.close();summary_extra={}
    finally:
        if args.pipeline:session._finish_processing();backend.close()
        else:client.close(force=True)
    with (args.output/'frames.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(record[0]));writer.writeheader();writer.writerows(record)
    summary=dict(test='synthetic horizontal translation of one recorded RGB-D view; no new camera capture',optimizer=args.optimizer,frames=len(record),lost=sum(r['status']==0 for r in record),keyframes=sum(r['status']==2 for r in record),**summary_extra)
    for key in ('vo_ms','graph_ms','processing_ms','fusion_ms'):
        summary[key+'_p50_p95']=np.percentile([r[key] for r in record],[50,95]).tolist()
    errors=[r['synthetic_position_error_mm'] for r in record if r['synthetic_position_error_mm'] is not None]
    if errors:summary['synthetic_keyframe_position_error_mm_p50_max']=np.percentile(errors,[50,100]).tolist()
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


def threaded(args,color,depth,k):
    frames=[render(color,depth,k,.005*min(max(0,i-10),args.frames-1-i)) for i in range(args.frames)]
    c=Calibration(fxir=k[0,0],fyir=k[1,1],cxir=k[0,2],cyir=k[1,2],fxrgb=k[0,0],fyrgb=k[1,1],cxrgb=k[0,2],cyrgb=k[1,2],R00=1,R11=1,R22=1)
    session=VideoSession(queue.Queue(),hz=10,optimizer=args.optimizer,dense_mode=args.dense_mode)
    class Camera:
        pair_cloud=None
        def __enter__(self):self.i=0;self.start=time.monotonic();return self
        def __exit__(self,*args):pass
        def read(self,**kwargs):
            if self.i==len(frames):session.stop.set();raise TimeoutError()
            time.sleep(max(0,self.start+self.i*.1-time.monotonic()))
            rgb,dep=frames[self.i];self.i+=1;stamp=1000+self.i*100
            return ((rgb,None,self.i,stamp),(dep,None,self.i,stamp)),c
    session.camera_factory=Camera;start=time.perf_counter();session.start()
    session.capture_thread.join(30);session.map_thread.join(60)
    if session.capture_thread.is_alive() or session.map_thread.is_alive():
        session.stop.set();raise TimeoutError('Offline replay did not stop')
    if session.failed:raise RuntimeError(str(list(session.events.queue)))
    wall=time.perf_counter()-start;saved=session.save(dataset=create_dataset(args.output,'pipeline_dataset'));_,observations,_=session_observations(saved)
    rows=list(csv.DictReader((saved/'online_metrics.csv').open()))
    summary=dict(test='10 Hz synthetic RGB-D camera replay through production capture/map/dense threads',
        optimizer=args.optimizer,dense_mode=args.dense_mode,camera_frames=args.frames,sampled=session.sampled,processed=session.processed,dropped=session.dropped,
        retained=len(session.names),observations=len(observations),dense_skipped=session.dense_dropped,
        tracking_lost=sum(int(r['tracking_status'])==0 for r in rows),wall_seconds=wall,saved_folder=str(saved))
    for key in ('processing_ms','odometry_ms','graph_ms'):
        summary[key+'_p50_p95']=np.percentile([float(r[key]) for r in rows],[50,95]).tolist()
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
