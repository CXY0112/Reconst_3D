"""Pixel-preserving RGB-D cleaning. No voxel merging, pose fitting or cloud API calls."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os,shutil,time,uuid
import cv2
import numpy as np

SCHEMA='aligned_rgbd_v1'


@dataclass(frozen=True)
class Parameters:
    radius:int=2
    iterations:int=2
    depth_sigma_mm:float=8.
    edge_mm:float=20.
    color_sigma:float=12.
    min_neighbors:int=3
    max_move_mm:float=10.

    def validate(self):
        values=list(asdict(self).values())
        if not all(np.isfinite(values)) or not isinstance(self.radius,int) or not isinstance(self.iterations,int) or not 1<=self.radius<=4 or not 1<=self.iterations<=5:
            raise ValueError('无效清洗参数')
        if min(self.depth_sigma_mm,self.edge_mm,self.color_sigma,self.max_move_mm)<=0 or not 0<=self.min_neighbors<(2*self.radius+1)**2:
            raise ValueError('无效清洗阈值')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():raise InterruptedError('已取消；原有数据保持不变')


def xyz_from_depth(depth,k):
    depth=np.asarray(depth,np.float32);k=np.asarray(k,float)
    if depth.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('深度或相机内参无效')
    v,u=np.indices(depth.shape,dtype=np.float32)
    result=np.stack(((u-k[0,2])*depth/k[0,0],(v-k[1,2])*depth/k[1,1],depth),-1).astype(np.float32)
    result[~np.isfinite(depth)|(depth<.15)|(depth>4)]=np.nan
    return result


def shift(array,dy,dx,fill=0):
    result=np.roll(array,(dy,dx),axis=(0,1))
    if dy>0:result[:dy]=fill
    elif dy<0:result[dy:]=fill
    if dx>0:result[:,:dx]=fill
    elif dx<0:result[:,dx:]=fill
    return result


def clean(rgb,depth,k,params=None,cancel=None):
    """Joint bilateral depth filter; original RGB and camera rays remain unchanged."""
    params=params or Parameters();params.validate();started=time.perf_counter()
    rgb=np.asarray(rgb);raw=np.asarray(depth,np.float32);k=np.asarray(k,float)
    if rgb.dtype!=np.uint8 or rgb.shape!=(*raw.shape,3):raise ValueError('RGB与深度必须共享像素网格')
    valid=np.isfinite(raw)&(raw>=.15)&(raw<=4)
    input_count=int(valid.sum())
    current=np.where(valid,raw,0).astype(np.float32)
    original=current.copy();lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    for _ in range(params.iterations):
        check_cancel(cancel);total=np.zeros(raw.shape,np.float32);weighted=np.zeros_like(total);neighbors=np.zeros(raw.shape,np.uint16)
        for dy in range(-params.radius,params.radius+1):
            check_cancel(cancel)
            for dx in range(-params.radius,params.radius+1):
                other=shift(current,dy,dx);ok=valid&shift(valid,dy,dx,False)
                difference=(other-current)*1000;ok&=abs(difference)<=params.edge_mm
                color_delta=shift(lab,dy,dx)-lab
                weight=np.exp(-.5*((difference/params.depth_sigma_mm)**2+
                    np.sum(color_delta**2,axis=2)/params.color_sigma**2+(dy*dy+dx*dx)/2.25))*ok
                total+=weight;weighted+=weight*other
                if dy or dx:neighbors+=ok
        valid&=neighbors>=params.min_neighbors
        candidate=weighted/np.maximum(total,1e-12)
        current=np.clip(candidate,original-params.max_move_mm/1000,original+params.max_move_mm/1000)
        current[~valid]=0
    result=np.where(valid,current,np.nan).astype(np.float32)
    xyz=xyz_from_depth(result,k);v,u=np.nonzero(valid)
    if len(u):
        p=xyz[valid];reproject=np.c_[p[:,0]/p[:,2]*k[0,0]+k[0,2],p[:,1]/p[:,2]*k[1,1]+k[1,2]]
        pixel_error=float(np.linalg.norm(reproject-np.c_[u,v],axis=1).max())
    else:pixel_error=0.
    moved=abs(result[valid]-raw[valid])*1000
    report=dict(algorithm='RGB-guided joint bilateral depth cleaning',parameters=asdict(params),
        width=rgb.shape[1],height=rgb.shape[0],input_valid=input_count,
        output_valid=int(valid.sum()),invalid_or_missing_filled=0,rgb_pixels_unchanged=True,
        movement_mm_p50_p95=np.percentile(moved,[50,95]).tolist() if len(moved) else [],
        max_movement_mm=float(moved.max()) if len(moved) else 0.,pixel_roundtrip_max_px=pixel_error,
        elapsed_seconds=time.perf_counter()-started,units='m',coordinates='OpenCV X right Y down Z forward',
        schema=SCHEMA,complete=True)
    return dict(rgb=rgb.copy(),depth=result,xyz=xyz,valid=valid,report=report)


def _safe_path(root,relative):
    path=(Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('数据清单包含目录外路径')
    return path


def sources(folder):
    """Import native sequences or accepted SDK observations; single PLY is insufficient."""
    folder=Path(folder).resolve()
    if (folder/'pointmap_dataset.json').is_file():
        manifest=read_json(folder/'pointmap_dataset.json')
        if manifest.get('schema')!=SCHEMA:raise ValueError('不支持的点图数据格式')
        rows=[]
        for row in manifest['frames']:
            base=_safe_path(folder,row['folder']);camera=read_json(base/'camera.json')
            rows.append(dict(capture=row['capture'],rgb=base/'rgb.png',depth=base/'depth_m.npy',
                             K=np.asarray(camera['K']),camera=camera,folder=folder))
    else:
        if (folder/'setp1').is_dir():folder=folder/'setp1'
        from tsdf_reconstruction import sdk_frames
        frames,k,_=sdk_frames(folder);rows=[]
        for f in frames:
            image=folder/'rgb_raw'/f['depth'].name
            if not image.is_file():raise ValueError('需要对应原始RGB，不能用不同网格的配准照片替代')
            rows.append(dict(capture=f['capture'],rgb=image,depth=f['depth'],K=k,
                camera=dict(K=k.tolist(),model='sdk_rgb_grid_v1',world_from_sdk=f['pose'].tolist(),
                            source=str(folder),stamp=f['depth'].stem),folder=folder,observation=f['observation']))
        # Use the same audited SDK RGB grid as the existing TSDF experiment.
        for index in sorted({0,len(rows)//2,len(rows)-1}):
            row=rows[index];rgb,depth=load(row)
            obs=row['observation']
            if obs['cloud_source']=='local_surface_v1':
                with np.load(obs['path'],allow_pickle=False) as packet:p=packet['points'].copy()
            else:p=np.load(obs['path'],allow_pickle=False).reshape(-1,3)/1000
            audit_sdk_grid(depth,k,p)
    if not rows:raise ValueError('没有可用RGB-D帧')
    if len({r['capture'] for r in rows})!=len(rows):raise ValueError('帧编号重复')
    for row in rows:
        if not row['rgb'].is_file() or not row['depth'].is_file():raise FileNotFoundError('RGB-D数据不完整')
    return rows


def load(row):
    bgr=cv2.imread(str(row['rgb']),cv2.IMREAD_COLOR)
    if bgr is None:raise ValueError('无法读取RGB：'+str(row['rgb']))
    if row['depth'].suffix=='.npy':depth=np.load(row['depth'],allow_pickle=False).astype(np.float32)
    else:
        depth=cv2.imread(str(row['depth']),cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype!=np.uint16:raise ValueError('需要uint16毫米深度图')
        depth=depth.astype(np.float32)/1000
    if depth.ndim!=2 or bgr.shape[:2]!=depth.shape:raise ValueError('RGB和深度的分辨率不一致')
    depth[~np.isfinite(depth)|(depth<.15)|(depth>4)]=np.nan
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),depth


def audit_sdk_grid(depth,k,points):
    points=np.asarray(points).reshape(-1,3)
    points=points[np.isfinite(points).all(1)&(points[:,2]>=.15)&(points[:,2]<=4)]
    points=points[::max(1,len(points)//10000)]
    u=np.rint(points[:,0]/points[:,2]*k[0,0]+k[0,2]).astype(int);v=np.rint(points[:,1]/points[:,2]*k[1,1]+k[1,2]).astype(int)
    inside=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]);p=points[inside]
    d=depth[v[inside],u[inside]];ok=np.isfinite(d)
    error=abs(d[ok]-p[ok,2])*1000
    if len(error)<100 or np.mean(error<=10)<.9:raise ValueError('SDK深度网格与RGB内参核验失败，停止使用未确认对齐的数据')
    return dict(checked_points=len(error),within10mm=float(np.mean(error<=10)))


def save_frame(folder,rgb,depth,camera,result=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if not cv2.imwrite(str(folder/'rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise OSError('RGB保存失败')
    np.save(folder/'depth_m.npy',np.asarray(depth,np.float32))
    write_json(folder/'camera.json',camera)
    if result is not None:
        np.save(folder/'xyz_opencv.npy',result['xyz'])
        np.save(folder/'xyz_pytorch3d.npy',result['xyz']*np.array([-1,-1,1],np.float32))
        cv2.imwrite(str(folder/'valid_mask.png'),result['valid'].astype(np.uint8)*255)
        write_json(folder/'report.json',result['report'])


def export_sequence(rows,destination,params=None,cancel=None,progress=None,overwrite=False):
    """Atomic export. Native reprocessing replaces only setp2, retaining original setp1."""
    destination=Path(destination).resolve()
    if not rows:raise ValueError('没有可导出的RGB-D帧')
    replacing=destination.exists() and overwrite
    if replacing:
        existing=sources(destination)
        if len(existing)!=len(rows) or any(a['capture']!=b['capture'] or a['rgb'].resolve()!=b['rgb'].resolve() or a['depth'].resolve()!=b['depth'].resolve() for a,b in zip(existing,rows)):
            raise ValueError('仅允许当前点图数据集覆盖自己的二阶段结果')
    else:
        for row in rows:
            if any(Path(row[key]).resolve().is_relative_to(destination) for key in ('rgb','depth')):raise ValueError('不能覆盖原始输入目录')
        if destination.exists():raise FileExistsError('导出目录已存在，请选择新名称：'+str(destination))
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/('.pending_pointmap_'+uuid.uuid4().hex)
    staging.mkdir();records=[];reports=[]
    try:
        for i,row in enumerate(rows):
            check_cancel(cancel);before={str(p):sha(p) for p in (row['rgb'],row['depth'])}
            rgb,depth=load(row);result=clean(rgb,depth,row['K'],params,cancel)
            name=row['depth'].parent.name if replacing else f'frame_{i+1:06d}'
            camera={**row['camera'],'K':np.asarray(row['K']).tolist()}
            if not replacing:save_frame(staging/'setp1'/name,rgb,depth,camera)
            save_frame(staging/'setp2'/name,result['rgb'],result['depth'],camera,result)
            if any(sha(p)!=h for p,h in before.items()):raise RuntimeError('处理期间输入数据发生变化')
            records.append(dict(capture=int(row['capture']),folder='setp1/'+name));reports.append(result['report'])
            if progress:progress((i+1)/len(rows),f'已清洗并保存 {i+1}/{len(rows)} 帧')
        check_cancel(cancel)
        write_json(staging/'pointmap_dataset.json',dict(schema=SCHEMA,frames=records,stages=[1,2],
            geometry_mode='independent RGB-D views; no cross-frame fusion',complete=True))
        write_json(staging/'setp2/report.json',dict(schema=SCHEMA,complete=True,frames=reports,
            input_valid=sum(r['input_valid'] for r in reports),output_valid=sum(r['output_valid'] for r in reports),
            rgb_pixels_unchanged=True,algorithm=reports[0]['algorithm']))
        if not replacing:os.rename(staging,destination)
        else:
            backup=destination/('.previous_pointmap_'+uuid.uuid4().hex)
            target=destination/'setp2'
            if target.exists():target.rename(backup)
            try:
                (staging/'setp2').rename(target)
                manifest=read_json(destination/'pointmap_dataset.json');manifest.update(stages=[1,2],complete=True)
                write_json(staging/'updated_manifest.json',manifest)
                os.replace(staging/'updated_manifest.json',destination/'pointmap_dataset.json')
            except BaseException:
                if target.exists():target.rename(staging/'setp2')
                if backup.exists():backup.rename(target)
                raise
            if backup.exists() and backup.parent==destination and backup.name.startswith('.previous_pointmap_'):shutil.rmtree(backup)
    finally:
        if staging.exists() and staging.parent==destination.parent and staging.name.startswith('.pending_pointmap_'):
            shutil.rmtree(staging)
    return destination
