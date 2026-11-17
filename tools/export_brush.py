"""Export retained, rectified RGB + optimized optical c2w poses to Nerfstudio.

OpenCV camera basis -> OpenGL camera basis: c2w_gl = c2w_cv @ diag(1,-1,-1,1).
Sparse RGB-D initialization uses training views only for the requested split.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import cv2
import numpy as np


BASIS=np.diag([1.,-1.,-1.,1.])


def export(source,output,eval_every=5,points_per_view=500,cancel=None):
    from preprocess import _check
    _check(cancel)
    source=Path(source).resolve();output=Path(output).resolve()
    if eval_every<2 or points_per_view<1:raise ValueError('Invalid export sampling/split')
    if output.exists():raise FileExistsError(output)
    graph=json.loads((source/'geometry_poses.json').read_text())
    if len(graph['node_ids'])!=len(graph['poses']):raise ValueError('Pose/node count mismatch')
    frames={r['capture']:r['stamp'] for r in map(json.loads,(source/'frames.jsonl').read_text().splitlines())}
    captures={r['node']:r['capture'] for r in map(json.loads,(source/'registrations.jsonl').read_text(encoding='utf-8').splitlines()) if r.get('accepted')}
    c=json.loads((source/'sdk_calibration.json').read_text());fx,fy,cx,cy=[c[k] for k in ['fxrgb','fyrgb','cxrgb','cyrgb']]
    output.mkdir(parents=True);(output/'images').mkdir()
    exported=[];points=[];colors=[];hashes={};max_error=0.;shape=None
    for i,(node,matrix) in enumerate(zip(graph['node_ids'],graph['poses'])):
        _check(cancel)
        pose=np.asarray(matrix,float)
        if (pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1])
                or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(pose[:3,:3])<.99):raise ValueError('Invalid pose')
        name=f'{frames[captures[node]]:.6f}.png';path=source/'rgb'/name;image=cv2.imread(str(path))
        if image is None:raise ValueError('缺少可读取的采集RGB图像：'+str(path))
        h,w=image.shape[:2]
        if shape is not None and shape!=(h,w):raise ValueError('Mixed image resolutions')
        shape=(h,w);shutil.copy2(path,output/'images'/name);hashes['rgb/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
        gl=pose@BASIS;exported.append(dict(file_path='images/'+name,transform_matrix=gl.tolist()))
        # Audit the exported pose convention on physical camera-frame points.
        cloud=np.load(source/'map_frames'/f'{node}.npy');cloud=cloud[np.isfinite(cloud).all(1)&(cloud[:,2]>.1)]
        if not len(cloud):continue
        selected=cloud[np.linspace(0,len(cloud)-1,min(len(cloud),points_per_view),dtype=int)]
        world=selected@pose[:3,:3].T+pose[:3,3]
        reconstructed=(world-gl[:3,3])@gl[:3,:3]@BASIS[:3,:3]
        max_error=max(max_error,float(np.max(np.abs(selected-reconstructed))))
        if i%eval_every==0:continue
        u=np.rint(selected[:,0]/selected[:,2]*fx+cx).astype(int);v=np.rint(selected[:,1]/selected[:,2]*fy+cy).astype(int)
        valid=(u>=0)&(u<w)&(v>=0)&(v<h)
        depth=cv2.imread(str(source/'depth'/name),-1)
        if depth is None or depth.shape!=(h,w):raise ValueError('缺少与RGB同网格的配准深度图')
        z=depth[np.clip(v,0,h-1),np.clip(u,0,w-1)]/1000
        valid&=(z>0)&(np.abs(z-selected[:,2])<=.03)
        points.append(world[valid]);colors.append(image[v[valid],u[valid],::-1])
    if not points or not any(len(p) for p in points):raise ValueError('No visible training-view initialization points')
    xyz=np.concatenate(points);rgb=np.concatenate(colors)
    _,ids=np.unique(np.floor(xyz/.02).astype(np.int64),axis=0,return_index=True);xyz=xyz[ids];rgb=rgb[ids]
    with (output/'initial.ply').open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(stream,np.column_stack((xyz,rgb)),fmt=['%.7f']*3+['%d']*3)
    transforms=dict(camera_model='OPENCV',fl_x=fx,fl_y=fy,cx=cx,cy=cy,w=w,h=h,k1=0,k2=0,p1=0,p2=0,ply_file_path='initial.ply',frames=exported)
    (output/'transforms.json').write_text(json.dumps(transforms,indent=2))
    report=dict(source=str(source),frames=len(exported),train_views=sum(i%eval_every!=0 for i in range(len(exported))),eval_views=sum(i%eval_every==0 for i in range(len(exported))),eval_split_every=eval_every,initial_points=len(xyz),initialization='training-view RGB-D samples only',max_coordinate_roundtrip_error_m=max_error,images_sha256=hashes,coordinates='original robot-world, OpenGL camera c2w; metres')
    (output/'export_report.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=export(args.source,args.output);print(json.dumps({k:v for k,v in report.items() if k!='images_sha256'},indent=2))
