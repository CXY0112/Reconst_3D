"""Read Brush Gaussian PLY and project in the saved RGB camera coordinate system.

No positions, rotations, scales or SH coefficients are rewritten on disk.
Projection follows Brush v0.3.0's pinhole Jacobian and 0.3 pixel covariance blur.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


class GaussianScene:
    def __init__(self, path):
        props=[];count=0
        with Path(path).open('rb') as stream:
            if stream.readline().strip()!=b'ply':raise ValueError('无效的高斯 PLY')
            for _ in range(256):
                line=stream.readline(1024).decode('ascii').strip()
                if line.startswith('format ') and line!='format binary_little_endian 1.0':
                    raise ValueError('需要 binary_little_endian 高斯 PLY')
                if line.startswith('element vertex '):count=int(line.split()[-1])
                if line.startswith('property '):
                    parts=line.split()
                    if len(parts)!=3 or parts[1]!='float':raise ValueError('不支持的高斯属性类型')
                    props.append(parts[2])
                if line=='end_header':break
            else:raise ValueError('高斯 PLY 头部不完整')
            data=np.fromfile(stream,dtype='<f4',count=count*len(props))
        if count<=0 or data.size!=count*len(props) or not np.isfinite(data).all():raise ValueError('高斯数据不完整或包含非有限值')
        data=data.reshape(count,len(props))
        def columns(names):return data[:,[props.index(n) for n in names]]
        self.means=columns(['x','y','z'])
        scales=np.exp(columns([f'scale_{i}' for i in range(3)]).astype(np.float64))
        quats=columns([f'rot_{i}' for i in range(4)])
        valid=np.linalg.norm(quats,axis=1)>1e-6
        # Zero rotations are invisible in Brush; keep them invisible here too.
        quats[~valid]=[1,0,0,0]
        rot=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        transform=rot*scales[:,None,:]
        self.covariance=(transform@transform.transpose(0,2,1)).astype(np.float32)
        if not np.isfinite(self.covariance).all():raise ValueError('高斯尺度超出有效范围')
        self.opacity=(1/(1+np.exp(-np.clip(columns(['opacity'])[:,0],-80,80))))*valid
        self.dc=columns([f'f_dc_{i}' for i in range(3)])
        rest=sorted((p for p in props if p.startswith('f_rest_')),key=lambda x:int(x[7:]))
        if len(rest) not in (0,9):raise ValueError('当前场景查看支持项目生成的 SH0 / SH1 高斯')
        # Standard Gaussian PLY stores channel-major SH coefficients.
        self.rest=columns(rest).reshape(count,3,-1).transpose(0,2,1) if rest else None

    def project(self, pose, intrinsics, width, height):
        """Return screen means, conics, opacity, colors, bounds, back-to-front."""
        basis=pose[:3,:3];position=pose[:3,3]
        p=(self.means-position)@basis
        ids=np.flatnonzero((p[:,2]>.01)&(self.opacity>=1/255))
        p=p[ids];z=p[:,2];focal=np.asarray(intrinsics[:2]);center=np.asarray(intrinsics[2:])
        uv=p[:,:2]/z[:,None];means=uv*focal+center
        clipped=np.clip(uv,(-.15*np.array([width,height])-center)/focal,
                        (1.15*np.array([width,height])-center)/focal)
        jac=np.zeros((len(p),2,3),np.float32)
        jac[:,0,0]=focal[0]/z;jac[:,1,1]=focal[1]/z
        jac[:,:,2]=-clipped*focal/z[:,None]
        world_j=jac@basis.T
        cov=world_j@self.covariance[ids]@world_j.transpose(0,2,1)
        cov[:,0,0]+=.3;cov[:,1,1]+=.3
        det=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2
        conic=np.column_stack((cov[:,1,1],-cov[:,0,1],cov[:,0,0]))/np.maximum(det[:,None],1e-24)
        opacity=self.opacity[ids]
        extent=np.sqrt(2*np.log(255*opacity[:,None])*np.diagonal(cov,axis1=1,axis2=2))
        visible=(det>1e-24)&(means[:,0]+extent[:,0]>0)&(means[:,0]-extent[:,0]<width)&(means[:,1]+extent[:,1]>0)&(means[:,1]-extent[:,1]<height)
        order=np.flatnonzero(visible);order=order[np.argsort(-z[order],kind='stable')]
        selected=ids[order]
        color=.5+.28209479177387814*self.dc[selected]
        if self.rest is not None:
            direction=self.means[selected]-position
            direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
            sh=direction[:,[1,2,0]]*[-.4886025119029199,.4886025119029199,-.4886025119029199]
            color+=np.sum(self.rest[selected]*sh[:,:,None],axis=1)
        return means[order],conic[order],opacity[order],np.maximum(color,0),extent[order]


def saved_cameras(model):
    """Nerfstudio OpenGL c2w -> OpenCV c2w, exactly once; preserve world Z-up."""
    path=Path(model).parent/'dataset/transforms.json'
    if not path.is_file():raise ValueError('缺少 dataset/transforms.json，无法恢复训练相机视角')
    data=json.loads(path.read_text(encoding='utf-8'))
    cameras=[]
    for frame in data['frames']:
        merged={**data,**frame}
        pose=np.asarray(frame['transform_matrix'],dtype=float)@np.diag([1.,-1.,-1.,1.])
        intr=np.array([merged[k] for k in ('fl_x','fl_y','cx','cy')],float)
        w,h=int(merged['w']),int(merged['h'])
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(pose[:3,:3])<.99:
            raise ValueError('无效的训练相机位姿')
        if min(w,h,*intr[:2])<=0 or not np.isfinite(intr).all():raise ValueError('无效的训练相机内参')
        cameras.append(dict(name=Path(frame['file_path']).name,pose=pose,intrinsics=intr,size=(w,h)))
    if not cameras:raise ValueError('未保存训练相机视角')
    return cameras
