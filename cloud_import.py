from runtime_paths import geometry_python
"""Read existing clouds without mutating files, inferring only explicit metadata."""
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent


@dataclass
class ImportedCloud:
    points: np.ndarray
    unit: str
    coordinates: str
    removed: int


def _metadata(path):
    unit,coordinates='m','world'
    if path.parent.name=='pointcloud_raw':
        return 'mm','camera'
    if path.name=='full_geometry_world.npy':
        return 'm','world'
    if path.suffix.lower()=='.ply':
        with path.open('rb') as stream:
            header=stream.read(65536).split(b'end_header',1)[0].decode('ascii',errors='ignore').lower()
        if 'x right y down z forward' in header or 'x right, y down, z forward' in header:
            coordinates='camera'
        if 'rtab-map world' in header:
            coordinates='world'
        if 'millimet' in header or 'units: mm' in header or 'unit: mm' in header:
            unit='mm'
    return unit,coordinates


def _array(points):
    points=np.asarray(points)
    if points.dtype.kind not in 'fiu' or points.ndim not in (1,2,3):
        raise ValueError('需要数值 XYZ 数组，形状为 N×3 或 H×W×3')
    if points.ndim==1:
        points=points.reshape(1,-1)
    if points.shape[-1]<3:
        raise ValueError('点云数组至少需要 X、Y、Z 三列')
    return np.asarray(points.reshape(-1,points.shape[-1])[:,:3],dtype=float)


def load_cloud(path,unit='auto',coordinates='auto'):
    path=Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'文件不存在：{path}')
    if unit not in ('auto','m','mm') or coordinates not in ('auto','world','camera'):
        raise ValueError('无效单位或坐标选项')
    detected_unit,detected_coordinates=_metadata(path)
    extension=path.suffix.lower()
    if extension=='.npy':
        points=np.load(path,allow_pickle=False)
    elif extension=='.npz':
        with np.load(path,allow_pickle=False) as archive:
            key=next((k for k in ('points_mm','points_world','points','point_cloud','xyz','arr_0') if k in archive),None)
            if key is None:
                raise ValueError('NPZ 中未找到 points_mm、points_world、points、point_cloud、xyz 或 arr_0 数组')
            points=archive[key].copy()
            if key=='points_mm':
                detected_unit,detected_coordinates='mm','camera'
            if key=='points_world':
                detected_coordinates='world'
            if 'unit' in archive:
                value=archive['unit'].item()
                value=value.decode() if isinstance(value,bytes) else str(value)
                if value in ('m','mm'):
                    detected_unit=value
            if 'coordinate_frame' in archive:
                value=archive['coordinate_frame'].item()
                value=value.decode() if isinstance(value,bytes) else str(value)
                if value in ('world','camera'):
                    detected_coordinates=value
    elif extension in ('.ply','.pcd'):
        (ROOT/'results').mkdir(exist_ok=True)
        # Temporary output is owned by this import and remains inside the workspace.
        with tempfile.TemporaryDirectory(prefix='.cloud_import_',dir=ROOT/'results') as folder:
            target=Path(folder)/'points.npy'
            env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
            result=subprocess.run([str(geometry_python()),str(ROOT/'cloud_read_worker.py'),str(path),str(target)],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',
                                   env=env,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode or not target.exists():
                raise ValueError('无法读取点云：'+(result.stderr.strip()[-800:] or '文件为空或格式无效'))
            points=np.load(target,allow_pickle=False)
    elif extension in ('.xyz','.txt','.csv'):
        delimiter=',' if extension=='.csv' else None
        with path.open('r',encoding='utf-8-sig') as stream:
            first=stream.readline().strip().lower().replace(',',' ').split()
        skip=1 if len(first)>=3 and first[:3]==['x','y','z'] else 0
        points=np.loadtxt(path,delimiter=delimiter,skiprows=skip,ndmin=2,encoding='utf-8-sig')
    else:
        raise ValueError('支持 PLY、PCD、NPY、NPZ、XYZ、TXT、CSV 点云文件')
    points=_array(points)
    valid=np.isfinite(points).all(axis=1)
    removed=int((~valid).sum()); points=points[valid]
    if not len(points):
        raise ValueError('文件中没有有效的 XYZ 点')
    unit=detected_unit if unit=='auto' else unit
    coordinates=detected_coordinates if coordinates=='auto' else coordinates
    if unit=='mm':
        points=points/1000
    if coordinates=='world':
        points=np.column_stack((-points[:,1],-points[:,2],points[:,0]))
    return ImportedCloud(np.ascontiguousarray(points,dtype=np.float32),unit,coordinates,removed)
