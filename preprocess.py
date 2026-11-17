"""Offline geometric baseline. Input is metres in optical display axes; poses stay fixed."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from cloud_import import ROOT, load_cloud
from reconstruct import save_cloud
from voxel_map import voxel_groups


class ProcessingCancelled(Exception):
    pass


@dataclass(frozen=True)
class Parameters:
    noise_mm: float = 10.
    radius_mm: float = 40.
    output_voxel_mm: float = 2.
    neighbors: int = 48
    max_displacement_mm: float = 10.
    normal_angle_deg: float = 25.
    layer_gap_mm: float = 6.
    isolation_factor: float = 3.
    batch_size: int = 8192
    kalman_floor_mm: float = 2.
    kalman_sampling: str = 'surface_uniform'
    surface_spacing_mm: float = 2.

    def validate(self):
        for key, value in asdict(self).items():
            if key=='kalman_sampling':continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} 必须为正数')
        if not 12 <= self.neighbors <= 128 or int(self.neighbors) != self.neighbors:
            raise ValueError('邻居数需要在 12–128 之间')
        if not 1 <= self.batch_size or int(self.batch_size) != self.batch_size:
            raise ValueError('batch_size 必须为正整数')
        if self.normal_angle_deg >= 90:
            raise ValueError('法线角阈值需要小于90度')
        if self.kalman_sampling not in ('voxel','surface_uniform'):
            raise ValueError('无效卡尔曼输出采样方式')


@dataclass
class ProcessingResult:
    points: np.ndarray
    removed: np.ndarray
    retained_indices: np.ndarray
    movement_mm: np.ndarray
    report: dict
    diagnostics: dict | None = None


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise ProcessingCancelled('已取消处理')


def _notify(progress, fraction, text):
    if progress:
        progress(float(fraction), text)


def _query(tree, points, k):
    return tree.query(points, k=k, workers=2)


def _fit(patch, weights):
    total = np.maximum(weights.sum(axis=1), 1e-12)
    center = np.einsum('nk,nki->ni', weights, patch)/total[:, None]
    delta = patch-center[:, None]
    covariance = np.einsum('nk,nki,nkj->nij', weights, delta, delta)/total[:, None, None]
    eigen, vectors = np.linalg.eigh(covariance)
    return center, eigen, vectors[:, :, 0]


def _geometry(points, spacing, params, progress, start, length, cancel):
    tree = cKDTree(points)
    n = len(points); k = min(params.neighbors+1, n)
    normals = np.zeros((n, 3), np.float32)
    valid = np.zeros(n, bool); boundary = np.ones(n, bool)
    radius = params.radius_mm/1000
    for a in range(0, n, params.batch_size):
        _check(cancel)
        b = min(n, a+params.batch_size)
        d, ids = _query(tree, points[a:b], k)
        patch = points[ids].astype(float)
        adaptive = np.minimum(np.maximum(radius, spacing[a:b]*3), radius*2)
        w = np.exp(-.5*(d/np.maximum(adaptive[:, None]*.55, 1e-9))**2)*(d<=adaptive[:, None])
        # Inverse surface sampling density; bounded to avoid sparse points dominating.
        density = spacing[ids]**2
        reference = np.maximum(np.median(density, axis=1), 1e-12)
        w *= np.clip(density/reference[:, None], .25, 4.)
        center, ev, normal = _fit(patch, w)
        h = np.einsum('nki,ni->nk', patch-center[:, None], normal)
        # Robust weighted plane fitting, rather than unconstrained coordinate averaging.
        robust = w/np.sqrt(1+(h/(params.noise_mm/1000))**2)
        center, ev, normal = _fit(patch, robust)
        good = ((d<=adaptive[:, None]).sum(1)>=12)&(ev[:, 1]>1e-7)&(ev[:, 0]<.25*ev[:, 1])
        tangent = center-points[a:b]
        tangent -= np.sum(tangent*normal, axis=1)[:, None]*normal
        edge = np.linalg.norm(tangent, axis=1)>.45*np.sqrt(np.maximum(ev[:, 1]+ev[:, 2], 1e-12))
        normals[a:b] = normal; valid[a:b] = good; boundary[a:b] = edge
        _notify(progress, start+length*b/n, '估计局部表面和边界')
    return tree, normals, valid, boundary




def quality(points, cancel=None, sample_count=5000):
    """Local shape proxies, not ground-truth accuracy or cross-frame registration error."""
    _check(cancel)
    if len(points)<12:
        return {'samples':0}
    rng = np.random.default_rng(20261015)
    sample = points[rng.choice(len(points), min(len(points), sample_count), replace=False)]
    d, ids = _query(cKDTree(points), sample, min(25, len(points)))
    patch = points[ids].astype(float); delta = patch-patch.mean(1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum('nki,nkj->nij', delta, delta)/patch.shape[1])
    planar = (ev[:, 1]>1e-7)&(ev[:, 0]<.25*ev[:, 1])&(d[:, -1]<.08)
    sigma = np.sqrt(np.maximum(ev[planar, 0], 0))*1000
    return {'samples':len(sample), 'planar_samples':int(planar.sum()),
            'neighbor_mm_p10_p50_p90':np.percentile(d[:, 1]*1000, [10, 50, 90]).tolist(),
            'planar_sigma_mm_p50_p90':np.percentile(sigma, [50, 90]).tolist() if len(sigma) else []}




def process_input(source, points, params=None, backend='kalman', progress=None, cancel=None):
    if backend != 'kalman':
        raise ValueError('当前版本仅保留卡尔曼表面融合')
    from kalman_fusion import process_session
    return process_session(source,params,before_points=points,progress=progress,cancel=cancel)


def load_input(path, unit='auto', coordinates='auto'):
    path = Path(path).resolve()
    if path.is_dir():
        path = path/'map_cloud.ply'
    return path, load_cloud(path, unit, coordinates)


def save_result(result, source, destination=None, metadata=None):
    from result_paths import stage_result_path
    source = Path(source).resolve()
    destination = Path(destination).resolve() if destination else stage_result_path(2)
    scan_source = source.name=='map_cloud.ply' or (source.parent/'map.db').is_file() or (source.parent/'geometry_poses.json').is_file()
    if source.parent==destination or (scan_source and source.parent in destination.parents):
        raise ValueError('请保存到源扫描目录之外的新目录')
    destination.mkdir(parents=True, exist_ok=False)
    world = lambda p:np.column_stack((p[:, 2], -p[:, 0], -p[:, 1]))
    # Write the success manifest last; an interrupted export is never marked complete.
    save_cloud(destination/'processed.ply', world(result.points), 'RTAB-Map world XYZ in metres, X forward Y left Z up')
    np.save(destination/'processed_world.npy', world(result.points))
    np.save(destination/'removed_world.npy', world(result.removed))
    np.save(destination/'retained_loaded_indices.npy', result.retained_indices)
    np.save(destination/'movement_mm.npy', result.movement_mm)
    if result.diagnostics is not None:
        attributes={key:value for key,value in result.diagnostics.items() if key!='normals'}
        if 'normals' in result.diagnostics:
            attributes['normals_world']=world(result.diagnostics['normals'])
        np.savez_compressed(destination/'surfel_attributes.npz',**attributes)
    report = {**result.report, 'stage':2, 'source':str(source), 'source_metadata':metadata or {},
              'saved_coordinates':'world: X forward Y left Z up', 'saved_units':'m',
              'index_note':result.report.get('index_note','retained_loaded_indices 对应 load_cloud 后的有效点数组；movement_mm 与之逐项对应，均为最终体素采样前。'),
              'complete':True}
    (destination/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return destination





