"""Resolve processed or explicitly skipped stage two without copying the raw map."""
from pathlib import Path
import json
import numpy as np
from saved_sources import metadata, reference


def source_info(source):
    source = Path(source).resolve()
    report = metadata(source)
    if (source/'processed_world.npy').is_file() and (source/'surfel_attributes.npz').is_file():
        return dict(kind='preprocessed', folder=source, cloud=source/'processed_world.npy',
                    session=reference(report.get('source_session'), source), report=report)
    if report.get('complete') and report.get('skipped') and report.get('backend') == 'skip':
        session = reference(report.get('source_session'), source)
    elif (source/'geometry_poses.json').is_file():
        session = source
    else:
        raise ValueError('请选择二阶段结果、跳过记录，或完整的一阶段扫描目录')
    if session is None:
        raise FileNotFoundError('找不到跳过预处理所引用的一阶段扫描目录')
    cloud = next((session/n for n in ('full_geometry_world.npy', 'map_cloud.ply') if (session/n).is_file()), None)
    if cloud is None:
        raise FileNotFoundError('源扫描缺少原始融合点云')
    return dict(kind='raw', folder=source, cloud=cloud, session=session, report=report)


def read_world_points(info):
    path = info['cloud']
    if path.suffix.lower() == '.npy':
        points = np.load(path, allow_pickle=False).astype(float)
    else:
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or len(points) < 200:
        raise ValueError('需要至少200个有效的米制世界坐标点')
    return points


def save_skip(source, output, point_count, captures=None):
    """Only a provenance record is saved; no renamed/copy 'processed' point cloud."""
    source = Path(source).resolve()
    session = source if source.is_dir() else source.parent
    if not (session/'geometry_poses.json').is_file() or not (session/'map_cloud.ply').is_file():
        raise ValueError('跳过预处理需要完整的一阶段扫描目录')
    if captures is None:
        captures = len(json.loads((session/'geometry_poses.json').read_text(encoding='utf-8'))['node_ids'])
    output = Path(output).resolve()
    if output == session or session in output.parents:
        raise ValueError('跳过记录必须保存在源扫描目录之外')
    from reconstruction_pipeline import file_hash
    report = dict(schema_version=2, stage=2, backend='skip', skipped=True, complete=True,
        source_session=str(session), source=str(session/'map_cloud.ply'), input_points=int(point_count),
        output_points=int(point_count), captures=int(captures), poses_modified=False,
        source_sha256={'geometry_poses.json':file_hash(session/'geometry_poses.json')},
        notes='Stage two explicitly bypassed. Raw map is referenced, not copied or filtered. Poisson estimates normals; SDK TSDF uses original depth observations.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def poisson_source(source):
    """Raw mode estimates/orients normals only; points are neither moved nor sampled."""
    info = source_info(source)
    points = read_world_points(info)
    files = [info['cloud']]
    if (info['folder']/'report.json').is_file():files.append(info['folder']/'report.json')
    if info['kind'] == 'preprocessed':
        files.append(info['folder']/'surfel_attributes.npz')
        with np.load(files[-1], allow_pickle=False) as data:
            normals = data['normals_world'].astype(float)
        normal_info = dict(method='saved stage-two surfel normals')
    else:
        import open3d as o3d
        from scipy.spatial import cKDTree
        from kalman_fusion import session_observations, OPTICAL_FROM_WORLD
        session, observations, _ = session_observations(info['session'])
        files += [session/'geometry_poses.json', session/'sdk_calibration.json', session/'frames.jsonl', session/'registrations.jsonl']
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.030, max_nn=30))
        normals = np.asarray(cloud.normals).copy()
        cameras = np.array([o['pose'][:3, 3] @ OPTICAL_FROM_WORLD for o in observations])
        _, closest = cKDTree(cameras).query(points, workers=2)
        flip = np.sum(normals*(cameras[closest]-points), axis=1) < 0
        normals[flip] *= -1
        normal_info = dict(method='Open3D PCA normals, oriented towards nearest retained camera',
                           radius_mm=30., max_nn=30, points_resampled=False, points_moved=False,
                           note='Nearest camera is a heuristic orientation, not a new registration or denoising step.')
    if normals.shape != points.shape:
        raise ValueError('点云与法线数量不一致')
    return info, points, normals, files, normal_info
