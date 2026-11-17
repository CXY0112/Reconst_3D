"""SDK RGB-grid depth TSDF using the unchanged saved observation poses."""
from pathlib import Path
import json
import time
import numpy as np
from geometry_inputs import source_info, read_world_points

CAMERA_MODEL = 'sdk_rgb_grid_v1'
DEFAULTS = dict(voxel_mm=5., sdf_trunc_mm=30., depth_trunc_m=4., depth_sampling_stride=1)


def sdk_frames(session):
    from kalman_fusion import session_observations, OPTICAL_FROM_WORLD
    session, observations, fingerprints = session_observations(session)
    calibration = json.loads((session/'sdk_calibration.json').read_text(encoding='utf-8'))
    k = np.array([[calibration['fxrgb'], 0, calibration['cxrgb']],
                  [0, calibration['fyrgb'], calibration['cyrgb']], [0, 0, 1.]], float)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('无效SDK深度网格内参')
    rows = {int(r['capture']):r for r in (json.loads(s) for s in (session/'frames.jsonl').read_text(encoding='utf-8').splitlines() if s.strip())}
    frames = []
    for observation in observations:
        stamp = f'{float(rows[observation["capture"]]["stamp"]):.6f}.png'
        depth = session/'depth_raw'/stamp
        if not depth.is_file():
            raise FileNotFoundError(f'TSDF需要保存的SDK原始深度图：{depth}')
        coordinates=observation['coordinates']
        if coordinates:
            if coordinates['raw_depth_frame']!='rgb' or coordinates['pointcloud_frame']!='rgb':
                raise ValueError('SDK RGB-grid TSDF requires RGB-frame raw depth and local cloud')
            pose=observation['rgb_pose'].copy()
        else:
            # Preserve historical datasets' original P_saved @ E interpretation.
            pose = observation['pose'].copy()
            pose[:3, :3] = OPTICAL_FROM_WORLD.T @ pose[:3, :3]
            pose[:3, 3] = OPTICAL_FROM_WORLD.T @ pose[:3, 3]
        frames.append(dict(capture=observation['capture'], pose=pose, depth=depth, observation=observation))
    return frames, k, fingerprints


def audit_grid(frames, k):
    """Reject silently applying RGB intrinsics to an incompatible SDK depth grid."""
    import open3d as o3d
    records = []
    for i in sorted({0, len(frames)//2, len(frames)-1}):
        f = frames[i]; obs = f['observation']
        raw = np.asarray(o3d.io.read_image(str(f['depth'])), float)/1000
        if raw.ndim != 2 or not np.any(raw > 0):raise ValueError('SDK深度图无有效像素')
        if obs['cloud_source'] == 'local_surface_v1':
            with np.load(obs['path'], allow_pickle=False) as packet:points = packet['points'].astype(float)
        else:points = np.load(obs['path'], allow_pickle=False).reshape(-1, 3).astype(float)/1000
        points = points[np.isfinite(points).all(1) & (points[:, 2] >= .15) & (points[:, 2] <= 4)]
        points = points[::max(1, len(points)//10000)]
        u = np.rint(k[0, 0]*points[:, 0]/points[:, 2]+k[0, 2]).astype(int)
        v = np.rint(k[1, 1]*points[:, 1]/points[:, 2]+k[1, 2]).astype(int)
        inside = (u >= 0) & (u < raw.shape[1]) & (v >= 0) & (v < raw.shape[0])
        observed = raw[v[inside], u[inside]]; z = points[inside, 2]
        valid = (observed >= .15) & (observed <= 4)
        errors = abs(observed[valid]-z[valid])*1000
        fraction = float((errors <= 10).mean()) if len(errors) else 0.
        records.append(dict(capture=f['capture'], checked_points=len(errors), within10mm=fraction,
                            p95_mm=float(np.percentile(errors, 95)) if len(errors) else None))
        if len(errors) < 100 or fraction < .9:
            raise ValueError('SDK点云与depth_raw/RGB内参对应核验失败；停止TSDF，请检查相机配准模式。'+str(records[-1]))
    return records


def run_tsdf(source, output, preview_packet=None):
    import open3d as o3d
    from reconstruction_pipeline import file_hash
    from mesh_processing import distances, scene_for
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    info = source_info(source)
    session = info['session']
    if session is None:raise ValueError('TSDF需要关联到一阶段原始扫描，单独点云不足以重建')
    if output == source or source in output.parents or output == session or session in output.parents:
        raise ValueError('TSDF输出必须在输入目录之外')
    print('1/6 Reading retained SDK RGB-D observations...', flush=True)
    frames, k, fingerprints = sdk_frames(session)
    files = {f'session/{name}':session/name for name in fingerprints}
    for f in frames:files['session/depth_raw/'+f['depth'].name] = f['depth']
    for index in sorted({0, len(frames)//2, len(frames)-1}):
        path = frames[index]['observation']['path']; files['session/'+path.relative_to(session).as_posix()] = path
    if (source/'report.json').is_file():files['input/report.json'] = source/'report.json'
    raw_info = source_info(session)
    files['reference_cloud'] = raw_info['cloud']
    hashes = {name:file_hash(path) for name, path in files.items()}
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=2, stage=3, method='tsdf', pipeline='sdk_depth_tsdf',
        input=str(source), source_session=str(session), input_kind='sdk_depth_observations', preprocessing_used=False,
        camera_model=CAMERA_MODEL, camera_model_note='K_rgb grid validated against SDK XYZ; explicit v2 RGB clouds use P_saved; unlabelled legacy uses P_saved@E; no source poses changed',
        parameters=DEFAULTS, input_sha256=hashes, complete=False, stage_seconds={}, decimation_attempts=[],
        units='metres', coordinates='world X forward Y left Z up', poses_modified=False,
        notes='Original retained SDK depth only; no stage-two cloud input, Poisson, smoothing, decimation, generated completion or texture. Cloud coverage is observation consistency, not ground-truth accuracy.')
    def save(): (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        print('2/6 Validating SDK point/depth projection...', flush=True)
        report['grid_audit'] = audit_grid(frames, k)
        report['stage_seconds']['load_and_validate'] = time.perf_counter()-started
        stamp = time.perf_counter()
        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.005, sdf_trunc=.030,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor, depth_sampling_stride=1)
        intrinsic = None; shape = None; total_pixels = 0
        print('3/6 Integrating original SDK depth into TSDF...', flush=True)
        for index, f in enumerate(frames):
            raw = np.asarray(o3d.io.read_image(str(f['depth'])))
            if raw.ndim != 2 or raw.dtype != np.uint16:raise ValueError('SDK深度必须为uint16毫米深度图')
            if intrinsic is None:
                shape = raw.shape
                intrinsic = o3d.camera.PinholeCameraIntrinsic(shape[1], shape[0], k[0, 0], k[1, 1], k[0, 2], k[1, 2])
            if raw.shape != shape:raise ValueError('同一扫描中深度图尺寸不一致')
            raw = np.where((raw >= 150) & (raw <= 4000), raw, 0).astype(np.uint16)
            total_pixels += int(np.count_nonzero(raw))
            rgb = o3d.geometry.Image(np.zeros((*shape, 3), np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, o3d.geometry.Image(raw),
                depth_scale=1000., depth_trunc=4., convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(f['pose']))
            print(f'RGB {index+1}/{len(frames)} TSDF depth', flush=True)
        report['stage_seconds']['tsdf_integration'] = time.perf_counter()-stamp
        report.update(captures=len(frames), valid_depth_samples=total_pixels)
        stamp = time.perf_counter(); print('4/6 Extracting native TSDF mesh...', flush=True)
        mesh = volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
        if not len(mesh.triangles) or not np.isfinite(np.asarray(mesh.vertices)).all():raise ValueError('TSDF没有生成有效网格')
        del volume
        report['stage_seconds']['mesh_extraction'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('5/6 Measuring against the original fused map...', flush=True)
        points = read_world_points(raw_info)
        error = distances(scene_for(mesh), points)
        report['input_points'] = len(points)
        report['metric_reference'] = 'original stage-one fused world cloud; not stage-two output'
        report['final'] = dict(vertices=len(mesh.vertices), triangles=len(mesh.triangles),
            input_coverage={str(mm):float((error <= mm).mean()) for mm in (5, 10, 15)}, input_to_mesh_mm_p95=float(np.percentile(error, 95)))
        report['stage_seconds']['reference_measurement'] = time.perf_counter()-stamp
        print('6/6 Verifying inputs and saving mesh...', flush=True)
        if hashes != {name:file_hash(path) for name, path in files.items()}:raise RuntimeError('TSDF输入在重建期间发生变化')
        if not o3d.io.write_triangle_mesh(str(output/'mesh.ply'), mesh, write_vertex_colors=False):raise OSError('TSDF网格保存失败')
        if preview_packet is not None:
            np.savez(preview_packet, vertices=np.asarray(mesh.vertices, np.float32), triangles=np.asarray(mesh.triangles, np.uint32), normals=np.asarray(mesh.vertex_normals, np.float32))
        report['final']['mesh_sha256'] = file_hash(output/'mesh.ply')
        report.update(complete=True, source_unchanged=True, elapsed_seconds=time.perf_counter()-started)
        save();return report
    except Exception as exc:
        report.update(error=str(exc), elapsed_seconds=time.perf_counter()-started);save();raise
