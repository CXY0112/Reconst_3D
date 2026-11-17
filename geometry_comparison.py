"""Read-only Poisson/NKSR pilot; independent from the production stage-three output."""
from pathlib import Path
import json
import time
import numpy as np


def nksr_scale_options(voxel_mm=None, detail_level=None):
    """Select exactly one scale control; NKSR's accepted default is detail_level=0.4."""
    if voxel_mm is None and detail_level is None: detail_level = .4
    if detail_level is not None:
        if voxel_mm is not None: raise ValueError('Choose detail_level or voxel_size, not both')
        if not np.isfinite(detail_level) or not 0 <= detail_level <= 1:
            raise ValueError('Detail level must be between 0 and 1')
        return {'detail_level': float(detail_level)}
    if not np.isfinite(voxel_mm) or voxel_mm <= 0: raise ValueError('Voxel size must be positive')
    return {'voxel_size': float(voxel_mm)/1000}


def load_source(source):
    from reconstruction_pipeline import file_hash
    source = Path(source).resolve()
    hashes = {name: file_hash(source/name) for name in ('processed_world.npy', 'surfel_attributes.npz', 'report.json')}
    points = np.load(source/'processed_world.npy', allow_pickle=False).astype(np.float64)
    with np.load(source/'surfel_attributes.npz', allow_pickle=False) as data:
        normals = data['normals_world'].astype(np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 200 or normals.shape != points.shape:
        raise ValueError('Need at least 200 points with matching normals')
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(points).all() or not np.isfinite(normals).all() or (lengths < .5).any():
        raise ValueError('Invalid source points or normals')
    return points, normals/lengths[:, None], hashes


def quantiles(values_mm):
    return dict(zip(('p50', 'p90', 'p95', 'p99', 'max'),
                    map(float, np.percentile(values_mm, [50, 90, 95, 99, 100]))))


def measure(mesh, points, tree):
    from mesh_processing import distances, scene_for, sampled_surface
    inward = distances(scene_for(mesh), points)
    outward = tree.query(sampled_surface(mesh), workers=2)[0]*1000
    triangles = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    _, components, _ = mesh.cluster_connected_triangles()
    return {'vertices': len(mesh.vertices), 'triangles': len(triangles),
            'area_m2': mesh.get_surface_area(), 'boundary_edges': int((counts == 1).sum()),
            'nonmanifold_edges': int((counts > 2).sum()), 'components': len(components),
            'input_to_mesh_mm': quantiles(inward), 'surface_to_input_mm': quantiles(outward),
            'input_coverage': {str(mm): float((inward <= mm).mean()) for mm in (5, 10, 15)},
            'sampled_surface_beyond_15mm_fraction': float((outward > 15).mean())}


def save_method(name, mesh, points, tree, output, seconds):
    import open3d as o3d
    mesh.remove_duplicated_triangles(); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles): raise ValueError(f'{name}: empty mesh')
    if not np.isfinite(np.asarray(mesh.vertices)).all(): raise ValueError(f'{name}: nonfinite vertices')
    folder = output/name; folder.mkdir()
    entry = {'status': 'complete', 'reconstruction_seconds': seconds, 'variants': {}}
    for variant in ('raw', 'supported'):
        if variant == 'supported':
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            vd = tree.query(vertices, workers=2)[0]
            cd = tree.query(vertices[faces].mean(axis=1), workers=2)[0]
            mesh.remove_triangles_by_mask((vd[faces].max(axis=1) > .015) | (cd > .015))
            mesh.remove_unreferenced_vertices()
            if not len(mesh.triangles): raise ValueError(f'{name}: support trim produced empty mesh')
        mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(folder/(variant+'.ply')), mesh, write_vertex_colors=False):
            raise OSError('Could not export comparison mesh')
        # Host UI does not need Open3D or the inference runtime to view results.
        def display(x): return np.ascontiguousarray(np.column_stack((-x[:, 1], -x[:, 2], x[:, 0])), dtype=np.float32)
        np.savez(folder/(variant+'.npz'), vertices=display(np.asarray(mesh.vertices)),
                 triangles=np.asarray(mesh.triangles, dtype=np.uint32), normals=display(np.asarray(mesh.vertex_normals)))
        entry['variants'][variant] = measure(mesh, points, tree)
    return entry


def run_comparison(source, output, nksr_python, voxel_mm=None, device='auto', detail_level=None):
    import os
    import subprocess
    import sys
    import traceback
    import open3d as o3d
    from scipy.spatial import cKDTree
    from reconstruction_pipeline import file_hash
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents: raise ValueError('Output must be outside stage-two input')
    scale = nksr_scale_options(voxel_mm, detail_level)
    selected_voxel_mm = scale['voxel_size']*1000 if 'voxel_size' in scale else None
    detail_level = scale.get('detail_level')
    points, normals, hashes = load_source(source)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'experiment': 'poisson_nksr', 'input': str(source),
              'input_points': len(points), 'input_sha256': hashes, 'units': 'metres',
              'coordinates': 'world X forward Y left Z up', 'complete': False,
              'parameters': {'poisson_depth': 9, 'poisson_scale': 1.05, 'nksr_voxel_mm': selected_voxel_mm,
                             'nksr_detail_level': scale.get('detail_level'), 'nksr_scale_mode': next(iter(scale)),
                             'nksr_mise_iter': 1, 'nksr_config': 'ks',
                             'support_mm': 15, 'surface_metric_samples': 100000, 'seed': 20261015,
                             'post_smoothing': False, 'decimation': False}, 'methods': {},
              'metric_note': 'Distances measure agreement with noisy input, not ground-truth accuracy.'}
    def save():
        (output/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        lines = ['# Poisson / NKSR 几何对照', '', f'输入：{source}（{len(points):,}点）', '',
                 f'NKSR尺度设置：{scale}；mise_iter=1', '',
                 '完整对照：'+('已完成' if report['complete'] else '尚未完成，不能判断NKSR质量'), '',
                 '| 算法/版本 | 三角面 | 输入→网格P50/P95(mm) | 10mm覆盖 | 表面采样超15mm |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for method, item in report['methods'].items():
            for variant, m in item.get('variants', {}).items():
                q = m['input_to_mesh_mm']
                lines.append(f'| {method}/{variant} | {m["triangles"]:,} | {q["p50"]:.2f}/{q["p95"]:.2f} | {m["input_coverage"]["10"]:.2%} | {m["sampled_surface_beyond_15mm_fraction"]:.2%} |')
            if item.get('status') != 'complete': lines.extend(['', f'{method}：{item["status"]}，参阅nksr_environment.log。'])
        lines.extend(['', '距离表示与有噪声输入的一致性，不是绝对精度。支持裁剪不是平滑或减面。',
                      '原始点云、法线、正式setp3和setp4不覆盖。源文件校验状态见comparison.json。'])
        (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(); tree = cKDTree(points)
    script = Path(__file__).resolve().parent/'tools/compare_geometry.py'
    try:
        print('Poisson: reconstructing identical full input...', flush=True)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points); cloud.normals = o3d.utility.Vector3dVector(normals)
        stamp = time.perf_counter()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=9, scale=1.05, n_threads=4)
        elapsed = time.perf_counter()-stamp
        report['methods']['poisson'] = save_method('poisson', mesh, points, tree, output, elapsed); save()
        print('NKSR: checking independent inference environment...', flush=True)
        env = os.environ.copy(); env['PYTHONIOENCODING'] = 'utf-8'
        from nksr_runtime import inference_command
        probe = subprocess.run(inference_command(nksr_python, script, ['--probe']), capture_output=True,
                               text=True, encoding='utf-8', errors='replace', env=env, timeout=90,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (output/'nksr_environment.log').write_text(probe.stdout+'\n'+probe.stderr, encoding='utf-8')
        if probe.returncode:
            report['methods']['nksr'] = {'status': 'unavailable', 'python': str(nksr_python),
                'error': (probe.stdout+'\n'+probe.stderr)[-4000:]}
        else:
            print('NKSR: reconstruction...', flush=True)
            packet = output/'nksr_inference.npz'
            scale_arguments = (['--detail-level', str(detail_level)] if detail_level is not None
                               else ['--voxel-mm', str(selected_voxel_mm)])
            with (output/'nksr_inference.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(inference_command(nksr_python, script, ['--infer',
                    '--source', str(source), '--packet', str(packet), *scale_arguments, '--device', device]),
                    stdout=log, stderr=subprocess.STDOUT, env=env,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode: raise RuntimeError('NKSR inference failed; see nksr_inference.log')
            with np.load(packet, allow_pickle=False) as data:
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']), o3d.utility.Vector3iVector(data['triangles']))
                elapsed = float(data['seconds'])
            report['methods']['nksr'] = save_method('nksr', mesh, points, tree, output, elapsed)
            report['methods']['nksr']['environment'] = probe.stdout
            packet.unlink()  # Only our temporary interchange file, after successful export.
    except Exception:
        report['error'] = traceback.format_exc(); raise
    finally:
        report['input_unchanged'] = hashes == {name: file_hash(source/name) for name in hashes}
        report['complete'] = report['input_unchanged'] and all(report['methods'].get(n, {}).get('status') == 'complete' for n in ('poisson', 'nksr'))
        save()
    print(f'Comparison complete: {report["complete"]}; saved to {output}', flush=True)
    return report
