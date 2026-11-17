"""Single-process Poisson -> supported mesh -> smooth -> simplify -> planar mesh."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plane_statistics(before, after, labels, patches):
    left, right = [], []
    for index, patch in enumerate(patches):
        mask = labels == index
        if mask.any():
            left.extend(np.abs((before[mask]-patch['origin']) @ patch['normal'])*1000)
            right.extend(np.abs((after[mask]-patch['origin']) @ patch['normal'])*1000)
    return {'assigned_vertices': len(left),
            'assigned_plane_abs_mm_before_p50_p90': np.percentile(left, [50, 90]).tolist() if left else [0., 0.],
            'assigned_plane_abs_mm_after_p50_p90': np.percentile(right, [50, 90]).tolist() if right else [0., 0.]}


def run_pipeline(source, output, debug=False, preview_packet=None, method='poisson',smoothing_iterations=30):
    if not isinstance(smoothing_iterations,int) or smoothing_iterations<1:raise ValueError('Smoothing iterations must be a positive integer')
    method=method.lower()
    from geometry_methods import METHODS,run_neural_pipeline
    if method not in {name.lower() for name in METHODS}:raise ValueError('Available reconstruction methods: Poisson, NKSR, TSDF')
    if method=='tsdf':
        from tsdf_reconstruction import run_tsdf
        return run_tsdf(source,output,preview_packet)
    if method!='poisson':return run_neural_pipeline(source,output,method,preview_packet)
    import open3d as o3d
    from scipy.spatial import cKDTree
    from mesh_refinement import constrained_taubin
    from mesh_processing import scene_for, sampled_surface, distances, select_decimation
    from plane_optimization import detect_planes, optimize_vertices
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the source folder')
    started = time.perf_counter()
    from geometry_inputs import poisson_source
    info,points,normals,files,normal_info = poisson_source(source)
    hashes = {p.name: file_hash(p) for p in files}
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 200 or normals.shape != points.shape:
        raise ValueError('At least 200 XYZ points and matching world normals are required')
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(points).all() or not np.isfinite(normals).all() or (lengths < .5).any():
        raise ValueError('Invalid source geometry or normals')
    normals /= lengths[:, None]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points); cloud.normals = o3d.utility.Vector3dVector(normals)
    tree = cKDTree(points)
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'stage': 3, 'method': 'poisson', 'pipeline': 'poisson_supported_smooth_qem_planar',
              'input': str(source), 'input_points': len(points), 'input_sha256': hashes,
              'input_kind':info['kind'],'preprocessing_used':info['kind']=='preprocessed',
              'source_session':str(info['session']) if info['session'] else None,'normal_estimation':normal_info,
              'complete': False, 'debug_outputs': debug, 'stage_seconds': {}, 'stages': {},
              'units': 'metres', 'coordinates': 'world X forward Y left Z up',
              'parameters': {'poisson_depth': 9, 'poisson_scale': 1.05, 'support_trim_mm': 15,'smoothing_iterations':smoothing_iterations,
                             'retention_schedule': [.4, .5, .6, .7, .8, .9, 1.],
                             'sampled_deviation_limit_mm': 15, 'area_ratio_range': [.95, 1.05],
                             'preserve_components_below_faces': 1000}}
    report['stage_seconds']['load_and_index'] = time.perf_counter()-started
    def save_report():
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    def snapshot(name, mesh):
        report['stages'][name] = {'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles)}
        if debug:
            directory = output/'debug'; directory.mkdir(exist_ok=True)
            if not o3d.io.write_triangle_mesh(str(directory/(name+'.ply')), mesh, write_vertex_colors=False):
                raise OSError('Could not save debug mesh')
    try:
        stamp = time.perf_counter(); print('1/6 Screened Poisson...', flush=True)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=9, scale=1.05, linear_fit=False, n_threads=4)
        mesh.remove_duplicated_triangles(); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
        snapshot('poisson', mesh)
        report['stage_seconds']['poisson'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('2/6 Observation support trim...', flush=True)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        vertex_distance, _ = tree.query(vertices, workers=2)
        center_distance, _ = tree.query(vertices[faces].mean(axis=1), workers=2)
        keep = (vertex_distance[faces].max(axis=1) <= .015) & (center_distance <= .015)
        mesh.remove_triangles_by_mask(~keep); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
        if not len(mesh.triangles): raise RuntimeError('Support trim produced an empty mesh')
        original = mesh
        snapshot('supported', original)
        original_scene = scene_for(original)
        original_samples = sampled_surface(original)
        report['stage_seconds']['support_trim_and_reference'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('3/6 Constrained Taubin smoothing...', flush=True)
        smooth_vertices, smoothing = constrained_taubin(np.asarray(original.vertices), np.asarray(original.triangles),iterations=smoothing_iterations)
        smoothed = o3d.geometry.TriangleMesh(original)
        smoothed.vertices = o3d.utility.Vector3dVector(smooth_vertices); smoothed.compute_vertex_normals()
        snapshot('smooth', smoothed); report['smoothing'] = smoothing
        report['stage_seconds']['smoothing'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('4/6 QEM: 40%, 50%, ..., 100% until accepted...', flush=True)
        reduced, attempts, component_info = select_decimation(smoothed, original, original_scene, original_samples)
        report['decimation_attempts'] = attempts; report['components'] = component_info
        snapshot('reduced', reduced)
        report['stage_seconds']['decimation_and_acceptance'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('5/6 Supported planar correction...', flush=True)
        patches, descriptions, rejected, _ = detect_planes(points, normals, tree=tree, cloud=cloud)
        before = np.asarray(reduced.vertices).copy()
        anchors = np.concatenate([original_scene.compute_closest_points(o3d.core.Tensor(
            before[start:start+50000].astype(np.float32)), nthreads=4)['points'].numpy()
            for start in range(0, len(before), 50000)])
        corrected, labels, plane_metrics = optimize_vertices(before, np.asarray(reduced.triangles),
            np.asarray(reduced.vertex_normals), patches, anchors, observations=points, observation_tree=tree)
        report['planes'] = descriptions; report['rejected_planes'] = rejected
        report['stage_seconds']['plane_detection_and_correction'] = time.perf_counter()-stamp
        stamp = time.perf_counter(); print('6/6 Final validation and export...', flush=True)
        # A single final acceptance check; reduce only the planar correction if
        # its cumulative surface deviation exceeds the existing 15mm/5% limits.
        faces = np.asarray(reduced.triangles)
        old_normal = np.cross(before[faces[:, 1]]-before[faces[:, 0]], before[faces[:, 2]]-before[faces[:, 0]])
        valid_face = np.linalg.norm(old_normal, axis=1) > 1e-15
        trials = []
        for strength in (1., .75, .5, .25, 0.):
            actual = before+(corrected-before)*strength
            final = o3d.geometry.TriangleMesh(reduced)
            final.vertices = o3d.utility.Vector3dVector(actual); final.compute_vertex_normals()
            final_scene = scene_for(final)
            outward = distances(original_scene, sampled_surface(final))
            inward = distances(final_scene, original_samples)
            max_deviation = float(max(outward.max(), inward.max()))
            area_ratio = final.get_surface_area()/original.get_surface_area()
            face_normal = np.cross(actual[faces[:, 1]]-actual[faces[:, 0]], actual[faces[:, 2]]-actual[faces[:, 0]])
            inversions = int((((old_normal*face_normal).sum(axis=1) <= 0) & valid_face).sum())
            accepted = max_deviation <= 15. and .95 <= area_ratio <= 1.05 and inversions == 0
            trials.append({'planar_strength': strength, 'sampled_two_way_max_mm': max_deviation,
                           'area_ratio': area_ratio, 'inverted_faces': inversions, 'accepted': accepted})
            if accepted: break
        else: raise RuntimeError('Final mesh failed cumulative geometry checks')
        displacement = np.linalg.norm(actual-before, axis=1)*1000
        moved = displacement > .01
        report['planar_acceptance'] = trials
        report['planar'] = {'patch_count': len(patches), 'moved_vertices': int(moved.sum()),
                            'movement_mm_p50_p90_p99_max': np.percentile(displacement, [50, 90, 99, 100]).tolist(),
                            'moved_total_anchor_distance_max_mm': float(np.linalg.norm(actual[moved]-anchors[moved], axis=1).max()*1000) if moved.any() else 0.,
                            **plane_statistics(before, actual, labels, patches)}
        input_distance = distances(final_scene, points)
        report['final'] = {'vertices': len(final.vertices), 'triangles': len(final.triangles),
                           'input_coverage': {str(mm): float(np.mean(input_distance <= mm)) for mm in (5, 10, 15)},
                           'input_to_mesh_mm_p95': float(np.percentile(input_distance,95)),
                           'retained_fraction_of_supported_faces': len(final.triangles)/len(original.triangles),
                           'input_to_mesh_mm_p50_p90_p99_max': np.percentile(input_distance, [50, 90, 99, 100]).tolist(),
                           'sampled_two_way_to_supported_mm_p50_p90_p99_max': np.percentile(np.r_[outward, inward], [50, 90, 99, 100]).tolist(),
                           'area_ratio_to_supported': area_ratio, 'planar_topology_unchanged': np.array_equal(np.asarray(final.triangles), faces)}
        if hashes != {p.name: file_hash(p) for p in files}: raise RuntimeError('Input data changed during reconstruction')
        if not o3d.io.write_triangle_mesh(str(output/'mesh.ply'), final, write_vertex_colors=False):
            raise OSError('Failed to save final mesh')
        report['final']['mesh_sha256'] = file_hash(output/'mesh.ply')
        if preview_packet is not None:
            np.savez(preview_packet, vertices=np.asarray(final.vertices, dtype=np.float32),
                     triangles=np.asarray(final.triangles, dtype=np.uint32),
                     normals=np.asarray(final.vertex_normals, dtype=np.float32))
        report['stage_seconds']['final_validation_and_export'] = time.perf_counter()-stamp
        report.update(complete=True, source_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                      notes='Sampled deviation limits are not exact Hausdorff guarantees. Point/plane agreement is not ground-truth accuracy. No BPA, RGB, pose changes, hole filling, routine intermediate exports or comparison renders. Face budgets reserve small components; exact targets may not be reached. Timing excludes worker interpreter/import startup.')
        save_report()
        print(f'Complete: {len(final.triangles):,} faces in {report["elapsed_seconds"]:.2f}s | {output}', flush=True)
        return report
    except Exception as exc:
        report.update(error=str(exc), elapsed_seconds=time.perf_counter()-started)
        save_report()
        raise
