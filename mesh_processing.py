"""Shared mesh operations for production and historical reconstruction pilots."""
import numpy as np


def retention_schedule():
    return tuple(value/10 for value in range(4, 11))


def scene_for(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def distances(scene, points):
    import open3d as o3d
    return np.concatenate([scene.compute_distance(o3d.core.Tensor(
        np.asarray(points[start:start+50000], dtype=np.float32)), nthreads=4).numpy()
        for start in range(0, len(points), 50000)])*1000


def sampled_surface(mesh, count=100000):
    import open3d as o3d
    o3d.utility.random.seed(20261015)
    return np.asarray(mesh.sample_points_uniformly(count).points)


def component_meshes(mesh, preserve_below=1000):
    import open3d as o3d
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels, sizes = np.asarray(labels), np.asarray(sizes)
    faces = np.asarray(mesh.triangles)
    def subset(mask):
        part = o3d.geometry.TriangleMesh()
        part.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        part.triangles = o3d.utility.Vector3iVector(faces[mask])
        part.remove_unreferenced_vertices()
        return part
    fixed = subset(sizes[labels] < preserve_below)
    large = [subset(labels == index) for index in np.flatnonzero(sizes >= preserve_below)]
    return fixed, large


def decimate_components(fixed, large, ratio, maximum_error=float('inf')):
    import open3d as o3d
    result = o3d.geometry.TriangleMesh(fixed)
    for part in large:
        result += part.simplify_quadric_decimation(int(len(part.triangles)*ratio),
                            maximum_error=maximum_error, boundary_weight=100.)
    return result


def decimate_to_total(fixed, large, target_faces):
    """Allocate a total face budget after reserving every small component."""
    import open3d as o3d
    result = o3d.geometry.TriangleMesh(fixed)
    large_faces = sum(len(part.triangles) for part in large)
    ratio = min(1., max(0., (target_faces-len(fixed.triangles))/max(large_faces, 1)))
    for part in large:
        target = max(4, round(len(part.triangles)*ratio))
        result += part.simplify_quadric_decimation(target, boundary_weight=100.)
    return result


def select_decimation(smoothed, original, reference_scene=None, reference_samples=None):
    """40%,50%,...,100%; accept the first sampled-distance/area compliant result."""
    reference_scene = reference_scene if reference_scene is not None else scene_for(original)
    reference_samples = reference_samples if reference_samples is not None else sampled_surface(original)
    original_area = original.get_surface_area()
    fixed, large = component_meshes(smoothed)
    attempts = []
    for ratio in retention_schedule():
        target = round(len(smoothed.triangles)*ratio)
        candidate = decimate_to_total(fixed, large, target) if ratio < 1 else type(smoothed)(smoothed)
        candidate.remove_degenerate_triangles(); candidate.remove_duplicated_triangles(); candidate.remove_unreferenced_vertices()
        candidate.compute_vertex_normals()
        candidate_scene = scene_for(candidate)
        outward = distances(reference_scene, sampled_surface(candidate))
        inward = distances(candidate_scene, reference_samples)
        area_ratio = candidate.get_surface_area()/original_area
        deviation = float(max(outward.max(), inward.max()))
        accepted = deviation <= 15. and .95 <= area_ratio <= 1.05
        attempts.append({'target_ratio': ratio, 'target_faces': target, 'actual_faces': len(candidate.triangles),
                         'sampled_two_way_max_mm': deviation, 'area_ratio': area_ratio, 'accepted': accepted})
        if accepted:
            return candidate, attempts, {'preserved_small_component_faces': len(fixed.triangles),
                                        'decimated_components': len(large)}
    raise RuntimeError('No face retention candidate passed geometry checks, including 100%')
