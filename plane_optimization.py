"""Conservative planar-patch validation and bounded normal-only mesh correction."""
import numpy as np
from scipy.spatial import cKDTree


def fit_patch(points, normals, center, rotation, extent):
    """Validate a detector proposal against observations; return None on ambiguity."""
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    rotation, center, extent = np.asarray(rotation), np.asarray(center), np.asarray(extent)
    local = (points-center) @ rotation
    footprint = ((np.abs(local[:, :2]) <= extent[:2]/2+.01).all(axis=1)
                 & (normals @ rotation[:, 2] >= np.cos(np.deg2rad(35))))
    volume = footprint & (np.abs(local[:, 2]) <= max(extent[2]/2, .025))
    selection = volume & (np.abs(local[:, 2]) <= .025)
    if selection.sum() < .7*max(int(volume.sum()), 1): return None, 'thick or multi-surface proposal'
    ids = np.flatnonzero(selection)
    if len(ids) < 200: return None, 'insufficient support'
    data = points[ids]
    normal = rotation[:, 2].copy()
    origin = np.median(data, axis=0)
    for _ in range(8):
        residual = (data-origin) @ normal
        weights = 1/(1+(residual/.006)**2)
        origin = (weights[:, None]*data).sum(axis=0)/weights.sum()
        delta = data-origin
        _, vectors = np.linalg.eigh((delta*weights[:, None]).T @ delta/weights.sum())
        normal = vectors[:, 0]
        if normal @ rotation[:, 2] < 0: normal = -normal
    residual = (data-origin) @ normal
    sigma = float(1.4826*np.median(np.abs(residual-np.median(residual))))
    p90 = float(np.percentile(np.abs(residual), 90))
    # Reject clear parallel layers rather than collapsing them onto one plane.
    low, high = np.percentile(residual, [25, 75])
    for _ in range(12):
        split = np.abs(residual-low) < np.abs(residual-high)
        if not split.any() or split.all(): break
        low, high = residual[split].mean(), residual[~split].mean()
    if split.any() and not split.all():
        separation = abs(high-low)
        spread = max(residual[split].std(), residual[~split].std())
        if separation > .008 and spread < .002 and .2 < split.mean() < .8:
            return None, 'parallel layers'
    if sigma > .008 or p90 > .015: return None, 'excessive plane residual'
    keep = np.abs(residual) <= .015
    ids, data, residual = ids[keep], data[keep], residual[keep]
    if len(ids) < 200: return None, 'insufficient inliers'
    axis = rotation[:, 0]-normal*(rotation[:, 0] @ normal)
    axis /= np.linalg.norm(axis)
    basis = np.column_stack((axis, np.cross(normal, axis)))
    uv = (data-origin) @ basis
    bounds = np.percentile(uv, [1, 99], axis=0)
    spans = bounds[1]-bounds[0]
    if spans.min() < .10 or spans.max() < .18: return None, 'small or narrow patch'
    uv_scaled = uv/np.maximum(spans, .1)
    u, v = uv_scaled.T
    design = np.column_stack((np.ones(len(u)), u, v, u*u, u*v, v*v))
    coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
    prediction = design @ coefficients
    bend = float(np.percentile(prediction, 95)-np.percentile(prediction, 5))
    explained = 1-float(np.var(residual-prediction))/max(float(np.var(residual)), 1e-12)
    if bend > .006 and explained > .5: return None, 'coherent curvature'
    return {'origin': origin, 'normal': normal, 'basis': basis, 'bounds': bounds,
            'support_points': data, 'support_ids': ids, 'sigma_mm': sigma*1000,
            'p90_mm': p90*1000, 'quadratic_bend_mm': bend*1000,
            'quadratic_explained_fraction': explained}, None


def optimize_vertices(vertices, faces, normals, patches, anchors, max_move_mm=8., total_limit_mm=15., observations=None, observation_tree=None):
    """Assign supported patches and move only along their normals, with fixed anchors."""
    from mesh_refinement import topology
    vertices, normals, anchors = np.asarray(vertices, float), np.asarray(normals, float), np.asarray(anchors, float)
    faces = np.asarray(faces, np.int64)
    labels = np.full(len(vertices), -1, np.int32)
    score = np.full(len(vertices), np.inf)
    delta = np.zeros_like(vertices)
    if observations is not None:
        observation_tree = observation_tree if observation_tree is not None else cKDTree(observations)
        observation_distance, observation_ids = observation_tree.query(vertices, workers=2)
    for index, patch in enumerate(patches):
        n, origin, basis = patch['normal'], patch['origin'], patch['basis']
        local = vertices-origin
        uv, distance = local @ basis, local @ n
        low, high = patch['bounds']
        edge_distance = np.minimum(uv-low, high-uv).min(axis=1)
        eligible = (edge_distance > 0) & (np.abs(distance) <= .025) & (np.abs(normals @ n) >= np.cos(np.deg2rad(40)))
        ids = np.flatnonzero(eligible)
        if not len(ids): continue
        if observations is not None:
            membership = np.zeros(len(observations), dtype=bool)
            membership[patch['global_ids']] = True
            ids = ids[membership[observation_ids[ids]] & (observation_distance[ids] <= .025)]
            if not len(ids): continue
        support_uv = (patch['support_points']-origin) @ basis
        nearby, _ = cKDTree(support_uv).query(uv[ids], workers=2)
        observation, _ = cKDTree(patch['support_points']).query(vertices[ids], workers=2)
        confidence = np.clip(edge_distance[ids]/.025, 0, 1)*np.clip((.020-nearby)/.010, 0, 1)
        candidate_score = np.abs(distance[ids]) + patch['sigma_mm']/1000
        accept = (nearby <= .020) & (observation <= .025) & (candidate_score < score[ids]) & (confidence > 0)
        ids, confidence, candidate_score = ids[accept], confidence[accept], candidate_score[accept]
        labels[ids] = index; score[ids] = candidate_score
        delta[ids] = -np.clip(distance[ids], -max_move_mm/1000, max_move_mm/1000)[:, None]*n*confidence[:, None]
    # Preserve sharp intersections; moving a hole boundary normal to its plane
    # is allowed, since connectivity and its in-plane contour remain unchanged.
    _, _, feature, _ = topology(vertices, faces, 65.)
    delta[feature] = 0
    relative = vertices-anchors
    a = (delta*delta).sum(axis=1)
    b = (relative*delta).sum(axis=1)
    c = (relative*relative).sum(axis=1)-(total_limit_mm/1000)**2
    scale = np.clip((-b+np.sqrt(np.maximum(b*b-a*c, 0)))/np.maximum(a, 1e-30), 0, 1)
    scale[c > 1e-10] = 0  # Existing over-budget geometry is retained, not moved.
    delta *= scale[:, None]
    result = vertices+delta
    old_normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    valid_old = np.linalg.norm(old_normal, axis=1) > 1e-15
    for _ in range(20):
        new_normal = np.cross(result[faces[:, 1]]-result[faces[:, 0]], result[faces[:, 2]]-result[faces[:, 0]])
        bad = ((old_normal*new_normal).sum(axis=1) <= 0) & valid_old
        if not bad.any(): break
        ids = np.unique(faces[bad]); result[ids] = vertices[ids]
    if bad.any(): raise RuntimeError('Planar correction caused inversions; lower movement limit')
    movement = np.linalg.norm(result-vertices, axis=1)*1000
    metrics = []
    all_before, all_after = [], []
    for index, patch in enumerate(patches):
        ids = labels == index
        if not ids.any(): continue
        before = (vertices[ids]-patch['origin']) @ patch['normal']*1000
        after = (result[ids]-patch['origin']) @ patch['normal']*1000
        all_before.append(np.abs(before)); all_after.append(np.abs(after))
        metrics.append({'patch': index, 'assigned_vertices': int(ids.sum()),
                        'moved_vertices': int((movement[ids] > .01).sum()),
                        'plane_abs_mm_before_p50_p90': np.percentile(np.abs(before), [50, 90]).tolist(),
                        'plane_abs_mm_after_p50_p90': np.percentile(np.abs(after), [50, 90]).tolist()})
    return result, labels, {'patches': metrics, 'moved_vertices': int((movement > .01).sum()),
                             'assigned_plane_abs_mm_before_p50_p90': np.percentile(np.concatenate(all_before), [50, 90]).tolist() if all_before else [0., 0.],
                             'assigned_plane_abs_mm_after_p50_p90': np.percentile(np.concatenate(all_after), [50, 90]).tolist() if all_after else [0., 0.],
                             'movement_mm_p50_p90_p99_max': np.percentile(movement, [50, 90, 99, 100]).tolist(),
                             'frozen_over_budget_vertices': int((c > 1e-10).sum()),
                             'newly_moved_total_anchor_distance_max_mm': float(np.linalg.norm(result[movement > 0]-anchors[movement > 0], axis=1).max()*1000) if (movement > 0).any() else 0.,
                             'flipped_faces': int(bad.sum())}


def detect_planes(points, normals, tree=None, cloud=None):
    """Detect and validate supported patches; share the observation index when supplied."""
    import open3d as o3d
    if cloud is None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points); cloud.normals = o3d.utility.Vector3dVector(normals)
    detector = cloud.voxel_down_sample(.015)
    o3d.utility.random.seed(20261015)
    proposals = detector.detect_planar_patches(normal_variance_threshold_deg=35, coplanarity_deg=70,
                        outlier_ratio=.5, min_plane_edge_length=.15, min_num_points=100,
                        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    if not proposals and len(detector.points) >= 200:
        # The octree detector can return no patches for almost entirely planar
        # inputs. A dominant RANSAC proposal still undergoes all validations.
        from types import SimpleNamespace
        coefficient, inliers = detector.segment_plane(.010, 3, 1000, probability=.999)
        if len(inliers) >= .6*len(detector.points):
            data = np.asarray(detector.points)[np.asarray(inliers)]
            n = np.asarray(coefficient[:3]); n *= 1 if n @ normals.mean(axis=0) >= 0 else -1
            axis = np.eye(3)[np.argmin(np.abs(n))]; axis -= n*(axis @ n); axis /= np.linalg.norm(axis)
            rotation = np.column_stack((axis, np.cross(n, axis), n))
            origin = np.median(data, axis=0)
            local = (data-origin) @ rotation
            bounds = np.percentile(local[:, :2], [1, 99], axis=0)
            origin += rotation[:, :2] @ bounds.mean(axis=0)
            proposals = [SimpleNamespace(center=origin, R=rotation, extent=np.r_[bounds[1]-bounds[0]+.02, .03])]
    print(f'Detected {len(proposals)} proposals on {len(detector.points):,} detector points; fitting against full observations...', flush=True)
    patches, rejected, descriptions = [], [], []
    tree = tree if tree is not None else cKDTree(points)
    for i, box in enumerate(proposals):
        ids = np.asarray(tree.query_ball_point(box.center, np.linalg.norm(box.extent)/2+.04), np.int64)
        patch, reason = fit_patch(points[ids], normals[ids], box.center, box.R, box.extent)
        extraction = 'robust planar patch'
        dominant_fraction = None
        if patch is None and reason not in ('coherent curvature', 'parallel layers'):
            # Separate a dominant observed surface from clutter inside a thick
            # detector box. Mesh eligibility still requires exact membership in
            # the final observed support, not merely proximity to its plane.
            local = (points[ids]-box.center) @ box.R
            selected = ((np.abs(local[:, :2]) <= box.extent[:2]/2+.01).all(axis=1)
                        & (np.abs(local[:, 2]) <= box.extent[2]/2+.01)
                        & (normals[ids] @ box.R[:, 2] >= np.cos(np.deg2rad(35))))
            subset_ids = ids[selected]
            if len(subset_ids) >= 400:
                subcloud = o3d.geometry.PointCloud()
                subcloud.points = o3d.utility.Vector3dVector(points[subset_ids])
                coefficient, inliers = subcloud.segment_plane(.010, 3, 500, probability=.999)
                dominant_fraction = len(inliers)/len(subset_ids)
                n = np.asarray(coefficient[:3]); n *= 1 if n @ box.R[:, 2] >= 0 else -1
                if dominant_fraction >= .6 and n @ box.R[:, 2] >= np.cos(np.deg2rad(15)):
                    axis = box.R[:, 0]-n*(box.R[:, 0] @ n); axis /= np.linalg.norm(axis)
                    rotation = np.column_stack((axis, np.cross(n, axis), n))
                    origin = np.median(points[subset_ids[np.asarray(inliers)]], axis=0)
                    extent = box.extent.copy(); extent[2] = .03
                    patch, reason = fit_patch(points[subset_ids], normals[subset_ids], origin, rotation, extent)
                    if patch is not None:
                        ids = subset_ids
                        extraction = 'dominant RANSAC surface + robust validation'
        if patch is None:
            rejected.append({'proposal': i, 'reason': reason}); continue
        patch['global_ids'] = ids[patch['support_ids']]
        descriptions.append({'patch': len(patches), 'proposal': i, 'support_points': len(patch['support_points']),
                             'origin': patch['origin'].tolist(), 'normal': patch['normal'].tolist(),
                             'sigma_mm': patch['sigma_mm'], 'p90_mm': patch['p90_mm'],
                             'quadratic_bend_mm': patch['quadratic_bend_mm'],
                             'quadratic_explained_fraction': patch['quadratic_explained_fraction'],
                             'extraction': extraction, 'dominant_ransac_fraction': dominant_fraction,
                             'bounds_uv_m': patch['bounds'].tolist(), 'basis': patch['basis'].tolist()})
        patches.append(patch)
    return patches, descriptions, rejected, cloud
