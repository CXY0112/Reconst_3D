"""Constrained Taubin vertex smoothing; no topology changes or hole filling."""
import numpy as np
from scipy.sparse import coo_matrix, diags


def topology(vertices, faces, feature_angle=65.):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    face_normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                           vertices[faces[:, 2]]-vertices[faces[:, 0]])
    face_normal /= np.maximum(np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-15)
    all_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edge_faces = np.tile(np.arange(len(faces)), 3)
    edges, inverse, counts = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    offsets = np.r_[0, np.cumsum(counts)]
    paired = counts == 2
    a = edge_faces[order[offsets[:-1][paired]]]
    b = edge_faces[order[offsets[:-1][paired]+1]]
    cosine = np.clip((face_normal[a]*face_normal[b]).sum(axis=1), -1, 1)
    angles = np.degrees(np.arccos(cosine))
    boundary = np.zeros(len(vertices), dtype=bool)
    boundary[edges[counts != 2].ravel()] = True
    feature = np.zeros(len(vertices), dtype=bool)
    feature[edges[paired][angles >= feature_angle].ravel()] = True
    return edges, boundary, feature, angles


def constrained_taubin(vertices, faces, iterations=30, max_move_mm=8., feature_angle=65.):
    if iterations < 1 or max_move_mm <= 0 or not 0 < feature_angle <= 180:
        raise ValueError('Invalid smoothing parameters')
    original = np.asarray(vertices, dtype=float).copy()
    faces = np.asarray(faces, dtype=np.int64)
    edges, boundary, feature, before_angles = topology(original, faces, feature_angle)
    locked = boundary | feature
    lengths = np.linalg.norm(original[edges[:, 0]]-original[edges[:, 1]], axis=1)
    weights = 1/np.maximum(lengths, 1e-9)
    adjacency = coo_matrix((np.r_[weights, weights],
                            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(original), len(original))).tocsr()
    averaging = diags(1/np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1e-15)) @ adjacency
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    locked |= degree == 0
    vertices = original.copy()
    for _ in range(iterations):
        for factor in (.45, -.47):
            candidate = vertices + factor*(averaging @ vertices-vertices)
            candidate[locked] = original[locked]
            delta = candidate-original
            length = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(1., max_move_mm/1000/np.maximum(length, 1e-15))[:, None]
            vertices = original+delta
    old_normal = np.cross(original[faces[:, 1]]-original[faces[:, 0]], original[faces[:, 2]]-original[faces[:, 0]])
    new_normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    inverted = ((old_normal*new_normal).sum(axis=1) <= 0) & (np.linalg.norm(old_normal, axis=1) > 1e-15)
    # If a face would invert, restore all its incident vertices to their inputs.
    # Iterate because restoring one face can affect its immediate neighbours.
    for _ in range(12):
        if not inverted.any(): break
        vertices[np.unique(faces[inverted])] = original[np.unique(faces[inverted])]
        new_normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
        inverted = ((old_normal*new_normal).sum(axis=1) <= 0) & (np.linalg.norm(old_normal, axis=1) > 1e-15)
    if inverted.any():
        raise RuntimeError('Smoothing caused local inversions; lower iteration count')
    movement = np.linalg.norm(vertices-original, axis=1)*1000
    after_angles = topology(vertices, faces, feature_angle)[3]
    metrics = {'iterations': iterations, 'lambda': .45, 'mu': -.47,
               'fixed_inverse_edge_length_weights': True, 'max_move_mm': max_move_mm,
               'feature_angle_deg': feature_angle, 'boundary_locked_vertices': int(boundary.sum()),
               'feature_locked_vertices': int(feature.sum()), 'locked_fraction': float(locked.mean()),
               'movement_mm_p50_p90_p99_max': np.percentile(movement, [50, 90, 99, 100]).tolist(),
               'adjacent_face_angle_deg_before_p50_p90': np.percentile(before_angles, [50, 90]).tolist(),
               'adjacent_face_angle_deg_after_p50_p90': np.percentile(after_angles, [50, 90]).tolist(),
               'flipped_faces': int(inverted.sum())}
    return vertices, metrics
