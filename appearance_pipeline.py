"""Conservative RGB vertex-color baseline. Geometry and camera poses remain fixed."""
from pathlib import Path
from saved_sources import reference
import hashlib
import json
import time
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from kalman_fusion import session_observations, OPTICAL_FROM_WORLD
from geometry_methods import is_geometry_report

PIPELINE = 'rgb_vertex_color_baseline'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_inputs(source, session=None):
    source = Path(source).resolve()
    geometry = json.loads((source/'report.json').read_text(encoding='utf-8'))
    if not is_geometry_report(geometry):
        raise ValueError('请选择第三阶段完整的正式几何重建目录')
    if session is None:
        session = reference(geometry.get('source_session'),source)
        if session is None:
            preprocessed = reference(geometry['input'], source)
            if preprocessed is None: raise FileNotFoundError('找不到关联的第二阶段结果，请手动指定源扫描')
            pre = json.loads((preprocessed/'report.json').read_text(encoding='utf-8'))
            session = reference(pre.get('source_session'), preprocessed)
        if not session: raise ValueError('缺少源扫描关联，请手动指定原始扫描目录')
    session, observations, fingerprints = session_observations(session)
    frames = {int(row['capture']):row for row in
              (json.loads(line) for line in (session/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    calibration = json.loads((session/'sdk_calibration.json').read_text(encoding='utf-8'))
    k = np.array([[calibration['fxrgb'], 0, calibration['cxrgb']],
                  [0, calibration['fyrgb'], calibration['cyrgb']], [0, 0, 1.]], float)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0: raise ValueError('无效RGB内参')
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = [[calibration[f'R{i}{j}'] for j in range(3)] for i in range(3)]
    extrinsic[:3,3] = [calibration[f'T{i}']/1000 for i in range(1,4)]
    result = []
    for observation in observations:
        if observation['coordinates']:
            # New manifests declare local cloud and raw depth frames separately;
            # registered photos always use the original RGB camera pose.
            pose=observation['rgb_pose'].copy()
            world_from_rgb=pose
        else:
            # Unlabelled historical sessions retain their former interpretation.
            pose = observation['pose'].copy()
            pose[:3,:3] = OPTICAL_FROM_WORLD.T @ pose[:3,:3]
            pose[:3,3] = OPTICAL_FROM_WORLD.T @ pose[:3,3]
            world_from_rgb = pose @ np.linalg.inv(extrinsic)
        name = f'{float(frames[observation["capture"]]["stamp"]):.6f}.png'
        rgb, depth = session/'rgb'/name, session/'depth'/name
        if geometry.get('camera_model')=='sdk_rgb_grid_v1':
            if observation['coordinates'] and observation['coordinates']['raw_depth_frame']!='rgb':
                raise ValueError('SDK RGB-grid appearance requires RGB-frame raw depth')
            # SDK TSDF uses its audited depth/XYZ grid, not the legacy registered-depth frame.
            world_from_rgb=pose;depth=session/'depth_raw'/name
        if not rgb.is_file() or not depth.is_file(): raise FileNotFoundError(f'缺少拍摄{observation["capture"]}的配准RGB/深度图')
        result.append({'capture':observation['capture'], 'pose':world_from_rgb, 'rgb':rgb, 'depth':depth})
    return source, session, result, k, fingerprints


def prepare_depth(depth, hole_radius=2, edge_limit=.03):
    """Nearest depth at most two pixels away; no inpainting across depth jumps."""
    depth = np.asarray(depth, float)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any(): return np.zeros_like(depth), np.zeros_like(valid)
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    filled = depth[tuple(indices)]
    low = minimum_filter(np.where(valid, depth, np.inf), size=3, mode='nearest')
    high = maximum_filter(np.where(valid, depth, 0), size=3, mode='nearest')
    stable = (high-low <= edge_limit) & (distance <= hole_radius)
    return filled, stable


def observe(vertices, normals, frame, k, image, depth, scene=None, depth_tolerance=.02, diagnostics=None,
            enhanced=False, strict_seen=None):
    """Return visible vertex IDs, sRGB and geometric weights for one accepted capture."""
    pose = np.asarray(frame['pose'])
    local = (vertices-pose[:3,3]) @ pose[:3,:3]
    h,w = depth.shape
    z = local[:,2]
    positive = z > .02
    uv = np.zeros((len(vertices),2), float)
    uv[positive] = (local[positive] @ k.T)[:,:2]/z[positive,None]
    u,v = np.rint(uv).astype(np.int64).T
    inside = positive & (u >= 2) & (u < w-2) & (v >= 2) & (v < h-2)
    ids = np.flatnonzero(inside)
    if diagnostics is not None: diagnostics['in_rgb_frustum']=len(ids)
    strict_depth, strict_stable = prepare_depth(depth)
    filled, stable = strict_depth, strict_stable
    if enhanced:
        filled,stable=prepare_depth(depth,hole_radius=3)
        depth_tolerance=.025
    ids = ids[stable[v[ids],u[ids]] & (np.abs(filled[v[ids],u[ids]]-z[ids]) <= depth_tolerance)]
    if diagnostics is not None: diagnostics['depth_consistent']=len(ids)
    directions = vertices[ids]-pose[:3,3]
    distances = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(distances[:,None],1e-12)
    incidence = np.abs(np.sum(normals[ids]*directions, axis=1))
    good = incidence >= .25
    ids,directions,distances,incidence = ids[good],directions[good],distances[good],incidence[good]
    if diagnostics is not None: diagnostics['angle_accepted']=len(ids)
    if scene is not None and len(ids):
        import open3d as o3d
        rays = np.column_stack((np.broadcast_to(pose[:3,3],directions.shape),directions)).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        # Reject another mesh surface in front, including geometry absent from depth.
        good = np.isfinite(hits) & (np.abs(hits-distances) <= .006)
        ids,distances,incidence = ids[good],distances[good],incidence[good]
    x,y = uv[ids].T; x0,y0 = np.floor(x).astype(int),np.floor(y).astype(int)
    a,b = (x-x0)[:,None],(y-y0)[:,None]
    colors = ((1-a)*(1-b)*image[y0,x0]+a*(1-b)*image[y0,x0+1]+
              (1-a)*b*image[y0+1,x0]+a*b*image[y0+1,x0+1])/255.
    border = np.minimum.reduce([x,y,w-1-x,h-1-y])
    weights = incidence**2 / np.maximum(distances**2,.04) * np.minimum(border/30,1)
    strict = strict_stable[v[ids],u[ids]] & (np.abs(strict_depth[v[ids],u[ids]]-z[ids])<=.02)
    if strict_seen is not None:strict_seen[ids[strict]]=True
    if diagnostics is not None:diagnostics['strict_visible_vertices']=int(strict.sum())
    if enhanced:weights[~strict]*=.35
    return ids, colors, weights


def combine_observations(count, samples, color_gate=.18):
    """Two-pass best-view anchored fusion; reject inconsistent RGB, never alter geometry."""
    best_weight=np.zeros(count); best=np.full((count,3),.65); best_capture=np.full(count,-1,np.int32)
    seen=np.zeros(count,np.uint16)
    for capture,ids,colors,weights in samples:
        seen[ids]+=1
        good=weights>best_weight[ids]; target=ids[good]
        best_weight[target]=weights[good];best[target]=colors[good];best_capture[target]=capture
    total=np.zeros(count);acc=np.zeros((count,3));used=np.zeros(count,np.uint16)
    conflicts=np.zeros(count,np.uint16);max_delta=np.zeros(count)
    for _,ids,colors,weights in samples:
        delta=np.linalg.norm(colors-best[ids],axis=1)/np.sqrt(3)
        max_delta[ids]=np.maximum(max_delta[ids],delta)
        good=delta<=color_gate
        conflicts[ids[~good]]+=1;target=ids[good];weight=weights[good]
        linear=np.where(colors[good]<=.04045,colors[good]/12.92,((colors[good]+.055)/1.055)**2.4)
        acc[target]+=linear*weight[:,None];total[target]+=weight;used[target]+=1
    covered=total>0
    linear=acc[covered]/total[covered,None]
    best[covered]=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    return np.clip(best,0,1),dict(covered=covered,observation_count=seen,fused_count=used,
        conflict_count=conflicts,best_capture=best_capture,max_color_delta=max_delta.astype(np.float32))


def run_pipeline(source, output, session=None, preview_packet=None, enhanced=True):
    import open3d as o3d
    started=time.perf_counter()
    source,session,frames,k,fingerprints=resolve_inputs(source,session)
    output=Path(output).resolve()
    tracked=[source/'mesh.ply',source/'report.json']+[session/name for name in fingerprints]
    tracked += [frame[key] for frame in frames for key in ('rgb','depth')]
    hashes={str(path):digest(path) for path in tracked}
    mesh=o3d.io.read_triangle_mesh(str(source/'mesh.ply'))
    if not len(mesh.triangles): raise ValueError('输入网格没有三角面')
    mesh.compute_vertex_normals()
    vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles);normals=np.asarray(mesh.vertex_normals)
    if not np.isfinite(vertices).all(): raise ValueError('输入网格含无效顶点')
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    samples=[];per_frame=[];strict_seen=np.zeros(len(vertices),bool)
    for index,frame in enumerate(frames):
        print(f'RGB {index+1}/{len(frames)} capture={frame["capture"]}',flush=True)
        image=np.asarray(o3d.io.read_image(str(frame['rgb'])),dtype=float)
        depth=np.asarray(o3d.io.read_image(str(frame['depth'])),dtype=float)/1000
        if image.ndim!=3 or image.shape[2]!=3 or depth.ndim!=2 or image.shape[:2]!=depth.shape:
            raise ValueError('配准RGB和深度尺寸或通道不一致')
        diagnostic={'capture':frame['capture']}
        ids,colors,weights=observe(vertices,normals,frame,k,image,depth,scene,diagnostics=diagnostic,
                                  enhanced=enhanced,strict_seen=strict_seen)
        samples.append((frame['capture'],ids,colors,weights))
        diagnostic['visible_vertices']=len(ids);per_frame.append(diagnostic)
    colors,attributes=combine_observations(len(vertices),samples)
    covered=attributes['covered'];multi=attributes['observation_count']>=2
    observed_colors=colors.copy()
    from color_completion import complete_colors
    colors,filled,regions,completion=complete_colors(vertices,triangles,normals,colors,covered) if enhanced else (
        colors,np.zeros(len(vertices),bool),np.full(len(vertices),-1),{'enabled':False})
    attributes.update(filled=filled,display_covered=covered|filled,surface_region=regions,
                      observed_colors=observed_colors.astype(np.float32),strict_covered=strict_seen)
    faces_covered=covered[triangles].all(axis=1)
    face_area=np.linalg.norm(np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]]),axis=1)/2
    mesh.vertex_colors=o3d.utility.Vector3dVector(colors)
    report={'schema_version':1,'stage':4,'pipeline':PIPELINE,'complete':False,
        'input':str(source),'source_session':str(session),'input_sha256':hashes,
        'coordinates':'world X forward Y left Z up','units':'metres',
        'geometry_modified':False,'poses_modified':False,'appearance':'vertex_colors_sRGB; no UV texture',
        'parameters':{'enhanced_coverage':enhanced,'depth_tolerance_mm':25 if enhanced else 20,'depth_hole_radius_pixels':3 if enhanced else 2,'depth_edge_limit_mm':30,
            'mesh_occlusion_tolerance_mm':6,'min_incidence_cos':.25,'color_gate_srgb_rms':.18,
            'exposure_compensation':False},
        'completion':completion,
        'final':{'vertices':len(vertices),'triangles':len(triangles)},'frames':per_frame,
        'coverage':{'colored_vertices':int(covered.sum()),'vertex_fraction':float(covered.mean()),
            'strict_colored_vertices':int(strict_seen.sum()),'recovered_observed_vertices':int((covered & ~strict_seen).sum()),
            'interpolated_vertices':int(filled.sum()),'display_vertex_fraction':float((covered|filled).mean()),
            'display_fully_colored_area_fraction':float(face_area[(covered|filled)[triangles].all(axis=1)].sum()/max(face_area.sum(),1e-15)),
            'fully_colored_faces':int(faces_covered.sum()),
            'fully_colored_area_fraction':float(face_area[faces_covered].sum()/max(face_area.sum(),1e-15)),
            'multi_view_vertices':int(multi.sum()),
            'conflicting_multi_view_fraction':float((attributes['conflict_count'][multi]>0).mean()) if multi.any() else 0.,
            'multi_view_color_delta_p50_p90':np.percentile(attributes['max_color_delta'][multi],[50,90]).tolist() if multi.any() else [0,0]},
        'limitations':['顶点着色基线，细节受顶点密度限制，尚未生成UV纹理',
            '蓝色标记的局部插值是估计颜色，不是RGB实测；不能补全网格几何孔洞或恢复缺失纹理',
            '颜色冲突可能来自曝光、光照、位姿或标定误差，本阶段不调整几何与位姿']}
    if not covered.any(): raise ValueError('没有可靠RGB覆盖：请检查源扫描、RGB标定及位姿对应关系')
    for path,value in hashes.items():
        if digest(path)!=value: raise RuntimeError('处理期间源文件发生改变：'+path)
    output.mkdir(parents=True,exist_ok=False)
    if not o3d.io.write_triangle_mesh(str(output/'mesh_colored.ply'),mesh,write_vertex_colors=True): raise OSError('着色网格保存失败')
    np.savez_compressed(output/'color_attributes.npz',**attributes)
    if preview_packet:
        np.savez(preview_packet,vertices=vertices.astype(np.float32),triangles=triangles.astype(np.uint32),
                 normals=normals.astype(np.float32),colors=colors.astype(np.float32))
    report.update(complete=True,source_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Colored {covered.sum()}/{len(vertices)} vertices -> {output}',flush=True)
    return report
