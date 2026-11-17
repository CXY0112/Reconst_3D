"""Isolated geometry-only reconstruction, or mesh preview."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import activate_geometry_packages
activate_geometry_packages()


def preview(source,packet):
    import numpy as np
    import open3d as o3d
    if Path(source).suffix.lower()=='.glb':
        import trimesh
        loaded=trimesh.load(str(source),force='scene')
        pieces=[]
        for node in loaded.graph.nodes_geometry:
            transform,name=loaded.graph[node];mesh=loaded.geometry[name].copy();mesh.apply_transform(transform)
            if hasattr(mesh.visual,'to_color'):mesh.visual=mesh.visual.to_color()
            pieces.append(mesh)
        if not pieces:raise ValueError('GLB不包含三角网格')
        joined=trimesh.util.concatenate(pieces)
        vertices=np.asarray(joined.vertices);faces=np.asarray(joined.faces);normals=np.asarray(joined.vertex_normals)
        colors=np.asarray(joined.visual.vertex_colors[:,:3],float)/255
    else:
        mesh=o3d.io.read_triangle_mesh(str(source));mesh.compute_vertex_normals()
        vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);normals=np.asarray(mesh.vertex_normals)
        colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or normals.shape!=vertices.shape or
        faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(vertices) or
        not np.isfinite(vertices).all() or not np.isfinite(normals).all()):raise ValueError('文件不包含有效三角网格')
    if colors is not None and (colors.shape!=vertices.shape or not np.isfinite(colors).all() or np.any((colors<0)|(colors>1))):
        raise ValueError('网格颜色无效')
    extra={} if colors is None else dict(colors=colors.astype(np.float32))
    np.savez(packet,vertices=vertices.astype(np.float32),triangles=faces.astype(np.uint32),normals=normals.astype(np.float32),**extra)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--session',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--method',choices=('poisson','tsdf','nksr'),default='poisson')
    p.add_argument('--smoothing-iterations',type=int,default=30)
    p.add_argument('--read',type=Path);p.add_argument('--packet',type=Path)
    a=p.parse_args()
    if a.read:preview(a.read,a.packet);return
    from reconstruction_pipeline import run_pipeline as geometry
    from scene_provenance import point_source_fingerprint as fingerprint
    def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.perf_counter();source=a.source.resolve();session=a.session.resolve() if a.session else None;out=a.output.resolve()
    before=fingerprint(source);raw_before=fingerprint(session) if session else None
    print('场景几何重建…',flush=True);geometry(source,out/'geometry',method=a.method,smoothing_iterations=a.smoothing_iterations)
    geo=json.loads((out/'geometry/report.json').read_text(encoding='utf-8'))
    geo['input']=os.path.relpath(source,out/'geometry');geo['source_session']=os.path.relpath(session,out/'geometry') if session else None
    write_json(out/'geometry/report.json',geo)
    if fingerprint(source)!=before or (session and fingerprint(session)!=raw_before):raise ValueError('重建期间来源点云或RGB-D发生变化')
    write_json(out/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,stage=3,method=a.method,
        source_relative=os.path.relpath(source,out),source_session_relative=os.path.relpath(session,out) if session else None,
        source_fingerprint=before,rgbd_observation_fingerprint=raw_before,model_relative='geometry/mesh.ply',geometry_only=True,
        geometry_relative='geometry/mesh.ply',coordinates='world X forward Y left Z up',units='metres',
        elapsed_seconds=time.perf_counter()-start,geometry=geo['final']))
    print('基础几何模型已完成',flush=True)


if __name__=='__main__':main()
