"""Read a triangle mesh in the isolated runtime; return a temporary viewer packet."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
from runtime_paths import activate_geometry_packages
activate_geometry_packages()

if __name__ == '__main__':
    import numpy as np
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(sys.argv[1])
    if not len(mesh.triangles): raise ValueError('文件不包含三角面，无法作为几何网格查看')
    mesh.compute_vertex_normals()
    colors = {'colors':np.asarray(mesh.vertex_colors,dtype=np.float32)} if mesh.has_vertex_colors() else {}
    np.savez(sys.argv[2], vertices=np.asarray(mesh.vertices, dtype=np.float32),
             triangles=np.asarray(mesh.triangles, dtype=np.uint32), normals=np.asarray(mesh.vertex_normals, dtype=np.float32), **colors)
