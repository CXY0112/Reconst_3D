"""One-shot PLY/PCD reader for the isolated Open3D Python runtime."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from runtime_paths import activate_geometry_packages
activate_geometry_packages()
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cloud=o3d.io.read_point_cloud(sys.argv[1],remove_nan_points=False,remove_infinite_points=False)
    points=np.asarray(cloud.points)
    if not len(points):
        raise ValueError('点云为空或文件格式无效')
    np.save(sys.argv[2],points)
