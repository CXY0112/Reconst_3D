"""Explicit optical camera frames for capture geometry (XYZ metres, images mm)."""
import numpy as np

SCHEMA = 'rgbd_coordinates_v2'


def camera_frame(value):
    if value not in ('ir', 'rgb'):
        raise ValueError(f'Unknown RGB-D coordinate frame: {value!r}')
    return value


def intrinsics(calibration, frame):
    frame = camera_frame(frame)
    return tuple(getattr(calibration, n+frame) for n in ('fx','fy','cx','cy'))


def ir_to_rgb(points, calibration):
    rotation = np.array([[getattr(calibration,f'R{i}{j}') for j in range(3)] for i in range(3)])
    translation = np.array([calibration.T1,calibration.T2,calibration.T3])/1000
    if (not np.isfinite(rotation).all() or not np.isfinite(translation).all() or
            not np.allclose(rotation@rotation.T,np.eye(3),atol=.05) or
            not np.isclose(np.linalg.det(rotation),1,atol=.05)):
        raise ValueError('Invalid SDK IR-to-RGB extrinsic')
    return np.asarray(points)@rotation.T+translation


def rgb_geometry(points, calibration, frame):
    return ir_to_rgb(points,calibration) if camera_frame(frame)=='ir' else points


def capture_geometry(raw_depth, calibration, sdk_cloud=None, *, depth_frame='ir', points_frame='rgb'):
    """SDK XYZ are mm; HP60C registered mode uses RGB for BOTH SDK outputs.

    Generic raw depth retains the legacy IR default. Callers must pass their
    camera's declared depth frame, including when a synchronized SDK cloud is absent.
    """
    camera_frame(depth_frame);camera_frame(points_frame)
    if sdk_cloud is None:
        # Geometry-only reconstruction workers need frame metadata without
        # importing the camera/OpenCV runtime.
        from reconstruct import depth_points
        return depth_points(raw_depth,calibration,stride=1,camera=depth_frame),depth_frame
    points = np.asarray(sdk_cloud,dtype=float).reshape(-1,3)/1000
    return points[np.isfinite(points).all(axis=1)&(points[:,2]>=.15)&(points[:,2]<=4)],points_frame


def coordinate_metadata(depth_frame, points_frame):
    return dict(coordinate_schema=SCHEMA,raw_depth_frame=camera_frame(depth_frame),
                pointcloud_frame=camera_frame(points_frame),registered_depth_frame='rgb',map_frame='rgb')


def recorded_coordinates(row):
    """Unlabelled historical files keep their historical interpretation at readers."""
    schema=row.get('coordinate_schema')
    if schema is None:return None
    if schema!=SCHEMA:raise ValueError(f'Unsupported coordinate schema: {schema!r}')
    if row.get('registered_depth_frame')!='rgb' or row.get('map_frame')!='rgb':
        raise ValueError('Registered depth and map poses must use RGB camera coordinates')
    return coordinate_metadata(row.get('raw_depth_frame'),row.get('pointcloud_frame'))
