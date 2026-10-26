"""Display camera axes from official camera-to-vehicle calibration.

This static preview is not a rolling-shutter reprojection accuracy certificate.
The authoritative point/image correspondences remain camera_projection pixels.
"""
import json
import numpy as np

from app.datasets.waymo import V
from app.infrastructure.artifacts import checked_asset


def camera_views(folder, scene):
    data = json.loads(checked_asset(folder, scene.calibration.path, scene.calibration.sha256).read_text(encoding='utf-8'))
    result = {}
    for row in data.get('camera_calibration', []):
        camera = row[V]
        if camera in result:
            raise ValueError('Duplicate camera calibration')
        prefix = '[CameraCalibrationComponent]'
        transform = np.asarray(row[f'{prefix}.extrinsic.transform'], dtype=float).reshape(4, 4)
        rotation = transform[:3, :3]
        if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError('Invalid camera-to-vehicle calibration')
        intrinsic = [float(row[f'{prefix}.intrinsic.{key}']) for key in ('f_u', 'f_v', 'c_u', 'c_v', 'k1', 'k2', 'p1', 'p2', 'k3')]
        width, height = row[f'{prefix}.width'], row[f'{prefix}.height']
        if not np.isfinite(intrinsic).all() or min(intrinsic[:2]) <= 0 or min(width, height) <= 0:
            raise ValueError('Invalid camera image calibration')
        matching = next((c for c in scene.cameras if c.camera_name == camera), None)
        if matching is None or (matching.width, matching.height) != (width, height):
            raise ValueError('Display camera dimensions differ from the scene')
        result[camera] = dict(right=(-rotation[:, 1]).tolist(), up=rotation[:, 2].tolist(),
            forward=rotation[:, 0].tolist(), origin=transform[:3, 3].tolist(),
            intrinsic=intrinsic, width=width, height=height)
    return result
