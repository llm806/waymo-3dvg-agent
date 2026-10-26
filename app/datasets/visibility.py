"""Geometric view suitability, separate from actual RGB visibility."""
import math
import numpy as np

VIEW_SCREENING_POLICY = 'recognizable-projection-ranking-v3'
LEGACY_VIEW_SCREENING_POLICY = 'potential-foreground-overlap-v2'


def foreground_occluders(target, rect, camera, projections, objects, calibration):
    """Conservative proposal screening, never a certificate of RGB visibility.

    Return overlapping native boxes that could be in front. The recognizable
    policy uses this risk to rank views, not to decide actual RGB identity.
    Only a box entirely behind the target can be dismissed geometrically.
    Overlapping depth intervals are ambiguous, not evidence of a clear view.
    All objects remain in the grounding competition pool.
    Camera +x is forward in Waymo; oriented extents give depth intervals.
    """
    from app.datasets.waymo import O, P, V
    values = (calibration or {}).get('[CameraCalibrationComponent].extrinsic.transform')
    if values is None:
        return []  # Legacy metadata fixtures; actual admission still checks RGB.
    transform = np.asarray(values, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid camera-to-vehicle transform')
    forward, origin = rotation[:, 0], transform[:3, 3]

    def depth(obj):
        angle = obj.heading_rad
        axes = np.array([[math.cos(angle), math.sin(angle), 0],
                         [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        radius = float(np.abs(axes @ forward) @ (np.asarray(obj.size_m) / 2))
        center = float((np.asarray(obj.center_m) - origin) @ forward)
        return center - radius, center + radius

    _, far = depth(target)
    by_id = {obj.object_id: obj for obj in objects}
    blocked = set()
    for projection in projections:
        other = by_id.get(projection[O])
        if (projection[V] != camera or other is None or other.object_id == target.object_id
                or not other.geometry_valid):
            continue
        other_near, other_far = depth(other)
        if other_far <= 0 or other_near > far + 0.2:
            continue
        cx, cy = (projection[f'{P}.box.center.{a}'] for a in 'xy')
        width, height = (projection[f'{P}.box.size.{a}'] for a in 'xy')
        overlap = (max(0, min(rect[2], cx + width/2) - max(rect[0], cx - width/2))
                   * max(0, min(rect[3], cy + height/2) - max(rect[1], cy - height/2)))
        if overlap > 0:
            blocked.add(other.object_id)
    return sorted(blocked)


def sign_front_facing(obj, calibration):
    """Waymo sign heading faces the information side; the reverse is unsuitable
    for this clear-view training policy. This is not an occlusion test.
    """
    if obj.category != 'sign':
        return None
    values = (calibration or {}).get('[CameraCalibrationComponent].extrinsic.transform')
    if values is None:
        return None
    transform = np.asarray(values, dtype=float).reshape(4, 4)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0,0,0,1]):
        raise ValueError('Invalid camera-to-vehicle transform')
    toward_camera = transform[:2, 3] - np.asarray(obj.center_m[:2])
    normal = np.array([math.cos(obj.heading_rad), math.sin(obj.heading_rad)])
    return bool(toward_camera @ normal > 0)
