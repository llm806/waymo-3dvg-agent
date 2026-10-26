"""Deterministic display diagnostics using official point-to-camera pixel indices.

These checks establish artifact/identity consistency, not independent projection accuracy.
Only the browser previews are sampled; source arrays and training inputs are untouched.
"""

from functools import lru_cache
from itertools import product
from datetime import datetime, timezone

import numpy as np

from app.infrastructure.artifacts import checked_asset
from app.agent.console import load_packet
from app.datasets.evidence import ROOT
from app.quality.geometry import inside_oriented_box
from app.datasets.training import read_scene
from app.datasets.waymo import CAMERAS, CATEGORIES, O, P, V, WaymoReader


from app.datasets.correspondence import associate_camera_boxes, apply_recorded_point_links, verified_native_objects, official_projection_boxes
from app.datasets.camera_view import camera_views


def native_camera_boxes(packet, scene, arrays):
    reader = WaymoReader(ROOT / 'waymo_trainset')
    components = ['camera_box', 'camera_to_lidar_box_association', 'projected_lidar_box']
    for name in components:
        source = packet.provenance['components'][name]
        actual = reader.provenance(name, packet.context)
        if (actual['file'], actual['sha256']) != (source['file'], source['sha256']):
            raise ValueError('Native camera label source mismatch: ' + name)
    rows = reader.rows('camera_box', packet.context, timestamp=packet.frame_timestamp_micros)
    associations = reader.rows('camera_to_lidar_box_association', packet.context, timestamp=packet.frame_timestamp_micros)
    objects = verified_native_objects(reader, packet)
    return apply_recorded_point_links(associate_camera_boxes(rows, associations, packet.target_object_id, packet.target.category),
        packet, scene=scene, arrays=arrays, objects=list(objects.values()),
        projections=reader.rows('projected_lidar_box', packet.context, timestamp=packet.frame_timestamp_micros))


def target_projection_boxes(packet):
    reader = WaymoReader(ROOT / 'waymo_trainset')
    actual = reader.provenance('projected_lidar_box', packet.context)
    expected = packet.provenance['components']['projected_lidar_box']
    if (actual['file'], actual['sha256']) != (expected['file'], expected['sha256']):
        raise ValueError('Projected label source mismatch')
    result = {}
    for row in reader.rows('projected_lidar_box', packet.context, timestamp=packet.frame_timestamp_micros):
        if row[O] != packet.target_object_id:
            continue
        if row[V] in result:
            raise ValueError('Duplicate target projected camera key')
        center = np.array([row[f'{P}.box.center.{a}'] for a in 'xy'], dtype=float)
        size = np.array([row[f'{P}.box.size.{a}'] for a in 'xy'], dtype=float)
        if not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError('Invalid projected camera box')
        if CATEGORIES.get(row[f'{P}.type'], 'unknown') != packet.target.category:
            raise ValueError('Projected target category mismatch')
        result[row[V]] = np.concatenate((center - size / 2, center + size / 2)).tolist()
    return result


def sample_indices(indices, limit):
    return indices[np.linspace(0, len(indices) - 1, min(len(indices), limit), dtype=int)] if len(indices) else indices


def corners(box):
    signs = np.array(list(product([-1, 1], repeat=3)))
    local = signs * np.asarray(box.size_m) / 2
    c, s = np.cos(box.heading_rad), np.sin(box.heading_rad)
    local[:, :2] = local[:, :2] @ np.array([[c, s], [-s, c]])
    return (local + box.center_m).tolist()


def scene_objects(packet, xyz):
    """Frame-local observability; suitability is not a text quality certificate."""
    from app.quality.contrast import POLICY, contrast_plan
    from collections import Counter
    from app.datasets.waymo import clip_projection
    reader = WaymoReader(ROOT / 'waymo_trainset')
    projections = reader.rows('projected_lidar_box',packet.context,timestamp=packet.frame_timestamp_micros)
    calibrations = {r[V]:r for r in reader.metadata_rows('camera_calibration',packet.context)}
    result = []
    projection_counts=Counter((row[O],row[V]) for row in projections)
    for obj in packet.candidates:
        views = {}
        for row in projections:
            if row[O] != obj.object_id or projection_counts[(row[O],row[V])]!=1 or CATEGORIES.get(row[f'{P}.type'],'unknown')!=obj.category:
                continue
            calibration = calibrations.get(row[V])
            if not calibration:
                continue
            bounded = clip_projection(row,calibration['[CameraCalibrationComponent].width'],calibration['[CameraCalibrationComponent].height'])
            if bounded:
                views[str(row[V])] = list(bounded)
        count = int(inside_oriented_box(xyz,obj.center_m,obj.size_m,obj.heading_rad).sum())
        observable = obj.category != 'unknown' and count > 0 and bool(views)
        plan = contrast_plan(packet.candidates,obj.object_id,packet.predicates)
        result.append({'object_id':obj.object_id,'category':obj.category,'center_m':obj.center_m,
            'corners':corners(obj),'converted_points':count,'projected_boxes':views,
            'observable':observable,'annotation_target':obj.object_id == packet.target_object_id,
            'distinguishable':plan.status == 'unique_in_gt_candidates',
            'reason': 'ready_for_text_validation' if observable and plan.status == 'unique_in_gt_candidates' else
                'requires_relation_expansion' if observable else 'no_points_or_valid_view_or_known_category'})
    return result


# Retain cache_clear for diagnostics/tests, but re-verify every live read.
@lru_cache(maxsize=0)
def inspect_packet(packet_id: str, manifest_hash: str) -> dict:
    packet = load_packet(packet_id)
    if not packet.scene or packet.scene.manifest_sha256 != manifest_hash:
        raise ValueError("Scene manifest mismatch")
    path = checked_asset(ROOT, packet.scene.manifest_path, manifest_hash)
    scene, arrays = read_scene(path.parent)
    if (scene.context, scene.frame_timestamp_micros) != (packet.context, packet.frame_timestamp_micros):
        raise ValueError("Scene/evidence identity mismatch")
    xyz, projection = arrays["points"][:, :3], arrays["camera_projection"]
    inside = inside_oriented_box(xyz, packet.target.center_m, packet.target.size_m, packet.target.heading_rad)
    previews = sample_indices(np.flatnonzero(~inside), 18000)
    target_indices = sample_indices(np.flatnonzero(inside), 4000)
    camera_boxes = native_camera_boxes(packet, scene, arrays)
    projection_mode = packet.task.get('image_identity_policy') == 'official-projection-v1'
    reader = WaymoReader(ROOT / 'waymo_trainset')
    projected = official_projection_boxes(reader.rows('projected_lidar_box', packet.context, timestamp=packet.frame_timestamp_micros) if 'projected_lidar_box' in packet.provenance.get('components', {}) else [],
        packet.candidates, {c.camera_name: (c.width, c.height) for c in scene.cameras})
    display_views = camera_views(path.parent, scene)
    linked_cameras = [camera for camera,group in camera_boxes.items() if any(b['selected_target'] for b in group)]
    if projection_mode:
        linked_cameras = [camera for camera, group in projected.items() if any(b['object_id'] == packet.target_object_id for b in group)]
    target_boxes = target_projection_boxes(packet)
    views = []
    for camera in scene.cameras:
        evidence_view = next((v for v in packet.views if v.camera_name == camera.camera_name), None)
        pixel_parts, target_parts, point_parts, target_point_parts = [], [], [], []
        out_of_bounds, recorded = 0, 0
        for offset in (0, 3):
            valid = projection[:, offset] == camera.camera_name
            uv = projection[:, offset + 1:offset + 3]
            bounded = (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
            recorded += int(valid.sum())
            out_of_bounds += int((valid & ~bounded).sum())
            pixel_parts.append(uv[valid & bounded])
            target_parts.append(uv[valid & bounded & inside])
            point_parts.append(xyz[valid & bounded])
            target_point_parts.append(xyz[valid & bounded & inside])
        pixels = np.concatenate(pixel_parts)
        target_pixels = np.concatenate(target_parts)
        rect = target_boxes.get(camera.camera_name)
        in_rect, rect_inside = None, None
        if rect:
            x1, y1, x2, y2 = rect
            in_rect = ((target_pixels[:, 0] >= x1) & (target_pixels[:, 0] <= x2)
                       & (target_pixels[:, 1] >= y1) & (target_pixels[:, 1] <= y2))
            rect_inside = 0 <= x1 < x2 <= camera.width and 0 <= y1 < y2 <= camera.height
        pixel_sample = sample_indices(np.arange(len(pixels)), 5000)
        target_sample = sample_indices(np.arange(len(target_pixels)), 2000)
        views.append({"camera_name": camera.camera_name, "camera_label": CAMERAS[camera.camera_name],
                      "width": camera.width, "height": camera.height, "projected_box_xyxy": rect,
                      "image_sha256_matches": not evidence_view or camera.image.sha256 == evidence_view.source_jpeg_sha256,
                      "dimensions_match": not evidence_view or (camera.width, camera.height) == (evidence_view.width, evidence_view.height),
                      "rectangle_in_image": rect_inside, "projection_entries": recorded,
                      "out_of_bounds_entries": out_of_bounds, "target_projected_entries": len(target_pixels),
                      "target_entries_in_rectangle": int(in_rect.sum()) if in_rect is not None else None,
                      "target_in_rectangle_fraction": float(in_rect.mean()) if in_rect is not None and len(in_rect) else None,
                      "pose_minus_frame_ms": (camera.pose_timestamp_s - packet.frame_timestamp_micros / 1e6) * 1000
                      if camera.pose_timestamp_s is not None else None,
                      "trigger_time_s": camera.trigger_time_s, "readout_done_time_s": camera.readout_done_time_s,
                      "pixels": pixels[pixel_sample].tolist(), "target_pixels": target_pixels[target_sample].tolist(),
                      "scene_points": np.concatenate(point_parts)[pixel_sample].tolist(),
                      "target_points": np.concatenate(target_point_parts)[target_sample].tolist(),
                      "source_image": f"/api/evidence/{packet_id}/scene-assets/camera-{camera.camera_name}.jpg"})
    return {"status": "diagnostic_only", "independent_alignment_verified": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "integrity_verified": True, "identity_verified": True,
            "coordinate_frame": scene.coordinate_frame, "manifest_sha256": manifest_hash,
            "point_count": scene.point_count, "target_points": int(inside.sum()),
            "native_label_target_points": packet.target.lidar_points,
            "point_count_matches_packet": scene.point_count == packet.scene.total_points,
            "target_point_count_matches_packet": int(inside.sum()) == packet.scene.target_points,
            "sensor_returns": scene.sensor_returns, "conversion_profile": scene.conversion["profile"],
            "sources": scene.source_components, "views": views, "scene_objects": scene_objects(packet,xyz),
            "packet_id": packet_id, "context": packet.context, "frame_timestamp_micros": packet.frame_timestamp_micros,
            "target_object_id": packet.target_object_id,
            "source_view": packet.source_view,
            "target_pair_confirmed": bool(linked_cameras),
            "target_camera_names": linked_cameras,
            "default_camera_name": next((v.camera_name for v in packet.views if CAMERAS[v.camera_name] == packet.source_view), packet.views[0].camera_name if packet.views else 1),
            "camera_label_sources": {name: packet.provenance['components'][name]
                                     for name in ['camera_box', 'camera_to_lidar_box_association']},
            "image_box_source": "official_lidar_box_projection" if projection_mode else "native_camera_box",
            "all_cameras": [{"camera_name": c.camera_name, "width": c.width, "height": c.height,
                             "native_boxes": camera_boxes.get(c.camera_name, []),
                             "projected_boxes": projected.get(c.camera_name, []), "view_axes": display_views.get(c.camera_name),
                             "image": f"/api/evidence/{packet_id}/scene-assets/{c.image.path}"} for c in scene.cameras],
            "preview": {"sampling": "deterministic evenly spaced indices; browser only",
                        "points": xyz[previews].tolist(), "target_points": xyz[target_indices].tolist(),
                        "target_corners": corners(packet.target),
                        "candidate_boxes": [{"object_id": b.object_id, "corners": corners(b)} for b in packet.candidates]},
            "limitations": ["像素来自官方 camera_projection；这是对应关系展示，未重新计算独立投影误差。",
                            "原生 camera_box 是 2D 真值；projected_lidar_box 是 3D 标签的二维投影，两者不能混称。",
                            "身份使用官方唯一关联或重新核验的真实点像素对应；仅凭投影框交叠不关联目标。",
                            "原生投影矩形可延伸至图像边界之外；展示裁切不能单独作为对齐错误的判据。",
                            "目标点按帧时刻原生 3D 框筛选；相机曝光时刻、运动和遮挡可能影响矩形内比例。",
                            "原生框点数与转换后框内点数统计口径不同，差值不作为对齐通过标准。",
                            "预览采样不修改完整场景点云；未设置未经校准的对齐通过阈值。"]}
