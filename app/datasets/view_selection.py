"""Official same-ID projection seeds and legacy native-camera pairing helpers."""
from collections import defaultdict

from app.datasets.correspondence import potential_native_cameras
from app.datasets.waymo import CAMERAS, CATEGORIES, O, P, V
from app.quality.visibility_policy import allows_partial, source_camera_inventory

CAMERA_PRIORITY = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def projected_image_seeds(projections, objects, camera_sizes, calibrations=None, require_complete=False,
                          require_unobstructed=False, require_front_facing=True):
    """Official C/T/O/V projection keys identify the same 3D object directly."""
    from app.datasets.waymo import clip_projection
    result = defaultdict(list)
    by_id = {o.object_id: o for o in objects}
    counts = defaultdict(int)
    for row in projections:
        counts[(row[O], row[V])] += 1
    for row in projections:
        obj, camera = by_id.get(row[O]), row[V]
        if (obj is None or counts[(row[O], camera)] != 1 or camera not in camera_sizes
                or CATEGORIES.get(row[f'{P}.type']) != obj.category or not obj.geometry_valid):
            continue
        from app.datasets.visibility import sign_front_facing, foreground_occluders
        front = sign_front_facing(obj, (calibrations or {}).get(camera))
        if require_front_facing and front is False:
            continue
        rect = clip_projection(row, *camera_sizes[camera])
        if rect is None:
            continue
        width,height = camera_sizes[camera]
        boundary = rect[0] <= 0 or rect[1] <= 0 or rect[2] >= width or rect[3] >= height
        if require_complete and boundary:
            continue
        occluders = foreground_occluders(obj, rect, camera, projections, objects, (calibrations or {}).get(camera))
        if require_unobstructed and occluders:
            continue
        area = (rect[2]-rect[0])*(rect[3]-rect[1])
        raw_area = row[f'{P}.box.size.x'] * row[f'{P}.box.size.y']
        result[obj.object_id].append({'source_view': CAMERAS[camera], 'source_camera_id': camera,
            'source_camera_object_id': None, 'source_projected_box_xyxy': list(rect),
            'projected_box_area': area, 'visible_fraction': min(1.0,area/raw_area),
            'image_selection_score': area*min(1.0,area/raw_area), 'seed_overlap_score': 1,
            'foreground_object_ids': occluders,
            'target_front_facing': front,
            'image_identity_status': 'official_lidar_box_projection'})
    for choices in result.values():
        choices.sort(key=lambda s: (bool(s['foreground_object_ids']), s['visible_fraction'] < 1,
            s['target_front_facing'] is False,
            -s['image_selection_score'], CAMERA_PRIORITY[s['source_camera_id']]))
    return result


def image_seeds(boxes, projections, objects, camera_sizes):
    by_id = {obj.object_id: obj for obj in objects}
    result = defaultdict(list)
    for camera in sorted(boxes, key=lambda c: CAMERA_PRIORITY.get(c, 9)):
        if camera not in camera_sizes:
            continue
        width, height = camera_sizes[camera]
        for box in boxes[camera]:
            if box['association_status'] == 'ambiguous':
                continue
            x1, y1, x2, y2 = box['xyxy']
            rect = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
            if min(rect[2]-rect[0], rect[3]-rect[1]) < 24:
                continue
            ids = box['laser_object_ids']
            official = len(ids) == 1
            if not official:
                ids = sorted({p[O] for p in projections if p[V] == camera
                    and CATEGORIES.get(p[f'{P}.type']) == box['category']
                    and potential_native_cameras({camera: [box]}, [p], box['category'])})
            for oid in ids:
                obj = by_id.get(oid)
                if obj is None or obj.category != box['category'] or not obj.lidar_points:
                    continue
                if not official and obj.lidar_points < 20:
                    continue  # Derived identity cannot meet the 20-pixel proof floor.
                overlap_score = 1.0 if official else max((projection_iou(rect, p) for p in projections
                    if p[O] == oid and p[V] == camera), default=0.0)
                result[oid].append({'source_view': CAMERAS[camera], 'source_camera_id': camera,
                    'source_camera_object_id': box['camera_object_id'], 'source_native_box_xyxy': rect,
                    'native_box_area': (rect[2]-rect[0])*(rect[3]-rect[1]),
                    'seed_overlap_score': overlap_score,
                    'image_identity_status': 'official_native_association' if official else 'pending_point_confirmation'})
    for candidates in result.values():
        candidates.sort(key=lambda s: (s['source_camera_id'] not in {1, 2, 3},
            s['image_identity_status'] != 'official_native_association', -s['seed_overlap_score'], -s['native_box_area'],
            CAMERA_PRIORITY.get(s['source_camera_id'], 9), s['source_camera_object_id']))
    return result


def projection_iou(rect, projection):
    cx, cy = (projection[f'{P}.box.center.{a}'] for a in 'xy')
    w, h = (projection[f'{P}.box.size.{a}'] for a in 'xy')
    intersection = max(0, min(rect[2], cx+w/2)-max(rect[0], cx-w/2)) * max(0, min(rect[3], cy+h/2)-max(rect[1], cy-h/2))
    union = (rect[2]-rect[0])*(rect[3]-rect[1]) + w*h - intersection
    return intersection/union if union > 0 else 0.0


def confirmed_source_views(views, camera=None, camera_object_id=None, resolve_provisional=False,
                           require_complete=False, allow_back_facing=False):
    candidates = [v for v in views if (v.appearance_identity_source == 'official_lidar_box_projection'
        or v.appearance_camera_object_id and v.appearance_identity_source in {'official_camera_lidar_association','native_box_point_correspondence'})
        and (allow_back_facing or v.target_front_facing is not False)
        and (not require_complete or v.projection_touches_boundary is False)
        and 'appearance' in v.artifacts
        and (v.appearance_identity_source != 'native_box_point_correspondence' or v.appearance_projected_points >= 20)
        and (camera is None or v.camera_name == camera)
        and (camera_object_id is None or v.appearance_camera_object_id == camera_object_id or resolve_provisional)]
    candidates.sort(key=lambda v: (v.appearance_camera_object_id != camera_object_id,
        CAMERA_PRIORITY.get(v.camera_name,9), not v.most_visible_camera))
    return candidates


def select_source_view(packet, camera):
    """Bind a new source to already frozen same-frame artifacts and identities."""
    partial = allows_partial(packet)
    choices = confirmed_source_views(packet.views, camera, require_complete=not partial, allow_back_facing=partial)
    if not choices or str(camera) not in (source_camera_inventory(packet) or {}):
        raise ValueError('No eligible identity-checked source view')
    view = choices[0]
    packet = packet.model_copy(deep=True)
    packet.source_view, packet.source_camera_object_id = view.camera_label, view.appearance_camera_object_id
    packet.views.sort(key=lambda v: v.camera_name != camera)
    packet.provenance['source_image_selection'] = {'camera_name': camera,
        'camera_object_id': view.appearance_camera_object_id, 'image_box_xyxy': view.appearance_box_xyxy,
        'identity_source': view.appearance_identity_source}
    packet.provenance['source_image_candidates'] = [c for c in packet.provenance.get('scene_image_candidates', [])
        if c['camera_name'] == camera]
    packet.provenance.pop('admission_plan', None)
    packet.ready_for_generation = True
    return packet
