"""Native camera/LiDAR identity joins shared by selection, preparation and display."""
import numpy as np
from app.datasets.waymo import CATEGORIES, O, V


def potential_native_cameras(boxes, projections, category):
    """Cheap metadata candidates only; actual points must confirm them later."""
    from app.datasets.waymo import P
    cameras = set()
    for projection in projections:
        cx, cy = (projection[f'{P}.box.center.{a}'] for a in 'xy')
        sx, sy = (projection[f'{P}.box.size.{a}'] for a in 'xy')
        for box in boxes.get(projection[V], []):
            if box['category'] != category or box['association_status'] == 'ambiguous':
                continue
            x1, y1, x2, y2 = box['xyxy']
            overlap = max(0, min(x2, cx+sx/2)-max(x1, cx-sx/2)) * max(0, min(y2, cy+sy/2)-max(y1, cy-sy/2))
            if min(x2-x1, y2-y1) >= 24 and overlap >= .1*(x2-x1)*(y2-y1):
                cameras.add(projection[V])
    return cameras


def native_evidence_cameras(packet):
    """A crop must retain an official 3D projection ID or a verified native 2D link."""
    return [v.camera_name for v in packet.views
            if v.appearance_identity_source in {'official_camera_lidar_association', 'native_box_point_correspondence', 'official_lidar_box_projection'}
            and (v.appearance_identity_source != 'native_box_point_correspondence' or v.appearance_projected_points >= 20)
            and (v.appearance_camera_object_id or v.appearance_identity_source == 'official_lidar_box_projection') and v.appearance_box_xyxy
            and 'appearance' in v.artifacts]


def has_multimodal_evidence(packet):
    valid_cameras = native_evidence_cameras(packet)
    if packet.task.get('image_first'):
        from app.datasets.waymo import CAMERAS
        valid_cameras = [v.camera_name for v in packet.views if v.camera_name in valid_cameras
            and CAMERAS.get(v.camera_name) == packet.source_view
            and v.appearance_camera_object_id == packet.source_camera_object_id]
    if packet.task.get('visibility_policy') == 'clear-source-rgb-v1':
        valid_cameras = [v.camera_name for v in packet.views if v.camera_name in valid_cameras and v.projection_touches_boundary is False]
    if packet.task.get('visibility_policy') == 'clear-source-rgb-v1' and packet.target.category == 'sign':
        valid_cameras = [v.camera_name for v in packet.views if v.camera_name in valid_cameras and v.target_front_facing is True]
    return bool(packet.target.lidar_points and packet.target.lidar_points > 0
                and packet.scene and packet.scene.target_points > 0
                and valid_cameras)


def verified_native_objects(reader, packet):
    """Bind displayed target and candidate geometry to the original frame rows."""
    from app.datasets.waymo import object_fact
    expected = packet.provenance.get('components', {}).get('lidar_box')
    actual = reader.provenance('lidar_box', packet.context)
    if not expected or (actual['file'], actual['sha256']) != (expected['file'], expected['sha256']):
        raise ValueError('Native 3D label source mismatch')
    facts = [object_fact(r) for r in reader.rows('lidar_box', packet.context, timestamp=packet.frame_timestamp_micros)]
    by_id = {o.object_id: o for o in facts}
    if (len(by_id) != len(facts) or len(packet.candidates) != len(by_id)
            or set(by_id) != {o.object_id for o in packet.candidates}
            or packet.target.object_id != packet.target_object_id):
        raise ValueError('Native 3D target/candidate identity mismatch')
    for obj in [*packet.candidates, packet.target]:
        native = by_id.get(obj.object_id)
        if native is None or (native.center_m, native.size_m, native.heading_rad, native.category) != (
                obj.center_m, obj.size_m, obj.heading_rad, obj.category):
            raise ValueError('3D box/category differs from the raw native frame')
    return by_id


def apply_recorded_point_links(boxes, packet, *, scene, arrays, objects, projections=None):
    """Recompute point links before display; a saved support count is not proof."""
    import math
    from app.datasets.waymo import CAMERAS
    from app.quality.geometry import inside_oriented_box
    if (scene.context, scene.frame_timestamp_micros) != (packet.context, packet.frame_timestamp_micros):
        raise ValueError('Scene/evidence frame mismatch')
    count = int(inside_oriented_box(arrays['points'][:, :3], packet.target.center_m,
        packet.target.size_m, packet.target.heading_rad).sum())
    if (not packet.scene or scene.point_count != packet.scene.total_points
            or count != packet.scene.target_points):
        raise ValueError('Scene/target point counts differ from the evidence')
    mapped = point_supported_native_boxes(boxes, objects, arrays, packet.target_object_id)
    cameras = {c.camera_name: c for c in scene.cameras}
    source_confirmed = False
    for view in packet.views:
        if view.appearance_identity_source not in {'official_camera_lidar_association', 'native_box_point_correspondence', 'official_lidar_box_projection'}:
            continue
        camera = cameras.get(view.camera_name)
        if (camera is None or view.camera_label != CAMERAS[view.camera_name]
                or view.source_jpeg_sha256 != camera.image.sha256
                or (view.width, view.height) != (camera.width, camera.height)):
            raise ValueError('Target image identity/dimensions differ from the scene')
        if view.appearance_identity_source == 'official_lidar_box_projection':
            from app.datasets.waymo import O, V, P, clip_projection
            rows = [r for r in (projections or []) if r[O] == packet.target_object_id and r[V] == view.camera_name]
            if len(rows) != 1 or CATEGORIES.get(rows[0][f'{P}.type']) != packet.target.category:
                raise ValueError('Official projection target ID/category is missing or ambiguous')
            row = rows[0]
            cx,cy = (row[f'{P}.box.center.{a}'] for a in 'xy')
            sx,sy = (row[f'{P}.box.size.{a}'] for a in 'xy')
            if (view.projected_box_xyxy != (cx-sx/2,cy-sy/2,cx+sx/2,cy+sy/2)
                    or view.appearance_box_xyxy != clip_projection(row,camera.width,camera.height)
                    or view.appearance_camera_object_id is not None):
                raise ValueError('Official projected target geometry differs from the raw label')
            if view.camera_label == packet.source_view and packet.source_camera_object_id is None:
                source_confirmed = True
            continue
        linked = [b for b in mapped.get(view.camera_name, []) if b['selected_target']]
        if len(linked) != 1 or linked[0]['camera_object_id'] != view.appearance_camera_object_id:
            raise ValueError('Recorded 2D/3D target link is not supported by current native evidence')
        box = linked[0]
        rect = box['xyxy']
        crop = (max(0, math.floor(rect[0])), max(0, math.floor(rect[1])),
                min(camera.width, math.ceil(rect[2])), min(camera.height, math.ceil(rect[3])))
        if (box.get('identity_source', 'official_camera_lidar_association') != view.appearance_identity_source
                or crop != view.appearance_box_xyxy
                or view.appearance_identity_source == 'native_box_point_correspondence'
                   and box.get('support_points') != view.appearance_projected_points):
            raise ValueError('Recorded target crop/point support differs from the native evidence')
        if view.camera_label == packet.source_view and view.appearance_camera_object_id == packet.source_camera_object_id:
            source_confirmed = True
    if packet.task.get('image_first') and not source_confirmed:
        raise ValueError('Source camera/native target identity could not be verified')
    return mapped


def official_projection_boxes(rows, objects, camera_sizes):
    """Image rectangles retain native 3D IDs; never invent camera_box IDs."""
    from app.datasets.waymo import O, V, P, clip_projection
    by_id = {o.object_id: o for o in objects}
    result, seen = {}, set()
    for row in rows:
        key = (row[V], row[O])
        if key in seen:
            raise ValueError('Duplicate official projection key')
        seen.add(key)
        obj = by_id.get(row[O])
        if obj is None or obj.category != CATEGORIES.get(row[f'{P}.type']):
            raise ValueError('Official projection has no matching native 3D box/category')
        if row[V] not in camera_sizes:
            continue
        rect = clip_projection(row, *camera_sizes[row[V]])
        if rect is not None:
            result.setdefault(row[V], []).append({'object_id': obj.object_id, 'category': obj.category,
                'xyxy': list(rect), 'identity_source': 'official_lidar_box_projection'})
    return result


def point_supported_native_boxes(boxes, objects, arrays, target_id):
    """Conservative derived matches; never relabel them as official ID links.

    Use actual 3D-box points and their dataset pixel indices, not projected-box
    IoU. Reject competing same-category/unknown objects, overlapping labels and
    weak pixel support. Other known categories may occlude the unit; they do not
    compete for its class-specific identity. RGB visibility is checked later.
    Thresholds are prototype rules, not a calibrated accuracy certificate.
    """
    if arrays is None:
        return boxes
    from app.quality.geometry import BoxPointIndex
    from collections import Counter
    result = {camera: [dict(b) for b in group] for camera, group in boxes.items()}
    xyz = arrays['points'][:, :3]
    point_index = BoxPointIndex(xyz)
    owner = np.full(len(xyz), -1, dtype=np.int32)
    hits = np.zeros(len(xyz), dtype=np.int16)
    for index, obj in enumerate(objects):
        if not obj.geometry_valid:
            continue
        mask = point_index.inside(obj.center_m, obj.size_m, obj.heading_rad)
        hits += mask
        owner[mask] = index
    owner[hits != 1] = -1
    projection = arrays['camera_projection']
    for camera, group in result.items():
        samples = []
        for offset in (0, 3):
            valid = (projection[:, offset] == camera) & (owner >= 0)
            samples.append(np.column_stack((owner[valid], projection[valid, offset+1:offset+3])))
        samples = np.unique(np.concatenate(samples), axis=0)
        totals = Counter(samples[:, 0].astype(int))
        proposals, unit_groups = [], []
        for box_index, box in enumerate(group):
            if box['laser_object_ids'] or box['association_status'] != 'unlinked':
                continue
            x1, y1, x2, y2 = box['xyxy']
            inside = samples[(samples[:, 1] >= x1) & (samples[:, 1] < x2)
                             & (samples[:, 2] >= y1) & (samples[:, 2] < y2)]
            counts = Counter(inside[:, 0].astype(int))
            compatible_points = sum(count for index, count in counts.items()
                if objects[index].category in {box['category'], 'unknown'})

            # Rectangular RGB labels overlap even when the physical cars do not.
            # Test the part unique to this label too; pixels inside another
            # same-class label cannot establish which rectangle owns the unit.
            exclusive = np.ones(len(inside), dtype=bool)
            for other in group:
                if other is box or other['category'] not in {box['category'], 'unknown'}:
                    continue
                u, v, w, h = other['xyxy']
                exclusive &= ~((inside[:, 1] >= u) & (inside[:, 1] < w)
                    & (inside[:, 2] >= v) & (inside[:, 2] < h))
            exclusive_counts = Counter(inside[exclusive, 0].astype(int))
            exclusive_total = sum(count for index, count in exclusive_counts.items()
                if objects[index].category in {box['category'], 'unknown'})

            def only_boundary_slivers(target_index):
                # A foreground car can contribute a narrow edge strip inside
                # another car's native RGB rectangle. Its partial overlap is
                # not a second full-label identity. Unknown objects, substantial
                # rival coverage or pixels away from the boundary block this.
                for other, count in counts.items():
                    if other == target_index or objects[other].category not in {box['category'], 'unknown'}:
                        continue
                    if objects[other].category == 'unknown' or count / totals[other] >= .2:
                        return False
                    pixels = inside[inside[:, 0] == other, 1:]
                    edge = ((pixels.max(axis=0) <= np.array([x1, y1]) + .2*np.array([x2-x1, y2-y1]))
                        | (pixels.min(axis=0) >= np.array([x2, y2]) - .2*np.array([x2-x1, y2-y1])))
                    if not edge.any():
                        return False
                return True

            members = []
            for index, count in counts.items():
                obj = objects[index]
                if obj.category != box['category'] or count < 20:
                    continue
                target_pixels = inside[inside[:, 0] == index, 1:]
                total = totals[index]
                span = np.ptp(target_pixels, axis=0)
                if count / total < .8 or min(span) < 8 or np.prod(span) < .02 * (x2-x1) * (y2-y1):
                    continue
                members.append((index, count))
                exposed_pixels = inside[exclusive & (inside[:, 0] == index), 1:]
                exposed_span = np.ptp(exposed_pixels, axis=0) if len(exposed_pixels) else np.zeros(2)
                exclusive_support = (exclusive_counts[index] >= 20
                    and exclusive_counts[index] / exclusive_total >= .9
                    and min(exposed_span) >= 8
                    and np.prod(exposed_span) >= .02 * (x2-x1) * (y2-y1)
                    and not any(objects[other].category == 'unknown' for other in counts))
                full_region_support = (count / total >= .9
                    and np.all(span >= .8*np.array([x2-x1, y2-y1]))
                    and (exclusive_support or only_boundary_slivers(index)))
                # An occluded unit need not fill 80% of its camera rectangle.
                # Its exposed pixels can independently establish the link:
                # >=20 uncontested owned pixels, >=90% exclusive purity,
                # >=80% of all its camera returns and the same span/area guards
                # above. Ambiguous labels/unknown native owners still block it.
                if count / compatible_points >= .9 or exclusive_support or full_region_support:
                    proposals.append((box_index, index, count))
            # Some machines have separate native boxes for boom and chassis,
            # while their camera annotation covers the whole assembly. Preserve
            # ALL native units; this is a set-valued coverage link, not an ID merge.
            if len(members) >= 2 and sum(n for _, n in members) / compatible_points >= .9:
                unit_groups.append((box_index, members))
        object_counts = Counter(index for _, index, _ in proposals)
        box_counts = Counter(box_index for box_index, _, _ in proposals)
        official_objects = {oid for b in group for oid in b['laser_object_ids']}
        for box_index, index, count in proposals:
            obj = objects[index]
            if object_counts[index] != 1 or box_counts[box_index] != 1 or obj.object_id in official_objects:
                continue
            result[camera][box_index].update(laser_object_ids=[obj.object_id],
                selected_target=obj.object_id == target_id, association_status='point_supported',
                identity_source='native_box_point_correspondence', support_points=count)
        group_counts = Counter(index for _, members in unit_groups for index, _ in members)
        for box_index, members in unit_groups:
            if any(group_counts[index] != 1 or objects[index].object_id in official_objects
                    or object_counts[index] for index, _ in members):
                continue
            result[camera][box_index].update(
                identity_source='native_camera_box_group_point_correspondence',
                native_unit_object_ids=sorted(objects[index].object_id for index, _ in members),
                native_unit_point_support={objects[index].object_id: count for index, count in members})
    return result


def associate_camera_boxes(rows, associations, target_id, target_category):
    """Never infer a camera/laser identity from position or a matching category."""
    links = {}
    for row in associations:
        key = (row[V], row['key.camera_object_id'])
        links.setdefault(key, set()).add(row[O])
    target_keys = {}
    object_keys = {}
    for key, ids in links.items():
        for oid in ids: object_keys.setdefault((key[0],oid),set()).add(key)
        if target_id in ids:
            target_keys.setdefault(key[0], set()).add(key)
    seen, result = set(), {}
    for row in rows:
        key = (row[V], row['key.camera_object_id'])
        if key in seen:
            raise ValueError('Duplicate native camera box key')
        seen.add(key)
        center = np.array([row[f'[CameraBoxComponent].box.center.{a}'] for a in 'xy'], dtype=float)
        size = np.array([row[f'[CameraBoxComponent].box.size.{a}'] for a in 'xy'], dtype=float)
        if not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError('Invalid native camera box geometry')
        category = CATEGORIES.get(row['[CameraBoxComponent].type'], 'unknown')
        associated = links.get(key, set())
        ambiguous = len(associated) > 1 or any(len(object_keys.get((key[0],oid),set()))>1 for oid in associated)
        is_target = not ambiguous and associated == {target_id} and category == target_category
        result.setdefault(row[V], []).append({
            'camera_object_id': key[1], 'category': category,
            'xyxy': np.concatenate((center - size / 2, center + size / 2)).tolist(),
            'selected_target': is_target, 'laser_object_ids': sorted(associated),
            'association_status': 'linked_target' if is_target else
                'ambiguous' if ambiguous else
                'category_mismatch' if target_id in associated else
                'linked_other' if associated else 'unlinked',
        })
    return result

