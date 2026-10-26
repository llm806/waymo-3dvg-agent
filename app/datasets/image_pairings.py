"""Image-native selection pool backed by verified same-frame 2D/3D identities.

Proof metadata is dataset-owned, not task-owned. Recreating a deleted scene must
reconfirm the frozen image box before a task can generate any text.
"""
import hashlib
import json
from collections import defaultdict

from app.infrastructure.artifacts import atomic_json, checked_asset, file_sha256
from app.datasets.correspondence import associate_camera_boxes, point_supported_native_boxes
from app.datasets.training import read_scene
from app.datasets.waymo import CAMERAS, O, T, V, P, object_fact, clip_projection
from app.quality.geometry import inside_oriented_box
from app.quality.contrast import contrast_plan, predicate_catalog


def visual_priority(pair):
    camera = pair.get('source_camera_id', 1)
    return (camera not in {1, 2, 3}, camera != 1, pair.get('category') != 'vehicle',
        -pair.get('image_selection_score', pair.get('native_box_area', 0)),
        -pair.get('native_box_area', 0), -pair.get('lidar_points', 0),
        pair.get('timestamp', 0), pair.get('source_camera_object_id', ''))


def frame_image_pairs(objects, boxes, arrays, camera_sizes, projections, timestamp):
    """Iterate native image boxes first; unresolved identities never enter the pool."""
    boxes = point_supported_native_boxes(boxes, objects, arrays, '')
    by_id = {o.object_id:o for o in objects}
    predicates = predicate_catalog(objects)
    counts, pairs = {}, []
    for camera, group in boxes.items():
        if camera not in camera_sizes:
            continue
        width, height = camera_sizes[camera]
        for box in group:
            ids = box['laser_object_ids']
            if len(ids) != 1 or box['association_status'] in {'ambiguous','category_mismatch'}:
                continue
            obj = by_id.get(ids[0])
            if obj is None or obj.category != box['category'] or not obj.geometry_valid or not obj.lidar_points:
                continue
            projected = [p for p in projections if p[O] == obj.object_id and p[V] == camera]
            if len(projected) != 1 or clip_projection(projected[0],width,height) is None:
                continue
            x1,y1,x2,y2 = box['xyxy']
            rect = [max(0,x1),max(0,y1),min(width,x2),min(height,y2)]
            if min(rect[2]-rect[0],rect[3]-rect[1]) < 24:
                continue
            area = (rect[2]-rect[0])*(rect[3]-rect[1])
            visible_fraction = area/((x2-x1)*(y2-y1))
            if visible_fraction < .85:
                continue  # Avoid large but mostly off-image slivers.
            if obj.object_id not in counts:
                counts[obj.object_id] = int(inside_oriented_box(arrays['points'][:,:3],obj.center_m,obj.size_m,obj.heading_rad).sum())
            if counts[obj.object_id] <= 0:
                continue
            occlusion = 0
            for other in group:
                if other is box or other['category'] != box['category']:
                    continue
                a,b,c,d = other['xyxy']
                overlap = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
                if overlap/area > .1:
                    occlusion += 1
            plan = contrast_plan(objects,obj.object_id,predicates)
            pairs.append({'object_id':obj.object_id,'category':obj.category,'timestamp':timestamp,
                'source_view':CAMERAS[camera],'source_camera_id':camera,
                'source_camera_object_id':box['camera_object_id'],'source_native_box_xyxy':rect,
                'native_box_area':area,'visible_fraction':visible_fraction,'overlapping_same_class_boxes':occlusion,
                'image_selection_score':area*visible_fraction/(1+occlusion),
                'image_identity_status':box.get('identity_source','official_native_association'),
                'pairing_status':'confirmed','lidar_points':counts[obj.object_id],
                'support_pixels':box.get('support_points',0),'valid_views':1,
                'safe_unary_unique':plan.status=='unique_in_gt_candidates','safe_spatial_unique':plan.status=='unique_in_gt_candidates',
                'remaining_candidates':len(plan.remaining_object_ids),
                'ego_direction':('front' if obj.center_m[0]>1 else 'behind' if obj.center_m[0]<-1 else 'side')+'-'+
                    ('left' if obj.center_m[1]>1 else 'right' if obj.center_m[1]<-1 else 'center')})
    return pairs


def refresh_pairing_pool(reader, root):
    pool_root = root/'runtime/image_pairings'
    tool_hash = hashlib.sha256(b''.join((root/'app'/p).read_bytes() for p in
        ['datasets/image_pairings.py','datasets/correspondence.py','datasets/waymo.py','quality/geometry.py','quality/contrast.py'])).hexdigest()
    records = {}
    for path in sorted(pool_root.glob('*/pairs.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('tool_sha256') == tool_hash and all(reader.provenance(k,record['context'])==v for k,v in record['sources'].items()):
            records[record['scene_id']] = record
    index_path = root/'runtime/scenes/index.json'
    scenes = json.loads(index_path.read_text(encoding='utf-8'))['scenes'] if index_path.exists() else []
    for entry in scenes:
        if entry['scene_id'] in records:
            continue
        scene_path = checked_asset(root,entry['path'],entry['manifest_sha256'])
        scene,arrays = read_scene(scene_path.parent)
        context,ts = scene.context,scene.frame_timestamp_micros
        sources = {k:reader.provenance(k,context) for k in set(scene.source_components)|{'camera_box','camera_to_lidar_box_association','projected_lidar_box'}}
        if any(sources[k] != v for k,v in scene.source_components.items()):
            raise ValueError('Image pairing scene source changed')
        objects = [object_fact(r) for r in reader.metadata_rows('lidar_box',context) if r[T]==ts]
        boxes = associate_camera_boxes([r for r in reader.metadata_rows('camera_box',context) if r[T]==ts],
            [r for r in reader.metadata_rows('camera_to_lidar_box_association',context) if r[T]==ts],'','')
        projections = [r for r in reader.metadata_rows('projected_lidar_box',context) if r[T]==ts]
        pairs = frame_image_pairs(objects,boxes,arrays,{c.camera_name:(c.width,c.height) for c in scene.cameras},projections,ts)
        record = {'scene_id':scene.scene_id,'context':context,'manifest_sha256':file_sha256(scene_path),
            'tool_sha256':tool_hash,'sources':sources,'pairs':pairs}
        atomic_json(pool_root/scene.scene_id/'pairs.json',record)
        records[scene.scene_id] = record
    return sorted([dict(p,context=r['context'],pairing_scene_id=r['scene_id'],pairing_manifest_sha256=r['manifest_sha256'])
        for r in records.values() for p in r['pairs']],
        key=lambda p:(p['context'],p['timestamp'],p['source_camera_id'],p['source_camera_object_id'],p['object_id']))


def apply_image_pairs(targets, pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair['context'],pair['object_id'])].append(pair)
    for target in targets:
        if target['status'] != 'selected':
            continue
        choices = grouped[(target['context'],target['object_id'])]
        if not choices:
            target.update(status='pending_pairing',eligibility_stage='unconfirmed_image_candidate')
            continue
        chosen = min(choices,key=visual_priority)
        target.update(selection=chosen,timestamp=chosen['timestamp'],source_view=chosen['source_view'],
            source_camera_id=chosen['source_camera_id'],source_camera_object_id=chosen['source_camera_object_id'],
            eligibility_stage='native_image_pair_confirmed')
    return targets
