"""Build symmetric identity-bound crops for contrastive appearance observation."""
import hashlib
import io
import math

from PIL import Image

IMAGE_INVENTORY_POLICY = 'native-and-projected-v3-native-rgb'


def image_object_id(camera, box):
    ids = box['laser_object_ids']
    if len(ids) == 1 and box['association_status'] not in {'ambiguous', 'category_mismatch'}:
        return ids[0]
    return f"image:{camera}:{box['camera_object_id']}"


def scene_image_inventory(boxes, projected_boxes=None):
    """Keep every native 2D box, even when it cannot produce a usable crop.

    Confirmed links reuse the existing 3D identity across cameras. Unresolved
    boxes keep camera-scoped identities; matching appearance is not an ID link.
    """
    native = [{'object_id': image_object_id(camera, box), 'category': box['category'],
             'camera_name': camera, 'camera_object_id': box['camera_object_id'], 'xyxy': box['xyxy'],
             'identity_source': box.get('identity_source', 'official_camera_lidar_association'),
             **({k:box[k] for k in ('native_unit_object_ids','native_unit_point_support')}
                if box.get('identity_source') == 'native_camera_box_group_point_correspondence' else {})}
            for camera in sorted(boxes) for box in boxes[camera]]
    if projected_boxes is None:
        return native
    # A verified native 2D link supplies the RGB region for the same 3D ID.
    # Keep projections as fallback and unresolved native boxes as possible rivals.
    merged = {(camera, box['object_id']): dict(box, camera_name=camera, camera_object_id=None)
              for camera in sorted(projected_boxes) for box in projected_boxes[camera]}
    for record in native:
        key = (record['camera_name'], record['object_id'])
        if key in merged:
            if merged[key]['category'] == record['category']:
                merged[key] = record
                continue
            record = {**record, 'object_id': f"image:{record['camera_name']}:{record['camera_object_id']}"}
            key = (record['camera_name'], record['object_id'])
        merged[key] = record
    return list(merged.values())


def build_candidate_crops(reader, context, timestamp, boxes, objects, folder, include_unlinked=False, preferred_camera=None,
                          arrays=None, projected_boxes=None, view_seeds=None, camera_images=None):
    by_id = {obj.object_id: obj for obj in objects}
    options = {}
    # Packet construction already decoded these exact same-frame originals.
    # Only cameras absent from that packet-local map need another Parquet read.
    images = {} if camera_images is None else dict(camera_images)
    def camera_image(camera):
        if camera not in images:
            rows = reader.rows('camera_image', context, timestamp=timestamp, camera=camera,
                               columns=['[CameraImageComponent].image'])
            images[camera] = None
            if len(rows) == 1:
                with Image.open(io.BytesIO(rows[0]['[CameraImageComponent].image'])) as opened:
                    images[camera] = opened.convert('RGB')
        return images[camera]
    for box in scene_image_inventory(boxes, projected_boxes):
        oid = box['object_id']
        obj = by_id.get(oid)
        if (obj is None and not include_unlinked) or (obj is not None and obj.category != box['category']):
            continue
        rect = box['xyxy']
        options.setdefault(oid, []).append(((rect[2]-rect[0])*(rect[3]-rect[1]), box['camera_name'], box))
    if arrays is not None and projected_boxes is None:
        # Native 3D signs/objects may have no camera_box label. Use actual
        # object-owned pixel returns to propose a crop, never box-projection IoU.
        import numpy as np
        from app.quality.geometry import inside_oriented_box
        owner = np.full(len(arrays['points']), -1, dtype=np.int32)
        hits = np.zeros(len(owner), dtype=np.int16)
        for index, obj in enumerate(objects):
            mask = inside_oriented_box(arrays['points'][:, :3], obj.center_m, obj.size_m, obj.heading_rad)
            owner[mask] = index
            hits += mask
        owner[hits != 1] = -1
        for index, obj in enumerate(objects):
            slots = arrays['camera_projection'][owner == index].reshape(-1, 3)
            for camera in range(1, 6):
                if any(option[1] == camera for option in options.get(obj.object_id, [])):
                    continue
                pixels = np.unique(slots[slots[:, 0] == camera, 1:], axis=0)
                if len(pixels) < 20:
                    continue
                image = camera_image(camera)
                if image is None:
                    continue
                pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < image.width)
                    & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.height)]
                if len(pixels) < 20 or min(np.ptp(pixels, axis=0)) < 8:
                    continue
                x1, y1 = pixels.min(axis=0) - 4
                x2, y2 = pixels.max(axis=0) + 5
                box = {'xyxy': [int(x1), int(y1), int(x2), int(y2)], 'category': obj.category,
                    'camera_object_id': None, 'identity_source': 'native_box_point_projection'}
                options.setdefault(obj.object_id, []).append(((x2-x1)*(y2-y1), camera, box))
    manifest = []
    # Keep the entire inventory for admission. The runtime observes bounded
    # batches; dropping objects here would leave unavoidable unknown competitors.
    ordered = sorted(options, key=lambda oid: hashlib.sha256(oid.encode()).hexdigest())
    if not include_unlinked:
        ordered = ordered[:32]
    for index, oid in enumerate(ordered):
        entry = None
        for _, camera, box in sorted(options[oid], key=lambda x: (x[1] != preferred_camera, -x[0], x[1])):
            image = camera_image(camera)
            if image is None:
                continue
            x1, y1, x2, y2 = box['xyxy']
            crop = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                    min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
            if min(crop[2]-crop[0], crop[3]-crop[1]) < (1 if projected_boxes is not None else 24):
                continue
            candidate_id = f'v{index:03d}'
            name = f'appearance-{candidate_id}.jpg' if entry is None else f'appearance-{candidate_id}-camera-{camera}.jpg'
            preview = image.crop(crop)
            if projected_boxes is not None and min(preview.size) < 10:
                # Preserve the small original region; provide valid surrounding
                # pixels instead of upscaling or dropping the object.
                from PIL import ImageDraw
                cx, cy = (crop[0]+crop[2])//2, (crop[1]+crop[3])//2
                region = (max(0, cx-max(32,(crop[2]-crop[0])//2)), max(0,cy-max(32,(crop[3]-crop[1])//2)),
                    min(image.width,cx+max(32,(crop[2]-crop[0])//2)+1),min(image.height,cy+max(32,(crop[3]-crop[1])//2)+1))
                preview = image.crop(region)
                ImageDraw.Draw(preview).rectangle((crop[0]-region[0],crop[1]-region[1],crop[2]-region[0]-1,crop[3]-region[1]-1), outline='#4bc5ff',width=1)
            preview.thumbnail((384, 384))
            preview.save(folder/name, quality=95)
            item = {'candidate_id': candidate_id, 'object_id': oid, 'camera_name': camera,
                             'camera_object_id': box['camera_object_id'], 'category': box['category'],
                             'identity_source': box.get('identity_source', 'official_camera_lidar_association') if oid in by_id else 'unlinked_native_box',
                             'artifact': name}
            seed = next((s for s in (view_seeds or {}).get(oid, []) if s['source_camera_id'] == camera), None)
            if seed and seed['foreground_object_ids']:
                item['potential_foreground_object_ids'] = seed['foreground_object_ids']
            if entry is None:
                entry = item
                manifest.append(entry)
                if arrays is not None:
                    # A larger context crop is an optional, bounded relook. Mark
                    # the same original region for every anonymous candidate.
                    dx, dy = max(32,(crop[2]-crop[0])//2), max(32,(crop[3]-crop[1])//2)
                    context_crop = (max(0,crop[0]-dx),max(0,crop[1]-dy),min(image.width,crop[2]+dx),min(image.height,crop[3]+dy))
                    from PIL import ImageDraw
                    contextual = image.crop(context_crop)
                    draw = ImageDraw.Draw(contextual)
                    draw.rectangle((crop[0]-context_crop[0],crop[1]-context_crop[1],
                        crop[2]-context_crop[0]-1,crop[3]-context_crop[1]-1),outline='#4bc5ff',width=2)
                    contextual.thumbnail((768,768))
                    context_name = f'appearance-{candidate_id}-context.jpg'
                    contextual.save(folder/context_name,quality=95)
                    entry['alternatives'] = [{**item, 'artifact': context_name}]
                    entry['visibility_artifact'] = context_name
                    entry['detail_artifact'] = name
                    entry['alternatives'][0]['detail_artifact'] = name
            else:
                entry.setdefault('alternatives', []).append(item)
            if arrays is None:
                break
    return manifest
