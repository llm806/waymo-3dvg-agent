"""Deterministic target selection and traceable multiview evidence artifacts."""

import hashlib
import io
import json
import math
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.infrastructure.artifacts import atomic_json, file_sha256, hash_assets
from app.schemas.contracts import EvidencePacket, ImageEvidence, ObjectFact, SceneEvidence
from app.quality.contrast import POLICY, contrast_plan, predicate_catalog
from app.datasets.training import points_inside_box, read_scene
from app.datasets.visibility import sign_front_facing, VIEW_SCREENING_POLICY, LEGACY_VIEW_SCREENING_POLICY
from app.quality.visibility_policy import RECOGNIZABLE_POLICY, allows_partial, checks_rgb_visibility, source_camera_inventory
from app.schemas.training_contracts import NativeBox, TaskProfile
from app.datasets.appearance_candidates import IMAGE_INVENTORY_POLICY
from app.datasets.waymo import B, C, CAMERAS, O, P, T, V, WaymoReader, clip_projection, object_fact, track_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "runtime" / "evidence"
TASK = {"task_id": "waymo-single-frame-object-v1", "language": "en",
        "target_box": "native lidar_box 7-DOF", "coordinate_frame": "frame vehicle: x forward, y left, z up",
        "annotation_inputs": ["native_boxes", "RGB", "projected_boxes", "calibration", "track"],
        "training_input_capabilities": ["geometry", "rgb"],
        "execution_policy": "evidence-fast-v1",
        "observation_policy": "identity-first-on-demand-v1",
        "source_light_policy": "visible-region-v1",
        "description_policy": "evidence-guided-ego-or-explicit-camera-v3",
        "reference_policy": "minimal-sufficient-grounding-v1",
        "scene_grounding_policy": "two-round-blind-scene-v1",
        "reference_preference_policy": "traffic-scene-v1",
        "landmark_description_policy": "spatially-qualified-v1",
        "vehicle_heading_policy": "vehicle-heading-landmark-v1",
        "claim_normalization_policy": "compound-claim-entailment-v1",
        "joint_reference_policy": "joint-target-landmark-v1",
        "vehicle_forward_region_policy": "vehicle-forward-region-v1",
        "spatial_relation_policy": "native-3d-spatial-relations-v1",
        "image_position_policy": "verified-boundary-v1",
        "requires_native_camera_link": True,
        "natural_description": True,
        "contrastive_appearance": True,
        "image_first": True,
        "image_identity_policy": "official-projection-v1",
        "image_inventory_policy": IMAGE_INVENTORY_POLICY,
        "visibility_policy": RECOGNIZABLE_POLICY,
        "view_selection_policy": "recognizable-multiview-v2",
        "admission_policy": "visible-distinct-v1",
        "annotation_search": "evidence-guided-v1",
        "image_reference_frame": "explicit camera name required for every image relation",
        "prohibited_essential_claims": ["unqualified_image_left_right", "temporal_motion", "object_id", "exact_coordinates", "numeric_distances", "distance_ranks"],
        "ground_protocol": "oracle native-box candidates; not point-cloud free-box grounding",
        "box_order": TaskProfile().box_order, "heading_convention": TaskProfile().heading_convention}


def audit_context(reader: WaymoReader, context: str) -> dict:
    boxes = reader.metadata_rows("lidar_box", context)
    camera_boxes = reader.metadata_rows("camera_box", context)
    associations = reader.metadata_rows("camera_to_lidar_box_association", context)
    lidar_keys = {(r[T], r[O]) for r in boxes}
    camera_keys = {(r[T], r[V], r["key.camera_object_id"]) for r in camera_boxes}
    images = reader.rows("camera_image", context, columns=[C, T, V])
    image_keys = {(r[T], r[V]) for r in images}
    projections = reader.metadata_rows("projected_lidar_box", context)
    category_by_key = {(r[T], r[O]): r[f"{B}.type"] for r in boxes}
    stats = {
        "context": context, "native_boxes": len(boxes),
        "zero_point_boxes": sum(r[f"{B}.num_lidar_points_in_box"] == 0 for r in boxes),
        "association_rows": len(associations),
        "association_missing_native_box": sum((r[T], r[O]) not in lidar_keys for r in associations),
        "association_missing_camera_box": sum((r[T], r[V], r["key.camera_object_id"]) not in camera_keys for r in associations),
        "projection_missing_native_box": sum((r[T], r[O]) not in lidar_keys for r in projections),
        "projection_missing_image": sum((r[T], r[V]) not in image_keys for r in projections),
        "projection_category_mismatch": sum((r[T], r[O]) in category_by_key and r[f"{P}.type"] != category_by_key[(r[T], r[O])] for r in projections),
        "interpretation": "association is partial, not an authoritative complete target map; zero-point boxes remain competing candidates",
    }
    return stats


def render_bev(objects: list[ObjectFact], target_id: str, path: Path, points_xyz=None) -> None:
    image = Image.new("RGB", (900, 900), "#f5f7f5")
    draw = ImageDraw.Draw(image)
    extent = max(30.0, max(abs(v) for o in objects for v in o.center_m[:2]) + 10)
    scale = 380 / extent
    if points_xyz is not None:
        # Rasterization is display-only; the training asset keeps every source point.
        columns = np.rint(450 - points_xyz[:, 1] * scale).astype(int)
        rows = np.rint(450 - points_xyz[:, 0] * scale).astype(int)
        valid = (rows >= 50) & (rows < 850) & (columns >= 50) & (columns < 850)
        pixels = np.asarray(image).copy()
        pixels[rows[valid], columns[valid]] = [147, 156, 150]
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)

    def pixel(x, y):
        return (450 - y * scale, 450 - x * scale)

    draw.line([(450, 50), (450, 850)], fill="#bac7be")
    draw.line([(50, 450), (850, 450)], fill="#bac7be")
    draw.text((460, 450), "EGO", fill="#22372b")
    label = "Official SDK point cloud + native boxes" if points_xyz is not None else "Native boxes only (not point cloud)"
    draw.text((20, 20), f"{label}. Forward up; left left. Extent +/-{extent:.0f}m", fill="#22372b")
    for o in objects:
        x, y, _ = o.center_m
        length, width, _ = o.size_m
        corners = []
        for dx, dy in [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]:
            corners.append(pixel(x + dx * math.cos(o.heading_rad) - dy * math.sin(o.heading_rad),
                                 y + dx * math.sin(o.heading_rad) + dy * math.cos(o.heading_rad)))
        color = "#d7352a" if o.object_id == target_id else "#537864"
        draw.line(corners + corners[:1], fill=color, width=3 if o.object_id == target_id else 1)
        if o.object_id == target_id:
            draw.text(pixel(x, y), "TARGET", fill=color)
    image.save(path)


def projected_appearance_support(arrays, target, camera, crop):
    """Require visible pixels of the actual 3D target, not just a projected box."""
    if arrays is None:
        return 0
    mask = points_inside_box(arrays["points"][:, :3], NativeBox(
        center_m=target.center_m, size_m=target.size_m, heading_rad=target.heading_rad))
    slots = arrays["camera_projection"][mask].reshape(-1, 3)
    slots = slots[(slots[:, 0] == camera) & (slots[:, 1] >= crop[0]) & (slots[:, 1] < crop[2])
                  & (slots[:, 2] >= crop[1]) & (slots[:, 2] < crop[3])]
    pixels = np.unique(slots[:, 1:], axis=0)
    if len(pixels) < 20 or min(crop[2] - crop[0], crop[3] - crop[1]) < 24:
        return 0
    # Reject a handful of projected pixels on an otherwise unrelated large crop.
    span = np.ptp(pixels, axis=0)
    if min(span) < 8 or float(np.prod(span)) < .02 * (crop[2] - crop[0]) * (crop[3] - crop[1]):
        return 0
    return len(pixels)


def build_packet(reader: WaymoReader, context: str, timestamp: int, object_id: str,
                 output_root: Path = EVIDENCE_ROOT, prepared_scene=None,
                 source_camera: int | None = None, source_camera_object_id: str | None = None,
                 resolve_provisional_source: bool = False) -> EvidencePacket:
    components = ["lidar_box", "projected_lidar_box", "lidar_camera_synced_box", "camera_image",
                  "camera_calibration", "vehicle_pose", "camera_box", "camera_to_lidar_box_association"]
    sources = {component: reader.provenance(component, context) for component in components}
    scene_manifest, scene_arrays, scene_path = None, None, None
    scene_index = ROOT / "runtime" / "scenes" / "index.json"
    if scene_index.exists():
        available = json.loads(scene_index.read_text(encoding="utf-8"))["scenes"]
        matches = [s for s in available if s["context"] == context and s["timestamp"] == timestamp]
        if len(matches) > 1:
            raise ValueError("Duplicate prepared scene identity")
        if matches:
            if not re.fullmatch(r"[0-9a-f]{24}", matches[0]["scene_id"]):
                raise ValueError("Invalid scene index identity")
            scene_path = ROOT / "runtime" / "scenes" / matches[0]["scene_id"] / "scene.json"
            scene_manifest, scene_arrays = prepared_scene if prepared_scene is not None else read_scene(scene_path.parent)
            if scene_manifest.scene_id != matches[0]["scene_id"]:
                raise ValueError("Prepared scene cache identity mismatch")
            if (scene_manifest.context, scene_manifest.frame_timestamp_micros) != (context, timestamp):
                raise ValueError("Prepared scene key mismatch")
    tool_hash = hashlib.sha256(b"".join((ROOT / "app" / name).read_bytes()
                                      for name in ["schemas/contracts.py", "datasets/waymo.py", "datasets/correspondence.py", "datasets/visibility.py", "datasets/view_selection.py", "datasets/appearance_candidates.py", "quality/visibility_policy.py", "quality/contrast.py", "datasets/evidence.py", "infrastructure/artifacts.py", "datasets/training.py", "schemas/training_contracts.py", "quality/geometry.py"])).hexdigest()
    identity = {"context": context, "timestamp": timestamp, "target": object_id,
                "source_camera": source_camera, "source_camera_object_id": source_camera_object_id,
                "resolve_provisional_source": resolve_provisional_source,
                "sources": sources, "tool_sha256": tool_hash, "policy": POLICY, "task": TASK,
                "scene_manifest_sha256": file_sha256(scene_path) if scene_path else None,
                "dependency_versions": {name: version(name) for name in ["pyarrow", "numpy", "pillow", "pydantic"]},
                "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()}
    packet_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    folder = output_root / packet_id
    if (folder / "packet.json").exists():
        return EvidencePacket.model_validate_json((folder / "packet.json").read_text(encoding="utf-8"))
    # Keep every native candidate, including zero-point/unknown objects.
    frame_rows = [r for r in reader.metadata_rows("lidar_box", context) if r[T] == timestamp]
    objects = sorted([object_fact(r) for r in frame_rows], key=lambda o: o.object_id)
    if len({o.object_id for o in objects}) != len(objects):
        raise ValueError("Duplicate native object key")
    target = next((o for o in objects if o.object_id == object_id), None)
    if target is None:
        raise ValueError("Target missing from native frame boxes")
    folder.mkdir(parents=True, exist_ok=True)
    # Identify partially written assets too, so failed preparation can be cleaned
    # up by task identity without guessing ownership from filenames.
    atomic_json(folder / 'preparation.json', {'packet_id': packet_id, 'context': context,
                'frame_timestamp_micros': timestamp, 'target_object_id': object_id,
                'scene': {'scene_id': scene_manifest.scene_id} if scene_manifest else None})
    catalog = predicate_catalog(objects)
    contrast = contrast_plan(objects, object_id, catalog)
    track = track_evidence(reader, context, object_id, timestamp)
    warnings = []
    if not track["category_consistent"]:
        warnings.append("track_category_inconsistent")
    if not target.lidar_points or target.lidar_points <= 0:
        warnings.append("target_has_no_lidar_points")
    calibrations = {r[V]: r for r in reader.metadata_rows("camera_calibration", context)}
    synced = [r for r in reader.metadata_rows("lidar_camera_synced_box", context) if r[T] == timestamp and r[O] == object_id]
    most_visible = synced[0]["[LiDARCameraSyncedBoxComponent].most_visible_camera_name"] if synced else None
    if not synced:
        warnings.append("missing_camera_synced_box")
    projections = [r for r in reader.metadata_rows("projected_lidar_box", context) if r[T] == timestamp and r[O] == object_id]
    projections.sort(key=lambda r: (r[V] != most_visible, -r[f"{P}.box.size.x"] * r[f"{P}.box.size.y"], r[V]))
    views = []
    camera_images = {}
    # Native 2D links remain auxiliary in the official same-ID projection mode.
    # A projected region can include an occluder; visibility is observed separately.
    from app.datasets.correspondence import associate_camera_boxes, point_supported_native_boxes
    native_boxes = associate_camera_boxes(
        [r for r in reader.metadata_rows("camera_box", context) if r[T] == timestamp],
        [r for r in reader.metadata_rows("camera_to_lidar_box_association", context) if r[T] == timestamp],
        object_id, target.category)
    native_boxes = point_supported_native_boxes(native_boxes, objects, scene_arrays, object_id)
    projections.sort(key=lambda r: (
        source_camera is not None and r[V] != source_camera,
        not any(b['selected_target'] for b in native_boxes.get(r[V], [])),
        r[V] != most_visible, -r[f'{P}.box.size.x'] * r[f'{P}.box.size.y'], r[V]))
    for projection in projections:
        if len(views) >= (5 if TASK.get('view_selection_policy') else POLICY["max_views"]):
            break
        camera = projection[V]
        if projection[f"{P}.type"] != next(r[f"{B}.type"] for r in frame_rows if r[O] == object_id):
            warnings.append(f"projection_category_mismatch:{camera}")
            continue
        calibration = calibrations.get(camera)
        if calibration is None:
            warnings.append(f"missing_calibration:{camera}")
            continue
        columns = [C, T, V, "[CameraImageComponent].image", "[CameraImageComponent].pose_timestamp",
                   "[CameraImageComponent].rolling_shutter_params.camera_trigger_time",
                   "[CameraImageComponent].rolling_shutter_params.camera_readout_done_time"]
        images = reader.rows("camera_image", context, timestamp=timestamp, camera=camera, columns=columns)
        if len(images) != 1:
            warnings.append(f"missing_or_duplicate_image:{camera}")
            continue
        row = images[0]
        jpeg = row["[CameraImageComponent].image"]
        with Image.open(io.BytesIO(jpeg)) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        if (width, height) != (calibration["[CameraCalibrationComponent].width"], calibration["[CameraCalibrationComponent].height"]):
            warnings.append(f"image_calibration_size_mismatch:{camera}")
            continue
        camera_images[camera] = image
        crop = clip_projection(projection, width, height)
        if crop is None or (not TASK.get('image_identity_policy') and (crop[2] - crop[0]) * (crop[3] - crop[1]) < POLICY["min_crop_area_px"]):
            warnings.append(f"unusable_projected_crop:{camera}")
            continue
        artifacts = {kind: f"camera-{camera}-{kind}.jpg" for kind in ["scene", "overlay", "crop"]}
        scene = image.copy()
        scene.thumbnail((1600, 1000))
        scene.save(folder / artifacts["scene"], quality=90)
        overlay = image.copy()
        ImageDraw.Draw(overlay).rectangle(crop, outline="#e33524", width=5)
        overlay.thumbnail((1600, 1000))
        overlay.save(folder / artifacts["overlay"], quality=90)
        cropped = image.crop(crop)
        cropped.thumbnail((640, 640))
        cropped.save(folder / artifacts["crop"], quality=90)
        linked = [b for b in native_boxes.get(camera, []) if b['selected_target']]
        native_crop = None
        if len(linked) == 1:
            x1, y1, x2, y2 = linked[0]['xyxy']
            bounded = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                       min(width, math.ceil(x2)), min(height, math.ceil(y2)))
            if min(bounded[2]-bounded[0], bounded[3]-bounded[1]) >= 24:
                native_crop = bounded
        rgb_crop = native_crop or crop
        if TASK.get('visibility_policy') in {'clear-source-rgb-v1', RECOGNIZABLE_POLICY}:
            dx, dy = max(64, rgb_crop[2]-rgb_crop[0]), max(64, rgb_crop[3]-rgb_crop[1])
            region = (max(0,rgb_crop[0]-dx),max(0,rgb_crop[1]-dy),min(width,rgb_crop[2]+dx),min(height,rgb_crop[3]+dy))
            contextual = image.crop(region)
            ImageDraw.Draw(contextual).rectangle((rgb_crop[0]-region[0],rgb_crop[1]-region[1],rgb_crop[2]-region[0]-1,rgb_crop[3]-region[1]-1),outline='#e33524',width=2)
            contextual.thumbnail((768,768))
            context_name = f'camera-{camera}-appearance-context.jpg'
            contextual.save(folder/context_name,quality=95)
            artifacts['appearance_context'] = context_name
        appearance_box, appearance_id, appearance_source, appearance_points = None, None, None, 0
        if native_crop is not None:
            # RGB identity uses the verified 2D unit. The official 3D projection
            # remains intact in projected_box_xyxy, crop and overlay artifacts.
            name = f'camera-{camera}-appearance.jpg'
            appearance = image.crop(native_crop)
            appearance.thumbnail((640, 640))
            appearance.save(folder/name, quality=95)
            artifacts['appearance'] = name
            appearance_box, appearance_id = native_crop, linked[0]['camera_object_id']
            appearance_source = linked[0].get('identity_source', 'official_camera_lidar_association')
            appearance_points = linked[0].get('support_points', 0)
        elif TASK.get('image_identity_policy') == 'official-projection-v1':
            # The official projected label already carries this native 3D ID.
            # Visibility is observed separately; a camera_box ID is not required.
            name = f"camera-{camera}-appearance.jpg"
            appearance = image.crop(crop)
            appearance.thumbnail((640, 640))
            appearance.save(folder / name, quality=95)
            artifacts['appearance'] = name
            appearance_box, appearance_source = crop, 'official_lidar_box_projection'
            if scene_arrays is not None:
                mask = points_inside_box(scene_arrays['points'][:, :3], NativeBox(
                    center_m=target.center_m, size_m=target.size_m, heading_rad=target.heading_rad))
                slots = scene_arrays['camera_projection'][mask].reshape(-1, 3)
                pixels = np.unique(slots[(slots[:, 0] == camera)
                    & (slots[:, 1] >= crop[0]) & (slots[:, 1] < crop[2])
                    & (slots[:, 2] >= crop[1]) & (slots[:, 2] < crop[3]), 1:], axis=0)
                appearance_points = len(pixels)
                if len(pixels):
                    marked = image.crop(crop)
                    marker = ImageDraw.Draw(marked)
                    for px, py in pixels:
                        marker.ellipse((px-crop[0]-1, py-crop[1]-1, px-crop[0]+1, py-crop[1]+1), fill='#4bc5ff')
                    marked.thumbnail((640,640))
                    marked_name = f'camera-{camera}-appearance-points.jpg'
                    marked.save(folder / marked_name, quality=95)
                    artifacts['appearance_points'] = marked_name
        elif not any(object_id in b["laser_object_ids"] for b in native_boxes.get(camera, [])):
            # Some Waymo segments contain no camera/laser ID association at all.
            # Stored official per-point pixels can still link this native 3D box
            # to visible target pixels. This is evidence, not alignment certification.
            appearance_points = projected_appearance_support(scene_arrays, target, camera, crop)
            if appearance_points:
                name = f"camera-{camera}-appearance.jpg"
                appearance = image.crop(crop)
                appearance.thumbnail((640, 640))
                appearance.save(folder / name, quality=95)
                artifacts["appearance"] = name
                appearance_box, appearance_source = crop, "official_target_point_projection"
        cx, cy = (projection[f"{P}.box.center.{a}"] for a in "xy")
        sx, sy = (projection[f"{P}.box.size.{a}"] for a in "xy")
        views.append(ImageEvidence(camera_name=camera, camera_label=CAMERAS[camera],
                                   projected_box_xyxy=(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2), crop_xyxy=crop,
                                   source_jpeg_sha256=hashlib.sha256(jpeg).hexdigest(), width=width, height=height,
                                   pose_timestamp_s=row[columns[4]], camera_trigger_time_s=row[columns[5]],
                                   camera_readout_done_time_s=row[columns[6]], most_visible_camera=camera == most_visible,
                                   artifacts=artifacts, appearance_box_xyxy=appearance_box,
                                   appearance_camera_object_id=appearance_id, appearance_identity_source=appearance_source,
                                   appearance_projected_points=appearance_points,
                                   target_front_facing=sign_front_facing(target, calibration),
                                   projection_touches_boundary=crop[0]<=0 or crop[1]<=0 or crop[2]>=width or crop[3]>=height))
    if not views:
        warnings.append("no_usable_camera_evidence")
    views.sort(key=lambda v: ("appearance" not in v.artifacts,
                              v.appearance_identity_source != "official_camera_lidar_association",
                              not v.most_visible_camera))
    warnings.append("projection_alignment_not_independently_validated")
    scene_evidence = None
    if scene_manifest is not None:
        native_box = NativeBox(center_m=target.center_m, size_m=target.size_m, heading_rad=target.heading_rad)
        target_points = int(points_inside_box(scene_arrays["points"][:, :3], native_box).sum())
        scene_evidence = SceneEvidence(scene_id=scene_manifest.scene_id, manifest_path=scene_path.relative_to(ROOT).as_posix(),
                                       manifest_sha256=file_sha256(scene_path), total_points=scene_manifest.point_count,
                                       target_points=target_points)
        if target_points == 0:
            warnings.append("target_unobserved_in_converted_point_cloud")
    else:
        warnings.append("point_cloud_not_prepared")
        warnings.append("BEV_uses_native_boxes_not_reconstructed_point_cloud")
    assets = {"bev": "bev.png"}
    render_bev(objects, object_id, folder / assets["bev"], scene_arrays["points"][:, :3] if scene_arrays is not None else None)
    packet = EvidencePacket(packet_id=packet_id, context=context, frame_timestamp_micros=timestamp,
                            target_object_id=object_id, task=TASK, policy=POLICY, target=target,
                            candidates=objects, predicates=catalog, contrast=contrast, views=views, track=track,
                            provenance={"components": sources, "tool_sha256": tool_hash, "artifact_sha256": {},
                                        "appearance_candidates": [],
                                        "dependency_versions": identity["dependency_versions"], "uv_lock_sha256": identity["uv_lock_sha256"],
                                        "joins": {"native_target": [C, T, O], "projection": [C, T, V, O],
                                                  "camera_image": [C, T, V], "calibration": [C, V]},
                                        "projection_semantics": "official projected rectangle; no segmentation/occlusion assertion",
                                        "camera_synced_box": synced},
                            warnings=warnings, assets=assets, scene=scene_evidence,
                            ready_for_generation=any((v.appearance_identity_source == 'official_lidar_box_projection' or v.appearance_identity_source in {'official_camera_lidar_association', 'native_box_point_correspondence'} and v.appearance_camera_object_id) and 'appearance' in v.artifacts for v in views) and target.category != "unknown" and bool(target.lidar_points and target.lidar_points > 0)
                            and track["category_consistent"]
                            and scene_evidence is not None and scene_evidence.target_points > 0)
    from app.datasets.view_selection import confirmed_source_views
    view_seeds = {}
    if TASK.get('view_selection_policy'):
        from app.datasets.view_selection import projected_image_seeds
        view_seeds = projected_image_seeds(
            [r for r in reader.metadata_rows('projected_lidar_box', context) if r[T] == timestamp],
            objects, {cam: (c['[CameraCalibrationComponent].width'], c['[CameraCalibrationComponent].height'])
                for cam, c in calibrations.items()}, calibrations, require_complete=not allows_partial(packet),
            require_front_facing=not allows_partial(packet))
        source_seeds = view_seeds.get(object_id, [])
        packet.provenance['source_view_screening'] = source_seeds
        packet.provenance['view_screening_policy'] = VIEW_SCREENING_POLICY if allows_partial(packet) else LEGACY_VIEW_SCREENING_POLICY
        safe_seeds = [s for s in source_seeds if not s['foreground_object_ids']]
        packet.provenance['unobstructed_source_cameras'] = {str(s['source_camera_id']): s for s in safe_seeds}
        if allows_partial(packet):
            packet.provenance['recognizable_source_cameras'] = {str(s['source_camera_id']): s for s in source_seeds}
    candidates = confirmed_source_views(packet.views, source_camera, source_camera_object_id, resolve_provisional_source,
        require_complete=checks_rgb_visibility(packet) and not allows_partial(packet), allow_back_facing=allows_partial(packet))
    if TASK.get('view_selection_policy'):
        all_choices = confirmed_source_views(packet.views, require_complete=not allows_partial(packet),
            allow_back_facing=allows_partial(packet))
        safe = source_camera_inventory(packet) or {}
        candidates = [v for v in all_choices if str(v.camera_name) in safe]
        candidates.sort(key=lambda v: (v.camera_name != source_camera,
            bool(safe[str(v.camera_name)]['foreground_object_ids']), v.projection_touches_boundary is not False,
            -safe[str(v.camera_name)]['image_selection_score']))
    if candidates:
        source = candidates[0]
        packet.source_view = CAMERAS[source.camera_name]
        packet.source_camera_object_id = source.appearance_camera_object_id
        packet.provenance['source_image_selection'] = {'camera_name': source.camera_name,
            'camera_object_id': source.appearance_camera_object_id, 'image_box_xyxy': source.appearance_box_xyxy,
            'identity_source': source.appearance_identity_source}
        if source_camera_object_id and source.appearance_camera_object_id != source_camera_object_id:
            packet.provenance['source_seed_resolution'] = {
                'requested_camera_name': source_camera, 'requested_camera_object_id': source_camera_object_id,
                'confirmed_camera_name': source.camera_name, 'confirmed_camera_object_id': source.appearance_camera_object_id,
                'reason': 'provisional_seed_resolved_by_native_target_identity'}
        packet.views.sort(key=lambda v: (v.camera_name != source.camera_name, 'appearance' not in v.artifacts))
    else:
        packet.ready_for_generation = False
        packet.warnings.append('selected_native_image_box_not_confirmed')
    if not allows_partial(packet) and packet.target.category == 'sign' and not any(v.camera_label == packet.source_view and v.target_front_facing is True for v in packet.views):
        packet.ready_for_generation = False
        packet.warnings.append('target_sign_not_front_facing_in_source_view')
    from app.datasets.appearance_candidates import build_candidate_crops, scene_image_inventory
    packet.provenance['scene_image_candidates'] = scene_image_inventory(native_boxes)
    if TASK.get('image_identity_policy') == 'official-projection-v1':
        from app.datasets.correspondence import official_projection_boxes
        all_projections = [r for r in reader.metadata_rows('projected_lidar_box', context) if r[T] == timestamp]
        projected_boxes = official_projection_boxes(all_projections, objects,
            {cam: (c['[CameraCalibrationComponent].width'], c['[CameraCalibrationComponent].height']) for cam,c in calibrations.items()})
        packet.provenance['native_image_context'] = packet.provenance['scene_image_candidates']
        packet.provenance['scene_image_candidates'] = scene_image_inventory(native_boxes, projected_boxes)
    if candidates:
        packet.provenance['source_image_candidates'] = [c for c in packet.provenance['scene_image_candidates']
            if c['camera_name'] == source.camera_name]
    packet.provenance['appearance_candidates'] = build_candidate_crops(reader, context, timestamp,
        native_boxes, objects, folder, include_unlinked=True,
        preferred_camera=candidates[0].camera_name if candidates else source_camera,
        arrays=scene_arrays if packet.task.get('annotation_search') else None,
        projected_boxes=projected_boxes if TASK.get('image_identity_policy') == 'official-projection-v1' else None,
        view_seeds=view_seeds, camera_images=camera_images)
    packet.provenance['artifact_sha256'] = hash_assets(folder,
        (p.name for p in folder.iterdir() if p.suffix in ('.jpg', '.png')))
    atomic_json(folder / "packet.json", packet.model_dump())
    (folder / 'preparation.json').unlink(missing_ok=True)
    return packet
