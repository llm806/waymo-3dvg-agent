"""Complete, immutable track census and deterministic representative-frame selection.

Retains every track and admits official same-ID image projection candidates.
Actual point visibility and text uniqueness are checked during preparation/run.
No text model participates in data selection.
"""

import hashlib
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from functools import lru_cache

from app.infrastructure.artifacts import atomic_json, file_sha256
from app.quality.contrast import contrast_plan, predicate_catalog
from app.quality.geometry import native_box_proximity
from app.datasets.evidence import ROOT
from app.datasets.view_selection import projected_image_seeds, CAMERA_PRIORITY
from app.datasets.waymo import B, C, CATEGORIES, O, P, T, V, WaymoReader, clip_projection, object_fact

CATALOG_ROOT = ROOT / "runtime" / "catalogs"
_display_lock = threading.Lock()


def recommendation_order(targets):
    """Round robin classes, then contexts/directions; never remove hard targets."""
    from itertools import zip_longest
    if any(t.get('eligibility_stage') == 'native_image_pair_confirmed' for t in targets):
        from app.datasets.image_pairings import visual_priority
        return sorted([t for t in targets if t['status']=='selected' and t.get('eligibility_stage')=='native_image_pair_confirmed'],
            key=lambda t:(*visual_priority(t['selection']),t['target_id']))
    if any(t.get('eligibility_stage') == 'official_projection_candidate' for t in targets):
        return sorted([t for t in targets if t['status'] == 'selected'], key=lambda t: (
            bool(t['selection'].get('foreground_object_ids')),
            t['selection'].get('visible_fraction', 1) < 1,
            t['selection'].get('range_xy_m', float('inf')),
            -t['selection'].get('image_selection_score', 0), t['target_id']))
    groups = defaultdict(lambda: defaultdict(list))
    for target in targets:
        if target['status'] == 'selected':
            tier = int(target.get('selection', {}).get('source_camera_id', 1) not in {1, 2, 3})
            groups[(tier, target['category'])][(target['context'], target.get('selection', {}).get('ego_direction', 'unknown'))].append(target)
    ordered = []
    for tier in (0, 1):
        per_class = []
        for group in sorted(k for k in groups if k[0] == tier):
            strata = []
            for key in sorted(groups[group]):
                strata.append(sorted(groups[group][key], key=lambda t: (
                    CAMERA_PRIORITY.get(t['selection'].get('source_camera_id', 1), 9),
                    -(t['selection'].get('valid_views', 0)), -(t['selection'].get('lidar_points', 0)), t['target_id'])))
            per_class.append([t for row in zip_longest(*strata) for t in row if t is not None])
        ordered.extend(t for row in zip_longest(*per_class) for t in row if t is not None)
    return ordered
SELECTION = {"version": "recognizable-multiview-v14", "unit": "official same-frame 3D ID and image projection",
             "admission_unit": "official_projection_candidate",
             "recommendation_policy": "prefer-low-overlap-recognizable-view-v7",
             "frame_order": ["near_coincident_native_identity", "potential_overlap", "in_frame", "range_xy_m", "projected_image_area_score", "camera_priority", "lidar_points", "timestamp"],
             "class_quota": None, "frames_per_track": 1,
             "policy_scope": "development selection, not a calibrated quality certificate",
             "eligibility": "Official projection identity permits preparation; actual point visibility, image visibility and sentence uniqueness are checked during annotation. No independent camera_box join is required",
             "zero_point_competitors": "retain every native candidate; exclude only as annotation targets"}


def context_targets(reader: WaymoReader, context: str) -> list[dict]:
    rows = reader.metadata_rows("lidar_box", context)
    frames, tracks = defaultdict(list), defaultdict(list)
    for row in rows:
        tracks[row[O]].append(row)
        frames[row[T]].append(row)
    poses = Counter(r[T] for r in reader.metadata_rows("vehicle_pose", context))
    images = Counter((r[T], r[V]) for r in reader.rows("camera_image", context, columns=[C, T, V]))
    calibrations = defaultdict(list)
    for row in reader.metadata_rows("camera_calibration", context):
        calibrations[row[V]].append(row)
    projections = defaultdict(list)
    for row in reader.metadata_rows("projected_lidar_box", context):
        projections[(row[T], row[O])].append(row)
    frame_diagnostics = {}
    expanded_catalogs = {}
    seed_frames = {}

    def expanded(timestamp, facts, predicates):
        if timestamp not in expanded_catalogs:
            from app.schemas.contracts import EvidencePacket
            from app.quality.contrast import POLICY
            from app.quality.relations import relation_catalog, vertical_rank_catalog
            # Both catalogs enumerate the entire frame symmetrically; no target
            # is favoured, and zero-point/unknown competitors remain present.
            sample = facts[0]
            packet = EvidencePacket(packet_id="catalog-preflight", context=context, frame_timestamp_micros=timestamp,
                target_object_id=sample.object_id, task={}, policy=POLICY, target=sample, candidates=facts,
                predicates=predicates, contrast=contrast_plan(facts, sample.object_id, predicates),
                views=[], track={}, provenance={})
            expanded_catalogs[timestamp] = [*predicates, *relation_catalog(packet)[0], *vertical_rank_catalog(packet)[0]]
        return expanded_catalogs[timestamp]

    def scene(timestamp):
        if timestamp not in frame_diagnostics:
            native = frames[timestamp]
            if len({r[O] for r in native}) != len(native):
                frame_diagnostics[timestamp] = (None, None, "duplicate_native_key")
            else:
                try:
                    facts = [object_fact(r) for r in native]
                    frame_diagnostics[timestamp] = (facts, predicate_catalog(facts), None)
                except (ValueError, TypeError, KeyError):
                    # Never remove an invalid competitor to make the target unique.
                    frame_diagnostics[timestamp] = (None, None, "invalid_scene_geometry")
        return frame_diagnostics[timestamp]

    result = []
    for object_id, track_rows in sorted(tracks.items()):
        categories = {r[f"{B}.type"] for r in track_rows}
        reasons, options = Counter(), []
        for row in track_rows:
            timestamp = row[T]
            reason = "track_category_inconsistent" if len(categories) != 1 else None
            facts, catalog, geometry_error = scene(timestamp)
            reason = reason or geometry_error
            try:
                fact = object_fact(row)
            except (ValueError, TypeError, KeyError):
                reasons[reason or "invalid_target_geometry"] += 1
                continue
            reason = reason or ("unknown_category" if fact.category == "unknown" else None)
            reason = reason or ("no_target_lidar_points" if not fact.lidar_points or fact.lidar_points <= 0 else None)
            reason = reason or ("missing_or_duplicate_vehicle_pose" if poses[timestamp] != 1 else None)
            views = set()
            for projection in projections[(timestamp, object_id)]:
                camera = projection[V]
                calibration = calibrations[camera]
                if images[(timestamp, camera)] != 1 or len(calibration) != 1 or projection[f"{P}.type"] != row[f"{B}.type"]:
                    continue
                width = calibration[0].get("[CameraCalibrationComponent].width")
                height = calibration[0].get("[CameraCalibrationComponent].height")
                if width and height:
                    try:
                        clipped = clip_projection(projection, width, height)
                        if clipped and (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) > 0:
                            views.add(camera)
                    except (ValueError, TypeError):
                        pass
            reason = reason or ("no_valid_projected_view" if not views else None)
            if reason:
                reasons[reason] += 1
                continue
            if timestamp not in seed_frames:
                sizes = {cam: (rows[0]['[CameraCalibrationComponent].width'], rows[0]['[CameraCalibrationComponent].height'])
                    for cam, rows in calibrations.items() if len(rows) == 1 and images[(timestamp, cam)] == 1}
                frame_projections = [p for (ts, _), group in projections.items() if ts == timestamp for p in group]
                seed_frames[timestamp] = projected_image_seeds(frame_projections, facts, sizes,
                    {cam: rows[0] for cam,rows in calibrations.items() if len(rows)==1}, require_front_facing=False)
            seeds = [seed for seed in seed_frames[timestamp].get(object_id, []) if seed['source_camera_id'] in views]
            if not seeds:
                reasons['no_projected_image_candidate'] += 1
                continue
            seed = seeds[0]
            plan = contrast_plan(facts, object_id, catalog)
            unary_unique = plan.status == "unique_in_gt_candidates"
            if not unary_unique:
                plan = contrast_plan(facts, object_id, expanded(timestamp, facts, catalog))
            options.append({**seed, "timestamp": timestamp, "lidar_points": fact.lidar_points, "range_xy_m": fact.range_xy_m,
                            "near_coincident_native_ids": sorted(o.object_id for o in facts if native_box_proximity(fact,o,planning=True)),
                            "ego_direction": ('front' if fact.center_m[0] > 1 else 'behind' if fact.center_m[0] < -1 else 'side')
                                + '-' + ('left' if fact.center_m[1] > 1 else 'right' if fact.center_m[1] < -1 else 'center'),
                            "valid_views": len(views), "safe_unary_unique": unary_unique,
                            "safe_spatial_unique": plan.status == 'unique_in_gt_candidates',
                            "remaining_candidates": len(plan.remaining_object_ids)})
        # Rank sources with separable native identities before a slightly
        # nearer/larger but coincident view. No frame or target is excluded;
        # actual RGB recognition and the full uniqueness proof still follow.
        options.sort(key=lambda f: (bool(f['near_coincident_native_ids']), bool(f['foreground_object_ids']), f['visible_fraction'] < 1, f['range_xy_m'],
            -f['image_selection_score'], CAMERA_PRIORITY[f['source_camera_id']],
            -f['valid_views'], -f['lidar_points'], f['timestamp']))
        selected = options[0] if options else None
        # Preserve a small, temporally varied fallback set instead of discarding
        # every other frame before actual RGB visibility is known.
        alternatives = []
        if selected:
            pending = options[1:]
            while pending and len(alternatives) < 2:
                seen_times = [selected['timestamp'], *(o['timestamp'] for o in alternatives)]
                distinct = [o for o in pending if min(abs(o['timestamp'] - t) for t in seen_times) >= 500000]
                choice = (distinct or pending)[0]
                alternatives.append(choice)
                pending = [o for o in pending if o['timestamp'] != choice['timestamp']]
        categories_text = sorted({CATEGORIES.get(r[f"{B}.type"], "unknown") for r in track_rows})
        identity = {"context": context, "object_id": object_id}
        result.append({"target_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24],
                       **identity, "category": categories_text[0] if len(categories_text) == 1 else "unknown",
                       "frame_count": len(track_rows), "start_timestamp": min(r[T] for r in track_rows),
                       "end_timestamp": max(r[T] for r in track_rows), "eligible_frame_count": len(options),
                       "status": "selected" if selected else "excluded", "selection": selected,
                       "alternative_selections": alternatives,
                       "eligibility_stage": "official_projection_candidate" if selected else "excluded",
                       "timestamp": selected["timestamp"] if selected else None,
                       "source_view": selected['source_view'] if selected else None,
                       "source_camera_id": selected['source_camera_id'] if selected else None,
                       "source_camera_object_id": selected['source_camera_object_id'] if selected else None,
                       "frame_exclusion_counts": dict(sorted(reasons.items()))})
    return result


def build_catalog(reader: WaymoReader, output_root: Path = CATALOG_ROOT) -> dict:
    targets, sources = [], {}
    for context in reader.contexts():
        sources[context] = {name: reader.provenance(name, context) for name in
                            ["lidar_box", "projected_lidar_box", "vehicle_pose", "camera_image", "camera_calibration", "camera_box", "camera_to_lidar_box_association"]}
        split = reader.official_split(context)
        targets.extend({**target, 'official_split': split} for target in context_targets(reader, context))
        # Bounded context working set; sensor images were never read.
        reader._cache.clear()
    from app.quality.contrast import POLICY
    identity = {"selection_policy": SELECTION, "geometry_policy": POLICY, "sources": sources,
                "confirmed_image_pairs": [],
                "tools": {name: file_sha256(ROOT / "app" / name) for name in ["datasets/catalog.py", "datasets/image_pairings.py", "datasets/view_selection.py", "datasets/visibility.py", "datasets/correspondence.py", "datasets/waymo.py", "quality/contrast.py", "quality/relations.py", "schemas/contracts.py"]}}
    catalog_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    for index, target in enumerate(targets, 1):
        target["ordinal"] = index
    for rank, target in enumerate(recommendation_order(targets), 1):
        target['recommendation_rank'] = rank
    selected = sum(t["status"] == "selected" for t in targets)
    result = {"schema_version": "1", "catalog_id": catalog_id, **identity,
              "summary": {"tracks": len(targets), "selected": selected, "excluded": sum(t['status']=='excluded' for t in targets),
                          "selected_frames": len({(t["context"], t["timestamp"]) for t in targets if t["status"] == "selected"}),
                          "contexts": len(sources), "pending_pairing":sum(t['status']=='pending_pairing' for t in targets)}, "targets": targets,
              "quality": "selection diagnostics only; independent accuracy not measured"}
    atomic_json(output_root / catalog_id / "catalog.json", result)
    atomic_json(output_root / "index.json", {"catalog_id": catalog_id, "summary": result["summary"]})
    return result


@lru_cache(maxsize=1)
def _display_catalog(path, signature):
    value = json.loads(path.read_text(encoding='utf-8'))
    fields = {'target_id','ordinal','context','object_id','category','status','timestamp',
        'frame_exclusion_counts','recommendation_rank','official_split'}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {**value, 'targets': [{k:v for k,v in t.items() if k in fields} for t in value['targets']]}, digest


def _catalog_path(catalog_id, root):
    if catalog_id is None:
        catalog_id = json.loads((root / "index.json").read_text(encoding="utf-8"))["catalog_id"]
    if len(catalog_id) != 24 or any(c not in "0123456789abcdef" for c in catalog_id):
        raise ValueError("Invalid catalog identity")
    return catalog_id, root / catalog_id / 'catalog.json'


def _display_snapshot(catalog_id, root):
    catalog_id, path = _catalog_path(catalog_id, root)
    # Share only a small display projection and the exact full-catalog digest.
    # Serialize cache misses to avoid concurrent polls decoding the same large
    # file. Check stat on every access, including replacement/in-place edits.
    with _display_lock:
        path = path.resolve()
        def marker():
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
        for _ in range(2):
            signature = marker()
            result, digest = _display_catalog(path, signature)
            if marker() == signature:
                if result['catalog_id'] != catalog_id:
                    raise ValueError('Catalog identity mismatch')
                return result, digest
        raise ValueError('Catalog changed during read')


def catalog_digest(catalog_id: str, root: Path = CATALOG_ROOT) -> str:
    """Exact frozen JSON hash, cached without retaining execution candidates."""
    return _display_snapshot(catalog_id, root)[1]


def load_catalog(catalog_id: str | None = None, root: Path = CATALOG_ROOT, *, compact=False) -> dict:
    if compact:
        # Display readers share one small projection, never the mutable/frozen
        # execution catalog. Every read checks replacement and in-place edits.
        return _display_snapshot(catalog_id, root)[0]
    catalog_id, path = _catalog_path(catalog_id, root)
    result = json.loads(path.read_text(encoding='utf-8'))
    if result["catalog_id"] != catalog_id:
        raise ValueError("Catalog identity mismatch")
    return result
