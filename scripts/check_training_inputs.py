"""Audit local full-scene model inputs; never publish annotation quality claims."""

import json
from pathlib import Path

from app.infrastructure.artifacts import atomic_json, file_sha256
from app.schemas.contracts import EvidencePacket
from app.datasets.training import points_inside_box, read_scene
from app.schemas.training_contracts import NativeBox
from app.datasets.waymo import WaymoReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / "runtime/scenes/index.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "runtime/evidence/index.json").read_text(encoding="utf-8"))
    reader = WaymoReader(ROOT / "waymo_trainset")
    scenes, errors, targets = [], list(source["errors"]), []
    expected_frames = {(p["context"], p["timestamp"]) for p in evidence["packets"]}
    seen = set()
    for entry in source["scenes"]:
        try:
            # Resolve through the declared relative path; SceneManifest checks its identity.
            path = (ROOT / entry["path"]).resolve()
            if not path.is_relative_to((ROOT / "runtime/scenes").resolve()):
                raise ValueError("Scene index escapes scene directory")
            scene, arrays = read_scene(path.parent)
            key = (scene.context, scene.frame_timestamp_micros)
            if key in seen or key != (entry["context"], entry["timestamp"]) or scene.scene_id != entry["scene_id"]:
                raise ValueError("Scene index identity mismatch/duplicate")
            if scene.official_split != reader.official_split(scene.context):
                raise ValueError('Scene must retain its original official split')
            for component, provenance in scene.source_components.items():
                actual = reader.provenance(component, scene.context)
                if actual["file"] != provenance["file"] or actual["sha256"] != provenance["sha256"]:
                    raise ValueError("Raw component source hash mismatch: " + component)
            seen.add(key)
            scenes.append(dict(scene_id=scene.scene_id, context=scene.context, timestamp=scene.frame_timestamp_micros,
                               points=scene.point_count, sensor_returns=scene.sensor_returns,
                               manifest_sha256=file_sha256(path), sdk_version=scene.conversion["sdk_version"]))
            for item in evidence["packets"]:
                if (item["context"], item["timestamp"]) != key:
                    continue
                packet = EvidencePacket.model_validate_json((ROOT / "runtime/evidence" / item["packet_id"] / "packet.json").read_text(encoding="utf-8"))
                box = NativeBox(center_m=packet.target.center_m, size_m=packet.target.size_m, heading_rad=packet.target.heading_rad)
                count = int(points_inside_box(arrays["points"][:, :3], box).sum())
                if count == 0:
                    raise ValueError("Selected target unobserved in converted point cloud")
                if packet.scene is None or packet.scene.scene_id != scene.scene_id or packet.scene.manifest_sha256 != file_sha256(path):
                    raise ValueError("Evidence does not reference this verified full scene")
                if packet.scene.target_points != count:
                    raise ValueError("Evidence target point count mismatch")
                targets.append(dict(packet_id=packet.packet_id, category=packet.target.category,
                                    native_label_points=packet.target.lidar_points, converted_points_within_box=count))
        except (ValueError, KeyError, OSError) as exc:
            errors.append(dict(entry=entry, reason=str(exc), error_type=type(exc).__name__))
    if seen != expected_frames:
        errors.append(dict(reason="Converted frame coverage disagrees with selected evidence", missing_frames=sorted(expected_frames - seen)))
    report = dict(scope="local source/input integrity and target observability; not label accuracy or independent alignment proof",
                  passed=not errors, scene_count=len(scenes), total_points=sum(s["points"] for s in scenes),
                  target_count=len(targets), scenes=scenes, targets=targets, errors=errors,
                  training_labels_released=0, source_alignment_verified=False,
                  independent_regrounding_accuracy=None, factual_error_rate=None)
    atomic_json(ROOT / "runtime/training-input-check.json", report)
    print(json.dumps({k: report[k] for k in ["passed", "scene_count", "total_points", "target_count", "training_labels_released", "errors"]}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
