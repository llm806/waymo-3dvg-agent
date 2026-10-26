"""Validate complete scene inputs and keep all target hints in supervision/metadata."""

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from app.infrastructure.artifacts import asset_path, checked_asset
from app.quality.geometry import inside_oriented_box
from app.schemas.training_contracts import NativeBox, SceneManifest, TaskProfile, TrainingSample


def points_inside_box(xyz: np.ndarray, box: NativeBox, tolerance_m: float = 0.01) -> np.ndarray:
    return inside_oriented_box(xyz, box.center_m, box.size_m, box.heading_rad, tolerance_m)


def read_scene(folder: Path, *, verify_hashes: bool = True,
               camera_ids: set[int] | None = None) -> tuple[SceneManifest, dict[str, np.ndarray]]:
    scene = SceneManifest.model_validate_json((folder / "scene.json").read_text(encoding="utf-8"))
    def resolve(ref):
        return checked_asset(folder, ref.path, ref.sha256) if verify_hashes else asset_path(folder, ref.path)
    points_file = resolve(scene.points)
    resolve(scene.calibration)
    if camera_ids is not None and not camera_ids <= {c.camera_name for c in scene.cameras}:
        raise ValueError('Unknown scene camera')
    for camera in scene.cameras:
        if camera_ids is not None and camera.camera_name not in camera_ids:
            continue
        image_path = resolve(camera.image)
        with Image.open(image_path) as image:
            if image.format != "JPEG" or image.size != (camera.width, camera.height):
                raise ValueError("Camera image encoding/dimensions disagree with calibration")
            image.verify()
    with np.load(points_file, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"points", "laser_id", "return_id", "range_image_pixel", "camera_projection", "nlz"}
    if set(arrays) != required:
        raise ValueError("Point cloud arrays do not match the conversion protocol")
    count = scene.point_count
    shapes = {"points": (count, 6), "laser_id": (count,), "return_id": (count,),
              "range_image_pixel": (count, 2), "camera_projection": (count, 6), "nlz": (count,)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid array shape or non-finite values: " + name)
    dtypes = {"points": np.float32, "laser_id": np.uint8, "return_id": np.uint8,
              "range_image_pixel": np.int32, "camera_projection": np.int32, "nlz": np.int8}
    if any(arrays[name].dtype != dtype for name, dtype in dtypes.items()):
        raise ValueError("Point cloud array dtypes do not match the official conversion protocol")
    if arrays["points"].dtype != np.float32 or np.any(arrays["points"][:, 5] <= 0):
        raise ValueError("Points must be float32 and use the official positive-range mask")
    if not np.isin(arrays["laser_id"], [1, 2, 3, 4, 5]).all() or not np.isin(arrays["return_id"], [1, 2]).all():
        raise ValueError("Unexpected sensor or return ID")
    if np.any(arrays["range_image_pixel"] < 0) or not np.isin(arrays["camera_projection"][:, [0, 3]], range(6)).all():
        raise ValueError("Invalid range pixel or camera projection ID")
    for key, expected in scene.sensor_returns.items():
        laser, return_id = map(int, key.split("/"))
        actual = np.count_nonzero((arrays["laser_id"] == laser) & (arrays["return_id"] == return_id))
        if actual != expected:
            raise ValueError("Sensor/return array counts mismatch")
    return scene, arrays


def training_inputs(folder: Path, task: TaskProfile, expression: str) -> dict:
    scene, arrays = read_scene(folder, verify_hashes=False)
    if scene.coordinate_frame != task.coordinate_frame:
        raise ValueError("Scene/task coordinate mismatch")
    inputs = {"points": arrays["points"], "point_features": scene.point_features,
              "expression": expression, "coordinate_frame": scene.coordinate_frame}
    if "rgb" in task.input_capabilities:
        if not scene.cameras:
            raise ValueError("RGB protocol requires scene images")
        inputs["images"] = {c.camera_name: asset_path(folder, c.image.path) for c in scene.cameras}
        inputs["calibration"] = json.loads(asset_path(folder, scene.calibration.path).read_text(encoding="utf-8"))
        inputs["point_camera_projection"] = arrays["camera_projection"]
    return inputs


def load_training_sample(dataset_root: Path, record: TrainingSample) -> tuple[dict, dict, dict]:
    scene_path = asset_path(dataset_root, record.scene.path)
    scene = SceneManifest.model_validate_json(scene_path.read_text(encoding="utf-8"))
    if (scene.context, scene.frame_timestamp_micros, scene.official_split) != (record.context, record.frame_timestamp_micros, record.official_split):
        raise ValueError("Sample and scene identity/split mismatch")
    calibration_path = asset_path(dataset_root, record.quality.calibration_report.path)
    validation_path = asset_path(dataset_root, record.quality.validation_report.path)
    if record.quality.protocol != 'independent-blind-audited-v1':
        raise ValueError('Legacy certificate cannot bypass independent frozen report validation')
    if record.quality.protocol == 'independent-blind-audited-v1':
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        if not all(r.get('complete') and r.get('calibration_gate_passed') for r in (calibration,validation)):
            raise ValueError('Incomplete independent release reports')
        from app.quality.research import RELEASE_MIN_REVIEWED, RELEASE_MAX_ERROR_UPPER95
        if (calibration.get('method_sha256') != validation.get('method_sha256')
                or calibration.get('method_configuration') != validation.get('method_configuration')
                or calibration.get('policy') != validation.get('policy')
                or any(r.get('policy',{}).get('min_reviewed',0)<RELEASE_MIN_REVIEWED
                    or r.get('policy',{}).get('max_error_upper95',1)>RELEASE_MAX_ERROR_UPPER95 for r in (calibration,validation))):
            raise ValueError('Independent release methods or fixed minimum quality policy differ')
        if set(calibration['contexts']) & set(validation['contexts']) or record.context in validation['contexts']:
            raise ValueError('Evaluation contexts cannot become training samples')
        if record.context not in calibration['contexts']:
            raise ValueError('Training context is outside the calibration study')
        from app.quality.research import validate_release_report, PROTOCOL
        if calibration.get('protocol') != validation.get('protocol'):
            raise ValueError('Independent audit input protocols differ')
        if calibration.get('protocol') == PROTOCOL and 'rgb' not in record.task.input_capabilities:
            raise ValueError('Multimodal audit requires RGB training inputs')
        for report in (calibration,validation):validate_release_report(report)
        accepted = next((r for r in calibration['cases'] if r['run_id'] == record.annotation_run_id),None)
        if (not accepted or not accepted['accepted'] or accepted['target_id'] != record.track_id
                or accepted['packet_id'] != record.evidence_packet_id or accepted['scene_sha256'] != record.scene.sha256
                or accepted['expression_sha256'] != hashlib.sha256(record.expression.encode()).hexdigest()
                or accepted['target_box'] != record.target_box.model_dump(mode='json')):
            raise ValueError('Training label differs from independently audited version')
    inputs = training_inputs(scene_path.parent, record.task, record.expression)
    supervision = {"target_box": np.asarray(record.target_box.vector(), dtype=np.float32)}
    metadata = {"sample_id": record.sample_id, "context": record.context, "timestamp": record.frame_timestamp_micros,
                "track_id": record.track_id, "category": record.category, "official_split": record.official_split}
    return inputs, supervision, metadata


def check_publishable_run(run: dict, packet_id: str) -> None:
    gate = run.get("quality_gate", {})
    if run.get("packet_id") != packet_id or run.get("provider_scope") != "real-model":
        raise ValueError("Run provenance does not identify a real-model annotation of this packet")
    if run.get("status") != "auto_pass" or not run.get("auto_published") or not run.get("diagnostic_pass"):
        raise ValueError("Run is not released for training; experimental holds cannot be exported")
    if not all(gate.get(name) is True for name in ("oracle_checks_passed", "counterfactual_checks_passed", "source_alignment_verified", "independent_risk_calibration")):
        raise ValueError("Independent quality/source gates are incomplete")
