"""Offline scene conversion using official Waymo v2 functions; run in preprocessing env."""

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
import traceback
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from app.infrastructure.artifacts import atomic_json, checked_asset, file_sha256
from app.infrastructure.conversion_progress import emit_progress
from app.infrastructure.conversion_environment import check_conversion_environment

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "runtime" / "scenes"
LASER = "key.laser_name"
PROFILE = {"version": "official-waymo-v2-both-returns-v1", "sensors": [1, 2, 3, 4, 5],
           "returns": [1, 2], "range_mask": "range > 0", "nlz_filter": "preserve flag; do not remove",
           "motion_compensation": "TOP per-pixel vehicle-to-world pose, then inverse frame pose; both returns",
           "features": ["x", "y", "z", "intensity", "elongation", "range"],
           "sampling": "none; complete scene; model-specific sampling belongs in the training adapter",
           "frame_accumulation": "none"}


def read_lidar_frame(reader,context,timestamp):
    """Overlap three independent Parquet reads inside the one serial worker.

    Resolve source identities on the owning thread. Each task opens a different
    file and retains the reader's bounded, single-thread Arrow decoding. SDK
    conversion, sensor/return ordering and publication stay serial and unchanged.
    """
    names=('lidar','lidar_pose','lidar_camera_projection')
    paths={name:reader.path(name,context) for name in names}
    with ThreadPoolExecutor(max_workers=3,thread_name_prefix='waymo-frame-read') as pool:
        futures={name:pool.submit(reader._frame_rows,paths[name],context,timestamp,None,None) for name in names}
        return {name:future.result() for name,future in futures.items()}


def convert_scene(reader, context: str, timestamp: int, progress=lambda stage: None) -> dict:
    # Import lazily: neither the API nor the ordinary annotation process loads TensorFlow.
    progress("reusing_sdk" if "waymo_open_dataset" in sys.modules else "loading_sdk")
    import numpy as np
    from app.quality.geometry import BoxPointIndex
    from app.datasets.waymo import B, CATEGORIES, O, V
    from waymo_open_dataset import v2
    from waymo_open_dataset.v2.perception.utils import lidar_utils

    started = time.perf_counter()
    progress("checking_scene_sources")
    source_components = {name: reader.provenance(name, context) for name in
                         ["lidar", "lidar_pose", "lidar_calibration", "lidar_camera_projection", "vehicle_pose",
                          "camera_image", "camera_calibration", "lidar_box"]}
    official_split = reader.official_split(context)
    identity = {"context": context, "timestamp": timestamp, "official_split": official_split, "profile": PROFILE,
                "source_components": source_components,
                "converter_sha256": file_sha256(Path(__file__)),
                "contract_sha256": file_sha256(ROOT / "app" / "schemas/training_contracts.py"),
                "training_tools_sha256": file_sha256(ROOT / "app" / "datasets/training.py"),
                "reader_sha256": file_sha256(ROOT / "app" / "datasets/waymo.py"),
                "geometry_sha256": file_sha256(ROOT / "app" / "quality/geometry.py"),
                "artifacts_sha256": file_sha256(ROOT / "app" / "infrastructure/artifacts.py"),
                "preprocessing_lock_sha256": file_sha256(ROOT / "preprocessing" / "uv.lock")}
    scene_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    folder = SCENE_ROOT / scene_id
    if (folder / "scene.json").exists():
        manifest = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
        for asset in [manifest["points"], manifest["calibration"], *[c["image"] for c in manifest["cameras"]]]:
            checked_asset(folder, asset["path"], asset["sha256"])
        return manifest
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / 'preparation.json', {'scene_id': scene_id, 'context': context, 'frame_timestamp_micros': timestamp})
    progress("reading_lidar")
    frame_inputs=read_lidar_frame(reader,context,timestamp)
    lidar_rows = frame_inputs.pop('lidar')
    calibrations = {r[LASER]: v2.LiDARCalibrationComponent.from_dict(r) for r in reader.metadata_rows("lidar_calibration", context)}
    poses = {r[LASER]: v2.LiDARPoseComponent.from_dict(r) for r in frame_inputs.pop('lidar_pose')}
    projections = {r[LASER]: v2.LiDARCameraProjectionComponent.from_dict(r) for r in frame_inputs.pop('lidar_camera_projection')}
    frame_rows = reader.rows("vehicle_pose", context, timestamp=timestamp)
    if len(frame_rows) != 1 or len(lidar_rows) != 5 or set(r[LASER] for r in lidar_rows) != set(PROFILE["sensors"]):
        raise ValueError("Missing/duplicate frame pose or LiDAR sensors")
    frame_pose = v2.VehiclePoseComponent.from_dict(frame_rows[0])
    if 1 not in poses:
        raise ValueError("TOP pixel pose is mandatory; cannot silently omit motion compensation")
    merged = {k: [] for k in ["points", "laser_id", "return_id", "range_image_pixel", "camera_projection", "nlz"]}
    sensor_returns = {}
    for row in sorted(lidar_rows, key=lambda r: r[LASER]):
        lidar = v2.LiDARComponent.from_dict(row)
        laser = row[LASER]
        if laser not in calibrations or laser not in projections:
            raise ValueError("Missing LiDAR calibration or camera projection")
        pixel_pose = poses[1].range_image_return1 if laser == 1 else None
        for return_id in PROFILE["returns"]:
            progress(f"converting_lidar:{laser}:{return_id}")
            ri = getattr(lidar, f"range_image_return{return_id}")
            cp = getattr(projections[laser], f"range_image_return{return_id}")
            tensor = ri.tensor.numpy()
            if tensor.ndim != 3 or tensor.shape[2] != 4 or cp.tensor.shape[:2] != tensor.shape[:2]:
                raise ValueError("Range/projection shape mismatch")
            mask = tensor[..., 0] > 0
            polar_xyz = lidar_utils.convert_range_image_to_point_cloud(
                ri, calibrations[laser], pixel_pose=pixel_pose,
                frame_pose=frame_pose if laser == 1 else None, keep_polar_features=True).numpy()
            camera_points = lidar_utils.extract_pointwise_camera_projection(ri, cp).numpy()
            # The official function produces [range,intensity,elongation,x,y,z].
            points = polar_xyz[:, [3, 4, 5, 1, 2, 0]].astype(np.float32)
            if len(points) != np.count_nonzero(mask) or not np.isfinite(points).all():
                raise ValueError("Official conversion yielded invalid point count or non-finite features")
            count = len(points)
            sensor_returns[f"{laser}/{return_id}"] = count
            for name, array in {"points": points, "laser_id": np.full(count, laser, np.uint8),
                                "return_id": np.full(count, return_id, np.uint8),
                                "range_image_pixel": np.argwhere(mask).astype(np.int32),
                                "camera_projection": camera_points.astype(np.int32),
                                "nlz": tensor[..., 3][mask].astype(np.int8)}.items():
                merged[name].append(array)
    arrays = {name: np.concatenate(parts) for name, parts in merged.items()}
    progress("saving_points")
    temporary = folder / "points.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(folder / "points.npz")
    progress("reading_images")
    camera_rows = reader.rows("camera_image", context, timestamp=timestamp)
    if len(camera_rows) != 5 or {r[V] for r in camera_rows} != {1, 2, 3, 4, 5}:
        raise ValueError("Complete frame camera inputs required")
    camera_calibrations = reader.metadata_rows("camera_calibration", context)
    dimensions = {r[V]: (r["[CameraCalibrationComponent].width"], r["[CameraCalibrationComponent].height"]) for r in camera_calibrations}
    cameras = []
    image_metadata = []
    for row in sorted(camera_rows, key=lambda r: r[V]):
        name = f"camera-{row[V]}.jpg"
        (folder / name).write_bytes(row["[CameraImageComponent].image"])
        width, height = dimensions[row[V]]
        cameras.append(dict(camera_name=row[V], image=dict(path=name, sha256=file_sha256(folder / name)),
                            width=width, height=height, pose_timestamp_s=row["[CameraImageComponent].pose_timestamp"],
                            trigger_time_s=row["[CameraImageComponent].rolling_shutter_params.camera_trigger_time"],
                            readout_done_time_s=row["[CameraImageComponent].rolling_shutter_params.camera_readout_done_time"]))
        image_metadata.append({k: value for k, value in row.items() if k != "[CameraImageComponent].image"})
    atomic_json(folder / "calibration.json", {"camera_calibration": camera_calibrations, "camera_frame_metadata": image_metadata,
                                             "lidar_calibration": reader.metadata_rows("lidar_calibration", context), "frame_pose": frame_rows[0]})
    box_audit = []
    point_index = BoxPointIndex(arrays['points'][:, :3])
    for row in reader.rows("lidar_box", context, timestamp=timestamp):
        center = [row[f"{B}.box.center.{axis}"] for axis in "xyz"]
        size = [row[f"{B}.box.size.{axis}"] for axis in "xyz"]
        box_audit.append({"object_id": row[O], "category": CATEGORIES.get(row[f"{B}.type"], "unknown"),
                          "native_label_points": row[f"{B}.num_lidar_points_in_box"],
                          "converted_points_within_box": int(point_index.inside(center, size, row[f"{B}.box.heading"]).sum())})
    camera_audit = {}
    for camera in cameras:
        slots = arrays["camera_projection"].reshape(-1, 3)
        selected = slots[slots[:, 0] == camera["camera_name"]]
        in_bounds = ((selected[:, 1] >= 0) & (selected[:, 1] < camera["width"]) &
                     (selected[:, 2] >= 0) & (selected[:, 2] < camera["height"]))
        camera_audit[str(camera["camera_name"])] = {"projected_point_slots": len(selected), "in_bounds": int(in_bounds.sum())}
    official_file = Path(inspect.getfile(lidar_utils))
    conversion = {**identity, "implementation": "official-waymo-sdk", "sdk_version": version("waymo-open-dataset-tf-2-12-0"),
                  "tensorflow_version": version("tensorflow"), "numpy_version": np.__version__, "python": platform.python_version(),
                  "official_lidar_utils_sha256": file_sha256(official_file), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    manifest = dict(schema_version="1.0", scene_id=scene_id, context=context, frame_timestamp_micros=timestamp, official_split=official_split,
                             coordinate_frame="frame vehicle: x forward, y left, z up",
                             point_features=PROFILE["features"],
                             points=dict(path="points.npz", sha256=file_sha256(folder / "points.npz")),
                             point_count=len(arrays["points"]), sensor_returns=sensor_returns,
                             calibration=dict(path="calibration.json", sha256=file_sha256(folder / "calibration.json")),
                             cameras=cameras, conversion=conversion, source_components=source_components,
                             diagnostics={"box_point_counts": box_audit, "stored_camera_projection": camera_audit,
                                          "scope": "official conversion and stored point projections; not an independent pixel/target alignment proof",
                                          "source_alignment_verified": False})
    # Publish the scene manifest last; an interrupted directory cannot be mistaken for a completed scene.
    progress("saving_scene")
    atomic_json(folder / "scene.json", manifest)
    (folder / 'preparation.json').unlink(missing_ok=True)
    return manifest


def serve():
    """One serial local pipe, SDK and metadata stay warm across frame requests."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    readers = None
    for line in sys.stdin:
        request = json.loads(line)
        context, timestamp = request["context"], int(request["timestamp"])
        progress_path = Path(request["progress_file"])
        # Requests come only from the fixed-argv local parent, never an HTTP path.
        progress_path.resolve().relative_to((ROOT / "runtime").resolve())
        request_id, stage = request["request_id"], "loading_converter"

        def progress(value, **extra):
            nonlocal stage
            stage = value
            # A pipe owns worker progress. The parent alone writes its Windows
            # display snapshot; the SDK never replaces a polled DrvFS file.
            emit_progress({"request_id": request_id, "stage": stage, "at": time.time(), **extra})

        def deadline():
            progress(stage, error_type="TimeoutError", done=True, exit_code=124)
            os._exit(124)  # Also bounds a stuck native TensorFlow operation.

        timer = threading.Timer(600, deadline)
        timer.daemon = True
        timer.start()
        try:
            progress("checking_converter_environment")
            check_conversion_environment()
            if readers is None:
                from app.datasets.waymo import ContextReaders
                readers = ContextReaders(ROOT / "waymo_trainset")
            reader = readers.get(context)
            scene = convert_scene(reader, context, timestamp, progress)
            index_path = SCENE_ROOT / "index.json"
            existing = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"scenes": []}
            scenes = {(s["context"], s["timestamp"]): s for s in existing["scenes"]}
            scenes[(context, timestamp)] = {"scene_id": scene["scene_id"], "context": context, "timestamp": timestamp,
                "point_count": scene["point_count"], "path": f"runtime/scenes/{scene['scene_id']}/scene.json",
                "manifest_sha256": file_sha256(SCENE_ROOT / scene["scene_id"] / "scene.json")}
            atomic_json(index_path, {"scope": "complete scene inputs, no text labels published",
                                    "scenes": [scenes[k] for k in sorted(scenes)], "errors": []})
            progress("scene_ready", done=True, exit_code=0)
        except Exception as exc:
            progress(stage, done=True, exit_code=1, error_type=type(exc).__name__,
                     reason=str(exc), traceback=traceback.format_exc())
        finally:
            timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context")
    parser.add_argument("--timestamp", type=int)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        serve()
        return
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if bool(args.context) != (args.timestamp is not None):
        parser.error("--context and --timestamp must be provided together")

    current_stage = "loading_converter"

    def progress(stage, **extra):
        nonlocal current_stage
        current_stage = stage
        emit_progress({"stage": stage, "at": time.time(), **extra}, args.progress_file)

    progress("checking_converter_environment")
    try:
        check_conversion_environment()
    except Exception as exc:
        progress(current_stage, error_type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
        raise SystemExit(1) from exc
    progress("loading_converter")
    from app.datasets.waymo import WaymoReader
    if args.context:
        frames = [(args.context, args.timestamp)]
    else:
        evidence = json.loads((ROOT / "runtime" / "evidence" / "index.json").read_text(encoding="utf-8"))
        frames = sorted({(p["context"], p["timestamp"]) for p in evidence["packets"]})
    if args.limit is not None:
        frames = frames[:args.limit]
    reader = WaymoReader(ROOT / "waymo_trainset")
    old_index = SCENE_ROOT / "index.json"
    existing = json.loads(old_index.read_text(encoding="utf-8")) if old_index.exists() else {"scenes": []}
    scenes = {(s["context"], s["timestamp"]): s for s in existing["scenes"]}
    errors = []
    for context, timestamp in frames:
        try:
            scene = convert_scene(reader, context, timestamp, progress)
            scenes[(context, timestamp)] = {"scene_id": scene["scene_id"], "context": context, "timestamp": timestamp,
                           "point_count": scene["point_count"], "path": f"runtime/scenes/{scene['scene_id']}/scene.json",
                           "manifest_sha256": file_sha256(SCENE_ROOT / scene['scene_id'] / "scene.json")}
            print(scene["scene_id"], scene["point_count"], "points", flush=True)
        except Exception as exc:
            progress(current_stage, error_type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
            errors.append({"context": context, "timestamp": timestamp, "error_type": type(exc).__name__, "reason": str(exc)})
    atomic_json(SCENE_ROOT / "index.json", {"scope": "complete scene inputs, no text labels published",
                                          "scenes": [scenes[key] for key in sorted(scenes)], "errors": errors})
    if errors:
        raise SystemExit(1)
    progress("scene_ready")


if __name__ == "__main__":
    main()
