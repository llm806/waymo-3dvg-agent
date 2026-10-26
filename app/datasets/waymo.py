"""Read real Waymo component columns without importing TensorFlow."""

from __future__ import annotations

import hashlib
import math
import copy
import threading
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from app.infrastructure.artifacts import file_sha256

C = "key.segment_context_name"
T = "key.frame_timestamp_micros"
O = "key.laser_object_id"
V = "key.camera_name"
B = "[LiDARBoxComponent]"
P = "[ProjectedLiDARBoxComponent]"
CATEGORIES = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
CAMERAS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}


def source_roots(root: Path) -> list[tuple[str | None, Path]]:
    root = Path(root)
    if root.name in {'waymo_trainset', 'waymo_valset'}:
        return [('training', root.parent / 'waymo_trainset'), ('validation', root.parent / 'waymo_valset')]
    return [(None, root)]


def raw_source_root(workspace: Path, source: dict) -> Path:
    # Internal provenance paths remain relative to their original component root.
    split = 'validation' if Path(source['file']).name.startswith('validation_') else 'training'
    return workspace / ('waymo_valset' if split == 'validation' else 'waymo_trainset')


class ContextReaders:
    """Keep at most two metadata readers warm when source cameras alternate contexts."""
    def __init__(self, root, *, source_cache=None, source_lock=None):
        self.root = root
        self.readers = {}
        # Fingerprints are small; retain them when bulky metadata readers leave
        # the two-context LRU. Every reuse still checks the file's stat marker.
        self.source_cache = {} if source_cache is None else source_cache
        self.source_lock = source_lock if source_lock is not None else threading.Lock()

    def get(self, context):
        reader = self.readers.pop(context, None)
        if reader is None:
            reader = WaymoReader(self.root, source_cache=self.source_cache, source_lock=self.source_lock)
        self.readers[context] = reader
        if len(self.readers) > 2:
            del self.readers[next(iter(self.readers))]
        return reader


class WaymoReader:
    def __init__(self, root: Path, *, source_cache=None, source_lock=None):
        self.root = root
        self._cache: dict[tuple[str, str], list[dict]] = {}
        self._paths: dict[tuple[str, str], Path] = {}
        self._source_provenance = {} if source_cache is None else source_cache
        self._source_lock = source_lock if source_lock is not None else threading.Lock()
        self._source_stats = {}
        self._image_frame = None
        self._context_index = None

    def contexts(self) -> list[str]:
        return sorted(self.context_splits())

    def context_splits(self) -> dict[str, str]:
        roots = source_roots(self.root)
        markers = tuple((str(root), (root / 'lidar_box').stat().st_mtime_ns if (root / 'lidar_box').is_dir() else None)
                        for _, root in roots)
        if self._context_index is not None and self._context_index[0] == markers:
            return dict(self._context_index[1])
        result = {}
        for expected, root in roots:
            for path in (root / 'lidar_box').glob('*.parquet'):
                split = next((s for s in ('training', 'validation') if path.stem.startswith(s + '_lidar_box_')), None)
                if split is None or expected is not None and split != expected:
                    raise ValueError('Wrong official split or data directory: ' + path.name)
                context = path.stem.removeprefix(split + '_lidar_box_')
                if not context or context in result:
                    raise ValueError('Context occurs in multiple official splits: ' + context)
                result[context] = split
        self._context_index = (markers, result)
        return dict(result)

    def source_root(self, split: str) -> Path:
        return next(root for expected, root in source_roots(self.root) if expected in {None, split})

    def official_split(self, context: str) -> str:
        try:
            return self.context_splits()[context]
        except KeyError as exc:
            raise ValueError('Unknown context') from exc

    def path(self, component: str, context: str) -> Path:
        split = self.official_split(context)
        key = (component, context)
        if key not in self._paths:
            path = self.source_root(split) / component / f"{split}_{component}_{context}.parquet"
            if not path.is_file():
                raise FileNotFoundError(component)
            self._paths[key] = path
        return self._paths[key]

    def rows(self, component: str, context: str, *, timestamp: int | None = None,
             columns: list[str] | None = None, camera: int | None = None) -> list[dict]:
        path = self.path(component, context)
        filters = ds.field(C) == context
        if timestamp is not None:
            filters = filters & (ds.field(T) == timestamp)
        if camera is not None:
            filters = filters & (ds.field(V) == camera)
        if timestamp is not None and component == 'camera_image':
            # One frame contains the five camera rows. Decoding the same
            # Parquet row group separately for every camera repeats expensive
            # image-byte decoding. Retain just one frame per metadata reader.
            stat = path.stat()
            key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, context, timestamp)
            if self._image_frame is None or self._image_frame[0] != key:
                names = set(pq.read_schema(path).names)
                if columns is not None and set(columns) - names:
                    raise ValueError('Unknown frame columns')
                rows = self._frame_rows(path, context, timestamp, None, None)
                self._image_frame = (key, names, rows)
            _, names, rows = self._image_frame
            if columns is not None and set(columns) - names:
                raise ValueError('Unknown frame columns')
            return [copy.deepcopy({name: row[name] for name in columns} if columns is not None else row)
                for row in rows if camera is None or row[V] == camera]
        if timestamp is not None and component in {"lidar", "lidar_pose", "lidar_camera_projection"}:
            return self._frame_rows(path, context, timestamp, columns, camera)
        return ds.dataset(path, format="parquet").to_table(columns=columns, filter=filters).to_pylist()

    @staticmethod
    def _frame_rows(path, context, timestamp, columns, camera):
        # Dataset filtering can decode many frames before applying the predicate.
        # Locate native keys first, then stream only the necessary row groups.
        key_columns = [C, T] + ([V] if camera is not None else [])
        rows = []
        with pq.ParquetFile(path, pre_buffer=False) as parquet:
            requested = key_columns + (columns if columns is not None else [])
            if set(requested) - set(parquet.schema_arrow.names):
                raise ValueError("Unknown frame columns")
            for group in range(parquet.num_row_groups):
                keys = parquet.read_row_group(group, columns=key_columns, use_threads=False).to_pylist()
                selected = {i for i, row in enumerate(keys) if row[C] == context and row[T] == timestamp
                    and (camera is None or row[V] == camera)}
                if not selected:
                    continue
                last = max(selected)
                metadata = parquet.metadata.row_group(group)
                # Bound Arrow decoding by both rows and estimated bytes. Only
                # selected native rows are materialized as Python tensor lists.
                bytes_per_row = max(1, metadata.total_byte_size // max(1, metadata.num_rows))
                batch_rows = max(1, min(8, 8 * 1024 * 1024 // bytes_per_row))
                batches = parquet.iter_batches(batch_size=batch_rows, row_groups=[group], columns=columns, use_threads=False)
                start = 0
                for batch in batches:
                    for index in sorted(selected):
                        if start <= index < start + len(batch):
                            rows.extend(batch.slice(index - start, 1).to_pylist())
                    start += len(batch)
                    if start > last:
                        break
        return rows

    def metadata_rows(self, component: str, context: str) -> list[dict]:
        key = (component, context)
        if key not in self._cache:
            self._cache[key] = self.rows(component, context)
        return self._cache[key]

    def provenance(self, component: str, context: str, columns: list[str] | None = None) -> dict:
        path = self.path(component, context)
        stat = path.stat()
        marker = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._source_stats.get(path) != marker:
            self._cache.pop((component, context), None)
            self._source_stats[path] = marker
        # Only the small immutable fingerprint tuples are shared across reader
        # threads. Coalesce hashing and bounded eviction; image/row caches are
        # private. Each reader still checks the actual file marker above.
        with self._source_lock:
            cached = self._source_provenance.get(path)
            if cached is None or cached[0] != marker:
                digest = file_sha256(path)
                with pq.ParquetFile(path) as parquet:
                    schema = parquet.schema_arrow.remove_metadata()
                cached = (marker, digest, tuple(schema.names),
                    hashlib.sha256(schema.serialize().to_pybytes()).hexdigest())
                self._source_provenance[path] = cached
                if len(self._source_provenance) > 1024:
                    del self._source_provenance[next(iter(self._source_provenance))]
        return {"file": path.relative_to(self.source_root(self.official_split(context))).as_posix(), "sha256": cached[1],
                "columns": list(columns or cached[2]), "schema_sha256": cached[3]}


def object_fact(row: dict) -> ObjectFact:
    from app.schemas.contracts import ObjectFact
    center = tuple(float(row[f"{B}.box.center.{a}"]) for a in "xyz")
    size = tuple(float(row[f"{B}.box.size.{a}"]) for a in "xyz")
    heading = float(row[f"{B}.box.heading"])
    valid = all(math.isfinite(x) for x in (*center, *size, heading)) and all(x > 0 for x in size)
    # Invalid numbers must never silently enter a fact contract or JSON trace.
    if not valid:
        raise ValueError(f"Invalid native geometry:{row[O]}")
    speed = tuple(row.get(f"{B}.speed.{a}") for a in "xyz")
    speed = tuple(float(x) if x is not None and math.isfinite(x) else None for x in speed)
    return ObjectFact(object_id=row[O], category=CATEGORIES.get(row[f"{B}.type"], "unknown"),
                      center_m=center, size_m=size, heading_rad=heading,
                      range_xy_m=math.hypot(*center[:2]),
                      lidar_points=row.get(f"{B}.num_lidar_points_in_box"),
                      speed_label_mps=speed, geometry_valid=valid)


def clip_projection(row: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    cx, cy = (row[f"{P}.box.center.{a}"] for a in "xy")
    sx, sy = (row[f"{P}.box.size.{a}"] for a in "xy")
    if not all(x is not None and math.isfinite(x) for x in (cx, cy, sx, sy)) or sx <= 0 or sy <= 0:
        return None
    box = (max(0, math.floor(cx - sx / 2)), max(0, math.floor(cy - sy / 2)),
           min(width, math.ceil(cx + sx / 2)), min(height, math.ceil(cy + sy / 2)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def track_evidence(reader: WaymoReader, context: str, object_id: str, timestamp: int) -> dict:
    rows = sorted((r for r in reader.metadata_rows("lidar_box", context) if r[O] == object_id), key=lambda r: r[T])
    poses = {r[T]: r["[VehiclePoseComponent].world_from_vehicle.transform"]
             for r in reader.metadata_rows("vehicle_pose", context)}
    nearest = sorted(rows, key=lambda r: (abs(r[T] - timestamp), r[T]))[:5]
    samples = []
    for row in sorted(nearest, key=lambda r: r[T]):
        transform = poses.get(row[T])
        if transform is None:
            continue
        matrix = np.asarray(transform, dtype=float).reshape(4, 4)
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid world_from_vehicle pose")
        point = np.array([row[f"{B}.box.center.{a}"] for a in "xyz"] + [1.0])
        samples.append({"timestamp_micros": row[T], "center_world_m": (matrix @ point)[:3].tolist()})
    speeds = []
    for a, b in zip(samples, samples[1:]):
        dt = (b["timestamp_micros"] - a["timestamp_micros"]) / 1e6
        if 0 < dt <= 0.3:
            speeds.append(float(np.linalg.norm(np.subtract(b["center_world_m"], a["center_world_m"])) / dt))
    return {"track_id": object_id, "frame_count": len(rows),
            "start_timestamp_micros": rows[0][T], "end_timestamp_micros": rows[-1][T],
            "category_consistent": len({r[f"{B}.type"] for r in rows}) == 1,
            "nearby_world_centers": samples,
            "median_world_displacement_speed_mps": float(np.median(speeds)) if speeds else None,
            "purpose": "audit/selection only; temporal motion cannot be a single-frame essential claim",
            "derivation": "world_from_vehicle @ native box center; differences over at most 0.3s; label jitter not calibrated"}
