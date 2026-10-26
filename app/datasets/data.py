"""Read-only Parquet inventory; sensor payloads are not read by this scan."""

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from app.datasets.waymo import source_roots

COMPONENTS = (
    "camera_box", "camera_calibration", "camera_image",
    "camera_to_lidar_box_association", "lidar", "lidar_box",
    "lidar_calibration", "lidar_camera_projection", "lidar_camera_synced_box",
    "lidar_pose", "projected_lidar_box", "stats", "vehicle_pose",
)


class ParquetFileInfo(BaseModel):
    path: str
    component: str
    context: str
    official_split: str = 'training'
    rows: int
    bytes: int
    row_groups: int
    frame_count: int | None
    key_columns: list[str]
    duplicate_keys: int
    null_key_rows: int
    schema_sha256: str
    content_sha256: str | None = None


class DatasetInventory(BaseModel):
    schema_version: str = "0.1"
    declared_dataset_version: str = "waymo-perception-v2.0.1"
    version_source: str = "user-declared; not independently verified"
    files: list[ParquetFileInfo] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    component_count: int = 0
    file_count: int = 0
    total_bytes: int = 0
    errors: list[str] = Field(default_factory=list)


def inspect_dataset(root: Path, *, hash_files: bool = False) -> DatasetInventory:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = DatasetInventory()
    component_contexts: dict[str, set[str]] = {}
    context_splits = {}
    for component in COMPONENTS:
        paths = sorted((path, expected, base) for expected, base in source_roots(root)
                       for path in (base / component).glob('*.parquet'))
        if not paths:
            result.errors.append(f"missing_component:{component}")
            continue
        component_contexts[component] = set()
        schemas = set()
        for path, expected, base in paths:
            try:
                parquet = pq.ParquetFile(path)
                schema = parquet.schema_arrow.remove_metadata()
                schema_hash = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()
                schemas.add(schema_hash)
                key_columns = [name for name in schema.names if name.startswith("key.")]
                if "key.segment_context_name" not in key_columns:
                    result.errors.append(f"missing_context_key:{component}/{path.name}")
                keys = parquet.read(columns=key_columns).to_pydict()
                rows = list(zip(*(keys[name] for name in key_columns))) if key_columns else []
                contexts = set(keys.get("key.segment_context_name", [])) - {None}
                # Empty components retain their context from the local filename.
                split = next((s for s in ('training', 'validation') if path.stem.startswith(f'{s}_{component}_')), None)
                if split is None or expected is not None and split != expected:
                    raise ValueError('Unknown official split in filename')
                filename_context = path.stem.removeprefix(f'{split}_{component}_')
                context = next(iter(contexts)) if len(contexts) == 1 else filename_context
                if context != filename_context or not filename_context:
                    result.errors.append(f'filename_context_mismatch:{component}/{path.name}')
                if context in context_splits and context_splits[context] != split:
                    result.errors.append(f'official_split_conflict:{context}')
                context_splits[context] = split
                if len(contexts) > 1:
                    result.errors.append(f"multiple_contexts:{component}/{path.name}")
                component_contexts[component].add(context)
                timestamps = keys.get("key.frame_timestamp_micros")
                duplicate_keys = len(rows) - len(set(rows))
                null_key_rows = sum(any(value is None for value in row) for row in rows)
                if duplicate_keys or null_key_rows:
                    result.errors.append(f"invalid_keys:{component}/{path.name}")
                digest = None
                if hash_files:
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                result.files.append(ParquetFileInfo(
                    path=(path.relative_to(root).as_posix() if base == root else path.relative_to(root.parent).as_posix()), component=component,
                    context=context, official_split=split, rows=parquet.metadata.num_rows, bytes=path.stat().st_size,
                    row_groups=parquet.metadata.num_row_groups,
                    frame_count=len(set(timestamps)) if timestamps is not None else None,
                    key_columns=key_columns, duplicate_keys=duplicate_keys,
                    null_key_rows=null_key_rows, schema_sha256=schema_hash,
                    content_sha256=digest,
                ))
            except (OSError, ValueError) as exc:
                result.errors.append(f"unreadable:{component}/{path.name}:{type(exc).__name__}")
        if len(schemas) > 1:
            result.errors.append(f"schema_mismatch:{component}")
    contexts = set().union(*component_contexts.values()) if component_contexts else set()
    for component, present in component_contexts.items():
        if present != contexts:
            result.errors.append(f"context_coverage_mismatch:{component}")
    result.contexts = sorted(contexts)
    result.component_count = len(component_contexts)
    result.file_count = len(result.files)
    result.total_bytes = sum(file.bytes for file in result.files)
    return result
