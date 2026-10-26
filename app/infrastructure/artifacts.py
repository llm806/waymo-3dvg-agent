"""Atomic writes and integrity checks shared by offline conversion and the harness."""

import hashlib
import errno
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from uuid import uuid4


def atomic_json(path: Path, value: dict) -> None:
    atomic_json_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def atomic_json_text(path: Path, serialized: str) -> None:
    """Atomically mirror JSON already serialized for the authoritative snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    # Windows scanners/readers can briefly lock the destination. Keep the old
    # complete artifact until replacement succeeds; never fall back to truncation.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError as exc:
            # DrvFS reports a Windows sharing violation as POSIX EACCES/EPERM,
            # without winerror. Apply the same bounded retry on both sides.
            transient = (getattr(exc, 'winerror', None) in {5, 32, 33}
                         or exc.errno in {errno.EACCES, errno.EPERM, errno.EBUSY})
            if not transient or attempt == 5:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(min(0.02 * 2**attempt, 0.2))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def asset_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError("Asset must be a relative POSIX path within its dataset")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset escapes dataset root")
    if not path.is_file():
        raise ValueError("Asset missing: " + relative)
    return path


def checked_asset(root: Path, relative: str, expected_sha256: str) -> Path:
    path = asset_path(root, relative)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Asset content hash mismatch: " + relative)
    return path


def hash_assets(root: Path, names, expected: dict | None = None) -> dict:
    """Hash every requested file with bounded I/O overlap; never cache trust.

    Large evidence packets contain hundreds of small crops. Opening them in
    sequence is expensive on Windows. All hashes and failures still propagate
    before the packet is published or used, including on the parallel path.
    """
    names = list(names)

    def digest(name):
        value = file_sha256(asset_path(root, name))
        if expected is not None and value != expected[name]:
            raise ValueError("Asset content hash mismatch: " + name)
        return name, value

    if len(names) < 8:
        return dict(map(digest, names))
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='evidence-hash') as pool:
        return dict(pool.map(digest, names))
