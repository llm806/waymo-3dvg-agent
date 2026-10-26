"""Conversion progress travels over stdout; disk snapshots are non-authoritative."""
import json
import sys

from app.infrastructure.artifacts import atomic_json

PROTOCOL = 'waymo-conversion-progress-v1'


def emit_progress(record, snapshot=None):
    # Completion must still reach the parent when a Windows reader/scanner has
    # locked the optional display file. Dataset/scene writes remain strict.
    print(json.dumps({'protocol': PROTOCOL, **record}, ensure_ascii=False), flush=True)
    if snapshot is not None:
        try:
            atomic_json(snapshot, record)
        except OSError as exc:
            print(json.dumps({'event': 'progress_snapshot_unavailable',
                              'error_type': type(exc).__name__}), file=sys.stderr, flush=True)
