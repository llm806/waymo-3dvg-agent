"""Read-only component joins for one sample; project only keys, never sensor payloads."""

from datetime import datetime, timezone
from functools import lru_cache

import pyarrow.parquet as pq

from app.datasets.data import COMPONENTS
from app.datasets.evidence import ROOT
from app.datasets.waymo import T, O, V, WaymoReader


@lru_cache(maxsize=26)
def source_snapshot(component, context, size, mtime_ns):
    # Rehash when a file changes; the stat arguments are intentional cache keys.
    return WaymoReader(ROOT / 'waymo_trainset').provenance(component, context)


def sample_links(packet, scene_sources):
    reader = WaymoReader(ROOT / 'waymo_trainset')
    result = []
    for name in COMPONENTS:
        item = {'component': name, 'status': 'missing', 'file': None, 'key_columns': [],
                'scope': None, 'rows_in_scope': None, 'target_rows': None,
                'camera_counts': {}, 'target_camera_counts': {}, 'laser_counts': {}, 'source_check': 'not_recorded'}
        try:
            path = reader.path(name, packet.context)
            stat = path.stat()
            actual = source_snapshot(name, packet.context, stat.st_size, stat.st_mtime_ns)
            keys = [k for k in pq.ParquetFile(path).schema_arrow.names if k.startswith('key.')]
            item.update(file=actual['file'], sha256=actual['sha256'], key_columns=keys,
                        scope='frame' if T in keys else 'context')
            expected = [s for s in [packet.provenance.get('components', {}).get(name), scene_sources.get(name)] if s]
            if expected and any((s['file'], s['sha256']) != (actual['file'], actual['sha256']) for s in expected):
                item.update(status='source_mismatch', source_check='mismatch')
            else:
                item['source_check'] = 'matches_recorded_sources' if expected else 'current_source_only'
                rows = reader.rows(name, packet.context, columns=keys,
                                   timestamp=packet.frame_timestamp_micros if T in keys else None)
                tuples = [tuple(row[k] for k in keys) for row in rows]
                invalid = any(None in key for key in tuples) or len(tuples) != len(set(tuples))
                item.update(status='invalid_keys' if invalid else 'linked' if rows else 'empty', rows_in_scope=len(rows))
                if O in keys:
                    item['target_rows'] = sum(row[O] == packet.target_object_id for row in rows)
                    if V in keys:
                        for row in rows:
                            if row[O] == packet.target_object_id:
                                sensor = str(row[V])
                                item['target_camera_counts'][sensor] = item['target_camera_counts'].get(sensor, 0) + 1
                for column, field in [(V, 'camera_counts'), ('key.laser_name', 'laser_counts')]:
                    if column in keys:
                        for row in rows:
                            sensor = str(row[column])
                            item[field][sensor] = item[field].get(sensor, 0) + 1
        except FileNotFoundError:
            item.update(status='missing')
        except (OSError, ValueError) as exc:
            item.update(status='unreadable', error=type(exc).__name__)
        result.append(item)
    return {'packet_id': packet.packet_id, 'context': packet.context,
            'frame_timestamp_micros': packet.frame_timestamp_micros, 'target_object_id': packet.target_object_id,
            'checked_at': datetime.now(timezone.utc).isoformat(), 'components': result,
            'interpretation': 'Key joins and source consistency only; not independent alignment verification. Empty optional associations are not guessed.'}
