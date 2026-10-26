"""Persist monotonic execution intervals; never count paused time as work."""
import time
import threading
from uuid import uuid4


class BatchTimer:
    def __init__(self, store, job):
        self.store, self.job = store, job
        self.session = uuid4().hex
        self.started = time.perf_counter()
        self.items = {}
        self.lock = threading.Lock()
        self.record('timing_execution_started', {'protocol': 'batch-timing-v1'})

    def record(self, kind, payload):
        with self.store.runs.connect() as db:
            self.store.event(db, self.job, kind, {'session': self.session, **payload})

    def begin_item(self, ordinal):
        with self.lock:
            self.items[ordinal] = time.perf_counter()
            self.record('timing_item_started', {'ordinal': ordinal})

    def finish_item(self, ordinal=None):
        with self.lock:
            for key in list(self.items) if ordinal is None else [ordinal]:
                started = self.items.pop(key, None)
                if started is not None:
                    self.record('timing_item_finished', {'ordinal': key,
                        'active_seconds': max(0, time.perf_counter() - started)})

    def close(self):
        self.finish_item()
        self.record('timing_execution_finished', {'active_seconds': max(0, time.perf_counter() - self.started)})


def summarize_timing(events, *, status, executor_active, now=None):
    """Closed intervals use monotonic durations. Interrupted ones stay unknown.

    Live estimates use server timestamps only while the executor lease is valid.
    Old tasks retain wall-clock timestamps but have no invented active duration.
    """
    now = time.time() if now is None else now
    sessions, items = {}, {}
    for event in events:
        kind, payload = event['kind'], event['payload']
        session = payload.get('session')
        if kind == 'timing_execution_started':
            sessions[session] = {'start': event['at'], 'seconds': None}
        elif kind == 'timing_execution_finished' and session in sessions:
            sessions[session].update(seconds=payload['active_seconds'], end=event['at'])
        elif kind == 'timing_item_started':
            items.setdefault(payload['ordinal'], []).append({'session': session, 'start': event['at'], 'seconds': None})
        elif kind == 'timing_item_finished':
            intervals = items.get(payload['ordinal'], [])
            if intervals and intervals[-1]['session'] == session:
                intervals[-1]['seconds'] = payload['active_seconds']
    latest_session = next(reversed(sessions), None)

    def duration(intervals):
        seconds, incomplete, running, recorded = 0.0, False, False, False
        for interval in intervals:
            if interval['seconds'] is not None:
                seconds += interval['seconds']
                recorded = True
            elif executor_active and interval.get('session') == latest_session:
                seconds += max(0, now - interval['start'])
                running = True
                recorded = True
            else:
                incomplete = True
        return {'active_seconds': seconds if intervals and not incomplete else None,
                'recorded_active_seconds': seconds if recorded else None,
                'history_incomplete': incomplete,
                'running': running, 'complete': bool(intervals) and not incomplete and not running}

    execution = duration([{**s, 'session': key} for key, s in sessions.items()])
    starts = [e['at'] for e in events if e['kind'] == 'batch_started_or_resumed']
    if len(starts) > len(sessions):
        execution.update(active_seconds=None, complete=False, history_incomplete=True)
    ends = [e['at'] for e in events if e['kind'] in {'batch_completed', 'batch_blocked'}]
    terminal = status in {'completed', 'blocked'}
    started_at = min(starts) if starts else (min(s['start'] for s in sessions.values()) if sessions else None)
    finished_at = max([*ends, *(s['end'] for s in sessions.values() if 'end' in s)]) if terminal and ends else None
    wall = max(0, (finished_at if finished_at is not None else now) - started_at) if started_at is not None else None
    return {**execution, 'protocol': 'batch-timing-v1' if sessions else None,
            'started_at': started_at, 'finished_at': finished_at, 'wall_seconds': wall,
            'wall_running': started_at is not None and not terminal,
            'status': ('running' if execution['running'] else 'complete' if execution['complete']
                       else 'incomplete' if sessions else 'unavailable' if starts else 'not_started'),
            'scope': 'This task only. Active time includes preparation, failed attempts and validation; excludes pauses. Wall time includes pauses.'}, {
                ordinal: duration(intervals) for ordinal, intervals in items.items()}
