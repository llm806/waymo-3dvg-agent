"""Local, bounded JSONL diagnostics. Never log model prompts or credentials."""

import json
import logging
import os
import re
import threading
import traceback
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4


_lock = threading.Lock()


class PerformanceLog:
    """Per-worker timing only; never part of evidence, quotas or acceptance.

    Phase intervals are sequential; named measurements are inclusive and may
    overlap them. Thread CPU excludes converter/Arrow/other worker threads.
    Append sessions so recovery cannot overwrite earlier attempts. An unfinished
    session remains explicitly incomplete, rather than acquiring invented time.
    """
    def __init__(self, folder, **identity):
        self.path = Path(folder) / 'performance.jsonl'
        self.identity = identity
        self.session = uuid4().hex
        self.started = time.perf_counter()
        self.cpu_started = time.thread_time()
        self.current = None
        self.metrics = {}
        self.closed = self.unavailable = False
        self._write('session_started')

    def _write(self, event, **fields):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {'protocol': 'annotation-performance-v1', 'event': event,
                'at': datetime.now(timezone.utc).isoformat(), 'session': self.session,
                **self.identity, **fields}
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        except OSError:
            # Telemetry must not turn a saved response into an unknown outcome.
            # The summary command reports missing/incomplete logs explicitly.
            self.unavailable = True

    def _end_phase(self, outcome):
        if self.current is not None:
            name, started, cpu = self.current
            self._write('phase_finished', phase=name, outcome=outcome,
                seconds=max(0, time.perf_counter() - started),
                thread_cpu_seconds=max(0, time.thread_time() - cpu))
            self.current = None

    def phase(self, name):
        if self.current is not None and self.current[0] == name:
            return
        self._end_phase('transition')
        self.current = (name, time.perf_counter(), time.thread_time())
        self._write('phase_started', phase=name)

    @contextmanager
    def measure(self, name):
        started, cpu, failed = time.perf_counter(), time.thread_time(), False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            metric = self.metrics.setdefault(name, {'count': 0, 'seconds': 0.0,
                'thread_cpu_seconds': 0.0, 'max_seconds': 0.0, 'errors': 0})
            elapsed = max(0, time.perf_counter() - started)
            metric['count'] += 1
            metric['seconds'] += elapsed
            metric['thread_cpu_seconds'] += max(0, time.thread_time() - cpu)
            metric['max_seconds'] = max(metric['max_seconds'], elapsed)
            metric['errors'] += int(failed)

    def close(self, outcome):
        if self.closed:
            return
        self.closed = True
        self._end_phase(outcome)
        self._write('session_finished', outcome=outcome, metrics=self.metrics,
            seconds=max(0, time.perf_counter() - self.started),
            thread_cpu_seconds=max(0, time.thread_time() - self.cpu_started),
            prior_write_failed=self.unavailable)


@contextmanager
def diagnostic_maintenance(root):
    """Close the active stream while task records are scrubbed from log files."""
    logger = logging.getLogger("waymo.diagnostics." + str((root / 'logs/application.jsonl').resolve()))
    with _lock:
        handlers = list(logger.handlers)
        for handler in handlers:
            handler.acquire()
            if handler.stream:
                handler.stream.close()
                handler.stream = None
        try:
            yield
        finally:
            for handler in reversed(handlers):
                handler.release()


def diagnostic_event(root: Path, event: str, *, level="INFO", exception=None, **fields):
    path = root / "logs" / "application.jsonl"
    logger = logging.getLogger("waymo.diagnostics." + str(path.resolve()))
    with _lock:
        if not logger.handlers:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3,
                                          encoding="utf-8", delay=True)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
    record = {"at": datetime.now(timezone.utc).isoformat(), "level": level, "event": event, **fields}
    if exception is not None:
        record.update(error_type=type(exception).__name__, error=str(exception),
                      traceback="".join(traceback.format_exception(exception)))
    message = json.dumps(record, ensure_ascii=False, allow_nan=False)
    # Defensive redaction also covers exceptions from dependencies. Requests,
    # headers and model bodies are deliberately never passed to this logger.
    for name, value in os.environ.items():
        if any(part in name.upper() for part in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")) and len(value) >= 4:
            message = message.replace(json.dumps(value)[1:-1], "[REDACTED]")
    message = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", message)
    logger.log(getattr(logging, level), message)
