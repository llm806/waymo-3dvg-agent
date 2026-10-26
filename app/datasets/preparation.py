"""Prepare one selected frame with the existing official converter, then its target.

Subprocess arguments are fixed and shell=False. Pause/stop lets the current
official conversion settle; no subsequent target/model step starts afterwards.
"""

import json
import os
import subprocess
import time
import queue
import threading
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path

from app.infrastructure.artifacts import atomic_json, checked_asset, hash_assets
from app.infrastructure.diagnostics import diagnostic_event
from app.infrastructure.conversion_progress import PROTOCOL
from app.infrastructure.conversion_environment import ConversionEnvironmentUnavailable
from app.datasets.evidence import EVIDENCE_ROOT, ROOT, build_packet
from app.datasets.training import read_scene
from app.datasets.waymo import ContextReaders


class PreparationCancelled(RuntimeError):
    pass


class ConversionInterrupted(ValueError):
    """A killed or timed-out local worker; source/schema errors are not retryable."""
    pass


class EvidenceRejected(ValueError):
    def __init__(self, packet):
        super().__init__("target_evidence_not_observable")
        self.packet = packet


class TargetNotDistinguishable(ValueError):
    pass


class ConversionSession:
    """Bounded serial SDK worker. Each frame retains its own progress and diagnostics."""
    def __init__(self):
        self.process = None
        self.log = None
        self.events = queue.Queue()
        self.reader_thread = None

    def _read_progress(self, process):
        for line in iter(process.stdout.readline, b''):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                record = {}
            if isinstance(record, dict) and record.get('protocol') == PROTOCOL:
                self.events.put(record)
            elif self.log:
                self.log.write(line)
                self.log.flush()

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                if process.stdin:
                    process.stdin.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)
        finally:
            if self.reader_thread:
                self.reader_thread.join(timeout=2)
                self.reader_thread = None
            if process is not None and process.stdout:
                process.stdout.close()
            if self.log:
                self.log.close()
                self.log = None

    def __call__(self, context, timestamp, heartbeat, log_path):
        progress_path = log_path.with_name("preparation-progress.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        linux_root = None
        if os.name == "nt":
            resolved = ROOT.resolve()
            linux_root = "/mnt/" + resolved.drive[0].lower() + "/" + resolved.as_posix()[3:]
        if self.process is None or self.process.poll() is not None:
            self.close()
            args = ["preprocessing/.venv/bin/python", "-u", "-B", "-m", "scripts.convert_pointcloud", "--worker"]
            if linux_root:
                args = ["wsl.exe", "--cd", linux_root, "--exec", *args]
            # The batch owns this process; stdout is kept inside its deletable folder.
            self.log = log_path.open("ab")
            self.events = queue.Queue()
            self.process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                env={**os.environ, "WSL_UTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.reader_thread = threading.Thread(target=self._read_progress, args=(self.process,), daemon=True)
            self.reader_thread.start()
            heartbeat("starting_converter")
        else:
            heartbeat("reusing_converter")
        request_id, started, stage = uuid4().hex, time.monotonic(), None
        request = {"context": context, "timestamp": timestamp, "request_id": request_id,
                   "progress_file": linux_root + "/" + progress_path.relative_to(ROOT).as_posix() if linux_root else str(progress_path)}
        try:
            try:
                self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                self.process.stdin.flush()
            except BrokenPipeError as exc:
                raise ConversionInterrupted("official_conversion_failed:broken_pipe") from exc
            progress = {}
            while True:
                changed = False
                while True:
                    try:
                        record = self.events.get_nowait()
                    except queue.Empty:
                        break
                    if record.get('request_id') == request_id:
                        progress, changed = record, True
                if changed:
                    try:
                        atomic_json(progress_path, progress)
                    except OSError as exc:
                        diagnostic_event(ROOT / 'runtime/annotation', 'progress_snapshot_unavailable',
                            error_type=type(exc).__name__, job_id=log_path.parent.parent.name)
                if progress.get("stage") and stage != progress["stage"]:
                    stage = progress["stage"]
                    heartbeat(stage)
                else:
                    heartbeat()
                code = progress.get("exit_code") if progress.get("done") else self.process.poll()
                if code is not None and not progress.get('done') and self.reader_thread.is_alive():
                    self.reader_thread.join(timeout=1)
                    # Drain final output before declaring a dead worker incomplete.
                    if not self.events.empty():
                        continue
                if code is not None:
                    record = {**progress, "context": context, "timestamp": timestamp,
                              "elapsed_seconds": round(time.monotonic() - started, 3), "exit_code": code}
                    with log_path.open("ab") as log:
                        log.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                    diagnostic_event(ROOT / "runtime/annotation", "converter_exited", context=context, timestamp=timestamp,
                        job_id=log_path.parent.parent.name, ordinal=log_path.parent.name,
                        exit_code=code, stage=stage, elapsed_seconds=record["elapsed_seconds"], persistent_worker=True)
                    if code and progress.get("error_type") == "ConversionEnvironmentUnavailable":
                        raise ConversionEnvironmentUnavailable("official_conversion_failed:" + progress["reason"])
                    if code and stage and not progress.get('done'):
                        # The local worker started this frame but disappeared
                        # without a structured source/schema failure. A fresh
                        # local attempt is safe; partial scenes are not published.
                        raise ConversionInterrupted("official_conversion_failed:" + str(code))
                    if code in {9, -9, 137, 124} and not progress.get("error_type"):
                        raise ConversionInterrupted("official_conversion_failed:" + str(code))
                    if code == 124 and progress.get("error_type") == "TimeoutError":
                        raise ConversionInterrupted("official_conversion_timeout")
                    if code:
                        raise ValueError("official_conversion_failed:" + str(progress.get("error_type") or code))
                    if not progress.get("done"):
                        raise ValueError("official_conversion_failed:missing_completion")
                    return
                if time.monotonic() - started > 630:
                    raise ConversionInterrupted("official_conversion_timeout")
                time.sleep(0.2)
        except BaseException:
            self.close()
            raise


def preflight_target(reader, context, timestamp, object_id):
    """Metadata observability only; converted points must confirm identity later."""
    from app.datasets.waymo import T, object_fact
    objects = [object_fact(r) for r in reader.metadata_rows("lidar_box", context) if r[T] == timestamp]
    if len({o.object_id for o in objects}) != len(objects):
        raise ValueError("Duplicate native object key")
    target = next(o for o in objects if o.object_id == object_id)
    from app.datasets.view_selection import projected_image_seeds
    from app.datasets.waymo import O, V
    projections = [r for r in reader.metadata_rows('projected_lidar_box', context) if r[T] == timestamp]
    calibrations = {r[V]: r for r in reader.metadata_rows('camera_calibration', context)}
    sizes = {r[V]: (r['[CameraCalibrationComponent].width'], r['[CameraCalibrationComponent].height'])
        for r in reader.metadata_rows('camera_calibration', context)}
    return bool(target.lidar_points and target.lidar_points > 0
        and projected_image_seeds(projections, objects, sizes, calibrations,
            require_front_facing=False).get(object_id))



def run_conversion(context, timestamp, heartbeat, log_path):
    """Publish real converter stages, retaining raw output and the exit code."""
    progress_path = log_path.with_name("preparation-progress.json")
    progress_path.unlink(missing_ok=True)
    args = ["preprocessing/.venv/bin/python", "-u", "-B", "-m", "scripts.convert_pointcloud",
            "--context", context, "--timestamp", str(timestamp), "--progress-file"]
    if os.name == "nt":
        resolved = ROOT.resolve()
        linux_root = "/mnt/" + resolved.drive[0].lower() + "/" + resolved.as_posix()[3:]
        progress_arg = linux_root + "/" + progress_path.relative_to(ROOT).as_posix()
        # The Linux deadline also bounds the child if the Windows WSL launcher
        # disappears. No shell expansion and no provider/network access.
        args = ["wsl.exe", "--cd", linux_root, "--exec", "timeout", "--kill-after=10s", "600s", *args, progress_arg]
    else:
        args.append(str(progress_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, stage = time.monotonic(), None
    heartbeat("starting_converter")
    with log_path.open("ab") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log,
                                   env={**os.environ, "WSL_UTF8": "1"},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while True:
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                    progress = {}
                if progress.get("stage") and progress["stage"] != stage:
                    stage = progress["stage"]
                    heartbeat(stage)
                else:
                    heartbeat()
                code = process.poll()
                if code is not None:
                    diagnostic_event(ROOT / "runtime" / "annotation", "converter_exited", context=context, timestamp=timestamp,
                                     job_id=log_path.parent.parent.name, ordinal=log_path.parent.name,
                                     exit_code=code, stage=stage, elapsed_seconds=round(time.monotonic() - started, 3))
                    if code:
                        # Structured child failures contain a complete local traceback.
                        failure = progress.get("error_type")
                        if failure == "ConversionEnvironmentUnavailable":
                            raise ConversionEnvironmentUnavailable("official_conversion_failed:" + progress["reason"])
                        raise ValueError("official_conversion_failed:" + (failure or str(code)))
                    return
                if time.monotonic() - started > 630:
                    raise ValueError("official_conversion_timeout")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


class TargetPreparer:
    def __init__(self):
        # Metadata/image caches belong to a preparation thread. Another target
        # switching context must never change the reader underneath this one.
        self._local = threading.local()
        self._source_cache = {}
        self._source_lock = threading.Lock()
        self.converter = ConversionSession()
        self._conversion_lock = threading.Lock()
        self.failed_frames = {}

    def _reader(self, context):
        if not hasattr(self._local, 'readers'):
            self._local.readers = ContextReaders(ROOT / 'waymo_trainset',
                source_cache=self._source_cache, source_lock=self._source_lock)
            self._local.frame_options = {}
        return self._local.readers.get(context)

    def close(self):
        # BatchExecution joins all preparation/annotation threads first.
        with self._conversion_lock:
            self.converter.close()
            self.failed_frames.clear()
            self._local = threading.local()
            self._source_cache.clear()

    @contextmanager
    def _conversion_slot(self, heartbeat, cancelled):
        acquired = False
        waiting = False
        try:
            while not acquired:
                if cancelled():
                    raise PreparationCancelled('preparation_cancelled')
                acquired = self._conversion_lock.acquire(timeout=0.1)
                if not acquired:
                    heartbeat(None if waiting else 'waiting_for_official_converter')
                    waiting = True
            if cancelled():
                raise PreparationCancelled('preparation_cancelled')
            heartbeat('checking_converted_scene')
            yield
        finally:
            if acquired:
                self._conversion_lock.release()

    def prepare_with_fallback(self, target, catalog, heartbeat, cancelled, log_path):
        """A failed representative frame does not reject an entire track.

        Keep the existing point/identity gates and the three-frame limit. The
        caller persists all failed-frame receipts before starting model work.
        """
        attempts, seen, first_error = [], set(), None
        options = [None, *(target.get('alternative_selections') or [])]
        for option in options:
            selected = target if option is None else {**target, 'timestamp':option['timestamp'],
                'selection':option, **{k:option.get(k) for k in
                    ('source_view','source_camera_id','source_camera_object_id')}}
            timestamp = selected['timestamp']
            if timestamp in seen:
                continue
            if len(seen) >= 3:
                break
            seen.add(timestamp)
            if cancelled():
                interrupted = PreparationCancelled('preparation_cancelled')
                interrupted.preparation_attempts = attempts
                raise interrupted
            path = log_path if option is None else log_path.parent / ('frame-' + str(timestamp)) / log_path.name
            try:
                return self(selected, catalog, heartbeat, cancelled, path), attempts
            except PreparationCancelled as exc:
                exc.preparation_attempts = attempts
                raise
            except (EvidenceRejected, TargetNotDistinguishable) as exc:
                first_error = first_error or exc
                packet = exc.packet if isinstance(exc, EvidenceRejected) else None
                reason = ('converted_target_points_missing' if packet and packet.scene
                    and packet.scene.target_points == 0 else 'native_image_evidence_missing' if packet
                    else str(exc) or 'no_multimodal_candidate')
                attempts.append({'timestamp':timestamp, 'camera_name':selected.get('source_camera_id'),
                    'reason':reason, 'packet_id':packet.packet_id if packet else None, 'packet':packet})
                heartbeat('trying_alternate_preparation_frame')
        # Preserve the primary failure and all rejected evidence, rather than
        # replacing it with an opaque last-frame exception.
        first_error.preparation_attempts = attempts
        raise first_error

    def _convert_scene(self, context, timestamp, heartbeat, cancelled, log_path):
        with self._conversion_slot(heartbeat, cancelled):
            # A peer may have published this frame while we waited. The SDK
            # worker and its scene-index read/modify/write remain serial.
            index_path = ROOT / 'runtime/scenes/index.json'
            index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {'scenes':[]}
            matches = [s for s in index['scenes'] if s['context'] == context and s['timestamp'] == timestamp]
            if len(matches) > 1:
                raise ValueError('Duplicate scene identity')
            if matches:
                return
            self._convert_scene_locked(context, timestamp, heartbeat, cancelled, log_path)

    def _convert_scene_locked(self, context, timestamp, heartbeat, cancelled, log_path):
        key = (context, timestamp)
        if key in self.failed_frames:
            raise ConversionInterrupted(self.failed_frames[key])
        for attempt in range(2):
            if cancelled():
                raise PreparationCancelled("preparation_cancelled")
            try:
                self.converter(context, timestamp, heartbeat, log_path)
                return
            except ConversionInterrupted as exc:
                diagnostic_event(ROOT / 'runtime/annotation', 'conversion_interrupted',
                    context=context, timestamp=timestamp, attempt=attempt + 1,
                    reason=str(exc), job_id=log_path.parent.parent.name, ordinal=log_path.parent.name)
                if attempt == 1:
                    self.failed_frames[key] = str(exc)
                    raise
                heartbeat("retrying_converter")

    def next_frame(self, current, catalog, rejected_timestamps, heartbeat, cancelled, log_path):
        reader = self._reader(current.context)
        frame_options = self._local.frame_options
        key = (catalog['catalog_id'], current.context, current.target_object_id)
        target = next(t for t in catalog['targets'] if (t['context'], t['object_id']) == key[1:])
        if key not in frame_options:
            options = target.get('alternative_selections')
            if options is None:
                # Older frozen catalogs did not retain backup frames. Recompute
                # metadata only; __call__ still verifies every frozen source.
                from app.datasets.catalog import context_targets
                heartbeat('finding_alternate_frame')
                refreshed = next(t for t in context_targets(reader, current.context)
                    if t['object_id'] == current.target_object_id)
                options = [refreshed['selection'], *refreshed['alternative_selections']]
            frame_options[key] = [o for o in options if o]
        for option in frame_options[key]:
            if option['timestamp'] in rejected_timestamps:
                continue
            replacement = {**target, 'timestamp': option['timestamp'], 'selection': option,
                **{k: option.get(k) for k in ('source_view', 'source_camera_id', 'source_camera_object_id')}}
            try:
                return self(replacement, catalog, heartbeat, cancelled,
                    log_path.parent / ('frame-' + str(option['timestamp'])) / log_path.name)
            except (EvidenceRejected, TargetNotDistinguishable, ConversionInterrupted) as exc:
                diagnostic_event(ROOT / 'runtime/annotation', 'alternate_frame_rejected',
                    context=current.context, object_id=current.target_object_id,
                    timestamp=option['timestamp'], reason=str(exc))
        return None

    def __call__(self, target, catalog, heartbeat, cancelled, log_path):
        context, timestamp = target["context"], target["timestamp"]
        heartbeat("checking_sources")
        reader = self._reader(context)
        for name, expected in catalog["sources"][context].items():
            heartbeat()
            if reader.provenance(name, context) != expected:
                raise ValueError("catalog_source_changed:" + name)
        if cancelled():
            raise PreparationCancelled("preparation_cancelled")
        heartbeat("checking_distinguishability")
        if not preflight_target(reader, context, timestamp, target["object_id"]):
            raise TargetNotDistinguishable("no_multimodal_candidate")
        index_path = ROOT / "runtime" / "scenes" / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"scenes": []}
        matches = [s for s in index["scenes"] if s["context"] == context and s["timestamp"] == timestamp]
        if len(matches) > 1:
            raise ValueError("Duplicate scene identity")
        if not matches:
            self._convert_scene(context, timestamp, heartbeat, cancelled, log_path)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            matches = [s for s in index["scenes"] if s["context"] == context and s["timestamp"] == timestamp]
            if cancelled():
                raise PreparationCancelled("paused_after_local_conversion_saved")
        if len(matches) != 1:
            raise ValueError("Official scene conversion did not publish selected frame")
        heartbeat("validating_scene")
        scene_path = checked_asset(ROOT, matches[0]["path"], matches[0]["manifest_sha256"]) if matches[0].get("manifest_sha256") else ROOT / matches[0]["path"]
        prepared_scene = read_scene(scene_path.parent)
        scene, _ = prepared_scene
        if (scene.context, scene.frame_timestamp_micros) != (context, timestamp):
            raise ValueError("Scene identity mismatch")
        for name, expected in scene.source_components.items():
            actual = reader.provenance(name, context)
            if actual["sha256"] != expected["sha256"] or actual["file"] != expected["file"]:
                raise ValueError("scene_source_changed:" + name)
        if cancelled():
            raise PreparationCancelled("preparation_cancelled")
        heartbeat("building_evidence")
        packet = build_packet(reader, context, timestamp, target["object_id"], prepared_scene=prepared_scene,
            source_camera=target.get('source_camera_id'), source_camera_object_id=target.get('source_camera_object_id'),
            resolve_provisional_source=target.get('selection', {}).get('image_identity_status') == 'pending_point_confirmation')
        if not packet.scene or packet.scene.total_points <= 0:
            raise ValueError("official_scene_not_attached")
        hashes = packet.provenance['artifact_sha256']
        hash_assets(EVIDENCE_ROOT / packet.packet_id, hashes, expected=hashes)
        from app.datasets.correspondence import has_multimodal_evidence
        if not has_multimodal_evidence(packet):
            raise EvidenceRejected(packet)
        return packet


def register_packet(packet):
    path = EVIDENCE_ROOT / "index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"packets": [], "association_audits": [], "errors": []}
    if not any(p["packet_id"] == packet.packet_id for p in index["packets"]):
        index["packets"].append({"packet_id": packet.packet_id, "context": packet.context,
                                 "timestamp": packet.frame_timestamp_micros, "object_id": packet.target_object_id,
                                 "category": packet.target.category, "views": len(packet.views),
                                 "candidate_count": len(packet.candidates), "same_class_count": packet.contrast.same_class_count,
                                 "contrast_status": packet.contrast.status, "ready_for_generation": packet.ready_for_generation,
                                 "warnings": packet.warnings})
        index["scope"] = "prepared targets; complete track coverage is in the versioned catalog"
        atomic_json(path, index)
