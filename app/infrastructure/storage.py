"""SQLite snapshot/event transactions and exclusive leases; stdlib only."""

import json
import sqlite3
import time
import threading
from contextlib import contextmanager
from pathlib import Path


class RunBusy(RuntimeError):
    pass


class CallLimit(RuntimeError):
    pass


class RunStoreUnavailable(RuntimeError):
    pass


def live_summary_sql(source):
    """Small transactional display/selection facts; never a validation snapshot."""
    return f"""json_object(
        'projection_format','human-approval-v1',
        'run_id',run_id,
        'provider_scope',json_extract({source},'$.provider_scope'),
        'status',json_extract({source},'$.status'),
        'stop_reason',json_extract({source},'$.stop_reason'),
        'diagnostic_pass',json(CASE json_type({source},'$.diagnostic_pass')
            WHEN 'true' THEN 'true' WHEN 'false' THEN 'false' ELSE 'null' END),
        'version',json_extract({source},'$.version'),
        'batch_id',json_extract({source},'$.batch_id'),
        'retired_from_execution',json_extract({source},'$.retired_from_execution'),
        'context',json_extract({source},'$.context'),
        'model_calls_used',COALESCE(json_extract({source},'$.model_calls_used'),0),
        'final_expression',json_extract({source},'$.final_expression'),
        'manual_review',json_extract({source},'$.manual_review'),
        'inflight',json_extract({source},'$.inflight'),
        'attempts',json(COALESCE((SELECT json_group_array(json_object(
            'generated',json_extract(value,'$.generated'))) FROM json_each({source},'$.attempts')),'[]')),
        'current_packet',json_object(
            'context',CASE WHEN json_type({source},'$.current_packet.context') IS NULL
                THEN json_extract({source},'$.context') ELSE json_extract({source},'$.current_packet.context') END,
            'target_object_id',json_extract({source},'$.current_packet.target_object_id'),
            'packet_id',json_extract({source},'$.current_packet.packet_id'),
            'frame_timestamp_micros',json_extract({source},'$.current_packet.frame_timestamp_micros'),
            'source_view',json_extract({source},'$.current_packet.source_view'),
            'source_camera_object_id',json_extract({source},'$.current_packet.source_camera_object_id'),
            'target',json_extract({source},'$.current_packet.target'),
            'scene',json_extract({source},'$.current_packet.scene'))
    )"""


class RunStore:
    _maintenance_locks = {}
    _maintenance_registry_lock = threading.Lock()

    @contextmanager
    def maintenance(self, checkpoint=None):
        with self._maintenance_registry_lock:
            lock = self._maintenance_locks.setdefault(str(self.path.resolve()), threading.RLock())
        if checkpoint is None:
            lock.acquire()
        else:
            while not lock.acquire(timeout=0.2):
                checkpoint()
        try:
            if checkpoint is not None:
                checkpoint()
            yield
        finally:
            lock.release()

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.initialized = False
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            from app.infrastructure.candidate_pool import initialize
            initialize(db)
            db.executescript("""
              CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL,
                owner TEXT, lease_until REAL NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
                payload TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY(run_id, seq));
              CREATE TABLE IF NOT EXISTS controls (
                run_id TEXT PRIMARY KEY, command TEXT NOT NULL, at REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS executor (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT, lease_until REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS runs_target ON runs (
                json_extract(snapshot,'$.current_packet.context'),
                json_extract(snapshot,'$.current_packet.target_object_id'),
                json_extract(snapshot,'$.provider_scope'));
              CREATE INDEX IF NOT EXISTS runs_batch ON runs (json_extract(snapshot,'$.batch_id'));
              CREATE INDEX IF NOT EXISTS runs_audit_parent ON runs (json_extract(snapshot,'$.parent_run_id'))
                WHERE json_extract(snapshot,'$.kind')='counterfactual_audit';
              CREATE TABLE IF NOT EXISTS run_summaries (
                run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS run_cleanup_cache (
                run_id TEXT PRIMARY KEY, generation TEXT NOT NULL, metadata TEXT);
              CREATE TABLE IF NOT EXISTS asset_cleanup_cache (
                path TEXT PRIMARY KEY, signature TEXT NOT NULL, metadata TEXT NOT NULL);
              CREATE TRIGGER IF NOT EXISTS cleanup_run_insert AFTER INSERT ON runs BEGIN
                INSERT OR REPLACE INTO run_cleanup_cache VALUES(NEW.run_id,hex(randomblob(16)),NULL);
              END;
              CREATE TRIGGER IF NOT EXISTS cleanup_run_update AFTER UPDATE OF snapshot ON runs BEGIN
                DELETE FROM run_cleanup_cache WHERE run_id=OLD.run_id;
                INSERT OR REPLACE INTO run_cleanup_cache VALUES(NEW.run_id,hex(randomblob(16)),NULL);
              END;
              CREATE TRIGGER IF NOT EXISTS cleanup_run_delete AFTER DELETE ON runs BEGIN
                DELETE FROM run_cleanup_cache WHERE run_id=OLD.run_id;
              END;
              INSERT OR IGNORE INTO run_cleanup_cache SELECT run_id,hex(randomblob(16)),NULL FROM runs;
              CREATE INDEX IF NOT EXISTS run_summaries_batch ON run_summaries(json_extract(snapshot,'$.batch_id'));
              CREATE INDEX IF NOT EXISTS run_summaries_target ON run_summaries(
                json_extract(snapshot,'$.current_packet.context'),
                json_extract(snapshot,'$.current_packet.target_object_id'),
                json_extract(snapshot,'$.provider_scope'));
              CREATE INDEX IF NOT EXISTS run_summaries_audit_parent ON run_summaries(json_extract(snapshot,'$.parent_run_id'))
                WHERE json_extract(snapshot,'$.kind')='counterfactual_audit';
              CREATE TRIGGER IF NOT EXISTS run_summary_insert AFTER INSERT ON runs BEGIN
                INSERT OR REPLACE INTO run_summaries VALUES(NEW.run_id,json_remove(NEW.snapshot,
                  '$.current_packet.candidates','$.current_packet.predicates','$.current_packet.provenance',
                  '$.current_packet.contrast','$.current_packet.track','$.current_packet.views'));
              END;
              CREATE TRIGGER IF NOT EXISTS run_summary_update AFTER UPDATE OF snapshot ON runs BEGIN
                DELETE FROM run_summaries WHERE run_id=OLD.run_id;
                INSERT OR REPLACE INTO run_summaries VALUES(NEW.run_id,json_remove(NEW.snapshot,
                  '$.current_packet.candidates','$.current_packet.predicates','$.current_packet.provenance',
                  '$.current_packet.contrast','$.current_packet.track','$.current_packet.views'));
              END;
              CREATE TRIGGER IF NOT EXISTS run_summary_delete AFTER DELETE ON runs BEGIN
                DELETE FROM run_summaries WHERE run_id=OLD.run_id;
              END;
              INSERT OR IGNORE INTO run_summaries SELECT run_id,json_remove(snapshot,
                '$.current_packet.candidates','$.current_packet.predicates','$.current_packet.provenance',
                '$.current_packet.contrast','$.current_packet.track','$.current_packet.views') FROM runs
                WHERE NOT EXISTS(SELECT 1 FROM run_summaries s WHERE s.run_id=runs.run_id);
            """)
            # Persist the small live projection in the same transaction as the
            # authoritative snapshot. Historical attempts can be very large;
            # progress/readiness must not parse them on every browser poll.
            projection = live_summary_sql('NEW.snapshot').replace("'run_id',run_id", "'run_id',NEW.run_id")
            old_live = db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='run_live_update'").fetchone()
            refresh_live = old_live is not None and 'human-approval-v1' not in old_live['sql']
            if refresh_live:
                db.executescript('DROP TRIGGER IF EXISTS run_live_insert; DROP TRIGGER IF EXISTS run_live_update;')
            db.executescript(f"""
                CREATE TABLE IF NOT EXISTS run_live_summaries (run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS run_live_summaries_batch ON run_live_summaries(json_extract(snapshot,'$.batch_id'));
                CREATE TRIGGER IF NOT EXISTS run_live_insert AFTER INSERT ON runs BEGIN
                    INSERT OR REPLACE INTO run_live_summaries VALUES(NEW.run_id,{projection});
                END;
                CREATE TRIGGER IF NOT EXISTS run_live_update AFTER UPDATE OF snapshot ON runs BEGIN
                    DELETE FROM run_live_summaries WHERE run_id=OLD.run_id;
                    INSERT OR REPLACE INTO run_live_summaries VALUES(NEW.run_id,{projection});
                END;
                CREATE TRIGGER IF NOT EXISTS run_live_delete AFTER DELETE ON runs BEGIN
                    DELETE FROM run_live_summaries WHERE run_id=OLD.run_id;
                END;
                INSERT OR IGNORE INTO run_live_summaries SELECT run_id,{live_summary_sql('snapshot')}
                    FROM run_summaries WHERE run_id NOT IN (SELECT run_id FROM run_live_summaries);
            """)
            if refresh_live:
                db.execute(f"INSERT OR REPLACE INTO run_live_summaries SELECT run_id,{live_summary_sql('snapshot')} FROM run_summaries")
        self.initialized = True

    @contextmanager
    def connect(self):
        # sqlite3.connect would silently create an empty, schema-less database
        # after runtime cleanup while an old server is still alive.
        if not self.path.parent.is_dir() or (self.initialized and not self.path.is_file()):
            raise RunStoreUnavailable("运行数据库目录已被移除，请重启后端服务以初始化空数据库。")
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def claim(self, run_id: str, owner: str, initial: dict) -> dict:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO runs(run_id,snapshot,owner,lease_until) VALUES(?,?,?,?)",
                           (run_id, json.dumps(initial), owner, now + 300))
                return initial
            snapshot = json.loads(row["snapshot"])
            if snapshot.get("status") in {"calibration_hold", "auto_pass", "abstained", "blocked", "needs_evidence"}:
                return snapshot
            if row["owner"] and row["owner"] != owner and row["lease_until"] > now:
                raise RunBusy("Run already has an active executor")
            db.execute("UPDATE runs SET owner=?,lease_until=? WHERE run_id=?", (owner, now + 300, run_id))
            return snapshot

    def save(self, run_id: str, owner: str, state: dict, kind: str, payload: dict) -> str:
        # Serialize/hash outside the writer. Completed records carry their small
        # deletion-reference index forward, avoiding another full legacy scan.
        snapshot = json.dumps(state,allow_nan=False)
        cleanup = None
        if state.get('status') in {'calibration_hold','auto_pass','abstained','needs_evidence','blocked','paused','stopped'}:
            from app.infrastructure.task_cleanup import TaskCleaner
            cleanup = json.dumps(TaskCleaner.run_metadata(run_id,snapshot,state))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner,lease_until,revision FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["owner"] != owner or row["lease_until"] < time.time():
                raise RunBusy("Lease is missing, expired or owned by another executor")
            revision = row["revision"] + 1
            if kind == "model_call_reserved" and state.get("cumulative_call_limit"):
                key = self.target_key(state)
                used = (self._target_usage(db, key, state.get('provider_scope'), run_id) if isinstance(key, tuple)
                    else sum(s.get("model_calls_used", 0) for r in db.execute("SELECT run_id,snapshot FROM runs")
                        if r['run_id'] != run_id and self.target_key(s := json.loads(r['snapshot'])) == key
                        and s.get('provider_scope') == state.get('provider_scope')))
                if used + state["model_calls_used"] > state["cumulative_call_limit"]:
                    raise CallLimit("Cumulative target call budget exhausted")
            if kind == "model_call_reserved" and state.get("batch_id"):
                batch = db.execute("SELECT spec FROM batches WHERE job_id=?", (state["batch_id"],)).fetchone()
                if not batch:
                    raise CallLimit("Batch budget owner missing")
                spec = json.loads(batch["spec"])
                if (spec.get('max_model_calls') is not None
                        and state.get('retry_authorization',{}).get('kind') != 'manual_review_retry'):
                    other = db.execute("""SELECT COALESCE(SUM(COALESCE(json_extract(snapshot,'$.model_calls_used'),0)),0)
                        FROM run_summaries WHERE json_extract(snapshot,'$.batch_id')=? AND run_id!=?""",
                        (state['batch_id'],run_id)).fetchone()[0]
                    if other + state["model_calls_used"] > spec["max_model_calls"]:
                        raise CallLimit("Batch shared call budget exhausted")
            db.execute("UPDATE runs SET snapshot=?,revision=?,lease_until=? WHERE run_id=?",
                       (snapshot, revision, time.time() + 300, run_id))
            if cleanup is not None:
                db.execute('UPDATE run_cleanup_cache SET metadata=? WHERE run_id=?',(cleanup,run_id))
            if kind == 'source_view_reselected' and state.get('batch_id'):
                # Scene binding and the durable frame-switch event commit together.
                db.execute('UPDATE batch_items SET packet_id=? WHERE job_id=? AND run_id=?',
                    (state['packet_id'], state['batch_id'], run_id))
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                       (run_id, revision, kind, json.dumps(payload, allow_nan=False), time.time()))
        return snapshot

    def target_key(self, state: dict):
        packet = state.get("current_packet", {})
        if packet.get("context") and packet.get("target_object_id"):
            return (packet["context"], packet["target_object_id"])
        if state.get("kind") == "counterfactual_audit" and state.get("parent_run_id"):
            parent = self.snapshot(state["parent_run_id"])
            packet = parent.get("current_packet", {})
            if packet.get("context") and packet.get("target_object_id"):
                return (packet["context"], packet["target_object_id"])
        return state.get("packet_id")

    def target_usage(self, packet, scope: str) -> int:
        return self.target_usage_for(packet.context, packet.target_object_id, scope)

    def target_usage_for(self, context, object_id, scope):
        with self.connect() as db:
            return self._target_usage(db, (context, object_id), scope)

    @staticmethod
    def _target_usage(db, key, scope, excluded=None):
        # Audit snapshots can inherit their target from a parent. Include them
        # exactly once, without loading every historical evidence packet.
        return db.execute("""SELECT COALESCE(SUM(used),0) FROM (
            SELECT COALESCE(json_extract(snapshot,'$.model_calls_used'),0) AS used FROM run_summaries
            WHERE json_extract(snapshot,'$.current_packet.context')=?
            AND json_extract(snapshot,'$.current_packet.target_object_id')=?
            AND json_extract(snapshot,'$.provider_scope')=? AND (? IS NULL OR run_id!=?)
            UNION ALL
            SELECT COALESCE(json_extract(a.snapshot,'$.model_calls_used'),0) FROM run_summaries a
            JOIN run_summaries p ON p.run_id=json_extract(a.snapshot,'$.parent_run_id')
            WHERE json_extract(a.snapshot,'$.kind')='counterfactual_audit'
            AND (COALESCE(json_extract(a.snapshot,'$.current_packet.context'),'')=''
                OR COALESCE(json_extract(a.snapshot,'$.current_packet.target_object_id'),'')='')
            AND json_extract(p.snapshot,'$.current_packet.context')=?
            AND json_extract(p.snapshot,'$.current_packet.target_object_id')=?
            AND json_extract(a.snapshot,'$.provider_scope')=? AND (? IS NULL OR a.run_id!=?)
            )""", (*key, scope, excluded, excluded, *key, scope, excluded, excluded)).fetchone()[0]

    def acquire_executor(self, owner: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM executor WHERE singleton=1").fetchone()
            if row and row["owner"] != owner and row["lease_until"] > time.time():
                raise RunBusy("Another single or batch executor holds the dataset lease")
            db.execute("INSERT OR REPLACE INTO executor VALUES(1,?,?)", (owner, time.time() + 300))

    def renew_executor(self, owner: str):
        with self.connect() as db:
            changed = db.execute("UPDATE executor SET lease_until=? WHERE singleton=1 AND owner=? AND lease_until>=?",
                                 (time.time() + 300, owner, time.time())).rowcount
            if changed != 1:
                raise RunBusy("Dataset executor lease expired")

    def release_executor(self, owner: str):
        with self.connect() as db:
            db.execute("DELETE FROM executor WHERE singleton=1 AND owner=?", (owner,))

    def release(self, run_id: str, owner: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE runs SET owner=NULL,lease_until=0 WHERE run_id=? AND owner=?", (run_id, owner))

    def renew(self, run_id: str, owner: str) -> None:
        now = time.time()
        with self.connect() as db:
            changed = db.execute("UPDATE runs SET lease_until=? WHERE run_id=? AND owner=? AND lease_until>=?",
                                 (now + 300, run_id, owner, now)).rowcount
            if changed != 1:
                raise RunBusy("Cannot renew another or expired executor's lease")

    def events(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT seq,kind,payload,at FROM events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def snapshot(self, run_id: str, *, missing_ok: bool = False) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT snapshot FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            if missing_ok:
                return None
            raise ValueError("Run not found")
        return json.loads(row["snapshot"])

    def snapshots(self, packet_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT snapshot FROM runs").fetchall()
        return [s for r in rows if (s := json.loads(r["snapshot"])).get("packet_id") == packet_id]

    def all_runs(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT snapshot FROM runs ORDER BY rowid DESC").fetchall()
        return [json.loads(r["snapshot"]) for r in rows]

    def report_runs(self, job):
        """Task-scoped metrics facts; stream one summary, never load all geometry."""
        fields = {'run_id','batch_id','kind','parent_run_id','status','diagnostic_pass',
            'final_expression','manual_review','revalidation_only','provider_scope','version',
            'code_hashes','model','base_url','uv_lock_sha256','max_repairs','context','category',
            'auto_published','model_calls_used','elapsed_active_seconds','counterfactual_audit'}
        projected = []
        with self.connect() as db:
            rows = db.execute("""WITH scoped AS (
                SELECT run_id FROM run_summaries WHERE json_extract(snapshot,'$.batch_id')=?
                UNION SELECT run_id FROM batch_items WHERE job_id=? AND run_id IS NOT NULL)
                SELECT s.snapshot FROM scoped q JOIN run_summaries s ON s.run_id=q.run_id
                JOIN runs r ON r.run_id=s.run_id ORDER BY r.rowid DESC""",(job,job))
            for row in rows:
                state = json.loads(row['snapshot'])
                small = {key:state[key] for key in fields if key in state}
                if (state.get('manual_review') or {}).get('approval_policy')=='localized-human-approval-v1':
                    small['current_packet']={key:state.get('current_packet',{}).get(key) for key in (
                        'packet_id','context','frame_timestamp_micros','target_object_id','source_view','source_camera_object_id','target','scene')}
                small['attempts'] = []
                for attempt in state.get('attempts',[]):
                    item = {key:attempt[key] for key in ('generated','counterfactual') if key in attempt}
                    if 'verification' in attempt:
                        check = attempt['verification']
                        item['verification'] = {'diagnostic_pass':check.get('diagnostic_pass')} if check else check
                    if 'ground_checks' in attempt:
                        item['ground_checks'] = [{'diagnostic_pass':check.get('diagnostic_pass')}
                            for check in attempt['ground_checks']]
                    small['attempts'].append(item)
                projected.append(small)
        return projected

    def excluded_target_keys(self):
        from app.infrastructure.candidate_pool import excluded_keys
        with self.connect() as db:
            return excluded_keys(db)

    def selection_runs(self) -> list[dict]:
        """Selection needs track/usage history, not complete geometric packets."""
        with self.connect() as db:
            rows = db.execute('SELECT snapshot FROM run_live_summaries ORDER BY rowid DESC').fetchall()
        return [json.loads(r['snapshot']) for r in rows]

    def preparation_attempted_keys(self, *, exclude_job=None):
        """Annotation attempts can fail before a model run is ever attached."""
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='batch_items'").fetchone():
                return set()
            rows = db.execute("""SELECT DISTINCT json_extract(i.target,'$.context') AS context,
                    json_extract(i.target,'$.object_id') AS object_id
                FROM batch_items i JOIN batches b ON b.job_id=i.job_id
                WHERE json_extract(b.spec,'$.prepare_only')=0 AND i.run_id IS NULL
                AND (? IS NULL OR i.job_id!=?)
                AND json_extract(i.target,'$.status')='selected'
                AND (i.packet_id IS NOT NULL OR i.status IN
                    ('preparing','running','excluded','failed','abstained','needs_evidence'))""",(exclude_job,exclude_job))
            return {(row['context'],row['object_id']) for row in rows}

    def candidate_retries(self):
        """Explicit, unused retry authorizations; never infer a fresh allowance."""
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='batch_items'").fetchone():
                return {}
            rows = db.execute("""SELECT json_extract(target,'$.context') AS context,
                json_extract(target,'$.object_id') AS object_id,
                json_extract(target,'$.retry_authorization') AS authorization
                FROM batch_items WHERE run_id IS NULL AND status IN ('pending','preparing')
                AND json_type(target,'$.retry_authorization')='object'
                AND json_extract(target,'$.retry_authorization.kind') IS NOT 'manual_review_retry'
                AND json_extract(target,'$.retry_authorization.consumed_run_id') IS NULL""").fetchall()
        result = {}
        for row in rows:
            authorization = json.loads(row['authorization'])
            key = row['context'],row['object_id']
            if authorization.get('authorized_at',0) >= result.get(key,{}).get('authorized_at',0):
                result[key] = authorization
        return result

    def create(self, initial: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO runs(run_id,snapshot) VALUES(?,?)",
                       (initial["run_id"], json.dumps(initial)))

    def executor_active(self, run_id: str) -> bool:
        with self.connect() as db:
            r = db.execute("SELECT owner,lease_until FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return bool(r and r["owner"] and r["lease_until"] > time.time())

    def control(self, run_id: str) -> str | None:
        with self.connect() as db:
            r = db.execute("SELECT command FROM controls WHERE run_id=?", (run_id,)).fetchone()
        return r["command"] if r else None

    def request_control(self, run_id: str, command: str) -> None:
        if command not in {"pause", "stop", "resume"}:
            raise ValueError("Unknown control")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if r is None:
                raise ValueError("Run not found")
            state = json.loads(r["snapshot"])
            if state.get("kind") == "counterfactual_audit" or state["status"] not in {"running", "paused", "stopped"}:
                raise ValueError("Run is not controllable")
            if command == "resume":
                if r["owner"] and r["lease_until"] > time.time():
                    raise RunBusy("Executor still active; wait for pause/stop acknowledgement")
                db.execute("DELETE FROM controls WHERE run_id=?", (run_id,))
            else:
                prior = db.execute("SELECT command FROM controls WHERE run_id=?", (run_id,)).fetchone()
                if prior and prior["command"] == "stop" and command == "pause":
                    raise ValueError("Stop already requested")
                if prior and prior["command"] == command:
                    return
                db.execute("INSERT OR REPLACE INTO controls VALUES(?,?,?)", (run_id, command, time.time()))
            revision = r["revision"] + 1
            db.execute("UPDATE runs SET revision=? WHERE run_id=?", (revision, run_id))
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                       (run_id, revision, "control_requested", json.dumps({"command": command}), time.time()))
            if command in {"pause", "stop"} and not (r["owner"] and r["lease_until"] > time.time()):
                desired = "paused" if command == "pause" else "stopped"
                if state["status"] != desired:
                    state["status"] = desired
                    revision += 1
                    db.execute("UPDATE runs SET snapshot=?,revision=? WHERE run_id=?",
                               (json.dumps(state), revision, run_id))
                    db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (run_id, revision, "control_applied",
                               json.dumps({"command": command, "phase": state.get("phase"),
                                           "resume_safe": state.get("inflight") is None, "executor_idle": True}), time.time()))
