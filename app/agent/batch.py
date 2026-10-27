"""Durable batch coordinator over independent single-target Agent Harnesses.

SQLite is authoritative. Execution revisions freeze catalog/code; the task ID
and budgets survive manual recovery. Each item links to its original run.
"""

import hashlib
import json
import os
import time
import threading
from collections import Counter
from collections import OrderedDict
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.agent_runtime import run_agent
from app.infrastructure.artifacts import file_sha256, atomic_json
from app.datasets.catalog import load_catalog, catalog_digest
from app.agent.console import load_packet
from app.datasets.evidence import ROOT
from app.infrastructure.diagnostics import diagnostic_event
from app.datasets.preparation import TargetPreparer, register_packet
from app.quality.relations import expand_relations
from app.infrastructure.storage import RunBusy
from app.agent.selection import attempted_item, diverse_candidate_order, passed_description, usable_description, unavailable_target_keys
from app.quality.sample_review import ReviewRetry, retry_source, review_status
from app.schemas.contracts import EvidencePacket


class BatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    dataset_split: Literal['train', 'validation'] = 'train'
    target_ids: list[str] | None = None
    prepare_only: bool = True
    # None means the task ends by its goal/candidate pool, not a call quota.
    # Explicit finite quotas remain readable for historical tasks/API clients.
    max_model_calls: int | None = Field(default=None, ge=0)
    calls_per_target: int = Field(default=12, ge=4, le=12)
    max_repairs: int = Field(default=2, ge=0, le=2)
    desired_count: int | None = Field(default=None, ge=1, le=2000)
    target_concurrency: int = Field(default=4, ge=1, le=4)
    review_retry: ReviewRetry | None = None

    @model_validator(mode='after')
    def check_goal(self):
        if self.prepare_only and self.desired_count is not None:
            raise ValueError('A description goal requires annotation, not preparation only')
        if self.review_retry and (self.prepare_only or self.desired_count is not None
                or not self.target_ids or len(self.target_ids) != 1):
            raise ValueError('Review retry requires one explicit annotation target')
        return self

    def call_allowance(self, consumed):
        """Calls available for one target within an optional task quota."""
        if self.max_model_calls is None:
            return self.calls_per_target
        return min(self.calls_per_target, max(0, self.max_model_calls - consumed))


def method_sources():
    files = [*sorted((ROOT / "app").rglob("*.py")), ROOT / "scripts" / "convert_pointcloud.py",
             ROOT / "uv.lock", ROOT / "preprocessing" / "uv.lock"]
    return files


def method_fingerprint():
    return hashlib.sha256(json.dumps({p.relative_to(ROOT).as_posix():file_sha256(p) for p in method_sources()},sort_keys=True).encode()).hexdigest()


def freeze_method(root, fingerprint):
    folder=root/'methods'/fingerprint
    if (folder/'manifest.json').exists():return
    hashes={}
    for source in method_sources():
        relative=source.relative_to(ROOT).as_posix()
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes();path.write_bytes(data);hashes[relative]=hashlib.sha256(data).hexdigest()
    if hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=fingerprint:
        raise ValueError('Method sources changed while freezing the experiment')
    atomic_json(folder/'manifest.json',{'method_sha256':fingerprint,'files':hashes,'scope':'backend, converter and dependency locks; no credentials or data'})


class BatchStore:
    def __init__(self, runs):
        self.runs = runs
        self.detail_cache = OrderedDict()
        self.detail_cache_lock = threading.Lock()
        with runs.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS batches (
                job_id TEXT PRIMARY KEY, spec TEXT NOT NULL, catalog_sha256 TEXT NOT NULL,
                method_sha256 TEXT NOT NULL, status TEXT NOT NULL, control TEXT,
                owner TEXT, reason TEXT, created_at REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS batch_items (
                job_id TEXT NOT NULL, ordinal INTEGER NOT NULL, target TEXT NOT NULL,
                status TEXT NOT NULL, packet_id TEXT, run_id TEXT, reason TEXT,
                reused INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job_id,ordinal));
              CREATE TABLE IF NOT EXISTS batch_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS batch_items_run ON batch_items(job_id,run_id);
              CREATE INDEX IF NOT EXISTS batch_events_job_kind ON batch_events(job_id,kind,seq);
              CREATE TABLE IF NOT EXISTS batch_queue_control (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), paused INTEGER NOT NULL);
              INSERT OR IGNORE INTO batch_queue_control VALUES(1,0);
              -- Derived presentation rows only. Random generations prevent an
              -- in-flight reader from publishing a stale row after a delete /
              -- recreate or an external snapshot edit with unchanged revision.
              CREATE TABLE IF NOT EXISTS sample_presentations (
                job_id TEXT NOT NULL, ordinal INTEGER NOT NULL, run_id TEXT,
                generation TEXT NOT NULL, format TEXT, presentation TEXT,
                PRIMARY KEY(job_id,ordinal));
              CREATE INDEX IF NOT EXISTS sample_presentations_run ON sample_presentations(run_id);
              CREATE TRIGGER IF NOT EXISTS sample_item_insert AFTER INSERT ON batch_items BEGIN
                INSERT OR REPLACE INTO sample_presentations(job_id,ordinal,run_id,generation)
                VALUES(NEW.job_id,NEW.ordinal,NEW.run_id,hex(randomblob(16)));
              END;
              CREATE TRIGGER IF NOT EXISTS sample_item_update AFTER UPDATE ON batch_items BEGIN
                DELETE FROM sample_presentations WHERE job_id=OLD.job_id AND ordinal=OLD.ordinal;
                INSERT OR REPLACE INTO sample_presentations(job_id,ordinal,run_id,generation)
                VALUES(NEW.job_id,NEW.ordinal,NEW.run_id,hex(randomblob(16)));
              END;
              CREATE TRIGGER IF NOT EXISTS sample_item_delete AFTER DELETE ON batch_items BEGIN
                DELETE FROM sample_presentations WHERE job_id=OLD.job_id AND ordinal=OLD.ordinal;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_batch_delete AFTER DELETE ON batches BEGIN
                DELETE FROM sample_presentations WHERE job_id=OLD.job_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_run_insert AFTER INSERT ON runs BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=NEW.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_run_update AFTER UPDATE OF snapshot ON runs BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=OLD.run_id OR run_id=NEW.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_run_delete AFTER DELETE ON runs BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=OLD.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_source_insert AFTER INSERT ON events
              WHEN NEW.kind='source_view_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=NEW.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_source_update AFTER UPDATE ON events
              WHEN OLD.kind='source_view_reselected' OR NEW.kind='source_view_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=OLD.run_id OR run_id=NEW.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_source_delete AFTER DELETE ON events
              WHEN OLD.kind='source_view_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE run_id=OLD.run_id;
              END;
              CREATE TRIGGER IF NOT EXISTS sample_preparation_insert AFTER INSERT ON batch_events
              WHEN NEW.kind='preparation_source_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE job_id=NEW.job_id AND ordinal=json_extract(NEW.payload,'$.ordinal');
              END;
              CREATE TRIGGER IF NOT EXISTS sample_preparation_update AFTER UPDATE ON batch_events
              WHEN OLD.kind='preparation_source_reselected' OR NEW.kind='preparation_source_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE (job_id=OLD.job_id AND ordinal=json_extract(OLD.payload,'$.ordinal'))
                   OR (job_id=NEW.job_id AND ordinal=json_extract(NEW.payload,'$.ordinal'));
              END;
              CREATE TRIGGER IF NOT EXISTS sample_preparation_delete AFTER DELETE ON batch_events
              WHEN OLD.kind='preparation_source_reselected' BEGIN
                UPDATE sample_presentations SET generation=hex(randomblob(16)),presentation=NULL,format=NULL
                WHERE job_id=OLD.job_id AND ordinal=json_extract(OLD.payload,'$.ordinal');
              END;
              INSERT OR IGNORE INTO sample_presentations(job_id,ordinal,run_id,generation)
                SELECT job_id,ordinal,run_id,hex(randomblob(16)) FROM batch_items;
            """)

    @staticmethod
    def event(db, job, kind, payload):
        db.execute("INSERT INTO batch_events(job_id,kind,payload,at) VALUES(?,?,?,?)",
                   (job, kind, json.dumps(payload, allow_nan=False), time.time()))

    def queue(self):
        """FIFO is the durable enqueue-event sequence, independent of task age."""
        with self.runs.connect() as db:
            rows = db.execute("""SELECT b.job_id,e.seq,e.payload FROM batches b
                JOIN batch_events e ON e.seq=(SELECT MAX(seq) FROM batch_events
                    WHERE job_id=b.job_id AND kind='batch_enqueued')
                WHERE b.status='queued' AND b.control IS NULL ORDER BY e.seq""").fetchall()
            paused = bool(db.execute('SELECT paused FROM batch_queue_control WHERE singleton=1').fetchone()[0])
        return [{**dict(row), 'intent':json.loads(row['payload']), 'position':index+1,
                 'queue_paused':paused} for index,row in enumerate(rows)]

    def suspend_queue(self):
        with self.runs.connect() as db:
            db.execute('UPDATE batch_queue_control SET paused=1 WHERE singleton=1')

    def continuation_budget(self, job):
        from app.infrastructure.task_lineage import read_lineage, lineage_family
        with self.runs.connect() as db:
            live, retained, parents = read_lineage(db)
        root, family = lineage_family(job, parents)
        root_spec = (retained | live)[root]['spec']
        with self.runs.connect() as db:
            spent = sum(self.reserved_calls(db, id_) for id_ in family & live.keys())
        spent += sum(retained[id_]['model_calls_reserved'] for id_ in (family & retained.keys()) - live.keys())
        return {'root':root, 'root_spec':root_spec, 'limit':root_spec['max_model_calls'],
                'spent':spent, 'is_continuation':job in parents, 'job_ids':sorted(family & live.keys())}

    def create(self, spec, catalog, continuation=None):
        if spec.review_retry:
            raise ValueError('人工返工必须在原任务内重试，不能新建任务。')
        if not spec.prepare_only:
            from app.agent.model_calls import image_model_error
            error = image_model_error(os.getenv('DASHSCOPE_MODEL', 'qwen3.8-flash'),
                os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'))
            if error:
                raise ValueError(error)
        if not spec.prepare_only and catalog.get('selection_policy') and catalog['selection_policy'].get('admission_unit') not in {'verified_native_image_pair', 'official_projection_candidate'}:
            raise ValueError('Catalog predates verified image-first selection; rebuild the catalog before annotation')
        requested = spec.target_ids
        official_split = 'training' if spec.dataset_split == 'train' else 'validation'
        split_targets = [t for t in catalog['targets'] if t.get('official_split', 'training') == official_split]
        available = {t['target_id'] for t in split_targets}
        if requested is not None and (not requested or len(set(requested)) != len(requested) or set(requested) - available):
            raise ValueError('所选物体与任务数据集不符，请选择对应官方训练集或验证集的候选。')
        if not spec.prepare_only and spec.call_allowance(0) < 4:
            raise ValueError("A model batch requires at least four reserved calls")
        specification = spec.model_dump()
        specification["provider"] = None if spec.prepare_only else {
            "model": os.getenv("DASHSCOPE_MODEL", "qwen3.8-flash"),
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")}
        if requested is not None:
            specification["target_ids"] = sorted(requested)
        digest = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        method = method_fingerprint()
        freeze_method(self.runs.path.parent,method)
        identity = [specification, digest, method]
        if continuation:
            identity.append({'continued_from': continuation['job_id'],
                             'inherited_runs': sorted(item['run_id'] for item in continuation['inherited_items'])})
        job = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        targets = [t for t in split_targets if requested is None or t['target_id'] in requested]
        retries = self.runs.candidate_retries()
        targets = [{**t,'retry_authorization':retries[t['context'],t['object_id']],
            'execution_attempt':retries[t['context'],t['object_id']]['token']}
            if (t.get('context'),t.get('object_id')) in retries else t for t in targets]
        if not targets:
            raise ValueError('所选数据集暂无候选，请先放入对应官方 training／validation Parquet 并重建目录。')
        if not spec.prepare_only and catalog.get('selection_policy',{}).get('admission_unit') == 'verified_native_image_pair':
            if requested is None:
                targets = [t for t in targets if t.get('eligibility_stage')=='native_image_pair_confirmed']
            if not targets or any(t.get('eligibility_stage')!='native_image_pair_confirmed' for t in targets):
                raise ValueError('Only confirmed native image / 3D pairs may enter annotation tasks; refresh the image pairing pool')
        if not spec.prepare_only and catalog.get('selection_policy', {}).get('admission_unit') == 'official_projection_candidate':
            if not targets or any(t.get('eligibility_stage') != 'official_projection_candidate' for t in targets if t['status'] == 'selected'):
                raise ValueError('Targets must retain their official same-frame projection identity')
        with self.runs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A deleted task can still own a continuation's cost history. A new
            # task with the same frozen specification must have a fresh identity.
            from app.infrastructure.task_lineage import read_lineage
            _, retained, _ = read_lineage(db)
            while job in retained:
                job = hashlib.sha256((job + ':new-task').encode()).hexdigest()[:24]
            if db.execute("SELECT 1 FROM batches WHERE job_id=?", (job,)).fetchone():
                return job, True
            if spec.desired_count is not None:
                targets = [t for t in targets if t['status'] == 'selected']
                targets = diverse_candidate_order(targets)
                if not targets:
                    raise ValueError('No eligible candidates in the requested pool')
                targets = [dict(target, queue_position=index) for index, target in enumerate(targets)]
            db.execute("INSERT INTO batches VALUES(?,?,?,?, 'ready',NULL,NULL,NULL,?)",
                       (job, json.dumps(specification), digest, method, time.time()))
            for target in targets:
                db.execute("INSERT INTO batch_items(job_id,ordinal,target,status,reason) VALUES(?,?,?,?,?)",
                           (job, target["ordinal"], json.dumps(target), "pending" if target["status"] == "selected" else "excluded",
                            None if target["status"] == "selected" else json.dumps(target["frame_exclusion_counts"])))
            self.event(db, job, "batch_created", {"total_tracks": len(targets), "prepare_only": spec.prepare_only,
                                                  "catalog_id": spec.catalog_id, "method_sha256": method,
                                                  "catalog_summary": catalog.get('summary', {})})
            if continuation:
                preparations = {}
                for receipt in db.execute("SELECT payload FROM batch_events WHERE job_id=? AND kind='preparation_source_reselected' ORDER BY seq",(continuation['job_id'],)):
                    payload = json.loads(receipt['payload'])
                    preparations.setdefault(payload['ordinal'], []).append(payload)
                for item in continuation['inherited_items']:
                    row = db.execute("SELECT ordinal,target FROM batch_items WHERE job_id=? AND json_extract(target,'$.target_id')=?",
                                     (job,item['target']['target_id'])).fetchone()
                    if row is None:
                        raise ValueError('已通过目标已不在候选目录中，无法继承；原结果保留。')
                    target = {**item['target'], 'queue_position': json.loads(row['target']).get('queue_position',0)}
                    db.execute("""UPDATE batch_items SET target=?,status='reused',packet_id=?,run_id=?,reused=1,reason=NULL
                        WHERE job_id=? AND ordinal=?""", (json.dumps(target),item['packet_id'],item['run_id'],job,row['ordinal']))
                    for receipt in preparations.get(item['ordinal'], []):
                        self.event(db,job,'preparation_source_reselected',
                            {**receipt,'ordinal':row['ordinal'],'inherited_from_job_id':continuation['job_id']})
                self.event(db,job,'continued_from',{k:v for k,v in continuation.items() if k!='inherited_items'})
        return job, False

    def summaries(self):
        """List durable tasks without loading sample snapshots or event histories."""
        with self.runs.connect() as db:
            headers = [dict(row) for row in db.execute('SELECT * FROM batches ORDER BY created_at DESC')]
            lease = db.execute('SELECT * FROM executor WHERE singleton=1').fetchone()
            counts = {}
            for row in db.execute('SELECT job_id,status,COUNT(*) AS n FROM batch_items GROUP BY job_id,status'):
                counts.setdefault(row['job_id'], {})[row['status']] = row['n']
            for row in db.execute("""SELECT i.job_id,i.status,json_extract(r.snapshot,'$.final_expression') AS text,
                json_extract(r.snapshot,'$.attempts') AS attempts,json_extract(r.snapshot,'$.manual_review') AS review,
                CASE WHEN json_extract(r.snapshot,'$.manual_review.verdict')='accepted' THEN r.snapshot END AS review_state
                FROM batch_items i JOIN run_live_summaries r ON r.run_id=i.run_id
                WHERE json_extract(r.snapshot,'$.manual_review.verdict') IN ('accepted','rejected','discarded')"""):
                verdict = review_status({'final_expression':row['text'], 'attempts':json.loads(row['attempts'] or '[]'),
                    'manual_review':json.loads(row['review'])})
                if row['review_state'] and usable_description(json.loads(row['review_state'])):
                    from app.quality.sample_review import manually_approved
                    if manually_approved(json.loads(row['review_state'])):
                        verdict='manual_approved'
                if verdict in {'rejected','discarded','manual_approved'}:
                    group=counts[row['job_id']]
                    group[row['status']] -= 1
                    if not group[row['status']]:group.pop(row['status'])
                    group[verdict]=group.get(verdict,0)+1
            continuations = {row['job_id']: {'source_job_id': row['source_job_id'], 'inherited_passed': row['inherited_passed']}
                for row in db.execute("""SELECT job_id,json_extract(payload,'$.job_id') AS source_job_id,
                    COALESCE(json_extract(payload,'$.inherited_passed'),0) AS inherited_passed
                    FROM batch_events WHERE kind='continued_from'""")}
        result = []
        queue = {row['job_id']:row for row in self.queue()}
        for header in headers:
            job = header['job_id']
            header['spec'] = json.loads(header['spec'])
            header['spec'].pop('target_ids', None)
            owner = header.pop('owner')
            header['executor_active'] = bool(lease and lease['owner'] == owner
                and lease['lease_until'] > time.time())
            result.append({**header, 'counts': counts.get(job, {}),
                'queue_position':queue.get(job,{}).get('position'),
                'queue_paused':queue.get(job,{}).get('queue_paused',False),
                'continuation': continuations.get(job),
                'selection_progress': {'requested': header['spec'].get('desired_count')}})
        return result

    def compact_states(self, db, job):
        """Bounded immutable JSON projections, keyed by snapshot generation.

        A lease/revision number alone misses external same-revision edits and
        delete/recreate cycles. Existing transactional random generations cover
        both; every poll rereads them before accepting a cached projection.
        """
        generations = db.execute("""SELECT s.run_id,c.generation FROM run_summaries s
            JOIN run_cleanup_cache c ON c.run_id=s.run_id WHERE s.run_id IN
            (SELECT run_id FROM batch_items WHERE job_id=?)""", (job,)).fetchall()
        projected, missing = {}, []
        with self.detail_cache_lock:
            for row in generations:
                rid = row['run_id']
                cached = self.detail_cache.get(rid)
                if cached and cached[0] == row['generation']:
                    projected[rid] = cached[1]
                    self.detail_cache.move_to_end(rid)
                else:
                    missing.append(rid)
        # Do not hold a Python lock across SQLite reads or JSON decoding.
        for start in range(0,len(missing),300):
            group = missing[start:start+300]
            placeholders = ','.join('?' for _ in group)
            rows = db.execute(f"""SELECT s.run_id,c.generation,json_remove(s.snapshot,
                '$.candidate_appearance','$.resolved_target_observation','$.tool_results',
                '$.chosen_action','$.code_hashes','$.scene_observation_reuse_receipts') AS snapshot
                FROM run_summaries s JOIN run_cleanup_cache c ON c.run_id=s.run_id
                WHERE s.run_id IN ({placeholders})""", group)
            small_rows = []
            for row in rows:
                state = json.loads(row['snapshot'])
                fields = {'run_id','status','diagnostic_pass','final_expression','manual_review','phase','version',
                    'stop_reason','inflight','model','base_url','model_calls_used','provider_scope',
                    'target_precheck_observation','rejected_source_views','admission'}
                small = {k:v for k,v in state.items() if k in fields}
                small['attempts'] = [{'generated': a.get('generated')} for a in state.get('attempts', [])]
                packet = state.get('current_packet') or {}
                small['current_packet'] = {k:v for k,v in packet.items() if k in {
                    'packet_id','context','target_object_id','frame_timestamp_micros','source_view','source_camera_object_id','target','scene'}}
                small['current_packet']['task'] = {'admission_policy': (packet.get('task') or {}).get('admission_policy')}
                small_rows.append((row['run_id'],row['generation'],json.dumps(small)))
            with self.detail_cache_lock:
                for rid, generation, raw in small_rows:
                    projected[rid] = raw
                    self.detail_cache[rid] = (generation,raw)
                    self.detail_cache.move_to_end(rid)
                while len(self.detail_cache)>8192:
                    self.detail_cache.popitem(last=False)
        # Return fresh objects so a renderer/caller cannot mutate cached facts.
        return {rid:json.loads(raw) for rid,raw in projected.items()}

    def progress(self, job):
        """Live counters/timers without decoding historical evidence or results.

        Read one SQLite snapshot so counts, reviews and the cost ledger agree.
        This is a display projection; it never authorizes or dispatches work.
        """
        with self.runs.connect() as db:
            db.execute('BEGIN')
            row = db.execute('SELECT * FROM batches WHERE job_id=?', (job,)).fetchone()
            if row is None:
                raise ValueError('Batch not found')
            header = dict(row)
            spec = json.loads(header.pop('spec'))
            lease = db.execute('SELECT * FROM executor WHERE singleton=1').fetchone()
            now = time.time()
            active = bool(lease and lease['owner'] == header.pop('owner') and lease['lease_until'] > now)
            counts, passed, attempted = Counter(), set(), 0
            for item in db.execute("""SELECT i.status,i.packet_id,i.run_id,
                    json_extract(i.target,'$.target_id') AS target_id,
                    json_extract(r.snapshot,'$.status') AS run_status,
                    json_extract(r.snapshot,'$.diagnostic_pass') AS diagnostic_pass,
                    json_extract(r.snapshot,'$.final_expression') AS final_expression,
                    json_extract(r.snapshot,'$.manual_review') AS manual_review,
                    CASE WHEN json_extract(r.snapshot,'$.manual_review.verdict')='accepted' THEN r.snapshot END AS review_state,
                    CASE WHEN json_type(r.snapshot,'$.manual_review')='object'
                        THEN json_extract(r.snapshot,'$.attempts') END AS attempts
                FROM batch_items i LEFT JOIN run_live_summaries r ON r.run_id=i.run_id
                WHERE i.job_id=?""", (job,)):
                if item['run_id'] and item['run_status'] is None:
                    raise ValueError('Run not found')
                verdict = None
                if item['manual_review']:
                    verdict = review_status({'final_expression': item['final_expression'],
                        'attempts': json.loads(item['attempts'] or '[]'),
                        'manual_review': json.loads(item['manual_review'])})
                human = False
                if item['review_state']:
                    from app.quality.sample_review import manually_approved
                    human=manually_approved(json.loads(item['review_state']))
                counts['manual_approved' if human else verdict if verdict in {'rejected','discarded'} else item['status']] += 1
                attempted += attempted_item(dict(item))
                if human or (item['run_status'] in {'calibration_hold','auto_pass'} and item['diagnostic_pass'] == 1
                        and item['final_expression'] and verdict not in {'rejected','discarded'}):
                    passed.add(item['target_id'])
            consumed = db.execute("""SELECT COALESCE(SUM(json_extract(snapshot,'$.model_calls_used')),0)
                FROM run_live_summaries WHERE json_extract(snapshot,'$.batch_id')=? OR run_id IN
                    (SELECT run_id FROM batch_items WHERE job_id=? AND reused=0)""", (job,job)).fetchone()[0]
            events = [{**dict(e), 'payload': json.loads(e['payload'])} for e in db.execute("""
                SELECT kind,at,payload FROM batch_events WHERE job_id=? AND kind IN
                ('batch_started_or_resumed','batch_completed','batch_blocked','timing_execution_started',
                 'timing_execution_finished','timing_item_started','timing_item_finished') ORDER BY seq""", (job,))]
            review_retries = self.review_retry_progress(db, job)
        from app.infrastructure.batch_timing import summarize_timing
        timing, item_timings = summarize_timing(events, status=header['status'], executor_active=active, now=now)
        return {'job_id': job, 'status': header['status'], 'control': header['control'], 'reason': header['reason'],
            'executor_active': active, 'counts': dict(counts), 'model_calls_reserved': consumed,
            'selection_progress': {'requested': spec.get('desired_count'), 'diagnostic_passed': len(passed),
                'attempted': attempted, 'candidate_pool': sum(counts.values())},
            'timing': timing, 'measured_at': now,
            'review_retries': review_retries,
            'active_item_timings': {str(k): v for k,v in item_timings.items() if v['running']}}

    @staticmethod
    def review_retry_progress(db, job, ordinal=None):
        """Latest review rounds only; do not decode historical evidence or prompts."""
        rows = db.execute("""SELECT i.ordinal,i.status,i.packet_id,i.run_id,
            COALESCE(i.reason,json_extract(r.snapshot,'$.stop_reason')) AS reason,
            json_extract(i.target,'$.retry_authorization.authorized_at') AS requested_at,
            json_extract(r.snapshot,'$.phase') AS phase,
            COALESCE(json_extract(r.snapshot,'$.model_calls_used'),0) AS calls,
            COALESCE(json_extract(r.snapshot,'$.model_call_budget'),12) AS call_budget,
            json_type(r.snapshot,'$.inflight')='object' AS model_pending,
            json_extract(r.snapshot,'$.feedback_check') AS feedback_check,
            json_extract(r.snapshot,'$.current_packet.task.feedback_acceptance_policy') AS feedback_policy,
            json_extract(r.snapshot,'$.current_packet.task.feedback_detail_policy') AS feedback_detail_policy,
            json_extract(r.snapshot,'$.current_packet.task.feedback_relation_policy') AS feedback_relation_policy,
            json_extract(r.snapshot,'$.current_packet.task.feedback_feature_policy') AS feedback_feature_policy,
            json_extract(r.snapshot,'$.current_packet.task.manual_review_feedback') AS feedback_request,
            json_extract(r.snapshot,'$.attempts[#-1].generated') AS feedback_generated,
            json_extract(r.snapshot,'$.diagnostic_pass')=1
                AND length(json_extract(r.snapshot,'$.final_expression'))>0
                AND json_extract(r.snapshot,'$.status') IN ('calibration_hold','auto_pass') AS automatic_passed
            FROM batch_items i LEFT JOIN run_summaries r ON r.run_id=i.run_id
            WHERE i.job_id=? AND json_type(i.target,'$.manual_review_retry')='object'
            AND (? IS NULL OR i.ordinal=?)
            ORDER BY i.status IN ('pending','preparing','running') DESC,requested_at DESC,i.ordinal DESC
            LIMIT 20""", (job,ordinal,ordinal)).fetchall()
        result = [{**dict(row), 'automatic_passed':bool(row['automatic_passed']),
            'model_pending':bool(row['model_pending']), 'preparation_stage':None} for row in rows]
        for item in result:
            check = item['feedback_check']
            policy = item.pop('feedback_policy')
            detail_policy = item.pop('feedback_detail_policy')
            relation_policy = item.pop('feedback_relation_policy')
            feature_policy = item.pop('feedback_feature_policy')
            request = item.pop('feedback_request')
            generated = item.pop('feedback_generated')
            item['feedback_check'] = json.loads(check) if check else None
            if request and not check and item['status'] not in {'pending','preparing','running','paused','stopped'}:
                from app.quality.joint_reference import ACCEPTANCE_POLICY, feedback_acceptance
                item['feedback_check'] = {**feedback_acceptance({'manual_review_feedback': json.loads(request),
                    **({'feedback_acceptance_policy': policy} if policy else {})},
                    json.loads(generated) if generated else None), 'historical': policy != ACCEPTANCE_POLICY}
            if request and detail_policy != 'explicit-feedback-details-v1' and item['status'] not in {'pending','preparing','running','paused','stopped'}:
                from app.quality.joint_reference import feedback_acceptance, NATURAL_CONCISION_POLICY
                supplement = feedback_acceptance({'manual_review_feedback':json.loads(request),
                    'feedback_acceptance_policy':NATURAL_CONCISION_POLICY,
                    'feedback_detail_policy':'explicit-feedback-details-v1'}, json.loads(generated) if generated else None)
                details = [r for r in supplement['requirements'] if r['kind'] in {'ego_distance','retained_phrase'}]
                if details:
                    # Presentation only: preserve the frozen run and original
                    # verdict, calls, and budget. No historical re-execution.
                    original = item['feedback_check'] or {}
                    item['feedback_check'] = {**original, 'historical':True,
                        'requirements':[*original.get('requirements',[]),*details]}
            if request and relation_policy != 'bilingual-landmark-feedback-v2-omissions' and item['status'] not in {'pending','preparing','running','paused','stopped'}:
                from app.quality.joint_reference import feedback_acceptance, NATURAL_CONCISION_POLICY, RELATION_FEEDBACK_POLICY
                supplement = feedback_acceptance({'manual_review_feedback':json.loads(request),
                    'feedback_acceptance_policy':NATURAL_CONCISION_POLICY,
                    'feedback_relation_policy':RELATION_FEEDBACK_POLICY}, json.loads(generated) if generated else None)
                original = item['feedback_check'] or {}
                known = {(r['kind'],r['label']) for r in original.get('requirements',[])}
                details = [r for r in supplement['requirements'] if r['kind']=='landmark_correction' and (r['kind'],r['label']) not in known]
                if details:
                    # A historically misread positive request must not remain
                    # beside the corrected omission as if both were required.
                    corrected = {r['label'].split('：',1)[-1] for r in details if r['label'].startswith('排除：')}
                    retained = [r for r in original.get('requirements',[]) if not (
                        r['kind']=='landmark_correction' and r['label'].split('：',1)[-1] in corrected)]
                    item['feedback_check'] = {**original, 'historical':True,
                        'requirements':[*retained,*details],
                        'passed':bool(original.get('passed')) and not any(r['status']=='missing' for r in details),
                        'blocking_errors':list(dict.fromkeys([*original.get('blocking_errors',[]),
                            *supplement.get('blocking_errors',[])]))}
            if request and feature_policy != 'verified-target-feature-feedback-v1' and item['status'] not in {'pending','preparing','running','paused','stopped'}:
                from app.quality.joint_reference import feedback_acceptance, TARGET_FEATURE_FEEDBACK_POLICY
                supplement = feedback_acceptance({'manual_review_feedback':json.loads(request),
                    'feedback_feature_policy':TARGET_FEATURE_FEEDBACK_POLICY}, json.loads(generated) if generated else None)
                details = [r for r in supplement['requirements'] if r['kind']=='target_feature']
                if details:
                    original = item['feedback_check'] or {}
                    item['feedback_check'] = {**original,'historical':True,
                        'requirements':[*original.get('requirements',[]),*details],
                        'passed':bool(original.get('passed')) and not any(r['status']=='missing' for r in details),
                        'blocking_errors':list(dict.fromkeys([*original.get('blocking_errors',[]),*supplement['blocking_errors']]))}
            if item['status'] != 'preparing':
                continue
            # The existing (job,kind,seq) index bounds this to the new round.
            # Waiting/completed rows need no scan of old preparation events.
            row = db.execute("""SELECT json_extract(payload,'$.stage') AS stage FROM batch_events
                WHERE job_id=? AND kind='preparation_progress' AND seq>COALESCE((
                    SELECT MAX(seq) FROM batch_events WHERE job_id=? AND kind='manual_review_retry_requested'
                    AND json_extract(payload,'$.ordinal')=?),0)
                AND at>=? AND json_extract(payload,'$.ordinal')=? ORDER BY seq DESC LIMIT 1""",
                (job,job,item['ordinal'],item['requested_at'] or 0,item['ordinal'])).fetchone()
            if row:
                item['preparation_stage'] = row['stage']
        return result

    def detail(self, job, *, compact=False):
        with self.runs.connect() as db:
            row = db.execute("SELECT * FROM batches WHERE job_id=?", (job,)).fetchone()
            if row is None:
                raise ValueError("Batch not found")
            header = dict(row)
            header["spec"] = json.loads(header["spec"])
            items = [{**dict(r), "target": json.loads(r["target"])} for r in
                     db.execute("SELECT * FROM batch_items WHERE job_id=? ORDER BY ordinal", (job,))]
            if header['spec'].get('desired_count') is not None:
                items.sort(key=lambda item: item['target']['queue_position'])
            if compact:
                # Keep all timing intervals, but decode only the latest preparation
                # updates needed by the live view. Older progress payloads contain
                # no additional timing evidence and dominate long task histories.
                event_rows = db.execute("""SELECT * FROM batch_events WHERE job_id=? AND (
                    kind IN ('continued_from','batch_started_or_resumed','batch_completed','batch_blocked',
                        'timing_execution_started','timing_execution_finished','timing_item_started','timing_item_finished')
                    OR seq IN (SELECT MAX(seq) FROM batch_events WHERE job_id=?
                        AND kind IN ('preparation_started','preparation_progress')
                        AND json_type(payload,'$.ordinal')='integer'
                        GROUP BY kind,json_extract(payload,'$.ordinal'))
                    OR seq=(SELECT MAX(seq) FROM batch_events WHERE job_id=?
                        AND json_type(payload,'$.ordinal')='integer')) ORDER BY seq""", (job,job,job))
            else:
                event_rows = db.execute("SELECT * FROM batch_events WHERE job_id=? ORDER BY seq", (job,))
            events = [{**dict(r), "payload": json.loads(r["payload"])} for r in event_rows]
            lease = db.execute("SELECT * FROM executor WHERE singleton=1").fetchone()
            # The list/detail renderer needs identity and outcome fields, not
            # full geometric inventories. Fetch all linked outcomes in one read;
            # authoritative snapshots used by validation/export stay untouched.
            if compact:
                states = self.compact_states(db,job)
            else:
                states = {r['run_id']: json.loads(r['snapshot']) for r in db.execute("""
                    SELECT run_id,snapshot FROM run_summaries
                    WHERE run_id IN (SELECT run_id FROM batch_items WHERE job_id=?)""", (job,))}
        active = bool(lease and lease["owner"] == header["owner"] and lease["lease_until"] > time.time())
        header["executor_active"] = active
        header.pop("owner")
        consumed = 0
        run_ids = set()
        passed = set()
        for item in items:
            item['source_view'] = item['target'].get('source_view')
            item['source_camera_object_id'] = item['target'].get('source_camera_object_id')
            item['source_view_status'] = 'candidate' if item['source_view'] else 'unrecorded'
            item['admission_status'] = 'rejected' if item.get('reason') == 'target_evidence_not_observable' else 'not_checked'
            item['result_label'] = ('证据未确认 · 已跳过' if item.get('reason') == 'target_evidence_not_observable' else None)
            item['retryable_evidence_failure'] = (not item.get('run_id')
                and item.get('reason') == 'target_evidence_not_observable'
                and bool(item['target'].get('alternative_selections')))
            if item.get('reason') == 'target_evidence_not_observable' and item.get('packet_id'):
                try:
                    rejected = load_packet(item['packet_id'])
                    if rejected.scene and rejected.scene.target_points == 0:
                        item['display_reason'] = 'converted_target_points_missing'
                        item['result_label'] = '当前帧目标无点 · 未生成'
                except (OSError, ValueError, KeyError):
                    pass  # Historical missing evidence retains its original reason.
            item['legacy_rules'] = item.get('reason') == 'target_spatially_ambiguous'
            item['retryable_preparation_failure'] = (not item.get('run_id')
                and item.get('reason') in {
                    'preparation:official_conversion_failed:PermissionError',
                    'preparation:official_conversion_failed:9',
                    'preparation:official_conversion_failed:-9',
                    'preparation:official_conversion_failed:137',
                    'preparation:official_conversion_failed:124',
                    'preparation:official_conversion_failed:1',
                    'preparation:official_conversion_failed:4294967295',
                    'preparation:official_conversion_failed:wsl_init_scope_stopping',
                    'preparation:official_conversion_failed:wsl_environment_check_failed',
                    'preparation:official_conversion_failed:broken_pipe',
                    'preparation:official_conversion_timeout'})
            if item["run_id"]:
                if item['run_id'] not in states:
                    raise ValueError('Run not found')
                state = states[item['run_id']]
                if usable_description(state):
                    passed.add(item['target']['target_id'])
                item["run_status"] = state["status"]
                item['manual_review'] = state.get('manual_review') if review_status(state) else None
                from app.quality.sample_review import manually_approved
                item['manual_approved'] = manually_approved(state)
                packet_snapshot = state.get('current_packet', {})
                item['display_target'] = {**item['target'], 'timestamp': packet_snapshot.get('frame_timestamp_micros', item['target'].get('timestamp'))}
                item['source_view'] = packet_snapshot.get('source_view') or item['source_view']
                if packet_snapshot.get('source_view'):
                    item['source_view_status'] = 'confirmed'
                item['source_camera_object_id'] = packet_snapshot.get('source_camera_object_id') or item['source_camera_object_id']
                from app.agent.agent_runtime import VERSION
                item['legacy_rules'] = (packet_snapshot.get('task', {}).get('admission_policy') != 'visible-distinct-v1'
                    or state.get('version') not in {None, VERSION})
                item['method_version'] = state.get('version')
                item["expression"] = state.get("final_expression") or next((a.get("generated", {}).get("expression")
                                                          for a in reversed(state.get("attempts", [])) if a.get("generated")), None)
                item["phase"] = state.get("phase")
                item['admission'] = state.get('admission')
                observed = state.get('target_precheck_observation') or {}
                item['visibility_observation'] = {
                    'visibility': observed.get('visibility'),
                    'visibility_reason': observed.get('visibility_reason'),
                } if observed else None
                if (not item['expression'] and state.get('stop_reason') == 'precheck_target_not_visible'
                        and any(r.get('reason') == 'precheck_no_distinguishing_cues'
                            and (r.get('observation') or {}).get('target_visible') is True
                            for r in state.get('rejected_source_views', []))):
                    # Keep the terminal reason unchanged, but do not describe
                    # an earlier recognized source as invisible after fallback.
                    item['display_reason'] = 'precheck_visible_but_ambiguous'
                item['admission_status'] = ('passed' if state.get('admission', {}).get('passed') else
                    'rejected' if state.get('admission') else 'pending')
                if not item['expression'] and state.get('status') == 'abstained':
                    item['result_label'] = '未生成描述 · 已跳过'
                if state.get('status') == 'needs_evidence':
                    item['result_label'] = '待补证据 · 保留候选'
                if item.get('reason') == 'invalid_visual_candidate_observation':
                    item['result_label'] = '场景观察未通过'
                if (item.get('reason') or '').startswith('precheck_') and state.get('status') != 'needs_evidence':
                    item['result_label'] = '预检查未通过 · 未标注'
                if not item['expression'] and (item.get('reason') or '').startswith('precheck_'):
                    item['result_label'] = '已尝试视角未通过 · 跳过该目标'
                item["model_calls_used"] = state.get("model_calls_used", 0)
                if (state.get("stop_reason") or '').startswith('model_call:') and state.get("inflight"):
                    from app.agent.model_calls import saved_call_diagnostic
                    failure = saved_call_diagnostic(self.runs.path.parent / item["run_id"] / state["inflight"]["folder"])
                    item["provider_diagnosis"] = failure.get("provider_diagnosis") or failure.get('network_diagnosis') or ('output_truncated' if failure.get('finish_reason') == 'length' else None)
                    item['failure_diagnostic'] = failure
                    item["retryable_parameter_failure"] = failure.get("http_status") == 400 and failure.get("provider_diagnosis") == "json_mode_instruction_missing"
                    item['retryable_connection_failure'] = (failure.get('remote_outcome') == 'not_sent'
                        and failure.get('retryable') is True
                        and state.get('stop_reason') == 'model_call:connection_failed'
                        or failure.get('temporary_provider_failure') is True
                        and state.get('stop_reason') == 'model_call:api_failed')
                    if item['retryable_connection_failure']:
                        item['result_label'] = '服务暂时失败 · 可重试'
                    elif failure.get('remote_outcome') == 'unknown':
                        item['result_label'] = '连接中断 · 调用结果待确认'
                from app.agent.model_calls import image_model_error
                if image_model_error(state.get('model', ''), state.get('base_url', '')):
                    # Explain historical failures without rewriting their traces
                    # or attributing a text-only model's guesses to the object.
                    item['provider_diagnosis'] = 'vision_model_text_only'
                    item['result_label'] = '模型不支持图像 · 视觉观察无效'
                item['retryable_visibility_failure'] = (state.get('status') == 'needs_evidence'
                    and state.get('stop_reason') in {'precheck_target_occluded_or_uncertain', 'precheck_source_visibility_conflict'}
                    and not item['expression'] and item['target'].get('eligible_frame_count', 0) > 1)
                if not item["reused"] and item["run_id"] not in run_ids:
                    consumed += item["model_calls_used"]
                    run_ids.add(item["run_id"])
        # Superseded recovery runs still belong to the task's cost ledger.
        with self.runs.connect() as db:
            consumed += db.execute("""SELECT COALESCE(SUM(json_extract(r.snapshot,'$.model_calls_used')),0)
                FROM run_summaries r WHERE json_extract(r.snapshot,'$.batch_id')=? AND NOT EXISTS
                (SELECT 1 FROM batch_items i WHERE i.job_id=? AND i.run_id=r.run_id AND i.reused=0)""",
                (job, job)).fetchone()[0]
            retry_usage = {}
            for item in items:
                if item.get('retryable_connection_failure') or item.get('retryable_visibility_failure'):
                    state = states[item['run_id']]
                    packet = state.get('current_packet', {})
                    key = (packet.get('context'), packet.get('target_object_id'), state.get('provider_scope'))
                    if key not in retry_usage:
                        retry_usage[key] = self.runs._target_usage(db, key[:2], key[2])
        for item in items:
            if item.get('retryable_connection_failure') or item.get('retryable_visibility_failure'):
                state = states[item['run_id']]
                packet = state.get('current_packet', {})
                used = retry_usage[packet.get('context'), packet.get('target_object_id'), state.get('provider_scope')]
                quota = header['spec'].get('max_model_calls')
                if 12 - used < 4 or quota is not None and quota - consumed < 4:
                    item['retryable_connection_failure'] = item['retryable_visibility_failure'] = False
                    if (item.get('reason') or '') == 'model_call:connection_failed':
                        item['result_label'] = '连接失败 · 剩余额度不足'
        selection_progress = {'requested': header['spec'].get('desired_count'),
            'diagnostic_passed': len(passed), 'attempted': sum(attempted_item(i) for i in items),
            'candidate_pool': len(items)}
        continuation = next(({'source_job_id': event['payload']['job_id'],
                              'inherited_passed': event['payload'].get('inherited_passed',0)}
                             for event in events if event['kind']=='continued_from'),None)
        from app.infrastructure.batch_timing import summarize_timing
        queue = next((row for row in self.queue() if row['job_id']==job), {})
        measured_at = time.time()
        timing, item_timings = summarize_timing(events, status=header['status'], executor_active=active, now=measured_at)
        for item in items:
            item['timing'] = item_timings.get(item['ordinal'], {'active_seconds': None, 'running': False, 'complete': False})
        return {**header, "items": items, "events": events, "counts": dict(Counter(
                    'manual_approved' if i.get('manual_approved') else i['manual_review']['verdict'] if (i.get('manual_review') or {}).get('verdict') in {'rejected','discarded'}
                    else i['status'] for i in items)),
                'continuation': continuation,
                'queue_position':queue.get('position'), 'queue_paused':queue.get('queue_paused',False),
                'timing': timing, 'measured_at': measured_at,
                'selection_progress': selection_progress,
                "model_calls_reserved": consumed, "training_eligible": False,
                "independent_quality": None, "billed_cost": None}

    def complete_selection(self, job, reason):
        """Settle unused candidates without labelling them as failed attempts."""
        with self.runs.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if reason != 'description_goal_reached' and db.execute("""SELECT 1 FROM batch_items
                    WHERE job_id=? AND status='pending' AND json_type(target,'$.manual_review_retry')='object' LIMIT 1""",(job,)).fetchone():
                return  # A late human retry must survive coordinator settlement.
            db.execute("UPDATE batch_items SET status='not_needed',reason=? WHERE job_id=? AND status='pending' AND run_id IS NULL AND packet_id IS NULL",
                       (reason, job))
            db.execute("UPDATE batches SET status='completed',reason=? WHERE job_id=?", (reason, job))
            self.event(db, job, 'batch_completed', {'reason': reason, 'training_eligible': False})

    @staticmethod
    def reserved_calls(db, job):
        # Includes retired owned runs and non-reused linked runs, exactly once.
        return db.execute("""SELECT COALESCE(SUM(json_extract(snapshot,'$.model_calls_used')),0)
            FROM run_live_summaries WHERE json_extract(snapshot,'$.batch_id')=? OR run_id IN
            (SELECT run_id FROM batch_items WHERE job_id=? AND reused=0)""", (job, job)).fetchone()[0]

    def execution_detail(self, job):
        """Frozen queue and ledger only; no UI outcome or event rendering."""
        with self.runs.connect() as db:
            row = db.execute('SELECT spec FROM batches WHERE job_id=?', (job,)).fetchone()
            if row is None:
                raise ValueError('Batch not found')
            specification = json.loads(row['spec'])
            items = [{**dict(r), 'target':json.loads(r['target'])} for r in db.execute(
                'SELECT * FROM batch_items WHERE job_id=? ORDER BY ordinal', (job,))]
            calls = self.reserved_calls(db, job)
        if specification.get('desired_count') is not None:
            items.sort(key=lambda item:item['target']['queue_position'])
        return {'spec':specification, 'items':items, 'model_calls_reserved':calls}

    def pending_review_retries(self, job):
        with self.runs.connect() as db:
            rows = db.execute("""SELECT * FROM batch_items WHERE job_id=? AND status='pending'
                AND json_type(target,'$.manual_review_retry')='object' ORDER BY ordinal""",(job,)).fetchall()
        return [{**dict(row),'target':json.loads(row['target'])} for row in rows]

    def start_detail(self, job):
        """An explicit pending review needs scheduling facts, not old UI results."""
        if not self.pending_review_retries(job):
            return self.detail(job, compact=True)
        with self.runs.connect() as db:
            row = db.execute('SELECT * FROM batches WHERE job_id=?', (job,)).fetchone()
            if row is None:
                raise ValueError('Batch not found')
            header = dict(row)
            lease = db.execute('SELECT * FROM executor WHERE singleton=1').fetchone()
        detail = self.execution_detail(job)
        progress = self.execution_progress(job)
        active = bool(lease and lease['owner']==header.pop('owner') and lease['lease_until']>time.time())
        return {**header, **detail, 'executor_active':active,
            'selection_progress':{'diagnostic_passed':progress['passed']}}

    def execution_progress(self, job):
        """Small coordinator view; avoid rendering every sample on each dispatch."""
        with self.runs.connect() as db:
            rows = db.execute("""SELECT json_extract(i.target,'$.target_id') AS target_id,r.snapshot
                FROM batch_items i JOIN run_live_summaries r ON r.run_id=i.run_id WHERE i.job_id=?
                AND (json_extract(r.snapshot,'$.status') IN ('calibration_hold','auto_pass')
                AND json_extract(r.snapshot,'$.diagnostic_pass')=1
                AND length(json_extract(r.snapshot,'$.final_expression'))>0
                OR json_extract(r.snapshot,'$.manual_review.approval_policy')='localized-human-approval-v1')""", (job,))
            passed = len({r['target_id'] for r in rows if usable_description(json.loads(r['snapshot']))})
            calls = self.reserved_calls(db, job)
        return {'passed': passed, 'calls': calls}

    def update(self, job, kind, payload, *, status=None, reason=None, ordinal=None, item_status=None,
               packet_id=None, run_id=None, reused=False):
        with self.runs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if status:
                db.execute("UPDATE batches SET status=?,reason=? WHERE job_id=?", (status, reason, job))
            if ordinal is not None:
                db.execute("UPDATE batch_items SET status=?,packet_id=COALESCE(?,packet_id),run_id=COALESCE(?,run_id),reason=?,reused=? WHERE job_id=? AND ordinal=?",
                           (item_status, packet_id, run_id, reason, int(reused), job, ordinal))
            self.event(db, job, kind, payload)

    def command(self, job):
        with self.runs.connect() as db:
            return db.execute("SELECT control FROM batches WHERE job_id=?", (job,)).fetchone()["control"]

    def attach_child(self, job, ordinal, initial):
        """Child snapshot and queue link commit together, before any external call."""
        from app.quality.sample_review import validate_sample_identity
        # Freeze the task's original identity and check the prepared input before
        # taking the writer. The full JSON serialization stays outside it too.
        with self.runs.connect() as db:
            item = db.execute('SELECT target,packet_id FROM batch_items WHERE job_id=? AND ordinal=?',(job,ordinal)).fetchone()
        if item is None:
            raise ValueError('Task target not found')
        target = json.loads(item['target'])
        initial['binding_target'] = {k:target[k] for k in ('context','object_id','timestamp','category') if k in target}
        initial['binding_target']['alternative_selections'] = [{'timestamp':o['timestamp']} for o in target.get('alternative_selections',[])]
        authorization = target.get('retry_authorization')
        if authorization:
            initial['retry_authorization'] = authorization
            if authorization.get('source_run_id'):
                initial['parent_run_id'] = authorization['source_run_id']
        validate_sample_identity(initial['binding_target'],initial,ordinal=ordinal,
            item_packet_id=item['packet_id'] if item['packet_id'] is not None else initial['packet_id'])
        snapshot = json.dumps(initial,allow_nan=False)
        with self.runs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute('SELECT target,packet_id FROM batch_items WHERE job_id=? AND ordinal=?',(job,ordinal)).fetchone()
            if current is None or tuple(current) != tuple(item):
                raise RunBusy('Task target changed before linking its run')
            if db.execute("SELECT 1 FROM runs WHERE run_id=?", (initial["run_id"],)).fetchone():
                raise RunBusy("Child run was concurrently created")
            if authorization:
                available = db.execute("""SELECT 1 FROM batch_items WHERE
                    json_extract(target,'$.retry_authorization.token')=?
                    AND json_extract(target,'$.retry_authorization.consumed_run_id') IS NULL
                    AND run_id IS NULL AND status IN ('pending','preparing') LIMIT 1""",
                    (authorization['token'],)).fetchone()
                if not available or authorization.get('consumed_run_id'):
                    raise RunBusy('该目标本轮重新入池授权已被使用，不能重复执行。')
                db.execute("""UPDATE batch_items SET target=json_set(target,
                    '$.retry_authorization.consumed_run_id',?) WHERE
                    json_extract(target,'$.retry_authorization.token')=? AND
                    json_extract(target,'$.retry_authorization.consumed_run_id') IS NULL""",
                    (initial['run_id'],authorization['token']))
            db.execute("INSERT INTO runs(run_id,snapshot) VALUES(?,?)", (initial["run_id"], snapshot))
            db.execute("UPDATE batch_items SET status='running',run_id=?,packet_id=? WHERE job_id=? AND ordinal=?",
                       (initial["run_id"], initial['packet_id'],job, ordinal))
            self.event(db, job, "child_checkpoint_linked", {"ordinal": ordinal, "run_id": initial["run_id"]})


class BatchService:
    def __init__(self, runner, preparer=None, catalog_loader=load_catalog, registrar=register_packet):
        self.loaded_method_sha256 = method_fingerprint()
        self.runner, self.catalog_loader = runner, catalog_loader
        self.store = BatchStore(runner.store)
        self.preparer, self.registrar = preparer or TargetPreparer(), registrar
        self.closing = False

    def ensure_loaded_code(self):
        loaded = getattr(self, 'loaded_method_sha256', None)
        if loaded is not None and loaded != method_fingerprint():
            raise ValueError('后端运行期间代码已更新，请先重启后端服务，再继续标注；已有样本和累计额度保持不变。')

    def create(self, spec, *, continuation=None):
        self.ensure_loaded_code()
        with self.runner.store.maintenance():
            catalog = self.catalog_loader(spec.catalog_id)
            if spec.review_retry:
                raise ValueError('人工返工请在原任务的样本中重试，不能新建任务。')
            job, reused = self.store.create(spec, catalog, continuation)
            diagnostic_event(self.runner.root, "batch_opened" if reused else "batch_created", job_id=job,
                             catalog_id=spec.catalog_id, prepare_only=spec.prepare_only,
                             target_count=len(self.store.detail(job)["items"]))
            return job, reused

    def retry_review(self, job, ordinal, reference, revision, *, retry_failed_run_id=None):
        """Requeue one settled item in place; keep every original run and bill."""
        self.ensure_loaded_code()
        source, review = retry_source(self.runner.store, reference, require_current=True)
        with self.runner.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM batch_items WHERE job_id=? AND ordinal=?',(job,ordinal)).fetchone()
            if row is None or row['run_id'] != (retry_failed_run_id or reference.source_run_id):
                raise ValueError('样本版本已变化，请刷新后重试。')
            if retry_failed_run_id:
                from app.quality.sample_review import empty_rework_source
                previous, _ = empty_rework_source(db,dict(row),retry_failed_run_id)
                if previous['run_id'] != reference.source_run_id or review.get('retry_failed_run_id') != retry_failed_run_id:
                    raise ValueError('反馈与本轮失败返工不匹配，请刷新后重新提交。')
            if (source['status'] in {'running','paused','stopped'} or source.get('inflight')
                    or source.get('stop_reason') in {'interrupted_response_outcome_unknown','model_call:interrupted_response'}):
                raise ValueError('该样本尚未结束或有结果未确定的请求，不能重放。')
            target = json.loads(row['target'])
            spent = self.runner.store._target_usage(db,
                (target['context'],target['object_id']),source['provider_scope'])
            authorization = {'token':uuid4().hex[:24], 'kind':'manual_review_retry',
                'source_job_id':job, 'source_run_id':retry_failed_run_id or reference.source_run_id,
                'review_id':reference.review_id, 'calls_before':spent, 'additional_calls':12,
                'call_limit':spent+12, 'authorized_at':time.time(), 'provider_scope':source['provider_scope']}
            if retry_failed_run_id:
                authorization.update(retry_failed_run_id=retry_failed_run_id,feedback_source_run_id=reference.source_run_id)
            target.update(retry_authorization=authorization, execution_attempt=authorization['token'],
                manual_review_retry=reference.model_dump())
            db.execute("""UPDATE batch_items SET target=?,status='pending',reason=NULL,
                run_id=NULL,packet_id=NULL,reused=0 WHERE job_id=? AND ordinal=?""",
                (json.dumps(target,allow_nan=False),job,ordinal))
            # The original task identity, specification, goal and creation time stay.
            db.execute("""UPDATE batches SET status=CASE WHEN status IN ('completed','blocked') THEN 'ready'
                ELSE status END,reason=NULL WHERE job_id=?""",(job,))
            self.store.event(db,job,'manual_review_retry_requested', {**authorization,
                'ordinal':ordinal, 'source_revision':revision, 'notes':review['notes']})
        return {'job_id':job, 'ordinal':ordinal, 'reused_job':True}

    def deletion_preview(self, job_ids, progress=None):
        from app.infrastructure.task_cleanup import TaskCleaner
        with self.runner.lock:
            if self.runner.active_batch in job_ids:
                raise RunBusy('所选任务正在执行，请先暂停该任务并等待保存后再删除。')
        return TaskCleaner(self.runner.store, ROOT, progress).preview(job_ids)

    def delete(self, job_ids, token, return_to_pool=None, progress=None):
        from app.infrastructure.task_cleanup import TaskCleaner
        with self.runner.lock:
            if self.runner.active_batch in job_ids:
                raise RunBusy('所选任务正在执行，请先暂停该任务并等待保存后再删除。')
        return TaskCleaner(self.runner.store, ROOT, progress).delete(job_ids, token, return_to_pool)

    def recover_deletions(self):
        from app.infrastructure.task_cleanup import TaskCleaner, run_parents
        cleaner = TaskCleaner(self.runner.store, ROOT)
        if (self.runner.root / '.task-delete').exists():
            from app.infrastructure.diagnostics import diagnostic_maintenance
            with self.runner.store.maintenance(), diagnostic_maintenance(self.runner.root):
                if cleaner.recover():
                    raise RuntimeError('部分任务文件仍被占用，无法完成删除恢复；请关闭占用文件的程序后重启服务。')
        with self.runner.store.connect() as db:
            no_jobs = db.execute('SELECT 1 FROM batches LIMIT 1').fetchone() is None
        if no_jobs:
            # Migrate leftovers from older deletion logic. Only descendants
            # whose source has disappeared qualify; standalone runs stay intact.
            states = {s['run_id']: s for s in self.runner.store.all_runs()}
            if any(parent not in states for s in states.values() for parent in run_parents(s)):
                preview = cleaner.preview([])
                if preview['summary']['runs'] and cleaner.delete([], preview['token'])['cleanup_pending']:
                    raise RuntimeError('孤立任务记录清理尚未完成，请关闭占用文件的程序后重启服务。')

    def recover_running(self):
        """Resume abandoned running jobs after lease expiry; respect explicit controls."""
        worker = self.runner
        if worker.closed or worker.active or worker.active_batch:
            return
        with worker.store.connect() as db:
            if db.execute('SELECT 1 FROM executor WHERE lease_until>?', (time.time(),)).fetchone():
                return
            jobs = [r['job_id'] for r in db.execute(
                "SELECT job_id FROM batches WHERE status='running' AND control IS NULL ORDER BY created_at")]
        for job in jobs:
            try:
                receipt = self.start(job, automatic=True)
            except RunBusy:
                return
            except (ValueError, OSError) as exc:
                # A new code/catalog/provider cannot execute an old frozen job.
                self.store.update(job, 'restart_recovery_paused', {'error_type': type(exc).__name__},
                    status='paused', reason='restart_configuration_or_input_changed')
                self.store.suspend_queue()
                continue
            if receipt:
                self.store.update(job, 'restart_recovery_started', {'scope': 'expired lease; original checkpoints and budgets'})
                return
        self.dispatch_queue()

    def dispatch_queue(self):
        """Only explicitly started/enqueued jobs may run when the slot is idle."""
        worker = self.runner
        if self.closing or worker.closed or worker.active or worker.active_batch:
            return
        queue = self.store.queue()
        if not queue or queue[0]['queue_paused']:
            return
        job = queue[0]['job_id']
        try:
            return self.start(job, queued_dispatch=True)
        except RunBusy:
            return  # Another owner still holds the shared execution lease.
        except (ValueError, OSError) as exc:
            self.store.update(job, 'queue_start_paused', {'error_type':type(exc).__name__},
                status='paused', reason='queue_configuration_or_input_changed')
            self.store.suspend_queue()
            diagnostic_event(worker.root, 'queue_start_paused', job_id=job, error_type=type(exc).__name__)

    def prepare_resume(self, job):
        from app.agent.batch_recovery import prepare_recovery, cleanup_temporary
        with self.runner.store.maintenance(), self.runner.lock:
            self.ensure_loaded_code()
            if self.runner.closed or self.runner.active or self.runner.active_batch:
                raise RunBusy('另有任务正在执行，请等待执行器退出后再继续。')
            receipt = prepare_recovery(self,job)
            cleanup_temporary(self,job)
            return receipt

    def start(self, job, *, automatic=False, queued_dispatch=False):
        worker = self.runner
        with worker.store.maintenance(), worker.lock:
            self.ensure_loaded_code()
            if worker.closed or self.closing:
                raise RunBusy('Executor is closing')
            detail = self.store.start_detail(job)
            queue = self.store.queue()
            if not automatic and not queued_dispatch and (worker.active_batch == job or detail['executor_active']):
                return {'job_id':job,'reused_result':True,'queued':False}
            with worker.store.connect() as db:
                leased = bool(db.execute('SELECT 1 FROM executor WHERE lease_until>?', (time.time(),)).fetchone()
                    or db.execute('SELECT 1 FROM runs WHERE owner IS NOT NULL AND lease_until>?', (time.time(),)).fetchone())
            occupied = bool(worker.active or worker.active_batch or leased)
            if queued_dispatch:
                if not queue or queue[0]['job_id']!=job or queue[0]['queue_paused'] or detail['status']!='queued':
                    return
                intent = queue[0]['intent']
                if occupied:
                    raise RunBusy('Another executor is active')
                provider = None if detail['spec']['prepare_only'] else {
                    'model':os.getenv('DASHSCOPE_MODEL','qwen3.8-flash'),
                    'base_url':os.getenv('DASHSCOPE_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')}
                catalog = self.catalog_loader(detail['spec']['catalog_id'])
                if (intent['method_sha256']!=method_fingerprint() or intent['provider']!=provider
                        or intent['catalog_sha256']!=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest()):
                    raise ValueError('Queued configuration changed; confirm start again')
            elif not automatic and (occupied or queue and queue[0]['job_id']!=job
                    and not (queue[0]['queue_paused'] and detail['status'] in {'paused','stopped','blocked'})):
                if detail['status'] not in {'ready','paused','stopped','blocked','queued'}:
                    raise ValueError('Task cannot be enqueued in its current state')
                catalog = self.catalog_loader(detail['spec']['catalog_id'])
                intent = {'method_sha256':method_fingerprint(),
                    'catalog_sha256':hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest(),
                    'provider':None if detail['spec']['prepare_only'] else {
                        'model':os.getenv('DASHSCOPE_MODEL','qwen3.8-flash'),
                        'base_url':os.getenv('DASHSCOPE_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')}}
                with worker.store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    if detail['status']!='queued':
                        db.execute("UPDATE batches SET status='queued',control=NULL,owner=NULL,reason=NULL WHERE job_id=?", (job,))
                        self.store.event(db,job,'batch_enqueued',intent)
                    else:
                        # Refresh consent without changing the original FIFO position.
                        row = next(row for row in queue if row['job_id']==job)
                        db.execute('UPDATE batch_events SET payload=? WHERE seq=?',(json.dumps(intent),row['seq']))
                    db.execute('UPDATE batch_queue_control SET paused=0 WHERE singleton=1')
                position = next(row['position'] for row in self.store.queue() if row['job_id']==job)
                return {'job_id':job,'reused_result':detail['status']=='queued','queued':True,'queue_position':position}
            elif occupied:
                raise RunBusy('Another annotation or preparation is executing')
            if automatic and (detail['status'] != 'running' or self.store.command(job)):
                return
            goal = detail['spec'].get('desired_count')
            reached = goal is not None and detail['selection_progress']['diagnostic_passed'] >= goal
            if reached:
                if detail['status'] != 'completed':
                    self.store.complete_selection(job,'description_goal_reached')
                return {"job_id": job, "reused_result": True}
            resumable = any(item['status'] in {'pending','preparing','running','budget_skipped'}
                or goal is not None and not item.get('run_id') and item['status']=='not_needed'
                    and item.get('reason')=='description_goal_reached'
                or any(item.get(key) for key in ('retryable_connection_failure','retryable_parameter_failure',
                    'retryable_preparation_failure','retryable_visibility_failure')) for item in detail['items'])
            if detail['status'] == 'completed' and (automatic or not resumable):
                return {"job_id": job, "reused_result": True}
            if not automatic:
                from app.agent.batch_recovery import prepare_recovery, cleanup_temporary
                requirements = self.resume_requirements(detail)
                if ((detail['status'] != 'ready' and not self.store.pending_review_retries(job))
                        or requirements and requirements['mode'] == 'recover'):
                    prepare_recovery(self,job)
                    detail = self.store.start_detail(job)
                cleanup_temporary(self,job)
            catalog = self.catalog_loader(detail["spec"]["catalog_id"])
            if hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest() != detail["catalog_sha256"]:
                raise ValueError("Catalog changed; recover the task before resuming")
            if detail["method_sha256"] != method_fingerprint():
                raise ValueError("Code/configuration changed; manual recovery is required")
            if detail["spec"].get("provider") is not None and detail["spec"]["provider"] != {
                    "model": os.getenv("DASHSCOPE_MODEL", "qwen3.8-flash"),
                    "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")}:
                raise ValueError("Model/provider configuration changed; manual recovery is required")
            with worker.store.connect() as db:
                if db.execute('SELECT 1 FROM runs WHERE owner IS NOT NULL AND lease_until>? LIMIT 1',
                        (time.time(),)).fetchone():
                    raise RunBusy("A single-target run remains active")
            owner = uuid4().hex
            worker.store.acquire_executor(owner)
            try:
                with worker.store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    current = db.execute('SELECT status,control FROM batches WHERE job_id=?', (job,)).fetchone()
                    if automatic and (current['status'] != 'running' or current['control']):
                        raise RunBusy('Batch was controlled during recovery')
                    db.execute("UPDATE batches SET status='running',control=NULL,owner=?,reason=NULL WHERE job_id=?", (owner, job))
                    db.execute('UPDATE batch_queue_control SET paused=0 WHERE singleton=1')
                    self.store.event(db, job, "batch_started_or_resumed", {'automatic_recovery': automatic})
                worker.active_batch = job
                diagnostic_event(worker.root, "batch_started", job_id=job, model_calls_reserved=detail["model_calls_reserved"])
                worker.pool.submit(self._execute, job, owner, catalog)
            except Exception:
                worker.active_batch = None
                worker.store.release_executor(owner)
                self.store.update(job,'executor_start_failed',{},status='paused',reason='executor_start_failed')
                self.store.suspend_queue()
                raise
        return {"job_id": job, "reused_result": False, 'queued':False}

    def display_catalog(self, catalog_id):
        return self.catalog_loader(catalog_id, compact=True) if self.catalog_loader is load_catalog else self.catalog_loader(catalog_id)

    def resume_requirements(self, detail):
        """Explain frozen-snapshot mismatches without dispatching any work."""
        if detail['executor_active'] or detail['status'] not in {'ready','paused','stopped','running','blocked','completed','queued'}:
            return None
        try:
            self.ensure_loaded_code()
        except ValueError as exc:
            return {'mode':'unavailable', 'reason':'backend_restart_required', 'message':str(exc)}
        if detail['method_sha256'] != method_fingerprint():
            return {'mode': 'recover', 'reason': 'code_changed',
                    'message': '系统已更新。继续时将保留已完成标注，清理中断进度，并在原任务中处理剩余数量。'}
        provider = detail['spec'].get('provider')
        if provider is not None and provider != {
                'model': os.getenv('DASHSCOPE_MODEL', 'qwen3.8-flash'),
                'base_url': os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}:
            return {'mode': 'recover', 'reason': 'provider_changed',
                    'message': '模型配置已改变。继续时将保留已完成标注，按当前配置在原任务中继续。'}
        try:
            catalog_id = detail['spec']['catalog_id']
            if self.catalog_loader is load_catalog:
                digest = catalog_digest(catalog_id)
            else:
                digest = hashlib.sha256(json.dumps(self.catalog_loader(catalog_id), sort_keys=True).encode()).hexdigest()
        except FileNotFoundError:
            return {'mode': 'unavailable', 'message': '原任务候选目录已不存在，请恢复目录后再继续。已有标注记录仍保留。'}
        if digest != detail['catalog_sha256']:
            return {'mode': 'recover', 'reason': 'catalog_changed',
                    'message': '候选目录已改变。继续时将保留已完成标注，重新核对待处理目标后在原任务中继续。'}
        return {'mode': 'resume', 'message': '可从原任务保存的检查点继续。'}

    def control(self, job, command):
        if command == "resume":
            return self.start(job)
        if command not in {"pause", "stop"}:
            raise ValueError("Unknown batch command")
        # Serialize cancellation with dispatch, so a queued job cannot start
        # after cancellation has been acknowledged.
        with self.runner.lock:
            return self._control(job, command)

    def _control(self, job, command):
        # Register intent transactionally, then forward to a linked call.
        # The worker rechecks between preparation and every child invocation.
        with self.runner.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,control FROM batches WHERE job_id=?", (job,)).fetchone()
            if row is None or row["status"] in {"completed", "blocked"}:
                raise ValueError("Batch is not controllable")
            if row["control"] == "stop" and command == "pause":
                raise ValueError("Stop already requested")
            db.execute("UPDATE batches SET control=? WHERE job_id=?", (command, job))
            if row['status']=='running':
                db.execute('UPDATE batch_queue_control SET paused=1 WHERE singleton=1')
            self.store.event(db, job, "batch_control_requested", {"command": command})
            current = db.execute("SELECT run_id FROM batch_items WHERE job_id=? AND status='running'", (job,)).fetchall()
        for item in current:
            if not item['run_id']:
                continue
            try:
                self.runner.store.request_control(item["run_id"], command)
            except ValueError:
                pass  # It settled between request and forwarding; batch intent persists.
        detail = self.store.detail(job)
        diagnostic_event(self.runner.root, "batch_control_requested", job_id=job, command=command,
                         executor_active=detail["executor_active"])
        if not detail["executor_active"]:
            self.store.update(job, "batch_control_applied", {"command": command},
                              status="paused" if command == "pause" else "stopped")
        return {"job_id": job, "command": command, "acknowledged": True}

    def _execute(self, job, owner, catalog):
        from app.agent.batch_execution import BatchExecution
        try:
            BatchExecution(self, job, owner, catalog).run()
        finally:
            with self.runner.store.connect() as db:
                row = db.execute('SELECT status FROM batches WHERE job_id=?',(job,)).fetchone()
            if row and row['status'] in {'completed','ready','running'} and self.store.pending_review_retries(job) and not self.store.command(job):
                self.start(job)
            elif row and row['status']=='completed':
                self.dispatch_queue()
            else:
                self.store.suspend_queue()

    def close(self):
        self.closing = True
        if self.runner.active_batch:
            try:
                self.control(self.runner.active_batch, "pause")
            except ValueError:
                # The final status may commit just before executor teardown.
                with self.runner.store.connect() as db:
                    row = db.execute('SELECT status FROM batches WHERE job_id=?',(self.runner.active_batch,)).fetchone()
                if row and row['status'] not in {'completed','blocked'}:
                    raise
