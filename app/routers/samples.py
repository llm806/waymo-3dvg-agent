"""Task-bound sample viewing and immediate manual localization feedback.

Manual feedback is a diagnostic, never an independent blind quality certificate.
Records live in batch_events so task deletion removes them with the task.
"""
import hashlib
import json
import random
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.infrastructure.artifacts import checked_asset
from app.schemas.contracts import EvidencePacket
from app.datasets.evidence import ROOT
from app.datasets.inspection import associate_camera_boxes, corners, sample_indices
from app.datasets.correspondence import apply_recorded_point_links, verified_native_objects, official_projection_boxes
from app.datasets.camera_view import camera_views
from app.routers.dependencies import Id, mutation_origin
from app.datasets.training import read_scene
from app.schemas.training_contracts import SceneManifest
from app.datasets.waymo import CAMERAS, WaymoReader
from app.quality.sample_review import REVIEW_EVENT, ReviewRetry, review_status, sample_revision, text_hash, expression as sample_expression, validate_sample_identity
from app.infrastructure.storage import RunBusy
from app.agent.selection import passed_description, usable_description
from app.quality.sample_review import manually_approved, approval_binding, MANUAL_APPROVAL_POLICY, previous_rework_display, empty_rework_source

router = APIRouter(prefix='/api/batches', tags=['task samples'])
REVIEW_KIND = 'sample_localization_confirmed'
# Bump when presentation/binding rules change. Run/item/source-event triggers
# invalidate individual durable rows even for edits that leave revision intact.
SAMPLE_PRESENTATION_VERSION = '4'


def root_for(request):
    return getattr(request.app.state, 'sample_root', ROOT)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sample(store, job_id, ordinal, job=None, *, allow_previous=False, retry_failed_run_id=None):
    if job is None:
        # Scene/image/review access needs one sample, never the whole batch.
        with store.runs.connect() as db:
            db.execute('BEGIN')
            header = db.execute('SELECT job_id,spec,status,created_at FROM batches WHERE job_id=?', (job_id,)).fetchone()
            if header is None:
                raise ValueError('Batch not found')
            raw = db.execute('''SELECT i.*,r.snapshot FROM batch_items i LEFT JOIN runs r ON r.run_id=i.run_id
                WHERE i.job_id=? AND i.ordinal=?''', (job_id, ordinal)).fetchone()
            if raw is None:
                raise ValueError('该任务中没有这条已生成描述的样本。')
            state = json.loads(raw['snapshot']) if raw['snapshot'] else {}
            item = {k: raw[k] for k in raw.keys() if k != 'snapshot'}
            item.update(target=json.loads(item['target']), expression=sample_expression(state))
            if retry_failed_run_id:
                item, state = empty_rework_source(db,item,retry_failed_run_id)
            elif allow_previous and not item['expression']:
                previous = previous_rework_display(db, item)
                if previous:
                    item, state = previous
                    # Viewing the input never authorizes reviewing or exporting it.
                    state = {**state, 'status':'running'}
            changes = [{'kind': 'source_view_reselected', 'payload': json.loads(r['payload'])} for r in db.execute(
                "SELECT payload FROM events WHERE run_id=? AND kind='source_view_reselected' ORDER BY seq", (item['run_id'],))]
            changes.extend({'kind':'preparation_source_reselected','payload':json.loads(r['payload'])} for r in db.execute(
                "SELECT payload FROM batch_events WHERE job_id=? AND kind='preparation_source_reselected' AND json_extract(payload,'$.ordinal')=? ORDER BY seq",
                (job_id,ordinal)))
            events = [{'kind': REVIEW_KIND, 'payload': json.loads(r['payload'])} for r in db.execute(
                "SELECT payload FROM batch_events WHERE job_id=? AND kind=? AND json_extract(payload,'$.ordinal')=? ORDER BY seq",
                (job_id, REVIEW_KIND, ordinal))]
        if not item.get('expression'):
            raise ValueError('该任务中没有这条已生成描述的样本。')
        job = {**dict(header), 'spec': json.loads(header['spec']), 'items': [item], 'events': events}
        packet, revision = bound_sample(job_id, item, state, changes)
        return job, item, state, packet, revision
    item = next((i for i in job['items'] if i['ordinal'] == ordinal), None)
    if item is None or not item.get('run_id') or not item.get('expression'):
        raise ValueError('该任务中没有这条已生成描述的样本。')
    state = store.runs.snapshot(item['run_id'])
    changes = [*store.runs.events(item['run_id']), *(e for e in job.get('events', [])
        if e['kind'] == 'preparation_source_reselected' and e['payload'].get('ordinal') == ordinal)]
    packet, revision = bound_sample(job_id, item, state, changes)
    return job, item, state, packet, revision


def bound_sample(job_id, item, state, events):
    packet = EvidencePacket.model_validate(state['current_packet'])
    validate_sample_identity(item['target'],state,events,ordinal=item['ordinal'],item_packet_id=item['packet_id'])
    expression = state.get('final_expression') or next((a['generated']['expression'] for a in reversed(state.get('attempts', [])) if a.get('generated')), None)
    if not expression or expression != item['expression']:
        raise ValueError('描述版本已变化，请重新打开样本。')
    revision = sample_revision(job_id, item['ordinal'], item['run_id'], expression, packet)
    return packet, revision


def mapping_for(packet):
    return {f'C{i+1:03d}': oid for i, oid in enumerate(sorted(o.object_id for o in packet.candidates))}


def recorded(job, ordinal, revision):
    return [e['payload'] for e in job['events'] if e['kind'] == REVIEW_KIND
            and e['payload'].get('ordinal') == ordinal and e['payload'].get('revision') == revision]


def native_labels(packet, root, scene, arrays):
    """Read exact native labels and recheck every saved target image link."""
    reader = WaymoReader(root / 'waymo_trainset')
    components = packet.provenance.get('components', {})
    required = ['lidar_box', 'camera_box', 'camera_to_lidar_box_association']
    if packet.task.get('image_identity_policy') == 'official-projection-v1' or 'projected_lidar_box' in components:
        required.append('projected_lidar_box')
    for name in required:
        expected = components.get(name)
        if not expected:
            raise ValueError('样本缺少原生真值框来源记录：' + name)
        actual = reader.provenance(name, packet.context)
        if (actual['file'], actual['sha256']) != (expected['file'], expected['sha256']):
            raise ValueError('原生真值框来源已变化：' + name)
    by_id = verified_native_objects(reader, packet)
    boxes = associate_camera_boxes(
        reader.rows('camera_box', packet.context, timestamp=packet.frame_timestamp_micros),
        reader.rows('camera_to_lidar_box_association', packet.context, timestamp=packet.frame_timestamp_micros),
        packet.target_object_id, packet.target.category)
    return by_id, apply_recorded_point_links(boxes, packet, scene=scene, arrays=arrays, objects=list(by_id.values()),
        projections=reader.rows('projected_lidar_box', packet.context, timestamp=packet.frame_timestamp_micros) if 'projected_lidar_box' in components else [])


def manifest(packet, root):
    if not packet.scene:
        raise ValueError('该样本没有保存对应的点云与图像场景。')
    path = checked_asset(root, packet.scene.manifest_path, packet.scene.manifest_sha256)
    scene = SceneManifest.model_validate_json(path.read_text(encoding='utf-8'))
    if (scene.context, scene.frame_timestamp_micros) != (packet.context, packet.frame_timestamp_micros):
        raise ValueError('场景与文本样本不属于同一采集帧。')
    return path, scene


def presentation_row(job_id, item, state, changes):
    """The same full binding validation used by the scene, computed once per edit."""
    from app.agent.agent_runtime import VERSION
    item = {**item, 'target': json.loads(item['target']), 'expression': sample_expression(state)}
    if not item['expression']:
        return None
    revision, decision, can_review, automatic_passed = None, None, False, False
    try:
        _, revision = bound_sample(job_id, item, state, changes)
        decision = state.get('manual_review') if review_status(state) else None
        can_review = state['status'] not in {'running', 'paused', 'stopped'}
        automatic_passed = passed_description(state)
    except (ValueError, KeyError):
        pass
    packet = state.get('current_packet', {})
    return {'ordinal': item['ordinal'], 'category': item['target']['category'],
        'source_view': packet.get('source_view') or item['target'].get('source_view'),
        'legacy_rules': (packet.get('task', {}).get('admission_policy') != 'visible-distinct-v1'
            or state.get('version') not in {None, VERSION}),
        'expression': item['expression'], 'run_id': item['run_id'], 'packet_id': item['packet_id'],
        'status': item['status'], 'revision': revision, 'can_review': can_review,
        'manual_review': decision, 'automatic_passed': automatic_passed,
        'manual_approved':bool(revision and manually_approved(state)), 'usable':bool(revision and usable_description(state))}


@router.get('/{job_id}/samples')
def samples(job_id: Id, request: Request):
    try:
        from app.agent.agent_runtime import VERSION
        store = request.app.state.batches.store
        cache_format = f'{SAMPLE_PRESENTATION_VERSION}:{VERSION}'
        result, pending = [], []
        # One consistent read. A hit never selects the snapshot column; misses
        # stream individually so cold reads do not retain hundreds of MB.
        with store.runs.connect() as db:
            db.execute('BEGIN')
            header = db.execute('SELECT created_at,status,spec FROM batches WHERE job_id=?', (job_id,)).fetchone()
            if header is None:
                raise ValueError('Batch not found')
            items = [dict(r) for r in db.execute('''SELECT i.*,c.generation,c.format,c.presentation
                FROM batch_items i JOIN sample_presentations c
                ON c.job_id=i.job_id AND c.ordinal=i.ordinal WHERE i.job_id=? ORDER BY i.ordinal''', (job_id,))]
            if json.loads(header['spec']).get('desired_count') is not None:
                items.sort(key=lambda i: json.loads(i['target'])['queue_position'])
            changes = {}
            for r in db.execute('''SELECT e.run_id,e.payload FROM events e WHERE e.kind='source_view_reselected'
                AND e.run_id IN (SELECT run_id FROM batch_items WHERE job_id=?) ORDER BY e.seq''', (job_id,)):
                changes.setdefault(r['run_id'], []).append({'kind': 'source_view_reselected', 'payload': json.loads(r['payload'])})
            preparations = {}
            for r in db.execute("SELECT payload FROM batch_events WHERE job_id=? AND kind='preparation_source_reselected' ORDER BY seq",(job_id,)):
                receipt = json.loads(r['payload'])
                preparations.setdefault(receipt.get('ordinal'), []).append({'kind':'preparation_source_reselected','payload':receipt})
            confirmations = {(p['ordinal'], p['revision']) for r in db.execute(
                'SELECT payload FROM batch_events WHERE job_id=? AND kind=?', (job_id, REVIEW_KIND))
                if (p := json.loads(r['payload'])).get('ordinal') and p.get('revision')}
            by_ordinal = {i['ordinal']: i for i in items}
            built = {}
            # Batch the misses too; do not issue one snapshot query per sample.
            misses = db.execute('''SELECT i.ordinal,r.snapshot FROM batch_items i
                JOIN sample_presentations c ON c.job_id=i.job_id AND c.ordinal=i.ordinal
                LEFT JOIN runs r ON r.run_id=i.run_id WHERE i.job_id=? AND i.run_id IS NOT NULL
                AND (c.format IS NULL OR c.format<>? OR c.presentation IS NULL) ORDER BY i.ordinal''', (job_id, cache_format)) if any(
                    i['run_id'] and (i['format'] != cache_format or i['presentation'] is None) for i in items) else ()
            for raw in misses:
                if raw['snapshot'] is None:
                    raise ValueError('Run not found')
                item = by_ordinal[raw['ordinal']]
                row = presentation_row(job_id, item, json.loads(raw['snapshot']),
                    [*preparations.get(item['ordinal'], []), *changes.get(item['run_id'], [])])
                built[item['ordinal']] = row
                pending.append((cache_format, json.dumps(row, ensure_ascii=False), job_id, item['ordinal'], item['generation']))
            for item in items:
                row = (built[item['ordinal']] if item['ordinal'] in built else json.loads(item['presentation'])) if item['run_id'] else None
                if row is None:
                    previous = previous_rework_display(db, item)
                    if previous:
                        old_item, old_state = previous
                        old_changes = [{'kind':'source_view_reselected', 'payload':json.loads(e['payload'])} for e in db.execute(
                            "SELECT payload FROM events WHERE run_id=? AND kind='source_view_reselected' ORDER BY seq", (old_item['run_id'],))]
                        old_item['target'] = json.dumps(old_item['target'])
                        row = presentation_row(job_id, old_item, old_state, [*preparations.get(item['ordinal'], []), *old_changes])
                        if row:
                            row.update(can_review=False, automatic_passed=False, manual_approved=False, usable=False, viewing_previous=True)
                            row['can_retry'] = False
                            if item.get('run_id'):
                                try:
                                    empty_rework_source(db,item,item['run_id'])
                                    row.update(can_retry=bool(row['revision']),retry_failed_run_id=item['run_id'])
                                except (ValueError,KeyError):
                                    pass
                if row is not None:
                    result.append({**row, 'reviewed': (row['ordinal'], row['revision']) in confirmations})
        if pending:
            # Never upgrade a read transaction or overwrite a newer generation.
            # Concurrent annotation/review/deletion may have invalidated any row.
            with store.runs.connect() as db:
                db.executemany('''UPDATE sample_presentations SET format=?,presentation=?
                    WHERE job_id=? AND ordinal=? AND generation=?''', pending)
        return {'job_id': job_id, 'created_at': header['created_at'], 'status': header['status'],
                'total_objects': len(items), 'samples': result,
                'task_goal':json.loads(header['spec']).get('desired_count'),
                'review_summary': {**{key: sum((s['manual_review'] or {}).get('verdict', 'unreviewed') == key for s in result)
                    for key in ('unreviewed', 'accepted', 'rejected', 'discarded')},
                    'usable': sum(s['usable'] for s in result)}}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/{job_id}/samples/{ordinal}/rework')
def rework(job_id: Id, ordinal: int, request: Request):
    """Selected round only, including failed drafts; no model calls or mutations."""
    store = request.app.state.batches.store
    try:
        with store.runs.connect() as db:
            db.execute('BEGIN')
            header = db.execute('SELECT status,control,reason,owner FROM batches WHERE job_id=?', (job_id,)).fetchone()
            if header is None:
                raise ValueError('Batch not found')
            rows = store.review_retry_progress(db, job_id, ordinal)
            if not rows:
                return {'job_id':job_id, 'ordinal':ordinal, 'item':None}
            item = rows[0]
            target = json.loads(db.execute('SELECT target FROM batch_items WHERE job_id=? AND ordinal=?', (job_id,ordinal)).fetchone()['target'])
            source = db.execute('SELECT snapshot FROM runs WHERE run_id=?', (target['manual_review_retry']['source_run_id'],)).fetchone()
            source = json.loads(source['snapshot']) if source else {}
            current = db.execute('SELECT snapshot FROM runs WHERE run_id=?', (item['run_id'],)).fetchone() if item['run_id'] else None
            current = json.loads(current['snapshot']) if current else {}
            attempts = []
            for attempt in current.get('attempts', []):
                if not attempt.get('generated'):
                    continue
                check = attempt.get('verification') or {}
                proof = check.get('multimodal_candidate_proof') or {}
                attempts.append({'expression':attempt['generated']['expression'], 'errors':check.get('errors', []),
                    'unsupported_claims':check.get('unsupported_claims', []),
                    'binding_retries':sum(bool(c.get('binding_retry_scheduled')) for c in attempt.get('ground_checks', [])),
                    'native_remaining':len(proof['native']) if 'native' in proof else None,
                    'image_remaining':len(proof['image_only']) if 'image_only' in proof else None,
                    'ground_rounds':len(attempt.get('ground_checks', []))})
            lease = db.execute('SELECT owner,lease_until FROM executor WHERE singleton=1').fetchone()
            now = time.time()
            task = {key:header[key] for key in ('status','control','reason')}
            task['executor_active'] = bool(lease and lease['owner']==header['owner'] and lease['lease_until']>now)
            item.update(previous_expression=sample_expression(source), feedback=(source.get('manual_review') or {}).get('notes'),
                        attempts=attempts, attempt_limit=current.get('max_repairs', 2)+1, manual_approved=manually_approved(current))
        return {'job_id':job_id, 'ordinal':ordinal, 'item':item, 'task':task, 'measured_at':now}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/{job_id}/samples/random')
def random_samples(job_id: Id, request: Request, count: int = Query(10, ge=1, le=1000), unreviewed: bool = True):
    rows = samples(job_id, request)['samples']
    pool = [s for s in rows if s['can_review'] and (s['manual_review'] or {}).get('verdict') != 'discarded'
        and (not unreviewed or s['manual_review'] is None)]
    chosen = random.SystemRandom().sample(pool, min(count, len(pool)))
    return {'job_id': job_id, 'ordinals': [s['ordinal'] for s in chosen], 'eligible': len(pool)}


@router.get('/{job_id}/samples/{ordinal}/scene')
def scene(job_id: Id, ordinal: int, request: Request, review: bool = False):
    try:
        store = request.app.state.batches.store
        job, item, state, packet, revision = sample(store, job_id, ordinal, allow_previous=True)
        path, _ = manifest(packet, root_for(request))
        scene, arrays = read_scene(path.parent)
        by_id, camera_boxes = native_labels(packet, root_for(request), scene, arrays)
        mapping = mapping_for(packet)
        labels = {oid: label for label, oid in mapping.items()}
        indices = sample_indices(np.arange(len(arrays['points'])), 22000)
        reader = WaymoReader(root_for(request) / 'waymo_trainset')
        projected = official_projection_boxes(reader.rows('projected_lidar_box', packet.context, timestamp=packet.frame_timestamp_micros) if 'projected_lidar_box' in packet.provenance.get('components', {}) else [],
            list(by_id.values()), {c.camera_name: (c.width, c.height) for c in scene.cameras})
        display_views = camera_views(path.parent, scene)
        cameras = []
        for camera in scene.cameras:
            boxes = []
            for n, box in enumerate(camera_boxes.get(camera.camera_name, [])):
                ids = box['laser_object_ids']
                linked = ids[0] if len(ids) == 1 and box['association_status'] != 'ambiguous' else None
                if linked not in by_id or by_id[linked].category != box['category']:
                    linked = None
                boxes.append({'box_id': f'I{camera.camera_name}-{n+1}', 'candidate_id': labels.get(linked),
                              'category': box['category'], 'xyxy': box['xyxy']})
            cameras.append({'camera_name': camera.camera_name, 'width': camera.width, 'height': camera.height,
                'image': f'/api/batches/{job_id}/samples/{ordinal}/images/{camera.camera_name}', 'native_boxes': boxes, 'view_axes': display_views.get(camera.camera_name),
                'projected_boxes': [{'box_id': f'P{camera.camera_name}-{n}', 'candidate_id': labels[b['object_id']],
                    'category': b['category'], 'xyxy': b['xyxy']} for n,b in enumerate(projected.get(camera.camera_name, []))]})
        previous = recorded(job, ordinal, revision)
        return {'job_id': job_id, 'ordinal': ordinal, 'revision': revision, 'expression': item['expression'],
            'source_view': None if review else packet.source_view,
            'image_box_source': 'official_lidar_box_projection' if packet.task.get('image_identity_policy') == 'official-projection-v1' else 'native_camera_box',
            'image_evidence': None if review else {
                'recommended_camera': next((c.camera_name for c in scene.cameras if CAMERAS[c.camera_name] == packet.source_view), next((v.camera_name for v in packet.views if v.most_visible_camera), None)),
                'point_projection_cameras': [v.camera_name for v in packet.views if v.appearance_identity_source == 'official_target_point_projection'],
                'lidar_points': packet.target.lidar_points,
            },
            'run_id': item['run_id'], 'can_confirm': state['status'] not in {'running', 'paused', 'stopped'},
            'manual_review': state.get('manual_review') if review_status(state) else None,
            'review': review, 'selected_candidate_id': None if review else labels[packet.target_object_id],
            'confirmation': previous[-1] if review and previous else None,
            'preview': {'points': arrays['points'][indices, :3].tolist(), 'target_points': [], 'target_corners': [],
                'candidate_boxes': [{'object_id': label, 'category': by_id[oid].category, 'corners': corners(by_id[oid])} for label, oid in mapping.items()],
                'sampling': 'target-independent full-scene preview; native 3D boxes'},
            'candidates': [{'candidate_id': label, 'category': by_id[oid].category} for label, oid in mapping.items()],
            'cameras': cameras}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: str = Field(pattern=r'^[a-f0-9]{64}$')
    verdict: Literal['accepted', 'rejected', 'discarded']
    notes: str = Field(default='', max_length=2000)
    approve_failed: bool = Field(default=False, strict=True)


@router.post('/{job_id}/samples/{ordinal}/review')
def review_sample(job_id: Id, ordinal: int, body: ReviewDecision, request: Request):
    return _review_sample(job_id,ordinal,body,request)


def _review_sample(job_id, ordinal, body, request, *, retry_failed_run_id=None):
    mutation_origin(request)
    store = request.app.state.batches.store
    try:
        with store.runs.maintenance():
            job, item, state, packet, revision = sample(store, job_id, ordinal,retry_failed_run_id=retry_failed_run_id)
            if retry_failed_run_id and body.verdict != 'rejected':
                raise ValueError('返工前版本仅可再次反馈返工，不能审批为有效样本。')
            if state['status'] in {'running', 'paused', 'stopped'}:
                raise ValueError('请等待本条标注结束后再审核。')
            if revision != body.revision:
                raise ValueError('描述版本已变化，请重新打开样本。')
            notes = body.notes.strip()
            if body.verdict == 'rejected' and not notes:
                raise ValueError('请填写不通过原因，便于 Agent 定向修正。')
            confirmations = recorded(job, ordinal, revision)
            if body.verdict == 'accepted' and (not confirmations or not confirmations[-1]['correct']):
                raise ValueError('请先完成正确的物体定位确认，再审核通过。')
            override = body.verdict=='accepted' and not passed_description(state)
            if override and not body.approve_failed:
                raise ValueError('自动核验未通过，请明确选择“人工审批通过并纳入可用集”。')
            if body.verdict=='accepted' and (state.get('inflight') or state['status'] not in {
                    'calibration_hold','auto_pass','needs_evidence','abstained','failed'}):
                raise ValueError('该样本尚未安全结束，请等待处理结束后再审批。')
            previous = state.get('manual_review', {})
            if (review_status(state) == body.verdict and previous.get('revision') == revision
                    and previous.get('notes') == notes and previous.get('retry_failed_run_id') == retry_failed_run_id
                    and (not override or manually_approved(state))):
                return previous
            decision = {'job_id': job_id, 'ordinal': ordinal, 'run_id': item['run_id'],
                'revision': revision, 'expression_sha256': text_hash(item['expression']),
                'verdict': body.verdict, 'notes': notes, 'at': time.time(),
                'scope': 'manual sample review with target exposure; not independent blind evaluation'}
            if override:
                decision.update(approval_policy=MANUAL_APPROVAL_POLICY, packet_binding=approval_binding(state['current_packet']),
                    localization_correct=True, confirmed_object_id=packet.target_object_id,
                    automatic_passed=False, automatic_failure_reason=state.get('stop_reason'),
                    confirmation_revision=confirmations[-1]['revision'])
            if retry_failed_run_id:
                decision['retry_failed_run_id'] = retry_failed_run_id
            decision['review_id'] = fingerprint(decision)
            with store.runs.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT snapshot,revision FROM runs WHERE run_id=?', (item['run_id'],)).fetchone()
                linked = db.execute('SELECT * FROM batch_items WHERE job_id=? AND ordinal=?', (job_id,ordinal)).fetchone()
                if retry_failed_run_id and linked:
                    current_source, _ = empty_rework_source(db,dict(linked),retry_failed_run_id)
                    valid_link = (current_source['run_id'],current_source['packet_id'])==(item['run_id'],item['packet_id'])
                else:
                    valid_link = linked is not None and (linked['run_id'],linked['packet_id'])==(item['run_id'],item['packet_id'])
                if not valid_link:
                    raise ValueError('样本已重新返工或变化，请刷新后再审批。')
                if row is None or fingerprint(json.loads(row['snapshot'])) != fingerprint(state):
                    raise ValueError('样本在审核期间发生变化，请刷新后重试。')
                state['manual_review'] = decision
                next_revision = row['revision'] + 1
                db.execute('UPDATE runs SET snapshot=?,revision=? WHERE run_id=?',
                    (json.dumps(state, allow_nan=False), next_revision, item['run_id']))
                db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                    (item['run_id'], next_revision, REVIEW_EVENT, json.dumps(decision), decision['at']))
                store.event(db, job_id, REVIEW_EVENT, decision)
            return decision
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


class RetryReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: str = Field(pattern=r'^[a-f0-9]{64}$')
    review_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    notes: str | None = Field(default=None, min_length=1, max_length=2000)
    retry_failed_run_id: str | None = Field(default=None,pattern=r'^[a-f0-9]{24}$')


class DeleteSample(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    packet_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    expression: str = Field(min_length=1, max_length=400)


@router.delete('/{job_id}/samples/{ordinal}')
def delete_sample(job_id: Id, ordinal: int, body: DeleteSample, request: Request):
    """Discard the displayed version using the existing review/export rules."""
    mutation_origin(request)
    try:
        _, item, _, _, revision = sample(request.app.state.batches.store, job_id, ordinal)
        if (item['run_id'], item['packet_id'], item['expression']) != (body.run_id, body.packet_id, body.expression):
            raise ValueError('这条标注已变化，请刷新后重新确认要删除的文本。')
        decision = review_sample(job_id, ordinal, ReviewDecision(revision=revision, verdict='discarded',
            notes='从标注结果列表删除。'), request)
        store = request.app.state.batches.store
        return {**decision, 'progress': store.execution_progress(job_id), 'jobs': store.summaries()}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/{job_id}/samples/{ordinal}/retry', status_code=202)
def retry_sample(job_id: Id, ordinal: int, body: RetryReview, request: Request):
    mutation_origin(request)
    service = request.app.state.batches
    try:
        with service.runner.store.maintenance():
            service.ensure_loaded_code()
            # Idempotency survives item relinking, process restart and repeat clicks.
            with service.runner.store.connect() as db:
                events = [json.loads(row['payload']) for row in db.execute(
                    "SELECT payload FROM batch_events WHERE job_id=? AND kind='manual_review_retry_requested' AND json_extract(payload,'$.ordinal')=? ORDER BY seq DESC",
                    (job_id,ordinal))]
            existing = next((event for event in events if event.get('source_revision')==body.revision
                and event.get('retry_failed_run_id')==body.retry_failed_run_id
                and (event.get('review_id')==body.review_id if body.review_id else event.get('notes')==(body.notes or '').strip())),None)
            if not existing:
                job, item, state, _, revision = sample(service.store, job_id, ordinal,retry_failed_run_id=body.retry_failed_run_id)
                if revision != body.revision:
                    raise ValueError('审核结论或样本版本已变化，请刷新后重试。')
                goal = job['spec'].get('desired_count')
                if goal is not None and service.store.execution_progress(job_id)['passed'] >= goal:
                    raise ValueError('本任务已达到目标条数，该条不能返工；请先手动删除一条有效标注，腾出名额后再重试。')
                if body.notes is not None:
                    decision = _review_sample(job_id,ordinal,ReviewDecision(revision=revision,
                        verdict='rejected',notes=body.notes),request,retry_failed_run_id=body.retry_failed_run_id)
                else:
                    decision = state.get('manual_review',{})
                    if review_status(state)!='rejected' or decision.get('review_id')!=body.review_id:
                        raise ValueError('请先填写反馈，或使用当前不通过审核的版本。')
                reference = ReviewRetry(source_run_id=item['run_id'],review_id=decision['review_id'])
                service.retry_review(job_id,ordinal,reference,revision,retry_failed_run_id=body.retry_failed_run_id)
            # An active coordinator reads this item's new round without restart.
            result = service.start(job_id)
            queued = result.get('queued',False)
            return {'job_id':job_id, 'ordinal':ordinal, 'started':not queued,
                'reused_job':True, 'queued':queued,
                'manual_review':service.runner.store.snapshot((existing or {}).get('feedback_source_run_id') or (existing or {}).get('source_run_id') or reference.source_run_id)['manual_review'],
                'message':'已加入原任务，等待执行。' if queued else '已在原任务内安排该样本重试；通过后计入本任务合格数量。'}
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/{job_id}/samples/{ordinal}/images/{camera}')
def image(job_id: Id, ordinal: int, camera: int, request: Request):
    try:
        _, _, _, packet, _ = sample(request.app.state.batches.store, job_id, ordinal, allow_previous=True)
        path, scene = manifest(packet, root_for(request))
        selected = next((c for c in scene.cameras if c.camera_name == camera), None)
        if selected is None:
            raise ValueError('该帧没有此相机图像。')
        return FileResponse(checked_asset(path.parent, selected.image.path, selected.image.sha256))
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


class Confirm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: str = Field(pattern=r'^[a-f0-9]{64}$')
    selected_candidate_id: str = Field(pattern=r'^C[0-9]{3,}$')


@router.post('/{job_id}/samples/{ordinal}/confirm')
def confirm(job_id: Id, ordinal: int, body: Confirm, request: Request):
    mutation_origin(request)
    store = request.app.state.batches.store
    try:
        with store.runs.maintenance():
            job, item, state, packet, revision = sample(store, job_id, ordinal)
            if state['status'] in {'running', 'paused', 'stopped'}:
                raise ValueError('该样本仍在标注中，请等待本条处理结束后再确认。')
            if body.revision != revision:
                raise ValueError('样本文本或场景版本已变化，请重新打开后检查。')
            path, _ = manifest(packet, root_for(request))
            source_scene, arrays = read_scene(path.parent)
            native_labels(packet, root_for(request), source_scene, arrays)
            mapping = mapping_for(packet)
            if body.selected_candidate_id not in mapping:
                raise ValueError('所选物体不属于当前场景。')
            previous = recorded(job, ordinal, revision)
            if previous and previous[-1]['selected_candidate_id'] == body.selected_candidate_id:
                return previous[-1]
            expected = next(label for label, oid in mapping.items() if oid == packet.target_object_id)
            correct = body.selected_candidate_id == expected
            result = {'ordinal': ordinal, 'revision': revision, 'run_id': item['run_id'],
                'expression_sha256': hashlib.sha256(item['expression'].encode()).hexdigest(),
                'selected_candidate_id': body.selected_candidate_id, 'expected_candidate_id': expected,
                'correct': correct, 'first_attempt_correct': previous[0]['correct'] if previous else correct,
                'attempt': len(previous) + 1, 'scope': 'manual localization with immediate feedback; not independent blind evaluation'}
            with store.runs.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                current = db.execute('SELECT run_id,packet_id FROM batch_items WHERE job_id=? AND ordinal=?', (job_id, ordinal)).fetchone()
                if current is None or current['run_id'] != item['run_id'] or current['packet_id'] != packet.packet_id:
                    raise ValueError('任务或样本已删除或变化，请刷新。')
                current_run = db.execute('SELECT snapshot FROM runs WHERE run_id=?', (item['run_id'],)).fetchone()
                if current_run is None or fingerprint(json.loads(current_run['snapshot'])) != fingerprint(state):
                    raise ValueError('标注记录在核对期间发生变化，请重新打开后检查。')
                store.event(db, job_id, REVIEW_KIND, result)
            return result
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
