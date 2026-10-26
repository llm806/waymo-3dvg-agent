"""Small durable track exclusions, independent of deletable task artifacts."""
import time
import hashlib
import json
import sqlite3
import re
from pathlib import Path


def initialize(db):
    db.execute("""CREATE TABLE IF NOT EXISTS candidate_exclusions (
        context TEXT NOT NULL, object_id TEXT NOT NULL, source TEXT NOT NULL,
        created_at REAL NOT NULL, PRIMARY KEY(context,object_id,source))""")


def exclude(db, keys, source):
    db.executemany('INSERT OR IGNORE INTO candidate_exclusions VALUES(?,?,?,?)',
                   [(context, object_id, source, time.time()) for context, object_id in sorted(set(keys))])


def excluded_keys(db):
    return {(row['context'], row['object_id']) for row in db.execute(
        'SELECT DISTINCT context,object_id FROM candidate_exclusions')}


def reopen_failed(store, job_id, *, additional_calls=0, ordinals=None, dry_run=False):
    """Explicitly reopen settled failures, without changing any run or bill.

    A per-track allowance is a recorded upper bound on cumulative usage, not a
    reset. The authorization is consumed atomically when its next run is linked.
    Current workers and successful samples are never reset or dispatched here.
    """
    from uuid import uuid4
    from collections import defaultdict
    from app.agent.selection import usable_description
    from app.agent.model_calls import saved_call_diagnostic
    if type(additional_calls) is not int or not 0 <= additional_calls <= 12:
        raise ValueError('本轮追加额度须为 0–12 次，历史费用不清零。')
    with store.connect() as db:
        header = db.execute('SELECT spec,status FROM batches WHERE job_id=?',(job_id,)).fetchone()
        if header is None:
            raise ValueError('Batch not found')
        if json.loads(header['spec']).get('prepare_only'):
            raise ValueError('数据准备任务不包含待重试标注。')
        rows = db.execute("""SELECT i.*,r.snapshot AS source_snapshot,v.revision AS source_revision
            FROM batch_items i LEFT JOIN run_summaries r ON r.run_id=i.run_id
            LEFT JOIN runs v ON v.run_id=i.run_id
            WHERE i.job_id=? AND (i.status IN ('needs_evidence','abstained','failed','excluded','budget_skipped')
                OR i.status='not_needed' AND i.reason='target_already_attempted')""",(job_id,)).fetchall()
        excluded = excluded_keys(db)
    history = defaultdict(list)
    for state in store.selection_runs():
        if state.get('provider_scope'):
            packet = state.get('current_packet',{})
            history[packet.get('context'),packet.get('target_object_id')].append(state)
    active_authorizations = store.candidate_retries()
    selected = None if ordinals is None else set(ordinals)
    changes, skipped = [], defaultdict(int)
    at, token = time.time(),uuid4().hex[:24]
    for row in rows:
        if selected is not None and row['ordinal'] not in selected:
            continue
        target = json.loads(row['target']);key = target['context'],target['object_id']
        source = json.loads(row['source_snapshot']) if row['source_snapshot'] else None
        scope = (source or {}).get('provider_scope','real-model')
        states = history[key]
        if target.get('status') != 'selected' or key in excluded:
            skipped['excluded'] += 1;continue
        if any(usable_description(s) for s in states) or source and usable_description(source):
            skipped['already_passed'] += 1;continue
        if key in active_authorizations:
            skipped['already_reopened'] += 1;continue
        if any(s.get('status') in {'running','paused'} for s in states):
            skipped['unfinished'] += 1;continue
        if any(s.get('stop_reason') in {'interrupted_response_outcome_unknown','model_call:interrupted_response'} for s in states):
            skipped['unknown_outcome'] += 1;continue
        if source and source.get('inflight'):
            folder = store.path.parent/source['run_id']/source['inflight']['folder']
            diagnostic = saved_call_diagnostic(folder)
            if (not (folder/'parsed-output.json').is_file() and diagnostic.get('remote_outcome') != 'not_sent'
                    and not (diagnostic.get('http_status') is not None and diagnostic.get('remote_outcome') != 'unknown')):
                skipped['unknown_outcome'] += 1;continue
        with store.connect() as db:
            spent = store._target_usage(db,key,scope)
        limit = max(12,target.get('retry_authorization',{}).get('call_limit',12),spent+additional_calls)
        if limit-spent < 4:
            skipped['insufficient_budget'] += 1;continue
        authorization = {'token':token+f'-{row["ordinal"]}', 'source_job_id':job_id,
            'source_run_id':row['run_id'], 'calls_before':spent, 'additional_calls':additional_calls,
            'call_limit':limit,'authorized_at':at,'provider_scope':scope}
        target.update(execution_attempt=authorization['token'],retry_authorization=authorization)
        changes.append((row,target,authorization))
    receipt = {'job_id':job_id,'started':False,'reopened':len(changes),'skipped':dict(skipped),
        'additional_calls_per_target':additional_calls,'ordinals':[r['ordinal'] for r,_,_ in changes]}
    if dry_run or not changes:
        return {**receipt,'dry_run':dry_run}
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        for row,target,authorization in changes:
            key=target['context'],target['object_id']
            current = db.execute('SELECT revision,owner,lease_until FROM runs WHERE run_id=?',(row['run_id'],)).fetchone()
            if row['run_id'] and (current is None or current['revision']!=row['source_revision']
                    or current['owner'] and current['lease_until']>time.time()):
                raise ValueError('所选失败记录已变化，请重新核对；未修改任何目标。')
            passed_rows = db.execute("""SELECT snapshot FROM run_live_summaries WHERE
                json_extract(snapshot,'$.current_packet.context')=? AND
                json_extract(snapshot,'$.current_packet.target_object_id')=? AND (
                json_extract(snapshot,'$.diagnostic_pass')=1 AND
                json_extract(snapshot,'$.status') IN ('calibration_hold','auto_pass') AND
                length(json_extract(snapshot,'$.final_expression'))>0 OR
                json_extract(snapshot,'$.manual_review.approval_policy')='localized-human-approval-v1')""",key)
            passed=any(usable_description(json.loads(r['snapshot'])) for r in passed_rows)
            if passed or store._target_usage(db,key,authorization['provider_scope'])!=authorization['calls_before']:
                raise ValueError('所选目标的结果或累计调用已变化，请重新核对；未修改任何目标。')
            changed = db.execute("""UPDATE batch_items SET target=?,status='pending',reason=NULL,
                run_id=NULL,packet_id=NULL,reused=0 WHERE job_id=? AND ordinal=? AND status=?
                AND run_id IS ? AND target=?""",(json.dumps(target,allow_nan=False),job_id,row['ordinal'],
                    row['status'],row['run_id'],row['target'])).rowcount
            if changed!=1:
                raise ValueError('所选候选状态已变化，请重新核对；未修改任何目标。')
        payload={**receipt,'authorizations':[authorization for _,_,authorization in changes],
            'scope':'explicit retry; original run snapshots, requests, usage, successful results and task control preserved'}
        db.execute('INSERT INTO batch_events(job_id,kind,payload,at) VALUES(?,?,?,?)',
            (job_id,'failed_candidates_reopened',json.dumps(payload,allow_nan=False),at))
    return receipt


def import_manifest(path, database, *, expected_count=None):
    """Restore exclusions from a compact export's small samples.jsonl only."""
    from app.datasets.export_loader import validate_sample
    raw = Path(path).read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    keys, samples = set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('导出清单必须由样本对象组成。')
        validate_sample(record)
        if (not all(isinstance(record.get(k), str) and record[k].strip() for k in ('context', 'track_id'))
                or not re.fullmatch('[a-f0-9]{24}', str(record.get('sample_id', '')))):
            raise ValueError('导出清单中的样本或原生轨迹身份无效。')
        keys.add((record['context'], record['track_id']))
        samples.add(record['sample_id'])
    if not records or len(keys) != len(records) or len(samples) != len(records):
        raise ValueError('导出清单必须包含非空且不重复的样本和轨迹。')
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f'导出清单应有 {expected_count} 条，实际为 {len(records)} 条。')
    database = Path(database).resolve(strict=True)
    # Read/write an existing store only; never silently create another DB.
    with sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=30) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN IMMEDIATE')
        initialize(db)
        source = 'export-manifest:' + hashlib.sha256(raw).hexdigest()
        before = db.total_changes
        exclude(db, keys, source)
        added = db.total_changes - before
    return {'records': len(records), 'tracks': len(keys), 'added': added, 'source': source}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='从已导出的 samples.jsonl 恢复候选池排除记录。')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--expected-count', type=int)
    arguments = parser.parse_args()
    print(json.dumps(import_manifest(arguments.manifest,
        Path(__file__).resolve().parents[2] / 'runtime/annotation/runs.sqlite3',
        expected_count=arguments.expected_count), ensure_ascii=False))
