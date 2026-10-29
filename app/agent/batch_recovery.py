"""Rewind incomplete items inside one durable task, keeping billed call history."""
import hashlib
import json
import os
import re
import time
from uuid import uuid4

from app.agent.selection import usable_description
from app.agent.model_calls import saved_call_diagnostic, image_model_error
from app.infrastructure.storage import RunBusy


def prepare_recovery(service, job, *, dry_run=False):
    # The caller holds the runner and maintenance locks; no worker may start.
    from app.agent.batch import method_fingerprint, freeze_method
    runs, store = service.runner.store, service.store
    service.ensure_loaded_code()
    detail = store.detail(job, compact=True)
    if detail['executor_active']:
        raise RunBusy('请等待当前任务保存并退出执行器后再继续。')
    with runs.connect() as db:
        if db.execute('SELECT 1 FROM executor WHERE lease_until>?', (time.time(),)).fetchone() or db.execute(
                'SELECT 1 FROM runs WHERE owner IS NOT NULL AND lease_until>?', (time.time(),)).fetchone():
            raise RunBusy('另有任务正在执行，请等待该任务完成或暂停。')
    goal = detail['spec'].get('desired_count')
    passed = detail['selection_progress']['diagnostic_passed']
    if goal is not None and passed >= goal:
        return {'job_id':job, 'started':False, 'retained_passed':passed, 'remaining_goal':0, 'reset_items':0}
    catalog = service.catalog_loader(detail['spec']['catalog_id'])
    admission = catalog.get('selection_policy', {}).get('admission_unit')
    if not detail['spec']['prepare_only'] and admission and admission not in {
            'verified_native_image_pair', 'official_projection_candidate'}:
        raise ValueError('候选目录使用旧准入规则，请更新目录后再继续；已有样本保持原样。')
    method = method_fingerprint()
    provider = None if detail['spec']['prepare_only'] else {
        'model': os.getenv('DASHSCOPE_MODEL', 'qwen3.8-flash'),
        'base_url': os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}
    if provider and (error := image_model_error(provider['model'], provider['base_url'])):
        raise ValueError(error)
    digest = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    changed = (method != detail['method_sha256'] or digest != detail['catalog_sha256']
               or provider != detail['spec'].get('provider'))
    if detail['status'] == 'ready' and not changed:
        return {'job_id':job,'started':False,'retained_passed':passed,
                'remaining_goal':None if goal is None else max(0,goal-passed),'reset_items':0}
    available = {target['target_id']:target for target in catalog['targets']}
    review_ordinals={item['ordinal'] for item in store.pending_review_retries(job)}
    review_only=bool(detail['status']=='ready' and review_ordinals and not any(
        item['status'] in {'pending','preparing','running'} and item['ordinal'] not in review_ordinals
        for item in detail['items']))
    changes, retired, uncertain, unknown_states = [], [], [], []
    with runs.connect() as db:
        completed = set()
        for row in db.execute("""SELECT run_id,snapshot FROM run_live_summaries
            WHERE run_id IN (SELECT run_id FROM batch_items WHERE job_id=?)""", (job,)):
            if usable_description(json.loads(row['snapshot'])):
                completed.add(row['run_id'])
    attempt = uuid4().hex[:24]
    for item in detail['items']:
        if review_only and item['ordinal'] not in review_ordinals:
            continue  # Updating code for one explicit review is not a batch-wide retry.
        if item.get('run_id') in completed:
            continue
        potential_retry = (any(item.get(key) for key in ('retryable_connection_failure','retryable_parameter_failure',
            'retryable_preparation_failure','retryable_visibility_failure','retryable_evidence_failure'))
            or item.get('run_status') == 'blocked')
        if item['status'] not in {'pending','preparing','running','budget_skipped','not_needed'} and not potential_retry:
            continue
        state = runs.snapshot(item['run_id']) if item.get('run_id') else None
        retryable = (item.get('retryable_connection_failure') or changed and (
            item.get('retryable_parameter_failure') or item.get('retryable_preparation_failure')
            or item.get('retryable_visibility_failure') or item.get('retryable_evidence_failure')) or state and state.get('status') == 'blocked'
            and state.get('stop_reason') != 'interrupted_response_outcome_unknown')
        unused_reserve = (goal is not None and passed < goal and item['status']=='not_needed'
            and item.get('reason')=='description_goal_reached' and not item.get('run_id') and not item.get('packet_id'))
        if item['status'] not in {'pending','preparing','running','budget_skipped'} and not retryable and not unused_reserve:
            continue
        target = available.get(item['target']['target_id'])
        if target is None or any(target.get(key) != item['target'].get(key) for key in ('context','object_id','timestamp')):
            raise ValueError('待处理目标的原生身份已变化，无法直接继续；已有样本保持原样。')
        if not detail['spec']['prepare_only'] and admission == 'verified_native_image_pair' and target.get('eligibility_stage') != 'native_image_pair_confirmed':
            raise ValueError('待处理目标尚未确认原生图像与 3D 身份，无法直接继续；已有样本保持原样。')
        if state and state.get('inflight'):
            folder = service.runner.root / state['run_id'] / state['inflight']['folder']
            diagnostic = saved_call_diagnostic(folder)
            result = json.loads((folder/'result.json').read_text(encoding='utf-8')) if (folder/'result.json').exists() else {}
            known = ((folder/'parsed-output.json').is_file() or diagnostic.get('remote_outcome') == 'not_sent'
                     or (diagnostic.get('http_status') is not None or result.get('status') in {'completed','invalid_output'})
                     and diagnostic.get('remote_outcome') != 'unknown')
            if not known:
                uncertain.append(item['ordinal'])
                unknown_states.append(state)
                changes.append((item, None, 'failed', 'interrupted_response_outcome_unknown'))
                continue
        pending = {**target, 'queue_position':item['target'].get('queue_position', item['ordinal'])}
        if item['target'].get('manual_review_retry'):
            pending['manual_review_retry'] = item['target']['manual_review_retry']
        if item['target'].get('retry_authorization'):
            authorization = dict(item['target']['retry_authorization'])
            if state and authorization.pop('consumed_run_id',None):
                # Resume an interrupted authorized round within its EXISTING
                # cumulative ceiling; do not grant another twelve calls.
                authorization.update(token=f'{attempt}-{item["ordinal"]}',
                    source_run_id=state['run_id'],continuation_of=authorization['token'],additional_calls=0)
            pending['retry_authorization'] = authorization
        if item['target'].get('execution_attempt'):
            pending['execution_attempt'] = item['target']['execution_attempt']
        if state or item.get('packet_id') or item['status'] != 'pending' or changed:
            pending['execution_attempt'] = attempt
        changes.append((item, pending, 'pending', None))
        if state:
            retired.append(state)
    budget = store.continuation_budget(job)
    quota = detail['spec'].get('max_model_calls')
    own_remaining = None if quota is None else quota-detail['model_calls_reserved']
    family_remaining = None if budget['limit'] is None else budget['limit']-budget['spent']
    if (not detail['spec']['prepare_only'] and any(value is not None and value < 4 for value in (own_remaining,family_remaining))
            and not any(target and target.get('manual_review_retry') for _,target,_,_ in changes)):
        raise ValueError('原任务累计剩余额度不足以完成下一条标注；已完成样本和费用记录保留。')
    if not any(target is not None for _,target,_,_ in changes):
        raise ValueError('没有可继续处理的目标；结果未知的请求不会重复发送。')
    specification = {**detail['spec'], 'provider':provider}
    receipt = {'job_id':job, 'started':False, 'retained_passed':passed,
        'remaining_goal':None if goal is None else max(0,goal-passed),
        'reset_items':sum(target is not None and bool(item.get('run_id') or item.get('packet_id')) for item,target,_,_ in changes),
        'skipped_unknown_outcome_targets':len(uncertain),
        'recovery_scope':'pending_review_retries' if review_only else 'unfinished_task',
        **({'review_ordinals':sorted(review_ordinals)} if review_only else {})}
    if dry_run:
        return {**receipt, 'dry_run':True, 'code_or_configuration_changed':changed,
            'ready_targets':sum(target is not None for _,target,_,_ in changes),
            'retired_run_ids':[state['run_id'] for state in retired],
            'skipped_unknown_ordinals':uncertain, 'model_calls_reserved':detail['model_calls_reserved']}
    freeze_method(service.runner.root, method)
    with runs.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = {key:detail[key] for key in ('spec','method_sha256','catalog_sha256')}
        current = {'spec':specification,'method_sha256':method,'catalog_sha256':digest}
        for state in [*retired,*unknown_states]:
            row = db.execute('SELECT revision FROM runs WHERE run_id=?', (state['run_id'],)).fetchone()
            unknown = state in unknown_states
            state = {**state, 'status':'blocked' if unknown else 'stopped',
                     'retired_from_execution':not unknown}
            if unknown:
                state['stop_reason'] = 'interrupted_response_outcome_unknown'
            revision = row['revision']+1
            db.execute('UPDATE runs SET snapshot=?,revision=?,owner=NULL,lease_until=0 WHERE run_id=?',
                       (json.dumps(state,allow_nan=False),revision,state['run_id']))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(state['run_id'],revision,'attempt_outcome_unknown' if unknown else 'attempt_retired',
                json.dumps({'job_id':job,'execution_attempt':attempt,'scope':'unfinished draft retired; calls retained'}),time.time()))
            db.execute('DELETE FROM controls WHERE run_id=?',(state['run_id'],))
        for item,target,status,reason in changes:
            if target is None:
                db.execute('UPDATE batch_items SET status=?,reason=? WHERE job_id=? AND ordinal=?',
                           (status,reason,job,item['ordinal']))
            else:
                db.execute('UPDATE batch_items SET target=?,status=?,reason=NULL,run_id=NULL,packet_id=NULL,reused=0 WHERE job_id=? AND ordinal=?',
                           (json.dumps(target,allow_nan=False),status,job,item['ordinal']))
        db.execute("UPDATE batches SET spec=?,method_sha256=?,catalog_sha256=?,status='ready',control=NULL,owner=NULL,reason=NULL WHERE job_id=?",
                   (json.dumps(specification,allow_nan=False),method,digest,job))
        store.event(db,job,'execution_revision',{'previous':previous,'current':current,'execution_attempt':attempt,
            'retired_run_ids':[state['run_id'] for state in retired],
            'reset_ordinals':[item['ordinal'] for item,target,_,_ in changes if target is not None], **receipt})
    return receipt


def cleanup_temporary(service, job):
    """Remove disposable files only; complete assets and model traces stay."""
    from app.infrastructure.task_cleanup import TaskCleaner, ids
    root = service.runner.root
    with service.runner.store.connect() as db:
        revisions = [json.loads(row['payload']) for row in db.execute(
            "SELECT payload FROM batch_events WHERE job_id=? AND kind='execution_revision'",(job,))]
    if not revisions:
        return
    folders = {root/'batches'/job/str(ordinal) for event in revisions for ordinal in event['reset_ordinals']}
    folders |= {root/run for event in revisions for run in event['retired_run_ids']}
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.rglob('*'):
            if path.is_file() and (path.name.endswith('.tmp') or path.name.endswith('.tmp.npz')
                                 or path.name == 'preparation-progress.json'):
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('临时文件路径越界，已停止清理。')
                path.unlink()
    # Standard runtime roots have shared converter/evidence cache pools. Test
    # and custom output roots still get the private-file cleanup above.
    if root.name != 'annotation' or root.parent.name != 'runtime':
        return
    cleaner = TaskCleaner(service.runner.store, root.parent.parent)
    candidates, protected = [], set()
    for name, manifest in (('scenes','scene.json'), ('evidence','packet.json')):
        pool = root.parent/name
        index = pool/'index.json'
        if index.exists():
            protected |= ids(json.loads(index.read_text(encoding='utf-8')))
        if not pool.exists():
            continue
        for folder in pool.iterdir():
            if folder.is_dir() and re.fullmatch('[a-f0-9]{24}',folder.name) and not (folder/manifest).exists():
                candidates.append(folder)
    candidates = [folder for folder in candidates if folder.name not in protected]
    if not candidates:
        return  # No unpublished cache needs the expensive full reference scan.
    protected |= ids(service.runner.store.all_runs())
    with service.runner.store.connect() as db:
        protected |= {row['packet_id'] for row in db.execute('SELECT packet_id FROM batch_items WHERE packet_id IS NOT NULL')}
    for folder in candidates:
        if folder.name not in protected:
            cleaner._remove(folder)
