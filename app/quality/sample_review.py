"""Shared sample identity, version-bound decisions and durable retry references."""
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

REVIEW_EVENT = 'sample_reviewed'
_NO_ITEM_PACKET = object()


class ReviewRetry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_run_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    review_id: str = Field(pattern=r'^[a-f0-9]{64}$')


def expression(state):
    return state.get('final_expression') or next((a['generated']['expression']
        for a in reversed(state.get('attempts', [])) if a.get('generated')), None)


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sample_revision(job, ordinal, run_id, text, packet):
    return hashlib.sha256(json.dumps({'job_id': job, 'ordinal': ordinal,
        'run_id': run_id, 'expression': text, 'packet': packet.model_dump(mode='json')},
        sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def review_status(state):
    review = state.get('manual_review') or {}
    text = expression(state)
    if text and review.get('expression_sha256') == text_hash(text):
        verdict = review.get('verdict')
        if verdict in {'accepted', 'rejected', 'discarded'}:
            return verdict
    return None


MANUAL_APPROVAL_POLICY = 'localized-human-approval-v1'


def approval_binding(packet):
    """Bind the human decision to the actual image/frame/native supervision."""
    fields = {key:packet.get(key) for key in ('packet_id','context','frame_timestamp_micros',
        'target_object_id','source_view','source_camera_object_id','target','scene')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def manually_approved(state):
    decision = state.get('manual_review') or {}
    packet = state.get('current_packet') or {}
    return bool(review_status(state)=='accepted'
        and decision.get('approval_policy')==MANUAL_APPROVAL_POLICY
        and decision.get('run_id')==state.get('run_id')
        and decision.get('packet_binding')==approval_binding(packet)
        and decision.get('localization_correct') is True
        and packet.get('target_object_id')
        and decision.get('confirmed_object_id')==packet['target_object_id']
        and state.get('status') in {'calibration_hold','auto_pass','needs_evidence','abstained','failed'}
        and not state.get('inflight'))


def previous_rework_display(db, item):
    """Keep the rejected input visible until the new round produces text."""
    target = item['target'] if isinstance(item['target'], dict) else json.loads(item['target'])
    source_id = target.get('manual_review_retry', {}).get('source_run_id')
    row = db.execute('SELECT snapshot FROM runs WHERE run_id=?', (source_id,)).fetchone() if source_id else None
    if row is None:
        return None
    source = json.loads(row['snapshot'])
    return {**item, 'target':target, 'run_id':source_id, 'packet_id':source['packet_id'],
            'expression':expression(source)}, source


def empty_rework_source(db, item, failed_run_id):
    """Authorize feedback on the visible input only after an empty round ends."""
    if item.get('run_id') != failed_run_id or item['status'] not in {'needs_evidence','abstained','failed','blocked'}:
        raise ValueError('返工版本已变化或尚未结束，请刷新后重试。')
    raw = db.execute('SELECT snapshot FROM runs WHERE run_id=?',(failed_run_id,)).fetchone()
    failed = json.loads(raw['snapshot']) if raw else {}
    if (expression(failed) or failed.get('status') not in {'needs_evidence','abstained','failed','blocked'}
            or failed.get('inflight') or failed.get('stop_reason') in {
                'interrupted_response_outcome_unknown','model_call:interrupted_response'}):
        raise ValueError('返工尚未安全结束或已产生新描述，请刷新后重试。')
    previous = previous_rework_display(db,item)
    if not previous or not expression(previous[1]):
        raise ValueError('找不到可供反馈的返工前描述。')
    old_packet, current_packet = previous[1]['current_packet'], failed.get('current_packet',{})
    if any(old_packet.get(k) != current_packet.get(k) for k in ('context','target_object_id')):
        raise ValueError('返工前后目标身份不一致。')
    return previous


def retry_source(runs, reference, *, require_current=False):
    state = runs.snapshot(reference.source_run_id)
    receipt = next((e['payload'] for e in runs.events(reference.source_run_id)
        if e['kind'] == REVIEW_EVENT and e['payload'].get('review_id') == reference.review_id), None)
    if (not receipt or receipt['verdict'] != 'rejected'
            or receipt['expression_sha256'] != text_hash(expression(state) or '')):
        raise ValueError('返工来源或描述版本不匹配。')
    if require_current and (review_status(state) != 'rejected'
            or state['manual_review']['review_id'] != reference.review_id):
        raise ValueError('请先将当前描述审核为不通过，再创建返工任务。')
    return state, receipt


def validate_sample_identity(target, state, events=(), *, ordinal=None, item_packet_id=_NO_ITEM_PACKET):
    """One metadata-only binding check for commit, viewing and export.

    Preparation receipts bind the accepted input to a frozen alternate. Runtime
    switches continue that chain. No files, arrays, model calls or writes occur.
    """
    packet = state['current_packet']
    bound = (state.get('input_packet_id', state['packet_id']), target['timestamp'])
    # Preparation can reject the catalog frame before a model run even exists.
    # Its receipt binds the run's input packet to a frozen alternate frame;
    # runtime source-view/frame events continue from that accepted input.
    receipts = [e['payload'] for e in events if e['kind'] == 'preparation_source_reselected'
        and e['payload'].get('ordinal') == ordinal and e['payload'].get('packet_id') == bound[0]]
    if state.get('preparation_source_reselection'):
        receipts.append(state['preparation_source_reselection'])
    for receipt in receipts:
        if receipt.get('packet_id') != bound[0]:
            raise ValueError('准备阶段换帧记录与输入证据不一致，已停止展示。')
        allowed = {target['timestamp'], *(o['timestamp'] for o in target.get('alternative_selections', []))}
        rejected = receipt.get('rejected_frames') or []
        timestamps = [r.get('timestamp') for r in rejected]
        accepted = receipt.get('timestamp')
        if (accepted not in allowed or not 1 <= len(rejected) <= 2
                or timestamps[0] != target['timestamp'] or len(set(timestamps)) != len(timestamps)
                or accepted in timestamps or any(t not in allowed for t in timestamps)
                or any(not r.get('reason') for r in rejected)
                or bound[1] not in {target['timestamp'],accepted}):
            raise ValueError('准备阶段换帧记录与冻结候选不一致，已停止展示。')
        bound = (bound[0],accepted)
    for event in events:
        change = event['payload']
        if event['kind'] == 'source_view_reselected' and change.get('timestamp') != change.get('from_timestamp'):
            if (change.get('from_packet_id'), change.get('from_timestamp')) != bound:
                raise ValueError('换帧记录与原任务身份不一致，已停止展示。')
            bound = (change.get('packet_id'), change.get('timestamp'))
    if ((item_packet_id is not _NO_ITEM_PACKET and packet['packet_id'] != item_packet_id) or state['packet_id'] != packet['packet_id']
            or (packet['packet_id'], packet['frame_timestamp_micros']) != bound
            or (packet['context'], packet['frame_timestamp_micros'], packet['target_object_id'])
            != (target['context'], bound[1], target['object_id'])
            or target.get('category',packet['target']['category']) != packet['target']['category']):
        raise ValueError('文本、任务物体与场景身份不一致，已停止展示。')
    return {'passed':True,'packet_id':packet['packet_id'],'frame_timestamp_micros':packet['frame_timestamp_micros'],'scope':'sample metadata identity; export still verifies native assets'}
