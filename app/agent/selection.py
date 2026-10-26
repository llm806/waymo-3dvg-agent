"""Deterministic candidate ordering and reuse protection for annotation batches."""
from collections import defaultdict
from itertools import zip_longest


def unavailable_target_keys(runs, method_version, *, include_prior_attempts=False, retries=None, preparation_attempts=None):
    attempted, processed = set(), set()
    usage = defaultdict(int)
    for state in runs:
        if state.get('provider_scope') != 'real-model':
            continue
        packet = state.get('current_packet') or {}
        key = (packet.get('context', state.get('context')), packet.get('target_object_id'))
        usage[key] += state.get('model_calls_used', 0)
        authorized = key in (retries or {})
        # A failed draft is not a completed annotation. Only an explicit unused
        # authorization releases its historical selection lock, once.
        protected = usable_description(state) or state.get('status') in {'running','paused'}
        if (not authorized or protected) and (include_prior_attempts or state.get('version') in {None, method_version}):
            attempted.add(key)
        if (not authorized or protected) and (state.get('final_expression') or any(a.get('generated') for a in state.get('attempts', []))):
            processed.add(key)
    attempted |= set(preparation_attempts or ()) - (retries or {}).keys()
    exhausted = {key for key, spent in usage.items() if spent > (retries or {}).get(key,{}).get('call_limit',12)-4}
    return attempted | processed | exhausted, processed


def candidate_priority(target):
    selection = target.get('selection') or {}
    return (-selection.get('image_selection_score', 0),
            selection.get('range_xy_m', float('inf')), target['ordinal'])


def diverse_candidate_order(targets):
    """Round robin source cameras; prefer larger image evidence within a camera.

    This only orders metadata candidates. RGB visibility and uniqueness remain
    mandatory checks, and the same track is never offered twice.
    """
    groups, seen = defaultdict(list), set()
    for target in sorted(targets, key=candidate_priority):
        key = (target['context'], target['object_id'])
        if key in seen:
            continue
        seen.add(key)
        groups[target.get('source_view', 'unrecorded')].append(target)
    return [target for row in zip_longest(*groups.values()) for target in row if target is not None]


def attempted_item(item):
    if item.get('run_id') or item.get('packet_id'):
        return True
    return item['status'] in {'preparing', 'excluded', 'failed', 'abstained', 'needs_evidence'}


def passed_description(state):
    return (state.get('status') in {'calibration_hold', 'auto_pass'}
            and state.get('diagnostic_pass') is True
            and bool(state.get('final_expression')))


def usable_description(state):
    from app.quality.sample_review import review_status, manually_approved
    return (passed_description(state) or manually_approved(state)) and review_status(state) not in {'rejected', 'discarded'}
