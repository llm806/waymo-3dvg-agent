"""Task-scoped research metrics with explicit denominators and missingness."""
from collections import Counter

from app.agent.selection import attempted_item, usable_description, passed_description
from app.quality.sample_review import manually_approved
from app.quality.evaluation import rate, summarize


def distribution(values):
    values = sorted(v for v in values if v is not None)

    def percentile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)

    return {'count': len(values), 'mean': sum(values) / len(values) if values else None,
            'p50': percentile(.5) if values else None, 'p95': percentile(.95) if values else None}


def task_metrics(batch, states, cost_states, calls, audits):
    """One track per queue item; exclude reuse from new-label speed/cost yields."""
    by_run = {s['run_id']: s for s in states}
    eligible = [i for i in batch['items'] if i['target']['status'] == 'selected']
    attempted = [i for i in eligible if attempted_item(i)]
    settled = [i for i in attempted if i['status'] not in {'pending', 'preparing', 'running'}]
    text = lambda i: bool(i.get('expression'))
    passed = lambda i: usable_description(by_run.get(i.get('run_id'), {}))
    accepted = [i for i in eligible if passed(i)]
    produced = [i for i in accepted if not i['reused'] and not by_run[i['run_id']].get('revalidation_only')]
    new_settled = [i for i in settled if not i['reused']]
    timed = lambda i: i.get('timing', {}).get('active_seconds') if i.get('timing', {}).get('complete') else None
    timing = batch['timing']
    active = timing['active_seconds'] if timing['complete'] else None
    wall = timing['wall_seconds'] if timing['finished_at'] is not None else None
    catalog = next((e['payload'].get('catalog_summary', {}) for e in batch['events'] if e['kind'] == 'batch_created'), {})
    n_pass, n_new, n_text = len(accepted), len(produced), sum(text(i) for i in attempted)
    n_automatic=sum(passed_description(by_run.get(i.get('run_id'),{})) for i in accepted)
    n_human=sum(manually_approved(by_run.get(i.get('run_id'),{})) for i in accepted)
    evaluation = summarize(cost_states)
    usage = [c.get('usage') or {} for c in calls]
    tokens_complete = (len(calls) == batch['model_calls_reserved'] and all(
        c.get('remote_outcome') == 'not_sent' or type(u.get('total_tokens')) is int for c, u in zip(calls, usage)))
    seconds_complete = len(calls) == batch['model_calls_reserved'] and all(c.get('elapsed_seconds') is not None for c in calls)
    failures = Counter(i.get('reason') or 'unrecorded' for i in settled if not passed(i))
    strata = []
    for category in sorted({i['target'].get('category', 'unknown') for i in eligible}):
        group = [i for i in eligible if i['target'].get('category', 'unknown') == category]
        tried = [i for i in attempted if i['target'].get('category', 'unknown') == category]
        completed = [i for i in settled if i['target'].get('category', 'unknown') == category]
        strata.append({'category': category, 'pool': len(group), 'attempted': len(tried),
            'settled': len(completed), 'passed': sum(passed(i) for i in group),
            'pass_per_settled': rate(sum(passed(i) for i in completed), len(completed))})
    return {'schema_version': 'task-metrics-v1', 'mode': 'preparation_only' if batch['spec']['prepare_only'] else 'annotation',
        'timing': timing, 'catalog_snapshot': catalog or None,
        'counts': {'pool': len(eligible), 'attempted': len(attempted), 'settled': len(settled),
                   'texts': n_text, 'passed': n_pass, 'new_passed': n_new,
                   'automatic_passed':n_automatic,'human_approved':n_human,
                   'reused': sum(bool(i['reused']) for i in attempted),
                   'unattempted': len(eligible) - len(attempted)},
        'coverage': {'pool_utilization': rate(n_pass, len(eligible)),
                     'dataset_utilization': rate(n_pass, catalog.get('tracks', 0)),
                     'eligible_catalog_utilization': rate(n_pass, catalog.get('selected', 0)),
                     'attempt_coverage': rate(len(attempted), len(eligible)),
                     'pass_per_settled': rate(sum(passed(i) for i in settled), len(settled))},
        'candidate_coverage': rate(n_text, len(attempted) if not batch['spec']['prepare_only'] else 0),
        'tool_localization_diagnostic': rate(n_automatic, n_text),
        'speed': {'item_active_seconds': distribution(timed(i) for i in new_settled),
                  'passed_item_active_seconds': distribution(timed(i) for i in produced),
                  'mean_active_seconds_per_new_pass': active / n_new if active is not None and n_new else None,
                  'new_passes_per_active_hour': n_new * 3600 / active if active else None,
                  'new_passes_per_wall_hour': n_new * 3600 / wall if wall else None},
        'cost': {'model_calls': len(calls), 'reserved_model_calls': batch['model_calls_reserved'],
                 'tokens': sum(u.get('total_tokens', 0) or 0 for u in usage), 'tokens_complete': tokens_complete,
                 'model_seconds': sum(c.get('elapsed_seconds', 0) or 0 for c in calls), 'model_seconds_complete': seconds_complete,
                 'calls_per_processed_target': rate(batch['model_calls_reserved'], len(attempted)),
                 'calls_per_new_pass': rate(batch['model_calls_reserved'], n_new),
                 'tokens_per_new_pass': sum(u.get('total_tokens', 0) or 0 for u in usage) / n_new if tokens_complete and n_new else None,
                 'outcomes': dict(Counter(c['status'] for c in calls)),
                 'successful_responses': sum(c['status'] == 'completed' for c in calls),
                 'known_not_sent': sum(c.get('remote_outcome') == 'not_sent' for c in calls),
                 'calls_with_usage': sum(bool(u) for u in usage), 'monetary_cost': None},
        'quality': {'scope': 'Per task-owned execution record, including superseded runs; revalidation-only records have no first draft.',
                    'first_draft_pass': evaluation['first_oracle_pass_per_checked_draft'],
                    'first_draft_incomplete': evaluation['first_draft_incomplete'],
                    'repair_success': evaluation['repair_success_per_initial_failure'],
                    'counterfactuals': evaluation['counterfactuals']},
        'independent_acceptance': audits['acceptance'], 'independent_audits': audits['cohorts'],
        'failure_reasons': [{'reason': key, 'count': count} for key, count in sorted(failures.items(), key=lambda x: (-x[1], x[0]))],
        'categories': strata,
        'denominators': {'all_tracks': len(batch['items']), 'eligible': len(eligible), 'texts': n_text, 'new_runs': len(cost_states)},
        'selection_progress': batch['selection_progress'],
        'definitions': {'unit': 'Unique object track in this task; multiple frames/cameras are not additional samples.',
            'dataset_utilization': 'Automatic-check-passed tracks in this task / all tracks in the frozen local catalog.',
            'pool_utilization': 'Automatic-check-passed tracks / eligible tracks in this task pool.',
            'pass_per_settled': 'Automatic-check-passed tracks / settled attempted tracks, including preparation failures.',
            'item_latency': 'Sum of processing intervals for a track, including preparation, retries and checks; excludes pauses.',
            'percentiles': 'Linear interpolation at (n-1)*q; only fully recorded, settled, non-reused items.',
            'mean_active_seconds_per_new_pass': 'Entire task active time / newly produced automatic-check-passed tracks; includes unsuccessful work.',
            'cost': 'All task-owned runs, including superseded runs; reused labels and post-hoc independent audits excluded.',
            'quality': 'First-draft, repair and counterfactual rates use execution-record denominators; coverage and throughput count unique tracks.',
            'scope': 'This task only; continuation jobs have separate reports. Creation waiting time excluded. Missing timing/usage remains unknown.'},
        'limitations': 'Automatic checks use native truth candidates and are not annotation accuracy. Independent acceptance requires sealed blind audits; local catalog utilization is not a population estimate.'}
