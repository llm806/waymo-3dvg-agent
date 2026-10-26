"""Read durable annotation timings without starting work or sending requests."""
import argparse
import json
import sqlite3
from statistics import median
from collections import Counter, defaultdict
from pathlib import Path


def profile_performance(paths):
    """Read only recorded intervals; partial last lines and crashes stay visible."""
    sessions, phases, measurements = {}, defaultdict(list), {}
    missing, unreadable, malformed = 0, 0, 0
    for path in paths:
        try:
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    if not isinstance(record, dict) or record.get('protocol') != 'annotation-performance-v1':
                        continue
                    key = (str(path), record['session'])
                    session = sessions.setdefault(key, {'started': False, 'finished': False, 'phase': None})
                    event = record['event']
                    if event == 'session_started':
                        session['started'] = True
                    elif event == 'phase_started':
                        session['phase'] = record['phase']
                    elif event == 'phase_finished':
                        phases[record['phase']].append(record)
                        session['phase'] = None
                    elif event == 'session_finished':
                        session.update(finished=True, seconds=record['seconds'],
                            prior_write_failed=record.get('prior_write_failed', False))
                        for name, value in record.get('metrics', {}).items():
                            total = measurements.setdefault(name, {'count': 0, 'seconds': 0.0,
                                'thread_cpu_seconds': 0.0, 'max_seconds': 0.0, 'errors': 0})
                            for field in ('count', 'seconds', 'thread_cpu_seconds', 'errors'):
                                total[field] += value[field]
                            total['max_seconds'] = max(total['max_seconds'], value['max_seconds'])
        except FileNotFoundError:
            missing += 1
        except OSError:
            unreadable += 1

    def totals(records):
        seconds = sorted(r['seconds'] for r in records)
        def percentile(q):
            index = (len(seconds) - 1) * q
            lower = int(index)
            return seconds[lower] + (seconds[min(lower + 1, len(seconds)-1)] - seconds[lower]) * (index-lower)
        return {'count': len(records), 'seconds': sum(seconds),
            'thread_cpu_seconds': sum(r['thread_cpu_seconds'] for r in records),
            'p50_seconds': percentile(.5), 'p95_seconds': percentile(.95), 'max_seconds': seconds[-1]}

    complete = bool(sessions) and all(s['started'] and s['finished'] and not s.get('prior_write_failed')
        for s in sessions.values()) and not (missing or unreadable or malformed)
    return {'status': 'complete' if complete else 'incomplete' if sessions else 'unavailable',
        'sessions': len(sessions), 'incomplete_sessions': sum(not s['finished'] for s in sessions.values()),
        'open_phases': [s['phase'] for s in sessions.values() if not s['finished'] and s['phase']],
        'missing_files': missing, 'unreadable_files': unreadable, 'malformed_lines': malformed,
        'completed_session_seconds': sum(s['seconds'] for s in sessions.values() if s['finished']) if sessions else None,
        'phases': dict(sorted(((name, totals(records)) for name, records in phases.items()),
            key=lambda item: -item[1]['seconds'])),
        'measurements': dict(sorted(measurements.items(), key=lambda item: -item[1]['seconds'])),
        'scope': 'Recorded phase intervals; measurements overlap phases. Target/alternate-frame/run scopes overlap. '
            'Concurrent worker seconds are not batch wall time. Thread CPU excludes other threads/processes. '
            'Missing logs and unfinished sessions have no inferred duration.'}


def profile_runs(root, run_ids=None, job_ids=None, *, recent=None):
    if recent is not None and (type(recent) is not int or recent<1 or run_ids or job_ids):
        raise ValueError('recent must be positive and cannot be combined with run/job IDs')
    root = Path(root)
    database = root / 'runs.sqlite3'
    if not database.is_file():
        raise ValueError('Annotation database not found')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        # Profiling selected runs must not deserialize every geometric packet
        # in a potentially multi-GB database. Only outcome/timing fields are used.
        table = 'run_summaries' if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_summaries'").fetchone() else 'runs'
        jobs = []
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='batches'").fetchone():
            selected_jobs = (dict.fromkeys(job_ids) if job_ids else
                [row[0] for row in db.execute('SELECT job_id FROM batches')] if not run_ids and recent is None else [])
            for job in selected_jobs:
                row = db.execute('SELECT status FROM batches WHERE job_id=?', (job,)).fetchone()
                if row is None:
                    continue
                items = db.execute('SELECT ordinal,run_id,status,reason FROM batch_items WHERE job_id=? ORDER BY ordinal',
                    (job,)).fetchall()
                from app.infrastructure.batch_timing import summarize_timing
                events = [{'kind':r[0], 'payload':json.loads(r[1]), 'at':r[2]} for r in db.execute(
                    "SELECT kind,payload,at FROM batch_events WHERE job_id=? AND (kind LIKE 'timing_%' "
                    "OR kind IN ('batch_started_or_resumed','batch_completed','batch_blocked')) ORDER BY seq", (job,))]
                timing, _ = summarize_timing(events, status=row[0], executor_active=False)
                jobs.append({'job_id':job, 'status':row[0], 'timing':timing,
                    'item_statuses':dict(Counter(i[2] for i in items)),
                    'item_reasons':dict(Counter(i[3] for i in items if i[3])),
                    'run_ids':list(dict.fromkeys(i[1] for i in items if i[1])),
                    'performance':profile_performance(root / 'batches' / job / str(i[0]) / 'performance.jsonl'
                        for i in items if i[2] not in {'not_needed', 'pending'}
                        or (root / 'batches' / job / str(i[0]) / 'performance.jsonl').exists())})
        if recent is not None:
            states = [json.loads(row[0]) for row in db.execute(f'''SELECT s.snapshot FROM runs r
                JOIN {table} s ON s.run_id=r.run_id ORDER BY r.rowid DESC LIMIT ?''', (recent,))]
        elif run_ids or job_ids:
            selected = dict.fromkeys([*(run_ids or []), *(rid for job in jobs for rid in job['run_ids'])])
            states = [json.loads(row[0]) for run_id in selected
                for row in db.execute(f'SELECT snapshot FROM {table} WHERE run_id=?', (run_id,))]
        else:
            states = [json.loads(row[0]) for row in db.execute(f'SELECT snapshot FROM {table}')]
    summaries = []
    for state in states:
        phases = defaultdict(lambda: {'calls': 0, 'seconds': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0,
            'first_content_seconds': [], 'response_seconds':[], 'image_counts':[],
            'cached_prompt_tokens':0,'cache_reporting_calls':0,'cache_reporting_prompt_tokens':0,
            'post_first_content_seconds':0.0,'stream_completion_tokens':0,
            'stream_log_seconds': 0.0, 'statuses':Counter(), 'missing_elapsed_calls':0})
        for path in sorted((root / state['run_id'] / 'model').glob('*/result.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            phase = phases[path.parent.name.split('-', 1)[1]]
            phase['calls'] += 1
            phase['statuses'][record.get('status', 'unknown')] += 1
            phase['missing_elapsed_calls'] += int(record.get('elapsed_seconds') is None)
            phase['seconds'] += record.get('elapsed_seconds') or 0
            phase['stream_log_seconds'] += record.get('stream_log_seconds') or 0
            usage = record.get('usage') or {}
            if record.get('elapsed_seconds') is not None:
                phase['response_seconds'].append(record['elapsed_seconds'])
            images=(record.get('request_shape') or {}).get('image_count')
            if images is not None:
                phase['image_counts'].append(images)
            cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if cached is not None:
                phase['cached_prompt_tokens']+=cached
                phase['cache_reporting_calls']+=1
                phase['cache_reporting_prompt_tokens']+=usage.get('prompt_tokens') or 0
            for field in ('prompt_tokens', 'completion_tokens'):
                phase[field] += usage.get(field, 0) or 0
            if record.get('first_content_seconds') is not None:
                phase['first_content_seconds'].append(record['first_content_seconds'])
                if record.get('elapsed_seconds') is not None and record['status']=='completed':
                    phase['post_first_content_seconds']+=max(0,record['elapsed_seconds']-record['first_content_seconds'])
                    phase['stream_completion_tokens']+=usage.get('completion_tokens') or 0
        total = sum(p['seconds'] for p in phases.values())
        for value in phases.values():
            times = value.pop('first_content_seconds')
            value['mean_first_content_seconds'] = round(sum(times) / len(times), 3) if times else None
            responses=value.pop('response_seconds');images=value.pop('image_counts')
            value['p50_response_seconds']=round(median(responses),3) if responses else None
            value['p50_image_count']=median(images) if images else None
            value['image_reporting_calls']=len(images)
            prompt=value.pop('cache_reporting_prompt_tokens')
            value['reported_prompt_cache_fraction']=round(value['cached_prompt_tokens']/prompt,4) if prompt else None
            stream_seconds=value.pop('post_first_content_seconds');tokens=value.pop('stream_completion_tokens')
            value['observed_stream_tokens_per_second']=round(tokens/stream_seconds,3) if stream_seconds else None
            value['seconds'] = round(value['seconds'], 3)
            value['stream_log_seconds'] = round(value['stream_log_seconds'], 3)
        summaries.append({'run_id': state['run_id'], 'kind': state.get('kind', 'annotation'),
            'status': state['status'], 'model_calls_reserved': state.get('model_calls_used', 0),
            'provider_scope':state.get('provider_scope'),'diagnostic_pass':state.get('diagnostic_pass'),
            'active_seconds': state.get('elapsed_active_seconds'), 'recorded_provider_seconds': round(total, 3),
            'stop_reason':state.get('stop_reason'),
            'performance':profile_performance([root / state['run_id'] / 'performance.jsonl']),
            'phases': dict(sorted(phases.items(), key=lambda item: -item[1]['seconds']))})
    return {'scope': 'Read-only recorded timings; no requests, source-data loading or inferred missing durations. '
        'Provider elapsed/observed stream rate include transport and local logging; phase/measurement totals overlap. '
        'Unreported cache/image fields are unknown, not zero. Recent selects creation order, not paired throughput evidence.',
        'recent_limit':recent,
        'requested_missing_run_ids':sorted(set(run_ids or []) - {s['run_id'] for s in states}),
        'requested_missing_job_ids':sorted(set(job_ids or []) - {j['job_id'] for j in jobs}),
        'runs': summaries, 'batches': jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runtime/annotation'))
    parser.add_argument('--run', action='append', dest='run_ids')
    parser.add_argument('--job', action='append', dest='job_ids')
    parser.add_argument('--recent',type=int,help='Inspect only the latest N created runs; no full batch-history scan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.recent is not None and (args.recent<1 or args.run_ids or args.job_ids):
        parser.error('--recent must be positive and cannot be combined with --run/--job')
    result = json.dumps(profile_runs(args.root, args.run_ids, args.job_ids,recent=args.recent), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
