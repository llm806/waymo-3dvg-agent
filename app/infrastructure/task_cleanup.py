"""Reference-aware task deletion, with recoverable filesystem transactions.

Only app-owned runtime assets are eligible. Selected task headers and exclusive
assets are removed; surviving dependents own necessary accounting and run refs.
"""

import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from pathlib import Path
from uuid import uuid4

from app.infrastructure.artifacts import atomic_json
from app.infrastructure.diagnostics import diagnostic_maintenance
from app.infrastructure.storage import RunBusy
from app.infrastructure.task_lineage import read_lineage, lineage_family

IDENTIFIERS = re.compile(r"(?<![a-f0-9])(?:[a-f0-9]{64}|[a-f0-9]{24})(?![a-f0-9])")
TABLES = ('batches', 'batch_items', 'batch_events', 'runs', 'events', 'controls',
          'audit_cohorts', 'audit_localizations', 'audit_answers', 'audit_answer_events', 'label_releases')


def ids(value):
    # Walk JSON values instead of serializing/scanning large numeric geometry
    # inventories. References may also occur in dict keys and embedded paths.
    found, pending = set(), [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if len(item) >= 24:
                found.update(IDENTIFIERS.findall(item))
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, int) and abs(item) >= 10 ** 23:
            found.update(IDENTIFIERS.findall(str(item)))
    return found


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_key(value):
    return value.get('context'), value.get('object_id', value.get('target_object_id')), value.get('timestamp', value.get('frame_timestamp_micros'))


def run_parents(state):
    return {parent for parent in (state.get('parent_run_id'), state.get('revalidation_source_run_id'),
            state.get('development_recovery_source_run_id'),
            state.get('development_validation_source_run_id'),
            state.get('current_packet', {}).get('task', {}).get('manual_review_feedback', {}).get('source_run_id'),
            (state.get('recovery') or {}).get('parent_run_id'),
            *state.get('observation_source_run_ids',[])) if parent}


class TaskCleaner:
    _manifest_cache = {}
    _confirmation_scopes = {}
    def __init__(self, runs, root, progress=None):
        self.runs, self.root = runs, Path(os.path.abspath(root))
        self.runtime = self.root / 'runtime'
        self.annotation = Path(os.path.abspath(runs.path.parent))
        self.journals = self.annotation / '.task-delete'
        self.safe(self.annotation)
        self.progress = progress
        self.cache_updates = []
        self.files_checked = 0
        self.asset_updates = []

    def report(self, **values):
        if self.progress:
            self.progress(**values)

    def safe(self, path):
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.runtime) or path == self.runtime:
            raise ValueError('删除路径必须位于项目运行目录内')
        for node in (path, *path.parents):
            if node == self.root:
                break
            try:
                attributes = node.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError('删除范围包含符号链接或目录联接，已停止以避免误删')
        # abspath removes '..'; containment plus lstat of every component rules
        # out escapes without repeating Windows realpath for every file.
        return path

    def tree(self, path):
        self.safe(path)
        pending = [path] if path.is_dir() else []
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    attributes = entry.stat(follow_symlinks=False)
                    if (stat.S_ISLNK(attributes.st_mode) or
                            getattr(attributes, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                        raise ValueError('删除范围包含符号链接或目录联接，已停止以避免误删')
                    if stat.S_ISDIR(attributes.st_mode):
                        pending.append(entry.path)
                    self.files_checked += 1
                    if self.files_checked % 200 == 0:
                        self.report(files_checked=self.files_checked)

    @staticmethod
    def run_metadata(run_id, raw, summary=None):
        decoded = isinstance(summary,dict)
        state = summary if decoded else json.loads(summary if summary is not None else raw)
        # IDs are ASCII, so native JSON serialization preserves them verbatim.
        # Escaped Unicode/control characters can change word boundaries; fall
        # back to the decoded walk for those uncommon imported/feedback rows.
        references = (ids(state if decoded else json.loads(raw)) if any(escape in raw for escape in ('\\u','\\b','\\f'))
                      else set(IDENTIFIERS.findall(raw)))
        return {'run_id': run_id, 'batch_id': state.get('batch_id'),
                'kind': state.get('kind'), 'parent_run_id': state.get('parent_run_id'),
                'observation_sources':state.get('observation_source_run_ids',[]),
                'parents': sorted(run_parents(state)), 'references': sorted(references),
                'model_calls_used': state.get('model_calls_used', 0),
                'snapshot_sha256': hashlib.sha256(raw.encode()).hexdigest()}

    def flush_cache(self):
        # Derived rows only, generation-checked and written in small independent
        # transactions. Never upgrade the long planning read into a writer.
        for start in range(0, len(self.cache_updates), 100):
            with self.runs.connect() as db:
                db.executemany('UPDATE run_cleanup_cache SET metadata=? WHERE run_id=? AND generation=?',
                               self.cache_updates[start:start+100])
        self.cache_updates.clear()
        for start in range(0,len(self.asset_updates),100):
            with self.runs.connect() as db:
                db.executemany('''INSERT INTO asset_cleanup_cache VALUES(?,?,?) ON CONFLICT(path)
                    DO UPDATE SET signature=excluded.signature,metadata=excluded.metadata''',self.asset_updates[start:start+100])
        self.asset_updates.clear()

    def pool(self, name, manifest):
        folder = self.safe(self.runtime / name)
        result = {}
        with self.runs.connect() as db:
            disk_cache = {row['path']:(tuple(json.loads(row['signature'])),json.loads(row['metadata']))
                          for row in db.execute('SELECT * FROM asset_cleanup_cache WHERE path LIKE ?',
                                                (str(folder)+os.sep+'%',))}
        if folder.exists():
            children = list(folder.iterdir())
            for index, child in enumerate(children, 1):
                if re.fullmatch('[a-f0-9]{24}', child.name):
                    path = self.safe(child / manifest)
                    if not path.is_file():
                        path = self.safe(child / 'preparation.json')
                    if path.is_file():
                        attributes = path.stat()
                        signature = (attributes.st_size, attributes.st_mtime_ns, attributes.st_ctime_ns)
                        cached = self._manifest_cache.get(str(path)) or disk_cache.get(str(path))
                        if cached and cached[0] == signature:
                            result[child.name] = cached[1]
                            self.report(stage='assets',message='正在核对缓存资源',completed=index,total=len(children))
                            continue
                        raw = path.read_text(encoding='utf-8')
                        value = json.loads(raw)
                        # Only identity and transitive references participate
                        # in deletion. Do not retain huge predicate inventories.
                        result[child.name] = {'context': value.get('context'),
                            'target_object_id': value.get('target_object_id'),
                            'frame_timestamp_micros': value.get('frame_timestamp_micros'),
                            'references': sorted(ids(value)),
                            'manifest_sha256': hashlib.sha256(raw.encode()).hexdigest()}
                        if len(self._manifest_cache) > 30000:
                            self._manifest_cache.clear()
                        self._manifest_cache[str(path)] = (signature, result[child.name])
                        self.asset_updates.append((str(path),json.dumps(signature),json.dumps(result[child.name])))
                self.report(stage='assets', message='正在核对场景资源' if name == 'scenes' else '正在核对图像证据',
                            completed=index,total=len(children))
        return result

    def guard(self, db, requested):
        """Reject active selections; unrelated workers may keep their leases."""
        lease = db.execute('SELECT * FROM executor WHERE singleton=1').fetchone()
        active_jobs = {row['job_id'] for row in db.execute('SELECT job_id FROM batches WHERE owner=?',
            (lease['owner'],))} if lease and lease['lease_until'] > time.time() else set()
        active_runs = [dict(row) for row in db.execute("""SELECT run_id,
            json_extract(snapshot,'$.batch_id') AS job_id,
            json_extract(snapshot,'$.current_packet.context') AS context
            FROM runs WHERE lease_until>? AND owner IS NOT NULL""", (time.time(),))]
        active_jobs |= {row['job_id'] for row in active_runs if row['job_id']}
        for row in active_runs:
            active_jobs |= {item['job_id'] for item in db.execute(
                'SELECT job_id FROM batch_items WHERE run_id=? AND reused=0', (row['run_id'],))}
        if active_jobs & set(requested):
            raise RunBusy('所选任务正在执行，请先暂停该任务并等待保存后再删除。')
        return active_jobs, active_runs

    def plan(self, db, requested, return_to_pool=None, preserve_shared=None):
        self.report(stage='references', message='正在核对任务与共享引用', completed=0, total=0)
        active_jobs, active_runs = self.guard(db, requested)
        active_run_ids = {row['run_id'] for row in active_runs}
        defer_shared = bool(active_jobs or active_runs or preserve_shared)
        tables = {r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = {name: [dict(r) for r in db.execute('SELECT * FROM ' + name + ' ORDER BY rowid')]
                for name in TABLES if name in tables and name not in {'runs', 'events', 'batch_events'}}
        jobs = {r['job_id']: r for r in rows['batches']}
        revision_methods = {}
        for row in db.execute("SELECT job_id,payload FROM batch_events WHERE kind='execution_revision'"):
            payload = json.loads(row['payload'])
            revision_methods.setdefault(row['job_id'],set()).update(
                payload[key]['method_sha256'] for key in ('previous','current'))
        if set(requested) - jobs.keys():
            raise ValueError('所选任务已不存在，请刷新任务列表。')
        selected = set(requested)
        if return_to_pool is None:
            return_to_pool = not any(jobs[job]['status'] == 'completed' for job in selected)
        if type(return_to_pool) is not bool:
            raise ValueError('请选择是否将已处理物体重新加入候选池。')
        _, retained, parents = read_lineage(db)
        selected_family = set().union(*(lineage_family(job, parents)[1] for job in selected))
        items = [{**r, 'target': json.loads(r['target'])} for r in rows['batch_items']]
        own_items = [r for r in items if r['job_id'] in selected]
        other_items = [r for r in items if r['job_id'] not in selected]
        active_contexts = {item['target'].get('context') for item in other_items if item['job_id'] in active_jobs}
        active_contexts |= {row['context'] for row in active_runs}
        active_contexts.discard(None)
        # An unused retry references its source; it does not own that source.
        # A live surviving source task already protects its entire ledger/assets.
        external_retry_sources = set()
        for item in own_items:
            source = item['target'].get('retry_authorization',{}).get('source_run_id')
            if source:
                owner = db.execute("SELECT json_extract(snapshot,'$.batch_id') FROM run_summaries WHERE run_id=?",(source,)).fetchone()
                if owner and owner[0] in jobs.keys() - selected:
                    external_retry_sources.add(source)
        has_evidence = (bool(selected_family & retained.keys())
            or any(r['run_id'] or r['packet_id'] or
                r['target'].get('retry_authorization',{}).get('source_run_id') not in external_retry_sources | {None}
                for r in own_items)
            or any(db.execute("SELECT 1 FROM runs WHERE json_extract(snapshot,'$.batch_id')=? LIMIT 1",
                (job,)).fetchone() for job in selected))
        # A pending candidate does not own a reusable scene cache. Deleting an
        # unprocessed task needs no global artifact scan. The last-task sweep
        # still inspects legacy orphan recovery chains, as before.
        inspect_assets = has_evidence or not (jobs.keys() - selected)
        metadata_only = not inspect_assets
        # Do not turn removal of a never-executed queue row into shared GC.
        # Retain all shared methods, reports, logs, scenes and evidence.
        defer_shared = defer_shared or metadata_only
        rows['runs'] = []
        generations = {r['run_id']:r['generation'] for r in db.execute('SELECT run_id,generation FROM run_cleanup_cache')}
        if inspect_assets:
            total = db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
            for index, row in enumerate(db.execute('''SELECT r.run_id,c.generation,c.metadata,
                    CASE WHEN c.metadata IS NULL THEN r.snapshot END AS snapshot,
                    CASE WHEN c.metadata IS NULL THEN s.snapshot END AS summary
                    FROM runs r LEFT JOIN run_cleanup_cache c ON c.run_id=r.run_id
                    LEFT JOIN run_summaries s ON s.run_id=r.run_id ORDER BY r.rowid'''), 1):
                state = json.loads(row['metadata']) if row['metadata'] else self.run_metadata(row['run_id'],row['snapshot'],row['summary'])
                rows['runs'].append(state)
                generations[row['run_id']] = row['generation']
                if not row['metadata']:
                    self.cache_updates.append((json.dumps(state),row['run_id'],row['generation']))
                self.report(stage='references', message='正在核对执行记录', completed=index, total=total)
        targets = {target_key(r['target']) for r in own_items}
        other_targets = {target_key(r['target']) for r in other_items}
        frames = {(c, t) for c, _, t in targets}
        other_frames = {(c, t) for c, _, t in other_targets}
        states = {r['run_id']: r for r in rows['runs']}
        candidates = {r['run_id'] for r in own_items if r['run_id']} | {rid for rid, s in states.items() if s.get('batch_id') in selected}
        # A pending retry can own a historical reference before its next child
        # exists. Release that history only after its original task is gone.
        retry_sources = {r['target'].get('retry_authorization',{}).get('source_run_id') for r in own_items}
        candidates |= {rid for rid in retry_sources if rid in states
            and states[rid].get('batch_id') and states[rid]['batch_id'] not in jobs}
        candidates |= {rid for job, record in retained.items() if job in selected_family for rid in record['run_ids'] if rid in states}
        # Remove only selected task headers. Transfer the minimum immutable
        # accounting and run references to surviving members of each family.
        # Full model traces stay in their original run folders, without copying.
        history_updates, protected_history, surviving_related = {}, {}, set()
        for job in jobs.keys() - selected:
            _, family = lineage_family(job, parents)
            history = {ancestor: record for ancestor, record in retained.items() if ancestor in family}
            for ancestor in selected & family:
                linked = {r['run_id'] for r in own_items if r['job_id'] == ancestor and r['run_id']}
                owned = {rid for rid, s in states.items() if s.get('batch_id') == ancestor}
                run_ids = linked | owned
                while True:
                    expanded = run_ids | {rid for rid, s in states.items() if set(s['parents']) & run_ids}
                    if expanded == run_ids:
                        break
                    run_ids = expanded
                counted = owned | {r['run_id'] for r in own_items if r['job_id'] == ancestor and not r['reused'] and r['run_id']}
                spec = json.loads(jobs[ancestor]['spec'])
                history[ancestor] = {'parent_job_id': parents.get(ancestor),
                    'spec': {key: spec.get(key) for key in ('max_model_calls', 'desired_count')},
                    'method_sha256': jobs[ancestor]['method_sha256'],
                    'method_sha256s': sorted(revision_methods.get(ancestor,set()) | {jobs[ancestor]['method_sha256']}),
                    'run_ids': sorted(run_ids),
                    'model_calls_reserved': sum(states[rid].get('model_calls_used', 0) for rid in counted if rid in states)}
            protected_history.update(history)
            if selected & family:
                history_updates[job] = {'tasks': history}
                surviving_related.add(job)
        if not (jobs.keys() - selected):
            # Repair legacy recoveries whose task was already deleted. Ordinary
            # standalone annotations are not orphan recoveries and stay intact.
            candidates |= {rid for rid, s in states.items()
                           if any(parent not in states for parent in s['parents'])}
        while True:
            released_sources={source for rid in candidates for source in states.get(rid,{}).get('observation_sources',[])
                if source in states and states[source].get('batch_id') and states[source]['batch_id'] not in jobs}
            expanded = candidates | released_sources | {rid for rid, s in states.items() if set(s['parents']) & candidates}
            if candidates == expanded:
                break
            candidates = expanded
        cohorts = {r['cohort_id']: {**r, 'spec': json.loads(r['spec']), 'frozen': json.loads(r['frozen']),
                                  'report': json.loads(r['report']) if r['report'] else None} for r in rows.get('audit_cohorts', [])}
        removed_cohorts = {cid for cid, c in cohorts.items() if c['spec'].get('job_id') in selected}
        keep_roots = {r['run_id'] for r in other_items if r['run_id']}
        keep_roots |= {r['target'].get('retry_authorization',{}).get('source_run_id') for r in other_items} & states.keys()
        keep_roots |= active_run_ids
        keep_roots |= {rid for record in protected_history.values() for rid in record['run_ids']}
        for cid, c in cohorts.items():
            if cid not in removed_cohorts:
                keep_roots |= ids(c) & states.keys()
        kept_runs = (states.keys() - candidates) | keep_roots
        while True:
            expanded = kept_runs | {parent for rid in kept_runs if rid in states for parent in states[rid]['parents'] if parent in states}
            if kept_runs == expanded:
                break
            kept_runs = expanded
        removed_runs = candidates - kept_runs
        releases = {r['release_id']: json.loads(r['manifest']) for r in rows.get('label_releases', [])}
        removed_reports = {digest(cohorts[c]['report']) for c in removed_cohorts if cohorts[c]['report']}
        removed_releases = set()
        for rid, release in releases.items():
            linked = bool(set(release.get('run_ids', [])) & removed_runs or ids(release.get('cohort_ids', {})) & removed_cohorts)
            # Older certificates did not store cohort IDs; recognize their bound reports.
            if 'cohort_ids' not in release:
                for name in ('calibration.json', 'validation.json'):
                    path = self.safe(self.runtime / 'releases' / rid / name)
                    if path.exists():
                        report = json.loads(path.read_text(encoding='utf-8'))
                        linked |= digest(report) in removed_reports or report.get('full_report_sha256') in removed_reports
            if linked:
                removed_releases.add(rid)
        protected_values = [protected_history, *other_items, *[s for rid, s in states.items() if rid not in removed_runs],
                            *[c for cid, c in cohorts.items() if cid not in removed_cohorts],
                            *[v for rid, v in releases.items() if rid not in removed_releases]]
        self.report(stage='assets', message='正在核对独占资源', completed=0, total=0)
        # Active workers protect the entire shared cache. Reading every huge
        # packet just to retain all of them adds no deletion safety.
        inspect_shared = inspect_assets and not defer_shared
        packets = self.pool('evidence', 'packet.json') if inspect_shared else {}
        scenes = self.pool('scenes', 'scene.json') if inspect_shared else {}
        own_refs = ids([own_items, [states[r] for r in removed_runs if r in states]])
        protected = ids(protected_values)
        packet_candidates = (own_refs & packets.keys()) | {pid for pid, p in packets.items() if target_key(p) in targets}
        if not (jobs.keys() - selected):
            packet_candidates |= packets.keys()  # Sweep unreferenced shared caches at the last task.
        protected |= {pid for pid, p in packets.items() if target_key(p) in other_targets}
        # The separate converter also publishes shared JSON indexes. Leave
        # those caches/indexes intact while any worker can be publishing them.
        if active_jobs or active_runs:
            protected |= packets.keys() | scenes.keys()
        removed_packets = packet_candidates - protected
        kept_packets = {pid: p for pid, p in packets.items() if pid not in removed_packets}
        protected |= ids(kept_packets)
        scene_candidates = ids([packets[p] for p in removed_packets]) & scenes.keys()
        scene_candidates |= {sid for sid, s in scenes.items() if (s.get('context'), s.get('frame_timestamp_micros')) in frames}
        if not (jobs.keys() - selected):
            scene_candidates |= scenes.keys()
        protected |= {sid for sid, s in scenes.items() if (s.get('context'), s.get('frame_timestamp_micros')) in other_frames}
        removed_scenes = scene_candidates - protected
        methods = {jobs[j]['method_sha256'] for j in selected}
        methods |= set().union(*(revision_methods.get(job,set()) for job in selected))
        methods |= {record['method_sha256'] for job, record in retained.items() if job in selected_family}
        methods |= {method for job, record in retained.items() if job in selected_family for method in record.get('method_sha256s',[])}
        protected |= set().union(*(methods_ for job,methods_ in revision_methods.items() if job not in selected))
        for state in states.values():
            if state['run_id'] not in removed_runs and state.get('batch_id') in selected:
                protected |= revision_methods.get(state['batch_id'],set()) | {jobs[state['batch_id']]['method_sha256']}
        protected |= ids([j for jid, j in jobs.items() if jid not in selected])
        protected |= {jobs[s['batch_id']]['method_sha256'] for rid, s in states.items()
                      if rid not in removed_runs and s.get('batch_id') in jobs}
        removed_methods = methods - protected
        if defer_shared:
            # Shared packets may still point at a frozen method. Without
            # opening that protected pool, conservatively retain its methods.
            removed_methods = set()
        if not inspect_assets:
            removed_methods = {method for method in removed_methods if not db.execute(
                'SELECT 1 FROM runs WHERE instr(snapshot,?)>0 LIMIT 1', (method,)).fetchone()}
        for values, length in ((selected | candidates | removed_cohorts | removed_releases | removed_packets | removed_scenes, 24), (methods, 64)):
            if any(not isinstance(v, str) or not re.fullmatch('[a-f0-9]{' + str(length) + '}', v) for v in values):
                raise ValueError('运行记录包含无效资源 ID，已停止删除。')
        for rid in removed_runs:
            if states.get(rid, {}).get('kind') == 'counterfactual_audit' and not re.fullmatch('[a-f0-9]{24}', str(states[rid].get('parent_run_id', ''))):
                raise ValueError('补检父记录 ID 无效，已停止删除。')
        deleted_ids = selected | removed_runs | removed_cohorts | removed_releases | removed_packets | removed_scenes | removed_methods
        paths = {self.annotation / 'batches' / j for j in selected}
        paths |= {self.runtime / 'evidence' / p for p in removed_packets}
        paths |= {self.runtime / 'scenes' / s for s in removed_scenes}
        paths |= {self.runtime / 'releases' / r for r in removed_releases}
        paths |= {self.annotation / 'methods' / m for m in removed_methods}
        for rid in removed_runs:
            state = states.get(rid, {})
            path = self.annotation / rid
            if state.get('kind') == 'counterfactual_audit':
                path = self.annotation / state['parent_run_id'] / 'audits' / rid
                for name in ('counterfactual-audit.json', 'counterfactual-analysis.json'):
                    report_path = path.parent.parent / name
                    if report_path.exists() and ids(json.loads(report_path.read_text(encoding='utf-8'))) & removed_runs:
                        paths.add(report_path)
            paths.add(path)
        rewrites = {}
        for area, key, removed in (('evidence', 'packets', removed_packets), ('scenes', 'scenes', removed_scenes)):
            path = self.safe(self.runtime / area / 'index.json')
            if removed and path.exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                value[key] = [r for r in value.get(key, []) if not ids(r) & removed]
                value['errors'] = [r for r in value.get('errors', []) if (r.get('context'), r.get('timestamp')) not in frames - other_frames]
                if value[key]:
                    rewrites[path] = json.dumps(value, ensure_ascii=False, indent=2).encode()
                else:
                    paths.add(path)
        if removed_runs and (self.annotation / 'index.json').exists():
            paths.add(self.annotation / 'index.json')  # Derived aggregate, rebuilt from SQLite on reads.
        reports = self.safe(self.runtime / 'reports')
        if reports.exists() and not metadata_only:
            self.tree(reports)
            paths |= {p for p in reports.rglob('*') if p.is_file() and ids(p.read_text(encoding='utf-8', errors='replace')) & deleted_ids}
        logs = self.safe(self.annotation / 'logs')
        orphan_targets = targets - other_targets
        orphan_frames = frames - other_frames
        orphan_pairs = {(c, o) for c, o, _ in targets} - {(c, o) for c, o, _ in other_targets}
        if logs.exists() and not defer_shared:
            for path in logs.iterdir():
                if not path.is_file():
                    continue
                self.safe(path)
                original = path.read_bytes()
                lines = original.decode('utf-8', errors='replace').splitlines(keepends=True)
                kept = []
                for line in lines:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        record = {}
                    references = ids(line)
                    if (record.get('job_id') in active_jobs or record.get('context') in active_contexts
                            or references & active_run_ids):
                        kept.append(line)
                        continue
                    if references & deleted_ids:
                        continue
                    if target_key(record) in orphan_targets or (record.get('context'), record.get('timestamp')) in orphan_frames or (record.get('context'), record.get('object_id')) in orphan_pairs:
                        continue
                    kept.append(line)
                if len(kept) != len(lines):
                    rewrites[path] = ''.join(kept).encode()
            if inspect_assets and not active_jobs and not active_runs and not kept_packets and not (scenes.keys() - removed_scenes):
                paths.add(logs / 'preparation-validation')
        paths = {self.safe(p) for p in paths if p.exists()}
        # Deleting an ancestor already covers every nested audit/file.
        paths = {p for p in paths if not any(parent in paths for parent in p.parents)}
        rewrites = {self.safe(p): v for p, v in rewrites.items() if not any(parent in paths for parent in (p, *p.parents))}
        path_list = sorted(paths | rewrites.keys())
        for index,path in enumerate(path_list,1):
            self.report(stage='paths',message='正在检查待删文件与目录',completed=index-1,total=len(path_list))
            self.tree(path)
            self.report(completed=index,total=len(path_list))
        summary = {'tasks': len(selected), 'following_tasks': 0, 'retained_tasks': len(surviving_related), 'runs': len(removed_runs),
                   'model_calls': sum(states[r].get('model_calls_used', 0) for r in removed_runs if r in states),
                   'cohorts': len(removed_cohorts), 'releases': len(removed_releases),
                   'evidence': len(removed_packets), 'scenes': len(removed_scenes)}
        public = {'job_ids': sorted(selected), 'jobs': [{'job_id': j, 'status': jobs[j]['status']} for j in sorted(selected)],
                  'summary': summary, 'retained_shared': {'runs': len(candidates & kept_runs),
                   'evidence': len(packet_candidates & protected), 'scenes': len(scene_candidates & protected)}}
        removal = {'batches': selected, 'runs': removed_runs, 'audit_cohorts': removed_cohorts,
                   'label_releases': removed_releases}
        revisions = {name: [tuple(db.execute('SELECT COUNT(*),MAX(rowid) FROM ' + name +
            ' WHERE ' + column + '=?', (key,)).fetchone()) for key in sorted(values)]
            for name, column, values in (('batch_events', 'job_id', selected), ('events', 'run_id', removed_runs))}
        # Bind frozen selected data and the actual removal/transfer scope, not
        # unrelated snapshots, global event counts or append-only log contents.
        selected_run_ids = {item['run_id'] for item in own_items if item['run_id']}
        public['token'] = digest([defer_shared,[row for row in rows['batches'] if row['job_id'] in selected], own_items,
            [state for state in rows['runs'] if state['batch_id'] in selected or state['run_id'] in selected_run_ids],
            [states[rid] for rid in sorted(removed_runs)], revisions,
            [cohorts[cid] for cid in sorted(removed_cohorts)],
            [releases[rid] for rid in sorted(removed_releases)],
            {pid: packets[pid] for pid in sorted(removed_packets | (own_refs & packets.keys()))},
            {sid: scenes[sid] for sid in sorted(removed_scenes | (own_refs & scenes.keys()))},
            {key: sorted(value) for key, value in removal.items()}, history_updates,
            sorted(str(path.relative_to(self.runtime)) for path in paths if path != self.annotation/'index.json')])
        # Pending/not-needed catalog rows were never processed by this task.
        # Preserve only native track identities; no text or large assets needed.
        from app.agent.selection import attempted_item
        pool_keys = {(r['target']['context'], r['target']['object_id']) for r in own_items
                     if r['target'].get('status') == 'selected' and attempted_item(r)}
        tokens = {'return': digest([public['token'], 'return_to_pool', True]),
                  'exclude': digest([public['token'], 'return_to_pool', False])}
        public.update(return_to_pool=return_to_pool, processed_tracks=len(pool_keys), candidate_tokens=tokens,
                      shared_cleanup_deferred=defer_shared)
        public['token'] = tokens['return' if return_to_pool else 'exclude']
        return public, paths, rewrites, {'batches': selected, 'runs': removed_runs, 'audit_cohorts': removed_cohorts,
                                       'label_releases': removed_releases, 'history_updates': history_updates,
                                       'excluded_tracks': sorted(pool_keys) if not return_to_pool else [],
                                       'log_policy': (deleted_ids,orphan_targets,orphan_frames,orphan_pairs),
                                       'guard': {'generations': generations, 'states': states,
                                           'strict_runs': selected_run_ids | removed_runs | {rid for rid,s in states.items() if s['batch_id'] in selected},
                                           'rows': rows, 'deleted_ids': deleted_ids,
                                           'revisions': revisions,
                                           'shared_removed': bool(removed_packets or removed_scenes),
                                           'removed_frames': {(s['context'],s['frame_timestamp_micros']) for sid,s in scenes.items() if sid in removed_scenes}}}

    def validate_commit(self, db, requested, removal):
        """Check only changed references under the writer; no directory/JSON I/O."""
        active_jobs, active_runs = self.guard(db,requested)
        frozen = removal['guard']
        invalid = '任务或关联数据已变化，请重新查看删除范围后确认。'
        if ({r['run_id'] for r in active_runs} & removal['runs']
                or (active_jobs or active_runs) and frozen['shared_removed']):
            raise ValueError(invalid)
        for table,column in (('batches','job_id'),('batch_items','job_id')):
            before = [r for r in frozen['rows'][table] if r['job_id'] in removal['batches']]
            after = [dict(r) for job in sorted(removal['batches']) for r in db.execute(
                'SELECT * FROM '+table+' WHERE '+column+'=? ORDER BY rowid',(job,))]
            if sorted(before,key=lambda r:json.dumps(r,sort_keys=True)) != sorted(after,key=lambda r:json.dumps(r,sort_keys=True)):
                raise ValueError(invalid)
        before_jobs = {r['job_id']:r for r in frozen['rows']['batches']}
        for row in db.execute('SELECT * FROM batches'):
            if row['job_id'] not in removal['batches'] and (ids(dict(row))-ids(before_jobs.get(row['job_id'],{}))) & frozen['deleted_ids']:
                raise ValueError(invalid)
        now = {r['run_id']:r['generation'] for r in db.execute('SELECT run_id,generation FROM run_cleanup_cache')}
        for run in frozen['strict_runs']:
            if now.get(run) != frozen['generations'].get(run):
                raise ValueError(invalid)
        for run,generation in now.items():
            if generation == frozen['generations'].get(run):
                continue
            row = db.execute('SELECT snapshot FROM runs WHERE run_id=?',(run,)).fetchone()
            if row:
                current = self.run_metadata(run,row['snapshot'])
                previous = set(frozen['states'].get(run,{}).get('references',[]))
                if (set(current['references']) - previous) & frozen['deleted_ids']:
                    raise ValueError(invalid)
        # New run/packet links or frame selections in surviving tasks must not
        # acquire a resource that is about to be removed. Status/lease/progress
        # changes unrelated to this scope are deliberately ignored.
        for row in db.execute('''SELECT job_id,run_id,packet_id,
                json_extract(target,'$.context') AS context,
                json_extract(target,'$.timestamp') AS timestamp FROM batch_items'''):
            if row['job_id'] in removal['batches']:
                continue
            if ({row['run_id'],row['packet_id']} & (frozen['deleted_ids'] - removal['batches'])
                    or (row['context'],row['timestamp']) in frozen['removed_frames']):
                raise ValueError(invalid)
        for table,column,values in (('batch_events','job_id',removal['batches']),('events','run_id',removal['runs'])):
            current = [tuple(db.execute('SELECT COUNT(*),MAX(rowid) FROM '+table+' WHERE '+column+'=?',(key,)).fetchone())
                       for key in sorted(values)]
            if current != frozen['revisions'][table]:
                raise ValueError(invalid)
        # Research writes normally share maintenance(), but validate their
        # references too so a second connection cannot publish a dangling label.
        for table,key,removed in (('audit_cohorts','cohort_id',removal['audit_cohorts']),
                                   ('label_releases','release_id',removal['label_releases'])):
            if table not in frozen['rows']:
                continue
            before = {r[key]:r for r in frozen['rows'][table]}
            for row in db.execute('SELECT * FROM '+table):
                current = dict(row)
                if current[key] in removed:
                    if current != before.get(current[key]):
                        raise ValueError(invalid)
                elif (ids(current) - ids(before.get(current[key],{}))) & frozen['deleted_ids']:
                    raise ValueError(invalid)

    def _remove(self, path):
        self.tree(path)
        if path.is_dir():
            removed = 0
            for directory,folders,files in os.walk(path,topdown=False,followlinks=False):
                for name in files:
                    (Path(directory)/name).unlink()
                    removed += 1
                    if removed % 100 == 0:
                        self.report(files_removed=removed)
                for name in folders:
                    (Path(directory)/name).rmdir()
            path.rmdir()
            self.report(files_removed=removed)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def filter_log(raw, policy):
        deleted,targets,frames,pairs = policy
        kept = []
        for line in raw.decode('utf-8',errors='replace').splitlines(keepends=True):
            try:
                record = json.loads(line)
            except ValueError:
                record = {}
            if (ids(line) & deleted or target_key(record) in targets
                    or (record.get('context'),record.get('timestamp')) in frames
                    or (record.get('context'),record.get('object_id')) in pairs):
                continue
            kept.append(line)
        return ''.join(kept).encode()

    def _restore(self, journal, manifest):
        for entry in reversed(manifest['entries']):
            if not isinstance(entry['index'], int) or entry['index'] < 0:
                raise ValueError('删除恢复记录无效')
            path = self.safe(self.runtime / entry['path'])
            backup = self.safe(journal / 'files' / str(entry['index']))
            if backup.exists():
                if path.exists():
                    self._remove(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                backup.replace(path)
        self._remove(journal)

    def recover(self):
        """Called before serving requests, and on deletion retry; no archives remain."""
        if not self.journals.exists():
            return False
        pending = False
        self.tree(self.journals)
        for journal in list(self.journals.iterdir()):
            manifest_path = journal / 'manifest.json'
            if not manifest_path.exists():
                self._remove(journal)  # No published manifest means no original file was touched.
                continue
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            with self.runs.connect() as db:
                exists = any(db.execute('SELECT 1 FROM batches WHERE job_id=?', (j,)).fetchone() for j in manifest['job_ids'])
                exists |= any(db.execute('SELECT 1 FROM runs WHERE run_id=?', (r,)).fetchone()
                              for r in manifest.get('run_ids', []))
            try:
                if manifest.get('committed') or not exists:
                    self._remove(journal)
                else:
                    self._restore(journal, manifest)
            except OSError:
                pending = True
        if not any(self.journals.iterdir()):
            self.journals.rmdir()
        return pending

    def preview(self, requested):
        self.report(stage='waiting',message='正在等待其他维护操作',completed=0,total=0)
        with self.runs.maintenance():
            with self.runs.connect() as db:
                self.guard(db,requested)
            if self.recover():
                raise OSError('上一批删除的文件仍被占用，请关闭占用程序后重试清理。')
            with self.runs.connect() as db:
                db.execute('BEGIN')
                result = self.plan(db,requested)[0]
            self.flush_cache()
            for token in result['candidate_tokens'].values():
                self._confirmation_scopes[(str(self.runs.path),token)] = result['shared_cleanup_deferred']
            while len(self._confirmation_scopes) > 512:
                self._confirmation_scopes.pop(next(iter(self._confirmation_scopes)),None)
            self.report(stage='ready',message='核对完成，等待确认',completed=1,total=1)
            return result

    def delete(self, requested, token, return_to_pool=None):
        self.report(stage='waiting',message='正在等待其他维护操作',completed=0,total=0)
        with self.runs.maintenance():
            with self.runs.connect() as db:
                self.guard(db,requested)
                exists = any(db.execute('SELECT 1 FROM batches WHERE job_id=?',(j,)).fetchone() for j in requested)
            pending = self.recover()
            if requested and not exists:
                return {'deleted_job_ids':requested,'cleanup_pending':pending}
            # All full scans and filesystem validation run in a read snapshot.
            # WAL writers can save model responses and renew their leases here.
            with self.runs.connect() as db:
                db.execute('BEGIN')
                preserve_shared = self._confirmation_scopes.get((str(self.runs.path),token))
                public,paths,rewrites,removal = self.plan(db,requested,return_to_pool,preserve_shared)
            self.flush_cache()
            if token != public['token']:
                raise ValueError('任务或关联数据已变化，请重新查看删除范围后确认。')
            journal = self.safe(self.journals / uuid4().hex)
            journal.mkdir(parents=True)
            entries = [{'path':p.relative_to(self.runtime).as_posix(),'index':i}
                       for i,p in enumerate(sorted(paths | rewrites.keys()))]
            manifest = {'job_ids':public['job_ids'],'run_ids':sorted(removal['runs']),
                        'entries':entries,'committed':False}
            atomic_json(journal/'manifest.json',manifest)
            committed = False
            write_seconds = 0
            try:
                self.report(stage='staging',message='正在暂存待删资源，可在失败时恢复',completed=0,total=len(entries))
                for index,entry in enumerate(entries,1):
                    path = self.safe(self.runtime/entry['path'])
                    backup = journal/'files'/str(entry['index'])
                    backup.parent.mkdir(parents=True,exist_ok=True)
                    # Only the last idle-task deletion rewrites shared logs.
                    # Protect that brief file publication, not the full scan.
                    if path in rewrites and path.is_relative_to(self.annotation/'logs'):
                        with diagnostic_maintenance(self.annotation):
                            path.replace(backup)
                            path.write_bytes(self.filter_log(backup.read_bytes(),removal['log_policy']))
                    else:
                        path.replace(backup)
                        if path in rewrites:
                            path.write_bytes(rewrites[path])
                    self.report(completed=index,total=len(entries))
                self.report(stage='committing',message='正在提交任务删除记录',completed=0,total=0)
                with self.runs.connect() as db:
                    # Erasing freed payload pages is not required for logical
                    # deletion; doing it here made large removals monopolize WAL.
                    db.execute('PRAGMA secure_delete=OFF')
                    db.execute('BEGIN IMMEDIATE')
                    started = time.monotonic()
                    self.validate_commit(db,requested,removal)
                    from app.infrastructure.candidate_pool import exclude
                    exclude(db,removal['excluded_tracks'],'deleted-tasks:'+digest(sorted(requested)))
                    for table,column,key in (('events','run_id','runs'),('controls','run_id','runs'),('runs','run_id','runs'),
                            ('batch_items','job_id','batches'),('batch_events','job_id','batches'),('batches','job_id','batches'),
                            ('audit_localizations','cohort_id','audit_cohorts'),('audit_answers','cohort_id','audit_cohorts'),
                            ('audit_answer_events','cohort_id','audit_cohorts'),('audit_cohorts','cohort_id','audit_cohorts'),
                            ('label_releases','release_id','label_releases')):
                        if removal[key]:
                            db.executemany('DELETE FROM '+table+' WHERE '+column+'=?',[(v,) for v in removal[key]])
                    for job,payload in removal['history_updates'].items():
                        db.execute("DELETE FROM batch_events WHERE job_id=? AND kind='retained_task_history'",(job,))
                        db.execute("INSERT INTO batch_events(job_id,kind,payload,at) VALUES(?,'retained_task_history',?,?)",
                                   (job,json.dumps(payload,allow_nan=False),time.time()))
                    db.commit()
                    committed = True
                    write_seconds = time.monotonic()-started
                self.report(database_write_seconds=round(write_seconds,4))
                atomic_json(journal/'manifest.json',{**manifest,'committed':True})
                self.report(stage='cleaning',message='任务已删除，正在清理独占文件',completed=0,total=0)
                self._remove(journal)
            except Exception:
                # connect() has rolled back and released its writer before any
                # filesystem restoration/cleanup, even when the disk is slow.
                if not committed:
                    self.report(stage='restoring',message='删除未提交，正在恢复原始记录与文件',completed=0,total=0)
                    self._restore(journal,manifest)
                    raise
                pending = True
            if self.journals.exists() and not any(self.journals.iterdir()):
                self.journals.rmdir()
            with self.runs.connect() as db:
                db.execute('PRAGMA busy_timeout=0')
                try:
                    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                except sqlite3.OperationalError as exc:
                    if getattr(exc,'sqlite_errorcode',0) & 0xff not in {sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED}:
                        raise
            self.report(stage='finished',message='删除完成' if not pending else '任务已删除，部分文件待清理',completed=1,total=1)
            return {'deleted_job_ids':public['job_ids'],'summary':public['summary'],'cleanup_pending':pending,
                    'database_write_seconds':round(write_seconds,4)}
