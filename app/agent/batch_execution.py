"""One batch coordinator, independent target workers, bounded preparation.

Only the coordinator dispatches targets or settles the batch. Workers retain
their own calls, leases and checkpoints. Native evidence preparation overlaps
on two threads; the official converter and evidence registry remain serial.
"""
import threading
import time
import json
import errno
import sqlite3
import os
import sys
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import OrderedDict, defaultdict, deque

from app.agent import batch as api
from app.agent.agent_runtime import VERSION, TERMINAL, ControlRequested, initial_run
from app.agent.model_calls import QwenCalls, saved_call_diagnostic, isolated_request_failure
from app.datasets.preparation import (TargetPreparer, PreparationCancelled,
    TargetNotDistinguishable, EvidenceRejected, ConversionInterrupted)
from app.infrastructure.batch_timing import BatchTimer
from app.infrastructure.conversion_environment import ConversionEnvironmentUnavailable
from app.infrastructure.diagnostics import diagnostic_event, PerformanceLog
from app.infrastructure.storage import RunBusy, RunStoreUnavailable


SETTLED = {'excluded', 'prepared', 'calibration_hold', 'abstained', 'needs_evidence',
    'failed', 'budget_skipped', 'reused', 'not_needed'}


class FrameAffinityQueue:
    """Schedule fresh targets after completed same-frame peers, never reuse trust.

    Frozen targets/queue positions stay intact. Reserved/prepared work retains
    priority; independent frames fill idle slots before duplicating a busy frame.
    Actual observation reuse still requires the existing pixel/identity checks.
    """
    def __init__(self, items, enabled=False):
        self.items = OrderedDict((i['ordinal'], i) for i in items)
        self.enabled = enabled
        self.keys, self.groups, self.warm = {}, defaultdict(deque), OrderedDict()
        self.protected = deque()
        for ordinal, item in self.items.items():
            target = item['target']
            key = (target.get('context'), target.get('timestamp'))
            if not key[0] or key[1] is None:
                key = ('unshared', ordinal)
            self.keys[ordinal] = key
            if item['status'] != 'pending' or item.get('run_id') or item.get('packet_id') or target.get('manual_review_retry'):
                self.protected.append(ordinal)
            else:
                self.groups[key].append(ordinal)

    def finished(self, ordinal):
        if self.enabled and ordinal in self.keys:
            self.warm[self.keys[ordinal]] = None

    def take(self, active=()):
        first = next(iter(self.items))  # StopIteration is the coordinator's exhaustion signal.
        if not self.enabled:
            return self.items.pop(first)
        while self.protected and self.protected[0] not in self.items:
            self.protected.popleft()
        if self.protected:
            return self.items.pop(self.protected.popleft())
        busy = {self.keys[o] for o in active if o in self.keys}
        for key in list(self.warm):
            group = self.groups[key]
            while group and group[0] not in self.items:
                group.popleft()
            if not group:
                del self.warm[key]
            elif key not in busy:
                return self.items.pop(group.popleft())
        ordinal = next((o for o in self.items if self.keys[o] not in busy), first)
        return self.items.pop(ordinal)


class BatchConfigurationError(ValueError):
    pass


def shared_failure(exc):
    """Data errors affect one target; storage, leases and config affect the batch."""
    return (isinstance(exc, (RunBusy, RunStoreUnavailable, sqlite3.Error, MemoryError,
                PermissionError, BatchConfigurationError))
        or isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EROFS, errno.EMFILE, errno.ENFILE}
        or isinstance(exc, OSError) and getattr(exc, 'winerror', None) in {39, 112})


class BatchExecution:
    def __init__(self, service, job, owner, catalog):
        self.service, self.store = service, service.store
        self.worker, self.job, self.owner, self.catalog = service.runner, job, owner, catalog
        self.preparation_concurrency = 2 if type(getattr(service,'preparer',None)) is TargetPreparer else 1
        self.preparation = threading.BoundedSemaphore(self.preparation_concurrency)
        self.registration = threading.Lock()
        self.abort = threading.Event()
        self.timer = None
        self.lease_lock = threading.Lock()
        self.last_lease_renewal = None
        self.provider_clients = {}
        self.client_lock = threading.Lock()
        from app.agent.appearance import SceneObservationCache
        self.observation_cache=SceneObservationCache()

    def target_calls(self):
        """Keep the native provider connection warm on each target thread.

        Checkpoints and request folders still belong to each individual run.
        Custom/test factories retain their original per-target lifetime.
        """
        if self.worker.calls_factory is not QwenCalls:
            return self.worker.calls_factory(), True
        if self.provider != {
                'model':os.getenv('DASHSCOPE_MODEL','qwen3.8-flash'),
                'base_url':os.getenv('DASHSCOPE_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')}:
            raise BatchConfigurationError('Provider changed during execution; pause and recover this task')
        with self.client_lock:
            thread = threading.get_ident()
            if thread not in self.provider_clients:
                self.provider_clients[thread] = QwenCalls()
            return self.provider_clients[thread], False

    def close_clients(self):
        # Called only after the target pool has joined, never during a request.
        for calls in self.provider_clients.values():
            try:
                calls.close()
            except Exception as exc:
                diagnostic_event(self.worker.root,'provider_close_failed',exception=exc,job_id=self.job)
        self.provider_clients.clear()
        self.observation_cache.clear()

    def register(self, packet):
        # The registry is an atomic JSON read/modify/write, not an append log.
        # Independent evidence builders must not overwrite each other's entry.
        with self.registration:
            self.service.registrar(packet)

    def renew_executor(self):
        # Coordinator and preparation waiters share the same 300-second lease.
        # Poll control frequently without committing a lease write every tick.
        with self.lease_lock:
            now = time.monotonic()
            if self.last_lease_renewal is None or now - self.last_lease_renewal >= 30:
                self.worker.store.renew_executor(self.owner)
                self.last_lease_renewal = time.monotonic()

    def cancelled(self):
        return self.abort.is_set() or bool(self.store.command(self.job))

    @contextmanager
    def preparation_slot(self, heartbeat=None, performance=None):
        acquired = False
        try:
            with performance.measure('preparation_slot_wait') if performance else nullcontext():
                while not acquired:
                    if self.cancelled():
                        raise PreparationCancelled('preparation_cancelled')
                    if heartbeat is not None:
                        heartbeat()
                    self.renew_executor()
                    acquired = self.preparation.acquire(timeout=0.1)
            if self.cancelled():
                raise PreparationCancelled('preparation_cancelled')
            yield
        finally:
            if acquired:
                self.preparation.release()

    def pause_children(self, command='pause'):
        with self.worker.store.connect() as db:
            rows = db.execute("SELECT run_id FROM batch_items WHERE job_id=? AND status='running'",
                (self.job,)).fetchall()
        for row in rows:
            if row['run_id']:
                try:
                    self.worker.store.request_control(row['run_id'], command)
                except ValueError:
                    pass  # A response may have settled since this query.

    def run(self):
        halt = None
        try:
            self.timer = BatchTimer(self.store, self.job)
            detail = self.store.execution_detail(self.job)
            specification = dict(detail['spec'])
            self.provider = specification.pop('provider', None)
            self.spec = api.BatchSpec.model_validate(specification)
            # Historical finite quotas retain their serial allocation behavior.
            ledger = self.store.continuation_budget(self.job)
            if ledger['limit'] is not None:
                available = max(0,ledger['limit']-ledger['spent']) + detail['model_calls_reserved']
                self.spec = self.spec.model_copy(update={'max_model_calls':available if self.spec.max_model_calls is None
                    else min(self.spec.max_model_calls,available)})
            parallelism = (1 if self.spec.prepare_only or self.spec.max_model_calls is not None
                else self.spec.target_concurrency)
            continuation = ledger['is_continuation']
            retry_authorizations = self.worker.store.candidate_retries()
            preparation_attempts = self.worker.store.preparation_attempted_keys(exclude_job=self.job)
            unavailable, processed = api.unavailable_target_keys(
                [state for state in self.worker.store.selection_runs() if not (
                    state.get('batch_id') == self.job and state.get('retired_from_execution'))],
                VERSION, include_prior_attempts=True, retries=retry_authorizations,
                preparation_attempts=preparation_attempts)
            if continuation:
                unavailable = processed | (preparation_attempts - retry_authorizations.keys())
            pending = FrameAffinityQueue((i for i in detail['items'] if i['status'] not in SETTLED),
                enabled=parallelism > 1 and self.preparation_concurrency > 1)
            exhausted, inflight, connection_failures, request_rejections = False, {}, 0, 0
            dispatched = set()
            self.store.update(self.job, 'execution_pipeline_started', {
                'target_concurrency': parallelism, 'preparation_concurrency': min(parallelism,self.preparation_concurrency),
                'model_concurrency': self.worker.model_slots.concurrency,
                'target_scheduling': 'completed-frame-affinity-v1' if pending.enabled else 'frozen-queue-order'})
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix='target') as pool:
                while not exhausted or inflight:
                    self.renew_executor()
                    command = self.store.command(self.job)
                    if command and halt is None:
                        halt = ('stopped' if command == 'stop' else 'paused', None)
                    for future in [f for f in inflight if f.done()]:
                        pending.finished(inflight.pop(future))
                        try:
                            outcome = future.result()
                        except Exception as exc:
                            diagnostic_event(self.worker.root, 'target_worker_failed', exception=exc, job_id=self.job)
                            outcome = ('blocked', type(exc).__name__)
                        if halt is None:
                            if outcome and outcome[0] == 'request_rejected':
                                request_rejections += 1
                                if request_rejections >= 3:
                                    halt = ('paused', 'repeated_request_rejections')
                            elif outcome and outcome[0] == 'connection':
                                request_rejections = 0
                                connection_failures += 1
                                delay = min(300, 15 * connection_failures)
                                self.worker.model_slots.backoff(delay)
                                self.store.update(self.job, 'provider_cooldown', {
                                    'consecutive_failures': connection_failures, 'delay_seconds': delay})
                            else:
                                request_rejections = 0
                                connection_failures = 0
                                halt = outcome
                    retries = [item for item in self.store.pending_review_retries(self.job)
                        if item['ordinal'] not in inflight.values()
                        and (item['ordinal'],item['target'].get('execution_attempt')) not in dispatched]
                    if retries and halt is None:
                        exhausted = False
                    if halt is not None:
                        self.abort.set()
                        self.pause_children('stop' if halt[0] == 'stopped' else 'pause')
                        exhausted = True
                    elif not exhausted and len(inflight) < parallelism:
                        # Outcomes cannot change dispatch while all success slots
                        # are occupied. Avoid rereading large run JSONs on every
                        # polling tick; refresh before allocating the next slot.
                        progress = self.store.execution_progress(self.job)
                        if self.spec.desired_count is not None and progress['passed'] >= self.spec.desired_count:
                            exhausted = True
                        while not exhausted and len(inflight) < parallelism:
                            # Each in-flight target occupies one success slot, including
                            # targets waiting for preparation. Never overshoot the goal.
                            if (self.spec.desired_count is not None
                                    and progress['passed'] + len(inflight) >= self.spec.desired_count):
                                break
                            try:
                                if retries:
                                    item = retries.pop(0)
                                else:
                                    item = pending.take(inflight.values())
                                    # Re-read the row: a reviewer can replace its round
                                    # while this coordinator still holds the old list.
                                    with self.worker.store.connect() as db:
                                        current = db.execute('SELECT * FROM batch_items WHERE job_id=? AND ordinal=?',
                                            (self.job,item['ordinal'])).fetchone()
                                    item = {**dict(current),'target':json.loads(current['target'])}
                                dispatch_key = (item['ordinal'],item['target'].get('execution_attempt'))
                                if dispatch_key in dispatched or item['ordinal'] in inflight.values() or item['status'] in SETTLED:
                                    continue
                            except StopIteration:
                                exhausted = True
                                break
                            if not item['run_id'] and self.spec.desired_count is not None:
                                if not item['target'].get('manual_review_retry') and self.spec.call_allowance(progress['calls']) < 4:
                                    exhausted = True
                                    break
                                key = (item['target']['context'], item['target']['object_id'])
                                if key in unavailable and not item['target'].get('manual_review_retry'):
                                    self.store.update(self.job, 'candidate_no_longer_available',
                                        {'ordinal': item['ordinal']}, ordinal=item['ordinal'],
                                        item_status='not_needed', reason='target_already_attempted')
                                    continue
                                unavailable.add(key)
                            dispatched.add(dispatch_key)
                            inflight[pool.submit(self.process_item, item)] = item['ordinal']
                    if inflight:
                        wait(inflight, timeout=0.1, return_when=FIRST_COMPLETED)
            if halt:
                if halt[0] == 'blocked':
                    with self.worker.store.connect() as db:
                        paused = db.execute("""SELECT 1 FROM batch_items i JOIN runs r ON r.run_id=i.run_id
                            WHERE i.job_id=? AND json_extract(r.snapshot,'$.status') IN ('paused','stopped') LIMIT 1""",
                            (self.job,)).fetchone()
                    if paused:
                        # Keep sibling checkpoints resumable in their original budget
                        # owner. Failed/unknown calls stay failed and are never replayed.
                        halt = ('paused', halt[1])
                self.store.update(self.job, 'batch_control_applied' if halt[0] != 'blocked' else 'batch_blocked',
                    {'reason': halt[1]}, status=halt[0], reason=halt[1])
            elif self.spec.desired_count is not None:
                progress = self.store.execution_progress(self.job)
                reason = ('description_goal_reached' if progress['passed'] >= self.spec.desired_count
                    else 'selection_budget_exhausted' if self.spec.call_allowance(progress['calls']) < 4
                    else 'candidate_pool_exhausted')
                self.store.complete_selection(self.job, reason)
            else:
                self.store.update(self.job, 'batch_completed', {'training_eligible': False}, status='completed')
        except Exception as exc:
            self.abort.set()
            self.pause_children()
            diagnostic_event(self.worker.root, 'batch_blocked', level='ERROR', exception=exc, job_id=self.job)
            self.store.update(self.job, 'batch_blocked', {'error_type': type(exc).__name__},
                status='blocked', reason=type(exc).__name__)
        finally:
            self.close_clients()
            try:
                if isinstance(self.service.preparer, TargetPreparer):
                    self.service.preparer.close()
            except Exception as exc:
                diagnostic_event(self.worker.root, 'converter_close_failed', exception=exc, job_id=self.job)
            finally:
                try:
                    if self.timer is not None:
                        self.timer.close()
                finally:
                    try:
                        self.worker.store.release_executor(self.owner)
                    finally:
                        with self.worker.lock:
                            self.worker.active_batch = None

    def process_item(self, item):
        ordinal = item['ordinal']
        self.timer.begin_item(ordinal)
        performance = PerformanceLog(self.worker.root / 'batches' / self.job / str(ordinal),
            job_id=self.job, ordinal=ordinal, scope='target')
        performance.phase('dispatch_checks')
        try:
            # Read at dispatch, including for queues created before a deletion.
            # Existing linked runs still settle in their original task/budget.
            if not item['run_id'] and (item['target']['context'], item['target']['object_id']) in self.worker.store.excluded_target_keys():
                self.store.update(self.job, 'candidate_excluded_after_deletion', {'ordinal': ordinal},
                    ordinal=ordinal, item_status='not_needed', reason='target_excluded_after_task_deletion')
                return None
            if not self.spec.prepare_only and not item['run_id']:
                used = self.worker.store.target_usage_for(
                    item['target']['context'], item['target']['object_id'], 'real-model')
                limit = item['target'].get('retry_authorization',{}).get('call_limit',12)
                remaining = min(12 if item['target'].get('manual_review_retry') else
                    self.spec.call_allowance(self.store.execution_progress(self.job)['calls']), limit - used)
                if remaining < 4:
                    self.store.update(self.job, 'target_budget_skipped', {'ordinal': ordinal},
                        ordinal=ordinal, item_status='budget_skipped', reason='insufficient_complete_validation_budget')
                    return None
            packet = self.prepare_item(item, performance)
            performance.phase('annotation_setup')
            if packet.task.get('annotation_search'):
                from app.agent.scene_grounding import POLICY, CLAIM_POLICY
                packet = packet.model_copy(deep=True)
                packet.task['scene_grounding_policy']=POLICY
                packet.task['scene_claim_policy']=CLAIM_POLICY
                packet.task['source_camera_scope_policy']='appearance-source-scope-v1'
                packet.task['attribute_repair_policy']='disputed-attribute-omission-v1'
                packet.task['scene_geometry_authority_policy']='native-table-authority-v1'
                packet.task['scene_relation_reading_policy']='explicit-subject-landmark-v2-bidirectional'
                packet.task['scene_binding_policy']='anonymous-region-binding-v1'
                packet.task['scene_region_card_policy']='explicit-anonymous-region-cards-v1'
                packet.task['scene_input_policy']='sentence-scoped-scene-input-v1'
                packet.task['feedback_revision_policy']='feedback-evidence-revision-v1'
                packet.task['feedback_acquisition_policy']='feedback-landmark-recheck-v1'
                packet.task['landmark_salience_policy']='aligned-landmark-preference-v1'
                packet.task['feedback_acceptance_policy']='feedback-acceptance-v5-natural-concision'
                packet.task['feedback_relation_policy']='bilingual-landmark-feedback-v2-omissions'
                packet.task['feedback_feature_policy']='verified-target-feature-feedback-v1'
                packet.task['sentence_claim_repair_policy']='sentence-claim-repair-v1'
                packet.task['sentence_style_policy']='simple-sufficient-sentence-v1'
                packet.task['target_naming_policy']='explicit-target-category-v1'
                packet.task['ego_wording_policy']='complete-ego-direction-v1'
                packet.task['landmark_locality_policy']='local-landmark-selection-v1'
                packet.task['feedback_detail_policy']='explicit-feedback-details-v1'
                packet.task['ego_proximity_policy']='native-ego-proximity-v1'
                packet.task['spatial_language_policy']='plain-spatial-language-v1'
                packet.task['manual_feedback_policy']='verified-feedback-planning-v1'
                from app.quality.description_search import TRAFFIC_SCENE_POLICY
                packet.task['reference_preference_policy']=TRAFFIC_SCENE_POLICY
                from app.quality.relations import VEHICLE_HEADING_POLICY
                packet.task['vehicle_heading_policy']=VEHICLE_HEADING_POLICY
                packet.task['claim_normalization_policy']='compound-claim-entailment-v1'
                packet.task['joint_reference_policy']='joint-target-landmark-v1'
                packet.task['vehicle_forward_region_policy']='vehicle-forward-region-v1'
                packet.task['spatial_relation_policy']='native-3d-spatial-relations-v1'
            if item['target'].get('execution_attempt'):
                packet = packet.model_copy(deep=True)
                packet.task['execution_attempt'] = item['target']['execution_attempt']
            reference = item['target'].get('manual_review_retry') or self.spec.review_retry
            if reference:
                from app.quality.sample_review import retry_source, expression, ReviewRetry
                if isinstance(reference,dict):
                    reference = ReviewRetry.model_validate(reference)
                source, receipt = retry_source(self.worker.store, reference)
                packet = packet.model_copy(deep=True)
                packet.task['manual_review_feedback'] = {
                    **reference.model_dump(), 'previous_expression': expression(source),
                    'reason': receipt['notes']}
            if self.spec.prepare_only:
                return None
            performance.phase('annotation')
            return self.annotate_item(item, packet)
        except PreparationCancelled:
            self.store.update(self.job, 'preparation_cancelled', {'ordinal': ordinal},
                ordinal=ordinal, item_status='pending')
        except TargetNotDistinguishable as exc:
            self.store.update(self.job, 'target_prefiltered', {'ordinal': ordinal, 'model_calls': 0},
                ordinal=ordinal, item_status='excluded', reason=str(exc) or 'no_multimodal_candidate')
        except EvidenceRejected as exc:
            self.assert_target(exc.packet, item['target'])
            self.store.update(self.job, 'evidence_rejected', {'ordinal': ordinal}, ordinal=ordinal,
                item_status='abstained', reason='target_evidence_not_observable', packet_id=exc.packet.packet_id)
        except ConversionEnvironmentUnavailable as exc:
            self.store.update(self.job, 'conversion_environment_unavailable', {'ordinal': ordinal},
                ordinal=ordinal, item_status='pending', reason='preparation:' + str(exc))
            return ('paused', 'preparation:' + str(exc))
        except ConversionInterrupted as exc:
            self.store.update(self.job, 'target_conversion_skipped', {'ordinal': ordinal},
                ordinal=ordinal, item_status='failed', reason='preparation:' + str(exc))
        except Exception as exc:
            diagnostic_event(self.worker.root, 'target_worker_failed', level='ERROR', exception=exc,
                job_id=self.job, ordinal=ordinal)
            self.store.update(self.job, 'target_worker_failed', {'ordinal': ordinal, 'error_type': type(exc).__name__},
                ordinal=ordinal, item_status='pending' if shared_failure(exc) else 'failed', reason=type(exc).__name__)
            if shared_failure(exc):
                return ('paused', type(exc).__name__)
        finally:
            performance.close('interrupted' if sys.exc_info()[0] else 'returned')
            self.timer.finish_item(ordinal)
        return None

    @staticmethod
    def assert_target(packet, target, *, alternate=False):
        if ((packet.context, packet.target_object_id) != (target['context'], target['object_id'])
                or not alternate and packet.frame_timestamp_micros != target['timestamp']
                or alternate and (packet.target.category != target['category']
                    or packet.frame_timestamp_micros not in
                        {target['timestamp'], *(o['timestamp'] for o in target.get('alternative_selections', []))})):
            raise ValueError('Prepared packet does not match frozen target')

    def record_preparation_rejections(self, ordinal, attempts):
        records = []
        for attempt in attempts:
            packet = attempt.get('packet')
            if packet is not None:
                self.register(packet)
            records.append({k:v for k,v in attempt.items() if k != 'packet'})
        if records:
            self.store.update(self.job, 'preparation_frames_rejected',
                {'ordinal':ordinal, 'attempts':records})
        return records

    def prepare_item(self, item, performance=None):
        if item['packet_id']:
            if performance:
                performance.phase('load_cached_evidence')
            input_id = (self.worker.store.snapshot(item['run_id']).get('input_packet_id', item['packet_id'])
                if item['run_id'] else item['packet_id'])
            return api.load_packet(input_id)
        ordinal, target = item['ordinal'], item['target']
        if performance:
            performance.phase('preparation_wait')
        with self.preparation_slot(performance=performance):
            if performance:
                performance.phase('preparation_setup')
            self.store.update(self.job, 'preparation_started', {'ordinal': ordinal, 'target_id': target['target_id']},
                ordinal=ordinal, item_status='preparing')
            started, last_stage = time.monotonic(), None

            def heartbeat(stage=None):
                nonlocal last_stage
                self.renew_executor()
                if stage and stage != last_stage:
                    if performance:
                        performance.phase(stage)
                    last_stage = stage
                    payload = {'ordinal': ordinal, 'stage': stage,
                        'elapsed_seconds': round(time.monotonic() - started, 3)}
                    self.store.update(self.job, 'preparation_progress', payload)
                    diagnostic_event(self.worker.root, 'preparation_progress', job_id=self.job, **payload)

            try:
                log_path = self.worker.root / 'batches' / self.job / str(ordinal) / 'preparation.log'
                method = getattr(self.service.preparer, 'prepare_with_fallback', None)
                if method is None:
                    packet, attempts = self.service.preparer(target, self.catalog, heartbeat, self.cancelled, log_path), []
                else:
                    packet, attempts = method(target, self.catalog, heartbeat, self.cancelled, log_path)
            except (EvidenceRejected, TargetNotDistinguishable, PreparationCancelled) as exc:
                self.record_preparation_rejections(ordinal, getattr(exc, 'preparation_attempts', []))
                if isinstance(exc, EvidenceRejected):
                    self.assert_target(exc.packet, target)
                    self.register(exc.packet)
                raise
            if performance:
                performance.phase('register_evidence')
            records = self.record_preparation_rejections(ordinal, attempts)
            self.assert_target(packet, target, alternate=bool(records))
            self.register(packet)
            payload = {'ordinal':ordinal,'packet_id':packet.packet_id}
            if records:
                payload.update(timestamp=packet.frame_timestamp_micros,
                    source_view=packet.source_view,rejected_frames=records)
            # The accepted packet link and its frame receipt are one checkpoint.
            self.store.update(self.job, 'preparation_source_reselected' if records else 'evidence_prepared', payload,
                ordinal=ordinal, item_status='prepared' if self.spec.prepare_only else 'pending', packet_id=packet.packet_id)
            return packet

    def annotate_item(self, item, packet):
        ordinal = item['ordinal']
        try:
            calls, close_after_target = self.target_calls()
        except ValueError as exc:
            raise BatchConfigurationError('Invalid model configuration') from exc
        try:
            if self.provider != {'model': calls.model, 'base_url': calls.base_url}:
                raise BatchConfigurationError('Provider changed during execution; pause and recover this task')
            if item['run_id']:
                prior = self.worker.store.snapshot(item['run_id'])
                budget, run_id = prior['model_call_budget'], prior['run_id']
                cumulative_limit = prior.get('cumulative_call_limit',12)
                if prior['status'] in {'running', 'paused', 'stopped'}:
                    self.worker.store.request_control(run_id, 'resume')
            else:
                remaining = (12 if item['target'].get('manual_review_retry') else
                    self.spec.call_allowance(self.store.execution_progress(self.job)['calls']))
                cumulative_limit = item['target'].get('retry_authorization',{}).get('call_limit',12)
                budget = min(remaining, cumulative_limit - self.worker.store.target_usage(packet, getattr(calls, 'scope', 'real-model')))
                minimum = 4 if packet.ready_for_generation else 5 if api.expand_relations(packet)[0].ready_for_generation else 1
                if budget < minimum:
                    self.store.update(self.job, 'target_budget_skipped', {'ordinal': ordinal}, ordinal=ordinal,
                        item_status='budget_skipped', reason='insufficient_complete_validation_budget')
                    return None
                initial = initial_run(packet, calls, self.spec.max_repairs, budget, cumulative_limit)
                run_id = initial['run_id']
                existing = self.worker.store.snapshot(run_id, missing_ok=True)
                if existing:
                    if existing['status'] not in TERMINAL:
                        raise ValueError('Existing unfinished run belongs to another job')
                    self.store.update(self.job, 'target_result_reused', {'ordinal': ordinal, 'run_id': run_id},
                        ordinal=ordinal, item_status='reused', run_id=run_id, reused=True)
                    return None
                initial['batch_id'] = self.job
                with self.worker.store.connect() as db:
                    row = db.execute("""SELECT payload FROM batch_events WHERE job_id=?
                        AND kind='preparation_source_reselected' AND json_extract(payload,'$.ordinal')=?
                        AND json_extract(payload,'$.packet_id')=? ORDER BY seq DESC LIMIT 1""",
                        (self.job, ordinal, packet.packet_id)).fetchone()
                if row:
                    receipt = json.loads(row['payload'])
                    initial['rejected_source_views'] = receipt['rejected_frames']
                    # Keep the accepted preparation binding with the immutable
                    # run too, so inherited results do not depend on live parents.
                    initial['preparation_source_reselection'] = receipt
                self.store.attach_child(self.job, ordinal, initial)
            self.store.update(self.job, 'target_started_or_resumed', {'ordinal': ordinal, 'run_id': run_id},
                ordinal=ordinal, item_status='running', run_id=run_id)
            if self.cancelled():
                self.worker.store.request_control(run_id, self.store.command(self.job) or 'pause')
            result = api.run_agent(packet, calls, self.spec.max_repairs, self.worker.root, budget, cumulative_limit,
                resume_run_id=run_id, executor_owner=self.owner, alternate_frame=self.alternate_frame(ordinal),
                model_slots=self.worker.model_slots,observation_cache=self.observation_cache)
            if result['status'] in {'paused', 'stopped'}:
                return (result['status'], None)
            self.store.update(self.job, 'target_finished', {'ordinal': ordinal, 'run_id': run_id,
                'status': result['status'], 'auto_published': False}, ordinal=ordinal,
                item_status='failed' if result['status'] == 'blocked' else result['status'],
                reason=result.get('stop_reason'), run_id=run_id)
            if result['status'] != 'blocked' or result.get('stop_reason') == 'model_call:invalid_output':
                return None
            failure = (saved_call_diagnostic(self.worker.root / run_id / result['inflight']['folder'])
                if result.get('inflight') else {})
            outcome = failure.get('remote_outcome')
            if result.get('stop_reason') == 'model_call:api_failed' and isolated_request_failure(failure):
                self.store.update(self.job, 'target_request_rejected', {'ordinal':ordinal, 'run_id':run_id,
                    'http_status':failure['http_status'], 'diagnostic':failure,
                    'action':'preserve rejected request; continue with another target'})
                return ('request_rejected', None)
            if (result.get('stop_reason') == 'model_call:api_failed'
                    and failure.get('temporary_provider_failure')):
                return ('connection', None)
            if result.get('stop_reason') == 'model_call:interrupted_response':
                self.store.update(self.job, 'target_interrupted_call_preserved',
                    {'ordinal': ordinal, 'run_id': run_id, 'remote_outcome': 'unknown'})
                return None
            if (result.get('stop_reason') == 'model_call:connection_failed'
                    and (outcome == 'unknown' or outcome == 'not_sent' and failure.get('retryable') is True)):
                self.store.update(self.job, 'target_connection_failure_preserved',
                    {'ordinal': ordinal, 'run_id': run_id, 'remote_outcome': outcome})
                return ('connection', None)
            if result.get('stop_reason', '').startswith('runtime:'):
                return None
            return ('blocked', 'child_run_blocked')
        finally:
            if close_after_target:
                calls.close()

    def alternate_frame(self, ordinal):
        def prepare(current, rejected, renew_child):
            method = getattr(self.service.preparer, 'next_frame', None)
            if method is None:
                return None
            performance = PerformanceLog(self.worker.root / 'batches' / self.job / str(ordinal),
                job_id=self.job, ordinal=ordinal, scope='alternate_frame')
            performance.phase('preparation_wait')
            def heartbeat(stage=None):
                renew_child()
                self.renew_executor()
                if stage:
                    performance.phase(stage)
                    self.store.update(self.job, 'preparation_progress',
                        {'ordinal': ordinal, 'stage': stage, 'alternate_frame': True})
            try:
                with self.preparation_slot(renew_child, performance):
                    performance.phase('alternate_frame_setup')
                    packet = method(current, self.catalog, rejected, heartbeat, self.cancelled,
                        self.worker.root / 'batches' / self.job / str(ordinal) / 'preparation.log')
                    if packet is not None:
                        performance.phase('register_evidence')
                        self.register(packet)
                    return packet
            except PreparationCancelled:
                raise ControlRequested(self.store.command(self.job) or 'pause')
            finally:
                performance.close('interrupted' if sys.exc_info()[0] else 'returned')
        return prepare
