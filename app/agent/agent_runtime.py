"""Agent chooses bounded actions; the harness alone controls validation and stopping."""

import hashlib
import json
import time
import sys
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from app.schemas.contracts import AgentAction, EvidencePacket, GeneratedExpression, GroundResult, VisualObservation, SceneVisualObservation, VisualAttribute
from app.quality.counterfactuals import CounterfactualPlan, counterfactual_summary, plan_counterfactuals, verify_counterfactual
from app.datasets.evidence import EVIDENCE_ROOT, ROOT, atomic_json
from app.agent.harness import CLAIM_PARSE_GUIDE, GROUND_GUIDE, appearance_messages, appearance_views, blind_payload, normalize_expression, repair_camera_qualifier, claim_payload, generation_messages, verify_tool_grounding, expression_policy_errors, expression_language_errors
from app.agent.model_calls import ModelCallFailure, format_retry_messages, release_auth_rejection, transient_retry_delay, request_repair_option
from app.quality.relations import expand_relations
from app.infrastructure.storage import CallLimit, RunStore
from app.infrastructure.diagnostics import diagnostic_event, PerformanceLog
from app.infrastructure.artifacts import atomic_json_text
from app.agent.appearance import resolve_target_observation, target_visibility_error, effective_observations, required_body_color
from app.quality.visibility_policy import SUPPORTED_POLICIES, allows_partial, source_camera_inventory

VERSION = "adaptive-evidence-harness-v75-native-spatial-planning"
FORMAT_RETRY_LIMIT = 2
CONNECTION_RETRY_DELAYS = (1.0, 2.0)
TERMINAL = {"calibration_hold", "auto_pass", "abstained", "blocked", "needs_evidence"}
ACTION_GUIDE = """You control evidence acquisition and expression repair for outdoor 3D grounding annotation.
Choose exactly one allowed action from the supplied tools. Use a tool only if it resolves a concrete
ambiguity or error. Expand geometry relations when it eliminates competitors; generate once evidence
is sufficient; rewrite after a language/grounding failure; abstain when evidence/budget is insufficient.
You cannot change data, coordinates, validators, budgets, ground inputs, acceptance rules or targets.
Do not claim success or propose unlisted tools. Return JSON {action, reason}; reason must be concise,
at most 300 characters. Match the supplied output schema exactly."""


class BudgetExhausted(RuntimeError):
    pass


class ControlRequested(Exception):
    def __init__(self, command):
        self.command = command


def quality_route(diagnostic_pass: bool) -> dict:
    # This is a global calibration hold, not a per-sample manual annotation queue.
    return {"status": "calibration_hold" if diagnostic_pass else "abstained",
            "auto_published": False, "diagnostic_pass": diagnostic_pass,
            "quality_gate": {"oracle_checks_passed": diagnostic_pass,
                             "independent_risk_calibration": False,
                             "source_alignment_verified": False,
                             "scope": "global experimental hold; production auto-pass is not enabled"}}


def decision_context(packet: EvidencePacket, state: dict) -> tuple[dict, dict | None]:
    tools = []
    expansion = None
    observation_cost = int(bool(packet.task.get('contrastive_appearance')
        and packet.provenance.get('appearance_candidates') and 'candidate_appearance' not in state))
    if not state["relations_acquired"]:
        updated, result = expand_relations(packet)
        # No costly/model tool is offered without a concrete useful outcome.
        if result["eliminated_candidates"] > 0:
            expansion = {"packet": updated, "result": result}
            tools.append({"action": "expand_relations", "problem": "unresolved competitors",
                          "eliminated_candidates": result["eliminated_candidates"], "model_calls": 0})
    if packet.ready_for_generation and not state["attempts"]:
        from app.agent.expression_rendering import render_verified_plan
        observations = effective_observations(packet, state.get('candidate_appearance'))
        if state.get('resolved_target_observation'):
            observations[packet.target_object_id] = state['resolved_target_observation']
        # Generation normally costs one call; preserve a simple fallback only
        # when the remaining budget can pay for both blind checks but no writer.
        available = state['model_call_budget'] - state['model_calls_used']
        local_generation = available < 3 + observation_cost and render_verified_plan(packet, observations) is not None
        if available < 3 + observation_cost and not local_generation:
            from app.quality.joint_reference import writing_plan, fallback_sentence
            contract = writing_plan(packet,observations)
            local_generation = contract and fallback_sentence(packet,contract,observations) is not None
        tools.append({"action": "generate", "problem": "expression absent",
            "model_calls": (2 if local_generation else 3) + observation_cost, "additional_counterfactual_calls": 2})
    if packet.ready_for_generation and state["attempts"] and len(state["attempts"]) <= state["max_repairs"]:
        from app.quality.joint_reference import writing_plan, fallback_sentence
        observations = effective_observations(packet,state.get('candidate_appearance'))
        if state.get('resolved_target_observation'):
            observations[packet.target_object_id] = state['resolved_target_observation']
        contract = writing_plan(packet,observations,attempts=state['attempts'])
        local_repair = contract and fallback_sentence(packet,contract,observations) is not None
        from app.quality.spatial_relations import native_only
        if native_only(packet) and state['model_call_budget']-state['model_calls_used']>=3+observation_cost:
            local_repair=False
        tools.append({"action": "rewrite", "problem": "latest validation failure", "model_calls": (2 if local_repair else 3) + observation_cost})
    available_calls = state["model_call_budget"] - state["model_calls_used"]
    tools = [t for t in tools if t["model_calls"] + (0 if state.get('admission', {}).get('passed')
        and packet.task.get('execution_policy') == 'evidence-fast-v1' else 1) <= available_calls]
    tools.append({"action": "abstain", "problem": "evidence or budget insufficient", "model_calls": 0})
    previous = state["attempts"][-1].get("verification", {}) if state["attempts"] else {}
    # Canonical state projection, not an ever-growing conversation or lossy LLM summary.
    return {"task": packet.task, "target_category": packet.target.category,
            "remaining_candidate_count": len(packet.contrast.remaining_object_ids),
            "all_candidate_count": len(packet.candidates), "safe_unique_plan": packet.contrast.status == 'unique_in_gt_candidates',
            "multimodal_evidence_ready": packet.ready_for_generation,
            "planned_cues": [p.phrase for p in packet.predicates if p.predicate_id in packet.contrast.selected_predicate_ids],
            "latest_failure": {"errors": previous.get("errors", [])[:8],
                               "unsupported_claims": previous.get("unsupported_claims", [])[:8]},
            "output_schema": AgentAction.model_json_schema(),
            "budget": {"available_model_calls": available_calls, "max_repairs": state["max_repairs"],
                       "attempts_so_far": len(state["attempts"])}, "allowed_tools": tools}, expansion


def initial_run(packet: EvidencePacket, calls, max_repairs: int, max_model_calls: int,
                cumulative_call_limit: int | None = None) -> dict:
    if packet.task.get('requires_native_camera_link') or getattr(calls, 'scope', 'real-model') == 'real-model':
        from app.datasets.correspondence import has_multimodal_evidence
        if not has_multimodal_evidence(packet):
            raise ValueError('no_unique_native_camera_evidence')
    if getattr(calls, 'scope', 'real-model') == 'real-model' and (packet.task.get('admission_policy') != 'visible-distinct-v1' or packet.task.get('visibility_policy') not in SUPPORTED_POLICIES):
        raise ValueError('Evidence predates visual admission; prepare the target again')
    if getattr(calls, 'scope', 'real-model') == 'real-model':
        from app.datasets.appearance_candidates import IMAGE_INVENTORY_POLICY
        if (packet.task.get('image_identity_policy') == 'official-projection-v1'
                and packet.task.get('image_inventory_policy') != IMAGE_INVENTORY_POLICY):
            raise ValueError('Evidence predates complete image inventory; prepare the target again')
        from app.datasets.visibility import VIEW_SCREENING_POLICY, LEGACY_VIEW_SCREENING_POLICY
        expected_screening = VIEW_SCREENING_POLICY if allows_partial(packet) else LEGACY_VIEW_SCREENING_POLICY
        if packet.provenance.get('view_screening_policy') != expected_screening:
            raise ValueError('Evidence uses a different view-screening policy; prepare the target again')
    if not 0 <= max_repairs <= 2 or not 1 <= max_model_calls <= 16:
        raise ValueError("Prototype budgets: repairs 0..2, model calls 1..16")
    hashes = {name: hashlib.sha256((ROOT / "app" / name).read_bytes()).hexdigest()
              for name in ["agent/agent_runtime.py", "agent/harness.py", "agent/scene_grounding.py", "agent/appearance.py", "agent/expression_rendering.py", "agent/observation_wire.py", "datasets/correspondence.py", "datasets/visibility.py", "quality/visibility_policy.py", "quality/admission.py", "quality/relations.py", "quality/joint_reference.py", "quality/spatial_relations.py", "quality/description_search.py", "quality/contrast.py", "quality/sample_review.py", "schemas/contracts.py", "quality/counterfactuals.py", "agent/model_calls.py", "infrastructure/storage.py", "infrastructure/model_slots.py"]}
    identity = {"packet_id": packet.packet_id, "model": calls.model, "base_url": calls.base_url,
                "packet_sha256": hashlib.sha256(packet.model_dump_json().encode()).hexdigest(),
                "version": VERSION, "code_hashes": hashes, "max_repairs": max_repairs,
                "model_call_budget": max_model_calls,
                "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()}
    if cumulative_call_limit is not None:
        identity["cumulative_call_limit"] = cumulative_call_limit
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return {**identity, "run_id": run_id, 'input_packet_id': packet.packet_id,
               'binding_target':{'context':packet.context,'object_id':packet.target_object_id,
                   'timestamp':packet.frame_timestamp_micros,'category':packet.target.category},
               "context": packet.context, "category": packet.target.category,
               "provider_scope": getattr(calls, "scope", "real-model"), "status": "running",
               "phase": "precheck_target" if packet.task.get('admission_policy') == 'visible-distinct-v1' else "decide",
               "created_at": time.time(),
               "model_calls_used": 0, "relations_acquired": False, "attempts": [], "inflight": None,
               "current_packet": packet.model_dump(mode="json"), "auto_published": False, "tool_results": []}


def run_agent(packet: EvidencePacket, calls, max_repairs: int, output_root: Path, max_model_calls: int,
              cumulative_call_limit: int | None = None, resume_run_id: str | None = None,
              executor_owner: str | None = None, alternate_frame=None, model_slots=None,observation_cache=None) -> dict:
    initial = initial_run(packet, calls, max_repairs, max_model_calls, cumulative_call_limit)
    run_id = resume_run_id or initial["run_id"]
    folder = output_root / run_id
    owner = uuid4().hex
    store = RunStore(output_root / "runs.sqlite3")
    if resume_run_id:
        initial = store.snapshot(run_id)
        # Resume must reproduce the exact protocol and provider, even after a deploy.
        expected = initial_run(packet, calls, max_repairs, max_model_calls, cumulative_call_limit)
        if expected["run_id"] != run_id:
            raise ValueError("Run protocol changed; resume with its recorded code/configuration")
    executor = executor_owner or owner
    store.acquire_executor(executor)
    try:
        state = store.claim(run_id, owner, initial)
    except Exception:
        if executor_owner is None:
            store.release_executor(executor)
        raise
    if state["status"] in TERMINAL:
        if executor_owner is None:
            store.release_executor(executor)
        return state
    state["status"] = "running"
    started = time.perf_counter()
    prior_elapsed = state.get("elapsed_active_seconds", 0)
    last_renewal = started
    observation_batches = {}
    performance = PerformanceLog(folder, run_id=run_id, job_id=state.get('batch_id'),
        scope='annotation', model=calls.model, method_version=VERSION)
    performance.phase('startup')

    def precheck_batches(current):
        from app.agent.appearance import precheck_observation_batches
        # The schedule uses frozen frame geometry, source view and crop
        # inventory only, never model answers. Recompute after every source or
        # frame switch; do not cache visibility or admission decisions.
        key = (current.packet_id, current.source_view, current.source_camera_object_id)
        if key not in observation_batches:
            observation_batches.clear()
            observation_batches[key] = precheck_observation_batches(current)
        batches = observation_batches[key]
        reused=set(state.get('reused_scene_observation_ids',[]))
        if reused:
            remaining=[m for batch in batches for m in batch if m['object_id'] not in reused]
            batches=[remaining[start:start+16] for start in range(0,len(remaining),16)]
        if state.get('precheck_identity_focus') == list(key):
            from app.agent.appearance import identity_first_batches
            batches=identity_first_batches(current, batches)
        # Late peer results shrink a batch in place. Never repack completed or
        # reserved batches: their durable index/request must keep its identity.
        late=state.get('precheck_reused_batch_ids',{})
        return [[m for m in batch if m['object_id'] not in late.get(str(index),[])]
            for index,batch in enumerate(batches)] if late else batches

    def heartbeat():
        nonlocal last_renewal
        if time.perf_counter() - last_renewal >= 30:
            store.renew(run_id, owner)
            store.renew_executor(executor)
            last_renewal = time.perf_counter()

    @performance.measure('checkpoint_save')
    def save(kind, payload=None):
        store.renew_executor(executor)
        state["elapsed_active_seconds"] = round(prior_elapsed + time.perf_counter() - started, 6)
        snapshot = store.save(run_id, owner, state, kind, payload or {})
        # JSON is a derived readable mirror; SQLite snapshot/events are authoritative.
        if isinstance(snapshot, str):
            atomic_json_text(folder / "checkpoint.json", snapshot)
        else:
            # Custom store wrappers may retain the historical None return.
            atomic_json(folder / "checkpoint.json", state)

    def check_control():
        command = store.control(run_id)
        if command in {'pause', 'stop'}:
            raise ControlRequested(command)

    def invoke(kind, messages, schema):
        if model_slots is None:
            return invoke_admitted(kind, messages, schema)
        with ExitStack() as stack:
            with performance.measure('model_slot_wait'):
                stack.enter_context(model_slots.slot(heartbeat, check_control))
            return invoke_admitted(kind, messages, schema)

    def invoke_admitted(kind, messages, schema):
        connection_retry = state.get("pending_connection_retry")
        if connection_retry is not None and state["inflight"] is None:
            if connection_retry["kind"] != kind:
                raise RuntimeError("Connection retry does not match durable phase")
            messages = json.loads((folder / connection_retry["failed_folder"] / "input.json").read_text(encoding="utf-8"))
            with performance.measure('retry_wait'):
                while time.time() < connection_retry.get('not_before', 0):
                    command = store.control(run_id)
                    if command in {'pause', 'stop'}:
                        raise ControlRequested(command)
                    heartbeat()
                    time.sleep(min(0.1, max(0, connection_retry['not_before'] - time.time())))
                if model_slots is not None:
                    model_slots.wait_retry(heartbeat, check_control)
        retry = state.get("pending_format_retry")
        if retry is not None and state["inflight"] is None:
            if retry["kind"] != kind:
                raise RuntimeError("Format retry does not match durable phase")
            messages = format_retry_messages(messages, schema, retry["errors"])
        pending = state["inflight"]
        if pending is not None:
            if pending["kind"] != kind:
                raise RuntimeError("Inflight stage does not match durable phase")
            call_folder = folder / pending["folder"]
            messages = json.loads((call_folder / "input.json").read_text(encoding="utf-8"))
        else:
            command = store.control(run_id)
            if command in {"pause", "stop"}:
                raise ControlRequested(command)
            if state["model_calls_used"] >= max_model_calls:
                raise BudgetExhausted()
            call_key = f"model/{state['model_calls_used']:02d}-{kind}"
            call_folder = folder / call_key
            atomic_json(call_folder / "input.json", messages)
            state["inflight"] = {"kind": kind, "folder": call_key}
            if connection_retry is not None:
                state['inflight']['connection_retry_attempt'] = connection_retry.get('attempt', 1)
                state['inflight']['format_retry_attempt'] = connection_retry.get('format_retry_attempt', 0)
                state['inflight']['provider_retry_attempt'] = connection_retry.get('provider_retry_attempt', 0)
            if retry is not None:
                state['inflight']['format_retry_attempt'] = retry.get('attempt', 1)
            state.pop("pending_format_retry", None)
            state.pop("pending_connection_retry", None)
            state["model_calls_used"] += 1
            try:
                save("model_call_reserved", {"kind": kind, "folder": call_key,
                                             "input_sha256": hashlib.sha256((call_folder / "input.json").read_bytes()).hexdigest()})
            except CallLimit:
                state["model_calls_used"] -= 1
                state["inflight"] = None
                raise BudgetExhausted()
        completed = call_folder / "parsed-output.json"
        if completed.exists():
            return schema.model_validate_json(completed.read_text(encoding="utf-8"))
        try:
            with performance.measure('model_call:' + kind):
                output = calls.call(messages, schema, call_folder, heartbeat=heartbeat,
                    **({'request_options':state['provider_request_options']} if state.get('provider_request_options') else {}))
        except ModelCallFailure as exc:
            trace = call_folder / "result.json"
            failure = json.loads(trace.read_text(encoding="utf-8")) if trace.exists() else {}
            if release_auth_rejection(state, failure):
                save('auth_rejection_reservation_released', {
                    'folder': state['inflight']['folder'], 'http_status': 401,
                    'result_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
                    'reason': 'invalid_api_key; no inference response or stream chunks'})
                raise
            option = request_repair_option(failure, state.get('provider_request_options', {}))
            if option and state['model_calls_used'] < max_model_calls:
                failed_folder = state['inflight']['folder']
                state.setdefault('provider_request_options', {})[option] = False
                state['request_repairs_used'] = state.get('request_repairs_used', 0) + 1
                state['inflight'] = None
                save('request_parameter_repair_scheduled', {'kind':kind, 'failed_folder':failed_folder,
                    'omitted_optional_parameter':option, 'reason':failure['provider_diagnosis']})
                return invoke_admitted(kind, messages, schema)
            retry_attempt = state['inflight'].get('connection_retry_attempt', 0)
            format_attempt = state['inflight'].get('format_retry_attempt', 0)
            provider_attempt = state['inflight'].get('provider_retry_attempt', 0)
            delay = transient_retry_delay(failure, provider_attempt)
            if delay is not None and state['model_calls_used'] < max_model_calls:
                state['provider_retries_used'] = state.get('provider_retries_used', 0) + 1
                state['inflight'] = None
                state['pending_connection_retry'] = {'kind': kind,
                    'failed_folder': call_folder.relative_to(folder).as_posix(),
                    'attempt': retry_attempt, 'format_retry_attempt': format_attempt,
                    'provider_retry_attempt': provider_attempt + 1, 'not_before': time.time() + delay}
                if model_slots is not None:
                    model_slots.backoff(delay)
                save('provider_retry_scheduled', {'http_status': failure.get('http_status'),
                    'failed_call_folder': call_folder.relative_to(folder).as_posix(),
                    'attempt': provider_attempt + 1, 'delay_seconds': delay})
                return invoke_admitted(kind, messages, schema)
            if (str(exc) == "connection_failed" and failure.get("remote_outcome") == "not_sent"
                    and failure.get("retryable") is True and retry_attempt < len(CONNECTION_RETRY_DELAYS)
                    and state["model_calls_used"] < max_model_calls):
                state["connection_retries_used"] = state.get("connection_retries_used", 0) + 1
                state["inflight"] = None
                delay = CONNECTION_RETRY_DELAYS[retry_attempt]
                state["pending_connection_retry"] = {"kind": kind, "failed_folder": call_folder.relative_to(folder).as_posix(),
                    "attempt": retry_attempt + 1, 'not_before': time.time() + delay,
                    'format_retry_attempt': format_attempt, 'provider_retry_attempt': provider_attempt}
                save("connection_retry_scheduled", {"failed_call_folder": str(call_folder.relative_to(folder)),
                     "remote_outcome": "not_sent", "network_diagnosis": failure.get("network_diagnosis"),
                     'attempt': retry_attempt + 1, 'delay_seconds': delay,
                     "scope": "bounded retries per node within original target and task budgets"})
                return invoke_admitted(kind, messages, schema)
            format_attempt = state['inflight'].get('format_retry_attempt', 0)
            if str(exc) != "invalid_output" or format_attempt >= FORMAT_RETRY_LIMIT or state["model_calls_used"] >= max_model_calls:
                raise
            # A recorded invalid response has a known outcome. This new call counts
            # against the same budget; unknown network outcomes never enter this path.
            errors = failure.get("validation_errors", [])
            state["format_retries_used"] = state.get("format_retries_used", 0) + 1
            state["inflight"] = None
            state["pending_format_retry"] = {"kind": kind, "errors": errors, 'attempt': format_attempt + 1}
            save("format_retry_scheduled", {"failed_call_folder": str(call_folder.relative_to(folder)), "errors": errors,
                'attempt': format_attempt + 1, 'limit': FORMAT_RETRY_LIMIT})
            return invoke_admitted(kind, messages, schema)
        atomic_json(completed, output.model_dump())
        save("model_call_completed", {"kind": kind, "folder": state["inflight"]["folder"]})
        return output

    @performance.measure('sample_identity_check')
    def check_sample_identity(stage):
        from app.quality.sample_review import validate_sample_identity
        target = state.get('binding_target') or initial['binding_target']
        task_item = None
        with store.connect() as db:
            events = [{'kind':'source_view_reselected','payload':json.loads(row['payload'])} for row in db.execute(
                "SELECT payload FROM events WHERE run_id=? AND kind='source_view_reselected' ORDER BY seq",(run_id,))]
            if state.get('batch_id') and not state.get('revalidation_only'):
                task_item = db.execute('SELECT target,packet_id FROM batch_items WHERE job_id=? AND run_id=?',
                    (state['batch_id'],run_id)).fetchone()
        try:
            if state.get('batch_id') and not state.get('revalidation_only'):
                if task_item is None:
                    raise ValueError('运行与任务物体的关联已缺失。')
                current_target = json.loads(task_item['target'])
                if any(current_target.get(k) != target.get(k) for k in ('context','object_id','timestamp')):
                    raise ValueError('任务冻结的物体身份已变化。')
            report = validate_sample_identity(target,state,events,
                **({'item_packet_id':task_item['packet_id']} if task_item is not None else {}))
        except ValueError as exc:
            report = {'passed':False,'error':str(exc),'scope':'sample metadata identity'}
        state['sample_identity_validation'] = {**report,'stage':stage}
        return report['passed']

    def finish(diagnostic_pass, reason):
        if diagnostic_pass and not check_sample_identity('before_success_commit'):
            diagnostic_pass, reason = False, 'sample_identity_binding_failed'
        if state["attempts"] and "counterfactual_plan" in state["attempts"][-1] and "counterfactual" not in state["attempts"][-1]:
            attempt = state["attempts"][-1]
            attempt["counterfactual"] = counterfactual_summary(CounterfactualPlan.model_validate(attempt["counterfactual_plan"]),
                                                               attempt.get("counterfactual_checks", []), reason)
        state.update(quality_route(diagnostic_pass), stop_reason=reason,
                     final_expression=state["attempts"][-1].get("generated", {}).get("expression") if state["attempts"] else None,
                     inflight=None)
        if packet.task.get('annotation_search'):
            state['annotatability'] = 'validated_candidate' if diagnostic_pass else 'not_decided'
            if not diagnostic_pass:
                state['status'] = 'needs_evidence'
        audit = state["attempts"][-1].get("counterfactual") if state["attempts"] else None
        state["quality_gate"]["counterfactual_checks_passed"] = audit.get("passed") if audit else None
        state["quality_gate"]["counterfactual_status"] = audit.get("status", "not_checked") if audit else "not_checked"
        checks = state["attempts"][-1].get("ground_checks", []) if state["attempts"] else []
        if checks:
            state["quality_gate"]["oracle_checks_passed"] = len(checks) == 2 and all(c["diagnostic_pass"] for c in checks)
            if any(c.get('protocol')=='two-round-blind-scene-v1' for c in checks):
                state['quality_gate']['oracle_checks_passed']=False
                state['quality_gate']['scene_grounding_checks_passed']=len(checks)==2 and all(c['diagnostic_pass'] for c in checks)
                state['quality_gate']['validation_protocol']='two-round-blind-scene-v1'
        appearance = state["attempts"][-1].get("appearance_check") if state["attempts"] else None
        state["quality_gate"]["appearance_checks_passed"] = appearance.get("passed") if appearance else None
        state['quality_gate']['contrastive_appearance_observed'] = 'candidate_appearance' in state
        state['quality_gate']['appearance_observed_candidate_count'] = len(state.get('candidate_appearance', {}))
        state['quality_gate']['admission_checks_passed'] = state.get('admission', {}).get('passed')
        state['quality_gate']['sample_identity_passed'] = state.get('sample_identity_validation',{}).get('passed')
        save("run_finished", {"status": state["status"], "reason": reason})
        if not diagnostic_pass:
            diagnostic_event(output_root, 'annotation_not_accepted', run_id=run_id,
                job_id=state.get('batch_id'), phase=state['phase'], reason=reason,
                generated=bool(state.get('final_expression')), model_calls_used=state['model_calls_used'])

    def try_alternate_source(current, reason):
        # Each rejected camera is remembered durably. A new camera needs its own
        # RGB visibility check; positive observations never transfer across views.
        if not current.task.get('view_selection_policy') or state['attempts']:
            return False
        camera = next((v.camera_name for v in current.views if v.camera_label == current.source_view), None)
        rejected = state.setdefault('rejected_source_views', [])
        timestamp = current.frame_timestamp_micros
        if not any(r['camera_name'] == camera and r.get('timestamp', timestamp) == timestamp for r in rejected):
            rejected.append({'camera_name': camera, 'reason': reason,
                'timestamp': timestamp, 'packet_id': current.packet_id,
                'observation': state.get('target_precheck_observation'),
                'admission': state.get('admission'),
                'distinguishing_cue_bound': state.get('distinguishing_cue_bound')})
        tried = {r['camera_name'] for r in rejected if r.get('timestamp', timestamp) == timestamp}
        from app.datasets.view_selection import confirmed_source_views, select_source_view
        safe = source_camera_inventory(current) or {}
        choices = [v for v in confirmed_source_views(current.views, require_complete=not allows_partial(current),
            allow_back_facing=allows_partial(current))
            if v.camera_name not in tried and str(v.camera_name) in safe]
        choices.sort(key=lambda v: -safe[str(v.camera_name)].get('image_selection_score', 0))
        if max_model_calls - state['model_calls_used'] < 4:
            return False
        if choices:
            updated = select_source_view(current, choices[0].camera_name)
        else:
            frames = sorted({r.get('timestamp', timestamp) for r in rejected})
            if alternate_frame is None or len(frames) >= 3:
                return False
            command = store.control(run_id)
            if command in {'pause', 'stop'}:
                raise ControlRequested(command)
            # Preparation and RGB observation are distinct: a new frame must pass
            # its own identity and clear-visibility checks before generation.
            save('alternate_frame_preparation_started', {'rejected_timestamps': frames, 'reason': reason})
            updated = alternate_frame(current, frames, heartbeat)
            if updated is None:
                return False
            if (updated.context != current.context or updated.target_object_id != current.target_object_id
                    or updated.target.category != current.target.category or updated.frame_timestamp_micros in frames
                    or any(updated.task.get(k) != current.task.get(k) for k in
                           ('admission_policy', 'visibility_policy', 'view_selection_policy', 'annotation_search'))):
                raise ValueError('Alternate frame changed target identity or validation protocol')
        if current.task.get('manual_review_feedback'):
            updated = updated.model_copy(deep=True)
            updated.task['manual_review_feedback'] = current.task['manual_review_feedback']
        from app.quality.description_search import traffic_scene_policy
        if traffic_scene_policy(current):
            updated = updated.model_copy(deep=True)
            updated.task['reference_preference_policy'] = current.task['reference_preference_policy']
        for key in ('target_precheck_observation', 'resolved_target_observation', 'candidate_appearance',
                    'admission', 'precheck_batch_index', 'precheck_batch_limit', 'relook_completed',
                    'relook_manifest', 'relook_kind', 'source_identity_rechecked', 'distinguishing_cue_bound',
                    'source_attributes_rechecked', 'source_attribute_recheck_manifest', 'precheck_identity_focus'):
            state.pop(key, None)
        state.pop('reused_scene_observation_ids',None)
        state.pop('precheck_reused_batch_ids',None)
        state.update(current_packet=updated.model_dump(mode='json'), packet_id=updated.packet_id, phase='precheck_target',
                     relations_acquired=False, inflight=None)
        save('source_view_reselected', {'from_camera': camera,
            'camera_name': next(v.camera_name for v in updated.views if v.camera_label == updated.source_view),
            'from_packet_id': current.packet_id, 'packet_id': updated.packet_id,
            'from_timestamp': timestamp, 'timestamp': updated.frame_timestamp_micros,
            'reason': reason, 'remaining_model_calls': max_model_calls - state['model_calls_used']})
        return True

    try:
        save("run_started_or_resumed", {"phase": state["phase"]})
        if not check_sample_identity('before_model_calls'):
            finish(False,'sample_identity_binding_failed')
            return state
        while state["status"] == "running":
            performance.phase(state['phase'])
            command = store.control(run_id)
            if command in {"pause", "stop"}:
                state["status"] = "paused" if command == "pause" else "stopped"
                save("control_applied", {"command": command, "phase": state["phase"],
                                         "resume_safe": state["inflight"] is None})
                break
            current = EvidencePacket.model_validate(state["current_packet"])
            appearance_observations = effective_observations(current, state.get('candidate_appearance')) if 'candidate_appearance' in state else None
            if 'resolved_target_observation' in state:
                appearance_observations = {**(appearance_observations or {}),
                    current.target_object_id: state['resolved_target_observation']}
            phase = state["phase"]
            if phase == 'precheck_target':
                if not current.provenance.get('scene_image_candidates'):
                    state['admission'] = {'passed': False, 'reason': 'precheck_scene_image_inventory_missing'}
                    finish(False, 'precheck_scene_image_inventory_missing')
                    break
                batches = precheck_batches(current)
                # A resumed reserved observation already counts against the budget.
                available = max_model_calls - state['model_calls_used'] + int(state['inflight'] is not None)
                required = 4 if current.task.get('annotation_search') else 4 + len(batches)
                if available < required:
                    finish(False, 'insufficient_validation_budget')
                    break
                observed = invoke('precheck_target', appearance_messages(current), VisualObservation)
                state.update(target_precheck_observation=observed.model_dump(), inflight=None)
                visibility_error = target_visibility_error(current, observed)
                if visibility_error:
                    if try_alternate_source(current, visibility_error):
                        continue
                    state['admission'] = {'passed': False, 'reason': visibility_error}
                    finish(False, visibility_error)
                    break
                if observation_cache is not None:
                    scope=(state.get('provider_scope'),state['model'],state['base_url'],state['code_hashes'])
                    with performance.measure('scene_observation_reuse_lookup'):
                        cached,receipts=observation_cache.reuse(current,[m for b in precheck_batches(current) for m in b],
                            scope,EVIDENCE_ROOT)
                    if receipts:
                        state.setdefault('candidate_appearance',{}).update(cached)
                        state['reused_scene_observation_ids']=sorted(cached)
                        state['observation_source_run_ids']=sorted(set(state.get('observation_source_run_ids',[]))
                            | {r['source_run_id'] for r in receipts})
                        state.setdefault('scene_observation_reuse_receipts',[]).extend(receipts)
                        batches=precheck_batches(current)
                if current.task.get('annotation_search'):
                    available = max_model_calls - state['model_calls_used']
                    # Keep one focused repair/fallback call as well as generation
                    # and both blind checks, rather than consuming all on crops.
                    limit = max(0, available - 4)
                    if allows_partial(current) and available >= 3:
                        # A focused scene batch plus local rendering and two
                        # blind checks fits the last three calls. Clear targets
                        # need this opportunity too; do not skip all evidence
                        # acquisition just because model generation costs more.
                        if available < 3:
                            finish(False, 'insufficient_validation_budget')
                            break
                        limit = max(1, limit)
                    from app.quality.admission import admission_plan, observation_plan_sufficient
                    effective = {**state.get('candidate_appearance',{}),current.target_object_id: observed.model_dump()}
                    probe, _ = expand_relations(current, effective)
                    if observation_plan_sufficient(current,admission_plan(probe, observed, effective)):
                        batches = []
                        state['candidate_appearance'] = effective
                        state['observation_strategy'] = 'visible target plus complete camera identity/order proof'
                    elif current.task.get('observation_policy') == 'identity-first-on-demand-v1':
                        from app.agent.appearance import identity_first_batches
                        focused = identity_first_batches(current, batches)
                        if len(focused) != len(batches) and len(focused) <= limit:
                            # Planning hypothesis only: could a consistent source
                            # identity observation suffice? Never persist this as
                            # observed evidence or use it to accept the target.
                            camera = next(v.camera_name for v in current.views if v.camera_label == current.source_view)
                            hypothetical = {**state.get('candidate_appearance',{}),current.target_object_id: {
                                **observed.model_dump(), 'observed_camera_name': camera}}
                            planned, relations = expand_relations(current, hypothetical, target_observation=observed)
                            if observation_plan_sufficient(current,admission_plan(planned, observed, hypothetical, relations['anchors'])):
                                state['precheck_identity_focus'] = [current.packet_id, current.source_view, current.source_camera_object_id]
                                state['observation_strategy'] = 'source identity first; acquire competitors only if proof still needs them'
                                batches = focused
                    state['precheck_batch_limit'] = min(len(batches), limit)
                    state['unobserved_crop_count'] = sum(len(b) for b in batches[limit:])
                    batches = batches[:limit]
                state['phase'] = 'precheck_candidates' if batches else 'precheck_decision'
                save('precheck_target_observed', {'target_visible': observed.target_visible,
                    'reused_scene_observations':len(state.get('reused_scene_observation_ids',[])),
                    'reuse_model_calls_added':0})
            elif phase == 'precheck_candidates':
                from app.agent.appearance import observation_messages, checked_observations
                batches = precheck_batches(current)
                batches = batches[:state.get('precheck_batch_limit', len(batches))]
                index = state.get('precheck_batch_index', 0)
                manifest = batches[index]
                reused_count=0
                if (observation_cache is not None and manifest and state['inflight'] is None
                        and not state.get('pending_format_retry') and not state.get('pending_connection_retry')):
                    # Other targets may finish observations while this target is
                    # reading earlier batches. Check again just before billing,
                    # without waiting on peers or altering a reserved request.
                    scope=(state.get('provider_scope'),state['model'],state['base_url'],state['code_hashes'])
                    with performance.measure('scene_observation_reuse_lookup'):
                        cached,receipts=observation_cache.reuse(current,manifest,scope,EVIDENCE_ROOT)
                    if receipts:
                        state.setdefault('candidate_appearance',{}).update(cached)
                        state.setdefault('precheck_reused_batch_ids',{})[str(index)]=sorted(cached)
                        state['observation_source_run_ids']=sorted(set(state.get('observation_source_run_ids',[]))
                            | {r['source_run_id'] for r in receipts})
                        state.setdefault('scene_observation_reuse_receipts',[]).extend(receipts)
                        reused_count=len(cached)
                        manifest=[m for m in manifest if m['object_id'] not in cached]
                before_calls=state['model_calls_used']
                if manifest:
                    observation = invoke('precheck_candidates', observation_messages(current, EVIDENCE_ROOT, manifest), SceneVisualObservation)
                    try:
                        records=checked_observations(current, observation, manifest)
                        state.setdefault('candidate_appearance', {}).update(records)
                    except ValueError:
                        state['admission'] = {'passed': False, 'reason': 'precheck_invalid_scene_observation'}
                        finish(False, 'precheck_invalid_scene_observation')
                        break
                    if observation_cache is not None:
                        scope=(state.get('provider_scope'),state['model'],state['base_url'],state['code_hashes'])
                        observation_cache.remember(current,manifest,records,scope,run_id,state['inflight']['folder'])
                sufficient = False
                checked_plan = {}
                visibility_error = None
                if current.task.get('annotation_search'):
                    from app.quality.admission import admission_plan, observation_plan_sufficient
                    primary = VisualObservation.model_validate(state['target_precheck_observation'])
                    visibility_error = target_visibility_error(current, primary, state['candidate_appearance'])
                    if not visibility_error:
                        probe, relations = expand_relations(current, state['candidate_appearance'], target_observation=primary)
                        checked_plan = admission_plan(probe, primary,state['candidate_appearance'], relations['anchors'])
                        sufficient = observation_plan_sufficient(current,checked_plan)
                from app.agent.appearance import source_attribute_conflicts
                conflicts = source_attribute_conflicts(current,
                    VisualObservation.model_validate(state['target_precheck_observation']), state['candidate_appearance'])
                # A focused correction may have consumed an extra request since
                # this schedule was planned. Keep writing and both blind checks.
                stop_observing = (sufficient or bool(visibility_error) or bool(conflicts)
                    or max_model_calls - state['model_calls_used'] <= 3)
                state.update(phase='precheck_decision' if stop_observing or index+1 == len(batches) else 'precheck_candidates',
                    precheck_batch_index=index+1, inflight=None)
                save('precheck_scene_observed', {'observed_candidates': len(state['candidate_appearance']),
                    'batch': index+1, 'batches': len(batches),
                    'fresh_scene_candidates':len(manifest),'reused_scene_candidates':reused_count,
                    'model_calls_added':state['model_calls_used']-before_calls,
                    'remaining_batches_skipped': len(batches)-index-1 if stop_observing else 0,
                    'unique_plan_found': checked_plan.get('unique_plan_found',False),
                    'preferred_reference_found': sufficient, 'visibility_error': visibility_error,
                    'source_attribute_conflicts': conflicts})
            elif phase == 'recheck_source_attributes':
                from app.agent.appearance import observation_messages, checked_observations
                manifest = state['source_attribute_recheck_manifest']
                # Focus on the actual competing units in their marked RGB
                # contexts, without supplying prior answers or suggested labels.
                checked = invoke('recheck_source_attributes',
                    observation_messages(current, EVIDENCE_ROOT, manifest), SceneVisualObservation)
                records = checked_observations(current, checked, manifest)
                target_record = records[current.target_object_id]
                observed = VisualObservation.model_validate({k: v for k, v in target_record.items()
                    if k in VisualObservation.model_fields})
                state.setdefault('source_attribute_rechecks', []).append({
                    'packet_id': current.packet_id, 'source_view': current.source_view,
                    'previous_observation': state['target_precheck_observation'],
                    'previous_candidate_observations': {oid: state.get('candidate_appearance', {}).get(oid)
                        for oid in records}, 'observations': records, 'observation': observed.model_dump()})
                # The earlier anonymous TARGET check remains an independent
                # comparator: disagreements still remove attributes. Focused
                # rival observations supersede the broad-batch readings, with
                # both versions retained above for audit.
                state.setdefault('candidate_appearance', {}).update({oid: record for oid, record in records.items()
                    if oid != current.target_object_id})
                state.update(target_precheck_observation=observed.model_dump(),
                    source_attributes_rechecked=True, phase='precheck_decision', inflight=None)
                save('source_attributes_rechecked', {'source_view': current.source_view})
            elif phase == 'relook_candidates':
                from app.agent.appearance import observation_messages, checked_observations, merge_observations
                manifest = state['relook_manifest']
                observation = invoke('relook_candidates', observation_messages(current, EVIDENCE_ROOT, manifest), SceneVisualObservation)
                records = checked_observations(current, observation, manifest)
                for oid, record in records.items():
                    state.setdefault('candidate_appearance', {})[oid] = merge_observations(
                        state.get('candidate_appearance', {}).get(oid), record, allow_partial=allows_partial(current))
                if state.pop('relook_kind', None) == 'source_identity':
                    state['source_identity_rechecked'] = True
                else:
                    state['relook_completed'] = True
                state.update(phase='precheck_decision', inflight=None)
                save('blocking_objects_reobserved', {'objects': list(records), 'reason': 'resolve missing or conflicting visual evidence before giving up'})
            elif phase == 'precheck_decision':
                from app.quality.admission import admission_plan, observation_plan_sufficient
                observed = VisualObservation.model_validate(state['target_precheck_observation'])
                current, result = expand_relations(current, state.get('candidate_appearance'), target_observation=observed)
                # Keep the exact predicate/landmark catalog behind a failed
                # cue bound as well as behind successful admission.
                state['current_packet'] = current.model_dump(mode='json')
                plan = admission_plan(current, observed, state.get('candidate_appearance', {}), result['anchors'])
                state['admission'] = plan
                state['relations_acquired'] = True
                state['tool_results'].append(result)
                needs_scene_search = plan['passed'] and not observation_plan_sufficient(current, plan)
                search_plan = (admission_plan(current, observed, state.get('candidate_appearance', {}),
                    result['anchors'], allow_pixel_fallback=False)
                    if needs_scene_search and plan.get('unique_plan_found') else plan)
                from app.agent.appearance import source_attribute_conflicts, source_attribute_gaps
                conflicts = source_attribute_conflicts(current, observed, state.get('candidate_appearance'))
                gaps = (source_attribute_gaps(current, observed, state.get('candidate_appearance'))
                    if needs_scene_search else [])
                if gaps and not conflicts:
                    # Acquisition planning only: spend a focused call when
                    # confirming the omitted cues could actually solve the
                    # current ambiguity. Never persist these hypothetical
                    # target attributes or consume the last fallback on a
                    # color shared by every remaining barricade.
                    attributes = [*observed.attributes, *[VisualAttribute.model_validate(a) for a in gaps]]
                    possible = False
                    if len(attributes) <= 6:
                        hypothetical = observed.model_copy(update={'attributes':attributes})
                        probe, relations = expand_relations(current, state.get('candidate_appearance'),
                            target_observation=hypothetical)
                        hypothetical_plan = admission_plan(probe, hypothetical,
                            state.get('candidate_appearance', {}), relations['anchors'])
                        possible = (observation_plan_sufficient(probe, hypothetical_plan)
                            if plan.get('unique_plan_found') else hypothetical_plan.get('unique_plan_found', False))
                    if not possible:
                        gaps = []
                if ((conflicts or gaps) and not state.get('source_attributes_rechecked')
                        and max_model_calls - state['model_calls_used'] >= 3):
                    from app.agent.appearance import source_identity_candidate, precheck_observation_batches
                    selected = source_identity_candidate(current)
                    if selected is not None:
                        blocking = set(search_plan.get('remaining_object_ids', []) + search_plan.get('remaining_image_object_ids', []))
                        manifest = [selected]
                        for batch in precheck_observation_batches(current):
                            for item in batch:
                                if item['object_id'] == current.target_object_id or item['object_id'] not in blocking:
                                    continue
                                same_camera = next((m for m in [item, *item.get('alternatives', [])]
                                    if m.get('camera_name') == selected['camera_name']), item)
                                manifest.append(same_camera)
                                if len(manifest) == 16:
                                    break
                            if len(manifest) == 16:
                                break
                        state.update(phase='recheck_source_attributes', source_attribute_recheck_manifest=manifest, inflight=None)
                        save('source_attribute_recheck_planned', {'conflicting_kinds': conflicts,
                            'omitted_attribute_hints': gaps,
                            'candidate_ids': [m['candidate_id'] for m in manifest],
                            'reason': 'Reobserve missing or contradictory source attributes and blocking rivals before acquiring unrelated evidence'})
                        continue
                if (plan['reason'] == 'precheck_target_identity_unconfirmed' and allows_partial(current)
                        and not state.get('source_identity_rechecked')
                        and max_model_calls - state['model_calls_used'] >= 3):
                    from app.agent.appearance import source_identity_candidate
                    item = source_identity_candidate(current)
                    if item is not None:
                        # One focused anonymous check before rejecting the source.
                        # Reserve both final blind checks even with local rendering.
                        state.update(phase='relook_candidates', relook_manifest=[item],
                            relook_kind='source_identity', inflight=None)
                        save('source_identity_recheck_planned', {'camera_name': item['camera_name'],
                            'candidate_id': item['candidate_id'], 'reason': plan['reason']})
                        continue
                # An attribute/identity correction interrupts the frozen crop
                # schedule; it must not discard its untouched tail. Even a
                # persistent color conflict may leave useful subtype/relations.
                # Resume by durable index, before generic relooks/frame changes,
                # only when the target remains admissible and calls still fit.
                pending_batches = precheck_batches(current)[:state.get('precheck_batch_limit', 0)]
                pending_index = state.get('precheck_batch_index', 0)
                if (needs_scene_search and (state.get('source_attributes_rechecked') or not conflicts)
                        and pending_index < len(pending_batches)
                        and max_model_calls - state['model_calls_used'] >= 4):
                    state.update(phase='precheck_candidates', inflight=None)
                    save('precheck_observation_plan_resumed', {
                        'next_batch':pending_index+1, 'remaining_batches':len(pending_batches)-pending_index,
                        'source_attribute_conflicts':conflicts,
                        'reason':'finish interrupted scheduled evidence before relooks or switching source'})
                    continue
                identity_conflicts = []
                if (plan['passed'] and current.task.get('annotation_search') and not plan.get('unique_plan_found')):
                    from app.quality.admission import distinguishing_cue_bound, native_identity_conflicts
                    bound = distinguishing_cue_bound(current, observed, state.get('candidate_appearance', {}))
                    if not bound['can_distinguish']:
                        identity_conflicts = native_identity_conflicts(current, bound['remaining_object_ids'])
                        if identity_conflicts:
                            state['distinguishing_cue_bound'] = {**bound,'native_identity_conflicts':identity_conflicts}
                            save('native_identity_ambiguity_detected', state['distinguishing_cue_bound'])
                            # Prefer a fresh source for the unresolved native
                            # identities while its full retry still fits. If
                            # unavailable, keep the ordinary bounded relook.
                            if try_alternate_source(current, 'precheck_native_identity_ambiguous'):
                                continue
                if (needs_scene_search and current.task.get('annotation_search')
                        and not state.get('relook_completed') and max_model_calls - state['model_calls_used'] >= 4):
                    if (not plan.get('unique_plan_found')
                            and current.task.get('landmark_description_policy')=='spatially-qualified-v1'
                            and max_model_calls-state['model_calls_used']==4
                            and try_alternate_source(current,'scene_relations_unresolved_reserve_complete_retry')):
                        # A generic reread here would leave only three calls and
                        # permanently close the same-track view/frame fallback.
                        continue
                    # An edge proof has already removed its rivals. Plan the
                    # quality repair against the ordinary-cue pool instead.
                    blocking = set(search_plan.get('remaining_object_ids', []) + search_plan.get('remaining_image_object_ids', []))
                    from app.agent.appearance import landmark_observation_blockers
                    landmark_requirements = landmark_observation_blockers(current, state.get('candidate_appearance'), blocking,
                        with_requirements=True)
                    landmark_blockers = set(landmark_requirements)
                    blocking.update(landmark_blockers)
                    manifest = []
                    source_camera = next(v.camera_name for v in current.views if v.camera_label == current.source_view)
                    landmark_priority = {oid:index for index,oid in enumerate(landmark_requirements)}
                    items = sorted(current.provenance.get('appearance_candidates', []),
                        key=lambda m:landmark_priority.get(m['object_id'],len(landmark_priority)))
                    for item in items:
                        if current.task.get('visibility_policy') and item['object_id'] == current.target_object_id:
                            continue  # The immutable source view already has its own visibility check.
                        record = state.get('candidate_appearance', {}).get(item['object_id'], {})
                        if item['object_id'] not in blocking:
                            continue
                        from app.quality.relations import VEHICLE_HEADING_POLICY
                        if (current.task.get('vehicle_heading_policy') == VEHICLE_HEADING_POLICY
                                and item['object_id'] in landmark_requirements):
                            source = next((m for m in [item,*item.get('alternatives',[])]
                                if m.get('camera_name') == source_camera),None)
                            if source is not None:
                                # Asking specifically about an unknown attribute
                                # is not a duplicate ordinary read of these pixels.
                                manifest.append({**source,'attribute_recheck':landmark_requirements[item['object_id']]})
                                if len(manifest) == 16:
                                    break
                                continue
                        alternatives = item.get('alternatives', [])
                        if current.task.get('observation_policy') == 'identity-first-on-demand-v1':
                            from app.agent.appearance import redundant_relook
                            alternatives = [choice for choice in alternatives if not redundant_relook(current,choice,record)]
                        if alternatives:
                            manifest.append(min(alternatives, key=lambda m: (
                                m.get('camera_name') != source_camera if item['object_id'] in landmark_blockers else False,
                                m['camera_name'] == item['camera_name'])))
                        elif not record:
                            manifest.append(item)
                        if len(manifest) == 16:
                            break
                    if manifest:
                        state.update(phase='relook_candidates', relook_manifest=manifest,
                            current_packet=current.model_dump(mode='json'), inflight=None)
                        save('evidence_relook_planned', {'objects': len(manifest),
                            'landmark_comparison_object_ids': sorted(landmark_blockers),
                            'landmark_attribute_requirements':landmark_requirements,
                            'replace_pixel_fallback': bool(plan.get('unique_plan_found'))})
                        continue
                from app.quality.joint_reference import writing_plan
                joint_observations = {**(appearance_observations or {}),current.target_object_id:
                    resolve_target_observation(observed,(appearance_observations or {}).get(current.target_object_id),
                        allow_partial=allows_partial(current)).model_dump()}
                joint_plan = writing_plan(current, joint_observations)
                if (plan['passed'] and not plan.get('unique_plan_found') and joint_plan
                        and joint_plan.get('requires_full_scene') and max_model_calls-state['model_calls_used']>=3):
                    plan.update(joint_plan,reason='joint_reference_requires_sentence_validation')
                if (plan['passed'] and current.task.get('annotation_search')
                        and not plan.get('unique_plan_found')):
                    from app.quality.admission import distinguishing_cue_bound
                    bound = distinguishing_cue_bound(current, observed, state.get('candidate_appearance', {}))
                    state['distinguishing_cue_bound'] = {**bound, **(
                        {'native_identity_conflicts':identity_conflicts} if identity_conflicts else {})}
                    if not bound['can_distinguish'] and not plan.get('requires_full_scene'):
                        # Further wording retries cannot create missing evidence.
                        # Leave the full evidence and rivals for audit; the batch
                        # proceeds to another object under its existing budget.
                        plan.update(passed=False, reason='precheck_native_identity_ambiguous'
                            if identity_conflicts else 'precheck_no_distinguishing_cues')
                        save('distinguishing_cues_exhausted', state['distinguishing_cue_bound'])
                if not plan['passed']:
                    if try_alternate_source(current, plan['reason']):
                        continue
                    finish(False, plan['reason'])
                    break
                current.provenance['admission_plan'] = plan
                state['current_packet'] = current.model_dump(mode='json')
                state['resolved_target_observation'] = resolve_target_observation(
                    observed, effective_observations(current, state.get('candidate_appearance')).get(current.target_object_id),
                    allow_partial=allows_partial(current)).model_dump()
                # Keep the anonymous scene observation intact for replay/audit.
                state.setdefault('candidate_appearance', {}).setdefault(current.target_object_id, observed.model_dump())
                state.update(phase='decide', inflight=None)
                save('annotation_admitted', plan)
            elif phase == "decide":
                context, _ = decision_context(current, state)
                if state["inflight"] is not None:
                    # Reserved calls already count against the budget. Validate resumed
                    # choices against the exact tools offered before reservation.
                    saved_input = folder / state["inflight"]["folder"] / "input.json"
                    context = json.loads(json.loads(saved_input.read_text(encoding="utf-8"))[-1]["content"])
                if len(context["allowed_tools"]) == 1:
                    reason = "no_useful_action_within_budget"
                    if current.task.get("execution_policy") == "evidence-fast-v1":
                        reason = ("description_validation_failed" if state["attempts"] else
                                  "insufficient_validation_budget" if context["budget"]["available_model_calls"] < (5 if current.task.get('contrastive_appearance') and current.provenance.get('appearance_candidates') and 'candidate_appearance' not in state else 4) else
                                  "target_spatially_ambiguous")
                    finish(False, reason)
                    break
                useful = [t for t in context["allowed_tools"] if t["action"] != "abstain"]
                free_expansion = next((t for t in useful if t['action'] == 'expand_relations'), None)
                if (current.task.get("execution_policy") == "evidence-fast-v1" and (len(useful) == 1 or free_expansion)
                        and state["inflight"] is None and not state.get("pending_format_retry")
                        and not state.get("pending_connection_retry")):
                    choice = AgentAction(action=(free_expansion or useful[0])["action"], reason="Verified useful local geometry tool or single bounded action; deterministic routing")
                    save("action_routed_locally", {"action": choice.action, "model_calls": 0})
                else:
                    choice = invoke("decide", [{"role": "system", "content": ACTION_GUIDE},
                                               {"role": "user", "content": json.dumps(context)}], AgentAction)
                allowed = {t["action"] for t in context["allowed_tools"]}
                if choice.action not in allowed:
                    state.update(status="blocked", stop_reason="agent_requested_disallowed_action", inflight=None)
                    save("action_rejected", choice.model_dump())
                    break
                state.update(inflight=None, phase="execute", chosen_action=choice.model_dump())
                save("action_chosen", choice.model_dump())
            elif phase == "execute":
                choice = AgentAction.model_validate(state["chosen_action"])
                if choice.action == "abstain":
                    finish(False, "agent_abstained")
                elif choice.action == "expand_relations":
                    updated, result = expand_relations(current)
                    state["relations_acquired"] = True
                    state["current_packet"] = updated.model_dump(mode="json")
                    state["tool_results"].append(result)
                    state["phase"] = "decide"
                    save("tool_completed", result)
                else:
                    if current.task.get('admission_policy') and not state.get('admission', {}).get('passed'):
                        raise RuntimeError('annotation_admission_required')
                    state["attempts"].append({"attempt": len(state["attempts"]), "ground_checks": []})
                    state["phase"] = ('observe_candidates' if current.task.get('contrastive_appearance')
                        and current.provenance.get('appearance_candidates') and 'candidate_appearance' not in state else 'generate')
                    save("phase_changed", {"phase": state['phase']})
            elif phase == 'observe_candidates':
                from app.agent.appearance import observation_messages, checked_observations
                observation = invoke('observe_candidates', observation_messages(current, EVIDENCE_ROOT), SceneVisualObservation)
                try:
                    state['candidate_appearance'] = checked_observations(current, observation)
                except ValueError:
                    state['attempts'][-1]['verification'] = {'diagnostic_pass': False, 'errors': ['invalid_visual_candidate_observation']}
                    finish(False, 'invalid_visual_candidate_observation')
                    break
                state.update(phase=state.pop('after_candidate_observation', 'generate'), inflight=None)
                save('scene_appearance_observed', {'observed_candidates': len(state['candidate_appearance']),
                    'scope': 'anonymous candidate crops; no target designation or generated sentence'})
            elif phase == "generate":
                previous = state["attempts"][-2] if len(state["attempts"]) > 1 else None
                failed_expressions = [a['generated']['expression'] for a in state['attempts'][:-1]
                    if a.get('generated') and a.get('verification',{}).get('diagnostic_pass') is False]
                generated = None
                from app.quality.joint_reference import writing_plan, writer_messages, WrittenSentence, fallback_sentence
                contract = writing_plan(current,appearance_observations or {},attempts=state['attempts'][:-1])
                from app.quality.spatial_relations import native_only
                if contract is None and previous and native_only(current):
                    finish(False,'precheck_no_distinguishing_cues')
                    break
                if contract:
                    local = (fallback_sentence(current,contract,appearance_observations or {})
                        if ((previous and not native_only(current)) or max_model_calls-state['model_calls_used']<3) and state['inflight'] is None and not state.get('pending_format_retry')
                        and not state.get('pending_connection_retry') else None)
                    sentence = local or invoke('generate',writer_messages(current,contract,previous,
                        appearance_observations or {}),WrittenSentence).expression
                    generated = GeneratedExpression(expression=sentence,predicate_ids=contract['predicate_ids'],
                        visual_attributes=contract['visual_attributes'])
                    state['attempts'][-1].update(writing_contract=contract,
                        generation_source=('local proposal from selected evidence after writer failure; still needs both blind checks'
                            if local else 'model sentence; evidence declarations bound by selected writing contract'))
                if (max_model_calls - state['model_calls_used'] < 3
                        and generated is None
                        and previous is None and state['inflight'] is None
                        and not state.get('pending_format_retry') and not state.get('pending_connection_retry')):
                    from app.agent.expression_rendering import render_verified_plan
                    generated = render_verified_plan(current, appearance_observations)
                if generated is None:
                    generated = invoke("generate", generation_messages(current, previous, appearance_observations,
                        failed_expressions=failed_expressions), GeneratedExpression)
                    state['attempts'][-1]['generation_source'] = 'model writer using confirmed RGB and verified spatial/landmark evidence'
                elif not contract:
                    state['attempts'][-1]['generation_source'] = 'simple verified fallback; fewer than three calls remain; not model inference'
                generated = generated.model_copy(update={"expression": normalize_expression(generated.expression)})
                from app.agent.harness import normalize_visual_predicate_aliases
                generated, alias_repair = normalize_visual_predicate_aliases(current, generated)
                if alias_repair:
                    state['attempts'][-1]['local_visual_declaration_repair'] = alias_repair
                generated, camera_repair = repair_camera_qualifier(current, generated)
                if camera_repair:
                    state['attempts'][-1]['local_camera_repair'] = camera_repair
                from app.agent.harness import repair_comparison_scope
                generated, scope_repair = repair_comparison_scope(current, generated)
                if scope_repair:
                    state['attempts'][-1]['local_comparison_scope_repair'] = scope_repair
                from app.agent.harness import repair_redundant_camera_qualifier
                generated, camera_simplification = repair_redundant_camera_qualifier(current,generated,appearance_observations)
                if camera_simplification:
                    state['attempts'][-1]['local_camera_simplification'] = camera_simplification
                state["attempts"][-1]["generated"] = generated.model_dump()
                policy_errors = expression_policy_errors(generated.expression) + expression_language_errors(generated.expression)
                from app.agent.harness import draft_reference_errors
                policy_errors.extend(draft_reference_errors(current,generated,appearance_observations))
                from app.quality.description_search import descriptive_attributes
                recognition = descriptive_attributes(current, appearance_observations)
                if not set((a.kind,a.value) for a in recognition) <= set((a.kind,a.value) for a in generated.visual_attributes):
                    policy_errors.append('confirmed_recognition_details_missing')
                if current.task.get('description_detail_policy'):
                    from app.agent.harness import description_predicates
                    if not set(description_predicates(current)) <= set(generated.predicate_ids):
                        policy_errors.append('supported_scene_location_missing')
                known_types = {a['value'] for a in (appearance_observations or {}).get(current.target_object_id, {}).get('attributes', [])
                    if a['kind']=='subtype'}
                from app.quality.description_search import minimal_reference_policy
                if (current.task.get('natural_description') and current.target.category in {'vehicle','sign'}
                        and not minimal_reference_policy(current)
                        and len(known_types)==1 and not any(a.kind=='subtype' for a in generated.visual_attributes)):
                    policy_errors.append('confirmed_core_type_missing')
                reviewed_text = current.task.get('manual_review_feedback', {}).get('previous_expression')
                normalized_text = ' '.join(generated.expression.casefold().split()).rstrip('.!?')
                rejected_text = ' '.join(normalize_expression(reviewed_text or '').casefold().split()).rstrip('.!?')
                if reviewed_text and normalized_text == rejected_text:
                    policy_errors.append('manual_review_rejected_text_repeated')
                if any(normalized_text == ' '.join(normalize_expression(a['generated']['expression']).casefold().split()).rstrip('.!?')
                        and set(generated.predicate_ids) == set(a['generated'].get('predicate_ids', []))
                        and {(v.kind,v.value) for v in generated.visual_attributes} ==
                            {(v['kind'],v['value']) for v in a['generated'].get('visual_attributes', [])}
                        for a in state['attempts'][:-1] if a.get('generated')
                        and a.get('verification', {}).get('diagnostic_pass') is False):
                    policy_errors.append('failed_draft_text_repeated')
                if policy_errors:
                    state["attempts"][-1]["verification"] = {"diagnostic_pass": False, "errors": policy_errors,
                        'invalid_predicate_ids':sorted(set(generated.predicate_ids)-{p.predicate_id for p in current.predicates}),
                        "repair_instruction": ("Correct the reported error using a prechecked unique reference. Rebuild predicate_ids from the NEW sentence using only available_target_facts; put appearance in visual_attributes. Never retain an unspoken order or visibility condition after changing a clause. Keep only truthful claims needed for uniqueness; no fixed attribute count, subtype or ego-direction clause is required. Keep explicit camera scope for image facts and do not invent evidence."
                            if minimal_reference_policy(current) else "Use the confirmed specific type, salient appearance and supported ego-relative location, or a verified relation to an identifiable landmark. Retain required_recognition_attributes and required_predicate_ids. Never use bottommost/topmost, second from bottom/top, image corners or abstract outline/height ranks; do not rename them as near/closest or delete a necessary cue without another full uniqueness proof. Use a prechecked unique cue combination, keep camera and ego axes distinct, and never invent attributes, landmarks, distances or motion.")}
                    state.update(phase="decide", inflight=None)
                else:
                    state.update(phase="appearance" if current.task.get('image_first') or generated.visual_attributes or any(v.appearance_identity_source == 'native_box_point_correspondence' for v in current.views) else "ground", inflight=None)
                save("expression_generated", {"attempt": len(state["attempts"]) - 1, "expression": generated.expression})
            elif phase == "appearance":
                attempt = state["attempts"][-1]
                generated = GeneratedExpression.model_validate(attempt["generated"])
                body_color = None
                if "rgb" not in current.task.get("training_input_capabilities", []) or not appearance_views(current):
                    check = {"passed": False, "errors": ["visual_evidence_unavailable"]}
                else:
                    # The precheck observed this immutable source crop without seeing
                    # any generated sentence; reuse it rather than billing twice.
                    if state.get('admission', {}).get('passed'):
                        observation_source = 'precheck'
                        observed = resolve_target_observation(
                            VisualObservation.model_validate(state['target_precheck_observation']),
                            effective_observations(current, state.get('candidate_appearance')).get(current.target_object_id),
                            allow_partial=allows_partial(current))
                    else:
                        cached = state.get('appearance_observation', {})
                        if (cached.get('packet_id') == current.packet_id
                                and cached.get('source_view') == current.source_view
                                and state['inflight'] is None and not state.get('pending_format_retry')
                                and not state.get('pending_connection_retry')):
                            raw = VisualObservation.model_validate(cached['observation'])
                            observation_source = 'run_cache'
                        else:
                            raw = invoke('appearance', appearance_messages(current), VisualObservation)
                            observation_source = 'model_call'
                            # Cache only within this run and immutable source. A
                            # failed identity check may still need a fresh look.
                            if target_visibility_error(current, raw, state.get('candidate_appearance')) is None:
                                state['appearance_observation'] = {'packet_id': current.packet_id,
                                    'source_view': current.source_view, 'observation': raw.model_dump()}
                            else:
                                state.pop('appearance_observation', None)
                        observed = resolve_target_observation(raw, None, allow_partial=allows_partial(current))
                    state['resolved_target_observation'] = observed.model_dump()
                    claimed = {(a.kind, a.value) for a in generated.visual_attributes}
                    confirmed = {(a.kind, a.value) for a in observed.attributes}
                    if ('subtype', 'pickup truck') in confirmed:
                        confirmed.add(('subtype', 'truck'))  # Entailment is one-way.
                    body_color = required_body_color(current, {current.target_object_id: observed.model_dump()})
                    missing_color = body_color is not None and ('color', body_color) not in claimed
                    passed = (target_visibility_error(current, observed, state.get('candidate_appearance')) is None
                        and claimed <= confirmed and not missing_color)
                    errors = [] if passed else ['visual_attributes_unconfirmed']
                    if missing_color:
                        errors = ['confirmed_vehicle_body_color_missing']
                    check = {"passed": passed, "observation": observed.model_dump(),
                             "observation_source": observation_source,
                             "errors": errors,
                             "scope": "separate image observation; same-provider consistency, not independent certification"}
                attempt["appearance_check"] = check
                state["inflight"] = None
                if check["passed"]:
                    state["phase"] = "ground"
                else:
                    attempt["verification"] = {"diagnostic_pass": False, "errors": check["errors"],
                        "repair_instruction": (
                            f"Include the confirmed body color {body_color!r} in the sentence and visual_attributes. Retain necessary distinguishing relations and remove unconfirmed claims."
                            if check['errors'] == ['confirmed_vehicle_body_color_missing'] else
                            "Remove all unconfirmed appearance claims. Retain supported geometry and confirmed details.")}
                    state["phase"] = "decide"
                save("appearance_checked", check)
            elif phase in {"ground", "ground_permuted"}:
                generated = GeneratedExpression.model_validate(state["attempts"][-1]["generated"])
                if (phase == 'ground' and state['inflight'] is None
                        and state['attempts'][-1].get('appearance_check', {}).get('passed') is True):
                    from app.agent.harness import declared_cue_failure
                    failure = declared_cue_failure(current, generated, appearance_observations)
                    if failure:
                        attempt=state['attempts'][-1]
                        attempt['verification'] = failure
                        from app.agent.scene_grounding import available
                        # Only ambiguity can request a new proof. Appearance,
                        # language, evidence identity and truthful claims remain gates.
                        from app.schemas.contracts import GroundResult as IdealParse
                        ideal=verify_tool_grounding(current,generated,IdealParse(candidate_ids=[],
                            parsed_predicate_ids=generated.predicate_ids,parsed_visual_attributes=generated.visual_attributes,
                            unsupported_claims=[]),{},appearance_observations,True)
                        ambiguity={'parsed_expression_not_symbolically_unique','image_expression_not_unique'}
                        fallback=(available(current) and state.get('admission',{}).get('passed') is True
                            and set(ideal['errors'])<=ambiguity and max_model_calls-state['model_calls_used']>=2)
                        if fallback:
                            attempt.update(symbolic_verification=failure,ground_checks=[])
                        state.update(phase='scene_ground' if fallback else 'decide', inflight=None)
                        save('declared_cues_rejected_locally', failure)
                        continue
                seed = 17 + 100 * (len(state["attempts"]) - 1) + (0 if phase == "ground" else 1)
                payload, mapping = claim_payload(current, generated.expression, seed)
                atomic_json(folder / f"private-map-{len(state['attempts']) - 1}-{phase}.json", mapping)
                from app.quality.description_search import minimal_reference_policy
                from app.agent.harness import MINIMAL_CLAIM_PARSE_GUIDE
                parse_guide = MINIMAL_CLAIM_PARSE_GUIDE if minimal_reference_policy(current) else CLAIM_PARSE_GUIDE
                grounded = invoke(phase, [{"role": "system", "content": parse_guide},
                                         {"role": "user", "content": json.dumps({**payload, "output_schema": GroundResult.model_json_schema()})}], GroundResult)
                feedback = verify_tool_grounding(current, generated, grounded, mapping,
                    appearance_observations=appearance_observations,
                    target_appearance_verified=state['attempts'][-1].get('appearance_check', {}).get('passed') is True)
                feedback["permutation_seed"] = seed
                feedback["permutation_scope"] = "scene-wide predicate vocabulary order"
                state["attempts"][-1]["ground_checks"].append(feedback)
                state["attempts"][-1]["verification"] = feedback
                state["inflight"] = None
                if feedback["diagnostic_pass"] and phase == "ground":
                    state["phase"] = "ground_permuted"
                elif feedback["diagnostic_pass"]:
                    plan = plan_counterfactuals(current, generated)
                    state["attempts"][-1]["counterfactual_plan"] = plan.model_dump()
                    state["attempts"][-1]["counterfactual_checks"] = []
                    if plan.status == "not_applicable":
                        state["attempts"][-1]["counterfactual"] = counterfactual_summary(plan, [])
                        finish(True, "oracle_passed_counterfactual_not_applicable")
                    elif max_model_calls - state["model_calls_used"] < len(plan.cases):
                        state["attempts"][-1]["counterfactual"] = counterfactual_summary(plan, [], "insufficient_shared_call_budget")
                        finish(True, "oracle_passed_counterfactual_budget_incomplete")
                    else:
                        state["phase"] = "counterfactual"
                        save("counterfactual_plan_created", plan.model_dump())
                else:
                    state["phase"] = "decide"
                save("validation_completed", feedback)
            elif phase in {'scene_ground','scene_ground_permuted'}:
                from app.agent.scene_grounding import messages, verify as verify_scene, SceneGroundResult
                attempt=state['attempts'][-1]
                seed=601+100*(len(state['attempts'])-1)+(0 if phase=='scene_ground' else 1)
                inputs,mapping,receipt=messages(current,attempt['generated']['expression'],seed)
                atomic_json(folder/f"private-map-{len(state['attempts'])-1}-{phase}.json",mapping)
                answer=invoke(phase,inputs,SceneGroundResult)
                feedback=verify_scene(current,answer,mapping,receipt,generated=attempt['generated'])
                attempt['ground_checks'].append(feedback)
                attempt['verification']=feedback
                state['inflight']=None
                if feedback['diagnostic_pass'] and phase=='scene_ground':
                    state['phase']='scene_ground_permuted'
                elif feedback['diagnostic_pass']:
                    attempt['validation_protocol']='two-round-blind-scene-v1'
                    finish(True,'two_round_blind_scene_grounding_passed')
                else:
                    state['phase']='decide'
                save('scene_grounding_completed',feedback)
            elif phase == "counterfactual":
                attempt = state["attempts"][-1]
                plan = CounterfactualPlan.model_validate(attempt["counterfactual_plan"])
                case = plan.cases[len(attempt["counterfactual_checks"])]
                payload, mapping = blind_payload(current, case.expression, seed=31)
                # Expectations, target identity and case type never enter the model request.
                grounded = invoke("counterfactual_" + case.case_id,
                                  [{"role": "system", "content": GROUND_GUIDE},
                                   {"role": "user", "content": json.dumps({**payload, "output_schema": GroundResult.model_json_schema()})}], GroundResult)
                check = verify_counterfactual(case, grounded, mapping)
                attempt["counterfactual_checks"].append(check)
                state["inflight"] = None
                save("counterfactual_checked", check)
                if len(attempt["counterfactual_checks"]) == len(plan.cases):
                    attempt["counterfactual"] = counterfactual_summary(plan, attempt["counterfactual_checks"])
                    # A reader inconsistency is a validation risk, not evidence that
                    # the known-true original text must be rewritten or published.
                    finish(True, "oracle_and_counterfactual_checks_passed" if attempt["counterfactual"]["passed"] else "oracle_passed_counterfactual_reader_inconsistent")
            else:
                raise RuntimeError("Unknown durable phase")
    except ControlRequested as exc:
        state["status"] = "paused" if exc.command == "pause" else "stopped"
        save("control_applied", {"command": exc.command, "phase": state["phase"], "resume_safe": True})
    except BudgetExhausted:
        finish(False, "model_call_budget_exhausted")
    except ModelCallFailure as exc:
        if state["attempts"] and "counterfactual_plan" in state["attempts"][-1]:
            attempt = state["attempts"][-1]
            attempt["counterfactual"] = counterfactual_summary(CounterfactualPlan.model_validate(attempt["counterfactual_plan"]),
                                                               attempt.get("counterfactual_checks", []), "model_call:" + str(exc))
        state.update(status="blocked", stop_reason="model_call:" + str(exc))
        save("model_call_failed", {"reason": str(exc)})
        from app.agent.model_calls import saved_call_diagnostic
        diagnostic_event(output_root, 'annotation_call_failed', level='ERROR', run_id=run_id,
            job_id=state.get('batch_id'), phase=state['phase'], reason=state['stop_reason'],
            diagnostic=saved_call_diagnostic(folder / state['inflight']['folder']) if state.get('inflight') else {})
    except RuntimeError as exc:
        state.update(status="blocked", stop_reason="runtime:" + str(exc))
        save("runtime_blocked", {"reason": str(exc)})
    finally:
        performance.close('interrupted' if sys.exc_info()[0] else state['status'])
        store.release(run_id, owner)
        if executor_owner is None:
            store.release_executor(executor)
    return state


def revalidate_saved_expression(source_run_id: str, calls, output_root: Path, max_model_calls: int = 3,
                               cumulative_call_limit: int = 12, *, repair_camera=False, repair_scope=False, recovery_batch_id=None,
                               upgrade_scene_grounding=False) -> dict:
    """Recheck a saved draft under this verifier version, without regenerating it.

    The source and prior failure remain intact. A separate, linked run uses the
    ordinary budget, lease, checkpoint and control mechanisms. This is protocol
    development revalidation, not a frozen-method automatic repair experiment.
    """
    store = RunStore(output_root / "runs.sqlite3")
    source = store.snapshot(source_run_id)
    if source.get("kind") or source.get("status") not in TERMINAL:
        raise ValueError("Revalidation requires a completed annotation source")
    generated = next((a["generated"] for a in reversed(source.get("attempts", [])) if a.get("generated")), None)
    generated = GeneratedExpression.model_validate(generated)
    saved = EvidencePacket.model_validate(source["current_packet"])
    if upgrade_scene_grounding:
        from app.agent.scene_grounding import POLICY
        saved.task={**saved.task,'scene_grounding_policy':POLICY}
    camera_repair = None
    if repair_camera:
        generated, camera_repair = repair_camera_qualifier(saved, generated)
        if camera_repair is None or not source.get('target_precheck_observation'):
            raise ValueError('No single verified camera qualifier can be repaired')
    scope_repair = None
    if repair_scope:
        from app.agent.harness import repair_comparison_scope
        generated, scope_repair = repair_comparison_scope(saved, generated)
        if scope_repair is None or not source.get('target_precheck_observation'):
            raise ValueError('No exact comparison adjective scope can be repaired')
    if saved.task.get('admission_policy') and not source.get('admission', {}).get('passed'):
        raise ValueError('Revalidation requires passed image-first admission')
    original = EvidencePacket.model_validate_json((EVIDENCE_ROOT / saved.packet_id / "packet.json").read_text(encoding="utf-8"))
    identity = lambda p: (p.context, p.frame_timestamp_micros, p.target.model_dump(), p.scene.manifest_sha256 if p.scene else None)
    if identity(saved) != identity(original):
        raise ValueError("Source target/scene differs from current evidence")
    if upgrade_scene_grounding:
        original.task={**original.task,'scene_grounding_policy':POLICY}
    if (source.get("model"), source.get("base_url"), source.get("provider_scope")) != (calls.model, calls.base_url, getattr(calls, "scope", "real-model")):
        raise ValueError("Revalidation provider differs from source")
    state = initial_run(original, calls, 0, max_model_calls, cumulative_call_limit)
    existing = store.snapshot(state['run_id'], missing_ok=True)
    if existing and (existing.get("revalidation_source_run_id") != source_run_id
                     or existing.get("attempts", [{}])[0].get("generated") != generated.model_dump()):
        raise ValueError("Revalidation identity collision; refusing to overwrite another result")
    if not existing:
        after_observation = ('appearance' if saved.task.get('image_first') or generated.visual_attributes or
            any(v.appearance_identity_source == 'native_box_point_correspondence' for v in saved.views) else 'ground')
        reuse_observations = (source.get('admission', {}).get('passed') is True
            and bool(source.get('target_precheck_observation'))
            and isinstance(source.get('candidate_appearance'), dict)
            and saved.source_view == original.source_view
            and saved.source_camera_object_id == original.source_camera_object_id
            and saved.views == original.views
            and saved.provenance.get('artifact_sha256') == original.provenance.get('artifact_sha256'))
        compare_candidates = bool(saved.task.get('contrastive_appearance') and saved.provenance.get('appearance_candidates')
            and not camera_repair and not reuse_observations)
        state.update(revalidation_only=True, revalidation_source_run_id=source_run_id,
            revalidation_scope="post-development verifier revalidation; not frozen-method repair success",
            current_packet=saved.model_dump(mode="json"), relations_acquired=True,
            phase='observe_candidates' if compare_candidates else after_observation,
            after_candidate_observation=after_observation,
            attempts=[{"attempt": 0, "generated": generated.model_dump(), "generation_source_run_id": source_run_id, "ground_checks": []}])
        if upgrade_scene_grounding:
            state['revalidation_scene_grounding_policy']=POLICY
        if source.get('admission'):
            state['admission'] = source['admission']
            state['target_precheck_observation'] = source['target_precheck_observation']
        if reuse_observations:
            state['candidate_appearance'] = source['candidate_appearance']
            state['revalidation_reused_candidate_observations'] = True
        if camera_repair:
            state['attempts'][0]['local_camera_repair'] = camera_repair
            if 'candidate_appearance' in source:
                state['candidate_appearance'] = source['candidate_appearance']
        if scope_repair:
            state['attempts'][0]['local_comparison_scope_repair'] = scope_repair
        if recovery_batch_id:
            with store.connect() as db:
                item = db.execute('SELECT target FROM batch_items WHERE job_id=?', (recovery_batch_id,)).fetchall()
            if len(item) != 1 or (json.loads(item[0]['target'])['context'], json.loads(item[0]['target'])['timestamp'],
                    json.loads(item[0]['target'])['object_id']) != (saved.context, saved.frame_timestamp_micros, saved.target_object_id):
                raise ValueError('Recovery task is not bound to this exact source frame and target')
            state['batch_id'] = recovery_batch_id
        store.create(state)
    return run_agent(original, calls, 0, output_root, max_model_calls, cumulative_call_limit, resume_run_id=state["run_id"])
