"""Post-hoc audit of a preserved run using the same trace, lease and call contracts."""

import hashlib
import json
import time
from uuid import uuid4

from app.infrastructure.artifacts import atomic_json, file_sha256
from app.schemas.contracts import EvidencePacket, GeneratedExpression, GroundResult
from app.quality.counterfactuals import VERSION, counterfactual_summary, plan_counterfactuals, verify_counterfactual
from app.agent.harness import GROUND_GUIDE, ROOT, blind_payload
from app.agent.model_calls import ModelCallFailure
from app.infrastructure.storage import CallLimit, RunStore


def audit_run(parent_run_id, calls, output_root, total_target_budget=12):
    """Serial CLI audit. At most two calls; preserve the source run unchanged."""
    store = RunStore(output_root / "runs.sqlite3")
    parent = store.snapshot(parent_run_id)
    scope = getattr(calls, "scope", "real-model")
    if parent.get("kind") == "counterfactual_audit" or parent.get("status") != "calibration_hold" or parent.get("diagnostic_pass") is not True:
        raise ValueError("Audit requires a completed original-expression oracle pass")
    if (parent["model"], parent["base_url"], parent["provider_scope"]) != (calls.model, calls.base_url, scope):
        raise ValueError("Audit model/provider differs from source run")
    packet = EvidencePacket.model_validate(parent["current_packet"])
    generated = GeneratedExpression.model_validate(parent["attempts"][-1]["generated"])
    plan = plan_counterfactuals(packet, generated)
    identity = dict(kind="counterfactual_audit", parent_run_id=parent_run_id, packet_id=packet.packet_id,
                    parent_snapshot_sha256=hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest(),
                    protocol=VERSION, model=calls.model, base_url=calls.base_url, provider_scope=scope,
                    total_target_budget=total_target_budget,
                    code_sha256={name: file_sha256(ROOT / "app" / name) for name in
                                 ["quality/counterfactual_audit.py", "quality/counterfactuals.py", "agent/harness.py", "agent/model_calls.py", "infrastructure/storage.py"]})
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    folder = output_root / parent_run_id / "audits" / run_id
    owner = uuid4().hex
    store.acquire_executor(owner)
    try:
        state = store.claim(run_id, owner, {**identity, "run_id": run_id, "status": "running", "model_calls_used": 0,
                                          "current_packet": packet.model_dump(mode="json"), "cumulative_call_limit": total_target_budget,
                                          "inflight": None, "checks": [], "auto_published": False, "plan": plan.model_dump()})
    except Exception:
        store.release_executor(owner)
        raise
    terminal = {"calibration_hold", "blocked"}

    def mirror():
        atomic_json(folder / "checkpoint.json", state)

    def save(kind, payload):
        store.renew_executor(owner)
        store.save(run_id, owner, state, kind, payload)
        mirror()

    def publish_report():
        report = state.get("report", counterfactual_summary(plan, state["checks"], state.get("stop_reason")))
        atomic_json(output_root / parent_run_id / "counterfactual-audit.json", {**report, "audit_run_id": run_id,
                    "parent_run_id": parent_run_id, "model_calls_used": state["model_calls_used"], "run_status": state["status"],
                    "post_hoc": True})

    if state["status"] in terminal:
        publish_report()
        store.release_executor(owner)
        return state
    renewed = time.perf_counter()

    def heartbeat():
        nonlocal renewed
        if time.perf_counter() - renewed >= 30:
            store.renew(run_id, owner)
            store.renew_executor(owner)
            renewed = time.perf_counter()

    try:
        save("audit_started_or_resumed", {"parent_run_id": parent_run_id})
        if plan.status == "ready":
            consumed = store.target_usage(packet, scope)
            if consumed + len(plan.cases) - len(state["checks"]) - bool(state["inflight"]) > total_target_budget:
                raise RuntimeError("Original and audit calls exceed the authorized target budget")
            for case in plan.cases[len(state["checks"]):]:
                payload, mapping = blind_payload(packet, case.expression, seed=31)
                stage = folder / case.case_id
                messages = [{"role": "system", "content": GROUND_GUIDE},
                            {"role": "user", "content": json.dumps({**payload, "output_schema": GroundResult.model_json_schema()})}]
                if state["inflight"] is None:
                    atomic_json(stage / "input.json", messages)
                    state["inflight"] = case.case_id
                    state["model_calls_used"] += 1
                    try:
                        save("model_call_reserved", {"case_id": case.case_id, "input_sha256": file_sha256(stage / "input.json")})
                    except CallLimit:
                        state["model_calls_used"] -= 1
                        state["inflight"] = None
                        raise RuntimeError("Cumulative track budget exhausted")
                elif state["inflight"] != case.case_id:
                    raise RuntimeError("Pending audit does not match its case")
                else:
                    messages = json.loads((stage / "input.json").read_text(encoding="utf-8"))
                parsed = stage / "parsed-output.json"
                grounded = GroundResult.model_validate_json(parsed.read_text()) if parsed.exists() else calls.call(messages, GroundResult, stage, heartbeat=heartbeat)
                atomic_json(parsed, grounded.model_dump())
                check = verify_counterfactual(case, grounded, mapping)
                state["checks"].append(check)
                state["inflight"] = None
                save("counterfactual_checked", check)
        state.update(status="calibration_hold", report=counterfactual_summary(plan, state["checks"]))
        save("audit_finished", {"status": state["report"]["status"]})
    except (ModelCallFailure, RuntimeError) as exc:
        state.update(status="blocked", stop_reason=str(exc))
        save("audit_blocked", {"reason": str(exc)})
    finally:
        store.release(run_id, owner)
        store.release_executor(owner)
    publish_report()
    return state


def recheck_saved_audit(parent_run_id, output_root):
    """Re-score recorded responses locally; preserve source run, checks and billing."""
    store = RunStore(output_root / "runs.sqlite3")
    parent = store.snapshot(parent_run_id)
    source = json.loads((output_root / parent_run_id / "counterfactual-audit.json").read_text(encoding="utf-8"))
    audit = store.snapshot(source["audit_run_id"])
    if audit.get("kind") != "counterfactual_audit" or audit.get("parent_run_id") != parent_run_id:
        raise ValueError("Audit/source identity mismatch")
    from app.quality.counterfactuals import CounterfactualPlan
    plan = CounterfactualPlan.model_validate(audit["plan"])
    packet = EvidencePacket.model_validate(parent["current_packet"])
    folder = output_root / parent_run_id / "audits" / audit["run_id"]
    checks = []
    for case in plan.cases:
        stage = folder / case.case_id
        parsed = stage / "parsed-output.json"
        if not parsed.exists():
            break
        payload, mapping = blind_payload(packet, case.expression, seed=31)
        saved = json.loads((stage / "input.json").read_text(encoding="utf-8"))
        saved_payload = json.loads(saved[-1]["content"])
        expected_candidates = json.loads(json.dumps(payload["candidates"]))
        if saved_payload["expression"] != case.expression or saved_payload["candidates"] != expected_candidates:
            raise ValueError("Recorded blind request does not match source frame/expression")
        grounded = GroundResult.model_validate_json(parsed.read_text(encoding="utf-8"))
        if audit["provider_scope"] == "real-model":
            raw = json.loads((stage / "result.json").read_text(encoding="utf-8"))
            if raw["status"] != "completed" or GroundResult.model_validate_json(raw["answer"]) != grounded:
                raise ValueError("Parsed output does not match completed provider response")
        checks.append(verify_counterfactual(case, grounded, mapping))
    report = {**counterfactual_summary(plan, checks), "source_audit_run_id": audit["run_id"], "parent_run_id": parent_run_id,
              "verifier_sha256": file_sha256(ROOT / "app/quality/counterfactuals.py"), "external_model_calls": 0,
              "analysis_scope": "local re-scoring of recorded outputs; source checks and runtime gates are unchanged"}
    atomic_json(folder / f"recheck-{report['verifier_sha256'][:16]}.json", report)
    atomic_json(output_root / parent_run_id / "counterfactual-analysis.json", report)
    return report
