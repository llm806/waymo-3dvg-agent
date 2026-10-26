"""Explicit denominators: runtime diagnostics are not independent annotation quality."""

from collections import Counter
import hashlib
import json


def rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def first_draft_outcome(run: dict) -> bool | None:
    if run.get("revalidation_only"):
        return None
    if not run.get("attempts"):
        return None
    first = run["attempts"][0]
    feedback = first.get('verification') or {}
    if feedback.get('diagnostic_pass') is False:
        return False  # Rules/appearance can reject a draft before a blind call.
    if "ground_checks" in first:
        checks = first["ground_checks"]
        if any(not check.get("diagnostic_pass") for check in checks):
            return False
        # A successful first order followed by a failed API request is incomplete.
        return True if len(checks) == 2 else None
    feedback = first.get("verification")
    return bool(feedback.get("diagnostic_pass")) if feedback else None


def final_oracle_pass(run: dict) -> bool:
    if run.get("revalidation_only"):
        return bool(run.get("diagnostic_pass"))
    if run.get("attempts"):
        return first_draft_outcome({"attempts": [run["attempts"][-1]]}) is True
    return bool(run.get("diagnostic_pass"))


def summarize(runs: list[dict]) -> dict:
    first_checked = [r for r in runs if first_draft_outcome(r) is not None]
    first_failed = [r for r in first_checked if first_draft_outcome(r) is False]
    passed = sum(final_oracle_pass(r) for r in runs)
    scopes = sorted({r.get("provider_scope", "unspecified") for r in runs})
    counterfactuals = [r.get("counterfactual_audit") or (r.get("attempts") or [{}])[-1].get("counterfactual", {}) for r in runs]
    cf_status = Counter(c.get("status", "not_checked") for c in counterfactuals)
    cf_complete = cf_status["passed"] + cf_status["failed"]
    strata = {}
    protocols = {}
    for run in runs:
        protocol = run.get("version", "unrecorded")
        if run.get('code_hashes'):
            identity={k:run.get(k) for k in ['code_hashes','model','base_url','uv_lock_sha256','max_repairs']}
            protocol+=' / '+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
        group_protocol = protocols.setdefault(protocol, {"workload": 0, "final_oracle_pass": 0})
        group_protocol["workload"] += 1
        group_protocol["final_oracle_pass"] += final_oracle_pass(run)
        key = run.get("context", "unknown") + "/" + run.get("category", "unknown")
        group = strata.setdefault(key, {"total": 0, "diagnostic_pass": 0, "auto_published": 0})
        group["total"] += 1
        group["diagnostic_pass"] += bool(run.get("diagnostic_pass"))
        group["auto_published"] += bool(run.get("auto_published"))
    return {"scope": "oracle native-box runtime diagnostics; not point-cloud grounding or gold quality",
            "provider_scopes": scopes, "workload_total": len(runs),
            "method_protocols": protocols,
            "status_counts": dict(Counter(r["status"] for r in runs)),
            "final_oracle_pass_per_workload": rate(passed, len(runs)),
            "first_oracle_pass_per_checked_draft": rate(sum(first_draft_outcome(r) is True for r in first_checked), len(first_checked)),
            "first_draft_incomplete": sum(not r.get("revalidation_only") and bool(r.get("attempts")) and first_draft_outcome(r) is None for r in runs),
            "revalidation_only_workloads": sum(bool(r.get("revalidation_only")) for r in runs),
            "repair_success_per_initial_failure": rate(sum(final_oracle_pass(r) for r in first_failed), len(first_failed)),
            "auto_publish_per_workload": rate(sum(bool(r.get("auto_published")) for r in runs), len(runs)),
            "reserved_model_calls": sum(r.get("model_calls_used", 0) for r in runs),
            "elapsed_active_seconds": sum(r.get("elapsed_active_seconds", 0) for r in runs),
            "counterfactuals": {"scope": "same-model oracle semantic checks; uncovered/incomplete remain in workload denominator",
                                "status_counts": dict(cf_status), "complete_per_workload": rate(cf_complete, len(runs)),
                                "pass_per_complete": rate(cf_status["passed"], cf_complete),
                                "pass_per_workload": rate(cf_status["passed"], len(runs))},
            "strata": strata,
            "independent_quality": {"regrounding_accuracy": None, "uniqueness_rate": None,
                                    "factual_error_rate": None, "human_acceptance_rate": None,
                                    "human_edit_rate": None,
                                    "reason": "independent held-out annotations and audit results not available"}}
