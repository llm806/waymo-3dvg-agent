import json
import re

from fastapi import APIRouter, HTTPException

from app.agent.harness import RUN_ROOT
from app.infrastructure.storage import RunStore
from app.quality.evaluation import summarize

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


@router.get("")
def index() -> dict:
    path = RUN_ROOT / "index.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    database = RUN_ROOT / "runs.sqlite3"
    if not database.exists():
        return report
    snapshots = RunStore(database).all_runs()
    old = {r["run_id"]: r for r in report.get("run_revisions", report["runs"])}
    latest, revisions = {}, []
    for state in snapshots:
        if state.get("kind") == "counterfactual_audit":
            continue
        # Keep derived post-hoc diagnostics, while live state comes from SQLite.
        row = {**old.get(state["run_id"], {}), **state}
        revisions.append(row)
        latest.setdefault(row["packet_id"], row)
    report.update(runs=list(latest.values()), run_revisions=revisions, processed=len(latest),
                  metrics=summarize(list(latest.values())))
    report["reserved_model_calls"] = sum(s.get("model_calls_used", 0) for s in snapshots)
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    elapsed, actual_calls = 0, 0
    for result_path in RUN_ROOT.glob("*/model/*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        actual_calls += 1
        elapsed += result.get("elapsed_seconds", 0)
        for key in tokens:
            tokens[key] += result.get("usage", {}).get(key + "_tokens", 0) or 0
    for result_path in RUN_ROOT.glob("*/audits/*/*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        actual_calls += 1
        elapsed += result.get("elapsed_seconds", 0)
        for key in tokens:
            tokens[key] += result.get("usage", {}).get(key + "_tokens", 0) or 0
    report.update(tokens=tokens, model_elapsed_seconds=elapsed, actual_model_calls=actual_calls)
    return report


@router.get("/{run_id}/trace")
def trace(run_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{24}", run_id):
        raise HTTPException(404, "Run not found")
    database = RUN_ROOT / "runs.sqlite3"
    if not database.exists():
        raise HTTPException(404, "Run not found")
    events = RunStore(database).events(run_id)
    if not events:
        raise HTTPException(404, "Run not found")
    return {"run_id": run_id, "events": events}
