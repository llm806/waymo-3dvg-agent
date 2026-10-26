"""Live console routes. Browsing never starts a model call."""


from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.routers.dependencies import Control, Id, mutation_origin
from app.agent.console import call_details, load_packet
from app.infrastructure.storage import RunBusy
from app.datasets.evidence import EVIDENCE_ROOT
import json

router = APIRouter(prefix="/api/console", tags=["console"])


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    packet_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    max_model_calls: int = Field(default=12, ge=1, le=12)
    max_repairs: int = Field(default=2, ge=0, le=2)


def runner(request):
    return request.app.state.console


@router.get("")
def index(request: Request) -> dict:
    worker = runner(request)
    runs = worker.store.all_runs()
    usage = {}
    for state in runs:
        if state.get("provider_scope") == "real-model":
            packet_id = state["packet_id"]
            usage[packet_id] = usage.get(packet_id, 0) + state.get("model_calls_used", 0)
    path = EVIDENCE_ROOT / "index.json"
    if path.exists():
        totals = {}
        for s in runs:
            if s.get("provider_scope") == "real-model":
                key = worker.store.target_key(s)
                totals[key] = totals.get(key, 0) + s.get("model_calls_used", 0)
        for p in json.loads(path.read_text(encoding="utf-8"))["packets"]:
            usage[p["packet_id"]] = totals.get((p["context"], p["object_id"]), usage.get(p["packet_id"], 0))
    active = worker.active or next((s["run_id"] for s in runs if s.get("status") == "running"
                                   and worker.store.executor_active(s["run_id"])), None)
    return {"active_run_id": active, "active_batch_id": worker.active_batch, "single_worker": True, "poll_interval_ms": 1500,
            "cumulative_target_call_limit": 12,
            "target_call_usage": usage,
            "runs": [{k: s.get(k) for k in ("run_id", "packet_id", "context", "category", "status", "phase",
                        "kind", "created_at", "model_calls_used", "model_call_budget", "stop_reason", "console_managed")}
                     for s in runs]}


@router.post("/runs", status_code=202)
def start(body: Start, request: Request, response: Response) -> dict:
    mutation_origin(request)
    try:
        receipt = runner(request).start(load_packet(body.packet_id), body.max_model_calls, body.max_repairs)
    except RunBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    state = runner(request).store.snapshot(receipt.run_id)
    if receipt.reused_result:
        response.status_code = 200
    return {"run_id": receipt.run_id, "packet_id": state["packet_id"],
            "reused_result": receipt.reused_result, "status": state["status"],
            "phase": state.get("phase"), "stop_reason": state.get("stop_reason"),
            "model_calls_used": state.get("model_calls_used", 0)}


@router.get("/runs/{run_id}")
def detail(run_id: Id, request: Request) -> dict:
    worker = runner(request)
    try:
        state = worker.store.snapshot(run_id)
    except ValueError as exc:
        raise HTTPException(404, "Run not found") from exc
    published=[]
    with worker.store.connect() as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='label_releases'").fetchone():
            published=[{'release_id':m['release_id'],'records':m['records']} for r in db.execute('SELECT manifest FROM label_releases')
                       if run_id in (m:=json.loads(r['manifest'])).get('run_ids',[])]
    return {"state": state, "published_releases":published, "events": worker.store.events(run_id),
            "control_requested": worker.store.control(run_id),
            "executor_active": worker.store.executor_active(run_id),
            "local_worker_active": worker.active == run_id,
            "calls": call_details(worker.root, state)}


@router.post("/runs/{run_id}/control", status_code=202)
def control(run_id: Id, body: Control, request: Request) -> dict:
    mutation_origin(request)
    worker = runner(request)
    try:
        state = worker.store.snapshot(run_id)
        if body.command == "resume":
            worker.start(load_packet(state["packet_id"]), state["model_call_budget"], state["max_repairs"], run_id)
        else:
            worker.store.request_control(run_id, body.command)
    except RunBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"run_id": run_id, "command": body.command, "acknowledged": True}
