from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import json
import sqlite3
import os
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.routers import annotations, datasets, evidence, console, batches, research, system
from app.agent.console import ConsoleRunner
from app.agent.batch import BatchService, method_fingerprint
from app.quality.research import ResearchStore
from app.infrastructure.storage import RunStoreUnavailable
from app.infrastructure.diagnostics import diagnostic_event
from app.routers.samples import router as sample_review_router
from app.datasets.export_jobs import ExportJobs
from app.infrastructure.deletion_operations import DeletionOperations

LOG_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "annotation"
LOADED_METHOD_SHA256 = method_fingerprint()


async def recover_batches(service, exports):
    while True:
        try:
            exports.expire()
            await asyncio.to_thread(service.recover_running)
        except Exception as exc:
            diagnostic_event(LOG_ROOT, 'batch_recovery_failed', level='ERROR', exception=exc)
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    app.state.console = ConsoleRunner()
    app.state.batches = BatchService(app.state.console)
    app.state.batches.loaded_method_sha256 = LOADED_METHOD_SHA256
    app.state.batches.recover_deletions()
    app.state.research = ResearchStore(app.state.console.store)
    app.state.exports = ExportJobs(app.state.batches, Path(__file__).resolve().parents[1])
    app.state.deletions = DeletionOperations(app.state.batches)
    diagnostic_event(LOG_ROOT, "service_started", pid=os.getpid())
    recovery = asyncio.create_task(recover_batches(app.state.batches, app.state.exports))
    try:
        yield
    finally:
        recovery.cancel()
        try:
            await recovery
        except asyncio.CancelledError:
            pass
        await asyncio.to_thread(app.state.exports.close)
        await asyncio.to_thread(app.state.deletions.close)
        app.state.batches.close()
        await asyncio.to_thread(app.state.console.close)
        diagnostic_event(LOG_ROOT, "service_stopped", pid=os.getpid())

app = FastAPI(title="Waymo 3DVG Agent", version="0.1.0", lifespan=lifespan,
              description="Evidence-grounded annotation for 3D visual grounding research.")
app.include_router(datasets.router)
app.include_router(evidence.router)
app.include_router(annotations.router)
app.include_router(console.router)
app.include_router(batches.router)
app.include_router(sample_review_router)
app.include_router(research.router)
app.include_router(system.router)


@app.middleware("http")
async def request_diagnostics(request, call_next):
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        diagnostic_event(LOG_ROOT, "request_failed", level="ERROR", exception=exc,
                         method=request.method, path=request.url.path)
        raise
    # Keep routine polling quiet, but retain slow reads for real service latency
    # analysis. This measures time to response headers, not streaming downloads.
    # No query strings, bodies, headers or credentials.
    elapsed = time.monotonic() - started
    slow_read = request.method == 'GET' and elapsed >= 1.0
    if request.url.path.startswith("/api/") and (request.method != "GET" or response.status_code >= 400 or slow_read):
        diagnostic_event(LOG_ROOT, "api_request", method=request.method,
                         path=getattr(request.scope.get('route'), 'path', request.url.path),
                         status=response.status_code, elapsed_seconds=round(elapsed, 3), slow_read=slow_read)
    return response


@app.exception_handler(RunStoreUnavailable)
async def store_unavailable(request, exc):
    diagnostic_event(LOG_ROOT, "store_unavailable", level="ERROR", exception=exc, path=request.url.path)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(json.JSONDecodeError)
async def corrupt_artifact(request, exc):
    diagnostic_event(LOG_ROOT, "artifact_json_invalid", level="ERROR", exception=exc, path=request.url.path)
    return JSONResponse(status_code=409,content={'detail':'已保存的 JSON 数据不完整或损坏，已停止读取；请检查本地记录，不会将其当作有效结果。','code':'artifact_json_invalid'})


@app.exception_handler(sqlite3.Error)
async def database_unavailable(request, exc):
    diagnostic_event(LOG_ROOT, "database_unavailable", level="ERROR", exception=exc, path=request.url.path)
    return JSONResponse(status_code=503,content={'detail':'运行数据库暂时无法读取，请检查本地服务与数据库。','code':'run_database_unavailable'})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "stage": "annotation-and-independent-audit; quality-calibration-pending", "version": app.version, "workspace_id":system.workspace_id(),
            'loaded_method_sha256': LOADED_METHOD_SHA256, 'disk_method_sha256': method_fingerprint()}


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
