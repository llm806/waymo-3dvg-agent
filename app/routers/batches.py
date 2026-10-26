"""Catalog browsing and explicit batch controls. GET requests never dispatch work."""

import json
import threading
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from app.agent.batch import BatchSpec
from app.datasets.catalog import load_catalog
from app.routers.dependencies import Control, Id, mutation_origin
from app.infrastructure.storage import RunBusy

router = APIRouter(prefix="/api/batches", tags=["batches"])
_report_slot = threading.BoundedSemaphore(1)


class DeletionPreview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_ids: list[Annotated[str, Field(pattern=r'^[a-f0-9]{24}$')]] = Field(min_length=1, max_length=1000)


class ReopenFailed(BaseModel):
    model_config = ConfigDict(extra='forbid')
    additional_calls: int = Field(default=0,ge=0,le=12,strict=True)
    ordinals: list[Annotated[int,Field(ge=1)]] | None = Field(default=None,max_length=10000)
    dry_run: bool = False


class Deletion(DeletionPreview):
    token: str = Field(pattern=r'^[a-f0-9]{64}$')
    return_to_pool: bool | None = Field(default=None, strict=True)


class DeletionOperation(DeletionPreview):
    action: Literal['preview','delete']
    token: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')
    return_to_pool: bool | None = Field(default=None,strict=True)


class DatasetExportRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    split: Literal['train', 'validation'] = 'train'
    export_scope: Literal['selected', 'all'] = 'selected'
    job_ids: list[Annotated[str, Field(pattern=r'^[a-f0-9]{24}$')]] | None = Field(default=None, max_length=1000)
    filename: str | None = Field(default=None, max_length=80, pattern=r'^[\w -]*$')


def export_disposition(split, filename):
    from urllib.parse import quote
    default_name = f'waymo-3dref-{split}.zip'
    name = filename.strip() + '.zip' if filename and filename.strip() else default_name
    return f'attachment; filename="{default_name}"; filename*=UTF-8\'\'{quote(name, safe="")}'


@router.get('/export/jobs/current')
def current_export(request: Request):
    return request.app.state.exports.current()


@router.post('/export/jobs', status_code=202)
def create_export(body: DatasetExportRequest, request: Request):
    mutation_origin(request)
    if body.export_scope == 'selected' and not body.job_ids:
        raise HTTPException(409, '请先选择要导出的任务。')
    if body.export_scope == 'all' and body.job_ids:
        raise HTTPException(409, '导出范围冲突，请刷新后重新选择。')
    try:
        return request.app.state.exports.create(body.job_ids if body.export_scope == 'selected' else None,
                                               body.split, body.filename)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/export/jobs/{export_id}')
def export_status(export_id: Id, request: Request):
    try:
        return request.app.state.exports.status(export_id)
    except KeyError as exc:
        raise HTTPException(404, '导出任务已失效（服务可能已重启），请重新导出。') from exc


@router.post('/export/jobs/{export_id}/cancel')
def cancel_export(export_id: Id, request: Request):
    mutation_origin(request)
    try:
        return request.app.state.exports.cancel(export_id)
    except KeyError as exc:
        raise HTTPException(404, '导出任务已失效，请刷新页面。') from exc


@router.get('/export/jobs/{export_id}/dataset.zip')
def download_export(export_id: Id, request: Request):
    exports = request.app.state.exports
    try:
        stream, job = exports.open_download(export_id)
    except KeyError as exc:
        raise HTTPException(404, '导出任务已失效，请重新导出。') from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return StreamingResponse(exports.download_chunks(export_id, stream), media_type='application/zip',
        headers={'Content-Disposition': export_disposition(job['split'], job['filename']),
                 'Content-Length': str(job['bytes_written']), 'Cache-Control': 'no-store'},
        background=BackgroundTask(exports.finish_download, export_id))


@router.post('/deletion/preview')
def deletion_preview(body: DeletionPreview, request: Request):
    mutation_origin(request)
    try:
        return request.app.state.batches.deletion_preview(list(dict.fromkeys(body.job_ids)))
    except (ValueError, OSError, RunBusy) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/deletion/operations',status_code=202)
def create_deletion_operation(body: DeletionOperation, request: Request):
    mutation_origin(request)
    if body.action == 'delete' and not body.token:
        raise HTTPException(409,'请先核对删除范围后确认。')
    try:
        return request.app.state.deletions.create(body.action,body.job_ids,body.token,body.return_to_pool)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/deletion/operations/{operation_id}')
def deletion_operation_status(operation_id: Id, request: Request):
    try:
        return request.app.state.deletions.status(operation_id)
    except KeyError as exc:
        raise HTTPException(404,'删除进度已失效，请刷新任务列表；已提交的删除不会撤销。') from exc


@router.post('/deletion')
def delete(body: Deletion, request: Request):
    mutation_origin(request)
    try:
        return request.app.state.batches.delete(list(dict.fromkeys(body.job_ids)), body.token, body.return_to_pool)
    except (ValueError, OSError, RunBusy) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/catalog")
def catalog(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=10000),
            status: str | None = None, category: str | None = None, context: str | None = None,
            compact: bool = False):
    try:
        value = load_catalog(compact=compact)
    except FileNotFoundError:
        return {"catalog_id": None, "summary": {}, "targets": [], "total": 0,
                "message": "先运行 uv run python -m scripts.build_catalog，扫描不调用模型。"}
    rows = [t for t in value["targets"] if (not status or t["status"] == status)
            and (not category or t["category"] == category) and (not context or t["context"] == context)]
    page = rows[offset:offset + limit]
    if compact:
        fields = {'target_id', 'ordinal', 'context', 'object_id', 'category', 'status', 'timestamp',
                  'frame_exclusion_counts', 'recommendation_rank', 'official_split'}
        page = [{k: v for k, v in t.items() if k in fields} for t in page]
    result = {"catalog_id": value["catalog_id"], "summary": value["summary"],
            "selection_policy": value["selection_policy"], "contexts": list(value["sources"]),
            "source_files": {c: s.get("lidar_box", {}).get("file") for c, s in value["sources"].items()},
            "targets": [{**t, "source_file": value["sources"].get(t["context"], {}).get("lidar_box", {}).get("file")}
                        for t in page],
            "total": len(rows), "offset": offset, "limit": limit}
    return JSONResponse(result, headers={'Cache-Control': 'no-store'}) if compact else result


@router.get("")
def index(request: Request):
    service = request.app.state.batches
    return {"active_batch_id": service.runner.active_batch,
            "jobs": service.store.summaries()}


@router.post("", status_code=201)
def create(body: BatchSpec, request: Request):
    mutation_origin(request)
    try:
        job, reused = request.app.state.batches.create(body)
        return {"job_id": job, "reused_job": reused, "started": False}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/export/dataset.zip')
def export_dataset(request: Request, split: Literal['train', 'validation'] = 'train',
                   job_ids: Annotated[list[Annotated[str, Field(pattern=r'^[a-f0-9]{24}$')]] | None, Query(max_length=1000)] = None,
                   filename: Annotated[str | None, Query(max_length=80, pattern=r'^[\w -]*$')] = None):
    from pathlib import Path
    from app.datasets.dataset_export import build_dataset, download_chunks
    service = request.app.state.batches
    root = getattr(request.app.state, 'sample_root', Path(service.runner.root).parents[1])
    try:
        stream = build_dataset(service, root, job_ids, intended_split=split)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, '数据集导出失败：' + str(exc)) from exc
    return StreamingResponse(download_chunks(stream), media_type='application/zip',
        headers={'Content-Disposition': export_disposition(split, filename)},
        background=BackgroundTask(stream.close))


@router.post('/export/dataset.zip')
def export_selected_dataset(request: Request,
        split: Annotated[Literal['train', 'validation'], Form()] = 'train',
        export_scope: Annotated[Literal['selected', 'all'], Form()] = 'selected',
        job_ids: Annotated[list[Annotated[str, Field(pattern=r'^[a-f0-9]{24}$')]] | None, Form(max_length=1000)] = None,
        filename: Annotated[str | None, Form(max_length=80, pattern=r'^[\w -]*$')] = None):
    mutation_origin(request)
    if export_scope == 'selected' and not job_ids:
        raise HTTPException(409, '请先选择要导出的任务。')
    if export_scope == 'all' and job_ids:
        raise HTTPException(409, '导出范围冲突，请刷新后重新选择。')
    return export_dataset(request, split, job_ids if export_scope == 'selected' else None, filename)


@router.get('/{job_id}/progress')
def progress(job_id: Id, request: Request):
    try:
        return JSONResponse(request.app.state.batches.store.progress(job_id), headers={'Cache-Control': 'no-store'})
    except ValueError as exc:
        raise HTTPException(404, 'Batch not found') from exc


@router.get("/{job_id}")
def detail(job_id: Id, request: Request, compact: bool = False):
    try:
        service = request.app.state.batches
        value = service.store.detail(job_id,compact=compact)
        if compact:
            target_fields = {'target_id', 'ordinal', 'context', 'object_id', 'category', 'status', 'timestamp',
                             'frame_exclusion_counts', 'recommendation_rank'}
            for item in value['items']:
                for key in ('target', 'display_target'):
                    if key in item:
                        item[key] = {k: v for k, v in item[key].items() if k in target_fields}
                item.pop('admission', None)
            value['spec'] = {k: v for k, v in value['spec'].items() if k != 'target_ids'}
            # The workspace uses the latest item event and per-item preparation
            # state. Full history stays available through the default endpoint.
            latest = {}
            last_item = None
            for event in value['events']:
                ordinal = event['payload'].get('ordinal')
                if isinstance(ordinal, int):
                    last_item = event
                    if event['kind'] in {'preparation_started', 'preparation_progress'}:
                        latest[event['kind'], ordinal] = event
            if last_item:
                latest['last_item'] = last_item
            value['events'] = sorted({e['seq']: e for e in latest.values()}.values(), key=lambda e: e['seq'])
        try:
            sources = service.display_catalog(value["spec"]["catalog_id"])["sources"]
        except FileNotFoundError:
            sources = {}
        result = {**value, 'resume': service.resume_requirements(value), "items": [{**item, "source_file":
            sources.get(item["target"].get("context"), {}).get("lidar_box", {}).get("file")}
            for item in value["items"]]}
        # These projections contain only JSON primitives. Encode inside this
        # synchronous endpoint's worker, avoiding recursive encoding of thousands
        # of records on the event loop. The default full-detail Python API stays.
        return JSONResponse(result, headers={'Cache-Control': 'no-store'}) if compact else result
    except ValueError as exc:
        raise HTTPException(404, "Batch not found") from exc


@router.post("/{job_id}/start", status_code=202)
def start(job_id: Id, request: Request):
    mutation_origin(request)
    try:
        return request.app.state.batches.start(job_id)
    except (ValueError, OSError, RunBusy) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{job_id}/control", status_code=202)
def control(job_id: Id, body: Control, request: Request):
    mutation_origin(request)
    try:
        return request.app.state.batches.control(job_id, body.command)
    except (ValueError, OSError, RunBusy) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/{job_id}/continue',status_code=200)
def continue_pending(job_id: Id, request: Request):
    """Prepare recovery inside the same task; /start or control resume executes it."""
    mutation_origin(request)
    try:
        return request.app.state.batches.prepare_resume(job_id)
    except (ValueError, OSError, RunBusy) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/{job_id}/reopen-failed')
def reopen_failed_candidates(job_id: Id, body: ReopenFailed, request: Request):
    mutation_origin(request)
    service=request.app.state.batches
    try:
        service.ensure_loaded_code()
        from app.infrastructure.candidate_pool import reopen_failed
        return reopen_failed(service.runner.store,job_id,additional_calls=body.additional_calls,
            ordinals=body.ordinals,dry_run=body.dry_run)
    except (ValueError,OSError,RunBusy) as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get("/{job_id}/report")
def report(job_id: Id, request: Request, download: bool = False):
    if not _report_slot.acquire(blocking=False):
        raise HTTPException(409,'正在读取任务统计，请等待本次统计结束后再试。')
    try:
        return build_report(job_id,request,download)
    finally:
        _report_slot.release()


def build_report(job_id, request, download=False):
    from app.quality.evaluation import summarize
    service = request.app.state.batches
    try:
        value = service.store.detail(job_id, compact=True)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    # Display polling omits these records; reports still need the frozen
    # dataset denominator and all execution revisions without old prep payloads.
    with service.runner.store.connect() as db:
        report_events = [{**dict(row),'payload':json.loads(row['payload'])} for row in db.execute(
            "SELECT * FROM batch_events WHERE job_id=? AND kind IN ('batch_created','execution_revision') ORDER BY seq",(job_id,))]
    value['events'] = sorted({event['seq']:event for event in [*value['events'],*report_events]}.values(),key=lambda event:event['seq'])
    linked = {i['run_id'] for i in value['items'] if i['run_id']}
    owned = {i['run_id'] for i in value['items'] if i['run_id'] and not i['reused']}
    snapshots = service.runner.store.report_runs(job_id)
    all_states = [s for s in snapshots if s['run_id'] in linked]
    # Recovery can replace an item's run link. Its earlier calls still count.
    states = [s for s in snapshots if s['run_id'] in owned or s.get('batch_id') == job_id]
    from app.agent.console import call_details
    actual = []
    for state in states:
        actual.extend({key:call[key] for key in ('status','usage','elapsed_seconds','remote_outcome') if key in call}
            for call in call_details(service.runner.root,state,include_requests=False))
    from app.quality.research import ResearchStore
    audits = ResearchStore(service.runner.store).job_metrics(job_id)
    from app.quality.batch_metrics import task_metrics
    core = task_metrics(value, all_states, states, actual, audits)
    result = {"job_id": job_id, "catalog_id": value["spec"]["catalog_id"], "core_metrics":core,
            'method_sha256': value['method_sha256'], 'configuration': value['spec'],
            'execution_revisions': [event['payload'] for event in value['events'] if event['kind']=='execution_revision'],
            'status': value['status'], 'reason': value['reason'],
            "track_denominator": len(value["items"]), "processing_counts": value["counts"],
            "model_calls_reserved": value["model_calls_reserved"],
            "evaluation": summarize(states),
            "training_eligible": False, "scope": "Task-level runtime diagnostics; independent quality is reported only by sealed blind audits."}
    if download:
        return Response(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename="batch-{job_id}-metrics.json"'})
    return result
