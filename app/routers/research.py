"""Explicit independent audit and release actions; read paths never call models."""
import json
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response
from pydantic import Field

from app.infrastructure.artifacts import checked_asset
from app.schemas.contracts import Contract, EvidencePacket
from app.datasets.evidence import ROOT
from app.datasets.inspection import corners, sample_indices
from app.quality.research import AuditAnswer, CohortSpec, ResearchStore, PROTOCOL
from app.routers.dependencies import Id, mutation_origin
from app.datasets.training import read_scene

router = APIRouter(prefix='/api/research',tags=['independent research audit'])


def service(request):
    return request.app.state.research


@router.get('')
def index(request: Request):
    store = service(request)
    with store.runs.connect() as db:
        releases = [json.loads(r['manifest']) for r in db.execute('SELECT manifest FROM label_releases ORDER BY created_at DESC')]
    return {'cohorts':store.list(),'releases':releases}


@router.post('/cohorts',status_code=201)
def create(body: CohortSpec,request: Request):
    mutation_origin(request)
    try:
        return service(request).create(body,request.app.state.batches.store.detail(body.job_id))
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/cohorts/{cohort_id}')
def cohort(cohort_id: Id,request: Request):
    try:
        return service(request).detail(cohort_id)
    except ValueError as exc:
        raise HTTPException(404,str(exc)) from exc


def assets(store,cohort_id,case_id):
    case,_ = store.case(cohort_id,case_id)
    packet = EvidencePacket.model_validate(case['packet'])
    scene_path = checked_asset(store.root,packet.scene.manifest_path,packet.scene.manifest_sha256)
    scene,arrays = read_scene(scene_path.parent)
    if (scene.context,scene.frame_timestamp_micros) != (packet.context,packet.frame_timestamp_micros):
        raise ValueError('Frozen case and scene identity mismatch')
    return case,packet,scene,arrays,scene_path


@router.get('/cohorts/{cohort_id}/cases/{case_id}')
def blind_case(cohort_id: Id,case_id: Id,request: Request):
    try:
        store = service(request)
        case,packet,scene,arrays,_ = assets(store,cohort_id,case_id)
        by_id = {o.object_id:o for o in packet.candidates}
        frozen = store.private(cohort_id)['frozen']
        points = arrays['points'][:,:3]
        selected = sample_indices(np.arange(len(points)),22000)
        return {'case_id':case_id,'expression':case['expression'],'localization':store.localization(cohort_id,case_id), 'protocol':frozen['protocol'],
            'coordinate_frame':scene.coordinate_frame, 'input_capabilities':frozen.get('input_capabilities', ['geometry']),
            'preview':{'sampling':'target-independent deterministic whole-scene preview',
                'points':points[selected].tolist(),'target_points':[],'target_corners':[],
                'candidate_boxes':[{'object_id':label,'category':by_id[oid].category,'corners':corners(by_id[oid])} for label,oid in case['mapping'].items()]},
            'candidates':[{'label':label,'category':by_id[oid].category,'center_m':by_id[oid].center_m,'size_m':by_id[oid].size_m} for label,oid in case['mapping'].items()],
            'saved_answer':store.private(cohort_id)['answers'].get(case_id),
            'cameras':[{'camera_name':c.camera_name,'width':c.width,'height':c.height,
                        'image':f'/api/research/cohorts/{cohort_id}/cases/{case_id}/images/{c.camera_name}'} for c in scene.cameras]}
    except (ValueError,OSError) as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/cohorts/{cohort_id}/cases/{case_id}/images/{camera}')
def image(cohort_id: Id,case_id: Id,camera: int,request: Request):
    try:
        store = service(request)
        _, cohort = store.case(cohort_id,case_id)
        if cohort['frozen']['protocol'] != PROTOCOL and store.localization(cohort_id,case_id) is None:
            raise ValueError('Commit the geometry-only localization before revealing auxiliary RGB')
        _,_,scene,_,path = assets(service(request),cohort_id,case_id)
        c = next((c for c in scene.cameras if c.camera_name == camera),None)
        if c is None:
            raise ValueError('Unknown camera')
        return FileResponse(checked_asset(path.parent,c.image.path,c.image.sha256))
    except (ValueError,OSError) as exc:
        raise HTTPException(409,str(exc)) from exc


class Localization(Contract):
    selected_labels: list[str] = Field(max_length=500)


@router.post('/cohorts/{cohort_id}/cases/{case_id}/localization')
def localization(cohort_id: Id,case_id: Id,body: Localization,request: Request):
    mutation_origin(request)
    try:
        return service(request).lock_localization(cohort_id,case_id,body.selected_labels)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/cohorts/{cohort_id}/cases/{case_id}/answer')
def answer(cohort_id: Id,case_id: Id,body: AuditAnswer,request: Request):
    mutation_origin(request)
    try:
        return service(request).submit(cohort_id,case_id,body)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/cohorts/{cohort_id}/seal')
def seal(cohort_id: Id,request: Request):
    mutation_origin(request)
    try:
        return service(request).seal(cohort_id)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/cohorts/{cohort_id}/report.json')
def report(cohort_id: Id,request: Request):
    value = cohort(cohort_id,request)
    if value['status'] != 'sealed':
        raise HTTPException(409,'Seal the audit before exporting scored results')
    return Response(json.dumps(value,ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="audit-{cohort_id}.json"'})


class ReleaseSpec(Contract):
    calibration_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    evaluation_id: str = Field(pattern=r'^[a-f0-9]{24}$')


@router.get('/comparison')
def comparison(request: Request, first_id: str = Query(pattern=r'^[a-f0-9]{24}$'), final_id: str = Query(pattern=r'^[a-f0-9]{24}$')):
    try:
        return service(request).compare_repair(first_id,final_id)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/releases',status_code=201)
def release(body: ReleaseSpec,request: Request):
    mutation_origin(request)
    try:
        return service(request).release(body.calibration_id,body.evaluation_id)
    except (ValueError,OSError) as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/releases/{release_id}/training.zip')
def package(release_id: Id,request: Request):
    try:
        return FileResponse(service(request).package(release_id),media_type='application/zip',filename=f'waymo-3dvg-{release_id}.zip')
    except (ValueError,OSError) as exc:
        raise HTTPException(409,str(exc)) from exc
