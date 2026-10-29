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


from app.quality.paper_study import StudySpec,Judgment


@router.get('/studies')
def studies(request: Request):
    return {'studies':request.app.state.paper_studies.list(request.app.state.batches)}


@router.post('/studies',status_code=201)
def create_study(body:StudySpec,request:Request):
    mutation_origin(request)
    try:return request.app.state.paper_studies.create(body,request.app.state.batches)
    except (ValueError,OSError) as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/studies/{study_id}')
def study_detail(study_id:Id,request:Request):
    try:return request.app.state.paper_studies.detail(study_id,request.app.state.batches)
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc


class StudyControl(Contract):
    action:Literal['start','pause','freeze','seal']


@router.post('/studies/{study_id}/control')
def study_control(study_id:Id,body:StudyControl,request:Request):
    mutation_origin(request);store=request.app.state.paper_studies;batch=request.app.state.batches
    try:
        s=store.private(study_id)
        if body.action=='freeze':return store.freeze(study_id,batch)
        if body.action=='seal':return store.seal(study_id,batch)
        if s['status'] in {'review','sealed'}:raise ValueError('已进入盲审的实验不能重新执行。')
        if body.action=='start':
            store.ensure_method(s['job_id']);batch.control(s['job_id'],'resume')
        else:batch.control(s['job_id'],'pause')
        return store.detail(study_id,batch)
    except (ValueError,OSError) as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/studies/{study_id}/report.{format}')
def study_export(study_id:Id,format:Literal['json','csv'],request:Request):
    store=request.app.state.paper_studies
    try:
        if format=='csv':data=store.export_csv(study_id)
        else:
            s=store.private(study_id)
            if s['status']!='sealed':raise ValueError('报告尚未封存。')
            data=json.dumps(s['report'],ensure_ascii=False,indent=2)
        return Response(data,media_type='text/csv' if format=='csv' else 'application/json',
            headers={'Content-Disposition':f'attachment; filename="paired-study-{study_id}.{format}"'})
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/review/{token}')
def study_inbox(token:str,request:Request):
    try:return request.app.state.paper_studies.inbox(token)
    except ValueError as exc:raise HTTPException(404,str(exc)) from exc


def paper_assets(token,cid,request):
    store=request.app.state.paper_studies
    case,review,revealed=store.case(token,cid)
    packet=EvidencePacket.model_validate(case['payload']['packet'])
    path=checked_asset(store.root,packet.scene.manifest_path,packet.scene.manifest_sha256)
    scene,arrays=read_scene(path.parent)
    if (scene.context,scene.frame_timestamp_micros)!=(packet.context,packet.frame_timestamp_micros):raise ValueError('冻结场景身份不一致。')
    camera=next(v.camera_name for v in packet.views if v.camera_label==packet.source_view)
    return case,review,revealed,packet,path,scene,arrays,camera


@router.get('/review/{token}/cases/{case_id}')
def study_case(token:str,case_id:Id,request:Request):
    try:
        case,review,revealed,p,path,scene,arrays,camera=paper_assets(token,case_id,request)
        from app.datasets.inspection import camera_views
        mapping=case['payload']['mapping'];labels={v:k for k,v in mapping.items()};by_id={o.object_id:o for o in p.candidates}
        selected=sample_indices(np.arange(len(arrays['points'])),22000)
        cam=next(c for c in scene.cameras if c.camera_name==camera)
        boxes=[{'box_id':f'P{i}','candidate_id':labels.get(b['object_id']),'category':b['category'],'xyxy':b['xyxy']}
            for i,b in enumerate(p.provenance.get('scene_image_candidates',[])) if b.get('camera_name')==camera]
        return {'case_id':case_id,'expression':case['payload']['expression'],'selection':review.get('selection'),
            'answer':review.get('answer'),'revealed':revealed,'expected':case['payload']['expected'] if revealed else None,
            'preview':{'points':arrays['points'][selected,:3].tolist(),'target_points':[],'target_corners':[],
                'candidate_boxes':[{'object_id':label,'category':by_id[oid].category,'corners':corners(by_id[oid])} for label,oid in mapping.items()],
                'sampling':'目标无关的全场景预览；完整单帧点云可下载'},
            'candidates':[{'candidate_id':label,'category':by_id[oid].category} for label,oid in mapping.items()],
            'cameras':[{'camera_name':camera,'width':cam.width,'height':cam.height,'view_axes':camera_views(path.parent,scene).get(camera),
                'native_boxes':boxes,'projected_boxes':boxes,'image':f'/api/research/review/{token}/cases/{case_id}/image'}],
            'full_cloud':f'/api/research/review/{token}/cases/{case_id}/cloud.npz','point_count':len(arrays['points'])}
    except (ValueError,OSError,StopIteration) as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/review/{token}/cases/{case_id}/image')
def study_image(token:str,case_id:Id,request:Request):
    try:
        _,_,_,_,path,scene,_,camera=paper_assets(token,case_id,request)
        c=next(c for c in scene.cameras if c.camera_name==camera)
        return FileResponse(checked_asset(path.parent,c.image.path,c.image.sha256))
    except (ValueError,OSError,StopIteration) as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/review/{token}/cases/{case_id}/cloud.npz')
def study_cloud(token:str,case_id:Id,request:Request):
    import io
    try:
        _,_,_,_,_,_,arrays,_=paper_assets(token,case_id,request)
        output=io.BytesIO();np.savez_compressed(output,points=arrays['points'])
        return Response(output.getvalue(),media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="single-frame-points.npz"'})
    except (ValueError,OSError) as exc:raise HTTPException(409,str(exc)) from exc


class PaperSelection(Contract):
    selected_labels:list[str]=Field(max_length=500)
    elapsed_seconds:float=Field(default=0,ge=0,le=86400,allow_inf_nan=False)


@router.post('/review/{token}/cases/{case_id}/localization')
def study_select(token:str,case_id:Id,body:PaperSelection,request:Request):
    mutation_origin(request)
    try:return request.app.state.paper_studies.select(token,case_id,body.selected_labels,body.elapsed_seconds)
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc


@router.post('/review/{token}/cases/{case_id}/answer')
def study_answer(token:str,case_id:Id,body:Judgment,request:Request):
    mutation_origin(request)
    try:return request.app.state.paper_studies.answer(token,case_id,body)
    except ValueError as exc:raise HTTPException(409,str(exc)) from exc


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
