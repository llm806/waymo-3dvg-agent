"""Core backend regressions; disposable inputs and local model substitutes only."""
import json
import math
import shutil
import time
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.schemas.contracts import (AgentAction, EvidencePacket, GeneratedExpression, GroundResult,
    ImageEvidence, ObjectFact, SceneEvidence, VisualAttribute, VisualObservation)
from app.schemas.training_contracts import TaskProfile
from app.datasets.evidence import TASK
from app.datasets.training import training_inputs, read_scene
from app.agent.harness import run_packet
from app.agent.model_calls import ModelCallFailure
from app.agent.appearance import target_visibility_error
from app.agent.batch import BatchService, BatchSpec, BatchStore
from app.agent.console import ConsoleRunner
from app.infrastructure.artifacts import atomic_json, file_sha256
from app.infrastructure.storage import RunStore, RunBusy
from app.infrastructure.task_cleanup import TaskCleaner
from app.quality.admission import admission_plan
from app.quality.contrast import POLICY, contrast_plan, predicate_catalog, remaining, ego_direction_match
from app.quality.research import ResearchStore
from app.quality.visibility_policy import RECOGNIZABLE_POLICY



def obj(name, x, y, category="vehicle", points=10, length=4):
    return ObjectFact(object_id=name, category=category, center_m=(x, y, 1), size_m=(length, 2, 2),
                      heading_rad=0, range_xy_m=math.hypot(x, y), lidar_points=points,
                      speed_label_mps=(None, None, None), geometry_valid=True)


def packet():
    candidates = [obj("target-unique-native-id", 10, 5), obj("negative-unique-native-id", 10, -5),
                  obj("sign-unique-native-id", 12, -3, "sign")]
    catalog = predicate_catalog(candidates)
    return EvidencePacket(packet_id="a" * 24, context="example", frame_timestamp_micros=10,
                          target_object_id=candidates[0].object_id,
                          task={k: v for k, v in TASK.items() if k not in {'scene_camera_scope_policy','vehicle_detail_policy','ego_wording_policy','target_naming_policy','landmark_locality_policy','sentence_style_policy','attribute_repair_policy','scene_geometry_authority_policy','feedback_acquisition_policy', 'landmark_salience_policy', 'feedback_acceptance_policy', 'feedback_revision_policy', 'scene_input_policy', 'scene_binding_policy', 'manual_feedback_policy', 'scene_relation_reading_policy', 'spatial_relation_policy', 'vehicle_forward_region_policy', 'joint_reference_policy', 'claim_normalization_policy', 'vehicle_heading_policy', 'scene_grounding_policy', 'landmark_description_policy', 'reference_preference_policy', 'image_position_policy', 'reference_policy', 'observation_policy', 'description_detail_policy', 'execution_policy', 'requires_native_camera_link', 'natural_description', 'contrastive_appearance', 'image_first', 'admission_policy', 'annotation_search', 'image_reference_frame', 'image_identity_policy', 'visibility_policy', 'view_selection_policy'}}, policy=POLICY,
                          target=candidates[0], candidates=candidates, predicates=catalog,
                          contrast=contrast_plan(candidates, candidates[0].object_id, catalog),
                          views=[], track={}, provenance={}, ready_for_generation=True)


class LocalCalls:
    model = "local-contract-test"
    base_url = "local://no-network"
    scope = "local-contract-test"

    def __init__(self, first_bad=False, crash_decision=False):
        self.count = 0
        self.generated = 0
        self.first_bad = first_bad
        self.crash_decision = crash_decision

    def call(self, messages, schema, folder, heartbeat=None):
        self.count += 1
        if heartbeat is not None:
            heartbeat()
        if schema is AgentAction:
            payload = json.loads(messages[-1]["content"])
            action = payload["allowed_tools"][0]["action"]
            result = AgentAction(action=action, reason="Exercise the offered action")
            if self.crash_decision:
                self.crash_decision = False
                # Emulate a process crash after output persistence but before phase commit.
                atomic_json(folder / "parsed-output.json", result.model_dump())
                raise KeyboardInterrupt()
            return result
        if schema is GeneratedExpression:
            self.generated += 1
            return GeneratedExpression(expression="The vehicle." if self.first_bad and self.generated == 1 else "The vehicle to the left of the ego vehicle.",
                                       predicate_ids=["category:vehicle", "lateral:left"])
        payload = json.loads(messages[-1]["content"])
        left = "left of" in payload["expression"]
        right = "right of" in payload["expression"]
        matching = [c["candidate_id"] for c in payload.get("candidates", [])
                    if c["category"] == "vehicle" and (not left or c["center_m"][1] > 1) and (not right or c["center_m"][1] < -1)]
        return GroundResult(candidate_ids=matching,
                            parsed_predicate_ids=["category:vehicle", *(["lateral:left"] if left else []), *(["lateral:right"] if right else [])],
                            unsupported_claims=[])


def rich_packet(tmp_path, monkeypatch):
    import app.agent.harness as harness
    evidence = packet()
    evidence.task = {**evidence.task, 'execution_policy': 'evidence-fast-v1'}
    folder = tmp_path / evidence.packet_id
    folder.mkdir(parents=True)
    for name in ['overlay.jpg', 'crop.jpg', 'appearance.jpg']:
        Image.new('RGB', (100, 100), 'white').save(folder / name)
    evidence.views = [ImageEvidence(camera_name=1, camera_label='FRONT', projected_box_xyxy=(0, 0, 100, 100),
        crop_xyxy=(0, 0, 100, 100), source_jpeg_sha256='a' * 64, width=100, height=100,
        pose_timestamp_s=None, camera_trigger_time_s=None, camera_readout_done_time_s=None,
        most_visible_camera=True, appearance_identity_source='official_camera_lidar_association',
        appearance_box_xyxy=(0, 0, 100, 100), appearance_camera_object_id='native-camera-id',
        artifacts={'overlay': 'overlay.jpg', 'crop': 'crop.jpg', 'appearance': 'appearance.jpg'})]
    evidence.provenance = {'artifact_sha256': {p.name: file_sha256(p) for p in folder.iterdir()}}
    monkeypatch.setattr(harness, 'EVIDENCE_ROOT', tmp_path)
    return evidence


class RichCalls(LocalCalls):
    def __init__(self, confirmed=True):
        super().__init__()
        self.confirmed = confirmed
        self.kinds = []

    def call(self, messages, schema, folder, heartbeat=None):
        self.count += 1
        self.kinds.append(schema)
        attributes = [VisualAttribute(kind='color', value='white'), VisualAttribute(kind='subtype', value='sedan')]
        if schema is GeneratedExpression:
            return GeneratedExpression(expression='The white sedan ahead and to the left of the ego vehicle.',
                predicate_ids=['category:vehicle', 'longitudinal:front', 'lateral:left'], visual_attributes=attributes)
        if schema is VisualObservation:
            return VisualObservation(target_visible=True, attributes=attributes if self.confirmed else [])
        assert schema is GroundResult  # A single useful action never needs a decision API call.
        sentence = json.loads(messages[-1]['content'])['expression']
        attributes = [a for a in attributes if a.value in sentence]
        return GroundResult(candidate_ids=[], parsed_predicate_ids=['category:vehicle', 'longitudinal:front', 'lateral:left'],
            unsupported_claims=[], parsed_visual_attributes=attributes)


def prepared(tmp_path, monkeypatch, ambiguous=False):
    p = rich_packet(tmp_path, monkeypatch)
    p.task.update(image_first=True, requires_native_camera_link=True, natural_description=True,
        contrastive_appearance=True, admission_policy='visible-distinct-v1')
    p.source_view, p.source_camera_object_id = 'FRONT', 'native-camera-id'
    p.scene = SceneEvidence(scene_id='b'*24, manifest_path='scene.json', manifest_sha256='b'*64,
        total_points=100, target_points=10)
    if ambiguous:
        p.candidates[1] = p.candidates[1].model_copy(update={'center_m':p.target.center_m})
        p.predicates = predicate_catalog(p.candidates)
    p.provenance['source_image_candidates'] = [{'object_id':p.target_object_id, 'category':'vehicle',
        'camera_object_id':p.source_camera_object_id}]
    p.provenance['scene_image_candidates'] = [dict(c, camera_name=1)
        for c in p.provenance['source_image_candidates']]
    return p


def visible(color='white'):
    return VisualObservation(target_visible=True, attributes=[VisualAttribute(kind='color',value=color)])


def strict_packet(tmp_path, monkeypatch, ambiguous=False):
    p=prepared(tmp_path,monkeypatch,ambiguous)
    p.task['visibility_policy']='clear-source-rgb-v1'
    view=p.views[0]
    view.projection_touches_boundary=False
    view.artifacts.update(scene='overlay.jpg',appearance_context='crop.jpg')
    return p


def clear():
    return VisualObservation(target_visible=True,visibility='clear',
        visibility_reason='The whole vehicle body is visible in the RGB context without an occluder.',attributes=visible().attributes)


def partial_observation():
    return VisualObservation(target_visible=True, visibility='partial',
        visibility_reason='White pickup cab is recognizable; the wall hides its rear.',
        attributes=[VisualAttribute(kind='color', value='white'), VisualAttribute(kind='subtype', value='pickup truck')],
        vehicle_colors={'body': 'white', 'roof': None})


class InvalidFormatCalls(LocalCalls):
    def __init__(self, failures=1, crash_after_retry=False):
        super().__init__()
        self.failures = failures
        self.crash_after_retry = crash_after_retry

    def call(self, messages, schema, folder, heartbeat=None):
        if schema is AgentAction and self.failures:
            self.count += 1
            self.failures -= 1
            atomic_json(folder / "result.json", {"status": "invalid_output", "validation_errors": [{"location": ["reason"], "type": "string_too_long", "message": "At most 300 characters"}]})
            raise ModelCallFailure("invalid_output")
        if schema is AgentAction and self.crash_after_retry:
            self.crash_after_retry = False
            self.crash_decision = True
        return super().call(messages, schema, folder, heartbeat)


def wait(worker):
    # This barrier runs after the coordinator on its single-worker executor.
    # File I/O load must not turn a successful task into a five-second timeout.
    worker.pool.submit(lambda: None).result(timeout=60)
    assert not worker.active_batch


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_MODEL", LocalCalls.model)
    monkeypatch.setenv("DASHSCOPE_BASE_URL", LocalCalls.base_url)
    catalog = {"catalog_id": "a" * 24, "sources": {}, "targets": []}
    prepared = []
    packets = {}
    for i in range(4):
        p = packet().model_copy(update={"packet_id": str(i) * 24, "context": "context-" + str(i)})
        packets[i + 1] = p
        catalog["targets"].append({"target_id": str(i + 1) * 24, "ordinal": i + 1, "context": p.context,
                                   "object_id": p.target_object_id, "timestamp": p.frame_timestamp_micros,
                                   "status": "selected", "frame_exclusion_counts": {}})
    catalog["targets"].append({"target_id": "f" * 24, "ordinal": 5, "status": "excluded", "frame_exclusion_counts": {"no_target_lidar_points": 2}})
    catalog['summary'] = {'tracks': 5, 'selected': 4, 'contexts': 4}
    instances = []

    class Calls(LocalCalls):
        def __init__(self):
            super().__init__(); instances.append(self)
        def close(self):
            pass

    worker = ConsoleRunner(tmp_path, Calls)

    def prepare(target, catalog, heartbeat, cancelled, log):
        prepared.append(target["ordinal"])
        return packets[target["ordinal"]]

    service = BatchService(worker, prepare, lambda _: catalog, lambda _: None)
    monkeypatch.setattr("app.agent.batch.load_packet", lambda pid: next(p for p in packets.values() if p.packet_id == pid))
    yield service, worker, catalog, prepared, instances
    service.close(); worker.close()


def pool(catalog):
    for target, camera, area in zip(catalog['targets'], ['FRONT', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT'], [100, 80, 90, 70]):
        target.update(source_view=camera, selection={'image_selection_score': area, 'range_xy_m': 10})
    return [t['target_id'] for t in catalog['targets'] if t['status'] == 'selected']


def spec(catalog, goal=1, budget=12):
    return BatchSpec(catalog_id=catalog['catalog_id'], target_ids=pool(catalog),
        prepare_only=False, desired_count=goal, max_model_calls=budget)


def outcomes(monkeypatch, worker, fail_ordinals=(), pause_after=None,
             fail_reason='precheck_target_occluded_or_uncertain', failure_calls=1):
    def run(packet, calls, repairs, root, budget, limit, *, resume_run_id, **kwargs):
        state = worker.store.snapshot(resume_run_id)
        ordinal = int(packet.packet_id[0]) + 1
        passed = ordinal not in fail_ordinals
        used = 4 if passed else failure_calls
        assert used <= budget
        calls.count += used
        owner = kwargs['executor_owner']
        worker.store.claim(resume_run_id, owner, state)
        state.update(status='calibration_hold' if passed else 'needs_evidence',
            diagnostic_pass=passed, model_calls_used=used, inflight=None,
            final_expression='The vehicle ahead.' if passed else 'An unverified failed draft.',
            stop_reason='oracle_passed' if passed else fail_reason)
        worker.store.save(resume_run_id, owner, state, 'model_call_reserved', {'local_test': True})
        worker.store.release(resume_run_id, owner)
        with worker.store.connect() as db:
            if ordinal == pause_after:
                db.execute("UPDATE batches SET control='pause' WHERE job_id=?", (state['batch_id'],))
        return state
    monkeypatch.setattr('app.agent.batch.run_agent', run)


def ident(n, size=24):
    return format(n, 'x').zfill(size)


def exercise_manual_reviews(root, monkeypatch, scene_folder):
    from fastapi import FastAPI
    from fastapi import HTTPException
    from app.routers import samples as routes
    from app.agent.agent_runtime import run_agent
    from app.datasets.dataset_export import rejection
    packets = {i:packet().model_copy(update={'packet_id':ident(9200+i),
        'context':f'review-context-{i}'}) for i in range(1,5)}
    catalog = {'catalog_id':ident(9200), 'sources':{}, 'targets':[
        {'target_id':ident(9300+i),'ordinal':i,'context':p.context,'object_id':p.target_object_id,
         'timestamp':p.frame_timestamp_micros,'category':'vehicle','status':'selected','frame_exclusion_counts':{}}
        for i,p in packets.items()]}

    from threading import Event
    retry_gate = {}

    class ReviewCalls(LocalCalls):
        def close(self):
            pass
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is GeneratedExpression:
                payload = json.loads(messages[-1]['content'][0]['text'])
                if payload['task'].get('manual_review_feedback'):
                    self.count += 1
                    assert payload['task']['manual_review_feedback']['reason'] == '请使用更清晰的左右关系。'
                    if retry_gate and runner.store.snapshot(payload['task']['manual_review_feedback']['source_run_id'])['current_packet']['context']==retry_gate['context']:
                        retry_gate['entered'].set()
                        assert retry_gate['release'].wait(20)
                    assert 'not verified visual evidence' in messages[0]['content']
                    text = ('The vehicle located to the left of the ego vehicle.'
                        if payload['task']['manual_review_feedback']['previous_expression'] == 'The vehicle on the left of the ego vehicle.'
                        else 'The vehicle on the left of the ego vehicle.')
                    return GeneratedExpression(expression=text,
                        predicate_ids=['category:vehicle','lateral:left'])
            return super().call(messages,schema,folder,heartbeat)

    runner = ConsoleRunner(root,ReviewCalls)
    service = BatchService(runner,lambda t,*args:packets[t['ordinal']],lambda _:catalog,lambda _:None)
    app = FastAPI();app.include_router(routes.router)
    app.state.batches=service;app.state.sample_root=scene_folder.parent
    with monkeypatch.context() as patch:
        patch.setattr('app.agent.batch.run_agent',run_agent)
        patch.setattr('app.agent.batch.load_packet',lambda pid:next(p for p in packets.values() if p.packet_id==pid))
        patch.setattr(routes,'manifest',lambda *args:(scene_folder/'scene.json',None))
        patch.setattr(routes,'native_labels',lambda p,*args:({o.object_id:o for o in p.candidates},{}))
        try:
            job,_=service.create(BatchSpec(catalog_id=catalog['catalog_id'],prepare_only=False,
                target_ids=[t['target_id'] for t in catalog['targets']],desired_count=5))
            service.start(job);wait(runner)
            request=SimpleNamespace(app=app,headers={},url=SimpleNamespace(hostname='127.0.0.1'))
            # The index must not invoke detail or open each run's history.
            with monkeypatch.context() as index_patch:
                def no_individual_read(*args, **kwargs):
                    raise AssertionError('Sample index must use a batched read')
                index_patch.setattr(service.store,'detail',no_individual_read)
                index_patch.setattr(runner.store,'snapshot',no_individual_read)
                index_patch.setattr(runner.store,'events',no_individual_read)
                listed=routes.samples(job,request)
                viewed=routes.scene(job,1,request,review=True)
                index_patch.setattr(routes,'sample_revision',no_individual_read)
                original_connect = runner.store.connect
                @contextmanager
                def index_only():
                    with original_connect() as db:
                        db.set_authorizer(lambda action, table, column, *_: sqlite3.SQLITE_DENY
                            if action == sqlite3.SQLITE_READ and ((table == 'runs' and column == 'snapshot') or table == 'events')
                            else sqlite3.SQLITE_OK)
                        yield db
                index_patch.setattr(runner.store,'connect',index_only)
                assert routes.samples(job,request)==listed
                # The compact index survives a fresh store instance/process;
                # neither request can select the large snapshot column.
                fresh = BatchStore.__new__(BatchStore);fresh.runs=runner.store
                other_request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(batches=SimpleNamespace(store=fresh))))
                assert routes.samples(job,other_request)==listed
                # An executor-only update must not reload full scene snapshots.
                from app.agent import agent_runtime
                index_patch.setattr(agent_runtime,'VERSION','next-executor-for-cache-test')
                changed=routes.samples(job,other_request)
                assert len(changed['samples'])==len(listed['samples'])
                assert all(row['legacy_rules'] for row in changed['samples'])
                assert all('_run_version' not in row for row in changed['samples'])
            assert listed['review_summary']=={'unreviewed':4,'accepted':0,'rejected':0,'discarded':0,'usable':4}
            drawn=routes.random_samples(job,request,count=10)['ordinals']
            assert len(drawn)==len(set(drawn))==4
            assert viewed['ordinal']==1 and viewed['selected_candidate_id'] is None and viewed['source_view'] is None
            with pytest.raises(HTTPException):routes.scene(job,999,request)
            first=next(s for s in listed['samples'] if s['ordinal']==1)
            assert first['revision']==viewed['revision']
            # Even an externally changed snapshot with the same SQL revision
            # must invalidate the presentation cache; authoritative checks stay.
            with runner.store.connect() as db:
                original=db.execute('SELECT snapshot FROM runs WHERE run_id=?',(first['run_id'],)).fetchone()['snapshot']
                modified=json.loads(original);modified['diagnostic_pass']=False
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(modified),first['run_id']))
            assert routes.samples(job,request)['review_summary']['usable']==3
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(original,first['run_id']))
            assert routes.samples(job,request)['review_summary']['usable']==4
            # A source-chain edit must invalidate even without a run revision.
            with runner.store.connect() as db:
                db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(first['run_id'],999999,'source_view_reselected',
                    json.dumps({'from_packet_id':'wrong','from_timestamp':-1,'packet_id':'wrong','timestamp':-2}),time.time()))
            assert routes.samples(job,request)['review_summary']['usable']==3
            with runner.store.connect() as db:
                db.execute('DELETE FROM events WHERE run_id=? AND seq=999999',(first['run_id'],))
            assert routes.samples(job,request)['review_summary']['usable']==4
            # Preparation receipts affect the same direct/detail/index binding
            # checks; changing them must invalidate a warm presentation too.
            with runner.store.connect() as db:
                db.execute("INSERT INTO batch_events(job_id,kind,payload,at) VALUES(?,'preparation_source_reselected',?,?)",
                    (job,json.dumps({'ordinal':1,'packet_id':first['packet_id'],'timestamp':999,
                        'rejected_frames':[{'timestamp':1,'reason':'converted_target_points_missing'}]}),time.time()))
            assert routes.samples(job,request)['review_summary']['usable']==3
            with pytest.raises(ValueError):routes.sample(service.store,job,1)
            with runner.store.connect() as db:
                db.execute("DELETE FROM batch_events WHERE job_id=? AND kind='preparation_source_reselected'",(job,))
            assert routes.samples(job,request)['review_summary']['usable']==4
            with runner.store.connect() as db:
                db.execute('UPDATE batch_items SET packet_id=? WHERE job_id=? AND ordinal=1',('f'*24,job))
            assert routes.samples(job,request)['review_summary']['usable']==3
            with runner.store.connect() as db:
                db.execute('UPDATE batch_items SET packet_id=? WHERE job_id=? AND ordinal=1',(first['packet_id'],job))
            assert routes.samples(job,request)['review_summary']['usable']==4
            # A writer racing a cold rebuild cannot publish a stale cache row.
            with runner.store.connect() as db:
                db.execute('UPDATE sample_presentations SET format=NULL,presentation=NULL WHERE job_id=? AND ordinal=1',(job,))
            render=routes.presentation_row
            with monkeypatch.context() as race_patch:
                def racing_render(job_id,item,state,changes):
                    if item['ordinal']==1:
                        with runner.store.connect() as db:
                            changed={**state,'diagnostic_pass':False}
                            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(changed),item['run_id']))
                    return render(job_id,item,state,changes)
                race_patch.setattr(routes,'presentation_row',racing_render)
                assert routes.samples(job,request)['review_summary']['usable']==4
            with runner.store.connect() as db:
                assert db.execute('SELECT presentation FROM sample_presentations WHERE job_id=? AND ordinal=1',(job,)).fetchone()['presentation'] is None
            assert routes.samples(job,request)['review_summary']['usable']==3
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(original,first['run_id']))
            assert routes.samples(job,request)['review_summary']['usable']==4
            body=routes.ReviewDecision(revision=first['revision'],verdict='accepted')
            with pytest.raises(HTTPException):routes.review_sample(job,1,body,request)
            with pytest.raises(HTTPException):
                routes.review_sample(job,1,body.model_copy(update={'revision':'0'*64}),request)
            labels=routes.mapping_for(packets[1])
            expected=next(k for k,v in labels.items() if v==packets[1].target_object_id)
            wrong=next(k for k in labels if k!=expected)
            for choice in (wrong,expected):
                routes.confirm(job,1,routes.Confirm(revision=first['revision'],selected_candidate_id=choice),request)
                if choice==wrong:
                    with pytest.raises(HTTPException):routes.review_sample(job,1,body,request)
                else:
                    assert routes.review_sample(job,1,body,request)['verdict']=='accepted'
            # Explicit human approval may accept a failed automatic version;
            # ordinary acceptance, stale decisions and wrong localization cannot.
            from app.quality.sample_review import manually_approved
            baseline=runner.store.snapshot(first['run_id'])
            failed_version=json.loads(json.dumps(baseline))
            failed_version.update(status='needs_evidence',diagnostic_pass=False,final_expression=None,
                stop_reason='description_validation_failed')
            failed_version['attempts'][-1]['ground_checks']=[{'diagnostic_pass':False,'errors':['image_expression_not_unique']}]
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(failed_version),first['run_id']))
                db.execute("UPDATE batch_items SET status='needs_evidence' WHERE job_id=? AND ordinal=1",(job,))
            with pytest.raises(HTTPException):
                routes.review_sample(job,1,body,request)
            override=body.model_copy(update={'approve_failed':True})
            routes.confirm(job,1,routes.Confirm(revision=first['revision'],selected_candidate_id=wrong),request)
            with pytest.raises(HTTPException):routes.review_sample(job,1,override,request)
            routes.confirm(job,1,routes.Confirm(revision=first['revision'],selected_candidate_id=expected),request)
            approved=routes.review_sample(job,1,override,request)
            assert approved['approval_policy']=='localized-human-approval-v1'
            snapshot=runner.store.snapshot(first['run_id'])
            assert manually_approved(snapshot) and snapshot['diagnostic_pass'] is False and snapshot['final_expression'] is None
            assert snapshot['attempts'][-1]['ground_checks']==failed_version['attempts'][-1]['ground_checks']
            human_row=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
            assert human_row['manual_approved'] and human_row['usable'] and not human_row['automatic_passed']
            assert routes.samples(job,request)['review_summary']['usable']==4
            assert service.store.execution_progress(job)['passed']==4
            assert service.store.progress(job)['counts']['manual_approved']==1
            assert service.store.summaries()[0]['counts']['manual_approved']==1
            assert service.store.detail(job,compact=True)['selection_progress']['diagnostic_passed']==4
            assert manually_approved(next(s for s in runner.store.report_runs(job) if s['run_id']==first['run_id']))
            for change in ({'run_id':'f'*24},{'inflight':{'pending':True}},{'final_expression':'Changed text.'},
                           {'current_packet':{**snapshot['current_packet'],'source_view':'SIDE_RIGHT'}}):
                assert not manually_approved({**snapshot,**change})
            assert routes.review_sample(job,1,override,request)==approved
            routes.review_sample(job,1,routes.ReviewDecision(revision=first['revision'],verdict='rejected',notes='撤回审批。'),request)
            assert service.store.execution_progress(job)['passed']==3
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(baseline),first['run_id']))
                db.execute("UPDATE batch_items SET status='calibration_hold' WHERE job_id=? AND ordinal=1",(job,))
            for number,verdict in ((1,'rejected'),(2,'discarded')):
                row=next(s for s in listed['samples'] if s['ordinal']==number)
                if number==2:
                    # Legacy results stay visible until an explicit per-row
                    # deletion; stale text/identity must never delete a new one.
                    assert row['legacy_rules'] and any(s['ordinal']==number for s in routes.samples(job,request)['samples'])
                    body=routes.DeleteSample(run_id=row['run_id'],packet_id=row['packet_id'],expression=row['expression'])
                    for stale in (body.model_copy(update={'expression':'A stale description.'}),
                            body.model_copy(update={'run_id':'f'*24}),body.model_copy(update={'packet_id':'e'*24})):
                        with pytest.raises(HTTPException):routes.delete_sample(job,number,stale,request)
                    before=runner.store.snapshot(row['run_id'])
                    result=routes.delete_sample(job,number,body,request)
                    assert routes.delete_sample(job,number,body,request)==result
                    assert result['progress']['passed']==2
                    assert service.store.detail(job)['selection_progress']['diagnostic_passed']==2
                    header=next(j for j in result['jobs'] if j['job_id']==job)
                    assert header['counts']=={'calibration_hold':2,'rejected':1,'discarded':1}
                    assert service.store.detail(job)['counts']==header['counts']
                    after=runner.store.snapshot(row['run_id'])
                    assert after['model_calls_used']==before['model_calls_used']
                    assert after['final_expression']==before['final_expression'] and after['attempts']==before['attempts']
                else:
                    result=routes.review_sample(job,number,routes.ReviewDecision(revision=row['revision'],
                        verdict=verdict,notes='请使用更清晰的左右关系。'),request)
                if number==1: rejected=result
                assert rejection(runner.store.snapshot(row['run_id']))=='manual_'+verdict
            refreshed=routes.samples(job,request)
            assert refreshed['review_summary']=={'unreviewed':2,'accepted':0,'rejected':1,'discarded':1,'usable':2}
            assert set(routes.random_samples(job,request,count=10)['ordinals'])=={3,4}
            assert set(routes.random_samples(job,request,count=10,unreviewed=False)['ordinals'])=={1,3,4}
            retry_body=routes.RetryReview(revision=first['revision'],review_id=rejected['review_id'])
            with pytest.raises(HTTPException):
                routes.retry_sample(job,1,retry_body.model_copy(update={'review_id':'f'*64}),request)
            # Candidate exhaustion below the goal still requeues in THIS task.
            with runner.store.connect() as db:
                original_spec=json.loads(db.execute('SELECT spec FROM batches WHERE job_id=?',(job,)).fetchone()['spec'])
                original_spec.update(max_model_calls=service.store.reserved_calls(db,job),desired_count=5)
                db.execute("UPDATE batches SET spec=?,reason='candidate_pool_exhausted' WHERE job_id=?",(json.dumps(original_spec),job))
                task_before=dict(db.execute('SELECT job_id,spec,created_at FROM batches WHERE job_id=?',(job,)).fetchone())
                tasks_before=db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]
            # The feedback path accepts a generated draft whose automatic
            # validation failed; it must not need a fresh global candidate.
            with runner.store.connect() as db:
                failed=runner.store.snapshot(first['run_id'])
                failed.update(status='needs_evidence',diagnostic_pass=False,final_expression=None)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(failed),first['run_id']))
            source_before=runner.store.snapshot(first['run_id'])
            original_start=service.start
            def visible_before_start(job_id,*args,**kwargs):
                progress=service.store.progress(job_id)
                retry=next(row for row in progress['review_retries'] if row['ordinal']==1)
                assert retry['status']=='pending' and retry['run_id'] is None
                assert retry['calls']==0 and retry['call_budget']==12 and not retry['automatic_passed']
                # An old preparation update for the same ordinal is not this round.
                assert retry['preparation_stage'] is None
                # Requeueing keeps the original text/scene available for viewing,
                # without allowing the rejected version to count or be edited.
                visible=next(s for s in routes.samples(job_id,request)['samples'] if s['ordinal']==1)
                assert visible['viewing_previous'] and not visible['can_review'] and not visible['automatic_passed']
                assert visible['run_id']==first['run_id']
                viewed=routes.sample(service.store,job_id,1,allow_previous=True)
                assert viewed[1]['run_id']==first['run_id'] and viewed[4]==visible['revision']
                assert viewed[2]['status']=='running'
                with pytest.raises(ValueError):
                    routes.sample(service.store,job_id,1)
                round_view=routes.rework(job_id,1,request)
                assert round_view['item']['status']=='pending' and round_view['item']['attempts']==[]
                assert round_view['item']['previous_expression']==first['expression']
                with runner.store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute("UPDATE batch_items SET status='preparing' WHERE job_id=? AND ordinal=1",(job_id,))
                    service.store.event(db,job_id,'preparation_progress',{'ordinal':1,'stage':'validating_scene'})
                    preparing=next(row for row in service.store.review_retry_progress(db,job_id) if row['ordinal']==1)
                    assert preparing['preparation_stage']=='validating_scene'
                    db.rollback()
                return original_start(job_id,*args,**kwargs)
            with monkeypatch.context() as scheduling_only:
                scheduling_only.setattr(service.store,'detail',lambda *a,**kw:pytest.fail(
                    'An explicit review retry must not expand historical UI results before dispatch'))
                scheduling_only.setattr(service,'start',visible_before_start)
                response=routes.retry_sample(job,1,retry_body,request)
            assert response['job_id']==job
            wait(runner)
            child=service.store.detail(job)
            assert child['status']=='completed' and child['spec']['desired_count']==5
            updated=runner.store.snapshot(child['items'][0]['run_id'])
            with runner.store.connect() as db:
                assert dict(db.execute('SELECT job_id,spec,created_at FROM batches WHERE job_id=?',(job,)).fetchone())==task_before
                assert db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]==tasks_before
            assert runner.store.snapshot(first['run_id'])==source_before
            grant=child['items'][0]['target']['retry_authorization']
            assert grant['additional_calls']==12 and grant['calls_before']==6 and grant['call_limit']==18
            assert grant['consumed_run_id']==updated['run_id']
            assert updated['cumulative_call_limit']==18 and updated['model_call_budget']==12
            assert updated['run_id']!=first['run_id'] and updated['diagnostic_pass']
            retry=service.store.progress(job)['review_retries'][0]
            assert retry['ordinal']==1 and retry['run_id']==updated['run_id'] and retry['automatic_passed']
            assert retry['calls']==updated['model_calls_used'] and retry['call_budget']==12
            round_view=routes.rework(job,1,request)
            assert round_view['item']['automatic_passed']
            assert round_view['item']['attempts'][-1]['expression']==updated['final_expression']
            assert round_view['item']['attempts'][-1]['ground_rounds']==2
            assert routes.rework(job,4,request)['item'] is None
            # Historical feedback diagnostics are read-only, including in the
            # small task progress list; automatic success remains its own fact.
            with runner.store.connect() as db:
                db.execute('BEGIN')
                historical=json.loads(json.dumps(updated))
                historical.pop('feedback_check',None)
                historical['current_packet']['task'].pop('feedback_acceptance_policy',None)
                historical['current_packet']['task']['manual_review_feedback']['reason']='请描述向右行驶的运动状态'
                encoded=json.dumps(historical)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(encoded,updated['run_id']))
                old_progress=BatchStore.review_retry_progress(db,job,1)[0]
                assert old_progress['automatic_passed']
                assert old_progress['feedback_check']['historical']
                assert old_progress['feedback_check']['passed'] is False
                assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',(updated['run_id'],)).fetchone()[0]==encoded
                historical['current_packet']['task'].pop('feedback_detail_policy',None)
                historical['current_packet']['task']['manual_review_feedback']['reason']='说明与自车距离很近'
                historical['feedback_check']={'requirements':[], 'passed':None, 'blocking_errors':[]}
                encoded=json.dumps(historical)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(encoded,updated['run_id']))
                detailed=BatchStore.review_retry_progress(db,job,1)[0]
                assert detailed['feedback_check']['historical'] and detailed['automatic_passed']
                assert detailed['feedback_check']['requirements'][0]['kind']=='ego_distance'
                assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',(updated['run_id'],)).fetchone()[0]==encoded
                historical['current_packet']['task'].pop('feedback_relation_policy',None)
                historical['current_packet']['task']['manual_review_feedback']['reason']='可以说明这个 silver sedan 在 silver SUV的 后方'
                encoded=json.dumps(historical)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(encoded,updated['run_id']))
                mixed=BatchStore.review_retry_progress(db,job,1)[0]
                assert mixed['automatic_passed'] and mixed['feedback_check']['historical']
                assert mixed['feedback_check']['requirements'][0]['status']=='missing'
                assert mixed['feedback_check']['requirements'][0]['required'] is False
                assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',(updated['run_id'],)).fetchone()[0]==encoded
                historical['current_packet']['task'].update(feedback_acceptance_policy='feedback-acceptance-v5-natural-concision',
                    feedback_relation_policy='bilingual-landmark-feedback-v1')
                historical['current_packet']['task']['manual_review_feedback']['reason']='不需要 有 in front of a sedan in the front view'
                historical['attempts'][-1]['generated']['expression']='The pedestrian in front of a silver sedan in the front view.'
                historical['feedback_check']={'requirements':[{'kind':'landmark_correction','label':'采用：in front of a sedan',
                    'status':'covered','required':True}], 'passed':True,'blocking_errors':[]}
                encoded=json.dumps(historical)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(encoded,updated['run_id']))
                corrected=BatchStore.review_retry_progress(db,job,1)[0]
                assert corrected['automatic_passed'] and corrected['feedback_check']['historical']
                assert not corrected['feedback_check']['passed']
                assert corrected['feedback_check']['requirements']==[r for r in corrected['feedback_check']['requirements'] if r['label'].startswith('排除：')]
                assert corrected['feedback_check']['blocking_errors']==['manual_feedback_rejected_relation_repeated']
                assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',(updated['run_id'],)).fetchone()[0]==encoded
                historical['current_packet']['task'].pop('feedback_feature_policy',None)
                historical['current_packet']['task']['manual_review_feedback']['reason']='目标在拿着摄像机'
                historical['attempts'][-1]['generated'].update(expression='The pedestrian holding a bag in the front view.',
                    visual_attributes=[{'kind':'feature','value':'handheld bag'}])
                historical['feedback_check']={'requirements':[],'passed':None,'blocking_errors':[]}
                encoded=json.dumps(historical)
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(encoded,updated['run_id']))
                held=BatchStore.review_retry_progress(db,job,1)[0]
                assert held['feedback_check']['historical'] and not held['feedback_check']['passed']
                assert held['feedback_check']['requirements'][0]['kind']=='target_feature'
                assert held['feedback_check']['requirements'][0]['status']=='missing'
                assert held['feedback_check']['blocking_errors']==['manual_feedback_target_feature_missing']
                assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',(updated['run_id'],)).fetchone()[0]==encoded
                db.rollback()
            assert updated['final_expression']!='The vehicle to the left of the ego vehicle.'
            assert len(updated['attempts'][-1]['ground_checks'])==2 and 'manual_review' not in updated
            assert service.store.execution_progress(job)['passed']==3
            assert runner.store.target_usage(packets[1],LocalCalls.scope)==12
            duplicate=routes.retry_sample(job,1,retry_body,request)
            assert duplicate['job_id']==job and runner.store.target_usage(packets[1],LocalCalls.scope)==12
            row=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
            next_body=routes.RetryReview(revision=row['revision'],notes='请使用更清晰的左右关系。')
            response=routes.retry_sample(job,1,next_body,request)
            assert response['job_id']==job
            wait(runner)
            child=service.store.detail(job)
            next_state=runner.store.snapshot(child['items'][0]['run_id'])
            assert next_state['diagnostic_pass'] and len(next_state['attempts'][-1]['ground_checks'])==2
            assert next_state['cumulative_call_limit']==24 and next_state['model_call_budget']==12
            assert runner.store.target_usage(packets[1],LocalCalls.scope)==12+next_state['model_calls_used']
            assert not runner.store.candidate_retries()
            # While another sample is executing, append a retry to this SAME
            # coordinator. It must notice the new round without a second task.
            retry_gate.update(context=packets[3].context,entered=Event(),release=Event())
            third=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==3)
            first_retry=routes.retry_sample(job,3,routes.RetryReview(revision=third['revision'],notes='请使用更清晰的左右关系。'),request)
            assert first_retry['job_id']==job and retry_gate['entered'].wait(10)
            try:
                progress=service.store.progress(job)
                retry=next(row for row in progress['review_retries'] if row['ordinal']==3)
                assert progress['executor_active'] and retry['status']=='running'
                assert retry['phase'] and retry['run_id'] and retry['model_pending']
                current=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
                pipeline_count=sum(e['kind']=='execution_pipeline_started' for e in service.store.detail(job)['events'])
                assert routes.retry_sample(job,1,routes.RetryReview(revision=current['revision'],notes='请使用更清晰的左右关系。'),request)['job_id']==job
                assert runner.active_batch==job
                assert sum(e['kind']=='execution_pipeline_started' for e in service.store.detail(job)['events'])==pipeline_count
            finally:
                retry_gate['release'].set()
            wait(runner)
            retry_gate.clear()
            assert service.store.execution_progress(job)['passed']==3
            # At the goal, reject BEFORE writing a rejection or granting calls.
            with runner.store.connect() as db:
                specification=json.loads(db.execute('SELECT spec FROM batches WHERE job_id=?',(job,)).fetchone()['spec'])
                specification['desired_count']=3
                db.execute('UPDATE batches SET spec=? WHERE job_id=?',(json.dumps(specification),job))
            current=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
            before=runner.store.snapshot(current['run_id'])
            with pytest.raises(HTTPException,match='先手动删除'):
                routes.retry_sample(job,1,routes.RetryReview(revision=current['revision'],notes='请使用更清晰的左右关系。'),request)
            assert runner.store.snapshot(current['run_id'])==before
            assert service.store.execution_progress(job)['passed']==3
            # An explicit deletion opens a slot; retry stays in the original task.
            other=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==3)
            routes.delete_sample(job,3,routes.DeleteSample(run_id=other['run_id'],packet_id=other['packet_id'],expression=other['expression']),request)
            assert service.store.execution_progress(job)['passed']==2
            assert routes.retry_sample(job,1,routes.RetryReview(revision=current['revision'],notes='请使用更清晰的左右关系。'),request)['job_id']==job
            wait(runner)
            assert service.store.execution_progress(job)['passed']==2  # One object's replacement is counted once.
            with runner.store.connect() as db:
                assert db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]==tasks_before
            # A settled rework with no draft displays its old input. It can be
            # retried with explicit feedback, but cannot approve/export that input.
            latest=service.store.detail(job)['items'][0]
            empty=runner.store.snapshot(latest['run_id'])
            empty.update(status='needs_evidence',diagnostic_pass=False,attempts=[],final_expression=None,
                inflight=None,stop_reason='precheck_no_distinguishing_cues')
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(empty),latest['run_id']))
                db.execute("UPDATE batch_items SET status='needs_evidence' WHERE job_id=? AND ordinal=1",(job,))
            shown=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
            assert shown['viewing_previous'] and shown['can_retry'] and not shown['can_review'] and not shown['usable']
            assert shown['retry_failed_run_id']==empty['run_id'] and shown['run_id']!=empty['run_id']
            again=routes.RetryReview(revision=shown['revision'],notes='请使用更清晰的左右关系。',retry_failed_run_id=empty['run_id'])
            for invalid in (again.model_copy(update={'retry_failed_run_id':'f'*24}),again.model_copy(update={'revision':'e'*64})):
                with pytest.raises(HTTPException):routes.retry_sample(job,1,invalid,request)
            with pytest.raises(HTTPException):
                routes.review_sample(job,1,routes.ReviewDecision(revision=shown['revision'],verdict='accepted',approve_failed=True),request)
            for status,inflight in [('running',None),('paused',None),('needs_evidence',{'pending':True})]:
                with runner.store.connect() as db:
                    db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps({**empty,'status':status,'inflight':inflight}),empty['run_id']))
                blocked=next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)
                assert not blocked['can_retry']
                with pytest.raises(HTTPException):routes.retry_sample(job,1,again,request)
            with runner.store.connect() as db:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(empty),empty['run_id']))
            spent=runner.store.target_usage(packets[1],LocalCalls.scope)
            with monkeypatch.context() as queue_only:
                queue_only.setattr(service,'start',lambda *_args,**_kwargs:{'queued':True})
                assert routes.retry_sample(job,1,again,request)['queued']
                grant=service.store.detail(job)['items'][0]['target']['retry_authorization']
                assert grant['call_limit']==spent+12 and grant['source_run_id']==empty['run_id']
                assert grant['feedback_source_run_id']==shown['run_id']
                assert routes.retry_sample(job,1,again,request)['queued']
                assert service.store.detail(job)['items'][0]['target']['retry_authorization']==grant
            assert runner.store.snapshot(empty['run_id'])==empty
            assert runner.store.target_usage(packets[1],LocalCalls.scope)==spent
            assert not next(s for s in routes.samples(job,request)['samples'] if s['ordinal']==1)['can_retry']
        finally:
            service.close();runner.close()


@pytest.fixture
def workspace(tmp_path):
    annotation = tmp_path / 'runtime/annotation'
    store = RunStore(annotation / 'runs.sqlite3')
    BatchStore(store)
    ResearchStore(store, tmp_path)
    cleaner = TaskCleaner(store, tmp_path)
    raw = tmp_path / 'waymo_trainset/input.parquet'
    raw.parent.mkdir(); raw.write_bytes(b'original data')
    config = tmp_path / '.env'; config.write_text('original configuration')

    def task(n, *, run=True, packet=None, scene=None, method=None):
        jid, rid, pid, sid, mid = ident(n), ident(n + 100), packet or ident(n + 200), scene or ident(n + 300), method or ident(n + 400, 64)
        target = {'target_id': ident(n + 500), 'context': 'context', 'object_id': 'object-' + str(n),
                  'timestamp': n, 'ordinal': 1, 'category': 'vehicle', 'status': 'selected'}
        p = {'packet_id': pid, 'context': target['context'], 'target_object_id': target['object_id'],
             'frame_timestamp_micros': target['timestamp'], 'scene': {'scene_id': sid,
              'manifest_path': f'runtime/scenes/{sid}/scene.json'}}
        packet_folder = tmp_path / 'runtime/evidence' / pid
        if not packet_folder.exists():
            atomic_json(packet_folder / 'packet.json', p)
            (packet_folder / 'image.jpg').write_bytes(b'image input')
        scene_folder = tmp_path / 'runtime/scenes' / sid
        if not scene_folder.exists():
            atomic_json(scene_folder / 'scene.json', {'scene_id': sid, 'context': 'context', 'frame_timestamp_micros': n})
            (scene_folder / 'points.npz').write_bytes(b'complete points')
        atomic_json(annotation / 'methods' / mid / 'manifest.json', {'method_sha256': mid})
        batch_folder = annotation / 'batches' / jid
        batch_folder.mkdir(parents=True); (batch_folder / 'preparation.log').write_text('task-specific preparation')
        state = {'run_id': rid, 'packet_id': pid, 'batch_id': jid, 'current_packet': p,
                 'model_calls_used': 4, 'status': 'calibration_hold'}
        with store.connect() as db:
            db.execute('INSERT INTO batches VALUES(?,?,?,?,?,?,?,?,?)',
                       (jid, json.dumps({'prepare_only': not run, 'catalog_id': ident(900),
                         'target_ids': [target['target_id']], 'max_model_calls': 12 if run else 0,
                         'calls_per_target': 12, 'max_repairs': 2}), 'hash', mid, 'completed', None, None, None, time.time()))
            db.execute('INSERT INTO batch_items VALUES(?,?,?,?,?,?,?,?)',
                       (jid, 1, json.dumps(target), 'calibration_hold' if run else 'failed', pid, rid if run else None, None, 0))
            db.execute('INSERT INTO batch_events(job_id,kind,payload,at) VALUES(?,?,?,?)', (jid, 'created', '{}', time.time()))
            if run:
                db.execute('INSERT INTO runs(run_id,snapshot) VALUES(?,?)', (rid, json.dumps(state)))
                db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (rid, 1, 'finished', json.dumps({'packet_id': pid}), time.time()))
                db.execute('INSERT INTO controls VALUES(?,?,?)', (rid, 'stop', time.time()))
        if run:
            atomic_json(annotation / rid / 'checkpoint.json', state)
            atomic_json(annotation / rid / 'model/0/input.json', {'prompt': 'candidate input'})
            atomic_json(annotation / rid / 'model/0/result.json', {'tokens': 20})
        return SimpleNamespace(job=jid, run=rid, packet=pid, scene=sid, method=mid, target=target, state=state)

    return SimpleNamespace(root=tmp_path, annotation=annotation, store=store, cleaner=cleaner, task=task, raw=raw, config=config)


@pytest.fixture
def scene(tmp_path):
    folder = tmp_path / "scene"
    folder.mkdir()
    arrays = dict(points=np.array([[0, 0, 0, .2, .1, 1], [0, 1.5, 0, .3, .1, 2], [3, 0, 0, .4, .1, 3]], np.float32),
                  laser_id=np.ones(3, np.uint8), return_id=np.ones(3, np.uint8),
                  range_image_pixel=np.zeros((3, 2), np.int32), camera_projection=np.zeros((3, 6), np.int32), nlz=np.zeros(3, np.int8))
    np.savez_compressed(folder / "points.npz", **arrays)
    atomic_json(folder / "calibration.json", {"camera_calibration": []})
    cameras = []
    for camera in range(1, 6):
        filename = f"camera-{camera}.jpg"
        Image.new("RGB", (8, 6)).save(folder / filename)
        cameras.append(dict(camera_name=camera, image=dict(path=filename, sha256=file_sha256(folder / filename)), width=8, height=6,
                            pose_timestamp_s=None, trigger_time_s=None, readout_done_time_s=None))
    manifest = dict(scene_id="a" * 24, context="context", frame_timestamp_micros=1, official_split="training",
                    points=dict(path="points.npz", sha256=file_sha256(folder / "points.npz")), point_count=3,
                    sensor_returns={f"{laser}/{ret}": 3 if (laser, ret) == (1, 1) else 0 for laser in range(1, 6) for ret in (1, 2)},
                    calibration=dict(path="calibration.json", sha256=file_sha256(folder / "calibration.json")), cameras=cameras,
                    conversion={}, source_components={}, diagnostics={})
    atomic_json(folder / "scene.json", manifest)
    return folder, arrays, manifest


def test_native_geometry_keeps_ambiguous_and_unknown_competitors():
    from app.quality.contrast import complete_ego_direction_ids, ego_wording_errors, EGO_WORDING_POLICY
    direction_packet = packet()
    direction_packet.task['ego_wording_policy'] = EGO_WORDING_POLICY
    for x,y,longitudinal,lateral in [(15.2,17.6,'front','left'),(15.2,-17.6,'front','right'),
            (-15.2,17.6,'behind','left'),(-15.2,-17.6,'behind','right')]:
        direction_packet.target = obj('direction',x,y)
        ids = ['longitudinal:'+longitudinal]
        assert complete_ego_direction_ids(direction_packet,ids) == [*ids,'lateral:'+lateral]
        assert ego_wording_errors(direction_packet,ids) == ['incomplete_ego_direction']
        assert not ego_wording_errors(direction_packet,[*ids,'lateral:'+lateral])
        assert not ego_wording_errors(direction_packet,[])  # Pure appearance needs no direction.
    direction_packet.target = obj('direction',20,0)
    assert not ego_wording_errors(direction_packet,['longitudinal:front'])
    # #5129: footprint-supported right remains useful even at <22.5 degrees.
    direction_packet.target = obj('direction',14.212,-3.907)
    assert complete_ego_direction_ids(direction_packet,['longitudinal:front'])==['longitudinal:front']
    direction_packet.task['ego_wording_policy']='proven-ego-components-v2'
    assert complete_ego_direction_ids(direction_packet,['longitudinal:front'])==['longitudinal:front','lateral:right']
    assert ego_wording_errors(direction_packet,['longitudinal:front'])==['incomplete_ego_direction']
    direction_packet.target=obj('direction',14.212,-1.1)  # Straddling the guard is UNKNOWN, not right.
    assert complete_ego_direction_ids(direction_packet,['longitudinal:front'])==['longitudinal:front']
    direction_packet.task['ego_wording_policy']=EGO_WORDING_POLICY
    direction_packet.target = obj('direction',15.2,17.6,points=0)
    assert 'ego_direction_not_supported' in ego_wording_errors(direction_packet,['longitudinal:front','lateral:left'])
    direction_packet.task.pop('ego_wording_policy')
    assert not ego_wording_errors(direction_packet,['longitudinal:front'])  # Frozen old semantics.
    from app.quality.contrast import ego_near_match, ego_near_errors
    for center, expected in [((3,4,0),True),((5.01,0,0),None),((7,0,0),None),((7.01,0,0),False),((0,0,8),False)]:
        candidate = obj('near',1,1).model_copy(update={'center_m':center,'lidar_points':10})
        assert ego_near_match(candidate) is expected
        assert ego_near_match(candidate.model_copy(update={'lidar_points':0})) is None
        assert ego_near_match(candidate.model_copy(update={'geometry_valid':False})) is None
    assert 'ego_distance_claim_mismatch' in ego_near_errors(packet(),'The vehicle is not close to the ego vehicle.',['range:ego_near'])
    from app.quality.relations import vehicle_heading_match, vehicle_forward_region_match
    # Heading-relative front/rear rotates with the landmark, unlike ego axes.
    for angle in (0., math.pi/2, -1.565, math.pi):
        anchor = obj('red-car',3,7).model_copy(update={'heading_rad':angle})
        def local_object(name,forward,side):
            return obj(name,3+forward*math.cos(angle)-side*math.sin(angle),
                7+forward*math.sin(angle)+side*math.cos(angle)).model_copy(update={'heading_rad':angle})
        ahead = local_object('ahead',10,.1)
        behind = local_object('behind',-10,.1)
        assert vehicle_heading_match(ahead,anchor) is True
        assert vehicle_heading_match(behind,anchor) is False
        assert vehicle_heading_match(behind,anchor,-1) is True
        assert vehicle_heading_match(ahead,anchor,-1) is False
        assert vehicle_heading_match(local_object('side',10,5),anchor) is False
        assert vehicle_forward_region_match(local_object('side',10,5),anchor) is True
        assert vehicle_forward_region_match(local_object('side',10,5),anchor,-1) is False
        assert vehicle_forward_region_match(local_object('overlap',3,5),anchor) is None
        assert vehicle_forward_region_match(local_object('above',10,2).model_copy(update={'center_m':(ahead.center_m[0],ahead.center_m[1],9)}),anchor) is False
        assert vehicle_forward_region_match(ahead,anchor.model_copy(update={'lidar_points':0})) is None
        assert vehicle_forward_region_match(ahead.model_copy(update={'lidar_points':0}),anchor) is None
        assert vehicle_forward_region_match(anchor,anchor) is False
        assert vehicle_heading_match(local_object('marginal',10,.8),anchor) is None
        assert vehicle_heading_match(local_object('overlap',3,0),anchor) is None
        assert vehicle_heading_match(ahead.model_copy(update={'geometry_valid':False}),anchor) is None
        assert vehicle_heading_match(ahead,anchor.model_copy(update={'lidar_points':0})) is None
        assert vehicle_heading_match(anchor,anchor) is False
    from app.quality.geometry import inside_oriented_box, BoxPointIndex
    rng = np.random.default_rng(10)
    for dtype in (np.float32, np.float64):
        xyz = rng.normal(size=(2000, 3)).astype(dtype)
        for heading in (0., .37, -1.7, math.pi / 2):
            center, size = np.array([.5,-.3,.2]), np.array([2.,3.,1.])
            # Include exact inclusive boundaries in addition to random points.
            points = np.vstack((xyz, center + [0,0,.51], center - [0,0,.51]))
            delta = points - center
            c, s = np.cos(heading), np.sin(heading)
            local = np.column_stack((delta[:,0]*c+delta[:,1]*s,
                -delta[:,0]*s+delta[:,1]*c, delta[:,2]))
            assert np.array_equal(inside_oriented_box(points, center, size, heading),
                np.all(np.abs(local) <= size/2+.01, axis=1))
    assert inside_oriented_box(np.empty((0,3)), (0,0,0), (1,1,1), 0).shape == (0,)
    # A conservative broad phase must preserve rotated faces/corners, remote
    # points and inclusive tolerance boundaries, including thin native parts.
    for heading in (0., .37, -1.7, math.pi / 2):
        for size in (np.array([.001, 50., .1]), np.array([10., .02, 8.])):
            c, s = np.cos(heading), np.sin(heading)
            local = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*(size/2+.01)
            world = np.column_stack((local[:,0]*c-local[:,1]*s, local[:,0]*s+local[:,1]*c, local[:,2]))
            points = np.vstack((world, rng.uniform(-10000,10000,(10000,3))))
            rotated = np.column_stack((points[:,0]*c+points[:,1]*s,-points[:,0]*s+points[:,1]*c,points[:,2]))
            assert np.array_equal(inside_oriented_box(points,(0,0,0),size,heading),
                np.all(np.abs(rotated)<=size/2+.01,axis=1))
            assert np.array_equal(BoxPointIndex(points).inside((0,0,0),size,heading),
                np.all(np.abs(rotated)<=size/2+.01,axis=1))
    for dtype in (np.float32, np.float64):
        cloud = rng.uniform(-1000,1000,(10000,3)).astype(dtype)
        index = BoxPointIndex(cloud)
        for heading in (0., .37, -1.7):
            for center in ((0.,0.,0.),(400.,-200.,.1),(1e8,1e8,1e8)):
                assert np.array_equal(index.inside(center,(12.,30.,5.),heading),
                    inside_oriented_box(cloud,center,(12.,30.,5.),heading))
        with pytest.raises(ValueError, match='positive sizes'):
            index.inside((0,0,0),(0,1,1),0)
    with pytest.raises(ValueError, match='Finite Nx3'):
        BoxPointIndex(np.full((3000,3), np.nan))
    target = obj('target', 10, 1.2).model_copy(update={'size_m': (10, 3, 3)})
    assert ego_direction_match(target, 1, True) is None
    rival = target.model_copy(update={'object_id':'rival', 'lidar_points':0})
    objects = [target, rival]
    plan = contrast_plan(objects, target.object_id, predicate_catalog(objects))
    assert set(plan.remaining_object_ids) == {'target', 'rival'}
    assert plan.status != 'unique_in_gt_candidates'


def test_partial_target_recheck_and_blind_checks_within_budget(tmp_path, monkeypatch):
    from app.agent import agent_runtime as runtime
    from app.schemas.contracts import SceneVisualObservation
    camera_packet = strict_packet(tmp_path/'camera-feedback-evidence',monkeypatch)
    # #1642: a large projected rectangle may contain a different pickup behind
    # the native target. Send the existing point guide AND the clean photograph;
    # do not turn the returns into visible paint or a segmentation mask.
    from app.agent.harness import appearance_messages
    guide_folder=tmp_path/'camera-feedback-evidence'/camera_packet.packet_id
    shutil.copyfile(guide_folder/'appearance.jpg',guide_folder/'point-guide.jpg')
    camera_packet.views[0].artifacts['appearance_points']='point-guide.jpg'
    camera_packet.provenance['artifact_sha256']['point-guide.jpg']=file_sha256(guide_folder/'point-guide.jpg')
    camera_packet.task['identity_point_guidance_policy']='box-point-owner-guide-v1'
    guide_messages=appearance_messages(camera_packet)
    guide_text=[c['text'] for c in guide_messages[-1]['content'] if c['type']=='text']
    assert any('NOT an instance mask' in text and 'do NOT borrow' in text for text in guide_text)
    assert 'Target appearance,' in guide_text[-1] and 'Target appearance_points' not in guide_text[-1]
    legacy_guide=camera_packet.model_copy(deep=True);legacy_guide.task.pop('identity_point_guidance_policy')
    assert not any('Target appearance_points' in c.get('text','') for c in appearance_messages(legacy_guide)[-1]['content'])
    camera_packet.candidates[0].category='pedestrian'
    camera_packet.target=camera_packet.candidates[0]
    camera_packet.predicates=predicate_catalog(camera_packet.candidates)
    camera_packet.task.update(feedback_feature_policy='verified-target-feature-feedback-v1',
        feedback_acceptance_policy='feedback-acceptance-v5-natural-concision',
        manual_review_feedback={'reason':'目标在拿着摄像机'})
    camera_packet.provenance['appearance_candidates']=[{'object_id':camera_packet.target_object_id,
        'candidate_id':'person0','category':'pedestrian','camera_name':1,'artifact':'appearance.jpg'}]
    camera_root=tmp_path/'camera-feedback-runs'
    bag_observation=VisualObservation(target_visible=True,visibility='clear',
        visibility_reason='The pedestrian is visible and has a bag.',
        attributes=[VisualAttribute(kind='feature',value='handheld bag')])
    camera_observation=bag_observation.model_copy(update={'visibility_reason':'The pedestrian is holding a camera.', 'attributes':[
        VisualAttribute(kind='feature',value='handheld camera')]})
    class CameraRecheckCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            assert schema is SceneVisualObservation
            payload=json.loads(messages[-1]['content'][0]['text'])
            assert payload['attribute_checks']==[{'candidate_id':'person0','kinds':['feature']}]
            assert 'manual_review_feedback' not in payload and 'previous_observation' not in payload
            self.count+=1
            RunStore(camera_root/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
            return SceneVisualObservation.model_validate({'candidates':[{
                **camera_observation.model_dump(),'candidate_id':'person0'}]})
    camera_calls=CameraRecheckCalls()
    camera_state=runtime.initial_run(camera_packet,camera_calls,1,12)
    camera_state.update(phase='precheck_decision',target_precheck_observation=bag_observation.model_dump(),
        candidate_appearance={camera_packet.target_object_id:{**bag_observation.model_dump(),'observed_camera_name':1}})
    RunStore(camera_root/'runs.sqlite3').create(camera_state)
    with monkeypatch.context() as focused:
        focused.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'camera-feedback-evidence')
        checked=runtime.run_agent(camera_packet,camera_calls,1,camera_root,12,resume_run_id=camera_state['run_id'])
    assert checked['status']=='paused' and camera_calls.count==checked['model_calls_used']==1
    assert checked['feedback_target_rechecked']
    assert checked['target_precheck_observation']['attributes']==[{'kind':'feature','value':'handheld camera'}]
    assert checked['source_attribute_rechecks'][0]['previous_observation']==bag_observation.model_dump()
    # #4102: gray is a request, not an observation. Reinspect only the requested
    # KINDS in a fresh anonymous read; a differing white result stays white.
    surface_packet=camera_packet.model_copy(deep=True)
    surface_packet.task.update(feedback_appearance_policy='explicit-appearance-feedback-v1',
        manual_review_feedback={'reason':'应该只看到这个标注牌的背面，是灰色的'})
    surface_root=tmp_path/'surface-feedback-runs'
    class SurfaceRecheckCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            assert schema is SceneVisualObservation
            payload=json.loads(messages[-1]['content'][0]['text'])
            assert payload['attribute_checks']==[{'candidate_id':'person0','kinds':['color','feature']}]
            assert 'manual_review_feedback' not in payload and 'previous_observation' not in payload
            self.count+=1
            RunStore(surface_root/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
            return SceneVisualObservation.model_validate({'candidates':[{
                **bag_observation.model_dump(),'attributes':[{'kind':'color','value':'white'}],
                'candidate_id':'person0'}]})
    surface_calls=SurfaceRecheckCalls()
    surface_state=runtime.initial_run(surface_packet,surface_calls,1,12)
    surface_state.update(phase='precheck_decision',target_precheck_observation=bag_observation.model_dump(),
        candidate_appearance={surface_packet.target_object_id:{**bag_observation.model_dump(),'observed_camera_name':1}})
    RunStore(surface_root/'runs.sqlite3').create(surface_state)
    with monkeypatch.context() as focused:
        focused.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'camera-feedback-evidence')
        surface=runtime.run_agent(surface_packet,surface_calls,1,surface_root,12,resume_run_id=surface_state['run_id'])
    assert surface['status']=='paused' and surface['model_calls_used']==surface_calls.count==1
    assert surface['target_precheck_observation']['attributes']==[{'kind':'color','value':'white'}]
    # Symbolic ambiguity can request a complete-scene proof; every unit is
    # included, IDs permute, and two fresh successful answers are mandatory.
    from app.agent import scene_grounding
    from app.agent.agent_runtime import initial_run, run_agent
    p= strict_packet(tmp_path/'blind-scene-evidence',monkeypatch)
    p.task.update(annotation_search='evidence-guided-v1',scene_grounding_policy=scene_grounding.POLICY,
        reference_policy='minimal-sufficient-grounding-v1')
    p.candidates[1]=p.candidates[1].model_copy(update={'center_m':(20,5,1)})
    p.predicates=predicate_catalog(p.candidates)
    p.provenance['scene_image_candidates']=[
        {'object_id':p.target_object_id,'category':'vehicle','camera_name':1,'xyxy':[20,20,80,99]},
        {'object_id':'image:unmatched','category':'unknown','camera_name':1,'xyxy':[45,10,55,20]}]
    p.ready_for_generation=True
    monkeypatch.setattr(scene_grounding,'EVIDENCE_ROOT',tmp_path/'blind-scene-evidence')
    generated=GeneratedExpression(expression='The white vehicle directly in front of the ego vehicle.',
        predicate_ids=['category:vehicle','longitudinal:front'],visual_attributes=visible().attributes)
    class SceneCalls(LocalCalls):
        def __init__(self,mode):super().__init__();self.mode=mode;self.payloads=[]
        def call(self,messages,schema,folder,heartbeat=None):
            assert schema is scene_grounding.SceneGroundResult
            self.count+=1
            payload=json.loads(messages[-1]['content'][0]['text']);self.payloads.append(payload)
            assert p.target_object_id not in json.dumps(messages) and 'manual_review_feedback' not in json.dumps(messages)
            assert len(payload['candidates'])==len(p.candidates)+1
            selected=next(c['candidate_id'] for c in payload['candidates'] if c.get('center_m',[None])[0]==10)
            rival=next(c['candidate_id'] for c in payload['candidates'] if c.get('identity')=='unmatched_image')
            if self.mode=='wrong_second' and self.count==2:selected=rival
            return schema(expression_supported=self.mode!='unsupported',candidate_ids=['C9999' if self.mode=='invalid' else selected],
                ambiguous_candidate_ids=[rival] if self.mode=='ambiguous' else [],
                unsupported_claims=['unsupported detail'] if self.mode=='unsupported' else [],reason='Whole-scene check.')
    for mode in ('passed','wrong_second','ambiguous','invalid','unsupported'):
        calls=SceneCalls(mode);root=tmp_path/f'scene-check-{mode}'
        initial=initial_run(p,calls,0,2)
        initial.update(phase='ground',relations_acquired=True,admission={'passed':True},
            attempts=[{'attempt':0,'generated':generated.model_dump(),'ground_checks':[],
                       'appearance_check':{'passed':True}}],candidate_appearance={p.target_object_id:clear().model_dump()})
        store=RunStore(root/'runs.sqlite3');store.create(initial)
        result=run_agent(p,calls,0,root,2,resume_run_id=initial['run_id'])
        assert result['diagnostic_pass']==(mode=='passed'),result
        assert calls.count==(2 if mode in {'passed','wrong_second'} else 1)
        if mode=='passed':
            assert result['quality_gate']['scene_grounding_checks_passed'] and not result['quality_gate']['oracle_checks_passed']
            assert calls.payloads[0]['candidates']!=calls.payloads[1]['candidates']
            assert run_agent(p,calls,0,root,2,resume_run_id=initial['run_id'])==result and calls.count==2
    proximity = p.model_copy(deep=True)
    proximity.task['ego_proximity_policy'] = 'native-ego-proximity-v1'
    near_text = 'The white vehicle in the front view is close to the ego vehicle.'
    inputs, mapping, receipt = scene_grounding.messages(proximity,near_text,17)
    table = json.loads(inputs[-1]['content'][0]['text'])['ego_proximity']['candidates']
    assert len(table) == len(mapping)
    assert all(row['close'] is None for row in table if mapping[row['candidate_id']]=='image:unmatched')
    ego_text='The white vehicle ahead and to the right of the ego vehicle.'
    proximity.task['scene_ego_direction_policy']='explicit-unary-ego-table-v1'
    inputs,ego_mapping,_=scene_grounding.messages(proximity,ego_text,17)
    ego_payload=json.loads(inputs[-1]['content'][0]['text'])
    unary=ego_payload['ego_directions']
    assert unary['required_conditions']==['lateral:right','longitudinal:front']
    assert len(unary['candidates'])==len(ego_mapping)
    assert all(row['longitudinal:front'] is None for row in unary['candidates'] if ego_mapping[row['candidate_id']]=='image:unmatched')
    assert 'Never search for an ego landmark ID' in inputs[0]['content']
    appearance_inputs,_,_=scene_grounding.messages(proximity,'The white vehicle in the front view.',17)
    assert 'ego_directions' not in json.loads(appearance_inputs[-1]['content'][0]['text'])
    target_id = next(cid for cid,oid in mapping.items() if oid==proximity.target_object_id)
    answer = scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[target_id],ambiguous_candidate_ids=[],unsupported_claims=[],reason='selected')
    # #1751: a rejected image-only ID is NOT a passing native result. Acquire
    # another source instead; geometry/semantic failures and old runs stay strict.
    identity_packet=proximity.model_copy(deep=True)
    identity_packet.task['scene_source_retry_policy']=scene_grounding.SOURCE_RETRY_POLICY
    identity_failure={'diagnostic_pass':False,'errors':['scene_grounding_not_unique_target'],
        'tool_grounded_candidates':['image:unmatched'],'ambiguous_candidate_ids':[], 'unsupported_claims':[]}
    assert scene_grounding.needs_identity_source_retry(identity_packet,identity_failure,{})
    ambiguous_failure={**identity_failure,'tool_grounded_candidates':[],
        'ambiguous_candidate_ids':['V01','V02'],'errors':[
            'scene_grounding_not_unique_target','scene_grounding_ambiguous','scene_grounding_unsupported_claims']}
    assert scene_grounding.needs_identity_source_retry(identity_packet,ambiguous_failure,
        {'V01':identity_packet.target_object_id,'V02':'image:unmatched'})
    legacy_identity=proximity.model_copy(deep=True)
    legacy_identity.task.pop('scene_source_retry_policy',None)
    assert not scene_grounding.needs_identity_source_retry(legacy_identity,identity_failure,{})
    assert not scene_grounding.needs_identity_source_retry(identity_packet,
        {**identity_failure,'unsupported_claims':['red']},{})
    assert not scene_grounding.needs_identity_source_retry(identity_packet,
        {**identity_failure,'errors':['joint_reference_pair_unverified','scene_grounding_not_unique_target']},{})
    assert not scene_grounding.needs_identity_source_retry(identity_packet,
        {**identity_failure,'tool_grounded_candidates':[identity_packet.target_object_id]}, {})
    native_retry=identity_packet.model_copy(deep=True)
    native_retry.task['scene_source_retry_policy']=scene_grounding.NATIVE_OWNER_RETRY_POLICY
    wrong_native={**identity_failure,'tool_grounded_candidates':['other-native']}
    assert scene_grounding.needs_identity_source_retry(native_retry,wrong_native,{})
    assert not scene_grounding.needs_identity_source_retry(identity_packet,wrong_native,{})
    assert not scene_grounding.needs_identity_source_retry(native_retry,
        {**wrong_native,'errors':['joint_reference_pair_unverified','scene_grounding_not_unique_target']},{})
    near_generated = {'expression':near_text,'predicate_ids':['category:vehicle','range:ego_near']}
    assert 'ego_distance_not_supported' in scene_grounding.verify(proximity,answer,mapping,receipt,generated=near_generated)['errors']
    # #4094: two white panels on one pole remain two native IDs. A literal
    # short inscription can distinguish them, but never supplies a target hint.
    from app.agent.observation_wire import CompactAttribute,unknown_optional_evidence,RGBVisualObservation
    from app.quality.description_search import appearance_phrase,descriptive_attributes
    word=VisualAttribute(kind='inscription',value='begin')
    assert word.value=='BEGIN' and CompactAttribute(k='inscription',v='begin').value=='BEGIN'
    raw={'attributes':[word.model_dump()]}
    assert unknown_optional_evidence(raw,RGBVisualObservation)==(raw,[])
    for unreadable in ['UNKNOWN','a '*25,'BEGIN\nNO PARKING','<script>']:
        with pytest.raises(ValueError):VisualAttribute(kind='inscription',value=unreadable)
    with pytest.raises(ValueError):VisualAttribute(kind='feature',value='BEGIN')
    literal=p.model_copy(deep=True)
    literal.task['sign_inscription_policy']='literal-sign-inscription-v1'
    literal.task['description_detail_policy']='confirmed-road-scene-v2'
    literal.candidates=literal.candidates[:2]
    for candidate in literal.candidates:candidate.category='sign'
    literal.target.category='sign';literal.predicates=predicate_catalog(literal.candidates)
    literal.views[0].target_front_facing=True
    other_id=literal.candidates[1].object_id
    literal.provenance['scene_image_candidates']=[
        {'object_id':literal.target_object_id,'category':'sign','camera_name':1,'xyxy':[10,50,40,65]},
        {'object_id':other_id,'category':'sign','camera_name':1,'xyxy':[10,10,40,45]},
        {'object_id':'image:unmatched','category':'sign','camera_name':1,'xyxy':[60,10,80,20]}]
    readings={literal.target_object_id:{**clear().model_dump(),'attributes':[word.model_dump()]},
        other_id:{**clear().model_dump(),'attributes':[{'kind':'inscription','value':'NO PARKING'}]}}
    from app.agent.appearance import filter_visual_candidates,sign_inscription_instruction
    assert filter_visual_candidates([literal.target_object_id,other_id,'image:unmatched'],[word],readings)[0]==[literal.target_object_id,other_id,'image:unmatched']
    assert word in descriptive_attributes(literal,readings)
    assert 'reading "BEGIN"' in appearance_phrase('sign',[word])
    assert 'BEGIN' not in sign_inscription_instruction(literal)  # No expected value in observation prompt.
    sentence='The sign reading "BEGIN" in the front view.'
    from app.quality.joint_reference import writer_contract_errors
    literal.task['manual_feedback_policy']='verified-feedback-planning-v1'
    literal_plan={'predicate_ids':['category:sign'],'visual_attributes':[word.model_dump()]}
    assert not writer_contract_errors(literal,literal_plan,sentence)
    assert 'declared_inscription_not_in_text' in writer_contract_errors(literal,literal_plan,'The white sign in the front view.')
    assert 'declared_inscription_not_in_text' in writer_contract_errors(literal,literal_plan,'The sign reading BEGINNING in the front view.')
    checks=[]
    for seed,phase in [(23,'scene_ground'),(24,'scene_ground_permuted')]:
        wire,ids,proof=scene_grounding.messages(literal,sentence,seed)
        selected_id=next(cid for cid,oid in ids.items() if oid==literal.target_object_id)
        good=scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[selected_id],
            ambiguous_candidate_ids=[],unsupported_claims=[],reason='Only this separately labeled panel reads the literal words.')
        result=scene_grounding.verify(literal,good,ids,proof,generated={'expression':sentence,'predicate_ids':['category:sign']})
        assert result['diagnostic_pass'], result['errors']
        checks.append({**result,'blind_round':phase})
        wrong_id=next(cid for cid,oid in ids.items() if oid==other_id)
        assert not scene_grounding.verify(literal,good.model_copy(update={'candidate_ids':[wrong_id]}),ids,proof,
            generated={'expression':sentence,'predicate_ids':['category:sign']})['diagnostic_pass']
    assert scene_grounding.two_rounds_passed(checks) and not scene_grounding.two_rounds_passed(checks[:1])
    swapped=literal.model_copy(deep=True);swapped.target_object_id=other_id;swapped.target=swapped.candidates[1]
    assert scene_grounding.messages(literal,sentence,23)[0]==scene_grounding.messages(swapped,sentence,23)[0]
    # #3161: rejecting an actually nonmatching object is not itself ambiguity.
    # Guide both fresh readers, but never reinterpret a failed answer's prose
    # as a successful result or discard genuinely unknown competitors.
    literal.task['scene_candidate_logic_policy']=scene_grounding.CANDIDATE_LOGIC_POLICY
    instructions,labels,receipt=scene_grounding.messages(literal,sentence,777)
    assert 'No rule requires every object in the category' in instructions[0]['content']
    camera_scoped=literal.model_copy(deep=True)
    camera_scoped.task['scene_camera_scope_policy']='named-camera-scope-v1'
    scope_messages,scope_ids,scope_receipt=scene_grounding.messages(camera_scoped,sentence,777)
    assert 'are CAMERA NAMES' in scope_messages[0]['content']
    assert scope_ids==labels and scope_receipt['scene_camera_scope_policy']=='named-camera-scope-v1'
    old_camera=literal.model_copy(deep=True);old_camera.task.pop('scene_camera_scope_policy',None)
    assert 'are CAMERA NAMES' not in scene_grounding.messages(old_camera,sentence,777)[0][0]['content']
    from app.agent.harness import claim_payload,literal_parse_instruction
    literal_parser=literal.model_copy(deep=True)
    literal_parser.task['blind_parse_scope_policy']='literal-text-parse-v1'
    parse_payload,parse_mapping=claim_payload(literal_parser,sentence,777)
    assert 'parsed_visual_attributes' in parse_payload['output_schema']['required']
    assert 'independent of predicate_vocabulary' in parse_payload['parse_channels']['visual_attributes']
    assert 'TEXT PARSING, not visual truth verification' in literal_parse_instruction(literal_parser)
    legacy_parser=literal_parser.model_copy(deep=True);legacy_parser.task.pop('blind_parse_scope_policy')
    assert literal_parse_instruction(legacy_parser)==''
    assert 'parse_channels' not in claim_payload(legacy_parser,sentence,777)[0]
    old_logic=literal.model_copy(deep=True);old_logic.task.pop('scene_candidate_logic_policy')
    assert 'No rule requires every object in the category' not in scene_grounding.messages(old_logic,sentence,777)[0][0]['content']
    ids_by_native={oid:cid for cid,oid in labels.items()}
    contradictory=scene_grounding.SceneGroundResult(expression_supported=False,candidate_ids=[],
        ambiguous_candidate_ids=[ids_by_native[literal.target_object_id],ids_by_native[other_id]],
        unsupported_claims=['BEGIN for the other sign'],
        reason='One sign clearly reads BEGIN and the other clearly reads NO PARKING, but I reject.')
    assert not scene_grounding.verify(literal,contradictory,labels,receipt)['diagnostic_pass']
    from app.agent.appearance import (merge_observations, resolve_target_observation, filter_visual_candidates,
        source_attribute_conflicts, precheck_observation_batches)
    from app.quality.relations import vertical_rank_catalog
    from app.agent.expression_rendering import render_verified_plan
    post = VisualObservation(target_visible=True,visibility='partial',visibility_reason='Sign back mounted on a post.',
        attributes=[VisualAttribute(kind='subtype',value='a-frame sign'),VisualAttribute(kind='shape',value='rectangular')])
    anonymous_post = {**post.model_dump(),'attributes':[]}
    resolved_post = resolve_target_observation(post,anonymous_post,allow_partial=True)
    assert resolved_post.target_visible and resolved_post.attributes == [VisualAttribute(kind='shape',value='rectangular')]
    supported_stand = {**anonymous_post,'attributes':[{'kind':'subtype','value':'a-frame sign'}]}
    assert VisualAttribute(kind='subtype',value='a-frame sign') in resolve_target_observation(
        post,supported_stand,allow_partial=True).attributes
    # A salient striped barricade must survive perception -> planning -> prose
    # -> both blind checks. Missing primary attributes require a fresh RGB
    # check; a generic folding stand cannot exclude a barricade by subtype.
    from app.agent.appearance import source_attribute_gaps
    from app.schemas.contracts import SceneVisualObservation
    import app.agent.agent_runtime as runtime
    barricade = strict_packet(tmp_path/'barricade-evidence', monkeypatch, ambiguous=True)
    barricade.task.update(annotation_search='evidence-guided-v1', visibility_policy=RECOGNIZABLE_POLICY,
        reference_policy='minimal-sufficient-grounding-v1', reference_preference_policy='scene-first-v1')
    barricade.candidates = barricade.candidates[:2]
    for candidate in barricade.candidates:
        candidate.category = 'sign'
    barricade.target = barricade.candidates[0]
    barricade.predicates = predicate_catalog(barricade.candidates)
    barricade.provenance['scene_image_candidates'] = [
        {'object_id':o.object_id,'category':'sign','camera_name':1,
         'camera_object_id':barricade.source_camera_object_id if i==0 else 'rival-sign',
         'xyxy':[10,10,90,90]} for i,o in enumerate(barricade.candidates)]
    barricade.provenance['source_image_candidates'] = barricade.provenance['scene_image_candidates']
    barricade.provenance['appearance_candidates'] = [
        {'object_id':o.object_id,'candidate_id':f'v{i}','category':'sign','camera_name':1,
         'artifact':'appearance.jpg'} for i,o in enumerate(barricade.candidates)]
    core = VisualAttribute(kind='subtype',value='construction barricade')
    from app.agent.appearance import sign_detail_instruction,sign_detail_recheck_allowed,SIGN_DETAIL_POLICY
    detail_packet=barricade.model_copy(deep=True)
    detail_packet.task['sign_detail_policy']=SIGN_DETAIL_POLICY
    assert 'one-way sign' in sign_detail_instruction(detail_packet)
    sign_type=VisualAttribute(kind='subtype',value='one-way sign')
    assert sign_type.model_dump()=={'kind':'subtype','value':'one-way sign'}
    feedback={'errors':['scene_grounding_unsupported_claims','scene_grounding_not_unique_target'],
        'unsupported_claims':['red and white stripes']}
    assert sign_detail_recheck_allowed(detail_packet,feedback,remaining_calls=5,attempts=1,max_repairs=2,used=False)
    assert not sign_detail_recheck_allowed(detail_packet,feedback,remaining_calls=4,attempts=1,max_repairs=2,used=False)
    assert not sign_detail_recheck_allowed(detail_packet,feedback,remaining_calls=5,attempts=1,max_repairs=2,used=True)
    assert not sign_detail_recheck_allowed(detail_packet,{**feedback,'errors':['joint_reference_pair_unverified']},remaining_calls=5,attempts=1,max_repairs=2,used=False)
    from app.agent.appearance import sign_disambiguation_manifest,SIGN_AMBIGUITY_POLICY
    from app.quality.joint_reference import feedback_distinctiveness_requested,feedback_acceptance
    detail_packet.task['sign_ambiguity_policy']=SIGN_AMBIGUITY_POLICY
    ambiguous_feedback={'errors':['scene_grounding_not_unique_target'],'unsupported_claims':[]}
    assert sign_detail_recheck_allowed(detail_packet,ambiguous_feedback,remaining_calls=5,attempts=1,max_repairs=2,used=False)
    legacy_detail=detail_packet.model_copy(deep=True);legacy_detail.task.pop('sign_ambiguity_policy')
    assert not sign_detail_recheck_allowed(legacy_detail,ambiguous_feedback,remaining_calls=5,attempts=1,max_repairs=2,used=False)
    detail_packet.task['manual_review_feedback']={'reason':'缺少区分线索，场景里有类似物体'}
    assert feedback_distinctiveness_requested(detail_packet.task)
    assert feedback_acceptance(detail_packet.task)['requirements'][0]['kind']=='distinctiveness'
    chinese_feedback={**detail_packet.task,'chinese_landmark_policy':'literal-chinese-landmarks-v1',
        'feedback_acceptance_policy':'feedback-acceptance-v5-natural-concision',
        'feedback_relation_policy':'bilingual-landmark-feedback-v3-spatial-details',
        'manual_review_feedback':{'reason':'你可以说 A silver SUV 在一个黑色车的后面'}}
    from app.quality.joint_reference import feedback_relations
    request=feedback_relations(chinese_feedback)
    assert len(request)==1 and request[0]['relation']=='vehicle_behind'
    assert request[0]['attributes']==[{'kind':'color','value':'black'}] and not request[0]['required']
    old_chinese={**chinese_feedback};old_chinese.pop('chinese_landmark_policy')
    assert not feedback_relations(old_chinese)
    from app.quality.joint_reference import fallback_sentence,local_wording_retry
    inscription_packet=detail_packet.model_copy(deep=True)
    inscription_packet.task.update(blind_parse_scope_policy='literal-text-parse-v1',
        distinctiveness_search_policy='distinctiveness-evidence-search-v1')
    inscription_plan={'predicate_ids':['category:sign','image_visibility:1:sign'],'visual_attributes':[
        {'kind':'color','value':'black'},{'kind':'subtype','value':'one-way sign'},
        {'kind':'inscription','value':'ONE WAY'}]}
    proposal=fallback_sentence(inscription_packet,inscription_plan,{})
    assert 'black one-way sign bearing the inscription "ONE WAY"' in proposal
    assert local_wording_retry(inscription_packet,{'ground_checks':[],
        'verification':{'errors':['manual_review_rejected_text_repeated']}})
    assert not local_wording_retry(inscription_packet,{'ground_checks':[{'diagnostic_pass':False}],
        'verification':{'errors':['manual_review_rejected_text_repeated']}})
    corrected={**chinese_feedback,'manual_review_feedback':{'reason':'不是在一个黑色车的后面，而是在一辆白色轿车的前面'}}
    requests=feedback_relations(corrected)
    assert [r['denied'] for r in requests]==[True,False]
    assert requests[1]['attributes']==[{'kind':'color','value':'white'},{'kind':'subtype','value':'sedan'}]
    manifest=sign_disambiguation_manifest(detail_packet,[o.object_id for o in detail_packet.candidates])
    assert {m['object_id'] for m in manifest}=={o.object_id for o in detail_packet.candidates}
    assert all(m['camera_name']==1 and 'inscription' in m['attribute_recheck'] for m in manifest)
    wrong_camera=detail_packet.model_copy(deep=True)
    wrong_camera.provenance['appearance_candidates'][1]['camera_name']=2
    assert len(sign_disambiguation_manifest(wrong_camera,[o.object_id for o in detail_packet.candidates]))==1
    # Common ambiguity feedback must trigger fresh acquisition on vehicles too.
    from app.agent.appearance import distinctiveness_manifest
    vehicle_detail=detail_packet.model_copy(deep=True)
    vehicle_detail.task['distinctiveness_search_policy']='distinctiveness-evidence-search-v1'
    for wording in ('缺少区分特征','区分特征不够','区分线索不足'):
        vehicle_detail.task['manual_review_feedback']={'reason':wording}
        assert feedback_distinctiveness_requested(vehicle_detail.task)
        assert feedback_acceptance(vehicle_detail.task)['requirements'][0]['status']=='pending'
    legacy_words=vehicle_detail.model_copy(deep=True)
    legacy_words.task.pop('distinctiveness_search_policy')
    assert not feedback_distinctiveness_requested(legacy_words.task)
    for unit in vehicle_detail.candidates:unit.category='vehicle'
    vehicle_detail.target=vehicle_detail.candidates[0]
    for item in vehicle_detail.provenance['appearance_candidates']:item['category']='vehicle'
    vehicle_manifest=distinctiveness_manifest(vehicle_detail)
    assert len(vehicle_manifest)==2
    assert all(m['camera_name']==1 and m['attribute_recheck']==['color','subtype','feature'] for m in vehicle_manifest)
    assert all('expected' not in m and 'reason' not in m for m in vehicle_manifest)
    no_link=vehicle_detail.model_copy(deep=True)
    no_link.provenance['appearance_candidates'][1]['camera_name']=2
    assert len(distinctiveness_manifest(no_link))==1
    from app.agent.appearance import vehicle_detail_instruction,filter_visual_candidates
    from app.quality.description_search import appearance_phrase
    old_vehicle=vehicle_detail.model_copy(deep=True);old_vehicle.task.pop('vehicle_detail_policy',None)
    assert vehicle_detail_instruction(old_vehicle)==''
    vehicle_detail.task['vehicle_detail_policy']='visible-vehicle-parts-v1'
    assert 'no detail in this list is expected' in vehicle_detail_instruction(vehicle_detail)
    for detail in ('rear spoiler','round taillight lenses','split grille'):
        part=VisualAttribute(kind='feature',value=detail)
        phrase=appearance_phrase('vehicle',[part])
        assert detail in phrase and 'a round taillight lenses' not in phrase
        remaining,_=filter_visual_candidates(['target','unknown'],[part],{
            'target':{'target_visible':True,'attributes':[part.model_dump()]}})
        assert remaining==['target','unknown']  # Missing detail never excludes a rival.
    stripes = VisualAttribute(kind='feature',value='orange and white stripes')
    initial = VisualObservation(target_visible=True,visibility='clear',
        visibility_reason='Portable barricade with orange and white striped panel.',attributes=[core])
    detailed = initial.model_copy(update={'attributes':[core,stripes]})
    records = {barricade.target_object_id:{**detailed.model_dump(),'observed_camera_name':1}}
    assert source_attribute_gaps(barricade,initial,records)==[stripes.model_dump()]
    assert not source_attribute_gaps(barricade,detailed,records)
    assert not source_attribute_gaps(barricade,initial,{barricade.target_object_id:{
        **records[barricade.target_object_id],'observed_camera_name':2}})
    assert not source_attribute_gaps(barricade,initial,{barricade.target_object_id:{
        **records[barricade.target_object_id],'target_visible':False,'visibility':'uncertain'}})
    assert resolve_target_observation(initial,records[barricade.target_object_id],allow_partial=True).attributes==[core]
    assert filter_visual_candidates(['rival'],[core],{'rival':{
        'target_visible':True,'attributes':[{'kind':'subtype','value':'a-frame sign'}]}})[0]==['rival']
    assert filter_visual_candidates(['unknown'],[stripes],{})[0]==['unknown']
    from app.agent.appearance import useful_landmark_ids
    schedule=barricade.model_copy(deep=True)
    extra=[obj(f'landmark-{i}',20+i*10,-5,category) for i,category in
        enumerate(('vehicle','vehicle','pedestrian','pedestrian'))]
    schedule.candidates.extend(extra)
    schedule.predicates=predicate_catalog(schedule.candidates)
    for i,landmark in enumerate(extra):
        schedule.provenance['scene_image_candidates'].append({'object_id':landmark.object_id,
            'category':landmark.category,'camera_name':1,'xyxy':[10+i*15,10,30+i*15,50]})
        schedule.provenance['appearance_candidates'].append({'object_id':landmark.object_id,
            'candidate_id':f'extra{i}','category':landmark.category,'camera_name':1,'artifact':'appearance.jpg'})
    assert useful_landmark_ids(schedule,{o.object_id for o in barricade.candidates})=={o.object_id for o in extra}

    class BarricadeCalls(LocalCalls):
        def __init__(self):
            super().__init__();self.stages=[];self.ids=[]
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1;self.stages.append(folder.name.split('-',1)[-1])
            if schema is VisualObservation:
                return initial
            if schema is SceneVisualObservation:
                ids=json.loads(messages[-1]['content'][0]['text'])['candidate_ids']
                return SceneVisualObservation.model_validate({'candidates':[
                    {**(detailed.model_dump() if cid=='v0' else initial.model_copy(update={
                        'attributes':[VisualAttribute(kind='subtype',value='a-frame sign')]}).model_dump()),
                     'candidate_id':cid,'absent_features':[] if cid=='v0' else [stripes.value]}
                    for cid in ids]})
            if schema is GeneratedExpression:
                payload=json.loads(messages[-1]['content'][0]['text'])
                self.ids=payload['prechecked_unique_cue_plan']['predicate_ids']
                return GeneratedExpression(expression='The construction barricade with orange and white stripes ahead and to the left of the ego vehicle.',
                    predicate_ids=self.ids,visual_attributes=[core,stripes])
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,
                parsed_visual_attributes=[core,stripes],unsupported_claims=[])
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'barricade-evidence')
        calls=BarricadeCalls()
        recovered=runtime.run_agent(barricade,calls,0,tmp_path/'barricade-runs',6)
    assert recovered['diagnostic_pass'] and recovered['model_calls_used']==calls.count==6
    assert calls.stages.count('recheck_source_attributes')==1
    assert len(recovered['attempts'][0]['ground_checks'])==2
    # A request for discriminating cues acquires new target/rival pixels
    # before writing, instead of merely changing a shared blue-sign template.
    ambiguity_root=tmp_path/'sign-ambiguity-runs'
    detail_packet.task['feedback_acceptance_policy']='feedback-acceptance-v5-natural-concision'
    class SignAmbiguityCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            assert schema is SceneVisualObservation
            payload=json.loads(messages[-1]['content'][0]['text'])
            assert set(payload['candidate_ids'])=={'v0','v1'}
            assert all('inscription' in row['kinds'] for row in payload['attribute_checks'])
            assert 'manual_review_feedback' not in payload and 'previous_observation' not in payload
            self.count+=1
            RunStore(ambiguity_root/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
            return SceneVisualObservation.model_validate({'candidates':[{
                'candidate_id':cid,'target_visible':True,'visibility':'clear','visibility_reason':'Readable panel.',
                'attributes':[{'kind':'inscription','value':'DROP OFF POINT' if cid=='v0' else 'POLICE'}]}
                for cid in payload['candidate_ids']]})
    ambiguity_calls=SignAmbiguityCalls()
    shared=VisualObservation(target_visible=True,visibility='clear',visibility_reason='Blue sign.',
        attributes=[VisualAttribute(kind='color',value='blue')])
    ambiguity_state=runtime.initial_run(detail_packet,ambiguity_calls,1,12,12)
    ambiguity_state.update(phase='precheck_decision',target_precheck_observation=shared.model_dump(),
        candidate_appearance={o.object_id:{**shared.model_dump(),'observed_camera_name':1} for o in detail_packet.candidates})
    RunStore(ambiguity_root/'runs.sqlite3').create(ambiguity_state)
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'barricade-evidence')
        ambiguity_result=runtime.run_agent(detail_packet,ambiguity_calls,1,ambiguity_root,12,12,resume_run_id=ambiguity_state['run_id'])
    assert ambiguity_result['status']=='paused' and ambiguity_result['model_calls_used']==ambiguity_calls.count==1
    assert ambiguity_result['feedback_target_rechecked'] and ambiguity_result['scene_sign_detail_rechecked']
    assert ambiguity_result['target_precheck_observation']['attributes']==[{'kind':'inscription','value':'DROP OFF POINT'}]
    assert ambiguity_result['source_attribute_rechecks'][0]['previous_observation']==shared.model_dump()
    # A real blind color rejection requests fresh pixels ONCE, then creates a
    # new fully checked draft. Never reinterpret the original denial as a pass.
    semantic=barricade.model_copy(deep=True)
    semantic.task['sign_detail_policy']=SIGN_DETAIL_POLICY
    semantic.task['scene_grounding_policy']=scene_grounding.POLICY
    red_stripes=VisualAttribute(kind='feature',value='red and white stripes')
    red_observation=initial.model_copy(update={'attributes':[core,red_stripes]})
    class StripeConflictCalls(BarricadeCalls):
        scene_reads=0
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is scene_grounding.SceneGroundResult:
                self.count+=1;self.stages.append('scene_ground')
                self.scene_reads+=1
                if self.scene_reads==1:
                    return scene_grounding.SceneGroundResult(expression_supported=False,candidate_ids=[],
                        ambiguous_candidate_ids=[],unsupported_claims=['red and white stripes'],
                        reason='The pattern is visible, but the specific red claim is not confirmed.')
                payload=json.loads(messages[-1]['content'][0]['text'])
                cid=next(row['candidate_id'] for row in payload['candidates'] if row.get('center_m')==list(semantic.target.center_m))
                return scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[cid],
                    ambiguous_candidate_ids=[],unsupported_claims=[],reason='The freshly observed barricade and stripe colors identify this candidate.')
            if schema is GeneratedExpression:
                self.count+=1;self.stages.append('generate');self.ids=['category:sign']
                return GeneratedExpression(expression='The construction barricade with orange and white stripes in the front view.',
                    predicate_ids=self.ids,visual_attributes=[core,stripes])
            return super().call(messages,schema,folder,heartbeat)
    semantic_calls=StripeConflictCalls()
    seed=initial_run(semantic,semantic_calls,2,12,12)
    red_draft=GeneratedExpression(expression='The sign with red and white stripes in the front view.',
        predicate_ids=['category:sign'],visual_attributes=[red_stripes])
    seed.update(phase='scene_ground',model_calls_used=3,admission={'passed':True},
        target_precheck_observation=red_observation.model_dump(),
        candidate_appearance={semantic.target_object_id:red_observation.model_dump(),semantic.candidates[1].object_id:{
            **initial.model_dump(),'attributes':[VisualAttribute(kind='subtype',value='a-frame sign').model_dump()],
            'absent_features':[stripes.value]}},
        attempts=[{'attempt':0,'generated':red_draft.model_dump(),'appearance_check':{'passed':True},'ground_checks':[]}])
    semantic_root=tmp_path/'stripe-conflict-runs'
    RunStore(semantic_root/'runs.sqlite3').create(seed)
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'barricade-evidence')
        check.setattr(scene_grounding,'EVIDENCE_ROOT',tmp_path/'barricade-evidence')
        checked=runtime.run_agent(semantic,semantic_calls,2,semantic_root,12,12,resume_run_id=seed['run_id'])
    assert checked['diagnostic_pass'] and checked['model_calls_used']==3+semantic_calls.count<=12
    assert semantic_calls.stages.count('recheck_source_attributes')==1
    assert not checked['attempts'][0]['ground_checks'][0]['diagnostic_pass']
    assert len(checked['attempts'][-1]['ground_checks'])==2
    assert 'orange and white stripes' in recovered['final_expression']
    # The same known cue without a rival's explicit negative evidence remains
    # ambiguous; recognizing the target must never delete the other barricade.
    no_absence={oid:{**r,'absent_features':[]} for oid,r in recovered['candidate_appearance'].items()}
    final_packet=EvidencePacket.model_validate(recovered['current_packet'])
    assert not admission_plan(final_packet,detailed,no_absence)['unique_plan_found']
    class SharedStripesCalls(BarricadeCalls):
        def call(self,*args,**kwargs):
            result=super().call(*args,**kwargs)
            if isinstance(result,SceneVisualObservation):
                for candidate in result.candidates:
                    candidate.absent_features=[]
            return result
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'barricade-evidence')
        shared_calls=SharedStripesCalls()
        shared=runtime.run_agent(barricade,shared_calls,0,tmp_path/'shared-stripes-runs',6)
    assert not shared['diagnostic_pass'] and shared['model_calls_used']==shared_calls.count==2
    assert 'recheck_source_attributes' not in shared_calls.stages  # Shared cues cannot solve the ambiguity.
    # Low light need not identify paint. Explicit source-region lamp checks
    # are separate from body recognition; unchecked/unclear/other-camera units
    # must remain, including image-only and zero-point native competitors.
    night=strict_packet(tmp_path/'night-evidence',monkeypatch,ambiguous=True)
    night.candidates=night.candidates[:2]
    night.candidates[1].lidar_points=0
    night.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY,
        reference_policy='minimal-sufficient-grounding-v1',reference_preference_policy='scene-first-v1',
        image_identity_policy='official-projection-v1',image_inventory_policy='native-and-projected-v3-native-rgb',
        source_light_policy='visible-region-v1')
    night.predicates=predicate_catalog(night.candidates)
    light=VisualAttribute(kind='feature',value='illuminated headlights')
    lamp_target=VisualObservation(target_visible=True,visibility='partial',
        visibility_reason='The vehicle silhouette and its own paired front lamps are recognizable.',
        attributes=[light],vehicle_colors={'body':None,'roof':None},visible_lights={
            'headlights':'present','taillights':'uncertain','complete_region_inspected':True,
            'reason':'The selected vehicle has a distinct pair of illuminated front lamps.'})
    light_ids=[night.target_object_id,night.candidates[1].object_id,'unlinked-night-vehicle']
    night.provenance['scene_image_candidates']=[{'object_id':oid,'category':'vehicle',
        'camera_name':1,'camera_object_id':night.source_camera_object_id if i==0 else f'night-{i}',
        'xyxy':[10,10+i*25,90,30+i*25]} for i,oid in enumerate(light_ids)]
    night.provenance['source_image_candidates']=night.provenance['scene_image_candidates']
    night.provenance['appearance_candidates']=[{'object_id':oid,'category':'vehicle',
        'camera_name':1,'candidate_id':f'n{i}','artifact':'appearance.jpg'} for i,oid in enumerate(light_ids)]
    class NightCalls(LocalCalls):
        complete=True
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            if schema is VisualObservation:return lamp_target
            if schema is SceneVisualObservation:
                ids=json.loads(messages[-1]['content'][0]['text'])['candidate_ids']
                return SceneVisualObservation.model_validate({'candidates':[
                    {**(lamp_target.model_dump() if cid=='n0' else {
                        'target_visible':False,'visibility':'uncertain','visibility_reason':'Body paint cannot be recognized.',
                        'attributes':[],'visible_lights':{'headlights':'absent','taillights':'uncertain',
                            'complete_region_inspected':self.complete,
                            'reason':'The full indicated image region is inspectable and contains no visible front lamp cue.'}}),
                     'candidate_id':cid} for cid in ids]})
            if schema is GeneratedExpression:
                payload=json.loads(messages[-1]['content'][0]['text'])
                self.ids=payload['prechecked_unique_cue_plan']['predicate_ids']
                return GeneratedExpression(expression='The vehicle with illuminated headlights in the front view, ahead and to the left of the ego vehicle.',
                    predicate_ids=self.ids,visual_attributes=[light])
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,
                parsed_visual_attributes=[light],unsupported_claims=[])
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'night-evidence')
        lamps=NightCalls();night_result=runtime.run_agent(night,lamps,0,tmp_path/'night-runs',5)
        unclear=NightCalls();unclear.complete=False
        unclear_result=runtime.run_agent(night,unclear,0,tmp_path/'night-unclear-runs',5)
    assert night_result['diagnostic_pass'] and lamps.count==5
    assert len(night_result['attempts'][0]['ground_checks'])==2
    assert not unclear_result['diagnostic_pass'] and unclear.count==2
    lamp_records=night_result['candidate_appearance'];rival_id=light_ids[1]
    assert filter_visual_candidates([rival_id],[light],lamp_records)[0]==[rival_id]  # No unqualified ego proof.
    assert filter_visual_candidates([rival_id],[light],lamp_records,visible_camera=1)[0]==[]
    from app.agent.appearance import effective_observations
    historical=night.model_copy(deep=True)
    historical.task.pop('source_light_policy')
    old_records=effective_observations(historical,lamp_records)
    assert filter_visual_candidates([rival_id],[light],old_records,visible_camera=1)[0]==[rival_id]
    for changed in ({'observed_camera_name':2},{'visibility_conflict':True},
                    {'observed_input_sha256':'changed'}, {'visible_light_region':None},
                    {'visible_lights':{'headlights':'uncertain','complete_region_inspected':True,'reason':'Glare.'}}):
        rival_record={**lamp_records[rival_id],**changed}
        assert filter_visual_candidates([rival_id],[light],{rival_id:rival_record},visible_camera=1)[0]==[rival_id]
    focus=night.model_copy(deep=True)
    focus.provenance['scene_image_candidates'][0]['xyxy']=[80,20,100,80]
    for index in range(20):
        oid=f'background-{index}'
        focus.provenance['scene_image_candidates'].append({'object_id':oid,'category':'vehicle',
            'camera_name':1,'xyxy':[1,1,20,10]})
        focus.provenance['appearance_candidates'].append({'object_id':oid,'candidate_id':f'a{index:02}',
            'category':'vehicle','camera_name':1,'artifact':'appearance.jpg'})
    focus.provenance['scene_image_candidates'].append({'object_id':'edge-unknown','category':'unknown',
        'camera_name':1,'xyxy':[95,1,100,15]})
    focus.provenance['appearance_candidates'].append({'object_id':'edge-unknown','candidate_id':'zz-edge',
        'category':'unknown','camera_name':1,'artifact':'appearance.jpg'})
    focused=precheck_observation_batches(focus)
    assert 'edge-unknown' in {m['object_id'] for m in focused[0]}
    assert sum(map(len,focused))==len(focus.provenance['appearance_candidates'])

    # A landmark's unclear lookalike is not necessarily a target competitor.
    # Acquire that missing comparison before giving up on a spatial sentence.
    relation=strict_packet(tmp_path/'relation-evidence',monkeypatch,ambiguous=True)
    relation.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY,
        reference_policy='minimal-sufficient-grounding-v1',reference_preference_policy='scene-first-v1',
        image_position_policy='verified-boundary-v1')
    relation.candidates=[relation.target,obj('gray-rival',30,5),obj('red-landmark',20,-3,'sign'),
        obj('uncertain-landmark-rival',40,-3,'sign')]
    relation.predicates=predicate_catalog(relation.candidates)
    relation.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':o.category,
        'camera_name':1,'camera_object_id':relation.source_camera_object_id if i==0 else f'rc{i}',
        'xyxy':([0,10,60,90] if i==0 else [20,20,80,90] if i==1 else [60,5,95,20])}
        for i,o in enumerate(relation.candidates)]
    relation.provenance['source_image_candidates']=relation.provenance['scene_image_candidates']
    relation.provenance['appearance_candidates']=[{'object_id':o.object_id,'category':o.category,
        'camera_name':1,'candidate_id':f'r{i}','artifact':'appearance.jpg'} for i,o in enumerate(relation.candidates)]
    last=relation.provenance['appearance_candidates'][-1]
    last['alternatives']=[dict(last)]  # Uncertain old observation warrants a real relook.
    gray=VisualObservation.model_validate({**clear().model_dump(),
        'attributes':[{'kind':'color','value':'gray'}], 'vehicle_colors':{'body':'gray','roof':None}})
    gray_partial=gray.model_copy(update={'visibility':'partial'})
    from app.quality.relations import expand_relations
    initial_records={o.object_id:{**(gray_partial if i==0 else gray).model_dump(), 'observed_camera_name':1}
        for i,o in enumerate(relation.candidates[:2])}
    initial_packet,initial_relations=expand_relations(relation,initial_records,target_observation=gray_partial)
    from app.quality.admission import observation_plan_sufficient
    initial_plan=admission_plan(initial_packet,gray_partial,initial_records,initial_relations['anchors'])
    assert initial_plan['unique_plan_found'] and not observation_plan_sufficient(initial_packet,initial_plan)
    assert not admission_plan(initial_packet,gray_partial,initial_records,initial_relations['anchors'],
        allow_pixel_fallback=False)['unique_plan_found']
    class RelationCalls(LocalCalls):
        def __init__(self):super().__init__();self.stages=[];self.ids=[]
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1;stage=folder.name.split('-',1)[-1];self.stages.append(stage)
            if schema is VisualObservation:return gray_partial
            if schema is SceneVisualObservation:
                ids=json.loads(messages[-1]['content'][0]['text'])['candidate_ids'];result=[]
                for cid in ids:
                    record=(gray_partial if cid=='r0' else gray).model_dump() if cid in {'r0','r1'} else {
                        'target_visible':True,'visibility':'clear','visibility_reason':'Visible colored sign panel.',
                        'attributes':[{'kind':'color','value':'red' if cid=='r2' else 'blue'}]}
                    if cid=='r3' and stage!='relook_candidates':
                        record={'target_visible':False,'visibility':'uncertain','visibility_reason':'Needs closer context.', 'attributes':[]}
                    result.append({**record,'candidate_id':cid})
                return SceneVisualObservation.model_validate({'candidates':result})
            if schema is GeneratedExpression:
                payload=json.loads(messages[-1]['content'][0]['text']);self.ids=payload['prechecked_unique_cue_plan']['predicate_ids']
                assert any(p['phrase']=='behind the red sign' for p in payload['available_target_facts'])
                return GeneratedExpression(expression='The gray vehicle ahead and to the left of the ego vehicle, behind the red sign.',
                    predicate_ids=self.ids,visual_attributes=gray.attributes)
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,parsed_visual_attributes=gray.attributes,unsupported_claims=[])
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'relation-evidence')
        relation_calls=RelationCalls()
        related=runtime.run_agent(relation,relation_calls,0,tmp_path/'relation-runs',6)
    assert related['diagnostic_pass'] and relation_calls.count==6 and relation_calls.stages.count('relook_candidates')==1
    assert len(related['attempts'][0]['ground_checks'])==2
    relook_event=next(e for e in RunStore(tmp_path/'relation-runs/runs.sqlite3').events(related['run_id'])
        if e['kind']=='evidence_relook_planned')
    assert 'uncertain-landmark-rival' in relook_event['payload']['landmark_comparison_object_ids']
    assert relook_event['payload']['replace_pixel_fallback']
    assert not any(id_.startswith(('image_boundary:','image_order:')) for id_ in related['admission']['predicate_ids'])

    # A nearly coincident native pedestrian remains a distinct identity. Both
    # clothing and a shared SUV relation would select the same two raw boxes.
    # Diagnose that conflict before spending calls rereading identical pixels.
    duplicate=strict_packet(tmp_path/'coincident-evidence',monkeypatch,ambiguous=True)
    duplicate.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY,
        reference_policy='minimal-sufficient-grounding-v1',reference_preference_policy='scene-first-v1')
    duplicate.target=duplicate.target.model_copy(update={'category':'pedestrian','size_m':(.8,.8,1.7)})
    rival=duplicate.target.model_copy(update={'object_id':'coincident-native-person',
        'center_m':(10.04,5.02,1.04)})
    duplicate.candidates=[duplicate.target,rival]
    duplicate.predicates=predicate_catalog(duplicate.candidates)
    duplicate.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':o.category,
        'camera_name':1,'camera_object_id':duplicate.source_camera_object_id if i==0 else 'overlap-person',
        'xyxy':[10,10,90,90]} for i,o in enumerate(duplicate.candidates)]
    duplicate.provenance['source_image_candidates']=duplicate.provenance['scene_image_candidates']
    duplicate.provenance['appearance_candidates']=[{'object_id':o.object_id,'category':o.category,
        'camera_name':1,'candidate_id':f'p{i}','artifact':'appearance.jpg'} for i,o in enumerate(duplicate.candidates)]
    person=VisualObservation(target_visible=True,visibility='partial',visibility_reason='Blue jacket and black trousers visible.',
        attributes=[VisualAttribute(kind='upper_clothing_color',value='blue'),
            VisualAttribute(kind='lower_clothing_color',value='black')])
    class CoincidentCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            if schema is VisualObservation:return person
            assert schema is SceneVisualObservation  # No writer or fake successful blind check.
            ids=json.loads(messages[-1]['content'][0]['text'])['candidate_ids']
            return SceneVisualObservation.model_validate({'candidates':[
                {**person.model_dump(),'candidate_id':cid} for cid in ids]})
    with monkeypatch.context() as check:
        check.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'coincident-evidence')
        coincident_calls=CoincidentCalls()
        unresolved=runtime.run_agent(duplicate,coincident_calls,0,tmp_path/'coincident-runs',8)
    assert coincident_calls.count==2 and not unresolved['diagnostic_pass']
    assert unresolved['stop_reason']=='precheck_native_identity_ambiguous'
    assert unresolved['distinguishing_cue_bound']['native_identity_conflicts'][0]['object_id']==rival.object_id
    assert len(unresolved['current_packet']['candidates'])==2 and not unresolved['attempts']
    from app.quality.admission import native_identity_conflicts
    separated=duplicate.model_copy(deep=True)
    separated.candidates[1]=rival.model_copy(update={'center_m':(10,7,0)})
    assert not native_identity_conflicts(separated,[duplicate.target_object_id,rival.object_id])
    assert not native_identity_conflicts(duplicate,[duplicate.target_object_id])
    # New frame selection should escape that overlap when the SAME tracks
    # separate later, even though the later frame is farther and smaller.
    from app.datasets.catalog import context_targets
    from app.datasets.waymo import B,C,O,P,T,V
    frame_rows=[];projections=[]
    for timestamp in (1,600001):
        for i,oid in enumerate(('person-a','person-b')):
            x=(10+i*.04) if timestamp==1 else (20+i*2)
            frame_rows.append({T:timestamp,O:oid,f'{B}.type':2,f'{B}.num_lidar_points_in_box':100,
                f'{B}.box.heading':0,**{f'{B}.box.center.{a}':v for a,v in zip('xyz',(x,5,1))},
                **{f'{B}.box.size.{a}':v for a,v in zip('xyz',(.8,.8,1.7))}})
            projections.append({T:timestamp,O:oid,V:1,f'{P}.type':2,
                f'{P}.box.center.x':50 if timestamp==1 else 30+40*i,f'{P}.box.center.y':50,
                f'{P}.box.size.x':40 if timestamp==1 else 20,f'{P}.box.size.y':60})
    class FrameReader:
        def metadata_rows(self,component,context):
            return {'lidar_box':frame_rows,'projected_lidar_box':projections,
                'vehicle_pose':[{T:t} for t in (1,600001)],
                'camera_calibration':[{V:1,'[CameraCalibrationComponent].width':100,
                    '[CameraCalibrationComponent].height':100}]}[component]
        def rows(self,component,context,**kwargs):
            assert component=='camera_image'
            return [{C:context,T:t,V:1} for t in (1,600001)]
    frame_targets=context_targets(FrameReader(),'separating-pedestrians')
    assert len(frame_targets)==2 and all(t['eligible_frame_count']==2 for t in frame_targets)
    assert all(t['timestamp']==600001 and not t['selection']['near_coincident_native_ids'] for t in frame_targets)
    assert all(t['alternative_selections'][0]['timestamp']==1 for t in frame_targets)
    assert frame_targets[0]['alternative_selections'][0]['near_coincident_native_ids']==['person-b']
    height = strict_packet(tmp_path/'height-proof', monkeypatch, ambiguous=True)
    height.task['annotation_search'] = True
    height.target = height.target.model_copy(update={'center_m':(10,5,4)})
    height.candidates[0] = height.target
    height.predicates = predicate_catalog(height.candidates)
    height.predicates.extend(vertical_rank_catalog(height)[0])
    observations = {o.object_id:{**clear().model_dump(),'observed_camera_name':1} for o in height.candidates}
    plan = admission_plan(height,clear(),observations)
    assert not plan['unique_plan_found']  # Technical center.z ranks are not everyday references.
    assert any(p.predicate_id.startswith('vertical_rank:highest:') for p in height.predicates)
    height.provenance['admission_plan'] = plan
    assert render_verified_plan(height,observations) is None
    from app.quality.admission import distinguishing_cue_bound
    assert not distinguishing_cue_bound(height,clear(),observations)['can_distinguish']
    unknown = height.candidates[1].model_copy(update={'category':'unknown','center_m':(10,5,4),'lidar_points':0})
    blocked = height.model_copy(deep=True)
    blocked.candidates[1] = unknown
    blocked.predicates = predicate_catalog(blocked.candidates)
    blocked.predicates.extend(vertical_rank_catalog(blocked)[0])
    ambiguous = admission_plan(blocked,clear(),observations)
    assert not ambiguous.get('unique_plan_found')  # Unknown/zero-point rivals cannot disappear.

    confirmed = {**partial_observation().model_dump(), 'observed_camera_name':1}
    unknown = {**confirmed, 'target_visible':False, 'visibility':'uncertain', 'attributes':[],
        'visibility_reason':'The tight crop does not establish identity.'}
    hidden = {**unknown, 'visibility':'occluded', 'visibility_reason':'The unit is fully hidden by the foreground vehicle.'}
    upgraded = merge_observations(unknown, confirmed, allow_partial=True)
    assert upgraded['target_visible'] and upgraded['attributes']
    assert merge_observations(upgraded, unknown, allow_partial=True)['attributes'] == upgraded['attributes']
    conflict = merge_observations(confirmed, hidden, allow_partial=True)
    assert not conflict['target_visible'] and conflict['camera_observations']['1']['visibility_conflict']
    assert not merge_observations(conflict, confirmed, allow_partial=True)['target_visible']
    alternate = merge_observations(confirmed, {**hidden,'observed_camera_name':2}, allow_partial=True)
    assert alternate['target_visible'] and alternate['camera_observations']['1']['target_visible']

    broad = {**confirmed, 'attributes':[{'kind':'subtype','value':'truck'}]}
    resolved = resolve_target_observation(partial_observation(), broad, allow_partial=True)
    assert VisualAttribute(kind='subtype',value='pickup truck') in resolved.attributes
    generic_primary = partial_observation().model_copy(update={'attributes':[VisualAttribute(kind='subtype',value='truck')]})
    assert resolve_target_observation(generic_primary, confirmed, allow_partial=True).attributes == generic_primary.attributes
    assert not any(a.kind == 'subtype' for a in resolve_target_observation(partial_observation(),
        {**broad,'attributes':[{'kind':'subtype','value':'suv'}]}, allow_partial=True).attributes)
    for claimed, observed in [('truck','pickup truck'),('pickup truck','truck')]:
        assert filter_visual_candidates(['rival'], [VisualAttribute(kind='subtype',value=claimed)],
            {'rival':{'target_visible':True,'attributes':[{'kind':'subtype','value':observed}]}})[0] == ['rival']
    assert filter_visual_candidates(['rival'], [VisualAttribute(kind='subtype',value='pickup truck')],
        {'rival':{'target_visible':True,'attributes':[{'kind':'subtype','value':'sedan'}]}})[0] == []
    for claimed, observed, expected in [('excavator boom','excavator',['rival']),
            ('excavator','excavator body',['rival']),('excavator boom','excavator body',[])]:
        assert filter_visual_candidates(['rival'],[VisualAttribute(kind='subtype',value=claimed)],
            {'rival':{'target_visible':True,'attributes':[{'kind':'subtype','value':observed}]}})[0] == expected

    p = strict_packet(tmp_path, monkeypatch)
    p.task.update(visibility_policy=RECOGNIZABLE_POLICY, annotation_search='evidence-guided-v1')
    observation = partial_observation()
    p.views[0].projection_touches_boundary = True
    assert target_visibility_error(p, observation) is None
    assert not admission_plan(p, observation, {})['passed']
    other = {p.target_object_id:{**observation.model_dump(), 'observed_camera_name':2}}
    assert target_visibility_error(p, observation, other) == 'precheck_target_identity_unconfirmed'
    other[p.target_object_id]['observed_camera_name'] = 1
    assert admission_plan(p, observation, other)['unique_plan_found']
    wrong = observation.model_copy(update={'attributes':[VisualAttribute(kind='color',value='silver'),
        VisualAttribute(kind='subtype',value='suv')], 'vehicle_colors':None})
    assert source_attribute_conflicts(p, wrong, other) == ['color', 'subtype']
    assert not source_attribute_conflicts(p, wrong, {p.target_object_id:{**other[p.target_object_id],
        'observed_camera_name':2}})
    p.provenance['appearance_candidates'] = [
        {'candidate_id':'target','object_id':p.target_object_id,'category':'vehicle','camera_name':1},
        {'candidate_id':'unused','object_id':'unrelated-pedestrian','category':'pedestrian','camera_name':2}]
    assert all(m['candidate_id'] != 'unused' for batch in precheck_observation_batches(p) for m in batch)
    p.provenance['scene_image_candidates'] = [
        {**p.provenance['source_image_candidates'][0], 'camera_name':1,'xyxy':[1,1,50,50]},
        {'object_id':'image:1:inside','category':'vehicle','camera_name':1,'xyxy':[60,1,90,50]},
        {'object_id':'image:2:outside','category':'vehicle','camera_name':2,'xyxy':[1,1,50,50]}]
    p.provenance['appearance_candidates'] += [
        {'candidate_id':'inside','object_id':'image:1:inside','category':'vehicle','camera_name':1},
        {'candidate_id':'outside','object_id':'image:2:outside','category':'vehicle','camera_name':2}]
    focused = [m['candidate_id'] for batch in precheck_observation_batches(p) for m in batch]
    assert 'inside' in focused and 'outside' not in focused
    assert len(p.provenance['scene_image_candidates']) == 3  # Unobserved rivals are not deleted from proof.
    for record, reason in (
        (unknown, 'precheck_target_identity_unconfirmed'),
        (hidden, 'precheck_source_visibility_conflict'),
        (conflict['camera_observations']['1'], 'precheck_source_visibility_conflict'),
    ):
        assert target_visibility_error(p, observation, {p.target_object_id:record}) == reason
        assert not admission_plan(p, observation, {p.target_object_id:record})['passed']
    p.candidates[1] = p.candidates[1].model_copy(update={'center_m':p.target.center_m})
    p.predicates = predicate_catalog(p.candidates)
    assert not admission_plan(p, observation, other)['unique_plan_found']
    p.source_view = 'SIDE_RIGHT'
    assert not admission_plan(p, observation, other)['passed']
    import app.agent.agent_runtime as runtime
    from app.agent.agent_runtime import initial_run
    from app.schemas.contracts import SceneVisualObservation

    # A model writer gets the verified landmark plan and source image, then
    # undergoes the same appearance check and both blind language checks.
    landmark_root = tmp_path/'landmark'
    landmark = strict_packet(landmark_root/'evidence', monkeypatch)
    landmark.task.update(annotation_search='evidence-guided-v1', visibility_policy=RECOGNIZABLE_POLICY)
    landmark.candidates[1].center_m = (30, 5, 1)
    landmark.candidates[2].center_m = (20, -3, 1)
    landmark.predicates = predicate_catalog(landmark.candidates)
    landmark.provenance['appearance_candidates'] = [
        {'candidate_id':f'v{i}', 'object_id':o.object_id, 'category':o.category,
         'camera_name':1, 'artifact':'appearance.jpg'} for i,o in enumerate(landmark.candidates)]
    landmark.provenance['scene_image_candidates'] = [
        {'object_id':o.object_id, 'category':o.category, 'camera_name':1,
         'camera_object_id':landmark.source_camera_object_id if i == 0 else f'camera-{i}'}
        for i,o in enumerate(landmark.candidates)]
    monkeypatch.setattr(runtime, 'EVIDENCE_ROOT', landmark_root/'evidence')

    class LandmarkCalls(RichCalls):
        def call(self, messages, schema, folder, heartbeat=None):
            if schema is VisualObservation:
                self.count += 1
                return clear().model_copy(update={'attributes':[
                    *clear().attributes, VisualAttribute(kind='subtype', value='sedan')]})
            if schema is SceneVisualObservation:
                self.count += 1
                return SceneVisualObservation.model_validate({'candidates':[
                    {**clear().model_dump(), 'candidate_id':f'v{i}',
                     'attributes':clear().model_dump()['attributes'] if i < 2 else []}
                    for i in range(3)]})
            if schema is GeneratedExpression:
                self.count += 1
                self.generated += 1
                payload = json.loads(messages[-1]['content'][0]['text'])
                assert payload['grounded_writing_brief']['generation_mode'].startswith('model writes')
                assert any(c['type']=='image_url' for c in messages[-1]['content'])
                return GeneratedExpression(expression='The white sedan ahead and to the left of the ego vehicle, behind the sign.',
                    predicate_ids=payload['prechecked_unique_cue_plan']['predicate_ids'],
                    visual_attributes=[VisualAttribute(kind='color',value='white'),VisualAttribute(kind='subtype',value='sedan')])
            result = super().call(messages, schema, folder, heartbeat)
            if schema is GroundResult:
                payload = json.loads(messages[-1]['content'])
                assert 'behind the sign' in payload['expression']
                relation = next(p['predicate_id'] for p in payload['predicate_vocabulary']
                    if p['phrase'] == 'behind the sign')
                result.parsed_predicate_ids.append(relation)
            return result

    landmark_calls = LandmarkCalls()
    schedules = []
    with monkeypatch.context() as schedule:
        def counted_batches(current):
            schedules.append((current.packet_id, current.source_view))
            return precheck_observation_batches(current)
        schedule.setattr('app.agent.appearance.precheck_observation_batches', counted_batches)
        rendered = runtime.run_agent(landmark, landmark_calls, 0, landmark_root/'runs', 5)
    assert schedules == [(landmark.packet_id, landmark.source_view)]
    assert rendered['diagnostic_pass'] and landmark_calls.count == rendered['model_calls_used'] == 5
    assert landmark_calls.generated == 1
    assert rendered['attempts'][0]['generation_source'].startswith('model')
    assert len(rendered['attempts'][0]['ground_checks']) == 2
    from app.agent.expression_rendering import render_verified_plan
    saved_packet = EvidencePacket.model_validate(rendered['current_packet'])
    local_text = render_verified_plan(saved_packet, rendered['candidate_appearance'])
    assert local_text == render_verified_plan(saved_packet, rendered['candidate_appearance'])
    # Every surface form retains the exact cue/attribute contract and the
    # registered landmark. Rendering adds no model calls or mutable RNG state.
    import app.agent.expression_rendering as rendering
    from app.agent.harness import expression_policy_errors, generation_messages
    variants = set()
    with monkeypatch.context() as style_patch:
        for seed in range(8):
            style_patch.setattr(rendering, '_style_seed', lambda p, value=seed: value)
            variant = render_verified_plan(saved_packet, rendered['candidate_appearance'])
            assert variant.predicate_ids == local_text.predicate_ids
            assert variant.visual_attributes == local_text.visual_attributes
            assert 'behind the sign' in variant.expression and 'the ego vehicle' in variant.expression
            assert 'ahead and to the left of the ego vehicle' in variant.expression.casefold()
            assert not expression_policy_errors(variant.expression)
            variants.add(variant.expression)
        style_patch.setattr(rendering, '_style_seed', lambda p: 3)
        messages = generation_messages(saved_packet, None, rendered['candidate_appearance'])
        payload = json.loads(messages[-1]['content'][0]['text'])
        assert payload['expression_style'] == rendering.generation_style(saved_packet)
        assert 'registered image relation' in payload['expression_style']
    assert len(variants) >= 3
    assert local_text == render_verified_plan(saved_packet, rendered['candidate_appearance'])
    # The new policy must still acquire a real landmark and complete both blind
    # checks when two same-looking vehicles share the same ego-relative region.
    from app.quality.description_search import SCENE_REFERENCE_POLICY
    scene_landmark = landmark.model_copy(deep=True)
    scene_landmark.task['reference_policy'] = SCENE_REFERENCE_POLICY
    for index, box in enumerate(scene_landmark.provenance['scene_image_candidates']):
        box['xyxy'] = [index*30, 1, index*30+25, 50]
    scene_calls = LandmarkCalls()
    scene_result = runtime.run_agent(scene_landmark, scene_calls, 0, landmark_root/'scene-runs', 5)
    assert scene_result['diagnostic_pass'] and scene_calls.count == 5
    assert 'behind the sign' in scene_result['final_expression']
    assert len(scene_result['attempts'][0]['ground_checks']) == 2
    assert not any(pid.startswith(('image_order:', 'image_visibility:')) for pid in
        scene_result['attempts'][0]['generated']['predicate_ids'])
    unobserved = {oid:r for oid,r in rendered['candidate_appearance'].items()
        if oid != landmark.candidates[2].object_id}
    assert render_verified_plan(saved_packet, unobserved) is None
    # A recognizable color/type and helmet can name a landmark, without
    # importing another camera's pixels or treating an unknown feature as absent.
    from app.quality.relations import relation_catalog
    from app.quality.description_search import confirmed_attributes, appearance_phrase
    salient = saved_packet.model_copy(deep=True)
    salient.task['reference_policy'] = SCENE_REFERENCE_POLICY
    records = {oid:dict(record) for oid,record in rendered['candidate_appearance'].items()}
    red_id = salient.candidates[1].object_id
    records[red_id] = {**records[red_id], 'vehicle_colors':None, 'attributes':[
        {'kind':'color','value':'red'},{'kind':'subtype','value':'sedan'}]}
    _, anchors = relation_catalog(salient,records)
    red_anchor = next(a for a in anchors if a['object_id']==red_id)
    assert red_anchor['phrase']=='the red sedan'
    assert {tuple(a.values()) for a in red_anchor['descriptor_visual_attributes']} == {('color','red'),('subtype','sedan')}
    elsewhere = {**records,red_id:{**records[red_id],'observed_camera_name':2}}
    assert all(a['object_id']!=red_id for a in relation_catalog(salient,elsewhere)[1])
    helmet_person = obj('helmet-person',20,-3,'pedestrian')
    salient.candidates[2] = helmet_person
    salient.predicates = predicate_catalog(salient.candidates)
    helmet = VisualAttribute(kind='helmet_color',value='black')
    records[helmet_person.object_id] = {**clear().model_dump(),'observed_camera_name':1,
        'attributes':[helmet.model_dump()]}
    helmet_anchor = next(a for a in relation_catalog(salient,records)[1] if a['object_id']==helmet_person.object_id)
    assert 'wearing a black helmet' in helmet_anchor['phrase']
    assert confirmed_attributes('vehicle',records[helmet_person.object_id])==[]
    assert appearance_phrase('pedestrian',[helmet])=='the pedestrian wearing a black helmet'
    rivals = {'black':records[helmet_person.object_id],
        'unknown':{**clear().model_dump(),'attributes':[]},
        'white':{**clear().model_dump(),'attributes':[{'kind':'helmet_color','value':'white'}]}}
    assert filter_visual_candidates(list(rivals),[helmet],rivals)[0]==['black','unknown']
    # A true relation is not necessarily a unique reference. Unknown appearance
    # on an overlapping rival must survive every suggested evidence bundle.
    from app.quality.admission import reference_options
    from app.quality.relations import expand_relations
    ambiguous_landmark = landmark.model_copy(deep=True)
    ambiguous_landmark.candidates[1].center_m = ambiguous_landmark.target.center_m
    ambiguous_landmark.predicates = predicate_catalog(ambiguous_landmark.candidates)
    ambiguous_landmark.provenance['admission_plan'] = {'passed':True,'unique_plan_found':False}
    ambiguous_records = {oid:{**record, 'attributes':[],'vehicle_colors':None}
        for oid,record in rendered['candidate_appearance'].items()}
    ambiguous_landmark,_ = expand_relations(ambiguous_landmark,ambiguous_records)
    assert not reference_options(ambiguous_landmark,ambiguous_records,['category:vehicle'])
    # The same evidence brief exposes ordinary landmark relations, never ranks.
    messages = generation_messages(salient,None,records)
    brief = json.loads(messages[-1]['content'][0]['text'])
    assert all(not f['predicate_id'].startswith('vertical_rank:') and not f['predicate_id'].endswith(
        ('bottom_edge_visible','top_edge_visible','height_visible')) for f in brief['available_target_facts'])
    saved_packet.task['manual_review_feedback'] = {'previous_expression':rendered['final_expression']}
    assert render_verified_plan(saved_packet, rendered['candidate_appearance']) is None

    # A color conflict interrupts batch 1 of 2. Its focused recheck cannot
    # replace the untouched second batch; resume its durable index, even after
    # pause/restart, and retain enough calls for writing and both blind parses.
    resumed_packet = strict_packet(tmp_path/'resume-observations-evidence',monkeypatch)
    resumed_packet.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY)
    resumed_packet.candidates=[resumed_packet.target,*[
        resumed_packet.target.model_copy(update={'object_id':f'resume-rival-{i}'}) for i in range(17)]]
    resumed_packet.predicates=predicate_catalog(resumed_packet.candidates)
    resumed_packet.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':'vehicle',
        'camera_name':1,'xyxy':[10,10,90,90],'identity_source':'official_camera_lidar_association'}
        for o in resumed_packet.candidates]
    resumed_packet.provenance['appearance_candidates']=[{'candidate_id':f'v{i:03d}','object_id':o.object_id,
        'camera_name':1,'category':'vehicle','artifact':'appearance.jpg'} for i,o in enumerate(resumed_packet.candidates)]
    from app.agent.appearance import precheck_observation_batches
    assert [len(b) for b in precheck_observation_batches(resumed_packet)]==[16,2]
    monkeypatch.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'resume-observations-evidence')
    resumed_root=tmp_path/'resume-observations-runs'
    class InterruptedObservationCalls(RichCalls):
        def __init__(self):
            super().__init__(); self.scene_inputs=[]
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is SceneVisualObservation:
                import re
                self.count+=1
                ids=[re.match(r'Candidate (v\d+);',part.get('text','')).group(1)
                    for part in messages[-1]['content'] if re.match(r'Candidate (v\d+);',part.get('text',''))]
                self.scene_inputs.append((folder.name,ids))
                recheck='recheck_source_attributes' in folder.name
                if recheck:
                    RunStore(resumed_root/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
                return SceneVisualObservation.model_validate({'candidates':[
                    {**clear().model_dump(),'candidate_id':cid,'attributes':
                        [{'kind':'color','value':'white'},{'kind':'subtype','value':'sedan'}] if cid=='v000' else
                        [{'kind':'color','value':'red'}]} for cid in ids]})
            result=super().call(messages,schema,folder,heartbeat)
            if schema is VisualObservation:
                result=clear()
                result.attributes=[VisualAttribute(kind='color',value='silver'),VisualAttribute(kind='subtype',value='sedan')]
            return result
    interrupted_calls=InterruptedObservationCalls()
    paused=runtime.run_agent(resumed_packet,interrupted_calls,0,resumed_root,8)
    assert paused['status']=='paused' and paused['precheck_batch_index']==1 and interrupted_calls.count==3
    RunStore(resumed_root/'runs.sqlite3').request_control(paused['run_id'],'resume')
    resumed=runtime.run_agent(resumed_packet,interrupted_calls,0,resumed_root,8,resume_run_id=paused['run_id'])
    assert resumed['diagnostic_pass'] and interrupted_calls.count==7
    assert interrupted_calls.scene_inputs[-1][1]==['v016','v017']
    assert resumed['precheck_batch_index']==2 and len(resumed['attempts'][0]['ground_checks'])==2
    assert any(e['kind']=='precheck_observation_plan_resumed' for e in RunStore(resumed_root/'runs.sqlite3').events(paused['run_id']))

    cases = [(True,4,True,4,'confirmed'), (False,4,True,4,'confirmed'), (False,5,True,5,'confirmed'),
        (True,5,True,5,'recover'), (True,5,False,3,'unknown'), (True,5,False,2,'hidden'),
        (True,5,True,5,'attributes'), (True,4,True,4,'clear_budget'),
        (True,4,True,4,'clear_needs_scene'), (True,4,True,4,'excavator')]
    for index, (image_plan, budget, passed, used, identity) in enumerate(cases):
        case = tmp_path/str(index)
        p = strict_packet(case/'evidence', monkeypatch, ambiguous=image_plan)
        p.task.update(visibility_policy=RECOGNIZABLE_POLICY, annotation_search='evidence-guided-v1')
        p.provenance['appearance_candidates'] = [{'candidate_id':'v000','object_id':p.target_object_id,
            'camera_name':1,'camera_object_id':p.source_camera_object_id,'category':'vehicle','artifact':'appearance.jpg'}]
        if image_plan:
            p.provenance['scene_image_candidates'] = [
                {**p.provenance['source_image_candidates'][0], 'camera_name':1, 'xyxy':[1,1,50,50]},
                {'object_id':p.candidates[1].object_id,'category':'vehicle','camera_name':2,'xyxy':[1,1,50,50]}]
        if identity == 'clear_needs_scene':
            p.provenance['scene_image_candidates'][1].update(camera_name=1,xyxy=[20,1,60,50])
            p.provenance['appearance_candidates'].append({'candidate_id':'v001',
                'object_id':p.candidates[1].object_id,'camera_name':1,'category':'vehicle','artifact':'appearance.jpg'})
        monkeypatch.setattr(runtime, 'EVIDENCE_ROOT', case/'evidence')
        attrs = partial_observation().attributes if image_plan else [
            VisualAttribute(kind='color',value='white'), VisualAttribute(kind='subtype',value='truck')]
        ids = ['category:vehicle','longitudinal:front','lateral:left']
        if image_plan:
            ids.append('image_visibility:1:vehicle')

        def case_observation():
            observed = partial_observation()
            if identity == 'excavator':
                observed = observed.model_copy(update={'attributes':[
                    VisualAttribute(kind='color', value='yellow'), VisualAttribute(kind='subtype', value='excavator')],
                    'vehicle_colors':None})
            return observed

        class Calls(LocalCalls):
            crash_primary = image_plan
            scene_calls = 0
            primary_calls = 0

            def call(self, messages, schema, folder, heartbeat=None):
                self.count += 1
                if schema is VisualObservation:
                    result = case_observation()
                    if identity.startswith('clear_'):
                        result = result.model_copy(update={'visibility':'clear'})
                    self.primary_calls += 1
                    if identity == 'attributes' and self.primary_calls == 1:
                        result = result.model_copy(update={'attributes':[
                            VisualAttribute(kind='color',value='silver'), VisualAttribute(kind='subtype',value='suv')],
                            'vehicle_colors':None})
                    if self.crash_primary:
                        self.crash_primary = False
                        atomic_json(folder/'parsed-output.json', result.model_dump())
                        raise KeyboardInterrupt()
                    return result
                if schema is SceneVisualObservation:
                    self.scene_calls += 1
                    result = case_observation().model_dump()
                    if identity in {'recover','unknown','hidden'} and (self.scene_calls == 1 or identity != 'recover'):
                        result.update(target_visible=False, attributes=[], visibility='occluded' if identity=='hidden' else 'uncertain',
                            visibility_reason='Hidden by a foreground vehicle.' if identity=='hidden' else 'Identity unclear in this crop.')
                    records = [{**result, 'candidate_id':'v000'}]
                    if identity == 'clear_needs_scene':
                        records.append({**clear().model_dump(),'candidate_id':'v001',
                            'attributes':[{'kind':'color','value':'red'}]})
                    return SceneVisualObservation.model_validate({'candidates':records})
                if schema is GeneratedExpression:
                    opening = 'In the front view, the visible' if image_plan else 'The'
                    return GeneratedExpression(expression=f'{opening} white {"pickup truck" if image_plan else "truck"} ahead and to the left of the ego vehicle.',
                        predicate_ids=ids, visual_attributes=attrs)
                assert schema is GroundResult
                sentence = json.loads(messages[-1]['content'])['expression']
                parsed_attrs = [VisualAttribute(kind='color',value='white')] if 'white' in sentence else []
                if 'yellow' in sentence:
                    parsed_attrs.append(VisualAttribute(kind='color',value='yellow'))
                if 'excavator' in sentence:
                    parsed_attrs.append(VisualAttribute(kind='subtype',value='excavator'))
                if 'pickup truck' in sentence:
                    parsed_attrs.append(VisualAttribute(kind='subtype',value='pickup truck'))
                elif 'truck' in sentence:
                    parsed_attrs.append(VisualAttribute(kind='subtype',value='truck'))
                parsed_ids = [pid for pid in ids if (not pid.startswith('image_visibility:') or 'front view' in sentence)
                    and (not pid.startswith(('longitudinal:', 'lateral:')) or 'ego vehicle' in sentence)]
                return GroundResult(candidate_ids=[], parsed_predicate_ids=parsed_ids,
                    unsupported_claims=[], parsed_visual_attributes=parsed_attrs)

        calls = Calls()
        if image_plan:
            with pytest.raises(KeyboardInterrupt):
                runtime.run_agent(p, calls, 0, case/'runs', budget)
            run_id = initial_run(p, calls, 0, budget)['run_id']
            result = runtime.run_agent(p, calls, 0, case/'runs', budget, resume_run_id=run_id)
        else:
            result = runtime.run_agent(p, calls, 0, case/'runs', budget)
        assert result['diagnostic_pass'] is passed
        assert result['model_calls_used'] == calls.count == used <= budget
        assert result.get('precheck_batch_index',0) == (0 if identity == 'clear_budget' else 1)
        if passed:
            assert len(result['attempts'][-1]['ground_checks']) == 2
            if identity == 'excavator':
                assert 'yellow excavator' in result['final_expression']
            if identity == 'attributes':
                assert result['source_attributes_rechecked'] and calls.scene_calls == 2
                assert result['source_attribute_rechecks'][0]['previous_observation']['attributes'][0]['value'] == 'silver'
                assert {(a['kind'], a['value']) for a in result['resolved_target_observation']['attributes']} == {
                    (a.kind, a.value) for a in partial_observation().attributes}
            if identity == 'recover':
                assert calls.scene_calls == 2 and result['source_identity_rechecked']
                atomic_json(case/'evidence'/p.packet_id/'packet.json', p.model_dump())
                before = calls.count
                rechecked = runtime.revalidate_saved_expression(result['run_id'], calls, case/'runs', max_model_calls=2)
                assert rechecked['diagnostic_pass'] and rechecked['revalidation_reused_candidate_observations']
                assert calls.count == before+2 and calls.scene_calls == 2
                assert len(rechecked['attempts'][0]['ground_checks']) == 2
        else:
            expected = {'unknown':'precheck_target_identity_unconfirmed', 'hidden':'precheck_source_visibility_conflict'}
            assert result['stop_reason'] == expected.get(identity, 'insufficient_validation_budget')
            assert not result['attempts']


    # New drafts must retain confirmed appearance and supported scene context,
    # even when camera order/visibility could already identify the target.
    detail = strict_packet(tmp_path/'recognition',monkeypatch)
    detail.task.update(annotation_search='evidence-guided-v1',
        description_detail_policy='confirmed-appearance-and-spatial-v1')
    detail.target = detail.target.model_copy(update={'category':'sign'})
    detail.views[0].target_front_facing = True
    monkeypatch.setattr(runtime,'EVIDENCE_ROOT',tmp_path/'recognition')
    detail.candidates[0] = detail.target
    detail.predicates = predicate_catalog(detail.candidates)
    detail.provenance['scene_image_candidates'] = [{
        **detail.provenance['source_image_candidates'][0], 'category':'sign',
        'camera_name':1,'xyxy':[10,10,90,90]}]
    detail.provenance['source_image_candidates'] = detail.provenance['scene_image_candidates']
    sign_attrs = [VisualAttribute(kind='color',value='green'),VisualAttribute(kind='shape',value='rectangular')]
    observed = VisualObservation(target_visible=True,visibility='clear',visibility_reason='Green rectangular sign visible.',
        attributes=sign_attrs)
    from app.quality.description_search import descriptive_attributes
    assert descriptive_attributes(detail,{detail.target_object_id:observed.model_dump()})==sign_attrs
    assert descriptive_attributes(detail,{detail.target_object_id:{**observed.model_dump(),'attributes':[]}})==[]

    class RecognitionCalls(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count += 1
            if schema is VisualObservation:
                return observed
            if schema is GeneratedExpression:
                self.generated += 1
                payload=json.loads(messages[-1]['content'][0]['text'])
                self.ids=payload['prechecked_unique_cue_plan']['predicate_ids']
                assert payload['required_recognition_attributes']==[a.model_dump() for a in sign_attrs]
                assert {'longitudinal:front','lateral:left'}<=set(payload['required_predicate_ids'])
                return GeneratedExpression(expression=(getattr(self,'first_text','The sign visible in the front view.') if self.generated==1 else
                    'The green rectangular sign visible in the front view, ahead and to the left of the ego vehicle.'),
                    predicate_ids=self.ids,visual_attributes=[] if self.generated==1 and not hasattr(self,'first_text') else sign_attrs)
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,
                parsed_visual_attributes=sign_attrs,unsupported_claims=[])
    calls=RecognitionCalls()
    detailed=runtime.run_agent(detail,calls,1,tmp_path/'recognition-runs',6)
    assert detailed['diagnostic_pass'] and calls.generated==2
    assert 'confirmed_recognition_details_missing' in detailed['attempts'][0]['verification']['errors']
    assert not detailed['attempts'][0]['ground_checks'] and len(detailed['attempts'][-1]['ground_checks'])==2
    assert 'green rectangular sign' in detailed['final_expression']
    detail.task['description_detail_policy'] = 'confirmed-road-scene-v2'
    calls = RecognitionCalls()
    calls.first_text = 'The bottom most green rectangular sign in the front view, ahead and to the left of the ego vehicle.'
    repaired = runtime.run_agent(detail,calls,1,tmp_path/'road-scene-repair',6)
    assert repaired['diagnostic_pass'] and calls.generated == 2
    assert 'image_vertical_or_corner_wording' in repaired['attempts'][0]['verification']['errors']
    assert not repaired['attempts'][0]['ground_checks']
    assert len(repaired['attempts'][-1]['ground_checks']) == 2

    # Removing a proven-redundant camera clause costs no extra model rewrite;
    # the resulting text still has its appearance check and both blind parses.
    calls = RecognitionCalls()
    calls.first_text = 'The green rectangular sign visible in the front view, ahead and to the left of the ego vehicle.'
    concise = runtime.run_agent(detail,calls,0,tmp_path/'concise-reference',4)
    assert concise['diagnostic_pass'] and calls.generated == 1 and concise['model_calls_used'] == 4
    assert concise['final_expression'] == 'The green rectangular sign ahead and to the left of the ego vehicle.'
    assert concise['attempts'][0]['local_camera_simplification']['before'] == calls.first_text
    assert concise['attempts'][0]['appearance_check']['passed']
    assert len(concise['attempts'][0]['ground_checks']) == 2
    from app.agent.harness import repair_redundant_camera_qualifier, draft_reference_errors
    reference = EvidencePacket.model_validate(concise['current_packet'])
    reference.task['visibility_policy'] = RECOGNIZABLE_POLICY
    target_records = {reference.target_object_id:{**observed.model_dump(),'observed_camera_name':1}}
    from app.quality.description_search import image_relation_catalog
    reference.predicates.extend(image_relation_catalog(reference,target_records)[0])
    visibility = next(q for q in reference.predicates if q.predicate_id == 'image_visibility:1:sign')
    old_draft = GeneratedExpression(expression=calls.first_text,
        predicate_ids=[*concise['attempts'][0]['generated']['predicate_ids'],visibility.predicate_id],
        visual_attributes=sign_attrs)
    shorter, change = repair_redundant_camera_qualifier(reference,old_draft,target_records)
    assert shorter.expression == concise['final_expression'] and change['removed_predicate_ids'] == [visibility.predicate_id]
    assert 'redundant_camera_qualifier' in draft_reference_errors(reference,old_draft,target_records)
    assert not draft_reference_errors(reference,shorter,target_records)
    assert 'front view' in old_draft.expression  # Do not mutate a source/frozen draft.
    brief = json.loads(generation_messages(reference,None,target_records)[-1]['content'][0]['text'])
    assert all(not f['predicate_id'].startswith('image_visibility:') for f in brief['available_target_facts'])
    # Missing RGB evidence, zero-point/unknown native rivals and unmatched 2D
    # candidates must prevent the supposedly harmless removal of a camera cue.
    for extra in ('unknown-native','unknown-image'):
        unresolved = reference.model_copy(deep=True)
        if extra == 'unknown-native':
            unresolved.candidates[1] = unresolved.candidates[1].model_copy(update={
                'category':'unknown','center_m':reference.target.center_m,'lidar_points':0})
            unresolved.predicates = [*predicate_catalog(unresolved.candidates),visibility]
        else:
            unresolved.provenance['scene_image_candidates'].append({
                'object_id':'image:2:unobserved','category':'unknown','camera_name':2})
        assert repair_redundant_camera_qualifier(unresolved,old_draft,target_records) == (old_draft,None)
    # An image order genuinely depends on the named camera, even when other
    # attributes could also distinguish this object.
    image_order = visibility.model_copy(update={'predicate_id':'image_order:1:sign:leftmost','phrase':'the leftmost sign in the front view'})
    ordered = reference.model_copy(deep=True)
    ordered.predicates.append(image_order)
    ordered_draft = old_draft.model_copy(update={
        'expression':'The leftmost sign in the front view is green, ahead and to the left of the ego vehicle.',
        'predicate_ids':[p for p in old_draft.predicate_ids if p != visibility.predicate_id] + [image_order.predicate_id]})
    assert repair_redundant_camera_qualifier(ordered,ordered_draft,target_records) == (ordered_draft,None)
    from app.agent.harness import repair_camera_qualifier,image_scope_errors
    missing_scope=ordered_draft.model_copy(update={'expression':'The leftmost sign is green, ahead and to the left of the ego vehicle.'})
    scoped,receipt=repair_camera_qualifier(ordered,missing_scope)
    assert receipt and receipt['kind']=='missing_verified_camera_qualifier'
    assert scoped.expression.endswith('in the front view.') and receipt['requires_two_blind_checks']
    assert scoped.predicate_ids==missing_scope.predicate_ids and scoped.visual_attributes==missing_scope.visual_attributes
    assert 'image_camera_not_explicit_in_text' in image_scope_errors(ordered,missing_scope.expression,[image_order])
    assert not image_scope_errors(ordered,scoped.expression,[image_order])
    unsupported_scope=ordered.model_copy(deep=True);unsupported_scope.provenance['admission_plan']['passed']=False
    assert repair_camera_qualifier(unsupported_scope,missing_scope)==(missing_scope,None)
    # Generic/ambiguous image wording is left to the writer, never silently
    # assigned to a camera. Missing explicit visibility is rejected before billing.
    ambiguous_scope=missing_scope.model_copy(update={'expression':'The leftmost sign in this image.'})
    assert repair_camera_qualifier(ordered,ambiguous_scope)==(ambiguous_scope,None)
    implicit=old_draft.model_copy(update={'expression':'The green rectangular sign in the front view.'})
    assert 'image_visibility_not_explicit_in_text' in draft_reference_errors(reference,implicit,target_records)
    explicit_reference = reference.model_copy(deep=True)
    explicit_reference.task['feedback_detail_policy'] = 'explicit-feedback-details-v1'
    explicit_reference.provenance['admission_plan']['passed'] = True
    repaired, receipt = repair_camera_qualifier(explicit_reference, implicit)
    assert receipt and receipt['requires_two_blind_checks']
    assert 'visible in the front view' in repaired.expression
    assert 'image_visibility_not_explicit_in_text' not in draft_reference_errors(explicit_reference,repaired,target_records)
    assert repaired.predicate_ids == implicit.predicate_ids
    negative = implicit.model_copy(update={'expression':'The sign is not in the front view.'})
    assert repair_camera_qualifier(explicit_reference,negative) == (negative,None)

    # New scene references must change acquisition, planning AND both final
    # checks, while legacy camera-order records remain valid above.
    from app.quality.description_search import SCENE_REFERENCE_POLICY, draft_reference_predicate
    from app.quality.admission import reference_options
    scene_reference = ordered.model_copy(deep=True)
    scene_reference.task['reference_policy'] = SCENE_REFERENCE_POLICY
    scene_plan = admission_plan(scene_reference,observed,target_records)
    assert scene_plan['unique_plan_found']
    assert not any(pid.startswith(('image_order:','image_visibility:')) for pid in scene_plan['predicate_ids'])
    scene_reference.provenance['admission_plan'] = scene_plan
    options = reference_options(scene_reference,target_records,scene_plan['predicate_ids'])
    assert options and all(not pid.startswith(('image_order:','image_visibility:'))
        for option in options for pid in option['predicate_ids'])
    writer = generation_messages(scene_reference,None,target_records)
    writer_brief = json.loads(writer[-1]['content'][0]['text'])
    assert all(not p['predicate_id'].startswith(('image_order:','image_visibility:'))
        for p in writer_brief['available_target_facts'])
    assert not writer_brief['comparison_scope_constraints']
    assert 'do not use image order' in writer[0]['content'].lower()
    assert 'non_scene_reference_cue' in draft_reference_errors(scene_reference,ordered_draft,target_records)
    hidden_order = ordered_draft.model_copy(update={'predicate_ids':scene_plan['predicate_ids']})
    assert 'image_order_not_scene_reference' in draft_reference_errors(scene_reference,hidden_order,target_records)
    assert not draft_reference_errors(ordered,ordered_draft,target_records)
    from app.agent.harness import verify_tool_grounding
    parsed_order = GroundResult(candidate_ids=[],parsed_predicate_ids=ordered_draft.predicate_ids,
        parsed_visual_attributes=sign_attrs,unsupported_claims=[])
    assert 'non_scene_reference_cue' in verify_tool_grounding(scene_reference,ordered_draft,parsed_order,{},
        target_records,target_appearance_verified=True)['errors']

    # An unobserved unit in another camera cannot disappear from an ego-based
    # scene description. Observe a promising landmark early, not after all rivals.
    schedule = scene_reference.model_copy(deep=True)
    schedule.task['visibility_policy'] = 'clear-source-rgb-v1'
    tid = schedule.target_object_id
    schedule.provenance['scene_image_candidates'] = [
        {'object_id':tid,'category':'sign','camera_name':1,'camera_object_id':schedule.source_camera_object_id,'xyxy':[10,10,90,90]},
        {'object_id':'unknown-other-view','category':'unknown','camera_name':2,'xyxy':[10,10,40,40]},
        {'object_id':'salient-landmark','category':'vehicle','camera_name':1,'xyxy':[10,10,90,90]}]
    schedule.provenance['appearance_candidates'] = [
        {'candidate_id':f'v{i:03d}','object_id':oid,'category':category,'camera_name':camera,'artifact':'crop.jpg'}
        for i,(oid,category,camera) in enumerate(((tid,'sign',1),('unknown-other-view','unknown',2),('salient-landmark','vehicle',1)))]
    with monkeypatch.context() as acquisition:
        acquisition.setattr('app.agent.appearance.useful_landmark_ids',lambda *args:{'salient-landmark'})
        from app.agent.appearance import precheck_observation_batches
        scheduled = [m for batch in precheck_observation_batches(schedule) for m in batch]
    assert [m['object_id'] for m in scheduled[:2]]==[tid,'salient-landmark']
    assert 'unknown-other-view' in {m['object_id'] for m in scheduled}
    assert not admission_plan(schedule,observed,target_records).get('unique_plan_found')

    class SceneCalls(RecognitionCalls):
        def __init__(self,first_bad=False):
            super().__init__();self.first_bad=first_bad
        def call(self,messages,schema,folder,heartbeat=None):
            value=super().call(messages,schema,folder,heartbeat)
            if schema is GeneratedExpression:
                return value.model_copy(update={'expression':
                    'The leftmost green rectangular sign in the front view.' if self.first_bad and self.generated==1 else
                    'The green rectangular sign ahead and to the left of the ego vehicle.',
                    'visual_attributes':sign_attrs})
            return value
    for first_bad,budget in ((False,4),(True,6)):
        natural=detail.model_copy(deep=True)
        natural.task['reference_policy']=SCENE_REFERENCE_POLICY
        calls=SceneCalls(first_bad)
        result=runtime.run_agent(natural,calls,1,tmp_path/f'natural-scene-{first_bad}',budget)
        assert result['diagnostic_pass'] and calls.generated==1+int(first_bad)
        assert result['final_expression']=='The green rectangular sign ahead and to the left of the ego vehicle.'
        assert len(result['attempts'][-1]['ground_checks'])==2
        if first_bad:
            assert 'image_order_not_scene_reference' in result['attempts'][0]['verification']['errors']
            assert not result['attempts'][0]['ground_checks']

    # Sufficient geometry does not require compulsory color, shape or subtype.
    minimal = detail.model_copy(deep=True)
    minimal.task['reference_policy'] = 'minimal-sufficient-grounding-v1'
    minimal.task.pop('description_detail_policy',None)
    class MinimalCalls(RecognitionCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is GeneratedExpression:
                self.count+=1; self.generated+=1
                self.ids=json.loads(messages[-1]['content'][0]['text'])['prechecked_unique_cue_plan']['predicate_ids']
                return GeneratedExpression(expression='The sign ahead and to the left of the ego vehicle.',
                    predicate_ids=self.ids,visual_attributes=[])
            if schema is GroundResult:
                self.count+=1
                return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,
                    parsed_visual_attributes=[],unsupported_claims=[])
            return super().call(messages,schema,folder,heartbeat)
    minimal_result=runtime.run_agent(minimal,MinimalCalls(),0,tmp_path/'sufficient-geometry',4)
    assert minimal_result['diagnostic_pass'] and len(minimal_result['attempts'][0]['ground_checks'])==2

    # Identity-first acquisition must retain the original rivals, use real
    # anonymous evidence, and resume an interrupted small batch without replay.
    observed_sizes = []
    for mode in ('scene-first','baseline','focused','hidden'):
        location = tmp_path/mode
        evidence = strict_packet(location/'evidence',monkeypatch)
        evidence.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY)
        if mode != 'baseline':
            evidence.task['observation_policy']='identity-first-on-demand-v1'
        evidence.provenance['scene_image_candidates'] = [
            {**evidence.provenance['source_image_candidates'][0],'camera_name':1,'xyxy':[1,1,50,50]},
            {'object_id':evidence.candidates[1].object_id,'category':'vehicle','camera_name':1,
                'camera_object_id':'rival-camera','xyxy':[60,1,90,50]}]
        evidence.provenance['appearance_candidates'] = [
            {'candidate_id':f'v{i:03d}','object_id':o.object_id,'category':o.category,
                'camera_name':1,'artifact':'appearance.jpg'} for i,o in enumerate(evidence.candidates[:2])]
        if mode=='scene-first':
            evidence.task.update(reference_policy='minimal-sufficient-grounding-v1',
                reference_preference_policy='scene-first-v1',image_position_policy='verified-boundary-v1',
                image_identity_policy='official-projection-v1',image_inventory_policy='native-and-projected-v3-native-rgb')
            evidence.provenance['scene_image_candidates'][0]['xyxy']=[0,1,50,evidence.views[0].height]
            evidence.provenance['scene_image_candidates'].append({'object_id':'unlinked','category':'unknown',
                'camera_name':1,'camera_object_id':'unlinked-camera','xyxy':[60,10,80,30]})
            evidence.provenance['appearance_candidates'].append({'candidate_id':'v002','object_id':'unlinked',
                'category':'unknown','camera_name':1,'artifact':'appearance.jpg'})
        monkeypatch.setattr(runtime,'EVIDENCE_ROOT',location/'evidence')

        class FocusCalls(LocalCalls):
            scene_sizes=[]
            interrupted=False

            def call(self,messages,schema,folder,heartbeat=None):
                self.count += 1
                if schema is VisualObservation:
                    return partial_observation()
                if schema is SceneVisualObservation:
                    payload=json.loads(messages[-1]['content'][0]['text'])
                    self.scene_sizes=[*self.scene_sizes,len(payload['candidate_ids'])]
                    observation=partial_observation().model_dump()
                    if mode=='hidden':
                        observation.update(target_visible=False,visibility='occluded',attributes=[],
                            visibility_reason='The selected unit is fully hidden by another vehicle.')
                    result=SceneVisualObservation.model_validate({'candidates':[
                        {**(clear().model_copy(update={'attributes':[VisualAttribute(kind='color',value='red')]}).model_dump()
                            if mode=='scene-first' and cid!='v000' else observation),
                         'candidate_id':cid} for cid in payload['candidate_ids']]})
                    if mode=='focused' and not self.interrupted:
                        self.interrupted=True
                        atomic_json(folder/'parsed-output.json',result.model_dump())
                        raise KeyboardInterrupt()
                    return result
                if schema is GeneratedExpression:
                    payload=json.loads(messages[-1]['content'][0]['text'])
                    self.ids=payload['prechecked_unique_cue_plan']['predicate_ids']
                    if mode=='scene-first':
                        assert not any(id_.startswith(('image_boundary:','image_order:')) for id_ in self.ids)
                        self.ids=['category:vehicle','longitudinal:front','lateral:left']
                        return GeneratedExpression(expression='The white pickup truck ahead and to the left of the ego vehicle.',
                            predicate_ids=self.ids,visual_attributes=partial_observation().attributes)
                    return GeneratedExpression(expression='The white pickup truck visible in the front view, ahead and to the left of the ego vehicle.',
                        predicate_ids=self.ids,visual_attributes=partial_observation().attributes)
                assert schema is GroundResult
                return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,
                    parsed_visual_attributes=partial_observation().attributes,unsupported_claims=[])

        calls=FocusCalls()
        if mode=='focused':
            with pytest.raises(KeyboardInterrupt):
                runtime.run_agent(evidence,calls,0,location/'runs',8)
            run_id=initial_run(evidence,calls,0,8)['run_id']
            result=runtime.run_agent(evidence,calls,0,location/'runs',8,resume_run_id=run_id)
        else:
            result=runtime.run_agent(evidence,calls,0,location/'runs',8)
        assert result['diagnostic_pass'] is (mode!='hidden')
        assert result['model_calls_used']==calls.count==(2 if mode=='hidden' else 5)
        assert len(result['current_packet']['candidates'])==len(evidence.candidates)
        observed_sizes.append(calls.scene_sizes)
        if mode in {'focused','hidden'}:
            assert result['precheck_identity_focus']==[evidence.packet_id,evidence.source_view,evidence.source_camera_object_id]
        if mode=='focused':
            assert len(result['attempts'][0]['ground_checks'])==2 and len(result['candidate_appearance'])==1
        if mode=='hidden':
            assert result['stop_reason']=='precheck_source_visibility_conflict' and not result['attempts']
        if mode=='scene-first':
            assert len(result['candidate_appearance'])==3 and 'precheck_identity_focus' not in result
            assert 'cut off' not in result['final_expression'] and 'ego vehicle' in result['final_expression']
    assert observed_sizes==[[3],[2],[1],[1]]
    from app.agent.appearance import checked_observations, redundant_relook
    rival=evidence.provenance['appearance_candidates'][1]
    record=checked_observations(evidence,SceneVisualObservation.model_validate({'candidates':[
        {**partial_observation().model_dump(),'candidate_id':rival['candidate_id']}]}),[rival])[rival['object_id']]
    assert redundant_relook(evidence,rival,record)
    assert not redundant_relook(evidence,{**rival,'camera_name':2},record)
    assert not redundant_relook(evidence,rival,{**record,'visibility':'uncertain'})
    assert not redundant_relook(evidence,rival,{**record,'visibility_conflict':True})
    assert not redundant_relook(evidence,rival,{k:v for k,v in record.items() if k!='observed_input_sha256'})
    evidence.provenance['artifact_sha256']['new-context.jpg']='f'*64
    assert not redundant_relook(evidence,{**rival,'visibility_artifact':'new-context.jpg'},record)

    # A recognized competitor with exactly the same context/detail bytes gets
    # no generic reread; an ambiguous target still fails the complete proof.
    evidence.candidates[1]=evidence.candidates[1].model_copy(update={'center_m':evidence.target.center_m})
    evidence.predicates=predicate_catalog(evidence.candidates)
    evidence.provenance['scene_image_candidates'][1]['xyxy']=[1,1,50,50]
    rival['alternatives']=[dict(rival)]
    mode='duplicate'
    calls=FocusCalls()
    result=runtime.run_agent(evidence,calls,0,tmp_path/'no-duplicate-reread',8)
    assert not result['diagnostic_pass'] and result['stop_reason']=='precheck_native_identity_ambiguous'
    assert result['distinguishing_cue_bound']['native_identity_conflicts']
    assert calls.scene_sizes==[2] and calls.count==2

    # Independent observations of the SAME original inputs can serve another
    # target in this batch. Its own RGB check and both blind readers stay fresh.
    from app.agent.appearance import SceneObservationCache,checked_observations
    location=tmp_path/'shared-scene-observations'
    base=strict_packet(location/'evidence',monkeypatch)
    base.task.update(annotation_search='evidence-guided-v1',visibility_policy=RECOGNIZABLE_POLICY,
        reference_policy='minimal-sufficient-grounding-v1',reference_preference_policy='scene-first-v1')
    other=obj('third-vehicle',14,-7);base.candidates.append(other)
    base.predicates=predicate_catalog(base.candidates)
    first_id,second_id=base.target_object_id,base.candidates[1].object_id
    ids=[first_id,second_id,other.object_id]
    boxes=[{'object_id':oid,'category':'vehicle','camera_name':1,'camera_object_id':f'camera-{index}',
        'xyxy':[5+index*30,10,25+index*30,80]} for index,oid in enumerate(ids)]
    base.provenance.update(scene_image_candidates=boxes,source_image_candidates=boxes,
        appearance_candidates=[{'candidate_id':f'v{index:03d}','object_id':oid,'category':'vehicle',
            'camera_name':1,'camera_object_id':f'camera-{index}','identity_source':'official_camera_lidar_association',
            'artifact':'appearance.jpg'} for index,oid in enumerate(ids)])
    known={first_id:partial_observation(),second_id:clear().model_copy(update={'attributes':[
        VisualAttribute(kind='color',value='red'),VisualAttribute(kind='subtype',value='sedan')]}),
        other.object_id:clear().model_copy(update={'attributes':[
            VisualAttribute(kind='color',value='black'),VisualAttribute(kind='subtype',value='suv')]})}
    class ReuseCalls(LocalCalls):
        def __init__(self,p):
            super().__init__();self.packet=p;self.scene_requests=[];self.fresh_target_checks=0
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            if schema is VisualObservation:
                self.fresh_target_checks+=1;return known[self.packet.target_object_id]
            if schema is SceneVisualObservation:
                request=json.loads(messages[-1]['content'][0]['text']);self.scene_requests.append(request['candidate_ids'])
                mapping={m['candidate_id']:m['object_id'] for m in self.packet.provenance['appearance_candidates']}
                return SceneVisualObservation.model_validate({'candidates':[
                    {**known[mapping[cid]].model_dump(),'candidate_id':cid} for cid in request['candidate_ids']]})
            if schema is GeneratedExpression:
                request=json.loads(messages[-1]['content'][0]['text'])
                self.ids=request['prechecked_unique_cue_plan']['predicate_ids'];self.attrs=known[self.packet.target_object_id].attributes
                from app.quality.description_search import appearance_phrase
                phrases=[q.phrase for q in self.packet.predicates if q.predicate_id in self.ids and q.kind!='category']
                return GeneratedExpression(expression=appearance_phrase('vehicle',self.attrs)+' '+' and '.join(phrases)+'.',
                    predicate_ids=self.ids,visual_attributes=self.attrs)
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=self.ids,parsed_visual_attributes=self.attrs,unsupported_claims=[])
    cache=SceneObservationCache()
    results=[];requests=[]
    monkeypatch.setattr(runtime,'EVIDENCE_ROOT',location/'evidence')
    for index,oid in enumerate(ids[:2]):
        current=base.model_copy(deep=True)
        current.packet_id=str(index+1)*24;current.target_object_id=oid
        current.target=next(o for o in current.candidates if o.object_id==oid)
        current.source_camera_object_id=f'camera-{index}'
        current.views[0].appearance_camera_object_id=current.source_camera_object_id
        shutil.copytree(location/'evidence'/base.packet_id,location/'evidence'/current.packet_id)
        calls=ReuseCalls(current)
        if index==1:
            original_save=RunStore.save
            def interrupt_after_reuse(store,run_id,owner,state,kind,payload):
                original_save(store,run_id,owner,state,kind,payload)
                if kind=='precheck_target_observed' and state.get('scene_observation_reuse_receipts'):
                    raise KeyboardInterrupt()
            with monkeypatch.context() as interrupted:
                interrupted.setattr(RunStore,'save',interrupt_after_reuse)
                with pytest.raises(KeyboardInterrupt):
                    runtime.run_agent(current,calls,0,location/'runs',8,observation_cache=cache)
            resumed_cache=SceneObservationCache()
            result=runtime.run_agent(current,calls,0,location/'runs',8,
                resume_run_id=initial_run(current,calls,0,8)['run_id'],observation_cache=resumed_cache)
            assert not resumed_cache.entries
        else:
            result=runtime.run_agent(current,calls,0,location/'runs',8,observation_cache=cache)
        assert result['diagnostic_pass'] and calls.fresh_target_checks==1
        assert len(result['attempts'][-1]['ground_checks'])==2
        assert len(result['current_packet']['candidates'])==len(current.candidates)
        results.append(result);requests.append(calls.scene_requests)
    assert [r['model_calls_used'] for r in results]==[5,4]
    assert requests[0] and not requests[1]
    assert results[1]['observation_source_run_ids']==[results[0]['run_id']]
    assert results[1]['candidate_appearance'][other.object_id]==results[0]['candidate_appearance'][other.object_id]
    receipt=results[1]['scene_observation_reuse_receipts'][0]
    assert receipt['source_call_folder']=='model/01-precheck_candidates' and receipt['source_run_id']==results[0]['run_id']
    baseline=current.model_copy(deep=True)
    baseline.task['execution_attempt']='without-observation-cache'
    baseline_calls=ReuseCalls(baseline)
    baseline_result=runtime.run_agent(baseline,baseline_calls,0,location/'runs',8)
    assert baseline_result['diagnostic_pass'] and baseline_result['model_calls_used']==5
    # A peer's evidence becomes available AFTER initial planning. Preserve empty
    # batch positions, skip the now-redundant request, and resume with a cold cache.
    from app.agent import appearance as appearance_module
    original_batches=appearance_module.precheck_observation_batches
    late_root=location/'late-runs'
    late_cache=SceneObservationCache()
    seed=base.model_copy(deep=True);seed.packet_id='1'*24
    seed.source_camera_object_id='camera-0';seed.views[0].appearance_camera_object_id='camera-0'
    seed_calls=ReuseCalls(seed)
    seeded=runtime.run_agent(seed,seed_calls,0,late_root,8,observation_cache=late_cache)
    assert seeded['diagnostic_pass']
    class DeferredCache:
        ready=False
        def reuse(self,*args):return late_cache.reuse(*args) if self.ready else ({},[])
        def remember(self,*args):return late_cache.remember(*args)
    deferred=DeferredCache()
    late_calls=ReuseCalls(current)
    ordinary_call=late_calls.call
    def complete_first_batch(*args,**kwargs):
        response=ordinary_call(*args,**kwargs)
        if args[1] is SceneVisualObservation:deferred.ready=True
        return response
    late_calls.call=complete_first_batch
    with monkeypatch.context() as late_schedule:
        late_schedule.setattr(appearance_module,'precheck_observation_batches',
            lambda p:[[m] for batch in original_batches(p) for m in batch])
        normal_save=RunStore.save
        def interrupt_late_reuse(store,run_id,owner,state,kind,payload):
            normal_save(store,run_id,owner,state,kind,payload)
            if kind=='precheck_scene_observed' and payload.get('reused_scene_candidates') and state['phase']=='precheck_decision':
                raise KeyboardInterrupt()
        with monkeypatch.context() as interrupted:
            interrupted.setattr(RunStore,'save',interrupt_late_reuse)
            with pytest.raises(KeyboardInterrupt):
                runtime.run_agent(current,late_calls,0,late_root,8,observation_cache=deferred)
        late_result=runtime.run_agent(current,late_calls,0,late_root,8,
            resume_run_id=initial_run(current,late_calls,0,8)['run_id'],observation_cache=SceneObservationCache())
        assert late_result['diagnostic_pass'] and len(late_result['attempts'][-1]['ground_checks'])==2
        fresh_ids=[cid for batch in late_calls.scene_requests for cid in batch]
        assert late_calls.fresh_target_checks==1 and 'v001' in fresh_ids and 'v002' not in fresh_ids
        assert len(fresh_ids)==len(set(fresh_ids))  # Uncached optional landmarks stay fresh, never replayed.
        assert late_result['precheck_reused_batch_ids'] and late_result['observation_source_run_ids']==[seeded['run_id']]
        comparison=current.model_copy(deep=True);comparison.task['execution_attempt']='late-cache-disabled'
        comparison_calls=ReuseCalls(comparison)
        comparison_result=runtime.run_agent(comparison,comparison_calls,0,tmp_path/'late-baseline',8)
        assert comparison_result['diagnostic_pass']
        assert comparison_result['model_calls_used']>late_result['model_calls_used']
    # Partial reuse changes the actual reserved manifest. A crash AFTER its
    # response was saved must restore that same manifest with an empty cache,
    # not resend the old full batch or reject its reduced candidate ID list.
    reserved_root=tmp_path/'reserved';reserved_cache=SceneObservationCache()
    reserved_source=runtime.run_agent(seed,ReuseCalls(seed),0,reserved_root,8,observation_cache=reserved_cache)
    class AppearingPeer:
        lookups=0
        def reuse(self,*args):
            self.lookups+=1
            return ({},[]) if self.lookups==1 else reserved_cache.reuse(*args)
        def remember(self,*args):
            reserved_cache.remember(*args)
            raise KeyboardInterrupt()
    reserved_calls=ReuseCalls(current)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_agent(current,reserved_calls,0,reserved_root,8,observation_cache=AppearingPeer())
    restored=runtime.run_agent(current,reserved_calls,0,reserved_root,8,
        resume_run_id=initial_run(current,reserved_calls,0,8)['run_id'],observation_cache=SceneObservationCache())
    assert restored['diagnostic_pass'] and restored['model_calls_used']==5
    assert reserved_calls.fresh_target_checks==1 and len(reserved_calls.scene_requests)==1
    assert 'v001' in reserved_calls.scene_requests[0] and 'v002' not in reserved_calls.scene_requests[0]
    assert restored['precheck_reused_batch_ids'] and restored['observation_source_run_ids']==[reserved_source['run_id']]
    assert len(restored['attempts'][-1]['ground_checks'])==2
    scope=(results[1]['provider_scope'],results[1]['model'],results[1]['base_url'],results[1]['code_hashes'])
    manifest=current.provenance['appearance_candidates']
    assert cache.reuse(current,manifest,('different-provider',),location/'evidence')==({},[])
    moved=current.model_copy(update={'frame_timestamp_micros':current.frame_timestamp_micros+1})
    assert cache.reuse(moved,manifest,scope,location/'evidence')==({},[])
    uncertain=SceneObservationCache(capacity=1)
    record={**results[0]['candidate_appearance'][other.object_id],'target_visible':False,'visibility':'uncertain'}
    uncertain.remember(current,manifest,{other.object_id:record},scope,results[0]['run_id'],'model/unknown')
    assert not uncertain.entries
    reliable=results[0]['candidate_appearance'][other.object_id]
    for extra,absent in [({'kind':'color','value':'red'},[]),({'kind':'feature','value':'roof rack'},['roof rack'])]:
        conflict={**reliable,'attributes':[*reliable['attributes'],extra],'absent_features':absent}
        uncertain.remember(current,manifest,{other.object_id:conflict},scope,results[0]['run_id'],'model/conflict')
        assert not uncertain.entries
    uncertain.remember(current,manifest,results[0]['candidate_appearance'],scope,results[0]['run_id'],'model/01-precheck_candidates')
    assert len(uncertain.entries)==1
    asset=location/'evidence'/current.packet_id/'appearance.jpg'
    asset.write_bytes(b'changed after preparation')
    with pytest.raises(ValueError,match='hash mismatch'):
        cache.reuse(current,manifest,scope,location/'evidence')
    cache.clear();assert not cache.entries


def test_projected_inventory_preserves_unresolved_native_competitors(tmp_path, monkeypatch):
    from app.datasets.appearance_candidates import scene_image_inventory, build_candidate_crops, IMAGE_INVENTORY_POLICY
    from app.quality.description_search import image_relation_catalog, expression_candidates
    from app.agent.harness import verify_tool_grounding, expression_policy_errors
    from app.agent.agent_runtime import initial_run
    from app.datasets.correspondence import point_supported_native_boxes

    # A camera identifies the red landmark, while native vehicle heading defines
    # front/rear. Same-camera unlinked objects cannot be removed by 2D geometry.
    from app.quality.relations import expand_relations, VEHICLE_HEADING_POLICY
    heading = strict_packet(tmp_path/'heading-evidence',monkeypatch)
    heading.task.update(annotation_search='evidence-guided-v1',reference_policy='minimal-sufficient-grounding-v1',
        reference_preference_policy='traffic-scene-v1',vehicle_heading_policy=VEHICLE_HEADING_POLICY,
        image_identity_policy='official-projection-v1',image_inventory_policy=IMAGE_INVENTORY_POLICY)
    heading.target.center_m=(10,5,1)
    red=obj('red-landmark',10,15).model_copy(update={'heading_rad':-math.pi/2})
    heading.target.heading_rad=-math.pi/2
    heading.candidates=[heading.target, obj('other-sedan',20,5),red]
    heading.predicates=predicate_catalog(heading.candidates)
    heading.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':o.category,
        'camera_name':1,'camera_object_id':heading.source_camera_object_id if i==0 else o.object_id,
        'xyxy':[i*60+1,5,i*60+50,55],'identity_source':'official_camera_lidar_association'}
        for i,o in enumerate(heading.candidates)]
    heading.provenance['source_image_candidates']=heading.provenance['scene_image_candidates']
    heading_records={o.object_id:{**clear().model_dump(),'observed_camera_name':1,
        'attributes':[{'kind':'color','value':'red' if o==red else 'white'},{'kind':'subtype','value':'sedan'}]}
        for o in heading.candidates}
    heading, expanded=expand_relations(heading,heading_records)
    rule=next(p for p in heading.predicates if p.predicate_id.startswith('image_vehicle_heading:1:front:')
        and 'red sedan' in p.phrase)
    assert rule.matches[heading.target_object_id] is True and rule.matches['other-sedan'] is False
    assert rule.matches[red.object_id] is False
    assert any(p.kind=='vehicle_heading' and p.matches[heading.target_object_id] is True for p in heading.predicates)
    attrs=[VisualAttribute(kind='color',value='white'),VisualAttribute(kind='subtype',value='sedan')]
    generated=GeneratedExpression(expression='The white sedan directly in front of the red sedan in the front view.',
        predicate_ids=['category:vehicle',rule.predicate_id],visual_attributes=attrs)
    parsed=GroundResult(candidate_ids=[],parsed_predicate_ids=generated.predicate_ids,
        parsed_visual_attributes=attrs,unsupported_claims=[])
    assert verify_tool_grounding(heading,generated,parsed,{},heading_records,True)['diagnostic_pass']
    rotated=heading.model_copy(deep=True)
    next(o for o in rotated.candidates if o.object_id==red.object_id).heading_rad=math.pi/2
    assert 'unconfirmed_image_reference_evidence' in verify_tool_grounding(
        rotated,generated,parsed,{},heading_records,True)['errors']
    # New existential landmarks allow two red cars when the WHOLE sentence
    # still selects one white target. Unknown pairs never count as exclusions.
    from app.quality.joint_reference import POLICY as JOINT_POLICY, writing_plan, fallback_sentence, writer_messages, WrittenSentence
    from app.agent import scene_grounding
    joint=heading.model_copy(deep=True)
    joint.task.update(joint_reference_policy=JOINT_POLICY,scene_grounding_policy=scene_grounding.POLICY)
    second=red.model_copy(update={'object_id':'second-red','center_m':(40,15,1)})
    joint.candidates.append(second)
    joint.provenance['scene_image_candidates'].append({'object_id':second.object_id,'category':'vehicle',
        'camera_name':1,'xyxy':[200,5,240,55],'identity_source':'official_camera_lidar_association'})
    joint_records={**heading_records,second.object_id:heading_records[red.object_id]}
    joint.predicates=predicate_catalog(joint.candidates)
    joint,_=expand_relations(joint,joint_records)
    joint_rule=next(p for p in joint.predicates if p.predicate_id.startswith('image_joint_heading:1:front:') and 'red sedan' in p.phrase)
    assert len(json.loads(joint_rule.value)['confirmed_anchor_ids'])==2
    assert joint_rule.matches[joint.target_object_id] is True and joint_rule.matches['other-sedan'] is False
    assert joint_rule.matches[red.object_id] is False
    joint_generated=GeneratedExpression(expression='The white sedan directly in front of a red sedan in the front view.',
        predicate_ids=['category:vehicle',joint_rule.predicate_id],visual_attributes=attrs)
    joint_parsed=parsed.model_copy(update={'parsed_predicate_ids':joint_generated.predicate_ids})
    assert verify_tool_grounding(joint,joint_generated,joint_parsed,{},joint_records,True)['diagnostic_pass']
    plan=writing_plan(joint,joint_records)
    assert plan['joint_reference'] and plan['unique_plan_found']
    joint.provenance['admission_plan']={'passed':True}
    wire=writer_messages(joint,plan,None,joint_records)
    contract=json.loads(wire[-1]['content'][0]['text'])
    assert set(contract['output_schema']['properties'])=={'expression'}
    assert any('a red sedan' in clause for clause in contract['required_meaning'])
    assert 'a red sedan' in fallback_sentence(joint,plan,joint_records)
    assert not any(p.predicate_id.startswith('image_joint_heading:') for p in heading.predicates)
    from app.agent.agent_runtime import run_agent
    joint.task.update(natural_description=True,execution_policy='evidence-fast-v1')
    class JointCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            if schema is WrittenSentence:
                # Omission must be rejected, then a local proposal retains the
                # selected contract without buying another writing request.
                return WrittenSentence(expression='The white sedan in the front view.')
            assert schema is GroundResult
            return GroundResult(candidate_ids=[],parsed_predicate_ids=plan['predicate_ids'],
                parsed_visual_attributes=plan['visual_attributes'],unsupported_claims=[])
    writer_calls=JointCalls();root=tmp_path/'joint-writer'
    initial=initial_run(joint,writer_calls,1,3)
    initial.update(phase='generate',relations_acquired=True,admission={'passed':True},
        candidate_appearance=joint_records,target_precheck_observation={k:v for k,v in joint_records[joint.target_object_id].items() if k!='observed_camera_name'},
        resolved_target_observation=joint_records[joint.target_object_id],attempts=[{'attempt':0,'ground_checks':[]}])
    RunStore(root/'runs.sqlite3').create(initial)
    result=run_agent(joint,writer_calls,1,root,3,resume_run_id=initial['run_id'])
    assert result['diagnostic_pass'] and writer_calls.count==3,result
    assert 'declared_joint_relation_not_in_text' in result['attempts'][0]['verification']['errors']
    assert result['attempts'][1]['generated']['predicate_ids']==plan['predicate_ids']
    assert result['attempts'][1]['generation_source'].startswith('local proposal')
    # Blind scene coordinates follow the supplied pixels; random IDs and
    # geometry-checked supporting pairs cannot smuggle in a designated target.
    monkeypatch.setattr(scene_grounding,'EVIDENCE_ROOT',tmp_path/'heading-evidence')
    joint.views[0].width=200; joint.views[0].height=200
    for seed in (601,602):
        msgs,mapping,receipt=scene_grounding.messages(joint,joint_generated.expression,seed)
        wire=json.loads(msgs[-1]['content'][0]['text'])
        assert wire['image_box_coordinates']=={'width':100,'height':100,'units':'provided image pixels'}
        assert joint.target_object_id not in json.dumps(msgs)
        for row in wire['candidates']:
            assert row['image_boxes_xyxy']==[[v/2 for v in box] for box in row['original_image_boxes_xyxy']]
        reverse={oid:cid for cid,oid in mapping.items()}
        graph=wire['native_vehicle_relations']
        anchor_row=next(row for row in graph['landmarks'] if row['landmark_candidate_id']==reverse[red.object_id])
        assert reverse[joint.target_object_id] in anchor_row['front']
        assert reverse['other-sedan'] not in anchor_row['front']+anchor_row['front_unknown']
        swapped=joint.model_copy(deep=True)
        swapped.target=next(o for o in swapped.candidates if o.object_id=='other-sedan')
        swapped.target_object_id=swapped.target.object_id
        assert scene_grounding.messages(swapped,joint_generated.expression,seed)[0]==msgs
        answer=scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[reverse[joint.target_object_id]],
            ambiguous_candidate_ids=[],unsupported_claims=[],reason='All pairs checked.',
            relation_pairs=[{'candidate_id':reverse[joint.target_object_id],'landmark_id':reverse[red.object_id]}])
        check=lambda a:scene_grounding.verify(joint,a,mapping,receipt,generated=joint_generated.model_dump())
        assert check(answer)['diagnostic_pass']
        assert not check(answer.model_copy(update={'relation_pairs':[]}))['diagnostic_pass']
        wrong=answer.model_copy(deep=True);wrong.relation_pairs[0].landmark_id=reverse[second.object_id]
        assert 'joint_reference_pair_unverified' in check(wrong)['errors']
        wrong.relation_pairs[0].landmark_id=reverse[joint.target_object_id]
        assert not check(wrong)['diagnostic_pass']
    # A descriptor/ID binding error remains a failed observation. One fresh
    # blind read is bounded by the existing budget and survives interruption.
    binding=joint.model_copy(deep=True)
    binding.task['scene_binding_policy']=scene_grounding.BINDING_POLICY
    msgs,mapping,receipt=scene_grounding.messages(binding,joint_generated.expression,601)
    reverse={oid:cid for cid,oid in mapping.items()}
    wrong=scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[reverse[binding.target_object_id]],
        ambiguous_candidate_ids=[],unsupported_claims=[],reason='Candidate binding example.',
        relation_pairs=[{'candidate_id':reverse[binding.target_object_id],'landmark_id':reverse['other-sedan'],
            'relation_key':'vehicle_directly_ahead'}])
    failure=scene_grounding.verify(binding,wrong,mapping,receipt,generated=joint_generated.model_dump())
    assert 'joint_reference_landmark_identity_mismatch' in failure['errors']
    retry=lambda f,used=0,left=2:scene_grounding.binding_retry_allowed(binding,f,retries_used=used,remaining_calls=left,phase='scene_ground')
    assert retry(failure) and not retry(failure,used=1) and not retry(failure,left=1)
    for field,value in [('expression_supported',False),('ambiguous_candidate_ids',[reverse['other-sedan']]),
            ('candidate_ids',[reverse['other-sedan']]),('unsupported_claims',['The relation is false.'])]:
        rejected=scene_grounding.verify(binding,wrong.model_copy(update={field:value}),mapping,receipt,generated=joint_generated.model_dump())
        assert not retry(rejected)
    unknown_binding=binding.model_copy(deep=True)
    for predicate in unknown_binding.predicates:
        if predicate.predicate_id==joint_rule.predicate_id:
            proof=json.loads(predicate.value);proof['possible_anchor_ids'].append('other-sedan');predicate.value=json.dumps(proof)
    assert not retry(scene_grounding.verify(unknown_binding,wrong,mapping,receipt,generated=joint_generated.model_dump()))
    swapped=binding.model_copy(deep=True);swapped.target_object_id='other-sedan';swapped.target=swapped.candidates[1]
    assert scene_grounding.messages(swapped,joint_generated.expression,601)[0]==msgs
    assert receipt['scene_binding_policy']==scene_grounding.BINDING_POLICY
    # More than a page of regions preserves EVERY box without a tall sheet.
    import io
    photo=io.BytesIO();Image.new('RGB',(100,100)).save(photo,format='JPEG')
    guides=scene_grounding.anonymous_image_guides(photo.getvalue(),[
        {'candidate_id':f'C{i:03d}','image_boxes_xyxy':[[0,0,70,70]]} for i in range(25)],binding=True)
    assert len(guides)==4
    assert all(Image.open(io.BytesIO(data)).height<=960 for _,data in guides)
    card_rows=[{'candidate_id':f'V{i:02d}','image_boxes_xyxy':[[0,0,70,70]]} for i in range(20)]
    cards=scene_grounding.anonymous_image_guides(photo.getvalue(),card_rows,binding=True,cards=True)
    assert len(cards)==21  # Overview plus one explicitly captioned image per region.
    for row,(caption,data) in zip(card_rows,cards[1:]):
        assert caption.startswith('Anonymous region card: '+row['candidate_id']+' region 1.')
        assert Image.open(io.BytesIO(data)).size==(640,480)
    dense_rows=card_rows+[{'candidate_id':f'V{i:02d}','image_boxes_xyxy':[[0,0,0,0]]} for i in range(20,25)]
    dense=scene_grounding.anonymous_image_guides(photo.getvalue(),dense_rows,binding=True,cards=True)
    assert len(dense)==8 and all(Image.open(io.BytesIO(data)).height<=720 for _,data in dense)
    captions=' '.join(c for c,_ in dense[1:])
    assert all(captions.count(row['candidate_id']+' region 1')==1 for row in dense_rows)
    card_packet=binding.model_copy(deep=True)
    card_packet.task['scene_region_card_policy']=scene_grounding.REGION_CARD_POLICY
    card_messages,_,card_receipt=scene_grounding.messages(card_packet,joint_generated.expression,601)
    card_packet.target_object_id='other-sedan';card_packet.target=card_packet.candidates[1]
    assert scene_grounding.messages(card_packet,joint_generated.expression,601)[0]==card_messages
    assert card_receipt['scene_region_card_policy']==scene_grounding.REGION_CARD_POLICY
    # Small regions must occupy readable card space, rather than remaining a
    # 30px island. Apply identically to every anonymous object and freeze V1.
    tiny_photo=io.BytesIO();Image.new('RGB',(100,100),'#ff0000').save(tiny_photo,format='JPEG')
    tiny_rows=[{'candidate_id':'V01','image_boxes_xyxy':[[10,10,20,25]]}]
    old_tiny=scene_grounding.anonymous_image_guides(tiny_photo.getvalue(),tiny_rows,binding=True,cards=True)
    new_tiny=scene_grounding.anonymous_image_guides(tiny_photo.getvalue(),tiny_rows,binding=True,cards=True,legible=True)
    def red_pixels(data):
        return sum(r>150 and g<80 and b<80 for r,g,b in Image.open(io.BytesIO(data)).getdata())
    assert red_pixels(new_tiny[1][1])>red_pixels(old_tiny[1][1])*10
    assert old_tiny[0]==new_tiny[0]  # Identical full-scene guide; no target emphasis.
    assert scene_grounding.anonymous_image_guides(tiny_photo.getvalue(),tiny_rows,binding=True,cards=True)==old_tiny
    legible_packet=card_packet.model_copy(deep=True)
    legible_packet.task['scene_region_card_policy']=scene_grounding.LEGIBLE_REGION_CARD_POLICY
    legible_messages,_,legible_receipt=scene_grounding.messages(legible_packet,joint_generated.expression,601)
    assert legible_receipt['scene_region_card_policy']==scene_grounding.LEGIBLE_REGION_CARD_POLICY
    assert 'not new visual evidence' in legible_messages[0]['content']
    legible_packet.target_object_id=binding.target_object_id;legible_packet.target=binding.target
    assert scene_grounding.messages(legible_packet,joint_generated.expression,601)[0]==legible_messages
    # Rejection still cannot be turned into a retry/pass by new presentation.
    assert not scene_grounding.binding_retry_allowed(card_packet,rejected,retries_used=0,remaining_calls=12,phase='scene_ground')
    class BindingCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            payload=json.loads(messages[-1]['content'][0]['text'])
            assert 'expected' not in payload and 'feedback' not in payload
            assert payload['region_identity_check']['policy']==scene_grounding.BINDING_POLICY
            rows=payload['candidates']
            tid=next(r['candidate_id'] for r in rows if r['original_image_boxes_xyxy']==[[1,5,50,55]])
            anchor=next(r['candidate_id'] for r in rows if r['original_image_boxes_xyxy']==([[61,5,110,55]] if self.count==1 else [[121,5,170,55]]))
            answer=schema(expression_supported=True,candidate_ids=[tid],ambiguous_candidate_ids=[],unsupported_claims=[],reason='Read the complete scene.',
                relation_pairs=[{'candidate_id':tid,'landmark_id':anchor,'relation_key':'vehicle_directly_ahead'}])
            if self.count==2:
                atomic_json(folder/'parsed-output.json',answer.model_dump())
                raise KeyboardInterrupt()
            return answer
    calls=BindingCalls();root=tmp_path/'binding-retry'
    initial=initial_run(binding,calls,0,3)
    initial.update(phase='scene_ground',relations_acquired=True,admission={'passed':True},
        candidate_appearance=joint_records,attempts=[{'attempt':0,'generated':joint_generated.model_dump(),
            'appearance_check':{'passed':True},'ground_checks':[]}])
    RunStore(root/'runs.sqlite3').create(initial)
    with pytest.raises(KeyboardInterrupt):
        run_agent(binding,calls,0,root,3,resume_run_id=initial['run_id'])
    result=run_agent(binding,calls,0,root,3,resume_run_id=initial['run_id'])
    assert result['diagnostic_pass'] and result['quality_gate']['scene_grounding_checks_passed'],result
    checks=result['attempts'][0]['ground_checks']
    assert calls.count==3 and result['model_calls_used']==3 and len(checks)==3
    assert not checks[0]['diagnostic_pass'] and checks[0]['binding_retry_scheduled']
    assert scene_grounding.two_rounds_passed(checks) and not scene_grounding.two_rounds_passed(checks[:2])
    assert len(list((root/result['run_id']).glob('private-map*')))==3
    rival=joint.model_copy(deep=True)
    next(o for o in rival.candidates if o.object_id=='other-sedan').center_m=(40,5,1)
    rival.predicates=predicate_catalog(rival.candidates)
    rival,_=expand_relations(rival,joint_records)
    assert not verify_tool_grounding(rival,joint_generated,joint_parsed,{},joint_records,True)['diagnostic_pass']
    unknown=joint.model_copy(deep=True)
    unknown.provenance['scene_image_candidates'].append({'object_id':'unknown-anchor','category':'unknown',
        'camera_name':1,'xyxy':[1,1,2,2]})
    unknown,_=expand_relations(unknown,joint_records)
    unknown_rule=next(p for p in unknown.predicates if p.predicate_id==joint_rule.predicate_id)
    assert unknown_rule.matches['other-sedan'] is None and unknown_rule.matches['unknown-anchor'] is None
    assert not verify_tool_grounding(unknown,joint_generated,joint_parsed,{},joint_records,True)['diagnostic_pass']
    # A visible target need not be exactly laterally aligned to be IN FRONT
    # of a vehicle. Keep this new vocabulary distinct from DIRECTLY in front.
    from app.quality.relations import VEHICLE_FORWARD_REGION_POLICY
    region=heading.model_copy(deep=True)
    region.task.update(joint_reference_policy=JOINT_POLICY,vehicle_forward_region_policy=VEHICLE_FORWARD_REGION_POLICY,
        scene_grounding_policy=scene_grounding.POLICY,natural_description=True,execution_policy='evidence-fast-v1')
    region.target.center_m=(12,5,1)
    next(o for o in region.candidates if o.object_id==region.target_object_id).center_m=region.target.center_m
    next(o for o in region.candidates if o.object_id=='other-sedan').center_m=(20,25,1)
    region.provenance['scene_image_candidates'].append({'object_id':'image:unresolved','category':'vehicle','camera_name':1,'xyxy':[80,5,90,20]})
    region.predicates=predicate_catalog(region.candidates)
    region,_=expand_relations(region,heading_records)
    ahead_rule=next(p for p in region.predicates if p.predicate_id.startswith('image_joint_heading:1:ahead:') and 'red sedan' in p.phrase)
    direct=next(p for p in region.predicates if p.predicate_id.startswith('image_joint_heading:1:front:') and 'red sedan' in p.phrase)
    assert ahead_rule.matches[region.target_object_id] is True and direct.matches[region.target_object_id] is None
    plain=GeneratedExpression(expression='The white sedan in front of a red sedan in the front view.',predicate_ids=['category:vehicle',ahead_rule.predicate_id],visual_attributes=attrs)
    from app.agent.harness import draft_reference_errors
    assert not draft_reference_errors(region,plain,heading_records)
    assert 'declared_joint_relation_not_in_text' in draft_reference_errors(region,plain.model_copy(update={'expression':plain.expression.replace('in front of','directly in front of')}),heading_records)
    assert 'declared_joint_relation_not_in_text' in draft_reference_errors(region,plain.model_copy(update={'predicate_ids':['category:vehicle',direct.predicate_id]}),heading_records)
    legacy=region.model_copy(deep=True);legacy.task.pop('vehicle_forward_region_policy')
    legacy,_=expand_relations(legacy,heading_records)
    assert not any(':ahead:' in p.predicate_id and p.predicate_id.startswith('image_joint_heading:') for p in legacy.predicates)
    class ForwardCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            if schema is WrittenSentence:return WrittenSentence(expression=plain.expression)
            assert schema is scene_grounding.SceneGroundResult
            payload=json.loads(messages[-1]['content'][0]['text'])
            target_id=next(c['candidate_id'] for c in payload['candidates'] if c.get('center_m')==[12,5,1])
            anchor_id=next(c['candidate_id'] for c in payload['candidates'] if c.get('center_m')==[10,15,1])
            graph=next(r for r in payload['native_vehicle_relations']['landmarks'] if r['landmark_candidate_id']==anchor_id)
            assert target_id in graph['ahead'] and target_id not in graph['front']
            return schema(expression_supported=True,candidate_ids=[target_id],ambiguous_candidate_ids=[],unsupported_claims=[],reason='Unique visible native pair.',
                relation_pairs=[{'candidate_id':target_id,'landmark_id':anchor_id}])
    forward_calls=ForwardCalls();root=tmp_path/'forward-region-run'
    initial=initial_run(region,forward_calls,0,3)
    initial.update(phase='precheck_decision',relations_acquired=True,relook_completed=True,candidate_appearance=heading_records,
        target_precheck_observation={k:v for k,v in heading_records[region.target_object_id].items() if k!='observed_camera_name'})
    RunStore(root/'runs.sqlite3').create(initial)
    result=run_agent(region,forward_calls,0,root,3,resume_run_id=initial['run_id'])
    assert result['diagnostic_pass'] and forward_calls.count==3,result
    assert len(result['attempts'][0]['ground_checks'])==2
    assert all(c['verified_relation_pairs']==[{'target_object_id':region.target_object_id,'landmark_object_id':red.object_id}] for c in result['attempts'][0]['ground_checks'])
    # Typed relations generalize to different landmark categories and frames.
    # Neither a left-of-sign nor a below-person clause is sufficient alone.
    from app.quality import spatial_relations as spatial
    typed=heading.model_copy(deep=True)
    typed.task.update(spatial_relation_policy=spatial.LEGACY_POLICY,joint_reference_policy=JOINT_POLICY,
        scene_grounding_policy=scene_grounding.POLICY,natural_description=True)
    typed.candidates=[typed.target,obj('upper-car',10,5),obj('right-car',10,5),
        obj('sign-anchor',10,5,'sign'),obj('person-anchor',10,5,'pedestrian')]
    locations=[[60,80,90,100],[60,0,90,20],[170,80,200,100],[130,110,160,140],[10,40,40,50]]
    typed.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':o.category,'camera_name':1,
        'camera_object_id':typed.source_camera_object_id if i==0 else o.object_id,'xyxy':locations[i],
        'identity_source':'official_camera_lidar_association'} for i,o in enumerate(typed.candidates)]
    typed.provenance['source_image_candidates']=typed.provenance['scene_image_candidates']
    typed_records={o.object_id:{**clear().model_dump(),'observed_camera_name':1,'attributes':
        ([{'kind':'color','value':'red'}] if o.category=='sign' else
         [{'kind':'upper_clothing_color','value':'blue'}] if o.category=='pedestrian' else
         [a.model_dump() for a in attrs])} for o in typed.candidates}
    typed.predicates=predicate_catalog(typed.candidates)
    typed,_=expand_relations(typed,typed_records)
    def spatial_rule(p,key,category):
        return next(r for r in p.predicates if r.predicate_id.startswith('image_joint_spatial:')
            and json.loads(r.value)['relation']==key and json.loads(r.value)['category']==category)
    left=spatial_rule(typed,'image_left','sign');below=spatial_rule(typed,'image_below','pedestrian')
    category=next(p for p in typed.predicates if p.predicate_id=='category:vehicle')
    for r in (left,below):
        assert len(expression_candidates(typed,[category,r],attrs,typed_records)['native'])==2
    assert expression_candidates(typed,[category,left,below],attrs,typed_records)['native']==[typed.target_object_id]
    proposal=GeneratedExpression(expression='The white sedan is to the left of a red sign and below a pedestrian wearing a blue top in the front view.',
        predicate_ids=[p.predicate_id for p in (category,left,below)],visual_attributes=attrs)
    assert not draft_reference_errors(typed,proposal,typed_records)
    parsed_compound=GroundResult(candidate_ids=[],parsed_predicate_ids=proposal.predicate_ids,
        parsed_visual_attributes=attrs,unsupported_claims=[])
    assert verify_tool_grounding(typed,proposal,parsed_compound,{},typed_records,True)['diagnostic_pass']
    assert 'declared_joint_relation_not_in_text' in draft_reference_errors(typed,
        proposal.model_copy(update={'expression':proposal.expression.replace('below','behind')}),typed_records)
    # Image-above is not native height; zero-point/unlinked vertical pairs stay unknown.
    native={o.object_id:o for o in typed.candidates};boxes={b['object_id']:b for b in typed.provenance['scene_image_candidates']}
    assert spatial.pair_match(spatial.BY_KEY['image_above'],'upper-car','person-anchor',boxes,native) is True
    assert spatial.pair_match(spatial.BY_KEY['higher_level'],'upper-car','person-anchor',boxes,native) is None
    elevated={**native,'upper-car':native['upper-car'].model_copy(update={'center_m':(10,5,10)})}
    assert spatial.pair_match(spatial.BY_KEY['higher_level'],'upper-car','person-anchor',boxes,elevated) is True
    elevated['upper-car']=elevated['upper-car'].model_copy(update={'lidar_points':0})
    assert spatial.pair_match(spatial.BY_KEY['higher_level'],'upper-car','person-anchor',boxes,elevated) is None
    # A recognized image-only cyclist is a valid 2D landmark, never a guessed 3D one.
    image_only=typed.model_copy(deep=True)
    image_only.provenance['scene_image_candidates'].append({'object_id':'image:cyclist','category':'cyclist','camera_name':1,'xyxy':[230,30,260,60]})
    more={**typed_records,'image:cyclist':{**clear().model_dump(),'observed_camera_name':1,'attributes':[{'kind':'helmet_color','value':'yellow'}]}}
    image_only,_=expand_relations(image_only,more)
    cycle_rule=spatial_rule(image_only,'image_left','cyclist')
    assert 'yellow helmet' in cycle_rule.phrase and cycle_rule.matches[typed.target_object_id] is True
    assert not any(json.loads(r.value)['category']=='cyclist' and json.loads(r.value)['reference_frame']=='native_vertical'
        for r in image_only.predicates if r.predicate_id.startswith('image_joint_spatial:'))
    # Unknown landmarks retain possible matches; self-reference cannot prove a target.
    uncertain=typed.model_copy(deep=True)
    uncertain.provenance['scene_image_candidates'].append({'object_id':'unknown-sign','category':'unknown','camera_name':1,'xyxy':[230,5,250,20]})
    uncertain,_=expand_relations(uncertain,typed_records)
    assert spatial_rule(uncertain,'image_left','sign').matches['right-car'] is None
    assert left.matches['sign-anchor'] is False
    # No color is mandatory for a recognized vehicle-heading landmark.
    colorless={oid:{**record,'attributes':[a for a in record['attributes'] if a['kind']!='color']} for oid,record in heading_records.items()}
    general_heading=heading.model_copy(deep=True);general_heading.task.update(spatial_relation_policy=spatial.LEGACY_POLICY,joint_reference_policy=JOINT_POLICY)
    general_heading,_=expand_relations(general_heading,colorless)
    assert any(r.predicate_id.startswith('image_joint_heading:') and 'a sedan' in r.phrase for r in general_heading.predicates)
    # Blind inputs stay invariant to target selection; both clauses need their own support.
    msgs,mapping,receipt=scene_grounding.messages(typed,proposal.expression,713)
    wire=json.loads(msgs[-1]['content'][0]['text']);reverse={oid:cid for cid,oid in mapping.items()}
    assert 'spatial_relations' in wire and 'native_vehicle_relations' not in wire
    assert typed.target_object_id not in json.dumps(msgs)
    swapped=typed.model_copy(deep=True);swapped.target=native['upper-car'];swapped.target_object_id='upper-car'
    assert scene_grounding.messages(swapped,proposal.expression,713)[0]==msgs
    tid=reverse[typed.target_object_id]
    answer=scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[tid],ambiguous_candidate_ids=[],unsupported_claims=[],reason='Both clauses resolve one subject.',
        relation_pairs=[{'candidate_id':tid,'landmark_id':reverse['sign-anchor'],'relation_key':'image_left'},
            {'candidate_id':tid,'landmark_id':reverse['person-anchor'],'relation_key':'image_below'}])
    check=lambda a:scene_grounding.verify(typed,a,mapping,receipt,generated=proposal.model_dump())
    assert check(answer)['diagnostic_pass']
    assert not check(answer.model_copy(update={'relation_pairs':answer.relation_pairs[:1]}))['diagnostic_pass']
    wrong=answer.model_copy(deep=True);wrong.relation_pairs[1].relation_key='lower_level'
    assert not check(wrong)['diagnostic_pass']
    wrong=answer.model_copy(deep=True);wrong.relation_pairs[1].landmark_id=reverse['upper-car']
    assert not check(wrong)['diagnostic_pass']
    plan={'predicate_ids':proposal.predicate_ids,'visual_attributes':[a.model_dump() for a in attrs],'unique_plan_found':True}
    sentence=fallback_sentence(typed,plan,typed_records)
    assert left.phrase.removesuffix(' in the front view') in sentence
    assert below.phrase.removesuffix(' in the front view') in sentence and ' is ' in sentence
    assert sentence.count('front view')==1
    # Preserve a proven conjunction instead of forcing one ambiguous relation.
    restricted=typed.model_copy(deep=True)
    restricted.predicates=[category,left,below]
    restricted.provenance['admission_plan']=plan
    restricted.task['manual_review_feedback']={'reason':'Use both spatial relations.'}
    assert writing_plan(restricted,typed_records)['predicate_ids']==proposal.predicate_ids
    assert spatial.requested_relations('位于 white van 的前方')=={'vehicle_ahead'}
    assert spatial.requested_relations('directly in front of a vehicle')=={'vehicle_directly_ahead'}
    assert spatial.requested_relations('位于标志牌左侧，同时在行人下方')=={'image_left','image_below'}
    preference=region.model_copy(deep=True)
    preference.task.update(spatial_relation_policy=spatial.LEGACY_POLICY,
        manual_review_feedback={'reason':'Use the relation in front of a red sedan.'})
    preference,_=expand_relations(preference,heading_records)
    # With equally unresolved candidates, honor the requested supported frame
    # instead of substituting an image-left relation with a shorter phrase.
    preference.predicates=[p for p in preference.predicates if p.predicate_id in {'category:vehicle',ahead_rule.predicate_id}]
    alternative=ahead_rule.model_copy(update={'predicate_id':'image_joint_spatial:1:image_left:preference',
        'value':json.dumps({'relation':'image_left','attributes':[{'kind':'color','value':'red'},{'kind':'subtype','value':'sedan'}]}),
        'phrase':'to the left of a red sedan in the front view.'})
    preference.predicates.append(alternative)
    assert ahead_rule.predicate_id in writing_plan(preference,heading_records)['predicate_ids']
    scoped=typed.model_copy(deep=True)
    scoped.task.pop('spatial_relation_policy')
    scoped,_=expand_relations(scoped,typed_records)
    assert not any(p.predicate_id.startswith('image_joint_spatial:') for p in scoped.predicates)
    # New annotation and rework use only native 3D geometry. Deliberately
    # contradictory pixels cannot create an above/below/left/right predicate.
    native_packet=typed.model_copy(deep=True)
    native_packet.task['spatial_relation_policy']=spatial.POLICY
    native_packet.predicates=predicate_catalog(native_packet.candidates)
    native_packet,_=expand_relations(native_packet,typed_records)
    from app.quality.description_search import draft_reference_predicate
    assert not any(p.predicate_id.startswith(('image_order:','image_anchor:','image_vehicle_heading:')) for p in native_packet.predicates)
    assert not any(p.predicate_id.startswith('image_joint_spatial:') and json.loads(p.value)['reference_frame']=='source_image'
        for p in native_packet.predicates)
    assert not draft_reference_predicate(native_packet,left)
    assert not draft_reference_predicate(native_packet,below)
    # Legacy axis proofs stay readable, but new writing must not confuse
    # longitudinal order with distance or a landmark's own heading.
    plain=native_packet.model_copy(deep=True)
    plain.task['spatial_language_policy']=spatial.PLAIN_SPATIAL_POLICY
    axis=left.model_copy(update={'predicate_id':'image_joint_spatial:1:native_behind:test'})
    legacy_axis=native_packet.model_copy(deep=True)
    legacy_axis.task.pop('spatial_language_policy',None)
    assert draft_reference_predicate(legacy_axis,axis)
    assert not draft_reference_predicate(plain,axis)
    plain,_=expand_relations(plain,typed_records)
    assert not any(p.predicate_id.startswith(tuple('image_joint_spatial:1:'+k+':'
        for k in ('native_left','native_right','native_ahead','native_behind'))) for p in plain.predicates)
    near=native[native_packet.target_object_id].model_copy(update={'center_m':(8.69,-3.74,0.95)})
    far=near.model_copy(update={'object_id':'far-car','center_m':(60.82,-0.14,1.15)})
    geometry={near.object_id:near,far.object_id:far}
    assert spatial.pair_match(spatial.BY_KEY['native_behind'],near.object_id,far.object_id,{},geometry) is True
    assert spatial.pair_match(spatial.BY_KEY['native_ahead'],near.object_id,far.object_id,{},geometry) is False
    assert spatial.pair_match(spatial.BY_KEY['native_below'],native_packet.target_object_id,'person-anchor',boxes,native) is None
    elevated={**native,'upper-car':native['upper-car'].model_copy(update={'center_m':(10,5,10)})}
    assert spatial.pair_match(spatial.BY_KEY['native_above'],'upper-car','person-anchor',boxes,elevated) is True
    elevated['upper-car']=elevated['upper-car'].model_copy(update={'center_m':(10,20,10)})
    assert spatial.pair_match(spatial.BY_KEY['native_above'],'upper-car','person-anchor',boxes,elevated) is None
    # #3261: adjacent thin sign panels can have an exact native stack proof
    # even though the historical broad 1m/inner-circle rule stays unknown.
    lower_panel=native_packet.target.model_copy(update={'category':'sign','center_m':(10,-4,2),
        'size_m':(.12,.6,.8),'heading_rad':0.,'lidar_points':20,'geometry_valid':True})
    upper_panel=lower_panel.model_copy(update={'object_id':'upper-panel','center_m':(10,-4,2.85),'size_m':(.15,1.4,.8)})
    panels={lower_panel.object_id:lower_panel,upper_panel.object_id:upper_panel}
    below=spatial.BY_KEY['native_below'];above=spatial.BY_KEY['native_above']
    assert spatial.pair_match(below,lower_panel.object_id,upper_panel.object_id,{},panels) is None
    assert spatial.pair_match(below,lower_panel.object_id,upper_panel.object_id,{},panels,sign_stack=True) is True
    assert spatial.pair_match(above,lower_panel.object_id,upper_panel.object_id,{},panels,sign_stack=True) is False
    for changed in ({'center_m':(10.5,-4,2.85)},{'center_m':(10,-4,2.6)},
            {'center_m':(10,-4,2.81)},{'heading_rad':math.pi/2},{'lidar_points':0},{'geometry_valid':False}):
        uncertain_panels={**panels,upper_panel.object_id:upper_panel.model_copy(update=changed)}
        assert spatial.pair_match(below,lower_panel.object_id,upper_panel.object_id,{},uncertain_panels,sign_stack=True) is None
    rotated={oid:o.model_copy(update={'heading_rad':math.pi/4}) for oid,o in panels.items()}
    assert spatial.pair_match(below,lower_panel.object_id,upper_panel.object_id,{},rotated,sign_stack=True) is True
    stack_packet=native_packet.model_copy(deep=True)
    stack_packet.target=lower_panel;stack_packet.target_object_id=lower_panel.object_id;stack_packet.candidates=list(panels.values())
    stack_packet.task.update(native_sign_stack_policy=spatial.SIGN_STACK_POLICY,sentence_style_policy='simple-sufficient-sentence-v1')
    panel_boxes=[{'object_id':oid,'category':'sign','camera_name':1,'xyxy':[10,10+i*40,60,40+i*40],
        'identity_source':'official_lidar_box_projection'} for i,oid in enumerate(panels)]
    stack_packet.provenance['scene_image_candidates']=panel_boxes
    panel_records={oid:{'target_visible':True,'visibility':'clear','visibility_reason':'Separately recognizable panel.',
        'observed_camera_name':1,'attributes':[{'kind':'color','value':'blue'},
            *([{'kind':'inscription','value':'POLICE'}] if oid==upper_panel.object_id else [])]} for oid in panels}
    panel_rules=spatial.catalog(stack_packet,1,panel_boxes,panel_records,list(panels))
    stack_rule=next(q for q in panel_rules if 'native_below:' in q.predicate_id and 'POLICE' in q.phrase)
    assert draft_reference_predicate(stack_packet,stack_rule)
    assert spatial.verified_pair(stack_packet,json.loads(stack_rule.value),lower_panel.object_id,upper_panel.object_id)
    old_stack=stack_packet.model_copy(deep=True);old_stack.task.pop('native_sign_stack_policy')
    assert not draft_reference_predicate(old_stack,stack_rule)
    assert not spatial.verified_pair(old_stack,json.loads(stack_rule.value),lower_panel.object_id,upper_panel.object_id)
    from app.agent.harness import image_scope_errors
    another='The blue sign below another sign reading POLICE in the front view.'
    assert 'declared_joint_relation_not_in_text' not in image_scope_errors(stack_packet,another,[stack_rule],check_order=True)
    assert 'declared_joint_relation_not_in_text' in image_scope_errors(stack_packet,another.replace('below ','above '),[stack_rule],check_order=True)
    assert 'declared_joint_relation_not_in_text' in image_scope_errors(old_stack,another,[stack_rule],check_order=True)
    stack_graph=spatial.scene_graph(stack_packet,1,{'A':lower_panel.object_id,'B':upper_panel.object_id})
    assert next(r for r in stack_graph['rows'] if r['relation_key']=='native_below' and r['landmark_id']=='B')['true_subject_ids']==['A']
    for key in ('native_left','native_right','native_ahead','native_behind','native_above','native_below'):
        assert spatial.pair_match(spatial.BY_KEY[key],'image:cyclist','person-anchor',boxes,native) is None
    # Native and image axes differ. A target to ego-right of the landmark
    # remains there after moving every image box to the opposite side.
    native_heading=heading.model_copy(deep=True)
    native_heading.task.pop('spatial_language_policy',None)  # Historical axis-proof coverage.
    native_heading.task.update(spatial_relation_policy=spatial.POLICY,joint_reference_policy=JOINT_POLICY,
        scene_grounding_policy=scene_grounding.POLICY,natural_description=True)
    native_heading,_=expand_relations(native_heading,heading_records)
    right=spatial_rule(native_heading,'native_right','vehicle')
    right=next(p for p in native_heading.predicates if p.predicate_id.startswith('image_joint_spatial:1:native_right:') and 'red sedan' in p.phrase)
    assert right.matches[native_heading.target_object_id] is True
    mirrored=native_heading.model_copy(deep=True)
    for b in mirrored.provenance['scene_image_candidates']:
        x,y,x2,y2=b['xyxy'];b['xyxy']=[1000-x2,1000-y2,1000-x,1000-y]
    mirrored,_=expand_relations(mirrored,heading_records)
    assert next(p for p in mirrored.predicates if p.predicate_id==right.predicate_id).matches==right.matches
    assert 'ego vehicle' in right.phrase
    msgs,mapping,receipt=scene_grounding.messages(native_heading,'The white sedan '+right.phrase+'.',801)
    graph=json.loads(msgs[-1]['content'][0]['text'])['spatial_relations']
    assert not any(d['reference_frame']=='source_image' for d in graph['definitions'])
    assert all(not row['relation_key'].startswith('image_') for row in graph['rows'])
    # Appearance-only language needs no invented landmark. New prompt scope is
    # target-independent; old packets keep the exact historical request.
    scoped=native_heading.model_copy(deep=True)
    scoped.task['scene_claim_policy']=scene_grounding.CLAIM_POLICY
    appearance_text='The white truck with a wooden cargo bed is visible in the front view.'
    appearance_claim={'predicate_ids':['category:vehicle'], 'visual_attributes':[]}
    for seed in (601,602):
        scoped_messages,scoped_map,scoped_receipt=scene_grounding.messages(scoped,appearance_text,seed)
        payload=json.loads(scoped_messages[-1]['content'][0]['text'])
        assert payload['claim_evaluation']['policy']==scene_grounding.CLAIM_POLICY
        assert 'No spatial relation, landmark' in scoped_messages[0]['content']
        assert 'Only IF the sentence asserts a landmark relation' in scoped_messages[0]['content']
        assert 'Resolve attributes AND relation jointly.' not in scoped_messages[0]['content']
        assert scoped_receipt['scene_claim_policy']==scene_grounding.CLAIM_POLICY
        swapped=scoped.model_copy(deep=True)
        swapped.target=next(o for o in swapped.candidates if o.object_id!=scoped.target_object_id)
        swapped.target_object_id=swapped.target.object_id
        assert scene_grounding.messages(swapped,appearance_text,seed)[0]==scoped_messages
        cid=next(cid for cid,oid in scoped_map.items() if oid==scoped.target_object_id)
        answer=scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[cid],
            ambiguous_candidate_ids=[],unsupported_claims=[],reason='All stated appearance details identify one object.')
        check=lambda a,g=appearance_claim:scene_grounding.verify(scoped,a,scoped_map,scoped_receipt,generated=g)
        assert check(answer)['diagnostic_pass']
        # Never reinterpret a model rejection as a pass based on its prose.
        rejected=answer.model_copy(update={'expression_supported':False,'candidate_ids':[],
            'ambiguous_candidate_ids':[cid],'unsupported_claims':[appearance_text]})
        assert not check(rejected)['diagnostic_pass']
        assert not check(answer.model_copy(update={'unsupported_claims':['wooden cargo bed']}))['diagnostic_pass']
        other=next(c for c in scoped_map if c!=cid)
        assert not check(answer.model_copy(update={'ambiguous_candidate_ids':[other]}))['diagnostic_pass']
        assert 'joint_reference_pair_missing' in check(answer,{'predicate_ids':[right.predicate_id]})['errors']
    legacy=native_heading.model_copy(deep=True)
    legacy.task.pop('scene_claim_policy',None)
    legacy_messages,_,legacy_receipt=scene_grounding.messages(legacy,appearance_text,601)
    assert 'claim_evaluation' not in json.loads(legacy_messages[-1]['content'][0]['text'])
    assert 'scene_claim_policy' not in legacy_receipt
    assert 'Resolve attributes AND relation jointly.' in legacy_messages[0]['content']
    # The same pair may be ego-right and vehicle-ahead. The sentence-only
    # index names subjects and landmarks explicitly without leaking feedback.
    indexed=native_heading.model_copy(deep=True)
    indexed.task['scene_relation_reading_policy']=spatial.READING_POLICY
    indexed.task['manual_review_feedback']={'reason':'private review hint'}
    text='The white sedan directly in front of a red sedan in the front view.'
    indexed_messages,indexed_map,indexed_receipt=scene_grounding.messages(indexed,text,801)
    index=json.loads(indexed_messages[-1]['content'][0]['text'])['sentence_relation_index']
    assert indexed_receipt['scene_relation_reading_policy']==spatial.READING_POLICY
    # Native spatial truth must never be delegated back to photo/lane impressions.
    authority = indexed.model_copy(deep=True)
    authority.task['scene_geometry_authority_policy'] = 'native-table-authority-v1'
    authority_inputs, _, authority_receipt = scene_grounding.messages(authority, joint_generated.expression, seed=609)
    assert authority_receipt['scene_geometry_authority_policy']=='native-table-authority-v1'
    assert 'PHOTOGRAPHS establish appearance and visible identity ONLY' in authority_inputs[0]['content']
    assert 'Do NOT judge any spatial assertion by apparent' in authority_inputs[0]['content']

    # The subject and landmark indices encode the exact same full relation table.
    table = spatial.sentence_relation_index({'definitions':[{'key':'vehicle_behind','reference_frame':'landmark_vehicle_heading','meaning':'behind'}],
        'checked_subject_ids':['V09','V02','V03'], 'rows':[
            {'relation_key':'vehicle_behind','landmark_id':'V02','true_subject_ids':['V09'],'unknown_subject_ids':['V03']},
            {'relation_key':'vehicle_behind','landmark_id':'V03','true_subject_ids':['V09'],'unknown_subject_ids':[]}]},
        'The black vehicle is behind a red car.')
    subject = next(r for r in table['by_subject'] if r['subject_id']=='V09')
    assert subject['true_landmark_ids']==['V02','V03']  # A second anchor does not negate the red anchor.
    assert next(r for r in table['by_subject'] if r['subject_id']=='V03')['unknown_landmark_ids']==['V02']
    assert all(r['relation_key']=='vehicle_behind' for r in table['by_subject'])

    assert {row['relation_key'] for row in index['rows']}=={'vehicle_directly_ahead'}
    reverse={v:k for k,v in indexed_map.items()}
    row=next(r for r in index['rows'] if r['landmark_id']==reverse[red.object_id])
    assert reverse[indexed.target_object_id] in row['true_subject_ids']
    assert f'{reverse[indexed.target_object_id]} is directly in front of {reverse[red.object_id]}' in ' '.join(row['true_statements'])
    assert 'private review hint' not in json.dumps(indexed_messages)
    swapped=indexed.model_copy(deep=True)
    swapped.target=next(o for o in swapped.candidates if o.object_id==red.object_id)
    swapped.target_object_id=red.object_id
    assert scene_grounding.messages(swapped,text,801)[0]==indexed_messages
    focused=indexed.model_copy(deep=True)
    focused.task['scene_input_policy']=scene_grounding.INPUT_POLICY
    focused.candidates.append(obj('unpictured-zero',100,100,points=0))
    focused_messages,focused_map,focused_receipt=scene_grounding.messages(focused,text,801)
    focused_wire=json.loads(focused_messages[-1]['content'][0]['text'])
    columns=focused_wire['unpictured_native_candidates']['columns']
    unpictured=[dict(zip(columns,row)) for row in focused_wire['unpictured_native_candidates']['rows']]
    assert {r['candidate_id'] for r in focused_wire['candidates']+unpictured}==set(focused_map)
    assert any(r['lidar_points']==0 for r in unpictured)
    assert all(r['candidate_id'].startswith('V') and r['image_boxes_xyxy'] for r in focused_wire['candidates'])
    assert all(r['candidate_id'].startswith('N') for r in unpictured)
    full_graph=spatial.scene_graph(focused,focused.views[0].camera_name,focused_map)
    keys={d['key'] for d in focused_wire['sentence_relation_index']['definitions']}
    assert focused_wire['spatial_relations']['rows']==[r for r in full_graph['rows'] if r['relation_key'] in keys]
    assert focused_wire['spatial_relations']['checked_subject_ids']==full_graph['checked_subject_ids']
    assert focused_wire['sentence_relation_index']==spatial.sentence_relation_index(full_graph,text)
    assert len(focused_wire['spatial_relations']['rows'])<len(full_graph['rows'])
    assert focused_receipt['scene_input_policy']==scene_grounding.INPUT_POLICY
    swapped=focused.model_copy(deep=True);swapped.target_object_id='other-sedan';swapped.target=swapped.candidates[1]
    assert scene_grounding.messages(swapped,text,801)[0]==focused_messages
    assert 'unpictured_native_candidates' not in json.loads(indexed_messages[-1]['content'][0]['text'])
    assert 'sentence_relation_index' not in json.loads(msgs[-1]['content'][0]['text'])
    assert spatial.sentence_relation_index(graph,appearance_text)['rows']==[]
    mixed=spatial.sentence_relation_keys('directly in front of a red car and in front of a van')
    assert mixed=={'vehicle_directly_ahead','vehicle_ahead'}
    assert spatial.sentence_relation_keys('farther behind than a red car relative to the ego vehicle')=={'native_behind'}
    assert spatial.sentence_relation_keys('ahead of the ego vehicle in the front view')==set()
    for original,index_row in [(r,next(x for x in index['rows'] if x['landmark_id']==r['landmark_id']))
            for r in graph['rows'] if r['relation_key']=='vehicle_directly_ahead']:
        assert original['unknown_subject_ids']==index_row['unknown_subject_ids']
    from app.agent.harness import generation_messages
    main_messages=generation_messages(native_heading,None,heading_records)
    main_payload=json.loads(main_messages[-1]['content'][0]['text'])
    assert 'native 3D' in main_payload['grounded_writing_brief']['spatial_evidence']
    assert 'NEVER use image above/below' in main_messages[0]['content']
    assert not any(f['predicate_id'].startswith(('image_anchor:','image_order:')) for f in main_payload['available_target_facts'])
    # A unique non-landmark plan wins over an ambiguous landmark both with
    # and without manual feedback; no spatial relation is mandatory.
    planning=native_heading.model_copy(deep=True)
    target_attrs=[{'kind':'color','value':'white'},{'kind':'subtype','value':'sedan'}]
    rank_free=next(p for p in planning.predicates if p.predicate_id=='lateral:left').model_copy(
        update={'matches':{o.object_id:o.object_id==planning.target_object_id for o in planning.candidates}})
    ambiguous_relation=right.model_copy(update={'matches':{o.object_id:True for o in planning.candidates}})
    planning.predicates=[next(p for p in planning.predicates if p.predicate_id=='category:vehicle'),rank_free,ambiguous_relation]
    planning.provenance['scene_image_candidates']=[]
    planning.provenance['admission_plan']={'unique_plan_found':True,'predicate_ids':['category:vehicle','lateral:left'],'visual_attributes':target_attrs}
    for feedback in ({},{'reason':'Use a spatial relation if helpful.'}):
        planning.task['manual_review_feedback']=feedback
        chosen=writing_plan(planning,heading_records)
        assert chosen['unique_plan_found'] and not chosen['joint_reference']
        assert chosen['predicate_ids']==['category:vehicle','lateral:left']
    from app.quality.joint_reference import FEEDBACK_POLICY,writer_contract_errors
    planning.task.update(manual_feedback_policy=FEEDBACK_POLICY,
        manual_review_feedback={'reason':'Use a red sedan relation directly in front of it.'})
    assert writing_plan(planning,heading_records)['unique_plan_found']
    assert not writing_plan(planning,heading_records)['joint_reference']
    feedback_packet=native_heading.model_copy(deep=True)
    next(o for o in feedback_packet.candidates if o.object_id=='other-sedan').center_m=feedback_packet.target.center_m
    feedback_packet.predicates=predicate_catalog(feedback_packet.candidates)
    feedback_packet,_=expand_relations(feedback_packet,heading_records)
    feedback_packet.provenance.pop('admission_plan',None)
    feedback_packet.task.update(manual_feedback_policy=FEEDBACK_POLICY,
        manual_review_feedback={'reason':'给出准确线索，比如 white sedan 在一个 red car 的正前方',
            'previous_expression':'The white sedan behind a van.'})
    feedback_plan=writing_plan(feedback_packet,heading_records)
    facts=[p for p in feedback_packet.predicates if p.predicate_id in feedback_plan['predicate_ids']]
    assert feedback_plan['feedback_alignment']['requested_relation_selected']
    assert any('directly in front of a red' in p.phrase for p in facts)
    assert any({(a['kind'],a['value']) for a in json.loads(p.value).get('attributes',[])}
        >= {('color','red'),('subtype','sedan')} for p in facts if p.predicate_id.startswith('image_joint_'))
    assert feedback_plan['requires_full_scene'] and not feedback_plan['unique_plan_found']
    scoped_feedback=feedback_packet.model_copy(deep=True)
    from app.schemas.contracts import Predicate
    scoped_feedback.task.update(feedback_relation_policy='bilingual-landmark-feedback-v1',
        feedback_acceptance_policy='feedback-acceptance-v5-natural-concision',
        manual_review_feedback={'reason':'可以说明在 red sedan 的正前方'})
    source_camera=next(v.camera_name for v in scoped_feedback.views if v.camera_label==scoped_feedback.source_view)
    scoped_feedback.predicates.append(Predicate(predicate_id=f'image_visibility:{source_camera}:vehicle',
        kind='image_relation', value=str(source_camera), phrase='visible in the front view',source='test',
        matches={o.object_id:o.object_id==scoped_feedback.target_object_id for o in scoped_feedback.candidates}))
    scoped_plan=writing_plan(scoped_feedback,heading_records)
    assert scoped_plan['unique_plan_found'] and scoped_plan['feedback_alignment']['requested_relation_selected']
    assert f'image_visibility:{source_camera}:vehicle' in scoped_plan['predicate_ids']
    intended_supported=scoped_feedback.model_copy(deep=True)
    intended_supported.task.update(feedback_intent_policy='purpose-bound-feedback-v1',
        manual_review_feedback={'reason':'你可以说明在 red sedan 的正前方，这样才能唯一定位物体'})
    supported_plan=writing_plan(intended_supported,heading_records)
    assert supported_plan and supported_plan['feedback_alignment']['requested_relation_selected']
    intended_supported.task['manual_review_feedback']['reason']='你可以说明在 green sedan 的正前方，这样才能唯一定位物体'
    assert writing_plan(intended_supported,heading_records) is None
    from app.quality.joint_reference import SIMPLE_SENTENCE_POLICY,simple_sentence_errors
    simple_packet=scoped_feedback.model_copy(deep=True)
    simple_packet.task.update(sentence_style_policy=SIMPLE_SENTENCE_POLICY,manual_review_feedback={})
    simple_plan=writing_plan(simple_packet,heading_records)
    assert simple_plan['unique_plan_found'] and not simple_plan['joint_reference']
    assert not any(i.startswith(('longitudinal:','lateral:')) for i in simple_plan['predicate_ids'])
    ambiguity_revision=simple_packet.model_copy(deep=True)
    ambiguity_revision.task.update(distinctiveness_search_policy='distinctiveness-evidence-search-v1',
        manual_review_feedback={'reason':'区分特征不够','rejected_appearance_bundle':simple_plan['visual_attributes']})
    new_cue=writing_plan(ambiguity_revision,heading_records)
    assert new_cue is None or not all(i.startswith(('category:','image_visibility:','image_attribute:'))
        for i in new_cue['predicate_ids']) or not set((a['kind'],a['value']) for a in new_cue['visual_attributes']) <= set(
            (a['kind'],a['value']) for a in simple_plan['visual_attributes'])
    lengthy="In the front-left view, the pedestrian wearing a red top, black pants, and holding a handheld bag is farther behind than the pedestrian in the black jacket relative to the ego vehicle's orientation."
    assert simple_sentence_errors(lengthy)==['description_too_long','technical_spatial_comparison']
    assert not simple_sentence_errors('The pedestrian in a red top in the front-left view.')
    assert not simple_sentence_errors('The red car behind the ego vehicle.')
    assert not simple_sentence_errors('The red car ahead and to the left of the ego vehicle.')
    from app.quality.description_search import draft_reference_predicate
    for simple_rule in simple_packet.predicates:
        if simple_rule.predicate_id.startswith('image_joint_spatial:'):
            assert not draft_reference_predicate(simple_packet,simple_rule)
    payload=json.loads(writer_messages(feedback_packet,feedback_plan,None,heading_records)[-1]['content'][0]['text'])
    assert payload['human_feedback']==feedback_packet.task['manual_review_feedback']['reason']
    assert payload['reviewed_sentence']=='The white sedan behind a van.'
    assert 'Do not add, reverse' in payload['writing_scope']
    assert writer_contract_errors(feedback_packet,feedback_plan,'The white sedan directly behind a red car.')
    assert not writer_contract_errors(feedback_packet,feedback_plan,fallback_sentence(feedback_packet,feedback_plan,heading_records))
    no_relation={'predicate_ids':['category:vehicle']}
    assert writer_contract_errors(feedback_packet,no_relation,'The white sedan behind a red car.')
    assert not writer_contract_errors(feedback_packet,no_relation,'The white sedan in the front view.')
    # #887: visibility modifies the subject, not another vehicle. Both writer
    # inputs and the low-budget fallback must preserve one shared camera scope.
    from app.quality.joint_reference import COMPOSITION_POLICY, composition_bundle
    composed=scoped_feedback.model_copy(deep=True)
    composed.task['sentence_composition_policy']=COMPOSITION_POLICY
    composed_sentence=fallback_sentence(composed,scoped_plan,heading_records)
    assert composed_sentence and composed_sentence.count('front view')==1
    assert 'visible in the front view is ' in composed_sentence
    assert 'and the vehicle' not in composed_sentence
    assert not writer_contract_errors(composed,scoped_plan,composed_sentence)
    composed_payload=json.loads(writer_messages(composed,scoped_plan,None,heading_records)[-1]['content'][0]['text'])
    assert 'visible in the front view' in composed_payload['required_meaning']
    assert not any(c.startswith('the vehicle visible') for c in composed_payload['required_meaning'])
    assert 'ONE target subject' in composed_payload['composition_instruction']
    malformed='The white sedan is directly in front of a red sedan in the front view and the vehicle visible in the front view.'
    malformed_errors=writer_contract_errors(composed,scoped_plan,malformed)
    assert 'detached_visibility_subject' in malformed_errors
    assert 'repeated_source_camera_wording' in malformed_errors
    assert composition_bundle(composed,scoped_plan['predicate_ids'])==composition_bundle(composed,
        [i for i in scoped_plan['predicate_ids'] if not i.startswith('image_visibility:')])
    assert 'image_visibility:5:vehicle' in composition_bundle(composed,[*scoped_plan['predicate_ids'],'image_visibility:5:vehicle'])
    appearance_ids=['category:vehicle',f'image_visibility:{source_camera}:vehicle']
    composed.task['source_camera_scope_policy']='appearance-source-scope-v1'
    scoped_attrs=scoped_plan['visual_attributes']
    assert composition_bundle(composed,appearance_ids,scoped_attrs)==composition_bundle(composed,
        ['category:vehicle'],scoped_attrs,expression='A white sedan in the front view.')
    assert composition_bundle(composed,appearance_ids,scoped_attrs)!=composition_bundle(composed,
        appearance_ids,scoped_attrs,expression='A white sedan.')
    assert composition_bundle(composed,appearance_ids)!={'category:vehicle'}
    # Frozen old runs keep their original strategy; no new guard is retroactive.
    frozen_composition=scoped_feedback.model_copy(deep=True)
    frozen_composition.task.pop('sentence_composition_policy',None)
    assert composition_bundle(frozen_composition,scoped_plan['predicate_ids'])==set(scoped_plan['predicate_ids'])
    # #1176: camera-scoped appearance must reach whole-scene checks even
    # before symbolic uniqueness can yield an image_attribute predicate.
    from app.agent.harness import draft_reference_errors
    from app.agent.scene_grounding import available as scene_available
    source_only = feedback_packet.model_copy(deep=True)
    source_only.task['source_camera_scope_policy'] = 'appearance-source-scope-v1'
    scoped_draft = GeneratedExpression(expression='The white sedan in the front view.',
        predicate_ids=['category:vehicle'], visual_attributes=target_attrs)
    assert scene_available(source_only)
    assert 'camera_qualifier_without_image_relation' not in draft_reference_errors(source_only,scoped_draft,heading_records)
    for invalid in ('legacy', 'wrong-camera', 'no-rgb', 'no-scene', 'no-attributes'):
        checked = source_only.model_copy(deep=True)
        draft = scoped_draft.model_copy(deep=True)
        if invalid == 'legacy': checked.task.pop('source_camera_scope_policy')
        if invalid == 'wrong-camera': draft.expression = 'The white sedan in the left-side view.'
        if invalid == 'no-rgb': checked.task['training_input_capabilities'] = ['geometry']
        if invalid == 'no-scene': checked.provenance['scene_image_candidates'] = []
        if invalid == 'no-attributes': draft.visual_attributes = []
        assert 'camera_qualifier_without_image_relation' in draft_reference_errors(checked,draft,heading_records)
    from app.quality.joint_reference import REVISION_POLICY,ego_requests
    ego_packet=native_heading.model_copy(deep=True)
    ego_packet.task.update(manual_feedback_policy=FEEDBACK_POLICY,feedback_revision_policy=REVISION_POLICY,
        manual_review_feedback={'reason':'缺少区分线索和相对空间位置','previous_expression':'The white sedan visible in the front view.'})
    ego_plan=writing_plan(ego_packet,heading_records)
    assert ego_plan and any(i.startswith(('longitudinal:','lateral:')) for i in ego_plan['predicate_ids'])
    assert ego_plan['feedback_alignment']['requested_spatial_cue']
    assert 'declared_ego_direction_not_in_text' in writer_contract_errors(ego_packet,ego_plan,'The white sedan visible in the front view.')
    assert not writer_contract_errors(ego_packet,ego_plan,fallback_sentence(ego_packet,ego_plan,heading_records))
    ego_packet.task['manual_review_feedback']['reason']='自车前方的一辆车'
    ego_plan=writing_plan(ego_packet,heading_records)
    assert 'longitudinal:front' in ego_plan['predicate_ids']
    assert ego_plan['feedback_alignment']['requested_relation_keys']==[]
    assert ego_plan['feedback_alignment']['requested_ego_predicate_ids']==['longitudinal:front']
    from app.quality.joint_reference import ACCEPTANCE_POLICY, feedback_acceptance, feedback_requirements
    from app.schemas.contracts import Predicate
    # #2984: a negated white landmark must never earn positive feedback credit.
    from app.quality.joint_reference import feedback_relations, landmark_salience, SALIENCE_POLICY, CONCISION_POLICY
    correction = {'feedback_acceptance_policy': ACCEPTANCE_POLICY, 'manual_review_feedback': {
        'reason': '明显是 behind a red car 啊，behind a white sedan是错误的'}}
    signed = feedback_relations(correction)
    assert len(signed) == 2 and not signed[0]['denied'] and signed[1]['denied']
    assert signed[0]['attributes'] == [{'kind':'color','value':'red'}]
    rejected = feedback_acceptance(correction, {'expression': 'The black vehicle is behind a white sedan in the front view.'})
    assert rejected['blocking_errors'] == ['manual_feedback_landmark_missing', 'manual_feedback_rejected_relation_repeated']
    assert feedback_acceptance(correction, {'expression': 'The black vehicle is behind a red sedan in the front view.'})['passed']
    assert 'manual_feedback_landmark_missing' in feedback_acceptance(correction, {'expression': 'The black vehicle is behind an orange car.'})['blocking_errors']
    assert not feedback_relations({**correction,'feedback_acceptance_policy':CONCISION_POLICY})
    # Mixed Chinese relation / English landmark must reach acquisition and
    # acceptance, without turning a suggestion into an observed fact.
    from app.quality.joint_reference import RELATION_FEEDBACK_POLICY
    bilingual = {**correction, 'feedback_relation_policy':RELATION_FEEDBACK_POLICY,
        'manual_review_feedback':{'reason':'可以说明 这个 silver sedan 在 silver SUV的 后方'}}
    request = feedback_relations(bilingual)[0]
    assert request['relation']=='vehicle_behind' and not request['required']
    assert request['attributes']==[{'kind':'color','value':'silver'},{'kind':'subtype','value':'suv'}]
    omitted = feedback_acceptance(bilingual, {'expression':'The silver sedan visible in the front-left view'})
    assert omitted['passed'] is False and not omitted['blocking_errors']
    assert omitted['requirements'][0]['status']=='missing'
    assert feedback_acceptance(bilingual, {'expression':'The silver sedan behind a silver SUV.'})['passed']
    assert not feedback_relations({k:v for k,v in bilingual.items() if k!='feedback_relation_policy'})
    spatial_feedback={**bilingual,'manual_review_feedback':{'reason':
        '你可以说 A white rectangular sign 在绿色 rectangular sign的下方，这样才能无歧义的唯一定位物体'}}
    spatial_request=feedback_relations(spatial_feedback)
    assert len(spatial_request)==1 and spatial_request[0]['relation']=='native_below'
    assert not spatial_request[0]['required'] and not spatial_request[0]['denied']
    from app.quality.joint_reference import FEEDBACK_INTENT_POLICY
    intended={**spatial_feedback,'feedback_intent_policy':FEEDBACK_INTENT_POLICY}
    assert feedback_relations(intended)[0]['required']
    assert feedback_acceptance(intended,{'expression':'A white rectangular sign reading BUS LANE ENDS.'})['blocking_errors']==['manual_feedback_landmark_missing']
    assert feedback_acceptance(intended,{'expression':'The white sign below a green rectangular sign.'})['passed']
    for reason in ['可以说明在绿色矩形标志牌的下方',
            '比如可以说明在绿色矩形标志牌的下方，这样才能唯一定位',
            '如果确认有绿色标志牌，可以说明在绿色矩形标志牌的下方，这样才能唯一定位',
            '可以说明在绿色矩形标志牌的下方。这样才能唯一定位是另外一个要求',
            '可以说明在绿色矩形标志牌的下方，可以说明在白色矩形标志牌的上方，这样才能唯一定位']:
        requests=feedback_relations({**intended,'manual_review_feedback':{'reason':reason}})
        assert not requests[0]['required']
        if len(requests)>1:
            assert requests[1]['required']
    assert spatial_request[0]['attributes']==[{'kind':'color','value':'green'},{'kind':'shape','value':'rectangular'}]
    ignored=feedback_acceptance(spatial_feedback,{'expression':'A white rectangular sign reading BUS LANE ENDS.'})
    assert not ignored['passed'] and not ignored['blocking_errors']
    assert ignored['requirements'][0]['status']=='missing' and '未采用' in ignored['requirements'][0]['reason']
    assert feedback_acceptance(spatial_feedback,{'expression':'The white sign below a green rectangular sign.'})['passed']
    assert not feedback_acceptance(spatial_feedback,{'expression':'The white sign below a green round sign.'})['passed']
    assert not feedback_relations({**spatial_feedback,'feedback_relation_policy':'bilingual-landmark-feedback-v2-omissions'})
    for reason,key,denied in [('必须说明在绿色矩形标志牌的下方','native_below',False),
            ('不要写 below a green rectangular sign','native_below',True),
            ('可以说明在 white rectangular sign 的上方','native_above',False)]:
        parsed=feedback_relations({**spatial_feedback,'manual_review_feedback':{'reason':reason}})
        assert len(parsed)==1 and parsed[0]['relation']==key and parsed[0]['denied']==denied
    for reason, key, denied in [('必须说明在 silver SUV 的后方','vehicle_behind',False),
            ('可以考虑，但必须说明在 silver SUV 的正前方','vehicle_directly_ahead',False),
            ('不在 silver SUV 的后方','vehicle_behind',True),
            ('在 silver SUV 的前面是错误的','vehicle_ahead',True)]:
        task={**bilingual,'manual_review_feedback':{'reason':reason}}
        parsed=feedback_relations(task)[0]
        assert parsed['relation']==key and parsed['denied']==denied and parsed['required']
        if not denied:
            assert 'manual_feedback_landmark_missing' in feedback_acceptance(task,{'expression':'The silver sedan.'})['blocking_errors']
    for reason in ('not behind a white sedan but behind a red car', '不是behind a white sedan，而是behind a red car'):
        parsed = feedback_relations({**correction, 'manual_review_feedback': {'reason': reason}})
        assert parsed[0]['denied'] and not parsed[1]['denied']
    for prefix in ('不需要 有 ', '无需使用 ', '不要写 ', '删除 ', '去掉 ',
            'do not include ', 'no need to use ', 'remove '):
        omission_task = {**bilingual,'manual_review_feedback':{'reason':prefix+'in front of a sedan in the front view'}}
        parsed = feedback_relations(omission_task)
        assert len(parsed)==1 and parsed[0]['denied'] and parsed[0]['required']
        repeated = feedback_acceptance(omission_task,{'expression':
            'The pedestrian in a black top is in front of a silver sedan in the front view.'})
        assert repeated['blocking_errors']==['manual_feedback_rejected_relation_repeated']
        assert feedback_acceptance(omission_task,{'expression':'The pedestrian in a black top in the front view.'})['passed']
    omission_task['manual_review_feedback']['reason']='不需要 有 in front of a sedan in the front view'
    frozen_omission = {**omission_task,'feedback_relation_policy':'bilingual-landmark-feedback-v1'}
    assert not feedback_relations(frozen_omission)[0]['denied']
    omission_plan_packet = scoped_feedback.model_copy(deep=True)
    omission_plan_packet.task.update(omission_task)
    omission_plan = writing_plan(omission_plan_packet,heading_records)
    assert omission_plan and not feedback_acceptance(omission_task,{'expression':
        ' and '.join(p.phrase for p in omission_plan_packet.predicates if p.predicate_id in omission_plan['predicate_ids'])})['blocking_errors']
    for reason in ('不要删除 in front of a sedan', '不要省略 in front of a sedan', 'do not omit in front of a sedan'):
        retained_task={**bilingual,'manual_review_feedback':{'reason':reason}}
        assert not feedback_relations(retained_task)[0]['denied']
    # Ranking may prefer alignment without narrowing the weak behind predicate.
    salience = native_heading.model_copy(deep=True)
    salience.task['landmark_salience_policy'] = SALIENCE_POLICY
    salience.target.center_m = (0,0,1)
    close = salience.target.model_copy(update={'object_id':'near-anchor','center_m':(15,0,1),'heading_rad':0,'lidar_points':100})
    remote = close.model_copy(update={'object_id':'side-anchor','center_m':(20,15,1)})
    salience.candidates += [close,remote]
    def salience_rule(anchor):
        return Predicate(predicate_id='image_joint_heading:1:rearward:'+anchor,kind='image_relation',
            value=json.dumps({'direction':'rearward','confirmed_anchor_ids':[anchor]}),phrase='behind a vehicle',
            matches={},source='test',required_capability='rgb')
    assert landmark_salience(salience,[salience_rule('near-anchor')]) < landmark_salience(salience,[salience_rule('side-anchor')])
    from app.quality.relations import LOCAL_LANDMARK_POLICY,local_landmark_predicate,local_landmark_pair
    local=salience.model_copy(deep=True)
    local.task['landmark_locality_policy']=LOCAL_LANDMARK_POLICY
    nearby_rule=salience_rule('near-anchor');distant_rule=salience_rule('side-anchor')
    before=distant_rule.model_dump()
    assert local_landmark_predicate(local,nearby_rule)
    assert not local_landmark_predicate(local,distant_rule)
    assert distant_rule.model_dump()==before  # No rewriting truth/competitor tables.
    mixed=nearby_rule.model_copy(update={'value':json.dumps({'direction':'rearward',
        'confirmed_anchor_ids':['near-anchor','side-anchor']})})
    assert not local_landmark_predicate(local,mixed)
    assert local_landmark_predicate(salience,distant_rule)  # Frozen historical behavior.
    assert local_landmark_pair(local,local.target,close)
    assert not local_landmark_pair(local,local.target,remote)
    assert not local_landmark_pair(local,local.target,close.model_copy(update={'lidar_points':0}))
    from app.agent.appearance import feedback_landmark_manifest, FEEDBACK_ACQUISITION_POLICY
    salience.task.update(correction, feedback_acquisition_policy=FEEDBACK_ACQUISITION_POLICY)
    source_camera = next(v.camera_name for v in salience.views if v.camera_label == salience.source_view)
    salience.provenance['appearance_candidates'] = [
        {'object_id':o.object_id,'candidate_id':str(i),'camera_name':source_camera}
        for i,o in enumerate([remote,close,salience.target])]
    manifest = feedback_landmark_manifest(salience)
    assert manifest[0]['object_id'] == 'near-anchor'
    assert all(m['object_id'] != salience.target_object_id for m in manifest)
    assert all(m['attribute_recheck'] == ['color'] for m in manifest)
    assert 'red' not in json.dumps(manifest)  # No desired answer leaks into anonymous observation.
    assert 'attribute_recheck' not in salience.provenance['appearance_candidates'][0]
    salience.task.update(bilingual)
    mixed_manifest=feedback_landmark_manifest(salience)
    assert mixed_manifest and all(m['attribute_recheck']==['color','subtype'] for m in mixed_manifest)
    assert 'silver' not in json.dumps(mixed_manifest)
    vertical_feedback=salience.model_copy(deep=True)
    vertical_feedback.task.update(spatial_feedback)
    vertical_feedback.target.category='sign'
    vertical_feedback.target.center_m=(10,0,1)
    vertical_feedback.target.size_m=(0.2,1,0.5)
    upper=vertical_feedback.target.model_copy(update={'object_id':'upper-sign','center_m':(10,0,4)})
    unaligned=upper.model_copy(update={'object_id':'unaligned-sign','center_m':(12,0,4)})
    unknown_upper=upper.model_copy(update={'object_id':'unknown-sign','lidar_points':0})
    vertical_feedback.candidates=[vertical_feedback.target,upper,unaligned,unknown_upper]
    vertical_feedback.provenance['appearance_candidates']=[{'object_id':o.object_id,'camera_name':source_camera}
        for o in vertical_feedback.candidates]
    vertical_manifest=feedback_landmark_manifest(vertical_feedback)
    assert [m['object_id'] for m in vertical_manifest]==['upper-sign']
    assert vertical_manifest[0]['attribute_recheck']==['color','shape']
    assert 'green' not in json.dumps(vertical_manifest) and 'rectangular' not in json.dumps(vertical_manifest)
    # Acquisition is a request for observations, never a new attribute or a
    # geometric proof. Zero-point and horizontally unaligned signs stay unknown.
    assert not vertical_feedback.task.get('observed_landmark_attributes')
    assert spatial.pair_match(spatial.BY_KEY['native_below'],vertical_feedback.target_object_id,
        unknown_upper.object_id,{}, {o.object_id:o for o in vertical_feedback.candidates}) is None
    from app.agent.appearance import feedback_landmark_report,explain_feedback_acquisition,FEEDBACK_ACQUISITION_REPORT_POLICY
    vehicle_report=feedback_packet.model_copy(deep=True)
    vehicle_report.task.update(feedback_acquisition_policy=FEEDBACK_ACQUISITION_REPORT_POLICY,
        chinese_landmark_policy='literal-chinese-landmarks-v1',
        feedback_acceptance_policy='feedback-acceptance-v5-natural-concision',
        feedback_relation_policy='bilingual-landmark-feedback-v3-spatial-details',
        manual_review_feedback={'reason':'可以在一个黑色车的后面'})
    vehicle_report.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':'vehicle','camera_name':source_camera}
        for o in vehicle_report.candidates]
    before=vehicle_report.model_dump()
    vehicle_acquisition=feedback_landmark_report(vehicle_report)
    assert vehicle_acquisition['requests'][0]['relation']=='vehicle_behind'
    assert vehicle_report.model_dump()==before  # Diagnostics never assert a desired color or relation.
    vehicle_report.candidates=[o.model_copy(update={'lidar_points':0}) for o in vehicle_report.candidates]
    unknown_acquisition=feedback_landmark_report(vehicle_report)['requests'][0]
    assert unknown_acquisition['native_supported_count']==0
    assert unknown_acquisition['unknown_count']==len(vehicle_report.candidates)-1
    vertical_feedback.task['feedback_acquisition_policy']=FEEDBACK_ACQUISITION_REPORT_POLICY
    vertical_feedback.provenance['scene_image_candidates']=[{'object_id':o.object_id,'category':'sign','camera_name':source_camera}
        for o in vertical_feedback.candidates]
    acquisition=feedback_landmark_report(vertical_feedback)
    assert acquisition['requests'][0]['native_supported_count']==1
    assert acquisition['requests'][0]['unknown_count']==2  # Unaligned and zero-point units retained.
    vertical_feedback.candidates=[o for o in vertical_feedback.candidates if o.object_id!=upper.object_id]
    acquisition=feedback_landmark_report(vertical_feedback)
    assert acquisition['requests'][0]['native_supported_count']==0
    assert acquisition['requests'][0]['unknown_count']==3  # Missing native ID remains unknown too.
    explained=explain_feedback_acquisition(ignored,acquisition)
    assert explained['requirements'][0]['acquisition_status']=='native_relation_unresolved'
    assert '3 个关系未知' in explained['requirements'][0]['reason']
    assert explained['passed']==ignored['passed'] and explained['blocking_errors']==ignored['blocking_errors']
    historical_acquisition=vertical_feedback.model_copy(deep=True)
    historical_acquisition.task['feedback_acquisition_policy']=FEEDBACK_ACQUISITION_POLICY
    assert feedback_landmark_report(historical_acquisition) is None
    frozen = salience.model_copy(deep=True)
    frozen.task.pop('feedback_acquisition_policy')
    assert not feedback_landmark_manifest(frozen)
    from app.quality.joint_reference import disputed_attribute_kinds
    attribute_failure = {'generated':{'visual_attributes':[{'kind':'color','value':'black'}]},
        'ground_checks':[{'unsupported_claims':['The subject is a truck but it is not black; it appears grey or blue.']}]}
    assert disputed_attribute_kinds(attribute_failure)=={'color'}
    pants_failure={'generated':{'visual_attributes':[{'kind':'upper_clothing_color','value':'black'},
        {'kind':'lower_clothing_color','value':'blue'}]},'ground_checks':[{'unsupported_claims':['blue pants']}]}
    assert not disputed_attribute_kinds(pants_failure)
    assert disputed_attribute_kinds(pants_failure,literal_spans=True)=={'lower_clothing_color'}
    for unsupported in ('blue pants for V15','not blue pants for V15','The other person has blue pants','blue pants behind a car'):
        assert not disputed_attribute_kinds({**pants_failure,'ground_checks':[{'unsupported_claims':[unsupported]}]},literal_spans=True)
    relation_failure = {**attribute_failure,'ground_checks':[{'unsupported_claims':['The black vehicle is not behind the red car.']}]}
    assert not disputed_attribute_kinds(relation_failure)
    repair_packet = feedback_packet.model_copy(deep=True)
    repair_packet.task['attribute_repair_policy']='disputed-attribute-omission-v1'
    first_plan = writing_plan(repair_packet,heading_records)
    old_color = next(a['value'] for a in first_plan['visual_attributes'] if a['kind']=='color')
    failed = {'generated':first_plan, 'ground_checks':[{'unsupported_claims':['The subject is not '+old_color+'.']}],
        'verification':{'diagnostic_pass':False,'errors':['scene_grounding_unsupported_claims']}}
    repaired = writing_plan(repair_packet,heading_records,attempts=[failed])
    assert repaired and repaired['omitted_disputed_attribute_kinds']==['color']
    assert not any(a['kind']=='color' for a in repaired['visual_attributes'])
    assert repaired['joint_reference'] and repaired['requires_full_scene']  # Cannot bypass fresh whole-scene checks.
    literal_repair=repair_packet.model_copy(deep=True);literal_repair.task['disputed_attribute_policy']='literal-disputed-attribute-v1'
    span_failed={**failed,'ground_checks':[{'unsupported_claims':['color:'+old_color]}]}
    span_plan=writing_plan(literal_repair,heading_records,attempts=[span_failed])
    assert span_plan and span_plan['omitted_disputed_attribute_kinds']==['color']
    assert not any(a['kind']=='color' for a in span_plan['visual_attributes'])


    # #5665: concision is an actionable rewrite constraint, not unrecognized feedback.
    from app.quality.joint_reference import SUGGESTIONS_POLICY, concision_budget
    concise_task = {'feedback_acceptance_policy': ACCEPTANCE_POLICY, 'manual_review_feedback': {
        'reason': '文本描述请简洁一些',
        'previous_expression': "In the front-left view, the pedestrian wearing a brown coat and holding a bag is located farther behind than the pedestrian in the black top relative to the ego vehicle's orientation."}}
    assert feedback_requirements(concise_task)['required_concise']
    assert concision_budget(concise_task) == 24
    long_draft = {'expression': ' '.join(['word'] * 29)}
    assert feedback_acceptance(concise_task, long_draft)['blocking_errors'] == ['manual_feedback_concision_missing']
    assert not feedback_acceptance(concise_task, long_draft, check_style=False)['blocking_errors']
    assert feedback_acceptance(concise_task, {'expression': 'The pedestrian in a brown top in the front-left view.'})['passed']
    for reason in ('不要缩短文本', '无需精简', 'do not shorten the text'):
        assert not feedback_requirements({**concise_task, 'manual_review_feedback': {'reason': reason}})['concise']
    assert not feedback_requirements({**concise_task, 'feedback_acceptance_policy': SUGGESTIONS_POLICY})['concise']
    brief_packet = ego_packet.model_copy(deep=True)
    brief_packet.task.update(concise_task)
    brief_plan = writing_plan(brief_packet, heading_records)
    assert brief_plan and brief_plan['unique_plan_found']
    brief_payload = json.loads(writer_messages(brief_packet, brief_plan, None, heading_records)[-1]['content'][0]['text'])
    assert 'at most 24' in brief_payload['concision_instruction']
    # #1176: natural brevity must not reward dropping articles or adding ego
    # clauses after a purely linguistic failure. Legacy percentage stays frozen.
    from app.quality.joint_reference import NATURAL_CONCISION_POLICY
    from app.agent.harness import expression_language_errors
    assert 'technical_ego_axis_comparison' in expression_language_errors(
        "The silver vehicle in the front view is farther behind than a silver vehicle relative to the ego vehicle's orientation.")
    natural_task = {**concise_task, 'feedback_acceptance_policy': NATURAL_CONCISION_POLICY}
    detail_task = {'feedback_acceptance_policy':NATURAL_CONCISION_POLICY,
        'feedback_detail_policy':'explicit-feedback-details-v1', 'manual_review_feedback':{
            'reason':'仅保留 The white vehicle in the left-side view 即可，然后说明与自车 距离很近'}}
    detail = feedback_acceptance(detail_task)
    assert detail['blocking_errors'] == ['manual_feedback_distance_unsupported']
    enabled_detail = {**detail_task,'ego_proximity_policy':'native-ego-proximity-v1'}
    assert not feedback_acceptance(enabled_detail)['blocking_errors']
    assert feedback_acceptance(enabled_detail,{'expression':'The white vehicle in the left-side view is close to the ego vehicle.',
        'predicate_ids':['category:vehicle','range:ego_near']})['passed']
    assert 'manual_feedback_distance_missing' in feedback_acceptance(enabled_detail,{
        'expression':'The white vehicle in the left-side view is to the left of the ego vehicle.',
        'predicate_ids':['category:vehicle','lateral:left']})['blocking_errors']
    assert {r['kind'] for r in detail['requirements']} == {'ego_distance','retained_phrase'}
    assert 'manual_feedback_distance_unsupported' in feedback_acceptance(detail_task, {
        'expression':'The white vehicle in the left-side view is to the left of the ego vehicle.',
        'predicate_ids':['lateral:left']})['blocking_errors']
    assert not feedback_acceptance({**detail_task,'feedback_detail_policy':None})['requirements']
    # #4826/#4102: generic color requests and explicit back-surface corrections
    # must constrain the plan and literal sentence, without manufacturing facts.
    from app.quality.joint_reference import feedback_target_appearance, APPEARANCE_FEEDBACK_POLICY
    color_task={'feedback_acceptance_policy':NATURAL_CONCISION_POLICY,
        'feedback_appearance_policy':APPEARANCE_FEEDBACK_POLICY,
        'manual_review_feedback':{'reason':'说明标注牌的颜色'}}
    assert feedback_target_appearance(color_task)==[{'kind':'color','value':None,'required':True,'label':'目标颜色'}]
    assert feedback_acceptance(color_task)['requirements'][0]['status']=='pending'
    assert 'manual_feedback_appearance_missing' in feedback_acceptance(color_task,{
        'expression':'The rectangular sign.','visual_attributes':[{'kind':'shape','value':'rectangular'}]})['blocking_errors']
    white={'expression':'The white rectangular sign.','visual_attributes':[{'kind':'color','value':'white'}]}
    assert feedback_acceptance(color_task,white)['passed']
    back_task={**color_task,'manual_review_feedback':{'reason':'表达不清晰，应该是只能看到这个标注牌的背面，是灰色的'}}
    assert len(feedback_target_appearance(back_task))==2
    assert 'manual_feedback_appearance_missing' in feedback_acceptance(back_task,white)['blocking_errors']
    gray_back={'expression':'The gray back of the rectangular sign in the front view.',
        'visual_attributes':[{'kind':'color','value':'gray'},{'kind':'feature','value':'visible rear face'}]}
    assert feedback_acceptance(back_task,gray_back)['passed']
    assert feedback_acceptance(back_task,{**gray_back,'expression':'The gray rectangular sign seen from behind.'})['passed']
    assert feedback_acceptance(back_task,{**gray_back,'expression':"The rectangular sign's gray back."})['passed']
    surface_wording=feedback_packet.model_copy(deep=True)
    surface_wording.target.category='sign'
    surface_wording.task.update(back_task)
    surface_wording.task['target_naming_policy']='explicit-target-category-v1'
    from app.quality.joint_reference import target_category_errors
    assert not target_category_errors(surface_wording,gray_back['expression'])
    assert target_category_errors(surface_wording,'The gray object behind a rectangular sign.')
    surface_wording.task.pop('feedback_appearance_policy')
    assert target_category_errors(surface_wording,gray_back['expression'])  # Frozen old parser stays strict.
    assert not feedback_acceptance({**back_task,'feedback_appearance_policy':None})['requirements']
    assert not feedback_target_appearance({**color_task,'manual_review_feedback':{'reason':'不要写颜色'}})
    assert not feedback_target_appearance({**color_task,'manual_review_feedback':{'reason':'在红色轿车后方'}})
    from app.quality.joint_reference import feedback_attribute_kinds
    mounting={**color_task,'manual_review_feedback':{'reason':'形状不对，你可以说它挂在 xx 杆子上，或柱子上'}}
    requests=feedback_target_appearance(mounting)
    assert requests==[{'kind':'feature','value':'pole mounted','required':False,'label':'安装在杆或柱上'}]
    assert feedback_attribute_kinds(mounting)==['feature','shape']
    # #3847: the native sign is a whole crossbuck. A rectangle around it or
    # the rectangular arms must not force rectangular into the schema/plan.
    from app.quality.joint_reference import feedback_target_shapes, writer_contract_errors
    from app.agent.appearance import sign_shape_instruction
    from app.agent.observation_wire import CompactAttribute, unknown_optional_evidence
    shape_task={**color_task,'sign_shape_policy':'whole-sign-shape-v1',
        'manual_review_feedback':{'reason':'形状是一个×，不是rectangular',
            'previous_expression':'A white rectangular sign reading RAILROAD CROSSING in the front view'}}
    assert feedback_target_shapes(shape_task)==[
        {'kind':'shape','value':'x shaped','denied':False,'required':True},
        {'kind':'shape','value':'rectangular','denied':True,'required':True}]
    assert feedback_attribute_kinds(shape_task)==['shape']
    assert len(feedback_acceptance(shape_task)['requirements'])==2
    wrong_shape={'expression':'The white rectangular sign in the front view.',
        'predicate_ids':['category:sign'],'visual_attributes':[{'kind':'shape','value':'rectangular'}]}
    assert set(feedback_acceptance(shape_task,wrong_shape)['blocking_errors'])=={
        'manual_feedback_rejected_shape_repeated','manual_feedback_shape_missing'}
    x_shape={'expression':'The white X-shaped sign in the front view.',
        'predicate_ids':['category:sign'],'visual_attributes':[{'kind':'shape','value':'x shaped'}]}
    assert feedback_acceptance(shape_task,x_shape)['passed']
    assert feedback_acceptance(shape_task,{**x_shape,'expression':'The white crossbuck sign in the front view.'})['passed']
    assert 'manual_feedback_shape_missing' in feedback_acceptance(shape_task,{
        **x_shape,'visual_attributes':[]})['blocking_errors']
    assert 'manual_feedback_rejected_shape_repeated' in feedback_acceptance(shape_task,{
        **x_shape,'expression':wrong_shape['expression']})['blocking_errors']
    assert feedback_acceptance(shape_task,{'predicate_ids':['category:sign'],
        'visual_attributes':x_shape['visual_attributes']},check_style=False)['passed']
    for marker in ('X形','x-shaped','X'):
        assert feedback_target_shapes({**shape_task,'manual_review_feedback':{'reason':'形状是'+marker}})[0]['value']=='x shaped'
    assert not feedback_target_shapes({**shape_task,'sign_shape_policy':None})
    assert not feedback_target_shapes({**shape_task,'manual_review_feedback':{'reason':'参照物形状是X形，不是矩形'}})
    assert not feedback_target_shapes({**shape_task,'manual_review_feedback':{
        **shape_task['manual_review_feedback'],'reason':'参照物形状是X形，不是rectangular'}})
    assert not feedback_target_shapes({**shape_task,'manual_review_feedback':{'reason':'below a rectangular sign'}})
    assert CompactAttribute.model_validate({'k':'shape','v':'x shaped'}).value=='x shaped'
    record={'target_visible':True,'visibility':'clear','visibility_reason':'Two crossed panels.',
        'attributes':[{'kind':'shape','value':'x shaped'}]}
    assert unknown_optional_evidence(record,VisualObservation)==(record,[])
    surface_wording.task.update(shape_task)
    assert 'two diagonal arms' in sign_shape_instruction(surface_wording)
    assert 'declared_shape_not_in_text' in writer_contract_errors(surface_wording,
        {'predicate_ids':['category:sign'],'visual_attributes':x_shape['visual_attributes']},wrong_shape['expression'])
    assert 'unselected_shape_in_text' in writer_contract_errors(surface_wording,
        {'predicate_ids':['category:sign'],'visual_attributes':[]},x_shape['expression'])
    assert not feedback_acceptance(mounting,{'expression':'The sign.','visual_attributes':[]})['blocking_errors']
    assert feedback_acceptance(mounting,{'expression':'The one-way sign mounted on a pole.',
        'visual_attributes':[{'kind':'subtype','value':'one-way sign'},{'kind':'feature','value':'pole mounted'}]})['passed']
    from app.quality.joint_reference import feedback_group_positions
    group_task={**color_task,'feedback_group_position_policy':'explicit-group-position-feedback-v1',
        'manual_review_feedback':{'reason':'你可以说 处于三个标注牌中间的标注牌'}}
    assert feedback_group_positions(group_task)==[{'required':False,'label':'三个标志牌中间的标志牌'}]
    optional_group=feedback_acceptance(group_task)
    assert optional_group['requirements'][0]['status']=='unsupported' and not optional_group['blocking_errors']
    assert not feedback_acceptance(group_task,{'expression':'The white sign reading PASSENGER LOADING ONLY in the front-right view.'})['blocking_errors']
    assert feedback_acceptance(group_task,{'expression':'The middle sign of the three signs.'})['blocking_errors']==['manual_feedback_group_position_unsupported']
    mandatory_group={**group_task,'manual_review_feedback':{'reason':'必须描述三个标志牌中间的标志牌'}}
    assert feedback_acceptance(mandatory_group)['blocking_errors']==['manual_feedback_group_position_unsupported']
    assert not feedback_group_positions({**group_task,'feedback_group_position_policy':None})
    assert not feedback_group_positions({**group_task,'manual_review_feedback':{'reason':'不要说三个标志牌中间的标志牌'}})
    from app.quality.joint_reference import feedback_gender_requests,person_wording_errors,PERSON_FEEDBACK_POLICY
    gender_task={**color_task,'person_feedback_policy':PERSON_FEEDBACK_POLICY,
        'manual_review_feedback':{'reason':'可以说明这个人的性别'}}
    assert feedback_gender_requests(gender_task)==[{'required':False}]
    gender_check=feedback_acceptance(gender_task,{'expression':'The pedestrian wearing a black top in the left-side view.'})
    assert gender_check['requirements'][0]['status']=='unsupported' and not gender_check['blocking_errors']
    mandatory_gender={**gender_task,'manual_review_feedback':{'reason':'必须说明这个人的性别'}}
    assert feedback_acceptance(mandatory_gender)['blocking_errors']==['manual_feedback_gender_unsupported']
    assert not feedback_gender_requests({**gender_task,'manual_review_feedback':{'reason':'不要描述性别'}})
    assert not feedback_gender_requests({**gender_task,'person_feedback_policy':None})
    neutral=surface_wording.model_copy(deep=True);neutral.target.category='pedestrian';neutral.task.update(gender_task)
    assert not person_wording_errors(neutral,'The pedestrian wearing a black top and carrying a backpack.')
    assert person_wording_errors(neutral,'The woman wearing a black top.')==['unsupported_person_gender_claim']
    surface_wording.task.update(group_task)
    assert 'unverified_group_position_in_text' in writer_contract_errors(surface_wording,
        {'predicate_ids':['category:sign'],'visual_attributes':[]},'The middle sign of three signs in the front view.')
    for reason in ['可以说明与自车距离很近','不要说明与自车距离很近']:
        optional = {**detail_task,'manual_review_feedback':{'reason':reason}}
        assert not feedback_acceptance(optional)['blocking_errors']
    keep = {**detail_task,'manual_review_feedback':{'reason':'仅保留 The white vehicle in the left-side view 即可'}}
    assert feedback_acceptance(keep,{'expression':'The white vehicle in the left-side view.'})['passed']
    assert feedback_acceptance(keep,{'expression':'The white vehicle in the left-side view is in front of a black vehicle.'})['blocking_errors'] == ['manual_feedback_retained_phrase_missing']
    assert feedback_acceptance(keep,{'expression':'The vehicle on the left.'})['blocking_errors'] == ['manual_feedback_retained_phrase_missing']
    assert concision_budget(natural_task) == len(concise_task['manual_review_feedback']['previous_expression'].split()) - 1
    bad = 'Front view: a pedestrian with black helmet, red top, blue bottoms, backpack is in front of ego vehicle.'
    assert set(expression_language_errors(bad)) >= {'telegraphic_camera_label','missing_reference_article','telegraphic_appearance_wording'}
    assert feedback_acceptance(natural_task, {'expression':bad})['blocking_errors'] == ['manual_feedback_concision_missing']
    for text in ('The pedestrian wearing a red top in the front view.',
                 'The pedestrian with a backpack is in front of the ego vehicle.'):
        assert not expression_language_errors(text)
        assert feedback_acceptance(natural_task, {'expression':text})['passed']
    assert expression_language_errors('In front view, a pedestrian with black helmet.')
    for sentence in ('The white pickup truck in the front-left view is the target.',
            'The white pickup truck is the selected object.',
            'The object of interest is the white pickup truck.'):
        assert 'annotation_meta_wording' in expression_language_errors(sentence)
        assert 'annotation_meta_wording' in simple_sentence_errors(sentence)
        assert 'annotation_meta_wording' in writer_contract_errors(simple_packet,simple_plan,sentence)
    assert not expression_language_errors('The white pickup truck in the front-left view.')
    from app.quality.joint_reference import feedback_target_features, TARGET_FEATURE_FEEDBACK_POLICY
    camera_task={'feedback_feature_policy':TARGET_FEATURE_FEEDBACK_POLICY,
        'manual_review_feedback':{'reason':'目标在拿着摄像机'}}
    assert feedback_target_features(camera_task)==[{'kind':'feature','value':'handheld camera','required':True}]
    bag_only={'expression':'The pedestrian holding a bag in the front view.',
        'visual_attributes':[{'kind':'feature','value':'handheld bag'}]}
    assert feedback_acceptance(camera_task,bag_only)['blocking_errors']==['manual_feedback_target_feature_missing']
    camera_sentence={'expression':'The pedestrian holding a camera in the front view.',
        'visual_attributes':[{'kind':'feature','value':'handheld camera'}]}
    assert feedback_acceptance(camera_task,camera_sentence)['passed']
    assert feedback_acceptance(camera_task,{**camera_sentence,'expression':bag_only['expression']})['blocking_errors']
    assert feedback_acceptance(camera_task,{**camera_sentence,'visual_attributes':[]})['blocking_errors']
    assert not feedback_target_features({'manual_review_feedback':camera_task['manual_review_feedback']})
    assert not feedback_target_features({**camera_task,'manual_review_feedback':{'reason':'目标不是拿着相机'}})
    from app.quality.description_search import appearance_phrase, descriptive_attributes
    camera_attrs=[VisualAttribute(kind='feature',value='handheld camera'),VisualAttribute(kind='feature',value='handheld bag')]
    assert appearance_phrase('pedestrian',camera_attrs)=='the pedestrian holding a camera and a bag'
    assert appearance_phrase('pedestrian',[VisualAttribute(kind='feature',value='binoculars')])=='the pedestrian using binoculars'
    camera_planning=simple_packet.model_copy(deep=True)
    camera_planning.task.update(camera_task)
    camera_planning.target.category='pedestrian'
    selected=descriptive_attributes(camera_planning,{camera_planning.target_object_id:{'target_visible':True,
        'visibility':'clear','visibility_reason':'The pedestrian holds a camera and has a bag.',
        'attributes':[a.model_dump() for a in camera_attrs]}})
    assert selected and selected[0].value=='handheld camera'
    assert not simple_sentence_errors('The white pickup truck visible in the front-left view.')
    from app.quality.joint_reference import ego_requests
    from app.quality.spatial_relations import sentence_relation_keys
    for first,axis in [('in front of','front'),('ahead','front'),('ahead of','front'),('behind','behind')]:
        for side in ('left','right'):
            sentence=f'The white pickup truck is {first} and to the {side} of the ego vehicle.'
            ids,remainder=ego_requests(sentence)
            assert ids == {'longitudinal:'+axis,'lateral:'+side}
            assert not sentence_relation_keys(sentence)
            assert not expression_language_errors(sentence)
    assert sentence_relation_keys('The white pickup truck is in front of a bus and to the left of the ego vehicle.')
    # Shared ego complements work in either order; the screenshot's right-first
    # wording must not leak an image/landmark relation into writer validation.
    from app.quality.contrast import expand_coordinated_ego_clause
    for first,axis in [('in front of','front'),('ahead','front'),('ahead of','front'),
                       ('behind','behind'),('to the rear of','behind')]:
        for side in ('left','right'):
            for clause in (f'{first} and to the {side} of', f'to the {side} of and {first}'):
                sentence=f'The blue pole-mounted sign in the front view, located {clause} the ego vehicle.'
                assert ego_requests(sentence)[0] == {'longitudinal:'+axis,'lateral:'+side}
                assert sentence_relation_keys(sentence) == set()
                expanded=expand_coordinated_ego_clause(sentence)
                assert expand_coordinated_ego_clause(expanded) == expanded
    original='The blue pole-mounted sign in the front view, located to the right of and in front of the ego vehicle.'
    coordinated_plan={**simple_plan,'predicate_ids':['category:sign','longitudinal:front','lateral:right'],
        'visual_attributes':[{'kind':'color','value':'blue'},{'kind':'feature','value':'pole mounted'}]}
    coordinated_packet=simple_packet.model_copy(deep=True)
    coordinated_packet.target.category='sign'
    coordinated_packet.target.center_m=[33.87,-9.47,3.34]
    coordinated_packet.target.size_m=[0.11,0.51,0.5]
    coordinated_packet.target.lidar_points=30
    coordinated_packet.task.update(manual_feedback_policy='verified-feedback-planning-v1',
        feedback_revision_policy='feedback-evidence-revision-v1',ego_wording_policy='proven-ego-components-v2')
    coordinated_packet.predicates=[Predicate(predicate_id=pid,kind=kind,value=value,phrase=phrase,source='test',
        matches={coordinated_packet.target_object_id:True}) for pid,kind,value,phrase in [
        ('category:sign','category','sign','sign'),
        ('longitudinal:front','longitudinal','front','in front of the ego vehicle'),
        ('lateral:right','lateral','right','to the right of the ego vehicle')]]
    assert not writer_contract_errors(coordinated_packet,coordinated_plan,original)
    assert 'declared_ego_direction_not_in_text' in writer_contract_errors(coordinated_packet,coordinated_plan,
        'The blue pole-mounted sign in front of the ego vehicle.')
    assert 'unselected_spatial_relation_in_text' in writer_contract_errors(coordinated_packet,coordinated_plan,
        'The blue pole-mounted sign to the right of a truck and in front of the ego vehicle.')
    # Explicit, distinct landmark complements must never become ego claims.
    for sentence,expected,relations in [
        ('to the right of a truck and in front of the ego vehicle', {'longitudinal:front'}, {'image_right'}),
        ('in front of a bus and to the left of the ego vehicle', {'lateral:left'}, {'vehicle_ahead'}),
        ('to the right of and in front of a truck', set(), {'vehicle_ahead','image_right'}),
        ('in the front-right view', set(), set()),
    ]:
        assert ego_requests(sentence)[0] == expected
        assert sentence_relation_keys(sentence) == relations
    small = feedback_packet.model_copy(deep=True)
    small.task.update(natural_task)
    small.predicates = [p for p in small.predicates if p.kind == 'category']
    unknown_records = {small.target_object_id:heading_records[small.target_object_id]}
    smaller = writing_plan(small,unknown_records)
    assert smaller and smaller['requires_full_scene'] and not smaller['unique_plan_found']
    assert set((a['kind'],a['value']) for a in smaller['visual_attributes'])=={('color','white'),('subtype','sedan')}
    legacy_small=small.model_copy(deep=True)
    legacy_small.task.pop('recognition_preference_policy',None)
    legacy_plan=writing_plan(legacy_small,unknown_records)
    assert 0 < len(legacy_plan['visual_attributes']) < len(unknown_records[small.target_object_id]['attributes'])
    wording_failure = {'generated':{**smaller,'expression':bad}, 'ground_checks':[],
        'verification':{'diagnostic_pass':False,'errors':['manual_review_rejected_text_repeated','missing_reference_article']}}
    assert writing_plan(small,unknown_records,attempts=[wording_failure]) == smaller
    semantic_failure = {**wording_failure, 'ground_checks':[{'unsupported_claims':['unresolved competitor']}],
        'verification':{'diagnostic_pass':False,'errors':['scene_grounding_ambiguous']}}
    assert writing_plan(small,unknown_records,attempts=[semantic_failure]) != smaller
    natural_payload = json.loads(writer_messages(small,smaller,wording_failure,unknown_records)[-1]['content'][0]['text'])
    assert 'no fixed percentage' in natural_payload['concision_instruction']
    assert 'articles' in natural_payload['concision_instruction']
    assert all(c['type']=='text' for c in writer_messages(small,smaller,wording_failure,unknown_records)[-1]['content'])
    assert 'source_target_region' not in natural_payload and natural_payload['source_camera_phrase']=='in the front view'
    assert 'mechanical_clothing_wording' in expression_language_errors('The pedestrian wearing blue lower clothing.')
    assert 'unselected_image_position_in_text' in writer_contract_errors(small,smaller,'The pedestrian on the right is wearing a red top.')
    assert 'unselected_image_position_in_text' in writer_contract_errors(small,smaller,'The pedestrian on the right side of the street.')
    assert 'unselected_visual_detail_in_text' in writer_contract_errors(small,smaller,'The pedestrian wearing a red striped shirt.')
    assert not writer_contract_errors(small,smaller,'The white sedan in the front view.')
    style_feedback = {**natural_task, 'manual_review_feedback': {'reason':
        '请使用规范通顺的英文。不要使用“Front view: a pedestrian with black helmet, red top, blue bottoms, backpack is in front of ego vehicle”这样的不规范、不通顺表达。'}}
    assert not feedback_requirements(style_feedback)['ego']
    assert not feedback_requirements(style_feedback)['required_spatial']
    assert not feedback_relations(style_feedback)
    style_only = small.model_copy(deep=True)
    style_only.task.update(style_feedback)
    style_plan = writing_plan(style_only,unknown_records)
    style_failure = {**wording_failure,'generated':{**style_plan,'expression':bad}}
    assert writing_plan(style_only,unknown_records,attempts=[style_failure]) == style_plan
    # Explicit relational corrections remain actionable even when quoted.
    still_correct = {**natural_task, 'manual_review_feedback': {'reason':'“behind a white sedan”是错误的，应采用 behind a red sedan。'}}
    assert any(r['denied'] for r in feedback_relations(still_correct))
    strict_ego=ego_packet.model_copy(deep=True)
    strict_ego.task['feedback_acceptance_policy']=ACCEPTANCE_POLICY
    strict_plan=writing_plan(strict_ego,heading_records)
    assert strict_plan and 'longitudinal:front' in strict_plan['predicate_ids']
    assert feedback_acceptance(strict_ego.task,{**strict_plan,'expression':fallback_sentence(strict_ego,strict_plan,heading_records)})['passed']
    assert feedback_acceptance(strict_ego.task,{'predicate_ids':['category:vehicle'],
        'expression':'The sedan visible in the front view.'})['errors']==['manual_feedback_spatial_missing']
    for ego_rule in strict_ego.predicates:
        if ego_rule.predicate_id=='longitudinal:front':ego_rule.matches[strict_ego.target_object_id]=None
    assert writing_plan(strict_ego,heading_records) is None
    # #5408: ego position covers only one part of this request. A native speed
    # label, heading, or a writer's motion words do not provide a motion proof.
    motion_task={**strict_ego.task,'manual_review_feedback':{'reason':'可以说明目标的运动状态，比如该灰色轿车在自车前方，是向右行驶的'}}
    motion_report=feedback_acceptance(motion_task,{'predicate_ids':['category:vehicle','longitudinal:front'],
        'expression':'The sedan in front of the ego vehicle is moving right.'})
    assert motion_report['passed'] is False and motion_report['errors']==['manual_feedback_motion_unsupported']
    assert motion_report['blocking_errors']==[]
    assert [r['status'] for r in motion_report['requirements']]==['unsupported','covered']
    example_text='可以说明目标的运动状态以减少歧义，比如该灰色轿车在自车前方，是向右行驶咖的'
    optional_task={**motion_task,'manual_review_feedback':{'reason':example_text}}
    optional=feedback_requirements(optional_task)
    assert optional['motion'] and optional['ego']==['longitudinal:front']
    assert not optional['required_motion'] and optional['required_ego']==[]
    assert not feedback_acceptance(optional_task,{'predicate_ids':['category:vehicle'],
        'expression':'The white sedan.'})['blocking_errors']
    optional_packet=ego_packet.model_copy(deep=True);optional_packet.task=optional_task
    assert writing_plan(optional_packet,heading_records) is not None
    for mandatory_text in ['必须描述运动状态，比如向右行驶', '可以说明颜色，但必须说明运动状态',
            '比如在自车前方，但必须保留运动状态', 'You must describe motion, for example moving right.']:
        mandatory_task={**motion_task,'manual_review_feedback':{'reason':mandatory_text}}
        assert feedback_acceptance(mandatory_task)['blocking_errors']==['manual_feedback_motion_unsupported']
    from app.quality.joint_reference import LEGACY_ACCEPTANCE_POLICY
    assert feedback_acceptance({**optional_task,'feedback_acceptance_policy':LEGACY_ACCEPTANCE_POLICY})['blocking_errors']==['manual_feedback_motion_unsupported']
    for text in ['Describe its motion.', 'The car is driving right.', '补充行驶方向', '是否静止', '不要忽略运动状态']:
        assert feedback_requirements({'manual_review_feedback':{'reason':text}})['motion']
    for text in ['不要描述运动，只说明自车前方', 'Do not infer motion; describe the sedan ahead of the ego vehicle.',
            '去掉向右行驶的描述', '车头朝右', '右侧的一辆车', '没有要求运动状态']:
        assert not feedback_requirements({'manual_review_feedback':{'reason':text}})['motion']
    # The same source view never silently stands for being ahead of ego.
    assert not ego_requests('visible in the front view')[0]
    assert ego_requests('in front of the ego vehicle')[0]=={'longitudinal:front'}
    assert 'unselected_ego_direction_in_text' in writer_contract_errors(ego_packet,{'predicate_ids':['category:vehicle']},'The sedan in front of the ego vehicle.')
    from app.quality.joint_reference import local_wording_retry
    wording_packet=ego_packet.model_copy(deep=True)
    wording_packet.task['distinctiveness_search_policy']='distinctiveness-evidence-search-v1'
    writing_reject={'ground_checks':[], 'verification':{'diagnostic_pass':False,
        'errors':['unselected_spatial_relation_in_text','declared_ego_direction_not_in_text']}}
    assert local_wording_retry(wording_packet,writing_reject)
    assert not local_wording_retry(wording_packet,{**writing_reject,'ground_checks':[{'diagnostic_pass':False}]})
    assert not local_wording_retry(wording_packet,{'verification':{'errors':['scene_grounding_ambiguous']}})
    wording_packet.task.pop('distinctiveness_search_policy')
    assert not local_wording_retry(wording_packet,writing_reject)
    # A true ego cue may still need two blind checks; retain the competitor.
    ambiguous_ego=ego_packet.model_copy(deep=True)
    next(o for o in ambiguous_ego.candidates if o.object_id=='other-sedan').center_m=ambiguous_ego.target.center_m
    ambiguous_ego.predicates=predicate_catalog(ambiguous_ego.candidates)
    ambiguous_ego,_=expand_relations(ambiguous_ego,heading_records)
    ambiguous_plan=writing_plan(ambiguous_ego,heading_records)
    assert 'longitudinal:front' in ambiguous_plan['predicate_ids'] and ambiguous_plan['requires_full_scene']
    # Local rejection of the reviewed text also retires its evidence bundle,
    # rather than paying for paraphrases before any ground check took place.
    old={'generated':ego_plan,'ground_checks':[],
        'verification':{'diagnostic_pass':False,'errors':['manual_review_rejected_text_repeated']}}
    revised=writing_plan(ego_packet,heading_records,attempts=[old])
    assert revised is None or set(revised['predicate_ids'])!=set(ego_plan['predicate_ids']) or revised['visual_attributes']!=ego_plan['visual_attributes']
    for p in ego_packet.predicates:
        if p.kind in {'longitudinal','lateral'}:p.matches[ego_packet.target_object_id]=None
    safe_ego=writing_plan(ego_packet,heading_records)
    assert safe_ego is None or not any(i.startswith(('longitudinal:','lateral:')) for i in safe_ego['predicate_ids'])
    # A requested but false/unknown relation never becomes usable evidence.
    for p in feedback_packet.predicates:
        if p.predicate_id.startswith(('image_joint_heading:','image_joint_spatial:')):
            p.matches[feedback_packet.target_object_id]=None
    safe_plan=writing_plan(feedback_packet,heading_records)
    assert safe_plan is None or not safe_plan['joint_reference']
    # A scene check rejected this relation: reselect a different bundle rather
    # than rendering the same failed relation with 'is ...' over and over.
    replanning=native_heading.model_copy(deep=True)
    next(o for o in replanning.candidates if o.object_id=='other-sedan').center_m=(20,25,1)
    replanning,_=expand_relations(replanning,heading_records)
    right=next(p for p in replanning.predicates if p.predicate_id==right.predicate_id)
    replanning.predicates=[next(p for p in replanning.predicates if p.predicate_id=='category:vehicle'),right]
    replanning.provenance.pop('admission_plan',None)
    first_plan=writing_plan(replanning,heading_records)
    assert first_plan and right.predicate_id in first_plan['predicate_ids']
    scoped_retry=replanning.model_copy(deep=True)
    scoped_retry.task.update(sentence_composition_policy=COMPOSITION_POLICY,
        feedback_relation_policy='bilingual-landmark-feedback-v1')
    scoped_retry.predicates.append(Predicate(predicate_id=f'image_visibility:{source_camera}:vehicle',
        kind='image_relation',value=str(source_camera),phrase='the vehicle visible in the front view',source='test',
        matches={o.object_id:True for o in scoped_retry.candidates}))
    pair_failure={'generated':first_plan,'ground_checks':[{}],
        'verification':{'diagnostic_pass':False,'errors':['joint_reference_pair_unverified','joint_reference_pair_missing']}}
    next_plan=writing_plan(scoped_retry,heading_records,attempts=[pair_failure])
    assert next_plan is None or composition_bundle(scoped_retry,next_plan['predicate_ids'])!=composition_bundle(scoped_retry,first_plan['predicate_ids'])
    assert scoped_retry.candidates==replanning.candidates
    failed={'generated':{**first_plan,'expression':'Rejected relation sentence.'},'ground_checks':[{}],
        'verification':{'diagnostic_pass':False,'errors':['scene_grounding_unsupported_claims']}}
    replacement=writing_plan(replanning,heading_records,attempts=[failed])
    assert replacement is None or right.predicate_id not in replacement['predicate_ids']
    omission={**failed,'ground_checks':[],'verification':{'diagnostic_pass':False,'errors':['declared_joint_relation_not_in_text']}}
    assert writing_plan(replanning,heading_records,attempts=[omission]) is not None
    # End-to-end control flow: an ambiguous attribute-only proposal may use
    # both fresh scene checks without being forced to invent a landmark.
    # After a rejected bundle, a new writer call uses a different contract.
    ordinary=native_heading.model_copy(deep=True)
    ordinary.task.update(execution_policy='evidence-fast-v1',visibility_policy=RECOGNIZABLE_POLICY,
        scene_claim_policy=scene_grounding.CLAIM_POLICY,manual_feedback_policy=FEEDBACK_POLICY,
        scene_relation_reading_policy=spatial.READING_POLICY,feedback_acceptance_policy=ACCEPTANCE_POLICY,
        manual_review_feedback={'reason':'可以考虑运动状态帮助消歧，比如向右行驶。'})
    next(o for o in ordinary.candidates if o.object_id=='other-sedan').center_m=ordinary.target.center_m
    ordinary.predicates=predicate_catalog(ordinary.candidates)
    ordinary,_=expand_relations(ordinary,heading_records)
    ordinary.provenance.pop('admission_plan',None)
    class NativeSceneCalls(LocalCalls):
        writes=0
        scenes=0
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            payload=json.loads(messages[-1]['content'][0]['text'])
            if schema is WrittenSentence:
                self.writes+=1
                assert 'Only required_meaning supplies spatial claims' in payload['writing_scope']
                assert not any('above' in phrase or 'below' in phrase for phrase in payload['required_meaning'])
                return schema(expression='The white sedan visible in the front view.' if any('visible' in phrase for phrase in payload['required_meaning']) else 'The white sedan.')
            assert schema is scene_grounding.SceneGroundResult
            self.scenes+=1
            assert payload['claim_evaluation']['policy']==scene_grounding.CLAIM_POLICY
            assert payload['sentence_relation_index']['rows']==[]
            if self.scenes==1:
                return schema(expression_supported=False,candidate_ids=[],ambiguous_candidate_ids=[],
                    unsupported_claims=['The current description needs a clearer scope.'],reason='Reject the first bundle.')
            target_id=next(row['candidate_id'] for row in payload['candidates'] if row['original_image_boxes_xyxy']==[[1,5,50,55]])
            assert all(not r['relation_key'].startswith('image_') for r in payload['spatial_relations']['rows'])
            return schema(expression_supported=True,candidate_ids=[target_id],ambiguous_candidate_ids=[],unsupported_claims=[],reason='Full scoped identity checked.')
    ordinary_calls=NativeSceneCalls();root=tmp_path/'native-scene-run'
    initial=initial_run(ordinary,ordinary_calls,1,5)
    initial.update(phase='precheck_decision',relations_acquired=True,relook_completed=True,candidate_appearance=heading_records,
        target_precheck_observation={k:v for k,v in heading_records[ordinary.target_object_id].items() if k!='observed_camera_name'})
    RunStore(root/'runs.sqlite3').create(initial)
    result=run_agent(ordinary,ordinary_calls,1,root,5,resume_run_id=initial['run_id'])
    assert result['diagnostic_pass'] and ordinary_calls.count==5,result
    assert result['feedback_check']['passed'] is False and not result['feedback_check']['blocking_errors']
    assert result['feedback_check']['requirements'][0]['required'] is False
    assert ordinary_calls.writes==2 and ordinary_calls.scenes==3
    assert all(not a['writing_contract']['joint_reference'] for a in result['attempts'])
    assert result['attempts'][0]['generated']['predicate_ids']!=result['attempts'][1]['generated']['predicate_ids']
    from app.agent.appearance import landmark_observation_blockers, redundant_relook, candidate_input_digest, observation_messages, SceneObservationCache
    unresolved=heading.model_copy(deep=True)
    unresolved.provenance['scene_image_candidates'].append({'object_id':'red-lookalike','category':'vehicle',
        'camera_name':1,'xyxy':[200,5,240,55]})
    unknown_type={**clear().model_dump(),'observed_camera_name':1,'attributes':[{'kind':'color','value':'red'}]}
    records={**heading_records,'red-lookalike':unknown_type}
    requirements=landmark_observation_blockers(unresolved,records,{'other-sedan'},with_requirements=True)
    assert requirements=={'red-lookalike':['subtype']}
    # A sedan still in the target pool can be a potential red landmark: being
    # its own front/rear neighbor is false. Planning must not discard it early.
    assert landmark_observation_blockers(unresolved,records,{red.object_id},with_requirements=True)==requirements
    ordinary={'candidate_id':'anonymous','object_id':'red-lookalike','category':'vehicle','camera_name':1,'artifact':'appearance.jpg'}
    focused={**ordinary,'attribute_recheck':requirements['red-lookalike']}
    record={**unknown_type,'observed_input_sha256':candidate_input_digest(unresolved,ordinary)}
    assert redundant_relook(unresolved,ordinary,record) and not redundant_relook(unresolved,focused,record)
    assert candidate_input_digest(unresolved,ordinary)!=candidate_input_digest(unresolved,focused)
    assert SceneObservationCache.key(unresolved,focused,('test',)) is None
    payload=json.loads(observation_messages(unresolved,tmp_path/'heading-evidence',[focused])[-1]['content'][0]['text'])
    assert payload['attribute_checks']==[{'candidate_id':'anonymous','kinds':['subtype']}]
    assert 'red-lookalike' not in json.dumps(payload)  # No source ID or suggested answer.
    for modification in ('unknown_image','unknown_native','duplicate_red','projection_only','legacy'):
        probe=heading.model_copy(deep=True)
        records=dict(heading_records)
        if modification in {'unknown_image','unknown_native','duplicate_red'}:
            oid='image:unknown' if modification=='unknown_image' else modification
            if modification!='unknown_image':
                probe.candidates.append(obj(oid,10,5,category='unknown' if modification=='unknown_native' else 'vehicle',points=0))
            probe.provenance['scene_image_candidates'].append({'object_id':oid,'category':'vehicle',
                'camera_name':1,'xyxy':[200,5,240,55],'identity_source':'official_lidar_box_projection'})
            # Non-red unknown-type objects still compete with the white target.
            records[oid]={**clear().model_dump(),'observed_camera_name':1,
                'attributes':[{'kind':'color','value':'red' if modification=='duplicate_red' else 'white'}]}
        elif modification=='projection_only':
            probe.provenance['scene_image_candidates'][-1]['identity_source']='official_lidar_box_projection'
        else:
            probe.task.pop('vehicle_heading_policy')
        probe.predicates=predicate_catalog(probe.candidates)
        probe,_=expand_relations(probe,records)
        current=next((p for p in probe.predicates if p.predicate_id==rule.predicate_id),None)
        if modification in {'duplicate_red','projection_only','legacy'}:
            assert current is None
        else:
            assert current is not None
            proof=expression_candidates(probe,[next(p for p in probe.predicates if p.predicate_id=='category:vehicle'),current],attrs,records)
            assert proof['image_only'] or len(proof['native'])>1

    # A sign in front of a vehicle does not create a second vehicle identity.
    # Same-class and unknown owners still compete, and weak/ambiguous links stay
    # separate. Identity linking does not itself certify RGB visibility.
    vehicle, sign = obj('native-car', 0, 0), obj('occluder', 10, 0, 'sign')
    grid = np.array([(x,y,1) for x in np.linspace(-1,1,5) for y in np.linspace(-.4,.4,5)],dtype=np.float32)
    pixels = np.array([[1, 10+10*(i%5), 20+10*(i//5), 0,0,0] for i in range(25)],dtype=np.int32)
    arrays = {'points':np.concatenate((grid, grid+[10,0,0])), 'camera_projection':np.tile(pixels,(2,1))}
    box = {'camera_object_id':'native-2d', 'laser_object_ids':[], 'association_status':'unlinked',
        'category':'vehicle', 'xyxy':[0,0,100,100], 'selected_target':False}
    mapped = point_supported_native_boxes({1:[box]}, [vehicle,sign], arrays, vehicle.object_id)
    assert mapped[1][0]['laser_object_ids'] == [vehicle.object_id]
    assert mapped[1][0]['identity_source'] == 'native_box_point_correspondence'
    assert mapped[1][0]['support_points'] == 25
    assert len(scene_image_inventory(mapped,{1:[{'object_id':vehicle.object_id,'category':'vehicle',
        'xyxy':[0,0,100,100],'identity_source':'official_lidar_box_projection'}]})) == 1
    assert not box['laser_object_ids']  # The original native record is preserved.
    for category in ('vehicle','unknown'):
        rival = sign.model_copy(update={'category':category})
        assert not point_supported_native_boxes({1:[box]},[vehicle,rival],arrays,vehicle.object_id)[1][0]['laser_object_ids']
    duplicates = point_supported_native_boxes({1:[box,dict(box,camera_object_id='duplicate')]},
        [vehicle,sign],arrays,vehicle.object_id)
    assert not any(b['laser_object_ids'] for b in duplicates[1])
    # The target fills the RGB label, while a known foreground vehicle only
    # contributes a low-coverage boundary strip. Pixel purity alone loses this
    # link; central overlap, substantial rival coverage and unknowns cannot use
    # the full-region alternative proof.
    target_pixels = np.array([[1,x,y,0,0,0] for x in (5,27,50,72,95)
        for y in (5,27,50,72,95)],dtype=np.int32)
    front_pixels = np.array([[1,10+15*i,95+i%3,0,0,0] for i in range(5)]
        + [[1,150+2*i,40+i,0,0,0] for i in range(35)],dtype=np.int32)
    foreground = sign.model_copy(update={'category':'vehicle'})
    edge_arrays = {'points':np.concatenate((grid,np.tile(grid+[10,0,0],(2,1))[:40])),
        'camera_projection':np.concatenate((target_pixels,front_pixels))}
    linked_edge = point_supported_native_boxes({1:[box]},[vehicle,foreground],edge_arrays,vehicle.object_id)
    assert linked_edge[1][0]['laser_object_ids'] == [vehicle.object_id]
    assert linked_edge[1][0]['support_points'] == 25
    for competing, projection in (
            (foreground.model_copy(update={'category':'unknown'}),front_pixels),
            (foreground,np.array([[1,10+15*i,50+i%3,0,0,0] for i in range(5)]+front_pixels[5:].tolist())),
            (foreground,front_pixels[:5])):
        modified = {'points':edge_arrays['points'][:25+len(projection)],
            'camera_projection':np.concatenate((target_pixels,projection))}
        assert not point_supported_native_boxes({1:[box]},[vehicle,competing],modified,vehicle.object_id)[1][0]['laser_object_ids']
    duplicate_edge = point_supported_native_boxes({1:[box,dict(box,camera_object_id='edge-duplicate')]},
        [vehicle,foreground],edge_arrays,vehicle.object_id)
    assert not any(b['laser_object_ids'] for b in duplicate_edge[1])
    # Another car's own RGB rectangle can overlap the target rectangle. Use
    # sufficient uncontested pixels, not purity over their shared rectangle.
    # This includes a substantially visible rival, not just an edge sliver.
    overlap_pixels = np.array([[1,x,y,0,0,0] for x in (25,27,29,31,35)
        for y in (10,20,30,40,50)],dtype=np.int32)
    overlap_arrays = {'points':np.concatenate((grid,grid+[10,0,0])),
        'camera_projection':np.concatenate((target_pixels,overlap_pixels))}
    overlap_box = dict(box,camera_object_id='other-car',xyxy=[20,0,40,55])
    overlapping = point_supported_native_boxes({1:[box,overlap_box]},[vehicle,foreground],
        overlap_arrays,vehicle.object_id)
    assert overlapping[1][0]['laser_object_ids'] == [vehicle.object_id]
    assert not overlapping[1][1]['laser_object_ids']  # No reverse/ambiguous match.
    # A partially occluded red car can own all its points in a compact exposed
    # patch. A neighboring truck contributes substantial shared-edge returns;
    # independently pure exposed pixels establish identity without requiring
    # the car's points to fill most of the camera rectangle.
    compact_pixels=np.array([[1,x,y,0,0,0] for x in (5,15,25,35,45)
        for y in (10,20,30,40,50)],dtype=np.int32)
    truck_pixels=np.array([[1,75+i%3,10+5*i,0,0,0] for i in range(10)]
        + [[1,150+i,20+i,0,0,0] for i in range(40)],dtype=np.int32)
    compact_arrays={'points':np.concatenate((grid,np.tile(grid+[10,0,0],(2,1)))),
        'camera_projection':np.concatenate((compact_pixels,truck_pixels))}
    truck_box=dict(box,camera_object_id='truck-rgb',xyxy=[60,0,100,100])
    compact_link=point_supported_native_boxes({1:[box,truck_box]},[vehicle,foreground],compact_arrays,vehicle.object_id)
    assert compact_link[1][0]['laser_object_ids']==[vehicle.object_id]
    assert compact_link[1][0]['support_points']==25
    for labels,rival_unit in (([box,truck_box],foreground.model_copy(update={'category':'unknown'})),
                              ([box,dict(box,camera_object_id='duplicate'),truck_box],foreground)):
        assert not point_supported_native_boxes({1:labels},[vehicle,rival_unit],compact_arrays,vehicle.object_id)[1][0]['laser_object_ids']
    for boxes, rival in (([box],foreground),
            ([box,overlap_box],foreground.model_copy(update={'category':'unknown'})),
            ([box,dict(box,camera_object_id='same-bounds')],foreground)):
        assert not point_supported_native_boxes({1:boxes},[vehicle,rival],
            overlap_arrays,vehicle.object_id)[1][0]['laser_object_ids']
    # A label that leaves only a narrow, weakly supported strip is insufficient.
    covered_box = dict(overlap_box,xyxy=[0,0,100,90])
    assert not point_supported_native_boxes({1:[box,covered_box]},[vehicle,foreground],
        overlap_arrays,vehicle.object_id)[1][0]['laser_object_ids']
    # Camera boxes can cover several separately annotated construction units.
    # Retain all members without inventing one-to-one identities or deleting rivals.
    assembly_arrays = {'points':np.concatenate((grid,grid+[10,0,0])),
        'camera_projection':np.concatenate((pixels,pixels+[0,40,0,0,0,0]))}
    assembly_objects = [vehicle,sign.model_copy(update={'category':'vehicle'})]
    group = point_supported_native_boxes({1:[box]},assembly_objects,assembly_arrays,vehicle.object_id)[1][0]
    assert not group['laser_object_ids']
    assert set(group['native_unit_object_ids']) == {o.object_id for o in assembly_objects}
    group_duplicates = point_supported_native_boxes({1:[box,dict(box,camera_object_id='duplicate')]},
        assembly_objects,assembly_arrays,vehicle.object_id)
    assert not any(b.get('native_unit_object_ids') for b in group_duplicates[1])
    weak = {'points':np.concatenate((grid,grid)),
        'camera_projection':np.concatenate((pixels,pixels+[0,150,0,0,0,0]))}
    assert not point_supported_native_boxes({1:[box]},[vehicle],weak,vehicle.object_id)[1][0]['laser_object_ids']

    p = strict_packet(tmp_path/'evidence', monkeypatch)
    p.task.update(annotation_search='evidence-guided-v1', image_identity_policy='official-projection-v1',
        image_inventory_policy=IMAGE_INVENTORY_POLICY, visibility_policy=RECOGNIZABLE_POLICY)
    p.source_camera_object_id = None
    p.views[0].appearance_camera_object_id = None
    p.views[0].appearance_identity_source = 'official_lidar_box_projection'
    p.provenance['recognizable_source_cameras'] = {'1': {}}
    p.provenance['view_screening_policy'] = 'recognizable-projection-ranking-v3'
    linked = {'camera_object_id':'linked', 'laser_object_ids':[p.target_object_id],
        'association_status':'linked_target', 'category':'vehicle', 'xyxy':[49,19,81,81]}
    unlinked = {**linked, 'camera_object_id':'unlinked', 'laser_object_ids':[],
        'association_status':'unlinked', 'xyxy':[1,20,20,80]}
    boxes = {1:[linked, unlinked], 2:[{**unlinked, 'camera_object_id':'other-view'}]}
    projected = {1:[{'object_id':p.target_object_id, 'category':'vehicle',
        'xyxy':[50,20,80,80], 'identity_source':'official_lidar_box_projection'}]}
    inventory = scene_image_inventory(boxes, projected)
    assert len(inventory) == 3  # Only the confirmed same-camera link is deduplicated.
    assert inventory[0]['xyxy'] == linked['xyxy']
    assert inventory[0]['camera_object_id'] == linked['camera_object_id']
    rival = 'image:1:unlinked'
    assert {c['object_id'] for c in inventory} == {p.target_object_id, rival, 'image:2:other-view'}
    assert next(c for c in inventory if c['object_id']==rival)['identity_source']=='unlinked_native_box'
    conflict = scene_image_inventory({1:[{**linked, 'category':'pedestrian'}]}, projected)
    assert len(conflict) == 2 and conflict[1]['object_id'] == 'image:1:linked'

    jpeg = (tmp_path/'evidence'/p.packet_id/'appearance.jpg').read_bytes()
    reader = SimpleNamespace(rows=lambda *a, **kw:[{'[CameraImageComponent].image':jpeg}])
    crops = tmp_path/'crops'; crops.mkdir()
    manifest = build_candidate_crops(reader, p.context, p.frame_timestamp_micros, boxes,
        p.candidates, crops, include_unlinked=True, preferred_camera=1, projected_boxes=projected,
        view_seeds={p.target_object_id:[{'source_camera_id':1,'foreground_object_ids':[p.candidates[1].object_id]}]})
    assert {c['object_id'] for c in manifest} == {c['object_id'] for c in inventory}
    assert all((crops/m['artifact']).is_file() for m in manifest)
    target_crop = next(m for m in manifest if m['object_id'] == p.target_object_id)
    assert target_crop['potential_foreground_object_ids'] == [p.candidates[1].object_id]
    with Image.open(crops/target_crop['artifact']) as image:
        assert image.size == (32,62)  # Verified native RGB box, not the 3D projection's (30,60).
    assert target_crop['camera_object_id'] == linked['camera_object_id']
    # Reusing decoded same-frame originals must preserve every crop byte and
    # fall back to the reader for cameras not already decoded by the target.
    import io
    with Image.open(io.BytesIO(jpeg)) as source:
        cached_image = source.convert('RGB')
    reads = []
    cached_reader = SimpleNamespace(rows=lambda *a, **kw:
        (reads.append(kw['camera']) or [{'[CameraImageComponent].image':jpeg}]))
    cached_crops = tmp_path/'cached-crops'; cached_crops.mkdir()
    reused = build_candidate_crops(cached_reader, p.context, p.frame_timestamp_micros, boxes,
        p.candidates, cached_crops, include_unlinked=True, preferred_camera=1, projected_boxes=projected,
        view_seeds={p.target_object_id:[{'source_camera_id':1,'foreground_object_ids':[p.candidates[1].object_id]}]},
        camera_images={1:cached_image})
    assert reused == manifest and reads == [2]
    for path in crops.iterdir():
        assert (cached_crops/path.name).read_bytes() == path.read_bytes()

    p.provenance['scene_image_candidates'] = inventory
    p.provenance['source_image_candidates'] = [c for c in inventory if c['camera_name'] == 1]
    observations = {p.target_object_id:{**clear().model_dump(), 'observed_camera_name':1}}
    rules, _ = image_relation_catalog(p, observations)
    p.predicates.extend(rules)
    ids = ['category:vehicle', 'longitudinal:front', 'lateral:left', 'image_visibility:1:vehicle']
    selected = [r for r in p.predicates if r.predicate_id in ids]
    proof = expression_candidates(p, selected, clear().attributes, observations)
    assert proof['image_only'] == [rival] and not proof['camera_identity_proof']
    expression = GeneratedExpression(expression='The white vehicle ahead and to the left of the ego vehicle, visible in the front view.',
        predicate_ids=ids, visual_attributes=clear().attributes)
    parsed = GroundResult(candidate_ids=[], parsed_predicate_ids=ids, unsupported_claims=[],
        parsed_visual_attributes=clear().attributes)
    checked = verify_tool_grounding(p, expression, parsed, {}, observations, target_appearance_verified=True)
    assert not checked['diagnostic_pass'] and 'image_expression_not_unique' in checked['errors']
    # A confirmed different appearance may exclude this rival; camera 2 is outside this sentence.
    observations[rival] = {**clear().model_dump(), 'observed_camera_name':1,
        'attributes':[{'kind':'color','value':'red'}]}
    assert expression_candidates(p, selected, clear().attributes, observations)['camera_identity_proof']
    # A unique intrinsic source-image reference needs neither an ordinal nor a
    # landmark. The verified native identity supplies its one 3D target; unrelated
    # camera boxes cannot block that explicitly scoped reference. Same-image
    # unknowns, conflicting RGB and missing identity must still block acceptance.
    simple=p.model_copy(deep=True)
    from app.quality.relations import expand_relations
    simple.source_camera_object_id='linked'
    simple.views[0].appearance_camera_object_id='linked'
    simple.views[0].appearance_identity_source='official_camera_lidar_association'
    simple.task['reference_policy']='minimal-sufficient-grounding-v1'
    simple.task.pop('description_detail_policy',None)
    simple,relation_result=expand_relations(simple,observations)
    simple_plan=admission_plan(simple,clear(),observations,relation_result['anchors'])
    assert simple_plan['unique_plan_found']
    assert any(pid.startswith('image_attribute:') for pid in simple_plan['predicate_ids'])
    assert not any(pid.startswith('image_order:') for pid in simple_plan['predicate_ids'])
    simple.provenance['admission_plan']=simple_plan
    from app.agent.expression_rendering import render_verified_plan
    direct_text=render_verified_plan(simple,observations)
    assert direct_text and 'white' in direct_text.expression
    assert 'leftmost' not in direct_text.expression and 'ego vehicle' not in direct_text.expression
    direct_parse=GroundResult(candidate_ids=[],parsed_predicate_ids=direct_text.predicate_ids,
        parsed_visual_attributes=direct_text.visual_attributes,unsupported_claims=[])
    assert verify_tool_grounding(simple,direct_text,direct_parse,{},observations,
        target_appearance_verified=True)['diagnostic_pass']
    missing_attr=direct_parse.model_copy(update={'parsed_visual_attributes':[]})
    assert not verify_tool_grounding(simple,direct_text,missing_attr,{},observations,
        target_appearance_verified=True)['diagnostic_pass']
    for ambiguous_records in ({k:v for k,v in observations.items() if k!=rival},
            {**observations,rival:{**observations[rival],'observed_camera_name':2}},
            {**observations,rival:{**observations[rival],'visibility_conflict':True,'target_visible':False}}):
        rules,_=image_relation_catalog(simple,ambiguous_records)
        assert not any(q.predicate_id.startswith('image_attribute:') for q in rules)
    assert not any(q.predicate_id.startswith('image_attribute:') for q in image_relation_catalog(p,observations)[0])

    # Partial objects at an ORIGINAL image boundary are usable references.
    # Distant unknown boxes cannot share that boundary, but nearby unresolved
    # boxes remain possible. Crop boundaries and wrong-camera RGB prove nothing.
    boundary=simple.model_copy(deep=True)
    boundary.task['image_position_policy']='verified-boundary-v1'
    boundary.provenance['scene_image_candidates']=[
        {**c,'xyxy':([0,20,80,boundary.views[0].height] if c['object_id']==boundary.target_object_id
                    else [20,5,40,18])} for c in inventory]
    partial_records={boundary.target_object_id:{**clear().model_dump(),
        'visibility':'partial','observed_camera_name':1}}
    boundary,expanded=expand_relations(boundary,partial_records)
    edge=next(q for q in boundary.predicates if q.predicate_id=='image_boundary:1:vehicle:left')
    assert edge.matches[boundary.target_object_id] is True and edge.matches[rival] is False
    boundary.provenance['admission_plan']={'passed':True,'unique_plan_found':True,
        'predicate_ids':['category:vehicle',edge.predicate_id], 'visual_attributes':clear().model_dump()['attributes']}
    partial_text=render_verified_plan(boundary,partial_records)
    assert partial_text and 'partly cut off by the left edge of the front view' in partial_text.expression
    partial_parse=GroundResult(candidate_ids=[],parsed_predicate_ids=partial_text.predicate_ids,
        parsed_visual_attributes=partial_text.visual_attributes,unsupported_claims=[])
    assert verify_tool_grounding(boundary,partial_text,partial_parse,{},partial_records,
        target_appearance_verified=True)['diagnostic_pass']
    from app.quality.admission import observation_plan_sufficient,reference_search_groups
    scene_first=boundary.model_copy(deep=True)
    scene_first.task['reference_preference_policy']='scene-first-v1'
    assert observation_plan_sufficient(boundary,boundary.provenance['admission_plan'])
    assert not observation_plan_sufficient(scene_first,boundary.provenance['admission_plan'])
    # A two-relation scene option must precede even a one-cue pixel fallback.
    relation=edge.model_copy(update={'predicate_id':'relative_left:a','kind':'relative_left','value':'a'})
    relation2=edge.model_copy(update={'predicate_id':'ahead_of:b','kind':'ahead_of','value':'b'})
    searched=[tuple(p.predicate_id for p in extras) for group in reference_search_groups(scene_first,[edge,relation,relation2])
        for extras in group]
    assert searched.index((relation.predicate_id,relation2.predicate_id))<searched.index((edge.predicate_id,))
    recognized={**partial_records,rival:{**clear().model_dump(),'observed_camera_name':1,
        'attributes':[{'kind':'color','value':'red'}]}}
    scene_first,expanded=expand_relations(scene_first,recognized)
    preferred=admission_plan(scene_first,VisualObservation.model_validate({k:v for k,v in partial_records[scene_first.target_object_id].items()
        if k in VisualObservation.model_fields}),
        recognized,expanded['anchors'])
    assert preferred['unique_plan_found'] and observation_plan_sufficient(scene_first,preferred)
    assert not any(id_.startswith(('image_boundary:','image_order:')) for id_ in preferred['predicate_ids'])
    assert {'longitudinal:front','lateral:left'}<=set(preferred['predicate_ids'])
    scene_first.provenance['admission_plan']=preferred
    from app.agent.harness import generation_messages
    brief=json.loads(generation_messages(scene_first,None,recognized)[-1]['content'][0]['text'])
    assert brief['preferred_recognition_attributes']==[{'kind':'color','value':'white'}]
    assert not brief['required_recognition_attributes']
    assert not any(f['predicate_id'].startswith(('image_boundary:','image_order:')) for f in brief['available_target_facts'])

    # New city-traffic writing never uses camera framing, including an old
    # prepared boundary-only plan or a paraphrase with the predicate omitted.
    # Keep the partial object's evidence and historical proof intact.
    from app.quality.description_search import TRAFFIC_SCENE_POLICY, draft_reference_predicate
    from app.agent.harness import draft_reference_errors, image_framing_description, TRAFFIC_TASK_GUIDE
    traffic=boundary.model_copy(deep=True)
    traffic.task['reference_preference_policy']=TRAFFIC_SCENE_POLICY
    assert not draft_reference_predicate(traffic,edge)
    assert render_verified_plan(traffic,partial_records) is None
    assert edge.predicate_id in {q.predicate_id for q in traffic.predicates}
    assert traffic.provenance['scene_image_candidates']==boundary.provenance['scene_image_candidates']
    assert 'non_scene_reference_cue' in verify_tool_grounding(traffic,partial_text,partial_parse,{},partial_records,
        target_appearance_verified=True)['errors']
    for text in ('The vehicle, partially cut off by the bottom edge of the left-side view.',
                 'The truck clipped at the image border.', 'The car at the edge of the front image.',
                 'The sedan partially outside the left-side view.', 'The car partly out of frame.'):
        disguised=partial_text.model_copy(update={'expression':text,'predicate_ids':['category:vehicle']})
        assert image_framing_description(text)
        assert 'image_framing_not_traffic_description' in draft_reference_errors(traffic,disguised,partial_records)
    for text in ('The silver pickup directly ahead of the ego vehicle.',
                 'The pedestrian partly occluded by the red sedan.', 'The car alongside the road edge.',
                 'The silver truck in the left-side view.'):
        assert not image_framing_description(text)
    traffic.task['reference_policy']='minimal-sufficient-grounding-v1'
    messages=generation_messages(traffic,None,partial_records)
    traffic_brief=json.loads(messages[-1]['content'][0]['text'])
    assert messages[0]['content'].startswith(TRAFFIC_TASK_GUIDE)
    assert traffic_brief['grounded_writing_brief']['forbidden_cues']==['image-edge truncation and camera framing']
    assert not any(f['predicate_id'].startswith('image_boundary:') for f in traffic_brief['available_target_facts'])
    assert all(not any(id_.startswith('image_boundary:') for id_ in option['predicate_ids'])
        for option in traffic_brief['prechecked_reference_options'])

    # Two silver sedans are not standalone landmarks. A truthful ego qualifier
    # can identify one; its spatial relation can then distinguish two black SUVs.
    qualified=simple.model_copy(deep=True)
    qualified.task['landmark_description_policy']='spatially-qualified-v1'
    landmark_a,landmark_b=obj('landmark-a',12,5),obj('landmark-b',12,-5)
    qualified.candidates.extend([landmark_a,landmark_b])
    qualified.predicates=predicate_catalog(qualified.candidates)
    negative=qualified.candidates[1].object_id
    qualified.provenance['scene_image_candidates']=[
        {'object_id':oid,'category':'vehicle','camera_name':1,'camera_object_id':oid,'xyxy':rect}
        for oid,rect in [(qualified.target_object_id,[100,20,150,80]),(negative,[270,20,320,80]),
                         ('landmark-a',[200,20,250,80]),('landmark-b',[20,20,70,80])]]
    def appearance(color,subtype):
        return {**clear().model_dump(),'observed_camera_name':1,
            'attributes':[{'kind':'color','value':color},{'kind':'subtype','value':subtype}]}
    qualified_records={qualified.target_object_id:appearance('black','suv'),negative:appearance('black','suv'),
        'landmark-a':appearance('silver','sedan'),'landmark-b':appearance('silver','sedan')}
    qualified,r=expand_relations(qualified,qualified_records)
    anchor=next(a for a in r['anchors'] if a['object_id']=='landmark-a' and a.get('camera_name')==1)
    assert 'silver sedan' in anchor['phrase'] and 'left of the ego vehicle' in anchor['phrase']
    assert anchor['descriptor_spatial_predicates']==['lateral:left']
    relation_id=f'image_anchor:1:left_of:{anchor["anchor_id"]}'
    chosen=[q for q in qualified.predicates if q.predicate_id in ['category:vehicle',relation_id]]
    black_suv=[VisualAttribute(kind='color',value='black'),VisualAttribute(kind='subtype',value='suv')]
    assert expression_candidates(qualified,chosen,black_suv,qualified_records)['camera_identity_proof']
    # Missing a native identity or an unknown extra object cannot be waved away
    # by a joint landmark descriptor.
    unresolved=qualified.model_copy(deep=True)
    unresolved.provenance['scene_image_candidates'].append({'object_id':'unknown-landmark','category':'unknown',
        'camera_name':1,'camera_object_id':None,'xyxy':[195,25,255,85]})
    from app.quality.description_search import _image_landmarks
    assert not any(a['object_id']=='landmark-a' for a in _image_landmarks(unresolved,1,
        unresolved.provenance['scene_image_candidates'],qualified_records))
    legacy_qualified=qualified.model_copy(deep=True)
    legacy_qualified.task.pop('landmark_description_policy')
    assert not any(a['object_id']=='landmark-a' for a in _image_landmarks(legacy_qualified,1,
        legacy_qualified.provenance['scene_image_candidates'],qualified_records))
    # A same-camera relation to an independently identifiable seed can qualify
    # an otherwise repeated landmark. The seed cannot select itself through it.
    chained=qualified.model_copy(deep=True)
    chained.candidates[-1]=obj('landmark-b',16,5)
    chained.predicates=predicate_catalog(chained.candidates)
    seed_id=chained.candidates[2].object_id
    chained.provenance['scene_image_candidates'].append({'object_id':seed_id,'category':'sign',
        'camera_name':1,'camera_object_id':seed_id,'xyxy':[80,1,90,15]})
    chain_records={**qualified_records,seed_id:{**clear().model_dump(),'observed_camera_name':1,
        'attributes':[{'kind':'shape','value':'rectangular'}]}}
    chained,chain_result=expand_relations(chained,chain_records)
    chained_anchor=next(a for a in chain_result['anchors'] if a['object_id']=='landmark-a' and a.get('camera_name')==1)
    assert chained_anchor['dependency_object_ids']==[seed_id]
    assert 'right of the rectangular sign' in chained_anchor['phrase']
    assert all(q.matches.get(seed_id) is False for q in chained.predicates
        if q.kind=='image_relation' and q.value==chained_anchor['anchor_id'])
    for uncertain in ({}, {**partial_records[boundary.target_object_id],'observed_camera_name':2},
            {**partial_records[boundary.target_object_id],'visibility_conflict':True}):
        assert not any(q.predicate_id.startswith('image_boundary:') for q in image_relation_catalog(
            boundary,{boundary.target_object_id:uncertain})[0])
    boundary.provenance['scene_image_candidates'][1]['xyxy']=[0,5,20,18]
    blocked=next(q for q in image_relation_catalog(boundary,partial_records)[0]
        if q.predicate_id==edge.predicate_id)
    assert blocked.matches[rival] is None
    boundary.predicates=[q for q in boundary.predicates if q.predicate_id!=edge.predicate_id]+[blocked]
    proof=expression_candidates(boundary,[q for q in boundary.predicates if q.predicate_id in partial_text.predicate_ids],
        partial_text.visual_attributes,partial_records)
    assert rival in proof['image_only'] and not proof['camera_identity_proof']
    boundary.task.pop('image_position_policy')
    assert not any(q.predicate_id.startswith('image_boundary:') for q in image_relation_catalog(boundary,partial_records)[0])

    # Fully hidden boxes must not win a VISIBLE order. Unknowns, another
    # camera's observations and conflicts still cannot prove it. Recognizable
    # partial units CAN supply a separated image order.
    from app.quality.description_search import visible_image_orders
    native = {o.object_id for o in p.candidates}
    neighbor = p.candidates[1].object_id
    scope_boxes = [
        {'object_id':p.target_object_id,'category':'vehicle','camera_name':1,'xyxy':[100,40,140,140]},
        {'object_id':neighbor,'category':'vehicle','camera_name':1,'xyxy':[200,40,240,140]},
        {'object_id':rival,'category':'vehicle','camera_name':1,'xyxy':[0,40,40,140]}]
    hidden = {'target_visible':False,'visibility':'occluded',
        'visibility_reason':'Completely hidden by the foreground vehicle.', 'attributes':[], 'observed_camera_name':1}
    scope_observations = {p.target_object_id:{**clear().model_dump(),'observed_camera_name':1},
        neighbor:{**clear().model_dump(),'observed_camera_name':1}, rival:hidden}
    all_ids = native | {rival}
    p.provenance['scene_image_candidates'] = scope_boxes
    def target_order(records):
        rules, _ = image_relation_catalog(p, records)
        return next((r for r in rules if r.predicate_id == 'image_order:1:vehicle:leftmost_visible'
            and r.matches.get(p.target_object_id)), None)
    order = target_order(scope_observations)
    assert order is not None and 'leftmost visible vehicle' in order.phrase
    scoped=p.model_copy(deep=True)
    scoped.task['description_detail_policy']='confirmed-appearance-and-spatial-v1'
    scoped.predicates.append(order)
    scoped.provenance['admission_plan']={'passed':True,'unique_plan_found':True,
        'predicate_ids':['category:vehicle',order.predicate_id],'visual_attributes':clear().model_dump()['attributes']}
    from app.agent.harness import generation_messages
    writing=json.loads(generation_messages(scoped,None,scope_observations)[-1]['content'][0]['text'])
    constraint=next(c for c in writing['comparison_scope_constraints'] if c['predicate_id']==order.predicate_id)
    assert constraint['visible_only'] and constraint['comparison_group_attributes']==[]
    assert {'kind':'color','value':'white'} in constraint['target_only_recognition_attributes']
    from app.agent.harness import repair_comparison_scope
    wrong_scope=GeneratedExpression(expression='The leftmost visible white vehicle in the front view is to the left of the ego vehicle.',
        predicate_ids=['category:vehicle',order.predicate_id,'lateral:left'],visual_attributes=clear().attributes)
    rewritten,repair=repair_comparison_scope(scoped,wrong_scope)
    assert rewritten.expression=='The leftmost visible vehicle in the front view is white and to the left of the ego vehicle.'
    assert rewritten.predicate_ids==wrong_scope.predicate_ids and rewritten.visual_attributes==wrong_scope.visual_attributes
    assert repair['before']==wrong_scope.expression and repair['requires_two_blind_checks']
    assert repair_comparison_scope(scoped,rewritten)==(rewritten,None)
    unknown_order=scoped.model_copy(deep=True)
    unknown_order.predicates[-1].matches[p.target_object_id]=None
    assert repair_comparison_scope(unknown_order,wrong_scope)==(wrong_scope,None)
    for uncertain in ({}, {**hidden,'visibility':'uncertain'},
            {**hidden,'observed_camera_name':2}, {**hidden,'visibility_conflict':True}):
        assert target_order({**scope_observations, rival:uncertain}) is None
    assert target_order({**scope_observations, p.target_object_id:{
        **scope_observations[p.target_object_id], 'visibility':'partial'}}) is not None
    assert not visible_image_orders(1,[scope_boxes[0],{**scope_boxes[1],'xyxy':[130,40,240,140]}],
        scope_observations,native,all_ids,'test')  # Overlapping extents do not establish order.
    # A lower visible sign stays distinguishable when a still-lower projection
    # is explicitly fully hidden. Unknowns, overlaps and other-camera readings
    # must still block the new vertical visible order.
    sign_packet = p.model_copy(deep=True)
    sign_packet.target.category = 'sign'
    for candidate in sign_packet.candidates:
        if candidate.object_id in {p.target_object_id, neighbor}:
            candidate.category = 'sign'
    sign_packet.provenance['scene_image_candidates'] = [
        {**scope_boxes[0], 'category':'sign', 'xyxy':[100,400,200,600]},
        {**scope_boxes[1], 'category':'sign', 'xyxy':[130,100,170,200]},
        {**scope_boxes[2], 'category':'sign', 'xyxy':[0,500,80,600]}]
    sign_packet.provenance['source_image_candidates'] = [dict(c,camera_object_id=None)
        for c in sign_packet.provenance['scene_image_candidates']]
    def bottom_order(records):
        return next((q for q in image_relation_catalog(sign_packet,records)[0]
            if q.predicate_id == 'image_order:1:sign:bottommost_visible' and q.matches[p.target_object_id]), None)
    bottom = bottom_order(scope_observations)
    assert bottom is not None
    for uncertain in ({}, {**hidden,'visibility':'uncertain'},
            {**hidden,'observed_camera_name':2}, {**hidden,'visibility_conflict':True}):
        assert bottom_order({**scope_observations,rival:uncertain}) is None
    sign_packet.provenance['scene_image_candidates'][1]['xyxy'] = [130,390,170,480]
    assert bottom_order(scope_observations) is None
    sign_packet.provenance['scene_image_candidates'][1]['xyxy'] = [130,100,170,200]
    sign_packet.predicates = [*predicate_catalog(sign_packet.candidates), bottom,
        next(q for q in image_relation_catalog(sign_packet,scope_observations)[0]
            if q.predicate_id == 'image_visibility:1:sign')]
    subtype = VisualAttribute(kind='subtype',value='a-frame sign')
    sign_records = {**scope_observations, p.target_object_id:{
        **scope_observations[p.target_object_id], 'attributes':[subtype.model_dump()]}}
    sign_observation = VisualObservation.model_validate({k:v for k,v in sign_records[p.target_object_id].items()
        if k in VisualObservation.model_fields})
    plan = admission_plan(sign_packet,sign_observation,sign_records)
    assert plan['unique_plan_found']
    sign_packet.provenance['admission_plan'] = plan
    from app.agent.expression_rendering import render_verified_plan
    sign_expression = render_verified_plan(sign_packet,sign_records)
    assert sign_expression is not None and 'a-frame sign' in sign_expression.expression
    assert 'bottom' not in sign_expression.expression  # Visibility + appearance already identifies it.
    sign_packet.provenance['admission_plan'] = {**plan, 'predicate_ids':['category:sign',bottom.predicate_id]}
    assert render_verified_plan(sign_packet,sign_records) is None
    # Frozen historical expressions remain verifiable, but new drafts cannot
    # turn vertical image ranks back into prose.
    sign_expression = GeneratedExpression(expression='The visible a-frame sign at the bottom of the front view.',
        predicate_ids=['category:sign',bottom.predicate_id],visual_attributes=[subtype])
    sign_parse = GroundResult(candidate_ids=[],parsed_predicate_ids=sign_expression.predicate_ids,
        parsed_visual_attributes=sign_expression.visual_attributes,unsupported_claims=[])
    assert verify_tool_grounding(sign_packet,sign_expression,sign_parse,{},sign_records,
        target_appearance_verified=True)['diagnostic_pass']
    import app.agent.expression_rendering as rendering
    sign_packet.provenance['admission_plan'] = plan
    sign_variants = set()
    with monkeypatch.context() as style_patch:
        for seed in range(8):
            style_patch.setattr(rendering, '_style_seed', lambda p, value=seed: value)
            variant = render_verified_plan(sign_packet,sign_records)
            assert variant.predicate_ids == plan['predicate_ids']
            assert subtype in variant.visual_attributes
            assert 'front view' in variant.expression and 'visible' in variant.expression
            assert 'a-frame sign' in variant.expression and 'bottom' not in variant.expression
            variant_parse = GroundResult(candidate_ids=[],parsed_predicate_ids=variant.predicate_ids,
                parsed_visual_attributes=variant.visual_attributes,unsupported_claims=[])
            assert verify_tool_grounding(sign_packet,variant,variant_parse,{},sign_records,
                target_appearance_verified=True)['diagnostic_pass']
            sign_variants.add(variant.expression)
    assert len(sign_variants) >= 2
    unqualified = sign_expression.model_copy(update={'expression':sign_expression.expression.replace('visible ', '')})
    assert 'image_visibility_scope_not_explicit' in verify_tool_grounding(sign_packet,unqualified,
        sign_parse,{},sign_records,target_appearance_verified=True)['errors']
    from app.agent.harness import expression_language_errors, CLAIM_PARSE_GUIDE
    assert 'Always include parsed_visual_attributes' in CLAIM_PARSE_GUIDE
    bad_text = ('The visible pedestrian wearing a black top and black bottoms and holding a bag, '
        'whose outline reaches lowest among visible pedestrians wearing a black top and black bottoms '
        'and holding a bag in the front-left view, is ahead and to the left of the ego vehicle.')
    assert {'mechanical_referring_expression','repeated_appearance_description'} <= set(expression_language_errors(bad_text))
    assert not expression_policy_errors(bad_text)  # Existing frozen exports keep their eligibility.
    assert not expression_language_errors('The pedestrian wearing a black top and holding a bag ahead of the ego vehicle.')
    for wording in ('bottommost','bottom-most','bottom most','topmost','top most','lowermost',
                    'second from the bottom','second from top','lowest red','highest green','at the bottom of the front view',
                    'in the lower right corner of the image','lower right corner'):
        assert 'image_vertical_or_corner_wording' in expression_language_errors(f'The {wording} sign ahead of the ego vehicle.')
    assert not expression_language_errors('The pedestrian wearing a black top and blue bottoms to the left of the ego vehicle.')
    from app.quality.description_search import ordinary_reference_predicate,descriptive_attributes
    from app.agent.harness import draft_reference_errors
    from app.quality.admission import _cue_readability
    road = sign_packet.model_copy(deep=True)
    road.task.update(natural_description=True,description_detail_policy='confirmed-road-scene-v2')
    for suffix in ('topmost','bottommost','topmost_visible','bottommost_visible','second_from_top','second_from_bottom',
                   'top_edge_visible','bottom_edge_visible','height_visible','left_edge_visible'):
        rejected = bottom.model_copy(update={'predicate_id':f'image_order:1:sign:{suffix}'})
        assert not ordinary_reference_predicate(rejected)
        road.predicates = [*sign_packet.predicates,rejected]
        fluent = GeneratedExpression(expression='The green sign ahead of the ego vehicle.',
            predicate_ids=['category:sign',rejected.predicate_id],visual_attributes=[])
        assert draft_reference_errors(road,fluent) == ['non_scene_reference_cue']
    road.predicates = sign_packet.predicates
    retired_plan = {**plan,'predicate_ids':['category:sign',bottom.predicate_id]}
    road.provenance['admission_plan'] = retired_plan
    brief = json.loads(generation_messages(road,None,sign_records)[-1]['content'][0]['text'])
    assert brief['prechecked_unique_cue_plan'] is None
    assert road.provenance['admission_plan'] == retired_plan
    assert all(not fact['predicate_id'].endswith('bottommost_visible') for fact in brief['available_target_facts'])
    assert all(bottom.predicate_id not in option['predicate_ids'] for option in brief['prechecked_reference_options'])
    assert ordinary_reference_predicate(order)
    landmark_cue = bottom.model_copy(update={'predicate_id':'relative_left:landmark','kind':'relative_left'})
    assert _cue_readability(landmark_cue) < _cue_readability(order)
    three_details = [VisualAttribute(kind='subtype',value='a-frame sign'),
                     VisualAttribute(kind='color',value='green'),VisualAttribute(kind='shape',value='rectangular')]
    confirmed = {road.target_object_id:{**clear().model_dump(),'attributes':[a.model_dump() for a in reversed(three_details)]}}
    assert descriptive_attributes(road,confirmed) == three_details
    conflicting = {road.target_object_id:{**confirmed[road.target_object_id],
        'attributes':[*confirmed[road.target_object_id]['attributes'],{'kind':'color','value':'red'}]}}
    assert all(a.kind != 'color' for a in descriptive_attributes(road,conflicting))
    from app.quality.description_search import visible_extent_orders
    extent_boxes = [{**scope_boxes[0],'identity_source':'official_camera_lidar_association'},
        {**scope_boxes[1],'xyxy':[130,40,240,120]},scope_boxes[2]]
    extents = visible_extent_orders(1,extent_boxes,scope_observations,native,all_ids,'test')
    assert any(q.predicate_id.endswith('bottom_edge_visible') and q.matches[p.target_object_id] for q in extents)
    assert any(q.predicate_id.endswith('height_visible') and q.matches[p.target_object_id] for q in extents)
    # Keep old geometric proofs inspectable, but never fall back to technical
    # edge/height prose when writing a new natural expression.
    extent_packet = p.model_copy(deep=True)
    part_records = scope_observations
    extent_packet.predicates = [*predicate_catalog(p.candidates), *extents]
    for rule in (q for q in extents if q.matches.get(p.target_object_id)):
        extent_packet.provenance['admission_plan'] = {'passed':True,'unique_plan_found':True,
            'predicate_ids':['category:vehicle',rule.predicate_id], 'visual_attributes':[]}
        assert render_verified_plan(extent_packet,part_records) is None
    assert not visible_extent_orders(1,[{**extent_boxes[0],'identity_source':'official_lidar_box_projection'},
        *extent_boxes[1:]],scope_observations,native,all_ids,'test')
    assert not visible_extent_orders(1,[extent_boxes[0],{**extent_boxes[1],'xyxy':[130,40,240,135]},
        scope_boxes[2]],scope_observations,native,all_ids,'test')  # Small edge/height gaps are ties.
    visible_rule = next(q for q in image_relation_catalog(p,scope_observations)[0]
        if q.predicate_id == 'image_visibility:1:vehicle')
    p.predicates = [*predicate_catalog(p.candidates), order, visible_rule]
    scoped_ids = ['category:vehicle','longitudinal:front','lateral:left',order.predicate_id]
    scoped_expression = GeneratedExpression(expression='The leftmost visible white vehicle in the front view, ahead and to the left of the ego vehicle.',
        predicate_ids=scoped_ids, visual_attributes=clear().attributes)
    scoped_parse = GroundResult(candidate_ids=[], parsed_predicate_ids=scoped_ids, unsupported_claims=[],
        parsed_visual_attributes=clear().attributes)
    scoped_check = verify_tool_grounding(p,scoped_expression,scoped_parse,{},scope_observations,
        target_appearance_verified=True)
    assert scoped_check['diagnostic_pass'], scoped_check['errors']
    redundant_parse = scoped_parse.model_copy(update={'parsed_predicate_ids':[
        *scoped_parse.parsed_predicate_ids,visible_rule.predicate_id]})
    assert verify_tool_grounding(p,scoped_expression,redundant_parse,{},scope_observations,
        target_appearance_verified=True)['diagnostic_pass']  # Visible order entails visibility.
    missing_scope = scoped_expression.model_copy(update={'expression':scoped_expression.expression.replace('visible ', '')})
    assert 'image_visibility_scope_not_explicit' in verify_tool_grounding(p,missing_scope,scoped_parse,{},
        scope_observations,target_appearance_verified=True)['errors']
    # A recognizable person must not be lost because a tiny, partly seen person
    # shares one garment color. Rank verified garment groups by visible RGB
    # extents, retaining unknown rivals' complete bounds and the same gap guard.
    ped = p.model_copy(deep=True)
    ped.target.category = 'pedestrian'
    for candidate in ped.candidates:
        if candidate.category == 'vehicle':
            candidate.category = 'pedestrian'
    upper = VisualAttribute(kind='upper_clothing_color',value='black')
    lower = VisualAttribute(kind='lower_clothing_color',value='black')
    ped_boxes = [
        {'object_id':ped.target_object_id,'category':'pedestrian','camera_name':1,
         'camera_object_id':None,'xyxy':[100,100,160,400],'identity_source':'official_camera_lidar_association'},
        {'object_id':neighbor,'category':'pedestrian','camera_name':1,
         'camera_object_id':None,'xyxy':[0,80,80,480],'identity_source':'official_camera_lidar_association'},
        {'object_id':rival,'category':'pedestrian','camera_name':1,
         'camera_object_id':None,'xyxy':[110,20,140,50],'identity_source':'official_camera_lidar_association'}]
    ped.provenance.update(scene_image_candidates=ped_boxes,source_image_candidates=ped_boxes)
    ped_records = {
        ped.target_object_id:{**clear().model_dump(),'observed_camera_name':1,'attributes':[upper.model_dump(),lower.model_dump()]},
        neighbor:{**clear().model_dump(),'observed_camera_name':1,'attributes':[
            {'kind':'upper_clothing_color','value':'green'},lower.model_dump()]},
        rival:{'target_visible':True,'visibility':'partial','attributes':[upper.model_dump()],'observed_camera_name':1}}
    def clothing_extent(records, boxes=ped_boxes):
        probe = ped.model_copy(deep=True)
        probe.provenance['scene_image_candidates'] = boxes
        return next((rule for rule in image_relation_catalog(probe,records)[0]
            if '~upper_clothing_color=black~lower_clothing_color=black:bottom_edge_visible' in rule.predicate_id
            and rule.matches.get(ped.target_object_id)), None)
    garment_rule = clothing_extent(ped_records)
    assert garment_rule is not None  # The tiny rival remains but cannot reach the target's lower edge.
    ped.predicates = [*predicate_catalog(ped.candidates), *image_relation_catalog(ped,ped_records)[0]]
    primary = VisualObservation.model_validate({k:v for k,v in ped_records[ped.target_object_id].items()
        if k in VisualObservation.model_fields})
    ped_plan = admission_plan(ped,primary,ped_records)
    assert ped_plan['passed'] and not ped_plan['unique_plan_found']
    ped.provenance['admission_plan'] = ped_plan
    assert render_verified_plan(ped,ped_records) is None
    # The original edge proof is still readable and validated for frozen records;
    # ordinary writing must find another cue/view rather than change its meaning.
    ped.provenance['admission_plan'] = {'passed':True,'unique_plan_found':True,
        'predicate_ids':['category:pedestrian',garment_rule.predicate_id]}
    caption = GeneratedExpression(
        expression='Among visible pedestrians wearing a black top and black bottoms in the front view, the one extending lowest.',
        predicate_ids=['category:pedestrian',garment_rule.predicate_id],visual_attributes=[upper,lower])
    assert expression_language_errors(caption.expression)
    ped_parse = GroundResult(candidate_ids=[],parsed_predicate_ids=caption.predicate_ids,
        parsed_visual_attributes=caption.visual_attributes,unsupported_claims=[])
    assert verify_tool_grounding(ped,caption,ped_parse,{},ped_records,target_appearance_verified=True)['diagnostic_pass']
    dropped_clothing = ped_parse.model_copy(update={'parsed_visual_attributes':[]})
    assert 'image_order_scope_not_explicit' in verify_tool_grounding(ped,caption,dropped_clothing,{},ped_records,
        target_appearance_verified=True)['errors']
    # New runs may complete redundant fields ONLY from this blind reader's
    # independently chosen qualified predicate; old frozen runs stay strict.
    from app.agent.harness import normalize_parsed_scope_attributes
    normalized, receipt = normalize_parsed_scope_attributes(ped.predicates,dropped_clothing)
    assert set((a.kind,a.value) for a in normalized.parsed_visual_attributes) == {(upper.kind,upper.value),(lower.kind,lower.value)}
    assert dropped_clothing.parsed_visual_attributes == [] and receipt['additions']
    current_ped = ped.model_copy(deep=True)
    current_ped.task['claim_normalization_policy']='compound-claim-entailment-v1'
    entailed = verify_tool_grounding(current_ped,caption,dropped_clothing,{},ped_records,target_appearance_verified=True)
    assert entailed['diagnostic_pass'] and entailed['raw_parsed_visual_attributes']==[]
    conflicting = dropped_clothing.model_copy(update={'parsed_visual_attributes':[
        VisualAttribute(kind='upper_clothing_color',value='green')]})
    assert 'visual_claims_disagree_with_text' in verify_tool_grounding(current_ped,caption,conflicting,{},ped_records,
        target_appearance_verified=True)['errors']
    unqualified = dropped_clothing.model_copy(update={'parsed_predicate_ids':['category:pedestrian']})
    assert normalize_parsed_scope_attributes(ped.predicates,unqualified)==(unqualified,None)
    # #1778: a compound appearance condition already means BOTH color/type.
    # Complete only the writer's own redundant fields, preserve conflicts and
    # still require an independent matching parse (including omitted words).
    from app.agent.harness import normalize_writer_scope_attributes, WRITER_SCOPE_POLICY
    writer_packet=current_ped.model_copy(deep=True)
    writer_packet.task['writer_claim_policy']=WRITER_SCOPE_POLICY
    attrs,repair=normalize_writer_scope_attributes(writer_packet,caption.predicate_ids,[])
    fixed=caption.model_copy(update={'visual_attributes':attrs})
    assert repair['raw_visual_attributes']==[] and repair['requires_appearance_and_two_blind_checks']
    assert verify_tool_grounding(writer_packet,fixed,dropped_clothing,{},ped_records,True)['diagnostic_pass']
    legacy_writer=current_ped.model_copy(deep=True)
    legacy_writer.task.pop('writer_claim_policy',None)
    assert normalize_writer_scope_attributes(legacy_writer,caption.predicate_ids,[]) == ([],None)
    assert normalize_writer_scope_attributes(writer_packet,['category:pedestrian'],[]) == ([],None)
    # Backpack fabric color is a separate observed attribute, never jacket or
    # trouser color. Only its own explicit claim entails backpack presence.
    from app.agent.appearance import filter_visual_candidates
    green_pack=VisualAttribute(kind='backpack_color',value='green')
    assert CompactAttribute.model_validate({'k':'backpack_color','v':'green'}).model_dump()==green_pack.model_dump()
    pack_attrs,pack_receipt=normalize_writer_scope_attributes(writer_packet,['category:pedestrian'],[green_pack])
    assert VisualAttribute(kind='feature',value='backpack') in pack_attrs and pack_receipt
    parsed_pack=GroundResult(candidate_ids=[],parsed_predicate_ids=['category:pedestrian'],
        parsed_visual_attributes=[green_pack],unsupported_claims=[])
    completed_pack,_=normalize_parsed_scope_attributes(writer_packet.predicates,parsed_pack)
    assert set((a.kind,a.value) for a in completed_pack.parsed_visual_attributes)=={('backpack_color','green'),('feature','backpack')}
    assert appearance_phrase('pedestrian',pack_attrs)=='the pedestrian with a green backpack'
    assert filter_visual_candidates(['black-pack','unknown-pack'],[green_pack],{
        'black-pack':{'target_visible':True,'attributes':[{'kind':'backpack_color','value':'black'}]}})[0]==['unknown-pack']
    wrong=[VisualAttribute(kind='upper_clothing_color',value='green')]
    attrs,_=normalize_writer_scope_attributes(writer_packet,caption.predicate_ids,wrong)
    assert wrong[0] in attrs and upper not in attrs
    assert 'visual_claims_disagree_with_text' in verify_tool_grounding(writer_packet,
        caption.model_copy(update={'visual_attributes':attrs}),ped_parse,{},ped_records,True)['errors']
    assert not verify_tool_grounding(writer_packet,fixed,unqualified,{},ped_records,True)['diagnostic_pass']
    # #695: a literal appearance condition and the same attributes plus source
    # visibility are equivalent. Derive each speaker's closure independently;
    # never borrow target facts, another camera, a missing attribute or a rank.
    from app.agent.harness import appearance_claim_closure, APPEARANCE_EQUIVALENCE_POLICY
    visible=Predicate(predicate_id='image_visibility:3:sign',kind='image_relation',value='3:sign',
        phrase='the sign visible in the front-right view',source='target-independent test inventory',matches={})
    appearance=visible.model_copy(update={'predicate_id':'image_attribute:3:sign~color=white~inscription=PASSENGER LOADING ONLY',
        'phrase':'the white sign reading PASSENGER LOADING ONLY in the front-right view'})
    attrs=[VisualAttribute(kind='color',value='white'),VisualAttribute(kind='inscription',value='PASSENGER LOADING ONLY')]
    sentence='The white sign reading PASSENGER LOADING ONLY visible in the front-right view'
    catalog=[visible,appearance]
    first,_=appearance_claim_closure(catalog,['category:sign',visible.predicate_id],attrs,sentence)
    second,_=appearance_claim_closure(catalog,['category:sign',appearance.predicate_id],attrs,sentence)
    assert first==second and appearance.predicate_id in first and visible.predicate_id in first
    for missing in ([],attrs[:1],[VisualAttribute(kind='color',value='red'),attrs[1]],
            [*attrs,VisualAttribute(kind='color',value='red')]):
        assert appearance.predicate_id not in appearance_claim_closure(catalog,
            ['category:sign',visible.predicate_id],missing,sentence)[0]
    assert visible.predicate_id not in appearance_claim_closure(catalog,
        ['category:sign',appearance.predicate_id],attrs,sentence.replace('visible ',''))[0]
    assert appearance.predicate_id not in appearance_claim_closure(catalog,
        ['category:sign','image_visibility:1:sign'],attrs,sentence)[0]
    ordered=appearance.model_copy(update={'predicate_id':appearance.predicate_id.replace('image_attribute:','image_order:')+':leftmost_visible'})
    assert ordered.predicate_id not in appearance_claim_closure([*catalog,ordered],first,attrs,sentence)[0]
    assert APPEARANCE_EQUIVALENCE_POLICY=='appearance-claim-equivalence-v1'
    extra = caption.model_copy(update={'visual_attributes':[*caption.visual_attributes,VisualAttribute(kind='feature',value='handheld bag')]})
    assert 'visual_claims_disagree_with_text' in verify_tool_grounding(current_ped,extra,dropped_clothing,{},ped_records,
        target_appearance_verified=True)['errors']
    horizontal = garment_rule.model_copy(update={
        'predicate_id':'image_order:1:pedestrian~upper_clothing_color=black:rightmost_visible',
        'value':'1:rightmost_visible',
        'phrase':'the rightmost visible pedestrian wearing a black top in the front view'})
    simpler = ped.model_copy(deep=True)
    simpler.predicates.append(horizontal)
    concise_plan = admission_plan(simpler,primary,ped_records)
    assert horizontal.predicate_id in concise_plan['predicate_ids']
    reordered = simpler.model_copy(update={'predicates':list(reversed(simpler.predicates))})
    assert admission_plan(reordered,primary,ped_records)['predicate_ids'] == concise_plan['predicate_ids']
    for bottom in (390,416,520):
        competing_boxes = [*ped_boxes[:2],{**ped_boxes[2],'xyxy':[110,20,140,bottom]}]
        assert clothing_extent(ped_records,competing_boxes) is None
    # Fully hidden exclusions must be explicit, unconflicted and bound to this camera.
    competing_boxes = [*ped_boxes[:2],{**ped_boxes[2],'xyxy':[110,20,140,520]}]
    assert clothing_extent({**ped_records,rival:hidden},competing_boxes) is not None
    for uncertain in ({**hidden,'visibility':'uncertain'}, {**hidden,'observed_camera_name':2},
            {**hidden,'visibility_conflict':True}):
        assert clothing_extent({**ped_records,rival:uncertain},competing_boxes) is None
    assembly_id = 'image:1:assembly'
    group_record = {'object_id':assembly_id,'category':'vehicle','camera_name':1,'xyxy':[90,10,250,150],
        'identity_source':'native_camera_box_group_point_correspondence',
        'native_unit_object_ids':[p.target_object_id,neighbor],
        'native_unit_point_support':{p.target_object_id:25,neighbor:25}}
    grouped_packet = p.model_copy(deep=True)
    grouped_packet.provenance['scene_image_candidates'] = [*scope_boxes[:2],group_record]
    base = [q for q in predicate_catalog(p.candidates) if q.predicate_id in
        ['category:vehicle','longitudinal:front','lateral:left']]
    proof = expression_candidates(grouped_packet,base,[],{})
    assert proof['native'] == [p.target_object_id] and not proof['image_only']
    group_orders = visible_extent_orders(1,[group_record],{assembly_id:clear().model_dump()},
        native,native|{assembly_id},'test')
    assert group_orders and all(q.matches[p.target_object_id] and q.matches[neighbor] for q in group_orders)
    from app.quality.description_search import scoped_image_orders
    from app.quality.description_search import confirmed_attributes, image_order_attributes, appearance_phrase
    from app.schemas.contracts import Predicate
    # Attribute scopes must generalize across categories, cameras and layout,
    # with no dependency on a particular ID, garment color or carried object.
    for category, facts, different, absent, group_phrase in (
            ('sign', [('shape','round')], [('shape','triangular')], [], 'round signs'),
            ('vehicle', [('color','blue'),('subtype','bus'),('feature','roof rack')],
                [('color','red'),('subtype','bus')], ['roof rack'], 'blue buses with a roof rack'),
            ('vehicle', [('feature','roof rack')], [], ['roof rack'], 'vehicles with a roof rack'),
            ('pedestrian', [('upper_clothing_color','red'),('feature','handheld bag')],
                [('upper_clothing_color','green')], ['handheld bag'], 'pedestrians wearing a red top and holding a bag'),
            ('pedestrian', [('upper_clothing_color','black'),('backpack_color','green'),('feature','backpack')],
                [('upper_clothing_color','black'),('backpack_color','black'),('feature','backpack')], [],
                'pedestrians wearing a black top with a green backpack'),
            ('cyclist', [('upper_clothing_color','blue'),('subtype','bicycle')],
                [('upper_clothing_color','red'),('subtype','motorcycle')], [], 'cyclists on a bicycle wearing a blue top')):
        expected = set(facts)
        for camera in range(1,6):
            boxes = [{**b,'category':category,'camera_name':camera} for b in ped_boxes]
            boxes[2]['category'] = 'unknown'
            records = {
                p.target_object_id:{**clear().model_dump(),'attributes':[{'kind':k,'value':v} for k,v in facts]},
                neighbor:{**clear().model_dump(),'attributes':[{'kind':k,'value':v} for k,v in different],
                    'absent_features':absent},
                rival:{'target_visible':True,'visibility':'partial','attributes':[]}}
            ids = native | set(records)
            rules = scoped_image_orders(camera,boxes,records,native,ids,'test',allow_partial=True)
            rule = next(q for q in rules if q.predicate_id.endswith(':bottom_edge_visible')
                and q.matches.get(p.target_object_id)
                and {(a.kind,a.value) for a in image_order_attributes(q)} == expected)
            # Input ordering, arbitrary IDs and a translated image preserve proof.
            renamed = {oid:f'arbitrary-{i}' for i,oid in enumerate(sorted(ids))}
            shifted = [{**b,'object_id':renamed[b['object_id']],
                'xyxy':[v+37 for v in b['xyxy']]} for b in reversed(boxes)]
            reordered = {renamed[oid]:record for oid,record in reversed(list(records.items()))}
            variants = scoped_image_orders(camera,shifted,reordered,{renamed[oid] for oid in native},
                set(renamed.values()),'test',allow_partial=True)
            variant = next(q for q in variants if q.predicate_id == rule.predicate_id)
            assert variant.matches[renamed[p.target_object_id]]
            assert {q.predicate_id:{renamed[oid]:match for oid,match in q.matches.items()} for q in rules} == {
                q.predicate_id:q.matches for q in variants}
            blocked = [*boxes[:2],{**boxes[2],'xyxy':[110,20,140,390]}]
            assert not any(q.predicate_id == rule.predicate_id for q in scoped_image_orders(
                camera,blocked,records,native,ids,'test',allow_partial=True))
            probe = ped.model_copy(deep=True)
            probe.target.category = category
            probe.views = [probe.views[0].model_copy(update={'camera_name':camera,'camera_label':probe.source_view})]
            probe.predicates = [Predicate(predicate_id=f'category:{category}',kind='category',value=category,
                phrase=category,matches={p.target_object_id:True},source='test'),rule]
            probe.provenance['admission_plan'] = {'passed':True,'unique_plan_found':True,
                'predicate_ids':[q.predicate_id for q in probe.predicates],'visual_attributes':[]}
            assert render_verified_plan(probe,records) is None
            from scripts.extract_comparison_supervision import semantics
            legacy = {'expression':f'Among visible {group_phrase} in the ' +
                {1:'front view',2:'front-left view',3:'front-right view',4:'left-side view',5:'right-side view'}[camera] +
                ', the one extending lowest.', 'camera_id':camera}
            auxiliary = semantics(legacy, rule.predicate_id, {'visual_attributes':[]})
            assert {(a['kind'],a['value']) for a in auxiliary['qualifiers']} == expected
            assert auxiliary['operator'] == 'bottom_edge_visible'
    conflicting = {'target_visible':True,'attributes':[
        {'kind':'color','value':'red'},{'kind':'color','value':'blue'},
        {'kind':'feature','value':'roof rack'}],'absent_features':['roof rack']}
    assert confirmed_attributes('vehicle',conflicting) == []
    assert appearance_phrase('vehicle',[VisualAttribute(kind='subtype',value='excavator body')],
        plural=True) == 'the excavator bodies'
    sign_attrs = [VisualAttribute(kind='shape',value='rectangular'),
        VisualAttribute(kind='feature',value='visible rear face'),
        VisualAttribute(kind='feature',value='pole mounted')]
    sign_phrase = appearance_phrase('sign',sign_attrs)
    assert sign_phrase == 'the rectangular sign mounted on a pole with its rear surface visible'
    from app.quality.spatial_relations import sentence_relation_keys
    assert not sentence_relation_keys(sign_phrase)
    sign_packet = p.model_copy(deep=True)
    sign_packet.target.category = 'sign'
    sign_packet.task.update(target_naming_policy='explicit-target-category-v1',manual_feedback_policy=FEEDBACK_POLICY)
    sign_plan = {'predicate_ids':[], 'visual_attributes':[a.model_dump() for a in sign_attrs]}
    assert 'target_category_omitted' in writer_contract_errors(sign_packet,sign_plan,'The rectangular object in the front-right view.')
    assert 'target_category_omitted' in writer_contract_errors(sign_packet,sign_plan,'The object behind a sign.')
    assert 'target_category_omitted' not in writer_contract_errors(sign_packet,sign_plan,sign_phrase)
    for sentence in ('A yellow pole mounted sign is visible in the front-right view.',
            'A small, yellow pole-mounted sign is visible in the front-right view.',
            'The yellow pole-mounted sign in the front-right view.',
            'The visible yellow sign in the front-right view.',
            'In the front-right view, the yellow pole mounted sign.',
            'From the front-right view, a yellow pole-mounted sign is visible.',
            'There is a yellow pole-mounted sign in the front-right view.'):
        assert 'target_category_omitted' not in writer_contract_errors(sign_packet,sign_plan,sentence)
    for sentence in ('The object mounted on a sign.', 'The person holding a sign.',
            'The yellow object beside a sign.', 'In the front-right view, the object behind a sign.'):
        assert 'target_category_omitted' in writer_contract_errors(sign_packet,sign_plan,sentence)
    from app.quality.joint_reference import writer_messages
    message = writer_messages(sign_packet,sign_plan,None,{})[-1]
    assert json.loads(message['content'][0]['text'])['target_category'] == 'sign'
    assert confirmed_attributes('sign',{'target_visible':True,'attributes':[a.model_dump() for a in sign_attrs]})
    assert not scoped_image_orders(1,[group_record],{assembly_id:clear().model_dump()},
        native,native|{assembly_id},'test',allow_partial=True)
    unit_orders = visible_image_orders(1,[*scope_boxes[:2],group_record],scope_observations,
        native,native|{assembly_id},'test')
    assert any(q.predicate_id.endswith('leftmost_visible') and q.matches[p.target_object_id] for q in unit_orders)
    assert not visible_image_orders(1,[scope_boxes[0],group_record],scope_observations,
        native,native|{assembly_id},'test')  # Missing member bounds cannot remove the assembly competitor.
    ambiguous = expression_candidates(grouped_packet,[q for q in base if q.kind=='category'],[],{})
    assert len(ambiguous['native']) > 1 and not ambiguous['camera_identity_proof']
    grouped_packet.provenance['scene_image_candidates'][-1]['native_unit_point_support'][neighbor] = 19
    assert expression_candidates(grouped_packet,base,[],{})['image_only'] == [assembly_id]
    p.task.pop('image_inventory_policy')
    with pytest.raises(ValueError, match='complete image inventory'):
        initial_run(p, SimpleNamespace(scope='real-model'), 0, 4)


def test_audit_release_and_training_require_matching_annotation_configuration(scene, monkeypatch):
    _check_paired_paper_study(scene[0].parent/'paper',monkeypatch)
    from copy import deepcopy
    import hashlib
    from app.quality.research import CohortSpec, PROTOCOL, wilson, validate_release_report
    from app.schemas.training_contracts import TrainingSample
    from app.datasets.training import load_training_sample

    folder, _, manifest = scene
    root = folder.parent
    store = RunStore(root/'runs/runs.sqlite3')
    research = ResearchStore(store, root)
    configuration = {'model':'model-A', 'base_url':'local://provider-A', 'calls_per_target':12, 'max_repairs':2}
    p = packet()
    p.scene = SceneEvidence(scene_id=manifest['scene_id'], manifest_path='scene/scene.json',
        manifest_sha256=file_sha256(folder/'scene.json'), total_points=3, target_points=1)
    state = {'run_id':'b'*24, 'provider_scope':'real-model', 'status':'calibration_hold',
        'model':configuration['model'], 'base_url':configuration['base_url'], 'max_repairs':2,
        'model_call_budget':8, 'current_packet':p.model_dump(mode='json'), 'final_expression':'The vehicle ahead.',
        'diagnostic_pass':True}
    with store.connect() as db:
        db.execute('INSERT INTO runs(run_id,snapshot) VALUES(?,?)', (state['run_id'],json.dumps(state)))
    batch = {'job_id':'c'*24, 'status':'completed', 'method_sha256':'d'*64,
        'spec':{'prepare_only':False, 'provider':{k:configuration[k] for k in ('model','base_url')},
            'calls_per_target':12, 'max_repairs':2},
        'items':[{'run_id':state['run_id'], 'target':{'context':p.context,'status':'selected'}}]}
    cohort = research.create(CohortSpec(job_id=batch['job_id'], sample_size=1), batch)
    frozen = research.private(cohort['cohort_id'])
    assert frozen['frozen']['method_configuration'] == configuration
    assert research.compute(frozen)['method_configuration'] == configuration
    with monkeypatch.context() as check:
        check.setattr(store, 'snapshot', lambda _: {**state, 'model':'different-model'})
        with pytest.raises(ValueError, match='run differs'):
            research.create(CohortSpec(job_id=batch['job_id']), batch)

    # Synthetic judgments exercise only certificate consistency, never research metrics.
    target_box = {'center_m':[0,0,0], 'size_m':[2,2,2], 'heading_rad':0}
    rows = [{'run_id':str(i), 'case_id':str(i), 'diagnostic_pass':True, 'accepted':True} for i in range(35)]
    rows[0].update(target_id='native-track',packet_id='a'*24,scene_sha256=p.scene.manifest_sha256,
        expression_sha256=hashlib.sha256(b'The vehicle ahead.').hexdigest(), target_box=target_box)
    report = {'protocol':PROTOCOL, 'complete':True, 'calibration_gate_passed':True,
        'prior_exposure_excluded':0, 'policy':{'min_reviewed':30,'max_error_upper95':0.1},
        'automatic_candidate_error':wilson(0,35), 'reviewed_independent':35, 'cases':rows,
        'method_sha256':batch['method_sha256'], 'method_configuration':configuration, 'contexts':['context']}
    validate_release_report(report)
    legacy = {k:v for k,v in report.items() if k != 'method_configuration'}
    with pytest.raises(ValueError):
        validate_release_report(legacy)

    def audited(purpose, context):
        return {'status':'sealed', 'spec':{'purpose':purpose,'text_version':'final'},
            'report':{**deepcopy(report),'contexts':[context]},
            'frozen':{'contexts':[context], 'method_sha256':batch['method_sha256'], 'protocol':PROTOCOL,
                'method_configuration':dict(configuration), 'cases':[]}}
    calibration, evaluation = audited('calibration','context'), audited('evaluation','held-out')
    monkeypatch.setattr(research, 'private', lambda key:calibration if key == 'cal' else evaluation)
    # Equal configurations reach label selection; they must not be rejected as a mismatch.
    with pytest.raises(ValueError, match='No independently audited accepted calibration cases'):
        research.release('cal','eval')
    for key, value in [('model','model-B'), ('base_url','local://provider-B'),
                       ('calls_per_target',8), ('max_repairs',1)]:
        changed = {**configuration, key:value}
        evaluation['frozen']['method_configuration'] = changed
        evaluation['report']['method_configuration'] = changed
        with pytest.raises(ValueError, match='configurations differ'):
            research.release('cal','eval')
    evaluation['frozen']['method_configuration'] = dict(configuration)
    with pytest.raises(ValueError, match='differs from its frozen'):
        research.release('cal','eval')

    def training_record(validation_configuration):
        validation = {**deepcopy(report), 'contexts':['held-out'], 'method_configuration':validation_configuration}
        atomic_json(root/'calibration.json',report)
        atomic_json(root/'validation.json',validation)
        return TrainingSample(sample_id='e'*24,context='context',frame_timestamp_micros=1,
            track_id='native-track',category='vehicle',official_split='training',
            scene={'path':'scene/scene.json','sha256':p.scene.manifest_sha256}, expression='The vehicle ahead.',
            task=TaskProfile(input_capabilities=('geometry','rgb')), required_capabilities=('geometry','rgb'),
            target_box=target_box, annotation_run_id='0',evidence_packet_id='a'*24,
            quality={'oracle_checks_passed':True,'counterfactual_checks_passed':None,
                'source_alignment_verified':True,'independent_risk_calibration':True,
                'calibration_report':{'path':'calibration.json','sha256':file_sha256(root/'calibration.json')},
                'validation_report':{'path':'validation.json','sha256':file_sha256(root/'validation.json')}})
    inputs, _, _ = load_training_sample(root, training_record(configuration))
    assert len(inputs['images']) == 5
    with pytest.raises(ValueError, match='release methods'):
        load_training_sample(root, training_record({**configuration,'model':'model-B'}))


def _check_paired_paper_study(root,monkeypatch):
    from app.quality.paper_study import PaperStudyStore,StudySpec,Judgment,BASELINE,baseline_messages,reserved_for_other
    from app.agent.agent_runtime import run_agent
    from app.agent.batch import method_fingerprint
    from app.quality.research import digest
    p=strict_packet(root/'evidence',monkeypatch)
    p.task['research_baseline_policy']=BASELINE
    class FixedCalls(LocalCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            self.count+=1
            assert 'candidates' in json.loads(messages[1]['content'][0]['text'])
            atomic_json(folder/'result.json',{'status':'completed','usage':{'prompt_tokens':30,'completion_tokens':10}})
            return schema(expression='The white vehicle ahead.')
    calls=FixedCalls()
    result=run_agent(p,calls,0,root/'fixed',1,13)
    assert calls.count==result['model_calls_used']==1 and result['research_baseline_delivered']
    assert not result['diagnostic_pass'] and not result['auto_published'] and not result['attempts'][0]['ground_checks']
    assert run_agent(p,calls,0,root/'fixed',1,13)==result and calls.count==1
    # The baseline gets the frozen original pixels/geometry, never acquired observations.
    prompt=baseline_messages(p,root/'evidence')
    assert 'output_schema' not in json.loads(prompt[1]['content'][0]['text'])
    store=RunStore(root/'runtime/a/runs.sqlite3');batches=BatchStore(store);papers=PaperStudyStore(store,root)
    job='c'*24;sid='d'*24
    spec=StudySpec(request_id=sid,title='Synthetic protocol regression',sample_size=2,dataset_split='train',reviewers=['A','B','C'])
    with store.connect() as db:
        db.execute('INSERT INTO paper_studies VALUES(?,?,?,?,?,?,NULL)',(sid,job,spec.model_dump_json(),method_fingerprint(),'ready',time.time()))
        db.executemany('INSERT INTO paper_reviewers VALUES(?,?,?,?)',[(f'token-{i}',sid,i,n) for i,n in enumerate(spec.reviewers)])
        db.execute('INSERT INTO paper_targets VALUES(?,?,?,?)',(p.context,p.target_object_id,sid,'1'*24))
        db.execute("INSERT INTO batches VALUES(?,?,?,?,'completed',NULL,NULL,NULL,?)",(job,json.dumps({'provider':{'model':'fixture','base_url':'local://fixture'}}),'a'*64,method_fingerprint(),time.time()))
    assert reserved_for_other(store,'e'*24,{'context':p.context,'object_id':p.target_object_id})
    assert not reserved_for_other(store,job,{'context':p.context,'object_id':p.target_object_id})
    assert not papers.inbox('token-2')['cases']
    items=[]
    for ordinal in (1,2):
        packet=p.model_copy(deep=True);packet.context=f'context-{ordinal}';packet.task.pop('research_baseline_policy')
        agent=ident(987000+ordinal);base=ident(988000+ordinal)
        for method,rid in [('agent',agent),('baseline',base)]:
            state={**result,'run_id':rid,'provider_scope':'real-model','batch_id':job,
                'status':'calibration_hold' if method=='agent' and ordinal==1 else 'needs_evidence',
                'current_packet':packet.model_dump(mode='json'),'model_calls_used':1,
                'diagnostic_pass':ordinal==1 and method=='agent','research_baseline_delivered':ordinal==1 and method=='baseline',
                'attempts':result['attempts'] if ordinal==1 else [],'final_expression':result['final_expression'] if ordinal==1 else None}
            store.create(state)
            atomic_json(store.path.parent/rid/'model/00-test/result.json',{'usage':{'prompt_tokens':30,'completion_tokens':10}})
        target={'target_id':str(ordinal)*24,'context':packet.context,'object_id':packet.target_object_id,'status':'selected'}
        item={'ordinal':ordinal,'target':target,'run_id':agent,'status':'needs_evidence','reason':None};items.append(item)
        with store.connect() as db:
            db.execute('INSERT INTO batch_items(job_id,ordinal,target,status,run_id) VALUES(?,?,?,?,?)',(job,ordinal,json.dumps(target),'needs_evidence',agent))
            db.execute('INSERT INTO paper_trials VALUES(?,?,?,?,?,?)',(sid,ordinal,packet.model_dump_json(),digest({'initial':ordinal}),json.dumps([agent]),base))
    batch={'job_id':job,'status':'completed','executor_active':False,'items':items,'spec':{'model':'fixture'}}
    service=SimpleNamespace(store=SimpleNamespace(detail=lambda _:batch,execution_progress=batches.execution_progress))
    detail=papers.freeze(sid,service)
    assert detail['cases']==4 and detail['status']=='review'
    cases,reviews,people=papers.records(sid);visible=[c for c in cases if c['payload']['expression']]
    assert len(visible)==2 and all('method' not in c for c in papers.inbox('token-0')['cases'])
    answer=Judgment(factual_correct=True,unique=True,input_sufficient=True,prior_target_exposure=False,notes='',elapsed_seconds=15)
    for token in ('token-0','token-1'):
        c=visible[0]
        with pytest.raises(ValueError,match='定位'):papers.answer(token,c['case_id'],answer)
        papers.select(token,c['case_id'],[c['payload']['expected']])
        assert not papers.case(token,c['case_id'])[2]  # Must finish ALL paired localizations before ANY reveal.
        with pytest.raises(ValueError,match='锁定'):papers.select(token,c['case_id'],[])
        for c in visible[1:]:papers.select(token,c['case_id'],[c['payload']['expected']])
        assert papers.case(token,c['case_id'])[2]
        for c in visible:
            judged=answer.model_copy(update={'factual_correct':not(token=='token-1' and c['method']=='baseline')})
            papers.answer(token,c['case_id'],judged)
    with pytest.raises(ValueError,match='裁定'):papers.seal(sid,service)
    adjud=papers.inbox('token-2');assert len(adjud['cases'])==1
    disputed=next(c for c in cases if c['case_id']==adjud['cases'][0]['case_id'])
    assert disputed['method']=='baseline'
    papers.select('token-2',disputed['case_id'],[disputed['payload']['expected']])
    papers.answer('token-2',disputed['case_id'],answer.model_copy(update={'factual_correct':False}))
    report=papers.seal(sid,service)['report']
    assert report['methods']['agent']['accepted_validity']['value']==1
    assert report['methods']['agent']['automatic_valid_coverage']['value']==.5
    assert report['methods']['baseline']['automatic_completion_rate']['value']==.5
    assert report['methods']['baseline']['automatic_valid_coverage']['value']==0
    assert report['methods']['agent']['cost']['calls']==2  # Includes the no-output task.
    assert report['methods']['agent']['cost']['estimated_cost'] is None
    assert report['paired_coverage_difference']==.5 and len(report['rows'])==4
    assert papers.export_csv(sid).startswith('\ufeffordinal,method,expression')
    with pytest.raises(ValueError,match='封存'):papers.select('token-0',visible[0]['case_id'],[])
    # Protection applies even to direct deletion APIs, not just the tab.
    with pytest.raises(ValueError,match='科研'):TaskCleaner(store,root).preview([job])
    # Prospective random sampling and actual paired dispatch use the existing
    # lease/queue/call ledger. No real provider is contacted in this regression.
    with monkeypatch.context() as scoped:
        scoped.setenv('DASHSCOPE_MODEL',LocalCalls.model);scoped.setenv('DASHSCOPE_BASE_URL',LocalCalls.base_url)
        initial=p.model_copy(deep=True);initial.task.pop('research_baseline_policy')
        initial.task.pop('admission_policy',None);initial.task.pop('visibility_policy',None)
        initial.task.pop('requires_native_camera_link',None);initial.task.pop('contrastive_appearance',None)
        initial.task.pop('image_first',None)
        catalog={'catalog_id':'a'*24,'sources':{},'targets':[],'summary':{'selected':3}}
        packets={}
        for n in (1,2,3):
            q=initial.model_copy(deep=True);q.context=f'prospective-{n}'
            q.packet_id=ident(991000+n);packets[n]=q
            shutil.copytree(root/'evidence'/p.packet_id,root/'evidence'/q.packet_id)
            catalog['targets'].append({'target_id':ident(992000+n),'ordinal':n,'context':q.context,
                'object_id':q.target_object_id,'timestamp':q.frame_timestamp_micros,'category':'vehicle',
                'official_split':'training','status':'selected','frame_exclusion_counts':{}})
        scoped.setattr('app.datasets.catalog.load_catalog',lambda **_:catalog)
        class BothCalls(RichCalls):
            def close(self):pass
            def call(self,messages,schema,folder,heartbeat=None):
                if schema.__name__=='BaselineSentence':
                    self.count+=1;return schema(expression='The white vehicle ahead.')
                return super().call(messages,schema,folder,heartbeat)
        worker=ConsoleRunner(root/'pr',BothCalls)
        paired_service=BatchService(worker,lambda t,*args:packets[t['ordinal']],lambda _:catalog,lambda _:None)
        prospective=PaperStudyStore(worker.store,root)
        try:
            frozen_spec=StudySpec(request_id='f'*24,title='Isolated prospective regression',sample_size=2,dataset_split='train',reviewers=['R1','R2','R3'])
            created=prospective.create(frozen_spec,paired_service)
            assert prospective.create(frozen_spec,paired_service)['job_id']==created['job_id']
            assert paired_service.store.detail(created['job_id'])['spec']['desired_count'] is None
            assert len(paired_service.store.detail(created['job_id'])['items'])==2
            paired_service.start(created['job_id']);wait(worker)
            with worker.store.connect() as db:
                trials=list(db.execute('SELECT * FROM paper_trials'))
            assert len(trials)==2
            for trial in trials:
                assert trial['baseline_run'] and worker.store.snapshot(trial['baseline_run'])['model_calls_used']==1
                assert worker.store.snapshot(json.loads(trial['agent_runs'])[-1])['cumulative_call_limit']==13
            assert prospective.detail(created['study_id'],paired_service)['batch']['calls']>=2
            with pytest.raises(ValueError,match='模拟'):prospective.freeze(created['study_id'],paired_service)
        finally:
            paired_service.close();worker.close()


def test_adaptive_loop_two_blind_orders_and_idempotent_completion(tmp_path, monkeypatch):
    # A stale unspoken attribute cannot force a longer sentence. Preserve the
    # failure, propose smaller claims, and run BOTH blind checks afresh.
    from app.agent.harness import SENTENCE_CLAIM_REPAIR_POLICY, sentence_claim_repair
    repair_packet=rich_packet(tmp_path/'sentence-claims',monkeypatch)
    repair_packet.task['sentence_claim_repair_policy']=SENTENCE_CLAIM_REPAIR_POLICY
    class StaleClaims(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            answer=super().call(messages,schema,folder,heartbeat)
            if schema in {GeneratedExpression,VisualObservation}:
                attrs=answer.visual_attributes if schema is GeneratedExpression else answer.attributes
                attrs.append(VisualAttribute(kind='feature',value='roof rack'))
            return answer
    repaired=run_packet(repair_packet,StaleClaims(),output_root=tmp_path/'sentence-runs',max_model_calls=6)
    assert repaired['diagnostic_pass'] and repaired['model_calls_used']==5
    assert len(repaired['attempts'])==2 and len(repaired['attempts'][1]['ground_checks'])==2
    first,last=repaired['attempts']
    assert 'visual_claims_disagree_with_text' in first['verification']['errors']
    assert first['generated']['expression']==last['generated']['expression']
    assert {'kind':'feature','value':'roof rack'} not in last['generated']['visual_attributes']
    parsed=GroundResult.model_validate(last['claim_repair']['source_parse'])
    spoken=GeneratedExpression.model_validate(first['generated'])
    spoken.expression+=' With a roof rack.'
    assert sentence_claim_repair(repair_packet,spoken,parsed,first['verification'],{}) is None
    conflict=parsed.model_copy(update={'parsed_visual_attributes':[VisualAttribute(kind='color',value='red')]})
    assert sentence_claim_repair(repair_packet,GeneratedExpression.model_validate(first['generated']),conflict,first['verification'],{}) is None
    # A redundant visual alias costs no extra call, but never removes a spatial
    # claim or replaces either independent blind parse.
    from app.agent.harness import normalize_visual_predicate_aliases, draft_reference_errors
    alias_packet=rich_packet(tmp_path/'alias-evidence',monkeypatch)
    alias_packet.task['natural_description']=True
    class AliasCalls(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            result=super().call(messages,schema,folder,heartbeat)
            if schema is GeneratedExpression:
                result.predicate_ids.extend(['subtype:sedan','color:white'])
            return result
    alias_calls=AliasCalls()
    alias_result=run_packet(alias_packet,alias_calls,output_root=tmp_path/'alias-runs',max_model_calls=4)
    assert alias_result['diagnostic_pass'] and alias_calls.count==4
    attempt=alias_result['attempts'][0]
    assert len(attempt['ground_checks'])==2
    assert attempt['local_visual_declaration_repair']['removed_duplicate_visual_ids']==['subtype:sedan','color:white']
    invalid=GeneratedExpression.model_validate(attempt['generated'])
    invalid.predicate_ids.extend(['relation:near_red_car','color:blue'])
    preserved,receipt=normalize_visual_predicate_aliases(alias_packet,invalid)
    assert receipt is None and preserved==invalid
    assert 'invalid_predicate_reference' in draft_reference_errors(alias_packet,preserved)
    from app.schemas.contracts import Predicate
    alias_packet.task['annotation_search']='evidence-guided-v1'
    alias_packet.predicates.append(Predicate(predicate_id='image_order:1:vehicle~subtype=sedan:leftmost_visible',
        kind='image_relation',value='1:leftmost_visible',phrase='the leftmost visible sedan in the front view',
        source='test',matches={o.object_id:o==alias_packet.target for o in alias_packet.candidates}))
    stale=GeneratedExpression(expression='The sedan near the red car in the front view.',
        predicate_ids=['category:vehicle',alias_packet.predicates[-1].predicate_id],
        visual_attributes=[VisualAttribute(kind='subtype',value='sedan')])
    errors=draft_reference_errors(alias_packet,stale)
    assert 'declared_image_order_not_in_text' in errors and 'image_visibility_scope_not_explicit' in errors
    lookup = RunStore(tmp_path/'lookup'/'runs.sqlite3')
    lookup.create({'run_id': 'wanted', 'packet_id': 'packet', 'status': 'paused'})
    lookup.create({'run_id': 'unrelated', 'packet_id': 'other'})
    decoded = []
    original_loads = json.loads
    def tracked_loads(value, *args, **kwargs):
        decoded.append(value)
        return original_loads(value, *args, **kwargs)
    # Point lookup never decodes unrelated history; missing runs remain explicit.
    with monkeypatch.context() as reading:
        reading.setattr('app.infrastructure.storage.json.loads', tracked_loads)
        assert lookup.snapshot('wanted', missing_ok=True)['packet_id'] == 'packet'
    assert len(decoded) == 1 and original_loads(decoded[0])['run_id'] == 'wanted'
    assert lookup.snapshot('missing', missing_ok=True) is None
    assert lookup.preparation_attempted_keys()==set()  # No batch schema yet.
    with pytest.raises(ValueError, match='Run not found'):
        lookup.snapshot('missing')
    from app.agent.selection import unavailable_target_keys
    for number, fields in enumerate((
        {'current_packet': {'context':'ctx', 'target_object_id':'track'}, 'model_calls_used':9},
        {'current_packet': {'context':'ctx', 'target_object_id':'generated'},
            'attempts':[{'generated':{'expression':'A vehicle.'},'tool_results':{'unused':'SELECTION_MUST_NOT_DECODE_GEOMETRY'}}]},
        {'context':'legacy', 'current_packet': {'target_object_id':'fallback'}},
        {'context':'legacy', 'current_packet': {'context':None, 'target_object_id':'null'}},
    )):
        lookup.create({'run_id':f'selection-{number}', 'provider_scope':'real-model', **fields})
    for include_prior in (False, True):
        assert unavailable_target_keys(lookup.selection_runs(), 'test', include_prior_attempts=include_prior) == (
            unavailable_target_keys(lookup.all_runs(), 'test', include_prior_attempts=include_prior))
    assert 'SELECTION_MUST_NOT_DECODE_GEOMETRY' not in json.dumps(lookup.selection_runs())
    # Persistent summaries preserve optional/null fields and booleans, track
    # external same-revision edits, and roll back atomically with the source.
    with lookup.connect() as db:
        for row in db.execute("""SELECT r.run_id,s.snapshot AS compact,json_remove(r.snapshot,
            '$.current_packet.candidates','$.current_packet.predicates','$.current_packet.provenance',
            '$.current_packet.contrast','$.current_packet.track','$.current_packet.views') AS expected
            FROM runs r JOIN run_summaries s USING(run_id)"""):
            assert json.loads(row['compact']) == json.loads(row['expected'])
        saved=db.execute('SELECT snapshot FROM runs WHERE run_id=?',('selection-0',)).fetchone()[0]
        changed=json.loads(saved);changed['model_calls_used']=10;changed['diagnostic_pass']=True
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(changed),'selection-0'))
        projected=json.loads(db.execute('SELECT snapshot FROM run_summaries WHERE run_id=?',('selection-0',)).fetchone()[0])
        assert projected['diagnostic_pass'] is True and projected['model_calls_used']==10
        live=json.loads(db.execute('SELECT snapshot FROM run_live_summaries WHERE run_id=?',('selection-0',)).fetchone()[0])
        assert live['diagnostic_pass'] is True and live['model_calls_used']==10
        db.rollback()
        assert db.execute('SELECT snapshot FROM runs WHERE run_id=?',('selection-0',)).fetchone()[0]==saved
        assert json.loads(db.execute('SELECT snapshot FROM run_summaries WHERE run_id=?',('selection-0',)).fetchone()[0])['model_calls_used']==9
        assert json.loads(db.execute('SELECT snapshot FROM run_live_summaries WHERE run_id=?',('selection-0',)).fetchone()[0])['model_calls_used']==9
        db.execute('DELETE FROM runs WHERE run_id=?',('unrelated',))
        assert db.execute('SELECT 1 FROM run_summaries WHERE run_id=?',('unrelated',)).fetchone() is None
        assert db.execute('SELECT 1 FROM run_live_summaries WHERE run_id=?',('unrelated',)).fetchone() is None
    calls = LocalCalls()
    result = run_packet(packet(), calls, output_root=tmp_path, max_model_calls=4)
    assert calls.count == result["model_calls_used"] == 4
    assert result["diagnostic_pass"] and result["status"] == "calibration_hold"
    assert result['sample_identity_validation']['stage'] == 'before_success_commit'
    assert result['quality_gate']['sample_identity_passed'] is True
    assert not result["auto_published"]
    assert len(result["attempts"][0]["ground_checks"]) == 2
    assert result["attempts"][0]["ground_checks"][0]["permutation_seed"] != result["attempts"][0]["ground_checks"][1]["permutation_seed"]
    assert run_packet(packet(), calls, output_root=tmp_path, max_model_calls=4) == result
    assert calls.count == 4
    events = RunStore(tmp_path / "runs.sqlite3").events(result["run_id"])
    assert any(e["kind"] == "action_chosen" for e in events)
    assert any(e["kind"] == "validation_completed" for e in events)

    # Performance telemetry is diagnostic only: completion replay creates no
    # extra session/calls, recovery appends, and broken writes cannot block work.
    from app.infrastructure.diagnostics import PerformanceLog
    from scripts.profile_annotation import profile_performance, profile_runs
    performance_path = tmp_path / result['run_id'] / 'performance.jsonl'
    recorded = profile_performance([performance_path])
    assert recorded['status'] == 'complete' and recorded['sessions'] == 1
    assert recorded['measurements']['sample_identity_check']['count'] == 2
    assert sum(v['count'] for k,v in recorded['measurements'].items() if k.startswith('model_call:')) == 4
    assert recorded['phases']['ground']['count'] >= 1
    assert profile_runs(tmp_path,[result['run_id']])['runs'][0]['performance'] == recorded
    reporting_root=tmp_path/'recorded-wire-timing'
    reporting=RunStore(reporting_root/'runs.sqlite3')
    older,newer=ident(99001),ident(99002)
    for rid in (older,newer):reporting.create({**result,'run_id':rid,'provider_scope':'local-test'})
    with reporting.connect() as db:
        db.execute("UPDATE runs SET snapshot=json_set(snapshot,'$.phase','edited') WHERE run_id=?",(older,))
    atomic_json(reporting_root/newer/'model/00-precheck_candidates/result.json',{
        'status':'completed','elapsed_seconds':10.,'first_content_seconds':2.,
        'request_shape':{'image_count':33},'usage':{'prompt_tokens':1000,'completion_tokens':80,
            'prompt_tokens_details':{'cached_tokens':500}}})
    atomic_json(reporting_root/newer/'model/01-precheck_candidates/result.json',{'status':'api_failed'})
    sampled=profile_runs(reporting_root,recent=1)
    assert sampled['recent_limit']==1 and sampled['batches']==[] and sampled['runs'][0]['run_id']==newer
    stats=sampled['runs'][0]['phases']['precheck_candidates']
    assert stats['p50_response_seconds']==10 and stats['mean_first_content_seconds']==2
    assert stats['p50_image_count']==33 and stats['image_reporting_calls']==1
    assert stats['reported_prompt_cache_fraction']==.5 and stats['cache_reporting_calls']==1
    assert stats['observed_stream_tokens_per_second']==10 and stats['missing_elapsed_calls']==1
    with pytest.raises(ValueError,match='recent'):
        profile_runs(reporting_root,[newer],recent=1)
    # Slow successful polling becomes observable without logging routine reads,
    # response bodies, query strings or starting the application lifespan.
    import asyncio
    import app.main as main_module
    from fastapi.responses import JSONResponse
    logged=[];offset=[0.];real_clock=time.monotonic
    with monkeypatch.context() as api_timing:
        api_timing.setattr(main_module.time,'monotonic',lambda:real_clock()+offset[0])
        api_timing.setattr(main_module,'diagnostic_event',lambda root,event,**fields:logged.append((event,fields)))
        request=SimpleNamespace(method='GET',url=SimpleNamespace(path='/api/batches/private-id'),
            scope={'route':SimpleNamespace(path='/api/batches/{job_id}')})
        async def fast_response(_):return JSONResponse({'private':'not logged'})
        async def slow_response(_):
            offset[0]+=1.5
            return JSONResponse({'private':'not logged'})
        asyncio.run(main_module.request_diagnostics(request,fast_response))
        assert not logged
        asyncio.run(main_module.request_diagnostics(request,slow_response))
    assert len(logged)==1 and logged[0][0]=='api_request'
    assert logged[0][1]['slow_read'] and logged[0][1]['elapsed_seconds']>=1.5
    assert logged[0][1]['path']=='/api/batches/{job_id}' and 'private' not in json.dumps(logged)
    telemetry = tmp_path / 'timing-only'
    with monkeypatch.context() as timing_clock:
        clocks = {'wall':0., 'cpu':0.}
        timing_clock.setattr('app.infrastructure.diagnostics.time.perf_counter', lambda: clocks['wall'])
        timing_clock.setattr('app.infrastructure.diagnostics.time.thread_time', lambda: clocks['cpu'])
        timer = PerformanceLog(telemetry, run_id='timing-only', scope='test')
        timer.phase('prepare')
        with timer.measure('wait'):
            clocks.update(wall=3.,cpu=1.)
        timer.phase('verify')
        with pytest.raises(RuntimeError,match='measurement failed'):
            with timer.measure('wait'):
                clocks.update(wall=5.,cpu=2.)
                raise RuntimeError('measurement failed')
        timer.close('paused')
        timer.close('paused')
        recovered_timer = PerformanceLog(telemetry, run_id='timing-only', scope='test')
        recovered_timer.phase('verify')
        clocks.update(wall=9.,cpu=3.)
        recovered_timer.close('finished')
    timed = profile_performance([telemetry/'performance.jsonl'])
    assert timed['status']=='complete' and timed['sessions']==2 and timed['completed_session_seconds']==9
    assert timed['phases']['prepare']['seconds']==3 and timed['phases']['verify']['seconds']==6
    assert timed['measurements']['wait']=={'count':2,'seconds':5.,'thread_cpu_seconds':2.,'max_seconds':3.,'errors':1}
    unfinished = PerformanceLog(telemetry, run_id='timing-only', scope='test')
    unfinished.phase('model_wait')
    with (telemetry/'performance.jsonl').open('a',encoding='utf-8') as log:
        log.write('{"partial":')
    partial = profile_performance([telemetry/'performance.jsonl'])
    assert partial['status']=='incomplete' and partial['incomplete_sessions']==1
    assert partial['open_phases']==['model_wait'] and partial['malformed_lines']==1
    assert profile_performance([tmp_path/'absent.jsonl'])['completed_session_seconds'] is None
    blocked_log = tmp_path / 'blocked-log'
    blocked_log.write_text('a file prevents a directory here',encoding='utf-8')
    broken = PerformanceLog(blocked_log, run_id='unavailable')
    with broken.measure('work'):
        pass
    broken.close('finished')
    assert broken.unavailable

    # Broken persisted identity is rejected before spending another model call.
    from app.agent.agent_runtime import initial_run, run_agent
    early_calls = LocalCalls()
    initial = initial_run(packet(),early_calls,0,4)
    initial['binding_target']['object_id'] = 'unrelated-track'
    RunStore(tmp_path/'bad-binding/runs.sqlite3').create(initial)
    early = run_agent(packet(),early_calls,0,tmp_path/'bad-binding',4,resume_run_id=initial['run_id'])
    assert not early['diagnostic_pass'] and early['stop_reason'] == 'sample_identity_binding_failed'
    assert early['model_calls_used'] == early_calls.count == 0
    from app.quality.joint_reference import ACCEPTANCE_POLICY
    motion_packet=packet()
    motion_packet.task.update(feedback_acceptance_policy=ACCEPTANCE_POLICY,
        manual_review_feedback={'reason':'目标在自车前方，向右行驶，补充运动状态'})
    motion_packet.target.speed_label_mps=(0.6,-7.3,0.04)
    motion_calls=LocalCalls()
    motion_root=tmp_path/'unsupported-feedback'
    motion_result=run_agent(motion_packet,motion_calls,2,motion_root,12)
    assert motion_result['stop_reason']=='manual_feedback_motion_unsupported'
    assert not motion_result['diagnostic_pass'] and not motion_result['attempts']
    assert motion_result['model_calls_used']==motion_calls.count==0
    assert run_agent(motion_packet,motion_calls,2,motion_root,12)==motion_result
    distance_packet = packet()
    distance_packet.task.pop('ego_proximity_policy',None)
    distance_packet.task.update(feedback_acceptance_policy=ACCEPTANCE_POLICY,
        feedback_detail_policy='explicit-feedback-details-v1',
        manual_review_feedback={'reason':'说明与自车距离很近'})
    distance_calls = LocalCalls()
    distance_result = run_agent(distance_packet,distance_calls,2,tmp_path/'distance-feedback',12)
    assert distance_result['stop_reason'] == 'manual_feedback_distance_unsupported'
    assert distance_result['model_calls_used'] == distance_calls.count == 0
    assert not distance_result['attempts'] and not distance_result['diagnostic_pass']
    # A resumed candidate can pass both localization checks yet still fail the
    # distinct feedback gate at commit. Neither blind result is rewritten.
    incomplete=packet()
    incomplete.task.update(feedback_acceptance_policy=ACCEPTANCE_POLICY,
        manual_review_feedback={'reason':'需要补充自车前方'})
    final_feedback_calls=LocalCalls()
    feedback_root=tmp_path/'final-feedback'
    initial=initial_run(incomplete,final_feedback_calls,0,4)
    initial.update(phase='ground',relations_acquired=True,attempts=[{'attempt':0,'ground_checks':[],
        'generated':{'expression':'The vehicle to the left of the ego vehicle.',
            'predicate_ids':['category:vehicle','lateral:left'],'visual_attributes':[]}}])
    RunStore(feedback_root/'runs.sqlite3').create(initial)
    final_feedback=run_agent(incomplete,final_feedback_calls,0,feedback_root,4,resume_run_id=initial['run_id'])
    assert final_feedback_calls.count==4  # Two blind parses plus legacy counterfactual checks.
    assert all(c['diagnostic_pass'] for c in final_feedback['attempts'][0]['ground_checks'])
    assert not final_feedback['diagnostic_pass'] and final_feedback['stop_reason']=='manual_feedback_spatial_missing'
    assert not final_feedback['feedback_check']['passed']
    # Even two successful blind parses cannot bypass the final identity gate.
    from app.quality import sample_review
    original_binding = sample_review.validate_sample_identity
    with monkeypatch.context() as drift:
        def changed_before_commit(target,state,events=(),**kwargs):
            if state.get('attempts') and len(state['attempts'][-1].get('ground_checks',[])) == 2:
                state = {**state,'packet_id':'f'*24}
            return original_binding(target,state,events,**kwargs)
        drift.setattr(sample_review,'validate_sample_identity',changed_before_commit)
        final_calls = LocalCalls()
        stopped = run_agent(packet(),final_calls,0,tmp_path/'final-binding',4)
    assert len(stopped['attempts'][-1]['ground_checks']) == 2
    assert not stopped['diagnostic_pass'] and not stopped['quality_gate']['sample_identity_passed']
    assert stopped['stop_reason'] == 'sample_identity_binding_failed' and final_calls.count == 4

    # A failed appearance claim followed by a wording repair reuses the raw
    # observation, including after pause/resume, rather than observing it twice.
    p = rich_packet(tmp_path/'repair-evidence', monkeypatch)
    class RepairCalls(RichCalls):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden

        def call(self, messages, schema, folder, heartbeat=None):
            if schema is GeneratedExpression:
                self.count += 1
                self.generated += 1
                color = 'red' if self.generated == 1 else 'white'
                if self.generated == 2 and not self.hidden:
                    payload = json.loads(messages[-1]['content'][0]['text'])
                    assert payload['observed_target_appearance']['attributes'][0]['value'] == 'white'
                return GeneratedExpression(expression=f'The {color} sedan ahead and to the left of the ego vehicle.',
                    predicate_ids=['category:vehicle','longitudinal:front','lateral:left'],
                    visual_attributes=[VisualAttribute(kind='color',value=color), VisualAttribute(kind='subtype',value='sedan')])
            result = super().call(messages, schema, folder, heartbeat)
            if schema is VisualObservation and self.kinds.count(VisualObservation) == 1:
                RunStore(root/'runs.sqlite3').request_control(folder.parents[1].name, 'pause')
                if self.hidden:
                    return result.model_copy(update={'target_visible':False, 'attributes':[]})
            return result

    from app.agent.agent_runtime import run_agent
    for hidden in (False, True):
        root = tmp_path/f'repair-runs-{hidden}'
        repair_calls = RepairCalls(hidden)
        paused = run_agent(p, repair_calls, 1, root, 6)
        assert paused['status'] == 'paused' and paused['model_calls_used'] == 2
        assert paused['attempts'][0]['appearance_check']['errors'] == ['visual_attributes_unconfirmed']
        assert ('appearance_observation' in paused) is not hidden
        RunStore(root/'runs.sqlite3').request_control(paused['run_id'], 'resume')
        repaired = run_agent(p, repair_calls, 1, root, 6, resume_run_id=paused['run_id'])
        assert repaired['diagnostic_pass'] and repaired['model_calls_used'] == repair_calls.count == 5 + int(hidden)
        assert repair_calls.kinds.count(VisualObservation) == 1 + int(hidden)
        assert repaired['attempts'][-1]['appearance_check']['observation_source'] == ('model_call' if hidden else 'run_cache')
        assert len(repaired['attempts'][-1]['ground_checks']) == 2

    p.task['manual_review_feedback'] = {'previous_expression':'The white sedan ahead and to the left of the ego vehicle.'}
    duplicate_calls = RichCalls()
    duplicate = run_agent(p, duplicate_calls, 0, tmp_path/'duplicate-review-text', 4)
    assert not duplicate['diagnostic_pass'] and duplicate_calls.count == 1
    assert duplicate['attempts'][0]['verification']['errors'] == ['manual_review_rejected_text_repeated']
    assert not duplicate['attempts'][0]['ground_checks']

    # An identical failed sentence AND declarations must not buy another blind
    # check; the next writer receives the failed wording and can repair it.
    p.task.pop('manual_review_feedback')
    class RepeatedCalls(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            result=super().call(messages,schema,folder,heartbeat)
            if schema is GeneratedExpression:
                self.generated += 1
                if self.generated > 1:
                    payload=json.loads(messages[-1]['content'][0]['text'])
                    assert result.expression in payload['failed_expressions_do_not_repeat']
                if self.generated == 3:
                    result.expression='The white sedan to the left of the ego vehicle and ahead of it.'
            if schema is GroundResult and self.generated == 1:
                result.parsed_visual_attributes=[]
            return result
    repeated_calls=RepeatedCalls()
    repeated=run_agent(p,repeated_calls,2,tmp_path/'repeated-failed-draft',8)
    assert repeated['diagnostic_pass'] and repeated_calls.count==7
    assert repeated['attempts'][1]['verification']['errors']==['failed_draft_text_repeated']
    assert not repeated['attempts'][1]['ground_checks']
    assert len(repeated['attempts'][2]['ground_checks'])==2
    # Same sentence with corrected declarations is a real repair, not a repeat.
    class DeclarationRepairCalls(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            result=super().call(messages,schema,folder,heartbeat)
            if schema is GeneratedExpression:
                self.generated += 1
                if self.generated == 1:
                    result.predicate_ids.append('nonexistent:relation')
            return result
    declaration_calls=DeclarationRepairCalls()
    declaration=run_agent(p,declaration_calls,1,tmp_path/'same-text-declaration-repair',7)
    assert declaration['diagnostic_pass'] and declaration_calls.count==6
    assert declaration['attempts'][0]['generated']['expression']==declaration['attempts'][1]['generated']['expression']


def test_format_correction_is_bounded_by_request_and_total_budget(tmp_path, monkeypatch):
    import copy
    from contextlib import nullcontext
    from pydantic import ValidationError
    from app.agent.model_calls import QwenCalls, validated_response, validation_diagnostics, format_retry_messages, image_model_error, release_auth_rejection
    from app.agent.observation_wire import CompactSceneObservation, RGBVisualObservation, VehicleVisualObservation, wire_response_type
    from app.agent.appearance import observation_attributes
    from app.schemas.contracts import SceneVisualObservation

    # Missing local override must agree with the project's flash default;
    # an explicit configured model still takes precedence. No request is sent.
    with monkeypatch.context() as config:
        config.setenv('DASHSCOPE_API_KEY', 'local-test-key')
        config.delenv('DASHSCOPE_MODEL', raising=False)
        config.setattr('app.agent.model_calls.OpenAI', lambda **kwargs: SimpleNamespace(close=lambda: None))
        default_calls = QwenCalls()
        assert default_calls.model == 'qwen3.8-flash'
        default_calls.close()
        config.setenv('DASHSCOPE_MODEL', 'configured-vision-model')
        configured_calls = QwenCalls()
        assert configured_calls.model == 'configured-vision-model'
        configured_calls.close()

    # The provider can return plausible JSON while ignoring unsupported images.
    # Reject documented text-only deployments before any SDK call or trace folder.
    for host in ('dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com', 'dashscope-us.aliyuncs.com',
                 'workspace.cn-beijing.maas.aliyuncs.com'):
        assert image_model_error('deepseek-v4-flash-0731', f'https://{host}/compatible-mode/v1')
    for model, url in (
        ('qwen3.7-flash-2026-07-15', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        ('deepseek-v4.1-flash', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        ('deepseek-flash', 'https://api.deepseek.com'),
        ('local-test', 'local://test'),
    ):
        assert image_model_error(model, url) is None
    with monkeypatch.context() as config:
        config.setenv('DASHSCOPE_API_KEY', 'local-test-key')
        config.setenv('DASHSCOPE_MODEL', 'deepseek-v4-flash-0731')
        config.setenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        with pytest.raises(ValueError, match='仅支持文本'):
            QwenCalls()
    text_only = QwenCalls.__new__(QwenCalls)
    text_only.model, text_only.base_url = 'deepseek-v4-flash-0731', 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    with pytest.raises(ValueError, match='仅支持文本'):
        text_only.call([], VisualObservation, tmp_path/'not-sent')
    assert not (tmp_path/'not-sent').exists()

    # One malformed optional rival attribute must not discard a whole batch.
    raw = {'c': [
        {'id': 'rival', 'seen': True, 'vis': 'clear', 'why': 'Visible body.',
         'a': [{'k': 'color', 'v': 'black'}, {'k': 'subtype', 'v': 'car'}],
         'vehicle_colors': {'body': 'black', 'roof': 'null'}},
        {'id': 'target', 'seen': True, 'vis': 'clear', 'why': 'Visible body.',
         'a': [{'k': 'color', 'v': 'silver'}, {'k': 'subtype', 'v': 'sedan'}],
         'vehicle_colors': {'body': 'silver', 'roof': None}}]}
    expected = ['rival', 'target']
    answer = json.dumps(raw)
    parsed, audit = validated_response(answer, CompactSceneObservation, expected)
    assert [c.candidate_id for c in parsed.candidates] == expected
    assert [a.model_dump() for a in parsed.candidates[0].attributes] == [{'kind': 'color', 'value': 'black'}]
    assert parsed.candidates[0].vehicle_colors.roof is None
    assert parsed.candidates[1].attributes[-1].model_dump() == {'kind': 'subtype', 'value': 'sedan'}
    assert len(audit['discarded_evidence']) == 2
    assert validated_response(json.dumps({'output': raw}), CompactSceneObservation, expected)[0] == parsed
    assert raw['c'][0]['a'][-1]['v'] == 'car'  # The original answer stays intact.
    # A hand-carried bag is supported end-to-end, including anonymous negatives.
    # A missing bag annotation or a worn backpack does not prove hand-bag absence.
    bag = VisualAttribute(kind='feature',value='handheld bag')
    bag_wire = copy.deepcopy(raw)
    bag_wire['c'][0].update(a=[{'k':'feature','v':'handheld bag'}],absent=[])
    bag_wire['c'][1].update(a=[],absent=['handheld bag'])
    checked, _ = validated_response(json.dumps(bag_wire), CompactSceneObservation, expected)
    assert [a.model_dump() for a in checked.candidates[0].attributes] == [bag.model_dump()]
    assert checked.candidates[1].absent_features == ['handheld bag']
    from app.agent.appearance import filter_visual_candidates, resolve_target_observation
    assert filter_visual_candidates(['unknown'],[bag],{})[0] == ['unknown']
    backpack = {**clear().model_dump(),'attributes':[{'kind':'feature','value':'backpack'}]}
    assert filter_visual_candidates(['backpack'],[bag],{'backpack':backpack})[0] == ['backpack']
    assert filter_visual_candidates(['absent'],[bag],{'absent':checked.candidates[1].model_dump()})[0] == []
    carrying = clear().model_copy(update={'attributes':[bag]})
    assert resolve_target_observation(carrying,backpack,allow_partial=True).attributes == [bag]
    from app.quality.description_search import appearance_phrase
    phrase = appearance_phrase('pedestrian',[bag,VisualAttribute(kind='upper_clothing_color',value='black'),
        VisualAttribute(kind='lower_clothing_color',value='black')])
    assert 'wearing a black top and black bottoms and holding a bag' in phrase
    assert 'black bag' not in phrase and 'cross' not in phrase

    conflict = copy.deepcopy(raw)
    conflict['c'][0]['a'] += [{'k': 'color', 'v': 'black and white'}]
    checked, _ = validated_response(json.dumps(conflict), CompactSceneObservation, expected)
    assert observation_attributes(checked.candidates[0].model_dump()) == []
    conflict['c'][0].update(a=[{'k': 'feature', 'v': 'roof rack'}, {'k': 'feature', 'v': 'red'}],
        absent=['roof rack'])
    checked, _ = validated_response(json.dumps(conflict), CompactSceneObservation, expected)
    assert not checked.candidates[0].attributes and not checked.candidates[0].absent_features
    primary = {'target_visible': True, 'visibility': 'clear', 'visibility_reason': 'Visible body.',
        'attributes': [{'kind': 'subtype', 'value': 'vehicle'}],
        'vehicle_colors': {'body': 'unknown', 'roof': 'null'}}
    for schema in (VisualObservation, VehicleVisualObservation):
        checked, _ = validated_response(json.dumps(primary), schema)
        assert not checked.attributes and checked.vehicle_colors.body is None
    # New wire observations require explicit visibility. Legacy snapshots remain readable.
    omitted = {k:v for k,v in checked.model_dump().items() if k != 'visibility'}
    assert VisualObservation.model_validate(omitted).visibility == 'uncertain'
    for schema in (RGBVisualObservation, VehicleVisualObservation):
        with pytest.raises(ValidationError):
            validated_response(json.dumps(omitted), schema)
        required = schema.model_json_schema()['required']
        assert 'visibility' in required and 'visibility_reason' in required
        message = [{'role':'user','content':json.dumps({'output_schema':schema.model_json_schema()})}]
        assert wire_response_type(message, VisualObservation) is schema
    excavator = {'kind':'subtype', 'value':'excavator'}
    assert VisualAttribute.model_validate(excavator).value == 'excavator'
    compact_excavator = copy.deepcopy(raw)
    compact_excavator['c'][0]['a'] = [{'k':'subtype','v':'excavator'}]
    checked, _ = validated_response(json.dumps(compact_excavator), CompactSceneObservation, expected)
    assert checked.candidates[0].attributes[0].value == 'excavator'
    canonical = {'candidates': [{**primary, 'candidate_id': 'rival'}]}
    assert not validated_response(json.dumps(canonical), SceneVisualObservation, ['rival'])[0].candidates[0].attributes

    # Identity, visibility, structure and claimed text attributes remain strict.
    for change in (
        lambda r: r['c'][0].pop('id'),
        lambda r: r['c'][0].update(id='target'),
        lambda r: r['c'][0].update(id='unexpected'),
        lambda r: r['c'][0].update(vis='hidden'),
        lambda r: r['c'][0].pop('seen'),
        lambda r: r['c'][0].pop('vis'),
        lambda r: r['c'][0].update(extra='value'),
        lambda r: r['c'][0]['a'][1].update(extra='value'),
        lambda r: r['c'][0]['a'][1].update(k='unknown_kind'),
        lambda r: r['c'][0].update(absent=['unknown_feature']),
        lambda r: r['c'][0].update(a=[{'k': 'subtype', 'v': 'car'}] * 7),
    ):
        invalid = copy.deepcopy(raw)
        change(invalid)
        with pytest.raises(ValidationError):
            validated_response(json.dumps(invalid), CompactSceneObservation, expected)
    for invalid in ('{', json.dumps({'output': raw, 'extra': True}), json.dumps({'output': {'output': raw}})):
        with pytest.raises(ValidationError):
            validated_response(invalid, CompactSceneObservation, expected)
    for schema, value in (
        (GeneratedExpression, {'expression': 'The car.', 'predicate_ids': ['p'],
            'visual_attributes': [{'kind': 'subtype', 'value': 'car'}]}),
        (GroundResult, {'candidate_ids': ['target'], 'parsed_predicate_ids': ['p'],
            'unsupported_claims': [], 'parsed_visual_attributes': [{'kind': 'subtype', 'value': 'vehicle'}]}),
    ):
        with pytest.raises(ValidationError):
            validated_response(json.dumps(value), schema)

    payload = {'output_schema': CompactSceneObservation.model_json_schema(), 'candidate_ids': expected}
    messages = [{'role': 'user', 'content': json.dumps(payload)}]
    invalid = copy.deepcopy(raw)
    invalid['c'][0]['vis'] = 'hidden'
    try:
        validated_response(json.dumps(invalid), CompactSceneObservation, expected)
    except ValidationError as exc:
        errors = validation_diagnostics(exc, json.dumps(invalid))
    assert errors[0]['candidate_id'] == 'rival' and errors[0]['received_value'] == 'hidden'
    feedback = json.loads(format_retry_messages(messages, SceneVisualObservation, errors)[-1]['content'])
    assert feedback['output_schema'] == payload['output_schema']
    assert feedback['format_feedback']['errors'] == errors

    # Exercise the SDK boundary and its durable audit without network calls.
    chunk = SimpleNamespace(id='local', model='local', usage=None,
        choices=[SimpleNamespace(finish_reason='stop', delta=SimpleNamespace(content=answer))],
        model_dump_json=lambda: '{}')
    sdk = QwenCalls.__new__(QwenCalls)
    sdk.model, sdk.base_url = 'local', 'local://no-network'
    sdk.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **request: nullcontext([chunk]))))
    folder = tmp_path/'sdk'
    result = sdk.call(messages, SceneVisualObservation, folder)
    saved = json.loads((folder/'result.json').read_text())
    assert result.candidates[1].candidate_id == 'target'
    assert saved['status'] == 'completed' and saved['answer'] == answer
    assert saved['response_normalization'] == audit
    assert sdk.call(messages, SceneVisualObservation, folder) == result
    # Many small deltas retain their complete raw trace with bounded flushes.
    original_open = Path.open
    flushed = []
    class TracedLog:
        def __init__(self, log):
            self.log = log
        def __enter__(self):
            self.log.__enter__()
            return self
        def __exit__(self, *args):
            return self.log.__exit__(*args)
        def write(self, line):
            return self.log.write(line)
        def flush(self):
            flushed.append(True)
            return self.log.flush()
    def traced_open(path, *args, **kwargs):
        log = original_open(path, *args, **kwargs)
        return TracedLog(log) if path.name == 'chunks.jsonl' else log
    prefix = SimpleNamespace(id='local', model='local', usage=None,
        choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content=''))],
        model_dump_json=lambda: '{}')
    with monkeypatch.context() as streaming:
        streaming.setattr(Path, 'open', traced_open)
        streaming.setattr(sdk.client.chat.completions, 'create',
            lambda **request: nullcontext([prefix] * 80 + [chunk]))
        assert sdk.call(messages, SceneVisualObservation, tmp_path/'buffered') == result
    assert 1 <= len(flushed) < 81
    assert (tmp_path/'buffered'/'chunks.jsonl').read_text().splitlines() == ['{}'] * 81
    # An interrupted request is preserved, never submitted twice after restart.
    interrupted_trace = {**saved, 'status':'running'}
    atomic_json(folder/'result.json',interrupted_trace)
    with pytest.raises(ModelCallFailure,match='interrupted_response'):
        sdk.call(messages,SceneVisualObservation,folder)
    assert json.loads((folder/'result.json').read_text()) == interrupted_trace
    # A stream that keeps yielding cannot hold a worker forever.
    with monkeypatch.context() as timed:
        timed.setattr('app.agent.model_calls.CALL_WALL_SECONDS',-1)
        with pytest.raises(ModelCallFailure,match='connection_failed'):
            sdk.call(messages,SceneVisualObservation,tmp_path/'deadline')
    deadline = json.loads((tmp_path/'deadline'/'result.json').read_text())
    assert deadline['remote_outcome']=='unknown' and deadline['retryable'] is False
    assert deadline['network_diagnosis']=='response_deadline_exceeded'
    import httpx
    def broken_stream():
        yield prefix
        raise httpx.ReadError('interrupted local stream')
    with monkeypatch.context() as interrupted:
        interrupted.setattr(sdk.client.chat.completions, 'create',
            lambda **request: nullcontext(broken_stream()))
        with pytest.raises(ModelCallFailure, match='connection_failed'):
            sdk.call(messages, SceneVisualObservation, tmp_path/'buffered-interruption')
    assert (tmp_path/'buffered-interruption'/'chunks.jsonl').read_text().splitlines() == ['{}']
    interrupted_record = json.loads((tmp_path/'buffered-interruption'/'result.json').read_text())
    assert interrupted_record['remote_outcome'] == 'unknown' and not interrupted_record['retryable']
    from openai import APIStatusError
    def refused(**request):
        raise APIStatusError('temporary rejection',response=httpx.Response(503,
            request=httpx.Request('POST','https://example.invalid'),headers={'Retry-After':'17'}),body={})
    with monkeypatch.context() as headers:
        headers.setattr(sdk.client.chat.completions,'create',refused)
        with pytest.raises(ModelCallFailure,match='api_failed'):
            sdk.call(messages,SceneVisualObservation,tmp_path/'retry-after')
    assert json.loads((tmp_path/'retry-after'/'result.json').read_text())['retry_after_seconds']==17

    from app.agent.model_calls import provider_diagnostic, request_repair_option, isolated_request_failure, saved_call_diagnostic
    def bad_request(**request):
        raise APIStatusError('request refused', response=httpx.Response(400,
            request=httpx.Request('POST','https://example.invalid')),
            body={'error':'image size invalid; private-input-must-not-be-logged'})
    with monkeypatch.context() as request_rejected:
        request_rejected.setattr(sdk.client.chat.completions,'create',bad_request)
        with pytest.raises(ModelCallFailure,match='api_failed'):
            sdk.call(messages,SceneVisualObservation,tmp_path/'request-rejected')
    failure=json.loads((tmp_path/'request-rejected/result.json').read_text())
    assert failure['provider_diagnosis']=='image_input_invalid'
    assert failure['provider_message_chars']>0 and len(failure['provider_message_sha256'])==64
    assert 'private-input-must-not-be-logged' not in json.dumps(failure)
    assert isolated_request_failure(saved_call_diagnostic(tmp_path/'request-rejected'))
    assert provider_diagnostic({'code':'Arrearage','message':'rejected'})['provider_diagnosis']=='provider_account_rejected'
    assert not isolated_request_failure({'http_status':400,'provider_diagnosis':'provider_account_rejected'})
    assert not isolated_request_failure({'http_status':400,'provider_response_received':True})
    assert not isolated_request_failure({'http_status':400,'remote_outcome':'unknown'})
    with monkeypatch.context() as repair_wire:
        repair_wire.setattr(sdk.client.chat.completions,'create',lambda **request:nullcontext([chunk]))
        sdk.call(messages,SceneVisualObservation,tmp_path/'repair-wire',
            request_options={'enable_thinking':False,'response_format':False})
    repaired_request=json.loads((tmp_path/'repair-wire/request.json').read_text())
    assert 'extra_body' not in repaired_request and 'response_format' not in repaired_request
    assert repaired_request['messages'] and repaired_request['model']==sdk.model
    repair_failure={'status':'api_failed','http_status':400,'stream_chunk_count':0,
        'provider_diagnosis':'thinking_mode_unsupported'}
    assert request_repair_option(repair_failure,{})=='enable_thinking'
    for change in ({'stream_chunk_count':1},{'response_id':'received'},{'answer':'partial'},
            {'usage':{'total_tokens':1}},{'http_status':401}):
        assert request_repair_option({**repair_failure,**change},{}) is None
    assert request_repair_option(repair_failure,{'enable_thinking':False}) is None

    class RepairCalls(LocalCalls):
        def __init__(self,always=False,pause=False):
            super().__init__();self.always,self.pause=always,pause
        def call(self,messages,schema,folder,heartbeat=None,request_options=None):
            if self.count==0 or self.always:
                self.count+=1
                atomic_json(folder/'result.json',repair_failure)
                if self.pause:
                    self.pause=False
                    RunStore(folder.parents[2]/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
                raise ModelCallFailure('api_failed')
            assert request_options=={'enable_thinking':False}
            return super().call(messages,schema,folder,heartbeat)
    repaired_calls=RepairCalls()
    repaired=run_packet(packet(),repaired_calls,output_root=tmp_path/'parameter-repair',max_model_calls=5)
    assert repaired['diagnostic_pass'] and repaired['request_repairs_used']==1
    assert repaired['model_calls_used']==repaired_calls.count==5
    assert (tmp_path/'parameter-repair'/repaired['run_id']/'model/00-decide/result.json').is_file()
    bounded_calls=RepairCalls(always=True)
    bounded=run_packet(packet(),bounded_calls,output_root=tmp_path/'parameter-repair-bounded',max_model_calls=6)
    assert bounded['status']=='blocked' and bounded_calls.count==bounded['model_calls_used']==2
    assert bounded['request_repairs_used']==1
    from app.agent.agent_runtime import run_agent
    paused_calls=RepairCalls(pause=True)
    paused=run_agent(packet(),paused_calls,0,tmp_path/'parameter-repair-paused',5)
    assert paused['status']=='paused' and paused['provider_request_options']=={'enable_thinking':False}
    RunStore(tmp_path/'parameter-repair-paused/runs.sqlite3').request_control(paused['run_id'],'resume')
    continued=run_agent(packet(),paused_calls,0,tmp_path/'parameter-repair-paused',5,resume_run_id=paused['run_id'])
    assert continued['diagnostic_pass'] and continued['model_calls_used']==paused_calls.count==5

    from app.agent.model_calls import transient_retry_delay
    temporary = {'status':'api_failed','http_status':429,'retry_after_seconds':21}
    assert transient_retry_delay(temporary,0)==21
    assert transient_retry_delay({**temporary,'retry_after_seconds':1000},0)==300
    for change in ({'http_status':401},{'http_status':400},{'response_id':'partly-received'},
            {'stream_chunk_count':1},{'answer':'partial'}):
        assert transient_retry_delay({**temporary,**change},0) is None
    assert transient_retry_delay(temporary,3) is None
    class TemporaryCalls(LocalCalls):
        def __init__(self,failures,pause=False):
            super().__init__()
            self.failures,self.pause=failures,pause
        def call(self,messages,schema,folder,heartbeat=None):
            if self.count < self.failures:
                self.count += 1
                atomic_json(folder/'result.json',{'status':'api_failed','http_status':429 if self.count==1 else 503})
                if self.pause:
                    self.pause=False
                    RunStore(folder.parents[2]/'runs.sqlite3').request_control(folder.parents[1].name,'pause')
                raise ModelCallFailure('api_failed')
            return super().call(messages,schema,folder,heartbeat)
    with monkeypatch.context() as retries:
        retries.setattr('app.agent.model_calls.TRANSIENT_RETRY_DELAYS',(0,0,0))
        calls=TemporaryCalls(2)
        recovered=run_packet(packet(),calls,output_root=tmp_path/'provider-retry',max_model_calls=6)
        assert recovered['diagnostic_pass'] and recovered['provider_retries_used']==2
        assert recovered['model_calls_used']==calls.count==6
        calls=TemporaryCalls(5)
        exhausted=run_packet(packet(),calls,output_root=tmp_path/'provider-exhausted',max_model_calls=8)
        assert exhausted['status']=='blocked' and exhausted['provider_retries_used']==3
        assert exhausted['model_calls_used']==calls.count==4
        from app.agent.agent_runtime import run_agent
        calls=TemporaryCalls(1,pause=True)
        paused=run_agent(packet(),calls,0,tmp_path/'provider-pause',6)
        assert paused['status']=='paused' and paused['pending_connection_retry']['provider_retry_attempt']==1
        RunStore(tmp_path/'provider-pause'/'runs.sqlite3').request_control(paused['run_id'],'resume')
        resumed=run_agent(packet(),calls,0,tmp_path/'provider-pause',6,resume_run_id=paused['run_id'])
        assert resumed['diagnostic_pass'] and resumed['model_calls_used']==calls.count==5

    calls = InvalidFormatCalls(failures=2)
    recovered = run_packet(packet(), calls, output_root=tmp_path/'recovered', max_model_calls=6)
    assert recovered['diagnostic_pass'] and recovered['format_retries_used'] == 2
    assert recovered['model_calls_used'] == calls.count == 6
    calls = InvalidFormatCalls(failures=3)
    stopped = run_packet(packet(), calls, output_root=tmp_path/'stopped', max_model_calls=8)
    assert stopped['status'] == 'blocked' and stopped['format_retries_used'] == 2
    assert stopped['model_calls_used'] == calls.count == 3
    assert not stopped['auto_published']

    rejected = {'status': 'api_failed', 'http_status': 401,
        'provider_error_code': 'invalid_api_key', 'stream_chunk_count': 0, 'answer': ''}
    for changed in ({'http_status': 403}, {'status': 'connection_failed'},
            {'provider_error_code': 'unknown'}, {'stream_chunk_count': 1},
            {'response_id': 'received'}, {'answer': '{}'}, {'usage': {'total_tokens': 1}}):
        state = {'model_calls_used': 1, 'inflight': {'folder': 'model/00-ground'}}
        assert not release_auth_rejection(state, {**rejected, **changed})
        assert state['model_calls_used'] == 1
    class AuthRejectedCalls(LocalCalls):
        def call(self, messages, schema, folder, heartbeat=None):
            self.count += 1
            atomic_json(folder/'result.json', rejected)
            from app.agent.model_calls import ModelCallFailure
            raise ModelCallFailure('api_failed')
    calls = AuthRejectedCalls()
    state = run_packet(packet(), calls, output_root=tmp_path/'auth', max_model_calls=4)
    assert state['status'] == 'blocked' and calls.count == 1  # Stop; no auth retry loop.
    assert state['model_calls_used'] == 0 and state['auth_rejections'] == 1
    assert not release_auth_rejection(state, rejected)  # Replays cannot release twice.
    assert any(e['kind'] == 'auth_rejection_reservation_released'
        for e in RunStore(tmp_path/'auth'/'runs.sqlite3').events(state['run_id']))



def test_ambiguous_source_tries_same_track_frames_within_budget(tmp_path, monkeypatch):
    recover, pause, budget = True, True, 8
    import shutil
    import app.agent.agent_runtime as runtime
    from app.infrastructure.storage import RunStore
    from app.quality.contrast import predicate_catalog
    from app.schemas.contracts import VisualObservation, VisualAttribute
    from app.datasets.preparation import TargetPreparer, EvidenceRejected, PreparationCancelled
    from app.agent.batch_execution import BatchExecution
    from app.datasets.correspondence import has_multimodal_evidence
    baseline = prepared(tmp_path/'preparation-fallback', monkeypatch)
    target = {'context':baseline.context,'object_id':baseline.target_object_id,'category':baseline.target.category,
        'timestamp':baseline.frame_timestamp_micros,'source_camera_id':1,
        'alternative_selections':[{'timestamp':baseline.frame_timestamp_micros+n,'source_camera_id':1}
            for n in (1,1,2,3)]}
    class FramePreparer(TargetPreparer):
        def __init__(self, good=True):
            self.visited, self.good = [], good
        def __call__(self, selected, *args):
            timestamp = selected['timestamp']; self.visited.append(timestamp)
            candidate = baseline.model_copy(deep=True)
            candidate.packet_id = ident(8000 + timestamp)
            candidate.frame_timestamp_micros = timestamp
            if not self.good or timestamp == target['timestamp']:
                candidate.scene = candidate.scene.model_copy(update={'target_points':0})
                raise EvidenceRejected(candidate)
            return candidate
    preparer = FramePreparer()
    recovered_packet, failures = preparer.prepare_with_fallback(target,{},lambda stage:None,lambda:False,tmp_path/'prepare.log')
    assert len(preparer.visited) == 2 and has_multimodal_evidence(recovered_packet)
    assert failures[0]['reason'] == 'converted_target_points_missing'
    assert not has_multimodal_evidence(failures[0]['packet'])
    BatchExecution.assert_target(recovered_packet,target,alternate=True)
    from app.routers.samples import bound_sample
    accepted = recovered_packet.model_dump(mode='json')
    binding_state = {'packet_id':recovered_packet.packet_id,'input_packet_id':recovered_packet.packet_id,
        'current_packet':accepted,'final_expression':'The vehicle ahead.'}
    binding_item = {'ordinal':1,'run_id':ident(8030),'target':target,'packet_id':recovered_packet.packet_id,
        'expression':binding_state['final_expression']}
    receipt = {'ordinal':1,'packet_id':recovered_packet.packet_id,'timestamp':recovered_packet.frame_timestamp_micros,
        'rejected_frames':[{k:v for k,v in attempt.items() if k != 'packet'} for attempt in failures]}
    preparation_event = {'kind':'preparation_source_reselected','payload':receipt}
    with pytest.raises(ValueError,match='身份不一致'):
        bound_sample(ident(8000),binding_item,binding_state,[])
    assert bound_sample(ident(8000),binding_item,binding_state,[preparation_event])[0].model_dump(mode='json') == recovered_packet.model_dump(mode='json')
    with pytest.raises(ValueError,match='身份不一致'):
        bound_sample(ident(8000),{**binding_item,'packet_id':None},binding_state,[preparation_event])
    # The run retains this binding if its original task is deleted after reuse.
    retained = {**binding_state,'preparation_source_reselection':receipt}
    assert bound_sample(ident(8001),binding_item,retained,[])[0].model_dump(mode='json') == recovered_packet.model_dump(mode='json')
    for invalid in ({**receipt,'timestamp':999}, {**receipt,'packet_id':'f'*24},
            {**receipt,'rejected_frames':[]}, {**receipt,'rejected_frames':[{'timestamp':999,'reason':'missing'}]}):
        with pytest.raises(ValueError):
            bound_sample(ident(8000),binding_item,{**binding_state,'preparation_source_reselection':invalid},[])
    # Runtime frame changes must continue from the actual prepared input,
    # rather than from the catalog's rejected representative frame.
    moved = recovered_packet.model_copy(update={'packet_id':ident(8040),'frame_timestamp_micros':target['timestamp']+2})
    runtime_change = {'kind':'source_view_reselected','payload':{'from_packet_id':recovered_packet.packet_id,
        'from_timestamp':recovered_packet.frame_timestamp_micros,'packet_id':moved.packet_id,'timestamp':moved.frame_timestamp_micros}}
    moved_state = {**binding_state,'packet_id':moved.packet_id,'current_packet':moved.model_dump(mode='json')}
    moved_item = {**binding_item,'packet_id':moved.packet_id}
    assert bound_sample(ident(8000),moved_item,moved_state,[preparation_event,runtime_change])[0].model_dump(mode='json') == moved.model_dump(mode='json')
    with pytest.raises(ValueError,match='frozen target'):
        BatchExecution.assert_target(recovered_packet.model_copy(update={'target_object_id':'other'}),target,alternate=True)
    with pytest.raises(ValueError,match='frozen target'):
        BatchExecution.assert_target(recovered_packet.model_copy(update={'frame_timestamp_micros':999}),target,alternate=True)
    preparer = FramePreparer(good=False)
    with pytest.raises(EvidenceRejected) as rejected:
        preparer.prepare_with_fallback(target,{},lambda stage:None,lambda:False,tmp_path/'all-failed.log')
    assert len(set(preparer.visited)) == len(preparer.visited) == len(rejected.value.preparation_attempts) == 3
    preparer = FramePreparer()
    with pytest.raises(PreparationCancelled) as interrupted:
        preparer.prepare_with_fallback(target,{},lambda stage:None,lambda:bool(preparer.visited),tmp_path/'paused.log')
    assert len(preparer.visited) == len(interrupted.value.preparation_attempts) == 1

    p = strict_packet(tmp_path / 'evidence', monkeypatch, ambiguous=True)
    p.task.update(annotation_search='evidence-guided-v1', view_selection_policy='unobstructed-multiview-v1')
    p.provenance['unobstructed_source_cameras'] = {'1': {'image_selection_score':100}}
    monkeypatch.setattr(runtime, 'EVIDENCE_ROOT', tmp_path / 'evidence')
    seen = []

    def alternate(current, rejected, heartbeat):
        # Preparing another frame can exceed the five-minute target lease.
        # Advance both clocks without sleeping and verify periodic renewals.
        wall, perf = time.time(), time.perf_counter()
        elapsed = [0]
        with monkeypatch.context() as slow:
            slow.setattr(time,'time',lambda:wall+elapsed[0])
            slow.setattr(time,'perf_counter',lambda:perf+elapsed[0])
            for _ in range(6):
                elapsed[0] += 60
                heartbeat()
            with RunStore(tmp_path/'runs/runs.sqlite3').connect() as db:
                assert db.execute('SELECT MIN(lease_until) FROM runs WHERE owner IS NOT NULL').fetchone()[0] > time.time()
        seen.append(list(rejected))
        new = p.model_copy(deep=True, update={'packet_id':str(len(seen))*24,
            'frame_timestamp_micros':current.frame_timestamp_micros+1000000})
        shutil.copytree(tmp_path/'evidence'/p.packet_id, tmp_path/'evidence'/new.packet_id)
        if recover:
            new.candidates[1] = new.candidates[1].model_copy(update={'center_m':(30, -10, 1)})
            new.predicates = predicate_catalog(new.candidates)
        return new

    class Calls(RichCalls):
        def call(self, messages, schema, folder, heartbeat=None):
            if schema is VisualObservation:
                self.count += 1
                self.kinds.append(schema)
                if pause and self.count == 2:
                    RunStore(tmp_path/'runs/runs.sqlite3').request_control(folder.parents[1].name, 'pause')
                return clear().model_copy(update={'attributes':[*clear().attributes,
                    VisualAttribute(kind='subtype', value='sedan')]})
            assert recover, 'Ambiguous frames must never generate text.'
            return super().call(messages, schema, folder, heartbeat)

    calls = Calls()
    result = runtime.run_agent(p, calls, 0, tmp_path/'runs', budget, alternate_frame=alternate)
    if pause:
        assert result['status'] == 'paused'
        RunStore(tmp_path/'runs/runs.sqlite3').request_control(result['run_id'], 'resume')
        result = runtime.run_agent(p, calls, 0, tmp_path/'runs', budget,
            resume_run_id=result['run_id'], alternate_frame=alternate)
    assert result['model_calls_used'] <= budget
    rejection = result['rejected_source_views'][0]
    assert rejection['reason'] == 'precheck_no_distinguishing_cues'
    assert rejection['packet_id'] == p.packet_id
    assert p.candidates[1].object_id in rejection['distinguishing_cue_bound']['remaining_object_ids']
    if recover:
        assert len(seen) == 1 and result['diagnostic_pass']
        assert len(result['attempts'][-1]['ground_checks']) == 2
        assert 'distinguishing_cue_bound' not in result  # No stale failure on the recovered frame.
        assert result['current_packet']['frame_timestamp_micros'] != p.frame_timestamp_micros
    else:
        assert len(seen) == (0 if budget == 4 else 2)  # Existing three-frame cap.
        assert not result['attempts'] and not result['diagnostic_pass']
        assert result['stop_reason'] == 'precheck_no_distinguishing_cues'

    # A suggested native relation gets one source recovery BEFORE writing,
    # even when a different appearance description could otherwise be used.
    # This spends the same original budget and never converts the suggestion
    # into observed attributes or geometric truth.
    feedback_case=p.model_copy(deep=True)
    feedback_case.task.update(feedback_acquisition_policy='feedback-landmark-recheck-v2-diagnostics',
        feedback_relation_policy='bilingual-landmark-feedback-v3-spatial-details',
        feedback_acceptance_policy='feedback-acceptance-v5-natural-concision',
        manual_review_feedback={'reason':'可以说明在绿色矩形标志牌的下方'})
    feedback_seen=[]
    def feedback_alternate(current,rejected,heartbeat):
        feedback_seen.append(current.packet_id)
        assert len(feedback_seen)==1
        new=current.model_copy(deep=True,update={'packet_id':ident(8890),
            'frame_timestamp_micros':current.frame_timestamp_micros+1000000})
        shutil.copytree(tmp_path/'evidence'/p.packet_id,tmp_path/'evidence'/new.packet_id)
        new.candidates[1]=new.candidates[1].model_copy(update={'center_m':(30,-10,1)})
        new.predicates=predicate_catalog(new.candidates)
        return new
    pause=False
    feedback_calls=Calls()
    feedback_result=runtime.run_agent(feedback_case,feedback_calls,0,tmp_path/'feedback-runs',8,
        alternate_frame=feedback_alternate)
    assert len(feedback_seen)==1 and feedback_result['feedback_source_retry_used']
    assert feedback_result['model_calls_used']==feedback_calls.count<=8
    assert feedback_result['rejected_source_views'][0]['reason']=='feedback_landmark_relation_unresolved'
    assert feedback_result['feedback_landmark_acquisition']['packet_id']==ident(8890)
    assert not feedback_result['feedback_check']['passed']  # Text acceptance is not feedback compliance.
    assert feedback_result['feedback_check']['requirements'][0]['status']=='missing'

    # A polite correction tied to unique identification is binding in new runs.
    # Preserve the old optional behavior above. After one source recovery fails,
    # retain the diagnostic and budget, with no irrelevant writer/ground calls.
    intended_case=feedback_case.model_copy(deep=True)
    intended_case.task.update(feedback_intent_policy='purpose-bound-feedback-v1',
        manual_review_feedback={'reason':'你可以说明在绿色矩形标志牌的下方，这样才能唯一定位物体'})
    intended_seen=[]
    def intended_alternate(current,rejected,heartbeat):
        intended_seen.append(current.packet_id)
        assert len(intended_seen)==1
        new=current.model_copy(deep=True,update={'packet_id':ident(8891),
            'frame_timestamp_micros':current.frame_timestamp_micros+1000000})
        shutil.copytree(tmp_path/'evidence'/p.packet_id,tmp_path/'evidence'/new.packet_id)
        return new
    intended_calls=Calls()
    intended_result=runtime.run_agent(intended_case,intended_calls,2,tmp_path/'intended-runs',8,
        alternate_frame=intended_alternate)
    assert len(intended_seen)==1 and intended_result['feedback_source_retry_used']
    assert intended_result['stop_reason']=='manual_feedback_landmark_missing'
    assert not intended_result['attempts'] and not intended_result['diagnostic_pass']
    assert intended_result['model_calls_used']==intended_calls.count<8
    assert all(kind.__name__ in {'VisualObservation','SceneVisualObservation'} for kind in intended_calls.kinds)
    assert intended_result['feedback_check']['blocking_errors']==['manual_feedback_landmark_missing']
    requirement=intended_result['feedback_check']['requirements'][0]
    assert requirement['required'] and requirement['status']=='missing'
    assert '要求' in requirement['reason'] and requirement['acquisition_status']=='no_supported_native_relation'

    # At eight spent calls, a generic reread would close the four-call recovery
    # window. Try the same track's next frame before spending that last reserve.
    location=tmp_path/'frame-reserve'
    reserve=strict_packet(location/'evidence',monkeypatch,ambiguous=True)
    reserve.task.update(annotation_search='evidence-guided-v1',view_selection_policy='unobstructed-multiview-v1',
        landmark_description_policy='spatially-qualified-v1')
    reserve.provenance['unobstructed_source_cameras']={'1':{'image_selection_score':100}}
    rival_id=reserve.candidates[1].object_id
    rival_crop={'candidate_id':'v001','object_id':rival_id,'category':'vehicle','camera_name':1,
        'artifact':'appearance.jpg'}
    reserve.provenance['appearance_candidates']=[{**rival_crop,'alternatives':[rival_crop]}]
    monkeypatch.setattr(runtime,'EVIDENCE_ROOT',location/'evidence')
    sedan=clear().model_copy(update={'attributes':[VisualAttribute(kind='color',value='white'),
        VisualAttribute(kind='subtype',value='sedan')]})
    class ReserveCalls(RichCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is VisualObservation:
                self.count+=1;return sedan
            assert schema in {GeneratedExpression,GroundResult},'Do not spend the frame reserve on a generic reread.'
            return super().call(messages,schema,folder,heartbeat)
    calls=ReserveCalls()
    from app.agent.agent_runtime import initial_run
    seed=initial_run(reserve,calls,0,12,12)
    seed.update(phase='precheck_decision',model_calls_used=8,target_precheck_observation=sedan.model_dump(),
        candidate_appearance={reserve.target_object_id:sedan.model_dump(),rival_id:sedan.model_dump()})
    RunStore(location/'runs/runs.sqlite3').create(seed)
    rescued=[]
    def next_reserved_frame(current,rejected,heartbeat):
        rescued.append(rejected)
        new=current.model_copy(deep=True)
        new.packet_id='f'*24;new.frame_timestamp_micros+=1000000
        new.candidates[1]=new.candidates[1].model_copy(update={'center_m':(30,-10,1)})
        new.predicates=predicate_catalog(new.candidates)
        new.provenance['appearance_candidates']=[]
        shutil.copytree(location/'evidence'/reserve.packet_id,location/'evidence'/new.packet_id)
        return new
    result=runtime.run_agent(reserve,calls,0,location/'runs',12,12,resume_run_id=seed['run_id'],
        alternate_frame=next_reserved_frame)
    assert rescued and result['diagnostic_pass'] and result['model_calls_used']==12 and calls.count==4
    assert len(result['attempts'][-1]['ground_checks'])==2
    assert result['rejected_source_views'][0]['reason']=='scene_relations_unresolved_reserve_complete_retry'

    # An image-only identity rejection AFTER drafting must seek new evidence,
    # rather than rewrite the same pixels until the draft limit is exhausted.
    from app.agent import scene_grounding
    recovery=tmp_path/'post-draft-source'
    original=strict_packet(recovery/'evidence',monkeypatch)
    original.task.update(annotation_search='evidence-guided-v1',
        view_selection_policy='unobstructed-multiview-v1',scene_grounding_policy=scene_grounding.POLICY,
        scene_source_retry_policy=scene_grounding.SOURCE_RETRY_POLICY)
    original.provenance['unobstructed_source_cameras']={'1':{'image_selection_score':100}}
    original.provenance['scene_image_candidates']=[
        {'object_id':original.target_object_id,'category':'vehicle','camera_name':1,'xyxy':[0,0,100,100]},
        {'object_id':'image:1:unresolved','category':'vehicle','camera_name':1,'xyxy':[0,0,90,90]}]
    draft=GeneratedExpression(expression='The white sedan ahead and to the left of the ego vehicle.',
        predicate_ids=['category:vehicle','longitudinal:front','lateral:left'],visual_attributes=sedan.attributes)
    class IdentityCalls(ReserveCalls):
        wrong_native=False
        def call(self,messages,schema,folder,heartbeat=None):
            if schema is scene_grounding.SceneGroundResult:
                self.count+=1
                payload=json.loads(messages[-1]['content'][0]['text'])
                cid=next(row['candidate_id'] for row in payload['candidates'] if (
                    row['identity']=='native_3d' and row.get('center_m')!=list(original.target.center_m)
                    if self.wrong_native else row['identity']=='unmatched_image'))
                return scene_grounding.SceneGroundResult(expression_supported=True,candidate_ids=[cid],
                    ambiguous_candidate_ids=[],unsupported_claims=[],reason='Image region recognized; native identity unresolved.')
            return super().call(messages,schema,folder,heartbeat)
    identity_calls=IdentityCalls()
    initial=initial_run(original,identity_calls,2,12,12)
    initial.update(phase='scene_ground',model_calls_used=3,admission={'passed':True},
        candidate_appearance={original.target_object_id:sedan.model_dump()},
        target_precheck_observation=sedan.model_dump(),attempts=[{'attempt':0,'generated':draft.model_dump(),
            'appearance_check':{'passed':True},'ground_checks':[]}])
    RunStore(recovery/'runs/runs.sqlite3').create(initial)
    changed=[]
    def better_frame(current,rejected,heartbeat):
        changed.append(rejected)
        updated=current.model_copy(deep=True)
        updated.packet_id='e'*24;updated.frame_timestamp_micros+=1000000
        updated.provenance['scene_image_candidates']=updated.provenance['scene_image_candidates'][:1]
        if not (recovery/'evidence'/updated.packet_id).exists():
            shutil.copytree(recovery/'evidence'/original.packet_id,recovery/'evidence'/updated.packet_id)
        return updated
    with monkeypatch.context() as context:
        context.setattr(runtime,'EVIDENCE_ROOT',recovery/'evidence')
        context.setattr(scene_grounding,'EVIDENCE_ROOT',recovery/'evidence')
        recovered=runtime.run_agent(original,identity_calls,2,recovery/'runs',12,12,
            resume_run_id=initial['run_id'],alternate_frame=better_frame)
    assert changed and recovered['diagnostic_pass']
    assert len(recovered['attempts'])==2 and len(recovered['attempts'][-1]['ground_checks'])==2
    assert recovered['attempts'][0]['ground_checks'][0]['errors']==['scene_grounding_not_unique_target']
    assert recovered['model_calls_used']==3+identity_calls.count <= 12
    assert recovered['rejected_source_views'][0]['reason']=='scene_image_identity_unresolved'
    # A different NATIVE owner needs the same new-source path in V2. The
    # previous wrong-ID rejection remains failed, and two NEW checks pass.
    native_original=original.model_copy(deep=True)
    native_original.task['scene_source_retry_policy']=scene_grounding.NATIVE_OWNER_RETRY_POLICY
    native_calls=IdentityCalls();native_calls.wrong_native=True
    native_initial=initial_run(native_original,native_calls,2,12,12)
    native_initial.update({k:initial[k] for k in ('phase','model_calls_used','admission',
        'candidate_appearance','target_precheck_observation','attempts')})
    native_root=recovery/'native-runs'
    RunStore(native_root/'runs.sqlite3').create(native_initial)
    # Reuse the independently prepared next-frame assets, not the result.
    with monkeypatch.context() as context:
        context.setattr(runtime,'EVIDENCE_ROOT',recovery/'evidence')
        context.setattr(scene_grounding,'EVIDENCE_ROOT',recovery/'evidence')
        native_recovered=runtime.run_agent(native_original,native_calls,2,native_root,12,12,
            resume_run_id=native_initial['run_id'],alternate_frame=better_frame)
    assert native_recovered['diagnostic_pass'] and native_recovered['model_calls_used']==3+native_calls.count<=12
    assert len(native_recovered['attempts'][-1]['ground_checks'])==2
    assert not native_recovered['attempts'][0]['ground_checks'][0]['diagnostic_pass']


def test_multiple_successes_count_unique_objects_until_goal(harness, monkeypatch, scene):
    # The task goal accepts the supported upper boundary without changing budgets.
    large_spec = BatchSpec(catalog_id='a' * 24, prepare_only=False, desired_count=2000)
    assert large_spec.desired_count == 2000 and large_spec.max_model_calls is None
    assert large_spec.calls_per_target == 12 and large_spec.target_concurrency == 4
    with pytest.raises(ValueError):
        BatchSpec(catalog_id='a' * 24, prepare_only=False, desired_count=2001)
    from app.agent.batch_execution import FrameAffinityQueue
    import copy
    queue_items = [{'ordinal': n, 'status': 'pending', 'target': {'context': 'ctx',
        'timestamp': frame}, 'queue_position': n} for n, frame in enumerate([1, 1, 2, 3, 4, 1, 2])]
    frozen_items = copy.deepcopy(queue_items)
    affinity = FrameAffinityQueue(queue_items, enabled=True)
    active = []
    for _ in range(4):
        active.append(affinity.take(active)['ordinal'])
    assert active == [0, 2, 3, 4]  # Fill slots without racing same-frame peers.
    affinity.finished(0)
    active.remove(0)
    assert affinity.take(active)['ordinal'] == 1
    assert affinity.take([1, *active])['ordinal'] == 5  # No artificial idle slot.
    assert affinity.take()['ordinal'] == 6
    with pytest.raises(StopIteration):
        affinity.take()
    assert queue_items == frozen_items
    fifo = FrameAffinityQueue(queue_items)
    assert [fifo.take()['ordinal'] for _ in queue_items] == list(range(7))
    reserved = copy.deepcopy(queue_items)
    reserved[-1]['run_id'] = 'recover-existing-request'
    affinity = FrameAffinityQueue(reserved, enabled=True)
    assert affinity.take()['ordinal'] == 6  # Durable recovery precedes fresh work.
    service, worker, catalog, prepared, calls = harness
    import app.datasets.catalog as catalog_module
    import app.agent.batch as batch_module
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor
    cache_root = worker.root/'catalog-cache'
    catalog_id = ident(9900)
    path = cache_root/catalog_id/'catalog.json'
    frozen = {'catalog_id':catalog_id,'sources':{'ctx':{'lidar_box':{'file':'source.parquet'}}},
        'targets':[{'target_id':'track-a','status':'selected','selection':{'native':'full-evidence'}}]}
    atomic_json(path,frozen)
    catalog_reads = []
    original_read = Path.read_text
    def tracked_read(file,*args,**kwargs):
        if file == path: catalog_reads.append(file)
        return original_read(file,*args,**kwargs)
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    with monkeypatch.context() as caching:
        caching.setattr(Path,'read_text',tracked_read)
        with ThreadPoolExecutor(max_workers=4) as readers:
            results = list(readers.map(lambda n:catalog_module.catalog_digest(catalog_id,cache_root)
                if n%2 else catalog_module.load_catalog(catalog_id,cache_root,compact=True),range(8)))
        assert len(catalog_reads)==1 and results[1]==digest(frozen)
        assert 'selection' not in results[0]['targets'][0]
        # Real detail display and resume checks share the cache; execution
        # still gets a fresh, complete catalog and cannot mutate cached facts.
        caching.setattr(batch_module,'load_catalog',lambda identity,**kw:catalog_module.load_catalog(identity,cache_root,**kw))
        caching.setattr(batch_module,'catalog_digest',lambda identity:catalog_module.catalog_digest(identity,cache_root))
        caching.setattr(service,'catalog_loader',batch_module.load_catalog)
        header = {'status':'paused','executor_active':False,'method_sha256':batch_module.method_fingerprint(),
            'catalog_sha256':digest(frozen),'spec':{'catalog_id':catalog_id}}
        assert service.display_catalog(catalog_id)['sources']==frozen['sources']
        assert service.resume_requirements(header)['mode']=='resume' and len(catalog_reads)==1
        complete = service.catalog_loader(catalog_id)
        assert complete==frozen
        complete['targets'][0]['selection']['native']='caller-change'
        assert catalog_module.catalog_digest(catalog_id,cache_root)==digest(frozen)
        # Same-length edits invalidate both display and the full frozen digest.
        before = path.stat()
        frozen['targets'][0]['target_id']='track-b'
        atomic_json(path,frozen)
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000_000))
        assert path.stat().st_size==before.st_size
        assert service.resume_requirements(header)['reason']=='catalog_changed'
        assert service.display_catalog(catalog_id)['targets'][0]['target_id']=='track-b'
        assert len(catalog_reads)==3 and catalog_module._display_catalog.cache_info().currsize==1
        path.unlink()
        assert service.resume_requirements(header)['mode']=='unavailable'
    # Native preparation overlaps across contexts, while the official SDK and
    # JSON registry stay serial. Barriers fail if preparation is still global
    # or a shared reader is switched underneath an in-flight evidence build.
    import app.datasets.preparation as preparation_module
    from types import SimpleNamespace
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event, Lock
    import app.agent.batch_execution as execution_module
    prep_root = worker.root/'parallel-prep'
    prep_root.mkdir()
    entered = Barrier(2)
    source_readers = []
    reader_pools = []
    registered = []
    counters = {'registry':0,'registry_max':0,'converter':0,'converter_max':0,'converted':0}
    count_lock = Lock()
    class Readers:
        def __init__(self, root, *, source_cache=None, source_lock=None):
            self.readers = {}
            reader_pools.append((source_cache,source_lock))
        def get(self, context):
            if context not in self.readers:
                self.readers[context] = SimpleNamespace(context=context,
                    provenance=lambda name, ctx: {'file':ctx,'sha256':ctx})
            return self.readers[context]
    packets = {}
    for n in (1,2):
        candidate = packet().model_copy(deep=True)
        candidate.context = str(n)
        candidate.frame_timestamp_micros = 1
        candidate.packet_id = ident(9700+n)
        candidate.scene = SceneEvidence(scene_id=ident(9710+n), manifest_path='scene.json',
            manifest_sha256='a'*64,total_points=10,target_points=1)
        candidate.provenance['artifact_sha256'] = {}
        packets[str(n)] = candidate
    def build(reader, context, timestamp, object_id, **kwargs):
        source_readers.append(reader)
        entered.wait(timeout=10)
        assert reader.context == context
        assert kwargs['prepared_scene'][0].context == context
        return packets[context]
    def read(path):
        context = path.name
        return SimpleNamespace(context=context,frame_timestamp_micros=1,
            source_components={'lidar_box':{'file':context,'sha256':context}}), {}
    def registrar(candidate):
        with count_lock:
            counters['registry'] += 1
            counters['registry_max'] = max(counters['registry_max'],counters['registry'])
        previous = list(registered)
        time.sleep(.01)
        registered[:] = [*previous,candidate.packet_id]
        with count_lock:counters['registry'] -= 1
    with monkeypatch.context() as concurrent_preparation:
        concurrent_preparation.setattr(preparation_module,'ROOT',prep_root)
        concurrent_preparation.setattr(preparation_module,'ContextReaders',Readers)
        concurrent_preparation.setattr(preparation_module,'preflight_target',lambda *a:True)
        concurrent_preparation.setattr(preparation_module,'read_scene',read)
        concurrent_preparation.setattr(preparation_module,'build_packet',build)
        concurrent_preparation.setattr(preparation_module,'hash_assets',lambda *a,**kw:kw['expected'])
        concurrent_preparation.setattr('app.datasets.correspondence.has_multimodal_evidence',lambda p:True)
        atomic_json(prep_root/'runtime/scenes/index.json',{'scenes':[
            {'context':ctx,'timestamp':1,'path':f'{ctx}/scene.json'} for ctx in packets]})
        native_preparer = preparation_module.TargetPreparer()
        store = SimpleNamespace(command=lambda job:None,update=lambda *a,**kw:None)
        native_service = SimpleNamespace(runner=worker,store=store,preparer=native_preparer,registrar=registrar)
        native_catalog = {'sources':{ctx:{'lidar_box':{'file':ctx,'sha256':ctx}} for ctx in packets}}
        pipeline = execution_module.BatchExecution(native_service,'parallel','owner',native_catalog)
        pipeline.spec = SimpleNamespace(prepare_only=False)
        pipeline.renew_executor = lambda:None
        assert pipeline.preparation_concurrency == 2
        def prepare_one(n):
            candidate = packets[str(n)]
            return pipeline.prepare_item({'ordinal':n,'packet_id':None,'run_id':None,'target':{
                'target_id':candidate.packet_id,'context':candidate.context,'timestamp':1,
                'object_id':candidate.target_object_id,'category':candidate.target.category}})
        with ThreadPoolExecutor(max_workers=2) as prep_pool:
            results = list(prep_pool.map(prepare_one,(1,2)))
        assert [p.context for p in results] == ['1','2']
        assert len(source_readers) == 2 and source_readers[0] is not source_readers[1]
        assert len(reader_pools) == 2 and reader_pools[0][0] is reader_pools[1][0]
        assert reader_pools[0][1] is reader_pools[1][1]
        assert set(registered) == {p.packet_id for p in results} and counters['registry_max'] == 1

        # Two stale scene-index misses converge on ONE conversion after waiting.
        converter_entered, release_converter, waiting_converter = Event(), Event(), Event()
        def convert(context,timestamp,heartbeat,log_path):
            with count_lock:
                counters['converter'] += 1
                counters['converted'] += 1
                counters['converter_max'] = max(counters['converter_max'],counters['converter'])
            converter_entered.set()
            assert release_converter.wait(10)
            atomic_json(prep_root/'runtime/scenes/index.json',{'scenes':[
                {'context':context,'timestamp':timestamp,'path':'converted/scene.json'}]})
            with count_lock:counters['converter'] -= 1
        native_preparer.converter = convert
        def convert_one(n):
            native_preparer._convert_scene('shared',2,
                lambda stage=None:waiting_converter.set() if stage=='waiting_for_official_converter' else None,
                lambda:False,prep_root/str(n)/'prep.log')
        with ThreadPoolExecutor(max_workers=2) as prep_pool:
            first = prep_pool.submit(convert_one,1)
            try:
                assert converter_entered.wait(10)
                second = prep_pool.submit(convert_one,2)
                assert waiting_converter.wait(10)
            finally:release_converter.set()
            first.result(timeout=10);second.result(timeout=10)
        assert counters['converter_max'] == counters['converted'] == 1
        # A converter waiter renews through heartbeat and can pause without ever
        # entering the worker; cancellation also releases its acquired lock.
        cancel, renewed = Event(), Event()
        native_preparer._conversion_lock.acquire()
        with ThreadPoolExecutor(max_workers=1) as prep_pool:
            def cancel_waiter():
                with native_preparer._conversion_slot(lambda stage=None:renewed.set(),cancel.is_set):
                    raise AssertionError('Cancelled waiter must not enter converter')
            paused = prep_pool.submit(cancel_waiter)
            try:
                assert renewed.wait(10)
                cancel.set()
                with pytest.raises(preparation_module.PreparationCancelled):paused.result(timeout=10)
            finally:native_preparer._conversion_lock.release()
        assert native_preparer._conversion_lock.acquire(blocking=False)
        native_preparer._conversion_lock.release()
        assert execution_module.BatchExecution(service,'custom','owner',catalog).preparation_concurrency == 1
    # Native provider sessions are private to a target thread and close after
    # its batch; custom factories retain per-target ownership and config checks.
    import app.agent.batch_execution as execution_module
    from concurrent.futures import ThreadPoolExecutor
    native_clients=[]
    class WarmCalls(LocalCalls):
        def __init__(self):
            super().__init__()
            self.closed=False
            native_clients.append(self)
        def close(self):
            self.closed=True
    with monkeypatch.context() as warm:
        warm.setattr(execution_module,'QwenCalls',WarmCalls)
        warm.setattr(worker,'calls_factory',WarmCalls)
        execution=execution_module.BatchExecution(service,'session-test','owner',catalog)
        execution.provider={'model':WarmCalls.model,'base_url':WarmCalls.base_url}
        first,owned=execution.target_calls()
        assert not owned and execution.target_calls()==(first,False)
        with ThreadPoolExecutor(max_workers=1) as session_pool:
            other,owned=session_pool.submit(execution.target_calls).result()
            assert other is not first and not owned
            assert session_pool.submit(execution.target_calls).result()==(other,False)
        warm.setenv('DASHSCOPE_MODEL','changed-model')
        with pytest.raises(execution_module.BatchConfigurationError):
            execution.target_calls()
        execution.close_clients()
        assert len(native_clients)==2 and all(c.closed for c in native_clients)
        warm.setattr(worker,'calls_factory',LocalCalls)
        custom,owned=execution.target_calls()
        assert owned and custom not in native_clients
    # An old zero-point representative is retried in-place after a code update;
    # the accepted alternate keeps the native track and the original task goal.
    import app.agent.agent_runtime as fallback_runtime
    from app.datasets.preparation import TargetPreparer, EvidenceRejected
    good = packet().model_copy(deep=True)
    good.packet_id, good.frame_timestamp_micros = ident(9801), 11
    good.scene = SceneEvidence(scene_id=ident(9800), manifest_path='scene.json',
        manifest_sha256='a'*64,total_points=10,target_points=1)
    zero = good.model_copy(deep=True)
    zero.packet_id, zero.frame_timestamp_micros = ident(9802), 10
    zero.scene = zero.scene.model_copy(update={'target_points':0})
    fallback_target = {'target_id':ident(9803),'ordinal':1,'context':good.context,
        'object_id':good.target_object_id,'category':good.target.category,'timestamp':10,
        'status':'selected','frame_exclusion_counts':{},'alternative_selections':[{'timestamp':11}]}
    fallback_catalog = {'catalog_id':ident(9804),'sources':{},'targets':[fallback_target]}
    registered = []
    class InitialFramePreparer(TargetPreparer):
        def __init__(self):
            pass
        def close(self):
            pass
        def __call__(self, selected, *args):
            if selected['timestamp'] == 10:
                raise EvidenceRejected(zero)
            return good
    fallback_calls = []
    class FallbackCalls(LocalCalls):
        def __init__(self):
            super().__init__()
            fallback_calls.append(self)
        def close(self):
            pass
    fallback_worker = ConsoleRunner(worker.root/'preparation-recovery',FallbackCalls)
    fallback_service = BatchService(fallback_worker,InitialFramePreparer(),lambda _:fallback_catalog,registered.append)
    with monkeypatch.context() as fallback_patch:
        fallback_patch.setattr('app.agent.batch.run_agent',fallback_runtime.run_agent)
        fallback_patch.setattr('app.agent.batch.load_packet',lambda pid:{zero.packet_id:zero,good.packet_id:good}[pid])
        try:
            fallback_job,_ = fallback_service.create(BatchSpec(catalog_id=fallback_catalog['catalog_id'],
                prepare_only=False,desired_count=1))
            fallback_service.store.update(fallback_job,'historical_preparation_rejection',{},status='paused',
                ordinal=1,item_status='abstained',reason='target_evidence_not_observable',packet_id=zero.packet_id)
            with fallback_worker.store.connect() as db:
                db.execute('UPDATE batches SET method_sha256=? WHERE job_id=?',('0'*64,fallback_job))
            old = fallback_service.store.detail(fallback_job)
            assert old['items'][0]['retryable_evidence_failure']
            assert old['items'][0]['display_reason'] == 'converted_target_points_missing'
            fallback_service.control(fallback_job,'resume');wait(fallback_worker)
            recovered = fallback_service.store.detail(fallback_job)
            assert recovered['status'] == 'completed'
            assert 4 <= recovered['model_calls_reserved'] == sum(c.count for c in fallback_calls) <= 12
            assert recovered['selection_progress']['diagnostic_passed'] == 1
            assert recovered['items'][0]['display_target']['timestamp'] == 11
            recovered_run = fallback_worker.store.snapshot(recovered['items'][0]['run_id'])
            assert recovered_run['rejected_source_views'][0]['timestamp'] == 10
            assert recovered_run['binding_target']['timestamp'] == 10
            assert recovered_run['sample_identity_validation']['passed']
            assert recovered_run['preparation_source_reselection']['timestamp'] == 11
            assert len(recovered_run['attempts'][-1]['ground_checks']) == 2
            assert {p.packet_id for p in registered} == {zero.packet_id,good.packet_id}
            assert any(e['kind']=='preparation_source_reselected' for e in recovered['events'])
            assert not any(e['kind']=='evidence_prepared' for e in recovered['events'])
            # A crash while recording the frame receipt rolls back its packet
            # link in the same transaction; the previous valid sample survives.
            original_event = fallback_service.store.event
            with monkeypatch.context() as interrupted_commit:
                def failed_receipt(db,job,kind,payload):
                    if kind == 'preparation_source_reselected':raise RuntimeError('interrupted receipt commit')
                    original_event(db,job,kind,payload)
                interrupted_commit.setattr(fallback_service.store,'event',failed_receipt)
                with pytest.raises(RuntimeError,match='interrupted receipt'):
                    fallback_service.store.update(fallback_job,'preparation_source_reselected',{'ordinal':1},
                        ordinal=1,item_status='pending',packet_id=zero.packet_id)
            assert fallback_service.store.detail(fallback_job)['items'][0]['packet_id'] == good.packet_id
        finally:
            fallback_service.close();fallback_worker.close()
    def forbidden_history_scan(*args, **kwargs):
        raise AssertionError('Target dispatch must use a primary-key run lookup')
    monkeypatch.setattr(worker.store, 'snapshots', forbidden_history_scan)
    with monkeypatch.context() as config:
        config.setenv('DASHSCOPE_MODEL', 'deepseek-v4-flash-0731')
        config.setenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        config.setenv('DASHSCOPE_API_KEY', 'local-test-key')
        import app.routers.system as readiness
        config.setattr(readiness, 'load_catalog', lambda **kwargs: {'summary':{'selected':0}, 'targets':[]})
        config.setattr(worker.store,'all_runs',lambda:pytest.fail('Readiness must not load complete run history'))
        status = readiness.status(SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(console=worker))))
        assert not status['annotation_ready']
        check = next(c for c in status['checks'] if c['name']=='模型配置')
        assert not check['ready'] and '仅支持文本' in check['detail']
        with pytest.raises(ValueError, match='仅支持文本'):
            service.create(spec(catalog))
        assert not prepared and not calls
    # Replacements must be able to exceed the former goal * 12 task quota.
    outcomes(monkeypatch, worker, fail_ordinals={1, 3}, pause_after=1, failure_calls=12)
    specification = BatchSpec(catalog_id=catalog['catalog_id'], target_ids=pool(catalog),
        prepare_only=False, desired_count=2, target_concurrency=1)
    job, _ = service.create(specification)
    service.start(job); wait(worker)
    detail = service.store.detail(job)
    assert detail['status'] == 'paused' and detail['model_calls_reserved'] == 12
    import app.routers.batches as workspace_routes
    workspace_request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(batches=service)))
    with worker.store.connect() as db:
        db.execute("UPDATE runs SET snapshot=json_set(snapshot,'$.candidate_appearance',json(?)) WHERE run_id=?",
            (json.dumps({'unused':'LIVE_VIEW_MUST_NOT_DECODE_EVIDENCE'}),detail['items'][0]['run_id']))
        service.store.event(db,job,'preparation_progress',{'ordinal':1,'stage':'OLD_PROGRESS_NOT_NEEDED'})
        service.store.event(db,job,'preparation_progress',{'ordinal':1,'stage':'building_evidence'})
        service.store.event(db,job,'preparation_progress',{'ordinal':3,'stage':'validating_scene'})
    full_view=workspace_routes.detail(job,workspace_request)
    with monkeypatch.context() as compact_reads:
        original_loads=json.loads
        def necessary_payload_only(value,*args,**kwargs):
            assert 'LIVE_VIEW_MUST_NOT_DECODE_EVIDENCE' not in value
            assert 'OLD_PROGRESS_NOT_NEEDED' not in value
            return original_loads(value,*args,**kwargs)
        compact_reads.setattr(json,'loads',necessary_payload_only)
        compact_response=workspace_routes.detail(job,workspace_request,compact=True)
        assert compact_response.media_type=='application/json'
        compact_view=json.loads(compact_response.body.decode())
    for field in ('counts','selection_progress','model_calls_reserved','resume','status'):
        assert compact_view[field]==full_view[field]
    live_view=json.loads(workspace_routes.progress(job,workspace_request).body)
    for field in ('counts','selection_progress','model_calls_reserved','status','control','executor_active'):
        assert live_view[field]==full_view[field]
    assert 'items' not in live_view and 'events' not in live_view
    assert len(json.dumps(live_view)) < 4000
    assert workspace_routes.progress(job,workspace_request).headers['cache-control']=='no-store'
    assert [(i['ordinal'],i['target']['context'],i['target']['object_id'],i.get('expression'))
        for i in compact_view['items']]==[(i['ordinal'],i['target']['context'],i['target']['object_id'],i.get('expression'))
        for i in full_view['items']]
    last_item_event=lambda v:next((e for e in reversed(v['events']) if isinstance(e['payload'].get('ordinal'),int)),None)
    assert last_item_event(compact_view)==last_item_event(full_view)
    for field in full_view['timing'].keys()-{'wall_seconds'}:
        assert compact_view['timing'][field]==full_view['timing'][field]
    for ordinal in (1,3):
        latest=lambda v:next(e for e in reversed(v['events']) if e['kind']=='preparation_progress' and e['payload']['ordinal']==ordinal)
        assert latest(compact_view)==latest(full_view)
    # Cache invalidation follows snapshot transactions, including same-revision
    # edits and an ID deleted/recreated within one transaction. Rollback restores
    # the original generation; neither a changed nor a mutable cached fact leaks.
    rid=detail['items'][0]['run_id']
    with worker.store.connect() as db:
        original=service.store.compact_states(db,job)[rid]
        for recreate in (False,True):
            db.execute('BEGIN IMMEDIATE')
            snapshot=db.execute('SELECT snapshot FROM runs WHERE run_id=?',(rid,)).fetchone()[0]
            replacement=json.loads(snapshot);replacement['phase']='cache-invalidated'
            if recreate:
                db.execute('DELETE FROM runs WHERE run_id=?',(rid,))
                db.execute('INSERT INTO runs(run_id,snapshot) VALUES(?,?)',(rid,json.dumps(replacement)))
            else:
                db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(replacement),rid))
            assert service.store.compact_states(db,job)[rid]['phase']=='cache-invalidated'
            assert json.loads(db.execute('SELECT snapshot FROM run_live_summaries WHERE run_id=?',(rid,)).fetchone()[0])['status']==replacement['status']
            db.rollback()
            assert service.store.compact_states(db,job)[rid]==original
        caller=service.store.compact_states(db,job)
        caller[rid]['phase']='caller-only'
        assert service.store.compact_states(db,job)[rid]==original
    with monkeypatch.context() as compact_catalog:
        compact_catalog.setattr(workspace_routes,'load_catalog',lambda **kwargs:{**catalog,
            'targets':[{**t,'context':t.get('context','excluded-context')} for t in catalog['targets']],
            'summary':{'selected':len(catalog['targets'])},'selection_policy':'test'})
        page=json.loads(workspace_routes.catalog(offset=0,limit=10000,compact=True).body)
        assert page['total']==len(catalog['targets'])
        assert [t['target_id'] for t in page['targets']]==[t['target_id'] for t in catalog['targets']]
        assert all('selection' not in t and 'alternative_selections' not in t for t in page['targets'])
    execution = service.store.execution_detail(job)
    assert execution['spec'] == detail['spec']
    assert execution['model_calls_reserved'] == detail['model_calls_reserved']
    assert [{k:item[k] for k in execution['items'][0]} for item in detail['items']] == execution['items']
    assert detail['timing']['complete'] and detail['timing']['active_seconds'] > 0
    from app.infrastructure.batch_timing import summarize_timing
    paused, _ = summarize_timing(detail['events'], status='paused', executor_active=False, now=time.time() + 600)
    assert paused['active_seconds'] == detail['timing']['active_seconds']
    outcomes(monkeypatch, worker, fail_ordinals={1, 3}, pause_after=3, failure_calls=12)
    service.control(job, 'resume'); wait(worker)
    detail = service.store.detail(job)
    assert detail['status'] == 'paused' and detail['model_calls_reserved'] == 24
    service.store.update(job, 'test_executor_interruption', {}, status='blocked', reason='test_interruption')
    from app.routers.batches import continue_pending
    request = SimpleNamespace(headers={}, app=SimpleNamespace(state=SimpleNamespace(batches=service)))
    recovery = continue_pending(job, request)
    service.start(recovery['job_id']); wait(worker)
    detail = service.store.detail(recovery['job_id'])
    assert recovery['job_id'] == job
    assert prepared == [1, 3, 4, 2]  # Preserve the original queue across recovery.
    assert detail['reason'] == 'description_goal_reached'
    assert detail['selection_progress']['diagnostic_passed'] == 2
    assert detail['spec']['max_model_calls'] is None
    assert detail['model_calls_reserved'] == 32
    assert sum(c.count for c in calls) == 32
    from scripts.profile_annotation import profile_runs
    timing_report = profile_runs(worker.root, job_ids=[job])
    batch_report = timing_report['batches'][0]
    assert batch_report['job_id'] == job and batch_report['status'] == 'completed'
    assert batch_report['performance']['status'] == 'complete'
    assert batch_report['performance']['measurements']['preparation_slot_wait']['count'] > 0
    assert {r['run_id'] for r in timing_report['runs']} == set(batch_report['run_ids'])
    assert timing_report['requested_missing_job_ids'] == []
    ledger = service.store.continuation_budget(job)
    assert ledger['limit'] is None and ledger['spent'] == 32
    from app.routers.batches import report
    report_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(batches=service)))
    # The task index must survive a slow/broken sample detail reader, and keep
    # completed tasks and their counters visible after a page/server reload.
    from app.routers.batches import index
    expected = {jid: service.store.detail(jid) for jid in ledger['job_ids']}
    with monkeypatch.context() as listing:
        def no_sample_reads(*args, **kwargs):
            raise AssertionError('The task index must not render individual samples')
        listing.setattr(service.store, 'detail', no_sample_reads)
        listing.setattr(worker.store, 'snapshot', no_sample_reads)
        listing.setattr(worker.store, 'all_runs', no_sample_reads)
        listing.setattr(service.store, 'execution_progress', no_sample_reads)
        summaries = index(report_request)['jobs']
    assert {row['job_id'] for row in summaries} == set(expected)
    for row in summaries:
        prior = expected[row['job_id']]
        for key in ('status', 'counts', 'executor_active'):
            assert row[key] == prior[key]
        assert row['selection_progress'] == {'requested': prior['spec'].get('desired_count')}
        assert row['continuation']==prior['continuation']
        assert row['spec'] == {k:v for k,v in prior['spec'].items() if k != 'target_ids'}
        assert not {'items', 'events', 'owner'} & row.keys()
    # Reports must keep the full-run metric semantics without loading every
    # run/evidence snapshot into memory or reading unrelated task histories.
    from app.quality.evaluation import summarize
    projected=worker.store.report_runs(recovery['job_id'])
    linked_ids={i['run_id'] for i in detail['items'] if i['run_id']}
    baseline=[state for state in worker.store.all_runs() if state.get('batch_id')==recovery['job_id'] or state['run_id'] in linked_ids]
    assert summarize(projected)==summarize(baseline)
    assert all('current_packet' not in state and 'candidate_appearance' not in state for state in projected)
    with monkeypatch.context() as bounded_report:
        bounded_report.setattr(worker.store,'all_runs',lambda:pytest.fail('A task report must not read the full database'))
        metrics = report(recovery['job_id'], report_request)['core_metrics']
    assert metrics['counts']['attempted'] == 4 and metrics['counts']['passed'] == 2
    assert metrics['coverage']['dataset_utilization']['value'] == 2 / 5
    assert metrics['coverage']['pool_utilization']['value'] == .5
    assert metrics['speed']['item_active_seconds']['count'] == 4
    assert metrics['speed']['mean_active_seconds_per_new_pass'] == detail['timing']['active_seconds'] / 2
    assert not metrics['cost']['tokens_complete'] and metrics['cost']['tokens_per_new_pass'] is None
    download = report(recovery['job_id'], report_request, download=True)
    exported = json.loads(download.body)
    assert exported['method_sha256'] == detail['method_sha256']
    assert exported['core_metrics'] == metrics
    assert 'attachment' in download.headers['content-disposition']
    from app.quality.batch_metrics import task_metrics
    final_states = [worker.store.snapshot(i['run_id']) for i in detail['items'] if i['run_id']]
    recorded_calls = [{'status':'completed', 'usage':{'total_tokens':10}, 'elapsed_seconds':1} for _ in range(32)]
    complete_cost = task_metrics(detail, final_states, final_states, recorded_calls, {'acceptance':None, 'cohorts':[]})['cost']
    assert complete_cost['tokens_complete'] and complete_cost['model_seconds_complete']
    assert complete_cost['tokens_per_new_pass'] == 160
    recorded_calls[-1].pop('usage')
    missing_usage = task_metrics(detail, final_states, final_states, recorded_calls, {'acceptance':None, 'cohorts':[]})['cost']
    assert not missing_usage['tokens_complete'] and missing_usage['tokens_per_new_pass'] is None
    # A superseded run no longer linked by the item is still part of task cost.
    retired = {'run_id': 'e' * 24, 'batch_id': job, 'status': 'needs_evidence',
        'model_calls_used': 1, 'current_packet': {'context':'retired', 'target_object_id':'retired'}}
    worker.store.create(retired)
    call = worker.root / retired['run_id'] / 'model/00-observe'
    atomic_json(call / 'input.json', [])
    atomic_json(call / 'result.json', {'status':'completed', 'usage':{'total_tokens':11}, 'elapsed_seconds':2})
    root_metrics = report(job, report_request)['core_metrics']
    assert root_metrics['counts']['attempted'] == root_metrics['counts']['settled'] == 4
    assert root_metrics['counts']['passed'] == 2 and root_metrics['speed']['new_passes_per_active_hour'] > 0
    assert root_metrics['cost']['reserved_model_calls'] == 33
    assert root_metrics['cost']['model_calls'] == 1 and root_metrics['cost']['tokens'] == 11
    assert root_metrics['failure_reasons'] == [{'reason':'precheck_target_occluded_or_uncertain', 'count':2}]

    # Deterministic pause/resume and crash cases: persisted monotonic intervals
    # exclude an 82-second pause; an interrupted interval is never fabricated.
    def event(kind, at, **payload):
        return {'kind': kind, 'at': at, 'payload': payload}
    timeline = [event('batch_started_or_resumed', 100),
        event('timing_execution_started', 100, session='one'),
        event('timing_item_started', 101, session='one', ordinal=1),
        event('timing_item_finished', 111, session='one', ordinal=1, active_seconds=10),
        event('timing_item_started', 112, session='one', ordinal=2),
        event('timing_item_finished', 117, session='one', ordinal=2, active_seconds=5),
        event('timing_execution_finished', 118, session='one', active_seconds=18),
        event('batch_started_or_resumed', 200),
        event('timing_execution_started', 200, session='two'),
        event('timing_item_started', 201, session='two', ordinal=2)]
    live, live_items = summarize_timing(timeline, status='running', executor_active=True, now=211)
    assert live['active_seconds'] == 29 and live_items[2]['active_seconds'] == 15
    interrupted, _ = summarize_timing(timeline, status='running', executor_active=False, now=10000)
    assert interrupted['active_seconds'] is None and interrupted['status'] == 'incomplete'
    assert interrupted['recorded_active_seconds'] == 18 and interrupted['history_incomplete']
    # A later live session keeps the closed subtotal and its own elapsed time,
    # without estimating the crashed session or counting paused/downtime gaps.
    recovered = [*timeline, event('batch_started_or_resumed', 300),
        event('timing_execution_started', 300, session='three')]
    partial, _ = summarize_timing(recovered, status='running', executor_active=True, now=311)
    assert partial['active_seconds'] is None and partial['recorded_active_seconds'] == 29
    assert partial['running'] and partial['history_incomplete'] and not partial['complete']
    recovered.append(event('timing_execution_finished', 320, session='three', active_seconds=20))
    paused_partial, _ = summarize_timing(recovered, status='paused', executor_active=False, now=10000)
    assert paused_partial['recorded_active_seconds'] == 38 and paused_partial['history_incomplete']
    assert not paused_partial['running'] and paused_partial['active_seconds'] is None
    missing_start, _ = summarize_timing([event('batch_started_or_resumed', 1), *timeline[:7]],
        status='paused', executor_active=False, now=10000)
    assert missing_start['active_seconds'] is None and missing_start['recorded_active_seconds'] == 18
    assert missing_start['history_incomplete']
    timeline += [event('timing_item_finished', 221, session='two', ordinal=2, active_seconds=20),
        event('batch_completed', 221), event('timing_execution_finished', 222, session='two', active_seconds=22)]
    finished, timed_items = summarize_timing(timeline, status='completed', executor_active=False, now=10000)
    assert finished['active_seconds'] == 40 and finished['wall_seconds'] == 122
    assert finished['recorded_active_seconds'] == 40 and not finished['history_incomplete']
    assert timed_items[1]['active_seconds'] == 10 and timed_items[2]['active_seconds'] == 25
    from app.quality.batch_metrics import distribution
    assert distribution([10, None, 25]) == {'count':2, 'mean':17.5, 'p50':17.5, 'p95':24.25}
    legacy, _ = summarize_timing([event('batch_started_or_resumed', 100), event('batch_completed', 200)],
        status='completed', executor_active=False, now=10000)
    assert legacy['active_seconds'] is None and legacy['wall_seconds'] == 100
    assert legacy['recorded_active_seconds'] is None and legacy['history_incomplete']

    # Explicit historical quotas still apply transactionally; no infinite
    # sentinel is serialized, and no old allowance is silently rewritten.
    from app.infrastructure.storage import CallLimit
    finite, _ = service.create(spec(catalog, goal=2, budget=4))
    state = {'run_id':'finite-test', 'batch_id':finite, 'provider_scope':'local-test',
        'current_packet':{'context':'finite', 'target_object_id':'one'}, 'model_calls_used':4}
    worker.store.claim(state['run_id'], 'finite-owner', state)
    worker.store.save(state['run_id'], 'finite-owner', state, 'model_call_reserved', {})
    state['model_calls_used'] = 5
    with pytest.raises(CallLimit, match='Batch shared'):
        worker.store.save(state['run_id'], 'finite-owner', state, 'model_call_reserved', {})
    assert worker.store.snapshot(state['run_id'])['model_calls_used'] == 4
    worker.store.release(state['run_id'], 'finite-owner')
    inherited={'run_id':'inherited-audit','kind':'counterfactual_audit','parent_run_id':state['run_id'],
        'provider_scope':'local-test','model_calls_used':2}
    direct={**inherited,'run_id':'direct-audit','current_packet':state['current_packet'],'model_calls_used':1}
    worker.store.create(inherited);worker.store.create(direct)
    assert worker.store.target_usage_for('finite','one','local-test')==7
    assert worker.store.target_usage_for('finite','one','another-provider')==0
    # A historical model incompatibility must not be presented as an invisible
    # object. The recorded failure reason itself remains available for audit.
    first = service.store.detail(job)['items'][0]
    old = worker.store.snapshot(first['run_id'])
    old.update(model='deepseek-v4-flash-0731', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    with worker.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?', (json.dumps(old), old['run_id']))
    historical = service.store.detail(job)['items'][0]
    assert historical['provider_diagnosis'] == 'vision_model_text_only'
    assert historical['reason'] == first['reason']
    old.update(model=LocalCalls.model,base_url=LocalCalls.base_url,final_expression=None,attempts=[],
        stop_reason='precheck_target_not_visible',rejected_source_views=[{
            'reason':'precheck_no_distinguishing_cues','observation':{'target_visible':True}}])
    with worker.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(old),old['run_id']))
    service.store.update(job,'test_visible_then_fallback',{},ordinal=first['ordinal'],
        item_status='needs_evidence',reason='precheck_target_not_visible')
    historical = service.store.detail(job)['items'][0]
    assert historical['reason'] == 'precheck_target_not_visible'
    assert historical['display_reason'] == 'precheck_visible_but_ambiguous'

    exercise_manual_reviews(worker.root/'manual-reviews',monkeypatch,scene[0])

    # Exercise both the lower request cap and the new four-request pipeline.
    # Waiting targets must not reserve calls; pause/resume must settle every run.
    import threading
    from app.infrastructure.model_slots import ModelSlots
    import app.agent.agent_runtime as runtime
    # A retry already holds its semaphore slot, but another worker can extend
    # the provider cooldown. It must wait without acquiring a second slot.
    with monkeypatch.context() as cooldown:
        clock=[100.0]
        cooldown.setattr('app.infrastructure.model_slots.time',SimpleNamespace(
            monotonic=lambda:clock[0],sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)))
        limits=ModelSlots(1,interval=0)
        with limits.slot(lambda:None,lambda:None):
            limits.backoff(2)
            limits.wait_retry(lambda:None,lambda:None)
            assert clock[0]>=102
            limits.backoff(10)
            def cancel_retry():
                raise runtime.ControlRequested('pause')
            with pytest.raises(runtime.ControlRequested):
                limits.wait_retry(lambda:None,cancel_retry)
    with monkeypatch.context() as settings:
        settings.delenv('ANNOTATION_MODEL_CONCURRENCY', raising=False)
        default_worker = ConsoleRunner(worker.root/'default-limits', LocalCalls)
        try:
            assert default_worker.model_slots.concurrency == 4
            assert BatchSpec(catalog_id=catalog['catalog_id']).target_concurrency == 4
            validation_catalog = json.loads(json.dumps(catalog))
            for target in validation_catalog['targets']:
                target['official_split'] = 'validation'
            split_store = BatchStore(default_worker.store)
            with pytest.raises(ValueError,match='暂无候选'):
                split_store.create(BatchSpec(catalog_id=catalog['catalog_id']), validation_catalog)
            validation_job, _ = split_store.create(BatchSpec(catalog_id=catalog['catalog_id'],
                dataset_split='validation'), validation_catalog)
            assert split_store.detail(validation_job)['spec']['dataset_split'] == 'validation'
            assert all(item['target']['official_split']=='validation' for item in split_store.detail(validation_job)['items'])
            with pytest.raises(ValueError,match='任务数据集不符'):
                split_store.create(BatchSpec(catalog_id=catalog['catalog_id'],target_ids=pool(catalog)),validation_catalog)
        finally:
            default_worker.close()
        settings.setenv('ANNOTATION_MODEL_CONCURRENCY', '2')
        configured_worker = ConsoleRunner(worker.root/'configured-limits', LocalCalls)
        try:
            assert configured_worker.model_slots.concurrency == 2
        finally:
            configured_worker.close()
        for value in ('0','5','invalid'):
            settings.setenv('ANNOTATION_MODEL_CONCURRENCY', value)
            with pytest.raises(ValueError):
                ConsoleRunner(worker.root/'invalid-limits', LocalCalls)
        assert not (worker.root/'invalid-limits').exists()
    for api_limit, target_limit in ((2, 3), (4, 4)):
        entered, release, first_request = threading.Event(), threading.Event(), threading.Event()
        guard = threading.Lock()
        counters = {'api':0, 'peak_api':0, 'prep':0, 'peak_prep':0, 'prepared':0, 'overlap':False, 'calls':0}
        parallel_packets = {i: packet().model_copy(update={'packet_id':ident(8000+i),
            'context':f'parallel-context-{i}'}) for i in range(1,6)}
        parallel_catalog = {'catalog_id':ident(8000),'sources':{},'targets':[
            {'target_id':ident(8100+i),'ordinal':i,'context':p.context,'object_id':p.target_object_id,
                'timestamp':p.frame_timestamp_micros,'status':'selected','frame_exclusion_counts':{}}
            for i,p in parallel_packets.items()]}

        class ConcurrentCalls(LocalCalls):
            def close(self):
                pass
            def call(self, messages, schema, folder, heartbeat=None):
                with guard:
                    counters['api'] += 1
                    counters['peak_api'] = max(counters['peak_api'], counters['api'])
                    counters['calls'] += 1
                    first_request.set()
                    if counters['api'] == api_limit:
                        entered.set()
                try:
                    assert release.wait(30)
                    time.sleep(0.03)
                    return super().call(messages,schema,folder,heartbeat)
                finally:
                    with guard:
                        counters['api'] -= 1

        def parallel_prepare(target,catalog,heartbeat,cancelled,log):
            with guard:
                counters['prep'] += 1
                counters['prepared'] += 1
                later = counters['prepared'] > 1
                counters['peak_prep'] = max(counters['peak_prep'],counters['prep'])
            try:
                if later:
                    assert first_request.wait(30)
                with guard:
                    counters['overlap'] |= counters['api'] > 0
                time.sleep(0.03)
                return parallel_packets[target['ordinal']]
            finally:
                with guard:
                    counters['prep'] -= 1

        concurrent_worker = ConsoleRunner(worker.root/f'concurrent-{api_limit}',ConcurrentCalls, model_concurrency=api_limit)
        concurrent_worker.model_slots = ModelSlots(api_limit,interval=0)
        concurrent_service = BatchService(concurrent_worker,parallel_prepare,lambda _:parallel_catalog,lambda _:None)
        with monkeypatch.context() as concurrent:
            concurrent.setattr('app.agent.batch.run_agent',runtime.run_agent)
            concurrent.setattr('app.agent.batch.load_packet',lambda pid:next(p for p in parallel_packets.values() if p.packet_id==pid))
            try:
                concurrent_job,_ = concurrent_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                    prepare_only=False,desired_count=target_limit,target_concurrency=target_limit))
                concurrent_service.start(concurrent_job)
                assert entered.wait(30)
                deadline = time.monotonic()+30
                while len(concurrent_worker.store.all_runs()) < target_limit and time.monotonic()<deadline:
                    time.sleep(0.01)
                waiting = concurrent_worker.store.all_runs()
                assert len(waiting)==target_limit and sorted(s['model_calls_used'] for s in waiting)==[0]*(target_limit-api_limit)+[1]*api_limit
                concurrent_service.control(concurrent_job,'pause')
                release.set();wait(concurrent_worker)
                paused = concurrent_service.store.detail(concurrent_job)
                assert paused['status']=='paused' and paused['model_calls_reserved']==api_limit
                assert all(s['status']=='paused' and s['model_calls_used']<=1 for s in concurrent_worker.store.all_runs())
                concurrent_service.control(concurrent_job,'resume');wait(concurrent_worker)
                finished = concurrent_service.store.detail(concurrent_job)
                assert finished['status']=='completed' and finished['selection_progress']['diagnostic_passed']==target_limit
                assert len([i for i in finished['items'] if i['run_id']])==target_limit
                assert finished['counts']['not_needed']==len(parallel_packets)-target_limit
                baseline_calls = LocalCalls()
                run_packet(packet(),baseline_calls,max_repairs=0,output_root=worker.root/f'serial-reference-{api_limit}',max_model_calls=12)
                assert counters['calls']==finished['model_calls_reserved']==baseline_calls.count*target_limit+api_limit
                assert counters['peak_api']==api_limit and counters['peak_prep']==1 and counters['overlap']
                assert finished['timing']['complete']
                item_seconds = sum(i['timing']['active_seconds'] or 0 for i in finished['items'])
                assert item_seconds > finished['timing']['active_seconds']  # Overlap is not double-counted.
            finally:
                release.set();concurrent_service.close();concurrent_worker.close()

    # A provider failure must not strand paused sibling checkpoints in a blocked
    # parent. Resume them in their original task; never replay the rejected call.
    failure_ready, reject = threading.Event(), threading.Event()
    failure_guard = threading.Lock()
    failure_mode, arrivals = True, 0
    class ProviderFailureCalls(LocalCalls):
        def close(self):
            pass
        def call(self,messages,schema,folder,heartbeat=None):
            nonlocal arrivals
            if failure_mode:
                with failure_guard:
                    arrivals += 1
                    first = arrivals == 1
                    if arrivals >= 2:
                        failure_ready.set()
                if first:
                    assert reject.wait(30)
                    atomic_json(folder/'result.json',{'status':'api_failed','http_status':401,
                        'provider_error_code':'invalid_api_key','stream_chunk_count':0,'answer':''})
                    raise ModelCallFailure('api_failed')
                # This is a synchronization barrier, not a five-second latency SLA.
                # Concurrent export/file I/O must not fabricate extra provider failures.
                deadline = time.monotonic()+30
                while not failure_worker.store.control(folder.parents[1].name) and time.monotonic()<deadline:
                    time.sleep(0.01)
                assert failure_worker.store.control(folder.parents[1].name)=='pause'
            return super().call(messages,schema,folder,heartbeat)
    failure_worker = ConsoleRunner(worker.root/'failure-siblings',ProviderFailureCalls)
    failure_worker.model_slots = ModelSlots(2,interval=0)
    failure_service = BatchService(failure_worker,lambda t,*args:parallel_packets[t['ordinal']],
        lambda _:parallel_catalog,lambda _:None)
    with monkeypatch.context() as failure_patch:
        failure_patch.setattr('app.agent.batch.run_agent',runtime.run_agent)
        failure_patch.setattr('app.agent.batch.load_packet',lambda pid:next(p for p in parallel_packets.values() if p.packet_id==pid))
        try:
            failure_job,_ = failure_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                prepare_only=False,desired_count=3,target_concurrency=3))
            failure_service.start(failure_job);assert failure_ready.wait(30)
            reject.set();wait(failure_worker)
            paused = failure_service.store.detail(failure_job)
            assert paused['status']=='paused' and paused['reason']=='child_run_blocked'
            failed = [i for i in paused['items'] if i['status']=='failed']
            assert len(failed)==1
            rejected_id = failed[0]['run_id']
            assert failure_worker.store.snapshot(rejected_id)['auth_rejections']==1
            failure_mode = False
            failure_service.control(failure_job,'resume');wait(failure_worker)
            completed = failure_service.store.detail(failure_job)
            assert completed['status']=='completed' and completed['selection_progress']['diagnostic_passed']==3
            assert failure_worker.store.snapshot(rejected_id)['model_calls_used']==0
            assert failure_worker.store.snapshot(rejected_id)['auth_rejections']==1
        finally:
            reject.set();failure_service.close();failure_worker.close()

    # Recover an abandoned batch, reusing a completed response whose phase
    # commit was interrupted. A corrupt target must not stop its siblings.
    from app.agent.agent_runtime import initial_run
    prepared_after_restart=[]
    storage_unavailable=False
    class RestartCalls(LocalCalls):
        def close(self):
            pass
    def restart_prepare(target,*args):
        prepared_after_restart.append(target['ordinal'])
        if storage_unavailable:
            raise PermissionError('storage unavailable')
        if target['ordinal']==3:
            raise ValueError('damaged target input')
        return parallel_packets[target['ordinal']]
    restart_worker=ConsoleRunner(worker.root/'r/runtime/annotation',RestartCalls)
    restart_worker.model_slots=ModelSlots(2,interval=0)
    restart_service=BatchService(restart_worker,restart_prepare,lambda _:parallel_catalog,lambda _:None)
    with monkeypatch.context() as recovery_patch:
        recovery_patch.setattr('app.agent.batch.run_agent',runtime.run_agent)
        recovery_patch.setattr('app.agent.batch.load_packet',lambda pid:next(p for p in parallel_packets.values() if p.packet_id==pid))
        try:
            restarted,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                prepare_only=False,desired_count=3,target_concurrency=2))
            original=parallel_packets[1]
            initial=initial_run(original,RestartCalls(),2,12,12)
            initial['batch_id']=restarted
            restart_service.store.update(restarted,'test_prepared',{},ordinal=1,item_status='pending',packet_id=original.packet_id)
            restart_service.store.attach_child(restarted,1,initial)
            with pytest.raises(KeyboardInterrupt):
                runtime.run_agent(original,LocalCalls(crash_decision=True),2,restart_worker.root,12,12,
                    resume_run_id=initial['run_id'])
            assert restart_worker.store.snapshot(initial['run_id'])['model_calls_used']==1
            restart_service.store.update(restarted,'test_crash',{},status='running')
            restart_worker.store.acquire_executor('previous-process')
            restart_service.recover_running()
            assert restart_worker.active_batch is None and not prepared_after_restart
            with restart_worker.store.connect() as db:
                db.execute('UPDATE executor SET lease_until=0')
                db.execute("UPDATE batches SET control='pause' WHERE job_id=?",(restarted,))
            restart_service.recover_running()
            assert restart_worker.active_batch is None
            with restart_worker.store.connect() as db:
                db.execute('UPDATE batches SET control=NULL WHERE job_id=?',(restarted,))
            restart_service.recover_running();wait(restart_worker)
            completed=restart_service.store.detail(restarted)
            assert completed['status']=='completed' and completed['selection_progress']['diagnostic_passed']==3
            assert len([i for i in completed['items'] if i['status']=='failed'])==1
            assert 1 not in prepared_after_restart and 3 in prepared_after_restart
            assert restart_worker.store.snapshot(initial['run_id'])['model_calls_used']==6
            reservations=[e for e in restart_worker.store.events(initial['run_id']) if e['kind']=='model_call_reserved']
            assert sum(e['payload']['kind']=='decide' for e in reservations)==1
            assert any(e['kind']=='restart_recovery_started' for e in completed['events'])
            # Shared storage errors pause with the item pending; an explicit
            # pause never gets undone by the automatic restart sweep.
            storage_unavailable=True
            storage_job,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                target_ids=[parallel_catalog['targets'][4]['target_id']],prepare_only=False,desired_count=1))
            restart_service.start(storage_job);wait(restart_worker)
            paused=restart_service.store.detail(storage_job)
            assert paused['status']=='paused' and paused['items'][0]['status']=='pending'
            assert paused['model_calls_reserved']==0
            storage_unavailable=False
            restart_service.recover_running()
            assert restart_worker.active_batch is None
            restart_service.control(storage_job,'resume');wait(restart_worker)
            assert restart_service.store.detail(storage_job)['status']=='completed'
            changed,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],prepare_only=True))
            with restart_worker.store.connect() as db:
                db.execute("UPDATE batches SET status='running',method_sha256='changed' WHERE job_id=?",(changed,))
            restart_service.recover_running()
            incompatible=restart_service.store.detail(changed)
            assert incompatible['status']=='paused' and incompatible['reason']=='restart_configuration_or_input_changed'
            # A code update rewinds unfinished items inside the same task.
            # Successful links and all billed calls stay, unknown RPCs skip.
            frozen,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                prepare_only=False,desired_count=4,max_repairs=1))
            successful=restart_worker.store.snapshot(completed['items'][0]['run_id'])
            successful={**successful,'run_id':ident(9001),'batch_id':frozen,'model_calls_used':4}
            restart_worker.store.create(successful)
            restart_service.store.update(frozen,'test_passed',{},ordinal=1,item_status='calibration_hold',
                packet_id=parallel_packets[1].packet_id,run_id=successful['run_id'])
            for ordinal, inflight in ((2,None),(3,{'folder':'unknown_request','kind':'decide'})):
                saved=initial_run(parallel_packets[ordinal],RestartCalls(),1,12,12)
                saved.update(batch_id=frozen,status='paused',model_calls_used=2 if ordinal==2 else 1,inflight=inflight)
                if ordinal==2:
                    with restart_worker.store.connect() as db:
                        target=json.loads(db.execute('SELECT target FROM batch_items WHERE job_id=? AND ordinal=2',
                            (frozen,)).fetchone()['target'])
                        target['retry_authorization']={'token':'explicit-round','source_run_id':None,
                            'call_limit':22,'additional_calls':12,'authorized_at':1}
                        db.execute('UPDATE batch_items SET target=? WHERE job_id=? AND ordinal=2',
                            (json.dumps(target),frozen))
                    saved['cumulative_call_limit']=22
                restart_service.store.attach_child(frozen,ordinal,saved)
            with restart_worker.store.connect() as db:
                db.execute("UPDATE batches SET method_sha256='prior-code',status='paused',reason='RunBusy' WHERE job_id=?",(frozen,))
            before=restart_service.store.detail(frozen)
            assert restart_service.resume_requirements(before)['mode']=='recover'
            with monkeypatch.context() as stale_process:
                stale_process.setattr(restart_service,'loaded_method_sha256','old-loaded-modules')
                assert restart_service.resume_requirements(before)['reason']=='backend_restart_required'
                with pytest.raises(ValueError,match='重启后端服务'):
                    restart_service.control(frozen,'resume')
                assert restart_service.store.detail(frozen)['status']=='paused'
                assert restart_worker.active_batch is None
            from app.agent.batch_recovery import prepare_recovery, cleanup_temporary
            # A completed task reopened for one review must not retry unrelated
            # preparation failures merely because the loaded code changed.
            scoped_review,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],prepare_only=False))
            with restart_worker.store.connect() as db:
                review_target=json.loads(db.execute('SELECT target FROM batch_items WHERE job_id=? AND ordinal=1',
                    (scoped_review,)).fetchone()['target'])
                review_target['manual_review_retry']={'source_run_id':successful['run_id'],'review_id':'1'*64}
                db.execute('UPDATE batch_items SET target=? WHERE job_id=? AND ordinal=1',(json.dumps(review_target),scoped_review))
                db.execute("UPDATE batch_items SET status='failed',reason='preparation:official_conversion_failed:1' WHERE job_id=? AND ordinal<>1",(scoped_review,))
                db.execute("UPDATE batches SET method_sha256='prior-code',status='ready' WHERE job_id=?",(scoped_review,))
            scoped_preview=prepare_recovery(restart_service,scoped_review,dry_run=True)
            assert scoped_preview['recovery_scope']=='pending_review_retries' and scoped_preview['review_ordinals']==[1]
            assert scoped_preview['ready_targets']==1 and scoped_preview['retired_run_ids']==[]
            with restart_worker.store.connect() as db:
                assert db.execute("SELECT COUNT(*) FROM batch_items WHERE job_id=? AND status='failed'",(scoped_review,)).fetchone()[0]==4
            with monkeypatch.context() as preview_only:
                preview_only.setattr('app.agent.batch.freeze_method',lambda *a:pytest.fail('A dry run must not freeze or write files'))
                preview=prepare_recovery(restart_service,frozen,dry_run=True)
            assert preview['dry_run'] and preview['retained_passed']==1 and preview['remaining_goal']==3
            assert preview['reset_items']==1 and preview['skipped_unknown_outcome_targets']==1
            assert restart_service.store.detail(frozen)['status']=='paused'
            assert restart_service.store.detail(frozen)['method_sha256']=='prior-code'
            assert restart_worker.store.snapshot(successful['run_id'])==successful
            request=SimpleNamespace(headers={},app=SimpleNamespace(state=SimpleNamespace(batches=restart_service)))
            receipt=continue_pending(frozen,request)
            assert receipt['job_id']==frozen and receipt['started'] is False
            assert receipt['retained_passed']==1 and receipt['remaining_goal']==3
            assert receipt['reset_items']==1 and receipt['skipped_unknown_outcome_targets']==1
            successor=restart_service.store.detail(receipt['job_id'])
            assert successor['continuation'] is None
            indexed=next(j for j in restart_service.store.summaries() if j['job_id']==receipt['job_id'])
            assert indexed['continuation']==successor['continuation']
            assert successor['status']=='ready' and successor['spec']['desired_count']==4
            assert successor['selection_progress']['diagnostic_passed']==1 and successor['model_calls_reserved']==7
            assert {i['ordinal'] for i in successor['items']}=={1,2,3,4,5}
            carried=next(i for i in successor['items'] if i['ordinal']==1)
            retry_round=next(i for i in successor['items'] if i['ordinal']==2)['target']['retry_authorization']
            assert retry_round['call_limit']==22 and retry_round['additional_calls']==0
            assert retry_round['continuation_of']=='explicit-round' and 'consumed_run_id' not in retry_round
            assert carried['run_id']==successful['run_id'] and not carried['reused']
            assert restart_worker.store.snapshot(successful['run_id'])==successful
            assert next(i for i in successor['items'] if i['ordinal']==2)['run_id'] is None
            assert next(i for i in successor['items'] if i['ordinal']==3)['status']=='failed'
            assert any(e['kind']=='execution_revision' and e['payload']['previous']['method_sha256']=='prior-code'
                       for e in successor['events'])
            assert restart_service.store.continuation_budget(receipt['job_id'])['spent']==7
            assert restart_service.store.detail(frozen)['status']=='ready'
            assert continue_pending(frozen,request)['job_id']==receipt['job_id']
            assert restart_worker.active_batch is None
            # A stop before the first model call must get a fresh attempt even
            # when its configuration/budget are identical. Reuse the same task
            # ID and successful sample; clean private/unpublished temp assets.
            fresh = packet().model_copy(deep=True,update={'packet_id':ident(9706),'context':'rewind-fresh'})
            parallel_packets[6] = fresh
            parallel_catalog['targets'].append({'target_id':ident(9806),'ordinal':6,'context':fresh.context,
                'object_id':fresh.target_object_id,'timestamp':fresh.frame_timestamp_micros,
                'status':'selected','frame_exclusion_counts':{}})
            rewind,_=restart_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                target_ids=[parallel_catalog['targets'][0]['target_id'],ident(9806)],
                prepare_only=False,desired_count=2,max_model_calls=12))
            restart_service.store.update(rewind,'test_saved_success',{},ordinal=1,item_status='reused',
                packet_id=parallel_packets[1].packet_id,run_id=successful['run_id'],reused=True)
            unfinished=initial_run(fresh,RestartCalls(),2,12,12)
            unfinished.update(batch_id=rewind,status='stopped')
            restart_service.store.attach_child(rewind,6,unfinished)
            restart_service.store.update(rewind,'test_stopped',{},status='stopped')
            disposable=restart_worker.root/unfinished['run_id']/'draft.tmp'
            atomic_json(disposable,{'unfinished':True})
            trace=restart_worker.root/unfinished['run_id']/'model/00-unused/input.json'
            atomic_json(trace,{'record':'retain original call inputs'})
            progress=restart_worker.root/'batches'/rewind/'6/preparation-progress.json'
            atomic_json(progress,{'stage':'interrupted'})
            incomplete=restart_worker.root.parent/'scenes'/ident(9810)
            atomic_json(incomplete/'preparation.json',{'context':fresh.context})
            (incomplete/'points.tmp.npz').write_bytes(b'unfinished conversion')
            complete_cache=restart_worker.root.parent/'scenes'/ident(9811)
            atomic_json(complete_cache/'scene.json',{'keep':'published cache'})
            before_jobs={row['job_id'] for row in restart_service.store.summaries()}
            restart_service.control(rewind,'resume');wait(restart_worker)
            resumed=restart_service.store.detail(rewind)
            assert resumed['status']=='completed' and resumed['selection_progress']['diagnostic_passed']==2
            assert resumed['model_calls_reserved']==6
            assert {row['job_id'] for row in restart_service.store.summaries()}==before_jobs
            assert next(i for i in resumed['items'] if i['ordinal']==1)['run_id']==successful['run_id']
            current_run=next(i for i in resumed['items'] if i['ordinal']==6)['run_id']
            assert current_run != unfinished['run_id']
            assert restart_worker.store.snapshot(current_run)['current_packet']['task']['execution_attempt']
            assert restart_worker.store.snapshot(unfinished['run_id'])['retired_from_execution'] is True
            assert restart_worker.store.snapshot(successful['run_id'])==successful
            assert trace.is_file() and not disposable.exists() and not progress.exists()
            assert not incomplete.exists() and (complete_cache/'scene.json').is_file()
            assert restart_service.control(rewind,'resume')['reused_result']
            with monkeypatch.context() as no_global_history:
                no_global_history.setattr(restart_worker.store,'all_runs',lambda:pytest.fail('No unpublished cache needs a global history scan'))
                cleanup_temporary(restart_service,rewind)
        finally:
            restart_service.close();restart_worker.close()

    # Several targets can encounter a temporary outage without halting the
    # batch. Every retry is bounded/accounted, and the next healthy target runs.
    outage_rejection = {'http_status':503, 'contexts':{'parallel-context-1','parallel-context-2'}}
    class OutageCalls(RestartCalls):
        def call(self,messages,schema,folder,heartbeat=None):
            context=outage_worker.store.snapshot(folder.parents[1].name)['current_packet']['context']
            if context in outage_rejection['contexts']:
                self.count += 1
                atomic_json(folder/'result.json',{'status':'api_failed','http_status':outage_rejection['http_status']})
                raise ModelCallFailure('api_failed')
            return super().call(messages,schema,folder,heartbeat)
    outage_worker=ConsoleRunner(worker.root/'provider-outage',OutageCalls)
    outage_worker.model_slots=ModelSlots(1,interval=0)
    cooldowns=[]
    outage_worker.model_slots.backoff=lambda seconds=30:cooldowns.append(seconds)
    outage_service=BatchService(outage_worker,lambda t,*args:parallel_packets[t['ordinal']],
        lambda _:parallel_catalog,lambda _:None)
    with monkeypatch.context() as outage_patch:
        outage_patch.setattr('app.agent.batch.run_agent',runtime.run_agent)
        outage_patch.setattr('app.agent.model_calls.TRANSIENT_RETRY_DELAYS',(0,0,0))
        try:
            outage_job,_=outage_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                prepare_only=False,desired_count=1,target_concurrency=1,max_repairs=0))
            outage_service.start(outage_job);wait(outage_worker)
            completed=outage_service.store.detail(outage_job)
            assert completed['status']=='completed' and completed['selection_progress']['diagnostic_passed']==1
            assert completed['counts']['failed']==2 and completed['model_calls_reserved']==14
            assert all(i.get('retryable_connection_failure') for i in completed['items'] if i['status']=='failed')
            assert [e['payload']['delay_seconds'] for e in completed['events'] if e['kind']=='provider_cooldown']==[15,30]
            assert 15 in cooldowns and 30 in cooldowns
        finally:
            outage_service.close();outage_worker.close()

    # The reported HTTP 400 must settle one target, not pause healthy siblings.
    # Repeated input refusals still stop after three targets to bound wasted calls.
    for rejected_count in (1,3):
        outage_rejection.update(http_status=400,
            contexts={f'parallel-context-{i}' for i in range(1,rejected_count+1)})
        outage_worker=ConsoleRunner(worker.root/f'input-rejection-{rejected_count}',OutageCalls)
        outage_worker.model_slots=ModelSlots(1,interval=0)
        outage_service=BatchService(outage_worker,lambda t,*args:parallel_packets[t['ordinal']],
            lambda _:parallel_catalog,lambda _:None)
        with monkeypatch.context() as refused_patch:
            refused_patch.setattr('app.agent.batch.run_agent',runtime.run_agent)
            try:
                refused_job,_=outage_service.create(BatchSpec(catalog_id=parallel_catalog['catalog_id'],
                    prepare_only=False,desired_count=1,target_concurrency=1,max_repairs=0))
                outage_service.start(refused_job);wait(outage_worker)
                refused=outage_service.store.detail(refused_job)
                assert refused['counts']['failed']==rejected_count
                rejection_events=[e for e in refused['events'] if e['kind']=='target_request_rejected']
                assert len(rejection_events)==rejected_count
                assert all(e['payload']['http_status']==400 for e in rejection_events)
                if rejected_count==1:
                    assert refused['status']=='completed' and refused['selection_progress']['diagnostic_passed']==1
                    assert refused['model_calls_reserved']==7
                else:
                    assert refused['status']=='paused' and refused['reason']=='repeated_request_rejections'
                    assert refused['model_calls_reserved']==3
            finally:
                outage_service.close();outage_worker.close()


    # Durable FIFO: explicit starts enqueue, pause holds successors, completion
    # advances one execution slot, and restart never starts an unqueued task.
    queued_packets={i:packet().model_copy(update={'packet_id':ident(18000+i),'context':f'queue-{i}'}) for i in range(1,5)}
    queued_catalog={'catalog_id':ident(18000),'sources':{},'targets':[
        {'target_id':ident(18100+i),'ordinal':i,'context':p.context,'object_id':p.target_object_id,
         'timestamp':p.frame_timestamp_micros,'status':'selected','frame_exclusion_counts':{}}
        for i,p in queued_packets.items()]}
    entered=threading.Event();release=threading.Event();prepared_order=[]
    def queue_prepare(target,catalog,heartbeat,cancelled,log):
        prepared_order.append(target['ordinal'])
        if target['ordinal']==1 and prepared_order.count(1)==1:
            entered.set()
            assert release.wait(10)
            if cancelled():
                from app.datasets.preparation import PreparationCancelled
                raise PreparationCancelled('paused before publishing evidence')
        return queued_packets[target['ordinal']]
    queue_worker=ConsoleRunner(worker.root/'queue-persistence',LocalCalls)
    queue_service=BatchService(queue_worker,queue_prepare,lambda _:queued_catalog,lambda _:None)
    queue_jobs=[queue_service.create(BatchSpec(catalog_id=queued_catalog['catalog_id'],
        target_ids=[t['target_id']],prepare_only=True))[0] for t in queued_catalog['targets']]
    def queue_wait(predicate):
        for _ in range(500):
            if predicate():return
            time.sleep(.01)
        pytest.fail('Queue did not settle')
    try:
        queue_service.start(queue_jobs[0]);assert entered.wait(10)
        assert queue_service.start(queue_jobs[1])['queue_position']==1
        assert queue_service.start(queue_jobs[1])['queue_position']==1
        assert queue_service.start(queue_jobs[2])['queue_position']==2
        queue_service.control(queue_jobs[2],'pause')
        assert queue_service.store.detail(queue_jobs[2])['status']=='paused'
        assert queue_service.start(queue_jobs[2])['queue_position']==2
        queue_service.control(queue_jobs[0],'pause');release.set()
        queue_wait(lambda:queue_worker.active_batch is None)
        queue_service.recover_running()
        assert queue_service.store.detail(queue_jobs[1])['status']=='queued'
        assert queue_service.store.detail(queue_jobs[1])['queue_paused']
        assert prepared_order==[1]
        queue_service.control(queue_jobs[0],'resume')
        queue_wait(lambda:all(queue_service.store.detail(j)['status']=='completed' for j in queue_jobs[:3]))
        queue_wait(lambda:queue_worker.active_batch is None)
        assert prepared_order==[1,1,2,3] and not queue_service.store.queue()
        assert all(queue_service.store.detail(j)['model_calls_reserved']==0 for j in queue_jobs)
        # Persist an idle queue exactly as a process exit would leave it.
        queue_worker.active='other-owner'
        assert queue_service.start(queue_jobs[3])['queued']
        queue_worker.active=None
        assert queue_service.store.detail(queue_jobs[3])['queue_position']==1
    finally:
        release.set();queue_service.close();queue_worker.close()
    restarted_worker=ConsoleRunner(queue_worker.root,LocalCalls)
    restarted_queue=BatchService(restarted_worker,queue_prepare,lambda _:queued_catalog,lambda _:None)
    try:
        assert restarted_queue.store.queue()[0]['job_id']==queue_jobs[3]
        # A changed queued method requires explicit confirmation, not silent execution.
        with restarted_worker.store.connect() as db:
            row=restarted_queue.store.queue()[0]
            intent={**row['intent'],'method_sha256':'old-method'}
            db.execute('UPDATE batch_events SET payload=? WHERE seq=?',(json.dumps(intent),row['seq']))
        restarted_queue.recover_running()
        assert restarted_queue.store.detail(queue_jobs[3])['status']=='paused'
        assert prepared_order==[1,1,2,3]
        restarted_queue.start(queue_jobs[3]);queue_wait(lambda:restarted_queue.store.detail(queue_jobs[3])['status']=='completed')
        assert prepared_order==[1,1,2,3,4]
    finally:
        restarted_queue.close();restarted_worker.close()

    # A preparation failure is an attempted annotation even without a model run.
    # Readiness and stale task dispatch must agree; unused/prepare-only rows do
    # not become attempted targets, and explicit reopening keeps the zero bill.
    import app.routers.system as system_routes
    from app.datasets.preparation import EvidenceRejected
    preparation_packets={i:packet().model_copy(update={'packet_id':ident(19000+i),'context':f'preparation-attempt-{i}'}) for i in (1,2)}
    preparation_catalog={'catalog_id':ident(19000),'sources':{},'summary':{'selected':2},'targets':[
        {'target_id':ident(19100+i),'ordinal':i,'context':p.context,'object_id':p.target_object_id,
            'timestamp':p.frame_timestamp_micros,'category':p.target.category,'status':'selected','frame_exclusion_counts':{}}
        for i,p in preparation_packets.items()]}
    prepared=[]
    def rejected_preparation(target,*args):
        prepared.append(target['ordinal'])
        if target['ordinal']==1:raise ValueError('test preparation failure')
        raise EvidenceRejected(preparation_packets[2])
    preparation_worker=ConsoleRunner(worker.root/'prep-pool',LocalCalls)
    preparation_service=BatchService(preparation_worker,rejected_preparation,lambda _:preparation_catalog,lambda _:None)
    import app.agent.batch as preparation_api
    previous_loader=preparation_api.load_packet
    monkeypatch.setattr(preparation_api,'load_packet',lambda pid:next(
        (p for p in preparation_packets.values() if p.packet_id==pid),None) or previous_loader(pid))
    from app.infrastructure.candidate_pool import reopen_failed
    try:
        first,_=preparation_service.create(BatchSpec(catalog_id=preparation_catalog['catalog_id'],prepare_only=False,desired_count=1))
        assert not preparation_worker.store.preparation_attempted_keys()
        # The original task may recover an interrupted preparation; only other
        # tasks must treat that already-started target as unavailable.
        with preparation_worker.store.connect() as db:
            db.execute("UPDATE batch_items SET status='preparing' WHERE job_id=? AND ordinal=1",(first,))
        assert preparation_worker.store.preparation_attempted_keys()
        assert not preparation_worker.store.preparation_attempted_keys(exclude_job=first)
        preparation_service.start(first);wait(preparation_worker)
        original=preparation_service.store.detail(first)
        assert {i['status'] for i in original['items']}=={'failed','abstained'}
        assert all(not i['run_id'] for i in original['items']) and original['model_calls_reserved']==0
        expected={(p.context,p.target_object_id) for p in preparation_packets.values()}
        assert preparation_worker.store.preparation_attempted_keys()==expected
        with monkeypatch.context() as readiness:
            readiness.setattr(system_routes,'load_catalog',lambda **kwargs:preparation_catalog)
            request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(console=preparation_worker)))
            value=system_routes.status(request)
            assert set(value['replacement_unavailable_target_ids'])=={t['target_id'] for t in preparation_catalog['targets']}
        # A client using its old candidate count cannot silently prepare again.
        stale,_=preparation_service.create(BatchSpec(catalog_id=preparation_catalog['catalog_id'],prepare_only=False,desired_count=2))
        preparation_service.start(stale);wait(preparation_worker)
        assert sorted(prepared)==[1,2]
        assert all(i['reason']=='target_already_attempted' for i in preparation_service.store.detail(stale)['items'])
        assert preparation_service.store.detail(stale)['model_calls_reserved']==0
        preparation_only,_=preparation_service.create(BatchSpec(catalog_id=preparation_catalog['catalog_id'],prepare_only=True))
        with preparation_worker.store.connect() as db:
            db.execute("UPDATE batch_items SET status='failed',target=json_set(target,'$.context','prepare-only') WHERE job_id=?",(preparation_only,))
        assert preparation_worker.store.preparation_attempted_keys()==expected
        assert reopen_failed(preparation_worker.store,first)['reopened']==2
        with monkeypatch.context() as readiness:
            readiness.setattr(system_routes,'load_catalog',lambda **kwargs:preparation_catalog)
            assert system_routes.status(request)['replacement_unavailable_target_ids']==[]
        assert preparation_service.store.detail(first)['model_calls_reserved']==0
    finally:
        preparation_service.close();preparation_worker.close()
        monkeypatch.setattr(preparation_api,'load_packet',previous_loader)

    # Reopening is an explicit, one-round authorization, never a quota reset.
    # A failed generated draft can re-enter; successful and unresolved records
    # retain their snapshots. The allowance follows the track into another task.
    from app.infrastructure.candidate_pool import reopen_failed
    retry_packets={i:packet().model_copy(update={'packet_id':str(i-1)*24,'context':f'retry-{i}'}) for i in (1,2,3)}
    retry_catalog={'catalog_id':'e'*24,'sources':{},'targets':[
        {'target_id':str(i)*24,'ordinal':i,'context':p.context,'object_id':p.target_object_id,
            'timestamp':p.frame_timestamp_micros,'category':p.target.category,'status':'selected','frame_exclusion_counts':{}}
        for i,p in retry_packets.items()]}
    retry_worker=ConsoleRunner(worker.root/'retry-pool',LocalCalls)
    retry_service=BatchService(retry_worker,lambda t,*args:retry_packets[t['ordinal']],lambda _:retry_catalog,lambda _:None)
    try:
        retry_job,_=retry_service.create(BatchSpec(catalog_id=retry_catalog['catalog_id'],target_ids=None,
            prepare_only=False,max_model_calls=None))
        outcomes(monkeypatch,retry_worker,fail_ordinals={1,3},failure_calls=10)
        retry_service.start(retry_job);wait(retry_worker)
        before=retry_service.store.detail(retry_job)
        linked={i['ordinal']:i['run_id'] for i in before['items']}
        unknown=retry_worker.store.snapshot(linked[3]);unknown['stop_reason']='interrupted_response_outcome_unknown'
        with retry_worker.store.connect() as db:
            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(unknown),linked[3]))
            frozen={r['run_id']:r['snapshot'] for r in db.execute('SELECT run_id,snapshot FROM runs')}
        assert reopen_failed(retry_worker.store,retry_job,dry_run=True)['skipped']['insufficient_budget']==1
        preview=reopen_failed(retry_worker.store,retry_job,additional_calls=12,dry_run=True)
        assert preview['reopened']==1 and preview['skipped']['unknown_outcome']==1
        receipt=reopen_failed(retry_worker.store,retry_job,additional_calls=12)
        assert receipt['reopened']==1 and not receipt['started']
        assert reopen_failed(retry_worker.store,retry_job,additional_calls=12)['reopened']==0
        reopened=retry_service.store.detail(retry_job)
        first=next(i for i in reopened['items'] if i['ordinal']==1)
        assert first['status']=='pending' and first['run_id'] is None
        assert first['target']['retry_authorization']['call_limit']==22
        assert reopened['model_calls_reserved']==before['model_calls_reserved']==24
        assert reopened['status']==before['status']=='completed'
        with retry_worker.store.connect() as db:
            assert {r['run_id']:r['snapshot'] for r in db.execute('SELECT run_id,snapshot FROM runs')}==frozen
        # Actual target re-dispatch uses the approved cumulative ceiling and
        # atomically consumes every copy of this authorization exactly once.
        child_job,_=retry_service.create(BatchSpec(catalog_id=retry_catalog['catalog_id'],
            target_ids=[retry_catalog['targets'][0]['target_id']],prepare_only=False,max_model_calls=None))
        outcomes(monkeypatch,retry_worker)
        retry_service.start(child_job);wait(retry_worker)
        child=retry_service.store.detail(child_job)['items'][0]
        state=retry_worker.store.snapshot(child['run_id'])
        assert state['diagnostic_pass'] and state['cumulative_call_limit']==22
        assert state['parent_run_id']==linked[1] and state['model_call_budget']==12
        assert retry_worker.store.target_usage_for('retry-1',retry_packets[1].target_object_id,LocalCalls.scope)==14
        assert not retry_worker.store.candidate_retries()
        from app.agent.selection import unavailable_target_keys
        example={'provider_scope':'real-model','status':'needs_evidence','diagnostic_pass':False,
            'version':'current','current_packet':{'context':'c','target_object_id':'o'},
            'final_expression':'Failed draft','model_calls_used':10}
        grants={('c','o'):{'call_limit':22}}
        assert ('c','o') in unavailable_target_keys([],'current',preparation_attempts={('c','o')})[0]
        assert ('c','o') not in unavailable_target_keys([],'current',preparation_attempts={('c','o')},retries=grants)[0]
        assert ('c','o') in unavailable_target_keys([example],'current',include_prior_attempts=True)[0]
        assert ('c','o') not in unavailable_target_keys([example],'current',include_prior_attempts=True,retries=grants)[0]
        assert ('c','o') in unavailable_target_keys([{**example,'status':'calibration_hold','diagnostic_pass':True}],
            'current',include_prior_attempts=True,retries=grants)[0]
    finally:
        retry_service.close();retry_worker.close()


def test_training_uses_full_scene_and_detects_changed_assets(scene, monkeypatch):
    folder, _, _ = scene
    from scripts.convert_pointcloud import read_lidar_frame
    import threading
    barrier=threading.Barrier(3)
    owner=threading.get_ident();threads=set()
    originals={name:[{'sensor':n,'values':[n,0.25,-1]} for n in (5,1,4,2,3)]
        for name in ('lidar','lidar_pose','lidar_camera_projection')}
    class ParallelFrameReader:
        def path(self,name,context):
            assert threading.get_ident()==owner and context=='context'
            return name
        def _frame_rows(self,path,context,timestamp,columns,camera):
            assert (context,timestamp,columns,camera)==('context',10,None,None)
            threads.add(threading.get_ident());barrier.wait(timeout=5)
            return originals[path]
    assert read_lidar_frame(ParallelFrameReader(),'context',10)==originals
    assert len(threads)==3 and owner not in threads
    class BrokenFrameReader(ParallelFrameReader):
        def _frame_rows(self,path,*args):
            if path=='lidar_pose':raise OSError('input read failed')
            return originals[path]
    with pytest.raises(OSError,match='input read failed'):
        read_lidar_frame(BrokenFrameReader(),'context',10)
    from app.infrastructure.artifacts import hash_assets
    # The bounded parallel path verifies every asset and still rejects a late
    # corrupt/missing file. There is no process-lifetime integrity cache.
    bulk = folder/'evidence-hashes'
    bulk.mkdir()
    for number in range(12):
        (bulk/f'{number}.jpg').write_bytes(f'image-{number}'.encode())
    expected = {p.name:file_sha256(p) for p in bulk.iterdir()}
    assert hash_assets(bulk, expected, expected=expected) == expected
    (bulk/'11.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        hash_assets(bulk, expected, expected=expected)
    (bulk/'11.jpg').unlink()
    with pytest.raises(ValueError, match='Asset missing'):
        hash_assets(bulk, expected, expected=expected)
    with pytest.raises(ValueError):
        hash_assets(bulk, ['../scene.json'])
    inputs = training_inputs(folder, TaskProfile(), 'The vehicle ahead.')
    assert set(inputs) == {'points', 'point_features', 'expression', 'coordinate_frame'}
    assert inputs['points'].shape == (3, 6)
    rgb = training_inputs(folder, TaskProfile(input_capabilities=('geometry', 'rgb')), 'The red vehicle.')
    assert len(rgb['images']) == 5 and rgb['point_camera_projection'].shape == (3, 6)
    with monkeypatch.context() as patch:
        def unexpected_hash(*args, **kwargs):
            raise AssertionError('Training must not hash scene files')
        patch.setattr('app.infrastructure.artifacts.file_sha256', unexpected_hash)
        assert len(training_inputs(folder, TaskProfile(input_capabilities=('geometry', 'rgb')), 'The red vehicle.')['images']) == 5
    for invalid in ('../outside.npz', '/outside.npz', 'C:/outside.npz'):
        from app.infrastructure.artifacts import asset_path
        with pytest.raises(ValueError):
            asset_path(folder, invalid)
    # A complete ZIP must work after extraction without the project or raw Parquet.
    import copy
    import importlib.util
    import zipfile
    from urllib.parse import unquote
    import pyarrow as pa
    import pyarrow.parquet as pq
    from app.datasets.dataset_export import build_dataset, rejection, download_chunks
    from app.datasets.waymo import B, C, T, O, WaymoReader
    from app.datasets.waymo import ContextReaders, V
    from app.datasets.data import COMPONENTS, inspect_dataset
    split_fixture = folder.parent/'official-split-inputs'
    for split, name, context in [('training','waymo_trainset','train-context'),('validation','waymo_valset','val-context')]:
        for component in COMPONENTS:
            component_root = split_fixture/name/component
            component_root.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{C:context,T:1}]),component_root/f'{split}_{component}_{context}.parquet')
    inventory = inspect_dataset(split_fixture/'waymo_trainset')
    assert not inventory.errors and inventory.file_count==26 and inventory.component_count==13
    assert {item.official_split for item in inventory.files}=={'training','validation'}
    split_reader = WaymoReader(split_fixture/'waymo_trainset')
    assert split_reader.contexts()==['train-context','val-context']
    assert split_reader.official_split('val-context')=='validation'
    conflicting_path = split_fixture/'waymo_valset/lidar_box/validation_lidar_box_train-context.parquet'
    pq.write_table(pa.Table.from_pylist([{C:'train-context',T:1}]),conflicting_path)
    with pytest.raises(ValueError,match='multiple official splits'):split_reader.contexts()
    assert any('official_split_conflict' in error for error in inspect_dataset(split_fixture/'waymo_trainset').errors)
    conflicting_path.unlink()
    wrong_directory = split_fixture/'waymo_trainset/lidar_box/validation_lidar_box_wrong.parquet'
    pq.write_table(pa.Table.from_pylist([{C:'wrong',T:1}]),wrong_directory)
    with pytest.raises(ValueError,match='Wrong official split'):split_reader.contexts()
    wrong_directory.unlink()
    # Evicting heavy context metadata must retain only small file fingerprints;
    # changed sources still invalidate the new and previously retained readers.
    fingerprint_root = folder.parent/'fingerprints'
    (fingerprint_root/'lidar_box').mkdir(parents=True)
    for number in range(3):
        pq.write_table(pa.Table.from_pylist([{C:f'ctx-{number}', T:1, 'value':number}]),
            fingerprint_root/'lidar_box'/f'training_lidar_box_ctx-{number}.parquet')
    pool = ContextReaders(fingerprint_root)
    original_hash = file_sha256
    hashes = []
    with monkeypatch.context() as fingerprints:
        def traced_hash(path):
            hashes.append(path)
            return original_hash(path)
        fingerprints.setattr('app.datasets.waymo.file_sha256', traced_hash)
        first_reader = pool.get('ctx-0')
        initial = first_reader.provenance('lidar_box','ctx-0')
        first_reader.metadata_rows('lidar_box','ctx-0')
        for number in (1,2):
            pool.get(f'ctx-{number}').provenance('lidar_box',f'ctx-{number}')
        replacement = pool.get('ctx-0')
        assert replacement is not first_reader and len(pool.readers) == 2
        assert replacement.provenance('lidar_box','ctx-0') == initial and len(hashes) == 3
        cached_rows = replacement.metadata_rows('lidar_box','ctx-0')
        narrowed = replacement.provenance('lidar_box','ctx-0', columns=[C])
        narrowed['columns'].append('external mutation')
        assert replacement.provenance('lidar_box','ctx-0') == initial
        path = replacement.path('lidar_box','ctx-0')
        pq.write_table(pa.Table.from_pylist([{C:'ctx-0', T:1, 'value':999}]), path)
        changed = replacement.provenance('lidar_box','ctx-0')
        assert changed['sha256'] != initial['sha256'] and len(hashes) == 4
        assert replacement.metadata_rows('lidar_box','ctx-0')[0]['value'] == 999
        assert cached_rows[0]['value'] == 0
        assert first_reader.provenance('lidar_box','ctx-0') == changed and len(hashes) == 4
        assert first_reader.metadata_rows('lidar_box','ctx-0')[0]['value'] == 999
        # Distinct preparation threads share only fingerprints. Same-file cold
        # requests hash once, but a real source change invalidates both readers.
        import threading
        from concurrent.futures import ThreadPoolExecutor
        shared_sources, source_lock = {}, threading.Lock()
        independent = [ContextReaders(fingerprint_root,source_cache=shared_sources,
            source_lock=source_lock).get('ctx-0') for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as fingerprint_executor:
            copies = list(fingerprint_executor.map(lambda r:r.provenance('lidar_box','ctx-0'),independent))
        assert copies == [changed,changed] and len(hashes) == 5
        assert all(r.metadata_rows('lidar_box','ctx-0')[0]['value'] == 999 for r in independent)
        pq.write_table(pa.Table.from_pylist([{C:'ctx-0',T:1,'value':1000}]),path)
        with ThreadPoolExecutor(max_workers=2) as fingerprint_executor:
            updated = list(fingerprint_executor.map(lambda r:r.provenance('lidar_box','ctx-0'),independent))
        assert updated[0] == updated[1] and updated[0]['sha256'] != changed['sha256'] and len(hashes) == 6
        assert all(r.metadata_rows('lidar_box','ctx-0')[0]['value'] == 1000 for r in independent)
    # Small Arrow batches must retain keys, ordering, selected columns and exact
    # frame/camera filtering across row-group and batch boundaries.
    (fingerprint_root/'camera_image').mkdir()
    frame_rows = [{C:'ctx-0',T:index//5,V:index%5+1,'payload':[index,index+1]}
        for index in range(45)]
    pq.write_table(pa.Table.from_pylist(frame_rows),
        fingerprint_root/'camera_image'/'training_camera_image_ctx-0.parquet', row_group_size=17)
    for timestamp in (0,3,6,8,100):
        for camera in (None,1,5):
            expected = [r for r in frame_rows if r[T]==timestamp and (camera is None or r[V]==camera)]
            assert replacement.rows('camera_image','ctx-0',timestamp=timestamp,camera=camera) == expected
            assert replacement.rows('camera_image','ctx-0',timestamp=timestamp,camera=camera,columns=['payload']) == (
                [{'payload':r['payload']} for r in expected])
    # Five camera filters share one frame decode; callers cannot mutate the
    # cache, and file changes / frame switches invalidate it immediately.
    with monkeypatch.context() as image_cache:
        decodes=[]
        original_frame_rows=WaymoReader._frame_rows
        def count_frame_rows(*args):
            decodes.append(args[2])
            return original_frame_rows(*args)
        image_cache.setattr(WaymoReader,'_frame_rows',staticmethod(count_frame_rows))
        for camera in range(1,6):
            replacement.rows('camera_image','ctx-0',timestamp=0,camera=camera,columns=['payload'])
        assert decodes==[0]
        result=replacement.rows('camera_image','ctx-0',timestamp=0,camera=1)
        result[0]['payload'][0]=-1
        assert replacement.rows('camera_image','ctx-0',timestamp=0,camera=1)[0]['payload'][0]==0
        with pytest.raises(ValueError,match='Unknown frame columns'):
            replacement.rows('camera_image','ctx-0',timestamp=0,columns=['missing'])
        image_path=replacement.path('camera_image','ctx-0')
        pq.write_table(pa.Table.from_pylist([{**r,'payload':[999]} for r in frame_rows]),image_path,row_group_size=17)
        assert replacement.rows('camera_image','ctx-0',timestamp=0,camera=1)[0]['payload']==[999]
        replacement.rows('camera_image','ctx-0',timestamp=1)
        assert decodes==[0,0,1]
    from app.routers.batches import export_dataset, export_selected_dataset
    root = folder.parent
    raw = root/'waymo_trainset/lidar_box'
    raw.mkdir(parents=True)
    objects = [obj('first-track', 0, 0, length=1), obj('second-track', 0, 1.5, length=1)]
    native = []
    for value in objects:
        row = {C:'context',T:1,O:value.object_id,f'{B}.type':1,f'{B}.box.heading':0,
            f'{B}.num_lidar_points_in_box':value.lidar_points}
        row.update({f'{B}.box.center.{axis}':value.center_m[i] for i,axis in enumerate('xyz')})
        row.update({f'{B}.box.size.{axis}':value.size_m[i] for i,axis in enumerate('xyz')})
        native.append(row)
    pq.write_table(pa.Table.from_pylist(native),raw/'training_lidar_box_context.parquet')
    manifest = json.loads((folder/'scene.json').read_text(encoding='utf-8'))
    manifest['conversion'] = {'implementation':'official-waymo-sdk'}
    # Frame pose differs from image pose: ego2cam must include this motion and
    # convert Waymo camera axes to OpenCV, not just invert the static extrinsic.
    from app.datasets.waymo import V
    frame_pose = np.array([[0,-1,0,10],[1,0,0,2],[0,0,1,3],[0,0,0,1]],dtype=float)
    image_pose = np.eye(4); image_pose[:3,3] = [8,2,3]
    calibration = {'camera_calibration':[], 'camera_frame_metadata':[], 'lidar_calibration':[],
        'frame_pose':{C:'context',T:1,'[VehiclePoseComponent].world_from_vehicle.transform':frame_pose.ravel().tolist()}}
    for camera in manifest['cameras']:
        camera.update(pose_timestamp_s=1.1,trigger_time_s=1.0,readout_done_time_s=1.2)
        extrinsic = np.eye(4); extrinsic[:3,3] = [camera['camera_name'],0,1]
        row = {C:'context',V:camera['camera_name'],'[CameraCalibrationComponent].width':8,
            '[CameraCalibrationComponent].height':6,'[CameraCalibrationComponent].extrinsic.transform':extrinsic.ravel().tolist(),
            '[CameraCalibrationComponent].rolling_shutter_direction':1}
        row.update({f'[CameraCalibrationComponent].intrinsic.{key}':value
            for key,value in zip(('f_u','f_v','c_u','c_v','k1','k2','p1','p2','k3'),(8,9,4,3,.1,.2,.01,.02,.3))})
        calibration['camera_calibration'].append(row)
        calibration['camera_frame_metadata'].append({C:'context',T:1,V:camera['camera_name'],
            '[CameraImageComponent].pose.transform':image_pose.ravel().tolist(),
            '[CameraImageComponent].pose_timestamp':1.1,
            '[CameraImageComponent].rolling_shutter_params.camera_trigger_time':1.0,
            '[CameraImageComponent].rolling_shutter_params.camera_readout_done_time':1.2,
            '[CameraImageComponent].rolling_shutter_params.shutter':.01})
    atomic_json(folder/'calibration.json',calibration)
    manifest['calibration']['sha256'] = file_sha256(folder/'calibration.json')
    atomic_json(folder/'scene.json',manifest)
    p = packet()
    p.context, p.frame_timestamp_micros = 'context', 1
    p.candidates = objects
    p.target, p.target_object_id = objects[0], objects[0].object_id
    p.scene = SceneEvidence(scene_id=manifest['scene_id'],manifest_path='scene/scene.json',
        manifest_sha256='a'*64,total_points=3,target_points=1)
    p.provenance = {'components':{'lidar_box':WaymoReader(root/'waymo_trainset').provenance('lidar_box','context')}}
    states = {}
    def saved(n, target=0, **updates):
        snapshot = p.model_copy(deep=True)
        snapshot.target = objects[target]
        snapshot.target_object_id = objects[target].object_id
        snapshot.source_view = 'FRONT' if target == 0 else 'SIDE_LEFT'
        state = {'run_id':f'{n:024x}','packet_id':p.packet_id,'status':'calibration_hold',
            'provider_scope':'real-model','diagnostic_pass':True,'created_at':n,
            'final_expression':'The vehicle ahead.','inflight':None,'current_packet':snapshot.model_dump(mode='json'),
            'attempts':[{'generated':{'expression':'The vehicle ahead.'},
                         'ground_checks':[{'diagnostic_pass':True},{'diagnostic_pass':True}]}]}
        state.update(updates); states[state['run_id']] = state
        return state
    old = saved(1)
    newest = saved(2)
    second = saved(3,1)
    failed = saved(4,status='needs_evidence')
    empty = saved(5,final_expression='  ')
    incomplete = saved(6,attempts=[])
    mock = saved(7,provider_scope='local-contract-test')
    assert [rejection(s) for s in (failed,empty,incomplete,mock)] == [
        'not_successful','no_final_text','validation_incomplete','not_real_model']
    assert rejection({**newest,'inflight':{'pending':True}}) == 'not_successful'
    bad_appearance = copy.deepcopy(newest)
    bad_appearance['attempts'][0]['generated']['visual_attributes'] = [{'kind':'color','value':'red'}]
    assert rejection(bad_appearance) == 'appearance_not_verified'
    runs = RunStore(root/'export-runs/runs.sqlite3')
    jobs = {}
    for number, runlist in enumerate(([old,newest,second,failed,empty,incomplete,mock],[newest]),1):
        jid = f'{number+20:024x}'
        items = []
        for ordinal,state in enumerate(runlist,1):
            target_id = state['current_packet']['target_object_id']
            items.append({'ordinal':ordinal,'run_id':state['run_id'],'packet_id':p.packet_id,
                'expression':state['final_expression'],'target':{'context':'context','timestamp':1,'object_id':target_id}})
        jobs[jid] = {'job_id':jid,'items':items,'status':'completed','executor_active':False}
    with runs.connect() as db:
        db.execute('CREATE TABLE batches(job_id TEXT,created_at REAL)')
        db.executemany('INSERT INTO batches VALUES(?,?)',[(j,i) for i,j in enumerate(jobs)])
        db.executemany('INSERT INTO runs(run_id,snapshot) VALUES(?,?)',[(r,json.dumps(s)) for r,s in states.items()])
    service = SimpleNamespace(runner=SimpleNamespace(store=runs,root=root/'runtime/annotation'),
        store=SimpleNamespace(detail=lambda job:jobs[job],summaries=lambda:list(jobs.values()),runs=runs))
    # Export uses the accepted preparation frame, not the failed catalog seed.
    # A frozen alternate alone is insufficient: its preparation receipt is required.
    owner = next(iter(jobs.values()))
    prepared_item = owner['items'][1]
    prepared_item['target'].update(timestamp=2,alternative_selections=[{'timestamp':1}])
    with pytest.raises(ValueError,match='物体 #2.*身份不一致'):
        build_dataset(service,root)
    owner['events'] = [{'kind':'preparation_source_reselected','payload':{
        'ordinal':2,'packet_id':p.packet_id,'timestamp':1,
        'rejected_frames':[{'timestamp':2,'reason':'converted_target_points_missing'}]}}]
    # An unused camera asset is neither needed for export nor packaged.
    (folder/'camera-3.jpg').unlink()
    with build_dataset(service,root) as stream:
        with zipfile.ZipFile(stream) as archive:
            assert archive.testzip() is None
            assert archive.namelist().count(f"scenes/{p.scene.scene_id}/points.npz") == 1
            assert not any('sha256' in key for key in json.loads(archive.read('manifest.json')))
            package = json.loads(archive.read('manifest.json'))
            assert package['records']==2 and package['scenes']==1
            for name in archive.namelist():
                if name.endswith(('.json','.jsonl')):
                    assert b'"schema_version"' not in archive.read(name)
            assert not {'quality','source_job_ids','excluded','task_count'} & set(package)
            assert sorted(name for name in archive.namelist() if name.endswith('.jpg')) == [
                f'scenes/{p.scene.scene_id}/camera-1.jpg',f'scenes/{p.scene.scene_id}/camera-4.jpg']
            camera_data = json.loads(archive.read(f'scenes/{p.scene.scene_id}/calibration.json'))
            assert set(camera_data['camera_models']) == {'1','4'}
            assert {row[V] for row in camera_data['camera_calibration']} == {1,4}
            assert {row[V] for row in camera_data['camera_frame_metadata']} == {1,4}
            extracted = root/'cloud-copy'
            archive.extractall(extracted)
    module_spec = importlib.util.spec_from_file_location('portable_grounding_loader',extracted/'loader.py')
    loader = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(loader)
    rows = list(loader.iter_samples(extracted))
    first = next(r for r in rows if r['track_id']=='first-track')
    assert first['sample_id']==newest['run_id']
    private_fields = {'quality','method_version','annotation_run_id','evidence_packet_id','job_ids','task'}
    assert all(not private_fields & set(r) for r in rows)
    assert package['heading_convention']==TaskProfile().heading_convention
    for record in rows:
        model_inputs, supervision, metadata = loader.load_sample(extracted,record)
        assert model_inputs['points'].shape==(3,6) and 'images' not in model_inputs
        expected_camera = 1 if record['track_id']=='first-track' else 4
        assert model_inputs['camera_id']==record['camera_id']==expected_camera
        assert model_inputs['image']==extracted/record['image_path']
        assert model_inputs['image'].name==f'camera-{expected_camera}.jpg'
        params = model_inputs['calibration']
        assert params['camera_id']==expected_camera and params['source_view']==record['source_view']
        assert params['K']==[[8,0,4],[0,9,3],[0,0,1]] and params['distortion']==[.1,.2,.01,.02,.3]
        if expected_camera == 1:
            assert np.allclose(np.asarray(params['ego2cam']) @ [2,-3,4,1],[-2,-3,4,1])
        assert all(key in model_inputs for key in ('laser_id','return_id','nlz','camera_projection'))
        assert 'target_box' not in model_inputs and 'scene_objects' not in model_inputs
        assert supervision['target_box'].shape==(7,) and len(supervision['scene_objects'])==2
        assert metadata['track_id']==record['track_id']
        assert not private_fields & set(metadata)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(batches=service,sample_root=root)),headers={},url=SimpleNamespace(hostname="127.0.0.1"))
    for split in ('train',):
        response = export_dataset(request,split)
        assert f'waymo-3dref-{split}.zip' in response.headers['content-disposition']
        # The temporary stream is owned by the response and cleaned on completion.
        response.background.func()
    # Portable shards merge into one root, including different cameras of one scene.
    from app.datasets.export_merge import merge_dataset
    import subprocess
    import sys
    parts = root/'parts'; parts.mkdir()
    shard_paths = []
    for index, record in enumerate(rows,1):
        shard = parts/f'part{index}'
        shutil.copytree(extracted,shard)
        (shard/'samples.jsonl').write_text(json.dumps(record)+'\n',encoding='utf-8')
        info = json.loads((shard/'manifest.json').read_text(encoding='utf-8'))
        info.update(records=1,intended_split='train')
        atomic_json(shard/'manifest.json',info)
        scene_file = shard/record['scene']
        one_scene = json.loads(scene_file.read_text(encoding='utf-8'))
        one_scene['cameras'] = [c for c in one_scene['cameras'] if c['camera_name']==record['camera_id']]
        atomic_json(scene_file,one_scene)
        data_file = scene_file.parent/'calibration.json'
        data = json.loads(data_file.read_text(encoding='utf-8'))
        for key in ('camera_calibration','camera_frame_metadata'):
            data[key] = [r for r in data[key] if r['key.camera_name']==record['camera_id']]
        data['camera_models'] = {str(record['camera_id']):data['camera_models'][str(record['camera_id'])]}
        atomic_json(data_file,data)
        for image in scene_file.parent.glob('camera-*.jpg'):
            if image.name!=f"camera-{record['camera_id']}.jpg":image.unlink()
        zipped = parts/f'trainset{index}.zip'
        with zipfile.ZipFile(zipped,'w') as archive:
            for file in shard.rglob('*'):
                if file.is_file():archive.write(file,file.relative_to(shard).as_posix())
        shard_paths.append(zipped)
    combined = root/'combined/trainset'
    report = merge_dataset(shard_paths,combined)
    assert report['dataset_name']=='Waymo-3DRef' and report['dataset_split']=='train'
    assert report['records']==2 and report['scenes']==1
    assert report['duplicates_removed_this_merge']==0
    assert 'schema_version' not in report
    assert not {'quality','source_job_ids','task_count'} & set(report)
    merged_scene = json.loads((combined/first['scene']).read_text(encoding='utf-8'))
    assert {c['camera_name'] for c in merged_scene['cameras']}=={1,4}
    for row in loader.iter_samples(combined):
        loader.load_sample(combined,row)
        assert not private_fields & set(row)
    from app.datasets.export_merge import merge_record
    with pytest.raises(ValueError,match='Unexpected fields'):
        merge_record({}, {}, {**rows[0], 'quality':{'diagnostic_pass':False}})
    assert merge_dataset(shard_paths[:1],combined)['records']==2
    baseline = (combined/'samples.jsonl').read_bytes()
    before_manifest = (combined/'manifest.json').read_bytes()
    conflicting = parts/'conflicting'
    shutil.copytree(parts/'part1',conflicting)
    changed = json.loads((conflicting/'samples.jsonl').read_text(encoding='utf-8'))
    changed['expression']='A conflicting expression.'
    (conflicting/'samples.jsonl').write_text(json.dumps(changed)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Conflicting annotation'):merge_dataset([conflicting],combined)
    assert (combined/'samples.jsonl').read_bytes()==baseline
    changed = dict(rows[0]);changed['sample_id']='f'*24
    (conflicting/'samples.jsonl').write_text(json.dumps(changed)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Different annotation versions'):merge_dataset([conflicting],combined)
    (conflicting/'samples.jsonl').write_text(json.dumps(rows[0])+'\n',encoding='utf-8')
    Image.new('RGB',(8,6),'green').save(conflicting/rows[0]['image_path'])
    with pytest.raises(ValueError,match='Conflicting scene asset'):merge_dataset([conflicting],combined)
    validation = parts/'validation'
    shutil.copytree(parts/'part1',validation)
    val_manifest = json.loads((validation/'manifest.json').read_text(encoding='utf-8'))
    val_manifest['intended_split']='validation';atomic_json(validation/'manifest.json',val_manifest)
    with pytest.raises(ValueError,match='split differs'):merge_dataset(shard_paths,combined.parent/'valset')
    with pytest.raises(ValueError,match='context overlap'):merge_dataset([validation],combined.parent/'valset')
    assert not (combined.parent/'valset').exists()
    assert (combined/'manifest.json').read_bytes()==before_manifest
    original_replace = Path.replace
    with monkeypatch.context() as failure:
        def failed_commit(path,target):
            if path.name=='dataset':raise OSError('simulated commit failure')
            return original_replace(path,target)
        failure.setattr(Path,'replace',failed_commit)
        with pytest.raises(OSError,match='simulated commit'):merge_dataset(shard_paths[:1],combined)
    assert (combined/'manifest.json').read_bytes()==before_manifest
    assert (combined/'samples.jsonl').read_bytes()==baseline
    unsafe = parts/'unsafe.zip'
    with zipfile.ZipFile(unsafe,'w') as archive:archive.writestr('../outside','bad')
    with pytest.raises(ValueError,match='Unsafe ZIP'):merge_dataset([unsafe],combined)
    check = subprocess.run([sys.executable,'-B',str(combined/'merge.py'),'--output',str(combined),str(shard_paths[0])],
        capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stderr
    assert 'Waymo-3DRef: 2 samples' in check.stdout
    independent = root/'independent/trainset'
    merge_dataset([parts/'part1'],independent)
    retained_image = (independent/rows[0]['image_path']).read_bytes()
    Image.new('RGB',(8,6),'purple').save(parts/'part1'/rows[0]['image_path'])
    assert (independent/rows[0]['image_path']).read_bytes()==retained_image
    unfinished_job, completed_job = list(jobs)
    for status, active in [('ready',False),('running',True),('preparing',True),('paused',False),('stopped',False),('blocked',False),('completed',True)]:
        jobs[unfinished_job].update(status=status,executor_active=active)
        with pytest.raises(ValueError,match='尚未完成或仍在执行'):
            build_dataset(service,root,[unfinished_job,completed_job])
    jobs[unfinished_job].update(status='running',executor_active=True)
    with build_dataset(service,root) as completed_stream:
        with zipfile.ZipFile(completed_stream) as archive:
            completed_manifest = json.loads(archive.read('manifest.json'))
            assert completed_manifest['records']==1
            assert len(archive.read('samples.jsonl').splitlines())==1
    jobs[completed_job].update(status='running',executor_active=True)
    with pytest.raises(ValueError,match='暂无已完成'):build_dataset(service,root)
    for job in jobs.values():job.update(status='completed',executor_active=False)
    selected_job = list(jobs)[1]
    with build_dataset(service,root,[selected_job,selected_job]) as selected_stream:
        with zipfile.ZipFile(selected_stream) as archive:
            scope = json.loads(archive.read('manifest.json'))
            chosen_rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
            assert scope['records']==1
            assert len(chosen_rows)==1 and chosen_rows[0]['track_id']=='first-track'
            assert chosen_rows[0]['sample_id']==newest['run_id']
            assert len([name for name in archive.namelist() if name.endswith('.jpg')])==1
            assert set(json.loads(archive.read(f'scenes/{p.scene.scene_id}/calibration.json'))['camera_models'])=={'1'}
    for ids, message in (([], '请选择'), (['f'*24], '不存在')):
        with pytest.raises(ValueError,match=message):build_dataset(service,root,ids)
    from fastapi import HTTPException
    with pytest.raises(HTTPException,match='请先选择'):export_selected_dataset(request)
    with pytest.raises(HTTPException,match='范围冲突'):export_selected_dataset(request,export_scope='all',job_ids=[selected_job])
    response = export_selected_dataset(request,job_ids=[selected_job],filename='trainset2')
    assert "filename*=UTF-8''trainset2.zip" in response.headers['content-disposition']
    response.background.func()
    response = export_dataset(request,'train',[selected_job],'trainset1')
    assert "filename*=UTF-8''trainset1.zip" in response.headers['content-disposition']
    response.background.func()
    with pytest.raises(HTTPException,match='导出类型与所选任务不一致'):
        export_dataset(request,'validation',[selected_job],'验证集2')
    stream = build_dataset(service,root)
    chunks = download_chunks(stream)
    assert next(chunks).startswith(b'PK'); chunks.close(); assert stream.closed
    # The asynchronous UI wraps the same export contract. Cancellation must
    # release both the maintenance lock and every owned temporary ZIP.
    from app.datasets.export_jobs import ExportJobs
    from app.datasets import dataset_export as export_code
    from app.routers.batches import router as batch_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import threading
    import time
    exports = ExportJobs(service,root)
    api = FastAPI()
    api.include_router(batch_router)
    api.state.exports = exports
    def settled(export_id, expected):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = exports.status(export_id)
            if status['status'] in expected:return status
            time.sleep(.01)
        pytest.fail('Export did not settle: ' + str(exports.status(export_id)))
    owned = []
    original_temporary = export_code.tempfile.SpooledTemporaryFile
    def temporary(*args, **kwargs):
        file = original_temporary(*args, **kwargs)
        file.rollover()  # Exercise deletion of disk files, not just BytesIO.
        owned.append(file)
        return file
    try:
        with monkeypatch.context() as capture, TestClient(api) as client:
            capture.setattr(export_code.tempfile,'SpooledTemporaryFile',temporary)
            assert client.post('/api/batches/export/jobs',json={}).status_code==409
            assert client.post('/api/batches/export/jobs',json={'export_scope':'all','job_ids':[selected_job]}).status_code==409
            assert client.post('/api/batches/export/jobs',json={'export_scope':'all','filename':'../bad'}).status_code==422
            with monkeypatch.context() as mixed:
                mixed.setitem(jobs[selected_job],'spec',{'dataset_split':'validation'})
                denied = client.post('/api/batches/export/jobs',json={'export_scope':'all'})
                assert denied.status_code == 409 and '分开导出' in denied.json()['detail']
                assert not exports.jobs and not owned
                with pytest.raises(ValueError,match='分开导出'):build_dataset(service,root)
                assert not owned
            response = client.post('/api/batches/export/jobs',json={'export_scope':'all','filename':'训练集'})
            assert response.status_code==202
            export_id = response.json()['export_id']
            assert client.get('/api/batches/export/jobs/current').json()['export_id']==export_id
            assert client.post('/api/batches/export/jobs',json={'export_scope':'all'}).status_code==409
            ready = settled(export_id,{'ready'})
            assert ready['bytes_written']>0
            response = client.get(f'/api/batches/export/jobs/{export_id}/dataset.zip')
            assert response.status_code==200 and int(response.headers['content-length'])==len(response.content)
            assert unquote(response.headers['content-disposition'].split("filename*=UTF-8''")[1])=='训练集.zip'
            new_zip = parts/'asynchronous.zip';new_zip.write_bytes(response.content)
            with zipfile.ZipFile(new_zip) as archive:
                assert archive.testzip() is None
                for path in extracted.rglob('*'):
                    if path.name=='manifest.json':
                        before = json.loads(path.read_text(encoding='utf-8'));before['intended_split']='train'
                        assert json.loads(archive.read('manifest.json'))==before
                    elif path.is_file():assert archive.read(path.relative_to(extracted).as_posix())==path.read_bytes()
            # Foreground and background packaging have the same training data.
            compatible = root/'compatibility/trainset'
            assert merge_dataset([extracted,new_zip],compatible)['records']==2
            for row in loader.iter_samples(compatible):loader.load_sample(compatible,row)
            assert merge_dataset([new_zip],compatible)['records']==2
            with pytest.raises(ValueError,match='Conflicting scene asset'):
                merge_dataset([conflicting],compatible)
            assert client.get(f'/api/batches/export/jobs/{export_id}').json()['status']=='downloaded'
            assert all(file.closed for file in owned)
            # A ready ZIP is removed immediately without ever downloading it.
            export_id = exports.create(None,'train',None)['export_id']
            settled(export_id,{'ready'})
            assert client.post(f'/api/batches/export/jobs/{export_id}/cancel',headers={'Origin':'https://other.example'}).status_code==403
            assert client.post(f'/api/batches/export/jobs/{export_id}/cancel').json()['status']=='cancelled'
            assert client.get(f'/api/batches/export/jobs/{export_id}/dataset.zip').status_code==409
            assert all(file.closed for file in owned)
            # Waiting for a deletion/export lock must remain cancellable.
            with runs.maintenance():
                export_id = exports.create(None,'train',None)['export_id']
                exports.cancel(export_id)
                settled(export_id,{'cancelled'})
            assert all(file.closed for file in owned)
            # Stop within archive.write, rather than only between samples.
            entered, release = threading.Event(), threading.Event()
            original_progress = exports._check
            def pause_packaging(job):
                original_progress(job)
                if 'points.npz' in job['message'] and not entered.is_set():
                    entered.set()
                    assert release.wait(5)
                    original_progress(job)
            with monkeypatch.context() as pause:
                pause.setattr(exports,'_check',pause_packaging)
                export_id = exports.create(None,'train',None)['export_id']
                assert entered.wait(5)
                try:
                    assert exports.cancel(export_id)['status']=='cancelling'
                finally:release.set()
                settled(export_id,{'cancelled'})
            assert all(file.closed for file in owned)
            # Abort a transfer after its first chunk, then clean its stream.
            export_id = exports.create(None,'train',None)['export_id']
            settled(export_id,{'ready'})
            download, _ = exports.open_download(export_id)
            chunks = exports.download_chunks(export_id,download)
            assert next(chunks).startswith(b'PK')
            assert exports.cancel(export_id)['status']=='cancelling'
            assert list(chunks)==[] and exports.status(export_id)['status']=='cancelled'
            assert download.closed
            # A browser disconnect and an asset failure also clean partial ZIPs.
            export_id = exports.create(None,'train',None)['export_id']
            settled(export_id,{'ready'})
            download, _ = exports.open_download(export_id)
            chunks = exports.download_chunks(export_id,download)
            next(chunks);chunks.close()
            assert exports.status(export_id)['status']=='failed' and download.closed
            with monkeypatch.context() as broken:
                def missing_asset(*args, **kwargs):raise OSError('simulated missing asset')
                broken.setattr(export_code,'verified_native_objects',missing_asset)
                export_id = exports.create(None,'train',None)['export_id']
                assert 'simulated missing asset' in settled(export_id,{'failed'})['message']
            assert all(file.closed for file in owned)
            export_id = exports.create(None,'train',None)['export_id']
            settled(export_id,{'ready'})
            exports.jobs[export_id]['updated_at'] -= exports.retention_seconds + 1
            exports.expire()
            assert exports.status(export_id)['status']=='expired' and all(file.closed for file in owned)
            export_id = exports.create(None,'train',None)['export_id']
            settled(export_id,{'ready'})
            exports.open_download(export_id)  # Reserved response, body never iterated.
    finally:exports.close()
    assert all(file.closed for file in owned)
    assert all(not Path(file.name).exists() for file in owned if isinstance(file.name,str))
    # A real validation source uses its own component root and keeps that split
    # through native verification, the portable ZIP, loader and merge tool.
    training_native = raw/'training_lidar_box_context.parquet'
    validation_native = root/'waymo_valset/lidar_box/validation_lidar_box_context.parquet'
    validation_native.parent.mkdir(parents=True)
    scene_path = folder/'scene.json'
    original_scene = scene_path.read_bytes()
    validation_state = copy.deepcopy(newest)
    validation_state['current_packet']['provenance']['components']['lidar_box']['file'] = 'lidar_box/validation_lidar_box_context.parquet'
    try:
        training_native.rename(validation_native)
        changed_scene = json.loads(original_scene)
        changed_scene['official_split'] = 'validation'
        atomic_json(scene_path, changed_scene)
        with runs.connect() as db:
            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(validation_state),newest['run_id']))
        with monkeypatch.context() as validation_scope:
            validation_scope.setitem(jobs[selected_job],'spec',{'dataset_split':'validation'})
            reader = WaymoReader(root/'waymo_trainset')
            assert reader.contexts() == ['context'] and reader.official_split('context') == 'validation'
            assert reader.path('lidar_box','context') == validation_native
            validation_output = root/'validation-portable'
            with build_dataset(service,root,[selected_job],intended_split='validation') as stream:
                with zipfile.ZipFile(stream) as archive:
                    assert json.loads(archive.read('manifest.json'))['intended_split']=='validation'
                    archive.extractall(validation_output)
            validation_row = next(loader.iter_samples(validation_output))
            assert validation_row['official_split']=='validation'
            loader.load_sample(validation_output,validation_row)
            validation_merged = root/'validation-only/valset'
            assert merge_dataset([validation_output],validation_merged)['records']==1
            response = export_dataset(request,'validation',[selected_job])
            assert 'waymo-3dref-validation.zip' in response.headers['content-disposition']
            response.background.func()
            with pytest.raises(ValueError,match='导出类型与所选任务不一致'):
                build_dataset(service,root,[selected_job],intended_split='train')
            # A task label cannot turn official validation pixels into training data.
            validation_scope.setitem(jobs[selected_job],'spec',{'dataset_split':'train'})
            with pytest.raises(ValueError,match='官方划分不一致'):
                build_dataset(service,root,[selected_job],intended_split='train')
    finally:
        validation_native.rename(training_native)
        scene_path.write_bytes(original_scene)
        with runs.connect() as db:
            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(newest),newest['run_id']))
    # Human approval preserves the portable record and auto-failure history;
    # it still requires matching identity/assets and cannot authorize changed text.
    from app.quality.sample_review import text_hash, approval_binding, MANUAL_APPROVAL_POLICY
    human_version=copy.deepcopy(newest)
    human_version.update(status='needs_evidence',diagnostic_pass=False,final_expression=None,
        stop_reason='description_validation_failed')
    human_version['attempts'][-1]['ground_checks']=[{'diagnostic_pass':False}]
    human_version['manual_review']={'verdict':'accepted','approval_policy':MANUAL_APPROVAL_POLICY,
        'run_id':human_version['run_id'],'expression_sha256':text_hash(newest['final_expression']),
        'packet_binding':approval_binding(human_version['current_packet']),'localization_correct':True,
        'confirmed_object_id':human_version['current_packet']['target_object_id']}
    assert rejection(human_version) is None
    assert rejection({**human_version,'inflight':{'pending':True}})=='not_successful'
    assert rejection({**human_version,'final_expression':'Changed text.'})=='not_successful'
    with runs.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(human_version),newest['run_id']))
    with build_dataset(service,root) as human_stream:
        with zipfile.ZipFile(human_stream) as archive:
            human_rows=[json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
            identity=human_version['current_packet']['target_object_id']
            assert next(r for r in human_rows if r['track_id']==identity)==next(r for r in rows if r['track_id']==identity)
            assert all(not {'quality','manual_review','approval_policy'} & set(row) for row in human_rows)
    assert runs.snapshot(newest['run_id'])['diagnostic_pass'] is False
    with runs.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(newest),newest['run_id']))
    # Manual rejection/discard applies to every reference to the same run. Do
    # not silently resurrect an older successful description of that track.
    from app.quality.sample_review import text_hash
    for verdict in ('rejected','discarded'):
        reviewed = {**newest,'manual_review':{'verdict':verdict,'expression_sha256':text_hash(newest['final_expression'])}}
        with runs.connect() as db:
            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(reviewed),newest['run_id']))
        with build_dataset(service,root) as reviewed_stream:
            with zipfile.ZipFile(reviewed_stream) as archive:
                current_rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
                assert len(current_rows)==1 and current_rows[0]['track_id']=='second-track'
    with runs.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(newest),newest['run_id']))
    mutated = copy.deepcopy(first); mutated['scene']='../outside.json'
    with pytest.raises(ValueError): loader.load_sample(extracted,mutated)
    for updates in ({'source_view':None}, {'source_view':'SIDE_LEFT'}, {'camera_id':4},
                    {'image_path':f'scenes/{p.scene.scene_id}/camera-4.jpg'}, {'image_path':'../outside.jpg'}):
        with pytest.raises(ValueError): loader.load_sample(extracted,{**first,**updates})
    # Loading the first query does not touch another query's selected image.
    (extracted/f'scenes/{p.scene.scene_id}/camera-4.jpg').unlink()
    loader.load_sample(extracted,first)
    with pytest.raises(ValueError,match='Missing dataset file'):
        loader.load_sample(extracted,next(row for row in rows if row['track_id']=='second-track'))
    exported_calibration = extracted/f'scenes/{p.scene.scene_id}/calibration.json'
    changed = json.loads(exported_calibration.read_text(encoding='utf-8'))
    changed['camera_models']['1']['ego2cam'][0][3] += 1
    atomic_json(exported_calibration,changed)
    with pytest.raises(ValueError,match='calibration differs'): loader.load_sample(extracted,first)
    saved_state = states[newest['run_id']]
    missing_view = copy.deepcopy(saved_state)
    missing_view['current_packet']['source_view'] = None
    with runs.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(missing_view),newest['run_id']))
    with pytest.raises(ValueError,match='source_view'): build_dataset(service,root)
    with runs.connect() as db:
        for rid,state in states.items():
            db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps({**state,'diagnostic_pass':False}),rid))
    with pytest.raises(ValueError,match='暂无'):build_dataset(service,root)
    with runs.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(saved_state),newest['run_id']))
    (folder/'camera-1.jpg').unlink()
    with pytest.raises(ValueError,match='Asset missing'):build_dataset(service,root)
    with (folder/'points.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError):
        read_scene(folder)


def test_task_cleanup_preserves_shared_inputs_and_removes_exclusive_history(workspace, monkeypatch):
    w = workspace
    first = w.task(1)
    second = w.task(2, packet=first.packet, scene=first.scene, method=first.method)
    # Same-task recovery may retain multiple frozen methods. Deleting one
    # task removes only its exclusive methods, including older revisions.
    first_revision, second_revision = 'a2'*32, 'b2'*32
    with w.store.connect() as db:
        for task, revision in ((first, first_revision), (second, second_revision)):
            BatchStore.event(db, task.job, 'execution_revision', {
                'previous': {'method_sha256': revision}, 'current': {'method_sha256': task.method}})
            atomic_json(w.annotation/'methods'/revision/'manifest.json', {'method_sha256': revision})
    # Unprocessed queues own no image/point resources. Their removal must not
    # parse unrelated saved annotations or scan every cached scene manifest.
    pending_job = ident(800)
    with w.store.connect() as db:
        db.execute("INSERT INTO batches SELECT ?,spec,catalog_sha256,method_sha256,'ready',NULL,NULL,NULL,created_at FROM batches WHERE job_id=?",
            (pending_job, first.job))
        db.execute("INSERT INTO batch_items SELECT ?,ordinal,target,'pending',NULL,NULL,NULL,0 FROM batch_items WHERE job_id=?",
            (pending_job, first.job))
    with monkeypatch.context() as fast_delete:
        def no_artifact_scan(*args):
            raise AssertionError('An unprocessed task must not scan the artifact pool')
        fast_delete.setattr(w.cleaner, 'pool', no_artifact_scan)
        queue_preview = w.cleaner.preview([pending_job])
        assert queue_preview['summary']['runs'] == queue_preview['summary']['scenes'] == 0
        assert not w.cleaner.delete([pending_job], queue_preview['token'])['cleanup_pending']
    with w.store.connect() as db:
        assert db.execute('SELECT count(*) FROM sample_presentations WHERE job_id=?',(pending_job,)).fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM sample_presentations WHERE job_id=?',(first.job,)).fetchone()[0] == 1
    assert (w.root/'runtime/evidence'/first.packet/'packet.json').exists()
    assert (w.annotation/first.run/'checkpoint.json').exists()
    with pytest.raises(ValueError):
        w.cleaner.safe(w.root/'runtime/../waymo_trainset/input.parquet')
    import stat
    guarded_file = w.annotation/first.run/'checkpoint.json'
    original_lstat = Path.lstat
    with monkeypatch.context() as junction:
        def reparse_point(path, *args, **kwargs):
            result = original_lstat(path, *args, **kwargs)
            if path == guarded_file:
                return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return result
        junction.setattr(Path, 'lstat', reparse_point)
        with pytest.raises(ValueError, match='目录联接'):
            w.cleaner.safe(guarded_file)
    from app.infrastructure.task_cleanup import ids, IDENTIFIERS
    references = {'nested': [first.run, {'path': f'runtime/scenes/{first.scene}/points.npz'}],
        first.method: ['a' + first.run, 123456789012345678901234]}
    assert ids(references) == set(IDENTIFIERS.findall(json.dumps(references, ensure_ascii=False)))
    for data in (references,{'escaped':'\b'+first.run,'unicode':'中'+first.run},
                 {'escaped':'\f'+first.method,'nested':{'key':first.packet}}):
        raw = json.dumps(data)
        assert set(TaskCleaner.run_metadata('test',raw,data)['references']) == ids(data)
    recovery = {**first.state, 'run_id': ident(701), 'batch_id': None,
                'provider_scope': 'real-model', 'recovery': {'parent_run_id': first.run}}
    revalidation = {**recovery, 'run_id': ident(702), 'recovery': None,
                    'revalidation_source_run_id': recovery['run_id']}
    development = {**recovery, 'run_id': ident(704), 'recovery': None,
                   'development_recovery_source_run_id': first.run}
    validation = {**recovery, 'run_id': ident(705), 'recovery': None,
                  'development_validation_source_run_id': first.run}
    validation_child = {**validation, 'run_id': ident(706),
                        'development_validation_source_run_id': validation['run_id']}
    standalone = {'run_id': ident(703), 'status': 'needs_evidence', 'provider_scope': 'real-model',
                  'current_packet': {'context': 'standalone', 'target_object_id': 'other'}}
    for state in (recovery, revalidation, development, validation, validation_child, standalone):
        w.store.create(state)
        atomic_json(w.annotation / state['run_id'] / 'checkpoint.json', state)
    preview = w.cleaner.preview([first.job])
    assert preview['summary']['runs'] == 6
    # Changes hidden deep inside a snapshot or manifest must still invalidate
    # the confirmation, even though deletion now uses compact reference graphs.
    original_snapshot = json.dumps(first.state)
    with w.store.connect() as db:
        changed = {**first.state, 'trace': {'expression': 'Changed after preview'}}
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?', (json.dumps(changed), first.run))
    with pytest.raises(ValueError, match='已变化'):
        w.cleaner.delete([first.job], preview['token'])
    with w.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?', (original_snapshot, first.run))
    manifest_path = w.root/'runtime/evidence'/first.packet/'packet.json'
    original_manifest = manifest_path.read_bytes()
    packet_data = json.loads(original_manifest)
    atomic_json(manifest_path, {**packet_data, 'description': 'Changed after preview'})
    with pytest.raises(ValueError, match='已变化'):
        w.cleaner.delete([first.job], preview['token'])
    manifest_path.write_bytes(original_manifest)
    # A small deletion must not compact the entire database, nor fetch every
    # historical event payload. The ordinary transaction still removes rows.
    statements = []
    from contextlib import contextmanager
    original_connect = w.store.connect
    @contextmanager
    def traced_connect():
        with original_connect() as db:
            db.set_trace_callback(statements.append)
            yield db
    monkeypatch.setattr(w.store, 'connect', traced_connect)
    result = w.cleaner.delete([first.job], preview['token'])
    assert not any('VACUUM' in sql.upper() or 'SELECT * FROM EVENTS ' in sql.upper() for sql in statements)
    assert not result['cleanup_pending']
    assert {s['run_id'] for s in w.store.all_runs()} == {second.run, standalone['run_id']}
    assert not (w.annotation / recovery['run_id']).exists()
    assert not (w.annotation / revalidation['run_id']).exists()
    assert not (w.annotation / development['run_id']).exists()
    assert not (w.annotation / validation['run_id']).exists()
    assert not (w.annotation / validation_child['run_id']).exists()
    assert (w.root/'runtime/evidence'/first.packet).exists()
    assert (w.root/'runtime/scenes'/first.scene).exists()
    assert not (w.annotation/'methods'/first_revision).exists()
    assert (w.annotation/'methods'/second_revision/'manifest.json').is_file()
    preview = w.cleaner.preview([second.job])
    assert not w.cleaner.delete([second.job], preview['token'])['cleanup_pending']
    with w.store.connect() as db:
        for table in ('batches', 'batch_items', 'batch_events', 'events', 'controls'):
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0] == 0
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not (w.root/'runtime/evidence'/first.packet).exists()
    assert not (w.root/'runtime/scenes'/first.scene).exists()
    assert not (w.annotation/'methods'/second_revision).exists()
    # A previous version could leave this exact recovery chain after deleting
    # its parent task. The same transactional cleanup repairs it explicitly.
    for state in (recovery, revalidation, development, validation, validation_child):
        w.store.create(state)
        atomic_json(w.annotation / state['run_id'] / 'checkpoint.json', state)
    journal = w.cleaner.journals / 'interrupted-orphan-cleanup'
    backup = journal / 'files/0'
    backup.parent.mkdir(parents=True)
    (w.annotation / recovery['run_id']).replace(backup)
    atomic_json(journal / 'manifest.json', {'job_ids': [], 'run_ids': [recovery['run_id']],
        'committed': False, 'entries': [{'path': 'annotation/' + recovery['run_id'], 'index': 0}]})
    assert not w.cleaner.recover()
    assert (w.annotation / recovery['run_id'] / 'checkpoint.json').is_file()
    preview = w.cleaner.preview([])
    assert preview['summary']['runs'] == 5
    # Startup repairs old orphan chains after all batches were deleted, while
    # preserving real standalone work. Running this twice is idempotent.
    monkeypatch.setattr('app.agent.batch.ROOT',w.root)
    service=BatchService.__new__(BatchService)
    service.runner=SimpleNamespace(root=w.annotation,store=w.store)
    service.recover_deletions()
    service.recover_deletions()
    assert w.store.all_runs() == [standalone]
    from app.agent.selection import unavailable_target_keys
    unavailable, _ = unavailable_target_keys(w.store.all_runs(), 'test', include_prior_attempts=True)
    assert ('context', 'object-1') not in unavailable
    assert ('standalone', 'other') in unavailable
    # Deleting a task header must not cascade into its continuations. Exercise
    # both a multi-level chain and a sibling, retaining immutable run files and
    # the original finite allowance until the final dependent task is removed.
    parent, middle, leaf, sibling = [w.task(n) for n in (11, 12, 13, 14)]
    batches = BatchStore(w.store)
    parent_revision = 'c2'*32
    atomic_json(w.annotation/'methods'/parent_revision/'manifest.json', {'method_sha256': parent_revision})
    parent.state.update(diagnostic_pass=True, final_expression='The vehicle ahead.')
    atomic_json(w.annotation/parent.run/'checkpoint.json', parent.state)
    with w.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?', (json.dumps(parent.state), parent.run))
        original_spec = json.loads(db.execute('SELECT spec FROM batches WHERE job_id=?', (parent.job,)).fetchone()[0])
        original_spec.update(max_model_calls=24, desired_count=4)
        db.execute('UPDATE batches SET spec=? WHERE job_id=?', (json.dumps(original_spec), parent.job))
        batches.event(db, parent.job, 'execution_revision', {
            'previous': {'method_sha256': parent_revision}, 'current': {'method_sha256': parent.method}})
        db.execute("UPDATE batch_items SET target=json_set(target,'$.queue_position',0) WHERE job_id=?", (parent.job,))
        for child, source in ((middle, parent), (leaf, middle), (sibling, parent)):
            batches.event(db, child.job, 'continued_from', {'job_id':source.job, 'inherited_passed':1})
        for child in (middle, leaf):
            db.execute("INSERT INTO batch_items SELECT ?,2,target,'reused',packet_id,run_id,NULL,1 FROM batch_items WHERE job_id=? AND ordinal=1",
                       (child.job, parent.job))
    original_run = (w.annotation/parent.run/'checkpoint.json').read_bytes()
    original_calls = (w.annotation/parent.run/'model/0/result.json').read_bytes()
    ledger = batches.continuation_budget(leaf.job)
    assert ledger['spent'] == 16 and ledger['limit'] == 24
    # A failed filesystem transaction must not leave transferred history or
    # missing task files behind. No real runtime records are involved.
    preview = w.cleaner.preview([middle.job])
    assert preview['job_ids'] == [middle.job] and preview['summary']['retained_tasks'] == 3
    with monkeypatch.context() as failure:
        original_replace = Path.replace
        def interrupted_move(path, target):
            if path == w.annotation/'batches'/middle.job:
                raise OSError('simulated locked task folder')
            return original_replace(path, target)
        failure.setattr(Path, 'replace', interrupted_move)
        with pytest.raises(OSError, match='simulated locked'):
            w.cleaner.delete([middle.job], preview['token'])
    assert batches.continuation_budget(leaf.job)['spent'] == 16
    with w.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM batch_events WHERE kind='retained_task_history'").fetchone()[0] == 0
    for task in (middle, parent, sibling):
        preview = w.cleaner.preview([task.job])
        result = w.cleaner.delete([task.job], preview['token'])
        assert result['deleted_job_ids'] == [task.job] and not result['cleanup_pending']
        batches = BatchStore(RunStore(w.store.path))  # Reload persisted history.
        ledger = batches.continuation_budget(leaf.job)
        assert ledger['spent'] == 16 and ledger['limit'] == 24
        assert ledger['root_spec']['desired_count'] == 4
        assert batches.detail(leaf.job)['model_calls_reserved'] == 4
        assert batches.detail(leaf.job)['selection_progress']['diagnostic_passed'] == 1
        assert (w.annotation/parent.run/'checkpoint.json').read_bytes() == original_run
        assert (w.annotation/parent.run/'model/0/result.json').read_bytes() == original_calls
        assert (w.annotation/'methods'/parent.method/'manifest.json').is_file()
        assert (w.annotation/'methods'/parent_revision/'manifest.json').is_file()
        assert w.store.snapshot(parent.run) == parent.state
    assert {j['job_id'] for j in batches.summaries()} == {leaf.job}
    assert {i['run_id'] for i in batches.detail(leaf.job)['items']} == {leaf.run, parent.run}
    assert batches.continuation_budget(parent.job)['spent'] == 16  # Historical review source.
    preview = w.cleaner.preview([leaf.job])
    assert preview['summary']['runs'] == 4
    assert not w.cleaner.delete([leaf.job], preview['token'])['cleanup_pending']
    assert w.store.all_runs() == [standalone]
    assert not (w.annotation/'methods'/parent.method).exists()
    assert not (w.annotation/'methods'/parent_revision).exists()
    # Keep a successor running while deleting its paused source. Its model
    # result is saved once between preview/commit and once during the deletion
    # transaction, without requesting pause or invalidating unrelated progress.
    import threading
    old, running = w.task(21), w.task(22)
    batches = BatchStore(w.store)
    owner = 'active-cleanup-regression'
    state = {**running.state, 'status':'running'}
    with w.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=?,revision=1 WHERE run_id=?', (json.dumps(state), running.run))
        db.execute("UPDATE batches SET status='paused' WHERE job_id=?", (old.job,))
        db.execute("UPDATE batches SET status='running',owner=? WHERE job_id=?", (owner, running.job))
        db.execute('DELETE FROM controls WHERE run_id=?', (running.run,))
        batches.event(db, running.job, 'continued_from', {'job_id':old.job,'inherited_passed':1})
        db.execute("INSERT INTO batch_items SELECT ?,2,target,'reused',packet_id,run_id,NULL,1 FROM batch_items WHERE job_id=?",
                   (running.job, old.job))
    w.store.acquire_executor(owner)
    w.store.claim(running.run, owner, state)
    service = BatchService.__new__(BatchService)
    service.runner = SimpleNamespace(root=w.annotation, store=w.store, lock=threading.Lock(),
                                     active=None, active_batch=running.job)
    service.store = batches
    with pytest.raises(RunBusy, match='所选任务正在执行'):
        service.deletion_preview([running.job])
    preview = service.deletion_preview([old.job])
    assert preview['job_ids'] == [old.job] and preview['summary']['runs'] == 0
    ready, during, entering_save, finished = [threading.Event() for _ in range(4)]
    failures = []
    active_log = w.annotation/'logs/application.jsonl'
    active_log.parent.mkdir(exist_ok=True)
    def continue_working():
        try:
            for used in (5,6):
                if used == 6:
                    assert during.wait(30)
                    entering_save.set()
                state['model_calls_used'] = used
                w.store.save(running.run, owner, state, 'model_call_reserved', {'used':used})
                w.store.renew(running.run,owner)
                w.store.renew_executor(owner)
                with w.store.connect() as db:
                    batches.event(db, running.job, 'progress_saved', {'used':used})
                    db.execute('UPDATE batch_items SET reason=? WHERE job_id=? AND ordinal=1',
                               (f'progress-{used}', running.job))
                with active_log.open('a',encoding='utf-8') as log:
                    log.write(json.dumps({'job_id':running.job,'context':'context','used':used})+'\n')
                # Preparation publishes another cache while the dialog is open.
                atomic_json(w.root/'runtime/scenes'/ident(8800+used)/'scene.json',
                            {'context':'context','frame_timestamp_micros':used})
                if used == 5:ready.set()
        except BaseException as exc:
            failures.append(exc)
        finally:finished.set()
    worker = threading.Thread(target=continue_working)
    worker.start()
    try:
        assert ready.wait(30), failures
        with monkeypatch.context() as concurrent_move:
            original_replace = Path.replace
            def move_while_saving(path, target):
                if path == w.annotation/'batches'/old.job:
                    during.set()
                    assert entering_save.wait(30)
                    # The writer must finish while filesystem staging is still
                    # held here. A BEGIN IMMEDIATE around staging deadlocks this.
                    assert finished.wait(2), failures
                return original_replace(path, target)
            concurrent_move.setattr(Path,'replace',move_while_saving)
            checkpoint_connect=w.store.connect
            checkpoints=[]
            class BusyCheckpoint:
                def __init__(self,db):self.db=db
                def __getattr__(self,name):return getattr(self.db,name)
                def execute(self,sql,*args):
                    if sql=='PRAGMA wal_checkpoint(TRUNCATE)':
                        checkpoints.append(sql)
                        error=sqlite3.OperationalError('database is locked')
                        error.sqlite_errorcode=sqlite3.SQLITE_BUSY
                        raise error
                    return self.db.execute(sql,*args)
            @contextmanager
            def busy_checkpoint_connection():
                with checkpoint_connect() as db:yield BusyCheckpoint(db)
            concurrent_move.setattr(w.store,'connect',busy_checkpoint_connection)
            result = service.delete([old.job], preview['token'])
            assert len(checkpoints)==1
        assert result['deleted_job_ids'] == [old.job] and not result['cleanup_pending']
        assert result['database_write_seconds'] < 2
        assert finished.wait(5)
        assert not failures
        assert w.store.snapshot(old.run) == old.state
        assert w.store.snapshot(running.run)['model_calls_used'] == 6
        assert service.runner.active_batch == running.job
        assert batches.detail(running.job)['executor_active']
        assert batches.continuation_budget(running.job)['spent'] == 10
        assert (w.root/'runtime/evidence'/old.packet/'packet.json').exists()
        assert (w.root/'runtime/scenes'/old.scene/'scene.json').exists()
        assert [json.loads(line)['used'] for line in active_log.read_text(encoding='utf-8').splitlines()] == [5,6]
        with w.store.connect() as db:
            assert db.execute('SELECT command FROM controls WHERE run_id=?',(running.run,)).fetchone() is None
        with pytest.raises(RunBusy, match='所选任务正在执行'):
            w.cleaner.preview([running.job])  # Direct API bypass is also guarded.
    finally:
        during.set();worker.join(5)
        w.store.release(running.run,owner)
        w.store.release_executor(owner)
    assert w.raw.read_bytes() == b'original data'
    assert w.config.read_text() == 'original configuration'

    # Candidate policy outlives deleted tasks; preview binds both choices without
    # rescanning assets when the confirmation checkbox changes.
    sealed = w.task(820)
    untouched = {**sealed.target, 'object_id':'never-started', 'ordinal':2}
    with w.store.connect() as db:
        db.execute('INSERT INTO batch_items VALUES(?,?,?,?,?,?,?,?)',
            (sealed.job,2,json.dumps(untouched),'pending',None,None,None,0))
    policy = w.cleaner.preview([sealed.job])
    assert policy['return_to_pool'] is False and policy['processed_tracks'] == 1
    assert policy['candidate_tokens']['exclude'] == policy['token']
    with pytest.raises(ValueError, match='已变化'):
        w.cleaner.delete([sealed.job],policy['token'],True)
    sealed_key = ('context',sealed.target['object_id'])
    assert sealed_key not in w.store.excluded_target_keys()
    original_replace = Path.replace
    with monkeypatch.context() as rollback:
        def fail_delete(path, target):
            if path == w.annotation/'batches'/sealed.job:
                raise OSError('simulated filesystem failure')
            return original_replace(path,target)
        rollback.setattr(Path,'replace',fail_delete)
        with pytest.raises(OSError,match='simulated'):
            w.cleaner.delete([sealed.job],policy['token'],False)
    assert sealed_key not in w.store.excluded_target_keys()
    assert w.store.snapshot(sealed.run) == sealed.state
    assert not w.cleaner.delete([sealed.job],policy['token'],False)['cleanup_pending']
    assert sealed_key in w.store.excluded_target_keys()
    assert ('context','never-started') not in w.store.excluded_target_keys()
    assert sealed_key in RunStore(w.store.path).excluded_target_keys()

    reopened = w.task(821)
    with w.store.connect() as db:
        db.execute("UPDATE batches SET status='paused' WHERE job_id=?",(reopened.job,))
    policy = w.cleaner.preview([reopened.job])
    assert policy['return_to_pool'] is True
    assert not w.cleaner.delete([reopened.job],policy['token'],True)['cleanup_pending']
    assert ('context',reopened.target['object_id']) not in w.store.excluded_target_keys()
    completed_return = w.task(822)
    policy = w.cleaner.preview([completed_return.job])
    assert not w.cleaner.delete([completed_return.job],policy['candidate_tokens']['return'],True)['cleanup_pending']
    assert ('context',completed_return.target['object_id']) not in w.store.excluded_target_keys()

    # Import exactly the retained export identities, atomically and idempotently;
    # the temporary JSONL may disappear immediately afterwards.
    from app.infrastructure.candidate_pool import import_manifest
    from app.datasets.export_loader import SAMPLE_FIELDS
    export_record = dict.fromkeys(SAMPLE_FIELDS)
    export_record.update(sample_id=ident(823),context='old-export',track_id='old-track')
    temporary_manifest = w.root/'first-batch-samples.jsonl'
    temporary_manifest.write_text(json.dumps(export_record)+'\n',encoding='utf-8')
    before_import = w.store.excluded_target_keys()
    with pytest.raises(ValueError,match='实际为'):
        import_manifest(temporary_manifest,w.store.path,expected_count=700)
    assert before_import == w.store.excluded_target_keys()
    assert import_manifest(temporary_manifest,w.store.path,expected_count=1)['added'] == 1
    assert import_manifest(temporary_manifest,w.store.path,expected_count=1)['added'] == 0
    temporary_manifest.unlink()
    assert ('old-export','old-track') in w.store.excluded_target_keys()

    # An already-created queue must check exclusions again immediately before
    # preparation/model calls, rather than relying on its stale candidate list.
    import app.agent.batch_execution as execution_module
    queued = w.task(824,run=False)
    api = SimpleNamespace(runner=SimpleNamespace(store=w.store,root=w.annotation),store=BatchStore(w.store))
    execution = execution_module.BatchExecution(api,queued.job,'test-owner',{})
    execution.timer = SimpleNamespace(begin_item=lambda _:None,finish_item=lambda _:None)
    blocked_item = {'ordinal':1,'target':{**queued.target,'context':'old-export','object_id':'old-track'},
                    'run_id':None,'packet_id':None}
    assert execution.process_item(blocked_item) is None
    with w.store.connect() as db:
        row = db.execute('SELECT status,reason FROM batch_items WHERE job_id=?',(queued.job,)).fetchone()
        assert (row['status'],row['reason']) == ('not_needed','target_excluded_after_task_deletion')

    import app.routers.system as system_routes
    with monkeypatch.context() as listing:
        listing.setattr(system_routes,'load_catalog',lambda **kwargs:{'summary':{'selected':1},'targets':[
            {'context':'old-export','object_id':'old-track','target_id':'exported-target'}]})
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(console=api.runner)))
        available = system_routes.status(request)
        assert available['replacement_unavailable_target_ids'] == ['exported-target']

    # Unchanged metadata is reused, while a newly acquired reference between
    # staging and commit aborts safely and restores the victim's files.
    victim = w.task(826)
    def writer_probe(**values):
        if values.get('stage') in {'references','assets','staging','cleaning'}:
            with w.store.connect() as writer:
                writer.execute('PRAGMA busy_timeout=20')
                writer.execute('BEGIN IMMEDIATE')
                writer.rollback()
    cleaner = TaskCleaner(w.store,w.root,writer_probe)
    cached = cleaner.preview([victim.job])
    with monkeypatch.context() as reuse:
        reuse.setattr(cleaner,'run_metadata',lambda *_:pytest.fail('Unchanged runs must use the reference cache'))
        assert cleaner.preview([victim.job])['token'] == cached['token']
    referenced = {'run_id':ident(827),'status':'needs_evidence','trace':{'links':[victim.run]}}
    with monkeypatch.context() as race:
        original = Path.replace
        def acquire_reference(path,target):
            if path == w.annotation/'batches'/victim.job:
                w.store.create(referenced)
            return original(path,target)
        race.setattr(Path,'replace',acquire_reference)
        with pytest.raises(ValueError,match='已变化'):
            cleaner.delete([victim.job],cached['token'])
    assert w.store.snapshot(victim.run) == victim.state
    assert (w.annotation/victim.run/'checkpoint.json').is_file()
    assert ('context',victim.target['object_id']) not in w.store.excluded_target_keys()

    # Background operations return quickly, expose real stage counts, dedup a
    # repeated submission, and retain errors/results for polling without replay.
    from app.infrastructure.deletion_operations import DeletionOperations
    queued_delete = w.task(828)
    gate,started = threading.Event(),threading.Event()
    class ProgressService:
        def deletion_preview(self,jobs,progress=None):
            progress(stage='references',completed=20,total=100,files_checked=200)
            started.set()
            assert gate.wait(5)
            return TaskCleaner(w.store,w.root,progress).preview(jobs)
        def delete(self,jobs,token,return_to_pool=None,progress=None):
            return TaskCleaner(w.store,w.root,progress).delete(jobs,token,return_to_pool)
    operations = DeletionOperations(ProgressService())
    try:
        check = operations.create('preview',[queued_delete.job])
        assert started.wait(2)
        assert operations.status(check['operation_id'])['completed'] == 20
        assert operations.create('preview',[queued_delete.job])['operation_id'] == check['operation_id']
        with pytest.raises(ValueError,match='正在执行'):
            operations.create('preview',[victim.job])
        gate.set()
        operations.jobs[check['operation_id']]['thread'].join(5)
        ready = operations.status(check['operation_id'])
        assert ready['status'] == 'completed' and ready['result']['summary']['tasks'] == 1
        rejected = operations.create('delete',[queued_delete.job],'0'*64,False)
        operations.jobs[rejected['operation_id']]['thread'].join(5)
        assert operations.status(rejected['operation_id'])['status'] == 'failed'
        deleted = operations.create('delete',[queued_delete.job],ready['result']['token'],False)
        operations.jobs[deleted['operation_id']]['thread'].join(5)
        final = operations.status(deleted['operation_id'])
        assert final['status'] == 'completed' and final['result']['deleted_job_ids'] == [queued_delete.job]
    finally:
        gate.set();operations.close()

    # Completion of an unrelated worker must not expand the previously confirmed
    # shared-resource scope, trigger a global scan, or invalidate confirmation.
    old_task,other_task = w.task(829),w.task(830)
    holder = 'completion-scope-test'
    with w.store.connect() as db:
        db.execute("UPDATE batches SET status='running',owner=? WHERE job_id=?",(holder,other_task.job))
    w.store.acquire_executor(holder)
    w.store.claim(other_task.run,holder,other_task.state)
    cleaner = TaskCleaner(w.store,w.root)
    confirmation = cleaner.preview([old_task.job])
    assert confirmation['shared_cleanup_deferred']
    w.store.release(other_task.run,holder);w.store.release_executor(holder)
    with w.store.connect() as db:
        db.execute("UPDATE batches SET status='completed',owner=NULL WHERE job_id=?",(other_task.job,))
    with monkeypatch.context() as stable_scope:
        stable_scope.setattr(cleaner,'pool',lambda *_:pytest.fail('An unrelated completion must not expand shared cleanup'))
        result = cleaner.delete([old_task.job],confirmation['token'])
    assert result['deleted_job_ids'] == [old_task.job]
    assert (w.root/'runtime/scenes'/old_task.scene).exists()

    # Deleting a never-executed retry with a live paused source is metadata-only.
    # Its source reference must not trigger a global scene/evidence/log sweep.
    live_source,empty_retry=w.task(841),w.task(842,run=False)
    with w.store.connect() as db:
        target=json.loads(db.execute('SELECT target FROM batch_items WHERE job_id=?',(empty_retry.job,)).fetchone()['target'])
        target['retry_authorization']={'source_run_id':live_source.run,'source_job_id':live_source.job,
            'kind':'manual_review_retry','token':'unused-feedback-round','additional_calls':12,'call_limit':16}
        db.execute("UPDATE batch_items SET target=?,status='pending',run_id=NULL,packet_id=NULL WHERE job_id=?",(json.dumps(target),empty_retry.job))
        db.execute("UPDATE batches SET status='paused',control='pause' WHERE job_id=?",(live_source.job,))
        db.execute("UPDATE batches SET status='queued',spec=json_set(spec,'$.prepare_only',json('false')) WHERE job_id=?",(empty_retry.job,))
        source_header=tuple(db.execute('SELECT * FROM batches WHERE job_id=?',(live_source.job,)).fetchone())
        source_events=list(db.execute('SELECT kind,payload FROM batch_events WHERE job_id=? ORDER BY seq',(live_source.job,)))
    with monkeypatch.context() as narrow_delete:
        narrow_delete.setattr(w.cleaner,'pool',lambda *a:pytest.fail('An unused retry must not scan shared assets'))
        confirmation=w.cleaner.preview([empty_retry.job])
        assert confirmation['shared_cleanup_deferred']
        assert confirmation['summary']['tasks']==1 and confirmation['summary']['runs']==0
        assert confirmation['summary']['evidence']==confirmation['summary']['scenes']==0
        result=w.cleaner.delete([empty_retry.job],confirmation['token'])
    assert result['deleted_job_ids']==[empty_retry.job]
    assert w.store.snapshot(live_source.run)==live_source.state
    assert (w.root/'runtime/scenes'/live_source.scene).exists()
    assert (w.root/'runtime/evidence'/live_source.packet).exists()
    assert (w.annotation/live_source.run/'model/0/result.json').exists()
    with w.store.connect() as db:
        assert tuple(db.execute('SELECT * FROM batches WHERE job_id=?',(live_source.job,)).fetchone())==source_header
        assert list(db.execute('SELECT kind,payload FROM batch_events WHERE job_id=? ORDER BY seq',(live_source.job,)))==source_events

    # An unused retry grant in a surviving task protects its original billed
    # run even before the new child is created. Last-reference deletion releases it.
    retry_source,retry_queue=w.task(831),w.task(832)
    with w.store.connect() as db:
        target=json.loads(db.execute('SELECT target FROM batch_items WHERE job_id=?',
            (retry_queue.job,)).fetchone()['target'])
        target['retry_authorization']={'source_run_id':retry_source.run,'call_limit':22,
            'token':'pending-retry','additional_calls':12}
        db.execute("UPDATE batch_items SET target=?,status='pending',run_id=NULL,packet_id=NULL WHERE job_id=?",
            (json.dumps(target),retry_queue.job))
    confirmation=w.cleaner.preview([retry_source.job])
    assert not w.cleaner.delete([retry_source.job],confirmation['token'])['cleanup_pending']
    assert w.store.snapshot(retry_source.run)==retry_source.state
    assert (w.annotation/retry_source.run/'model/0/result.json').exists()
    confirmation=w.cleaner.preview([retry_queue.job])
    assert not w.cleaner.delete([retry_queue.job],confirmation['token'])['cleanup_pending']
    assert w.store.snapshot(retry_source.run,missing_ok=True) is None

    # A consuming annotation retains the actual calls behind reused perception.
    # Deleting its source header keeps that ledger; last-consumer deletion frees it.
    observation_source,observation_consumer=w.task(833),w.task(834)
    consumer_state={**observation_consumer.state,'observation_source_run_ids':[observation_source.run]}
    with w.store.connect() as db:
        db.execute('UPDATE runs SET snapshot=? WHERE run_id=?',(json.dumps(consumer_state),observation_consumer.run))
    confirmation=w.cleaner.preview([observation_source.job])
    assert not w.cleaner.delete([observation_source.job],confirmation['token'])['cleanup_pending']
    assert w.store.snapshot(observation_source.run)==observation_source.state
    assert (w.annotation/observation_source.run/'model/0/result.json').exists()
    confirmation=w.cleaner.preview([observation_consumer.job])
    assert not w.cleaner.delete([observation_consumer.job],confirmation['token'])['cleanup_pending']
    assert w.store.snapshot(observation_source.run,missing_ok=True) is None

