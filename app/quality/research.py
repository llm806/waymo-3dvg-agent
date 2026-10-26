"""Frozen blind audit cohorts and conservative, evidence-bound research releases.

Audits measure human native-candidate re-grounding, not free-box model accuracy.
No audit, metric or certificate is synthesized by the annotation generator.
"""
import hashlib
import json
import math
import random
from collections import Counter
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4
from functools import wraps

from pydantic import Field, model_validator

from app.infrastructure.artifacts import atomic_json, checked_asset, file_sha256
from app.schemas.contracts import Contract, EvidencePacket
from app.datasets.evidence import ROOT
from app.datasets.training import read_scene
from app.datasets.waymo import raw_source_root

LEGACY_PROTOCOL = 'blind-native-candidate-human-v1'
PROTOCOL = 'blind-native-candidate-multimodal-human-v2'
RELEASE_MIN_REVIEWED = 30
RELEASE_MAX_ERROR_UPPER95 = 0.1


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.runs.maintenance():
            return method(self, *args, **kwargs)
    return wrapped


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def final_text(state):
    return state.get('final_expression') or next((a['generated']['expression'] for a in reversed(state.get('attempts', [])) if a.get('generated')), None)


def wilson(successes, total):
    if not total:
        return {'numerator': 0, 'denominator': 0, 'value': None, 'ci95': None}
    z = 1.959963984540054
    p = successes / total
    divisor = 1 + z*z/total
    center = (p + z*z/(2*total)) / divisor
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / divisor
    return {'numerator': successes, 'denominator': total, 'value': p, 'ci95': [max(0, center-radius), min(1, center+radius)]}


class AnnotationConfiguration(Contract):
    # Inference parameters and the cumulative track limit are fixed in hashed code.
    # These settings vary independently of code; cohort size is not a per-target method.
    model: str = Field(min_length=1, pattern=r'\S')
    base_url: str = Field(min_length=1, pattern=r'\S')
    calls_per_target: int = Field(strict=True, ge=4, le=12)
    max_repairs: int = Field(strict=True, ge=0, le=2)


def validate_release_report(report):
    """A boolean flag alone is not an independent quality certificate."""
    AnnotationConfiguration.model_validate(report.get('method_configuration'))
    policy=report.get('policy',{});risk=report.get('automatic_candidate_error',{})
    n=risk.get('denominator');errors=risk.get('numerator');reviewed=report.get('reviewed_independent')
    if (not all(isinstance(v,int) and not isinstance(v,bool) for v in (n,errors,reviewed))
            or not report.get('complete') or not report.get('calibration_gate_passed')
            or report.get('prior_exposure_excluded')!=0
            or policy.get('min_reviewed',0)<RELEASE_MIN_REVIEWED
            or policy.get('max_error_upper95',1)>RELEASE_MAX_ERROR_UPPER95
            or not 0<=errors<=n<=reviewed or n<policy['min_reviewed']):
        raise ValueError('Independent report has insufficient or inconsistent quality observations')
    recalculated=wilson(errors,n)
    if risk!=recalculated or recalculated['ci95'][1]>policy['max_error_upper95']:
        raise ValueError('Independent error risk does not satisfy the recalculated uncertainty bound')
    rows=report.get('cases')
    if rows is not None:
        if len(rows)!=reviewed or len({r['run_id'] for r in rows})!=reviewed:
            raise ValueError('Independent report case count or identity is inconsistent')
        diagnosed=[r for r in rows if r.get('diagnostic_pass')]
        if len(diagnosed)!=n or sum(not r['accepted'] for r in diagnosed)!=errors:
            raise ValueError('Independent report risk differs from its actual case judgments')
    elif not report.get('case_gold_withheld_from_training_package'):
        raise ValueError('Independent report is missing its audited case records')


class CohortSpec(Contract):
    job_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    purpose: Literal['development', 'calibration', 'evaluation'] = 'development'
    sample_size: int = Field(default=20, ge=1, le=500)
    contexts: list[str] | None = None
    seed: int = Field(default=42, ge=0, le=2147483647)
    min_reviewed: int = Field(default=30, ge=10, le=500)
    max_error_upper95: float = Field(default=0.1, gt=0, lt=0.5)
    text_version: Literal['final', 'first_draft'] = 'final'


class AuditAnswer(Contract):
    reviewer: str = Field(min_length=1, max_length=80)
    selected_labels: list[str] = Field(max_length=500)
    factual_correct: bool
    # The old geometry-only criterion is retained solely for frozen v1 cohorts.
    geometry_sufficient: bool | None = None
    input_sufficient: bool | None = None
    alignment_correct: bool
    prior_target_exposure: bool
    notes: str = Field(default='', max_length=2000)

    @model_validator(mode='after')
    def unique_labels(self):
        self.reviewer=self.reviewer.strip()
        if not self.reviewer:raise ValueError('Evaluator identity must not be blank')
        if len(set(self.selected_labels)) != len(self.selected_labels):
            raise ValueError('Duplicate candidate selection')
        if self.geometry_sufficient is None and self.input_sufficient is None:
            raise ValueError('Input sufficiency judgment is required')
        return self


class ResearchStore:
    def __init__(self, runs, root: Path = ROOT):
        self.runs, self.root = runs, root
        with runs.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS audit_cohorts (
                cohort_id TEXT PRIMARY KEY, spec TEXT NOT NULL, frozen TEXT NOT NULL,
                status TEXT NOT NULL, created_at REAL NOT NULL, report TEXT);
              CREATE TABLE IF NOT EXISTS audit_localizations (
                cohort_id TEXT NOT NULL, case_id TEXT NOT NULL, selected_labels TEXT NOT NULL,
                created_at REAL NOT NULL, PRIMARY KEY(cohort_id,case_id));
              CREATE TABLE IF NOT EXISTS audit_answers (
                cohort_id TEXT NOT NULL, case_id TEXT NOT NULL, answer TEXT NOT NULL,
                updated_at REAL NOT NULL, PRIMARY KEY(cohort_id,case_id));
              CREATE TABLE IF NOT EXISTS audit_answer_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, cohort_id TEXT NOT NULL,
                case_id TEXT NOT NULL, answer TEXT NOT NULL, at REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS label_releases (
                release_id TEXT PRIMARY KEY, manifest TEXT NOT NULL, created_at REAL NOT NULL);
            """)

    @serialized
    def create(self, spec, batch):
        if batch['job_id'] != spec.job_id or batch['status'] not in {'completed', 'blocked'} or batch.get('executor_active') or batch['spec']['prepare_only']:
            raise ValueError('Audit requires a settled real annotation batch with no active executor')
        configuration = AnnotationConfiguration.model_validate({
            **(batch['spec'].get('provider') or {}),
            'calls_per_target': batch['spec'].get('calls_per_target'),
            'max_repairs': batch['spec'].get('max_repairs')}).model_dump()
        states, workload = [], []
        for item in batch['items']:
            if batch['spec'].get('desired_count') is not None:
                from app.agent.selection import attempted_item
                if not attempted_item(item):
                    continue
            if spec.contexts is not None and item['target']['context'] not in spec.contexts:
                continue
            workload.append(item)
            if item.get('run_id'):
                state = self.runs.snapshot(item['run_id'])
                if state.get('provider_scope') != 'real-model':
                    raise ValueError('Local test runs cannot enter independent research evaluation')
                if state.get('status') in {'running', 'paused', 'stopped'}:
                    raise ValueError('Unfinished child runs cannot enter a frozen audit')
                if (any(state.get(k) != configuration[k] for k in ('model', 'base_url', 'max_repairs'))
                        or type(state.get('model_call_budget')) is not int
                        or not 1 <= state['model_call_budget'] <= configuration['calls_per_target']):
                    raise ValueError('Annotation run differs from the frozen model or execution configuration')
                text = final_text(state) if spec.text_version == 'final' else (state.get('attempts') or [{}])[0].get('generated',{}).get('expression')
                if text:
                    states.append(state)
        if not workload or not states:
            raise ValueError('No real candidate texts in the requested cohort scope')
        contexts = sorted({i['target']['context'] for i in workload})
        if spec.contexts is not None and (not spec.contexts or set(spec.contexts) != set(contexts)):
            raise ValueError('Unknown or empty audit context selection')
        states.sort(key=lambda s: s['run_id'])
        random.Random(spec.seed).shuffle(states)
        selected = states[:spec.sample_size]
        identity = digest({'protocol': PROTOCOL, 'spec': spec.model_dump(), 'method': batch['method_sha256'],
            'method_configuration': configuration, 'states': [digest(s) for s in selected]})[:24]
        salt = uuid4().hex
        cases = []
        for state in selected:
            packet = EvidencePacket.model_validate(state['current_packet'])
            from app.datasets.appearance_candidates import IMAGE_INVENTORY_POLICY
            if (packet.task.get('image_identity_policy') == 'official-projection-v1'
                    and packet.task.get('image_inventory_policy') != IMAGE_INVENTORY_POLICY):
                raise ValueError('Annotation predates complete image inventory; prepare and annotate again')
            if not packet.scene:
                raise ValueError('Audit requires complete official scenes')
            object_ids = sorted(p.object_id for p in packet.candidates)
            random.Random(digest([salt, packet.context, packet.frame_timestamp_micros])).shuffle(object_ids)
            mapping = {f'C{i+1:03d}': oid for i, oid in enumerate(object_ids)}
            from app.quality.evaluation import first_draft_outcome
            text = final_text(state) if spec.text_version == 'final' else state['attempts'][0]['generated']['expression']
            diagnostic = bool(state.get('diagnostic_pass')) if spec.text_version == 'final' else first_draft_outcome(state) is True
            cases.append({'case_id': digest([salt, state['run_id']])[:24], 'run_id': state['run_id'],
                          'run_sha256': digest(state), 'expression': text,
                          'packet': packet.model_dump(mode='json'), 'mapping': mapping,
                          'expected': next(k for k,v in mapping.items() if v == packet.target_object_id),
                          'diagnostic_pass': diagnostic})
        frozen = {'protocol': PROTOCOL, 'input_capabilities': ['geometry', 'rgb'], 'method_sha256': batch['method_sha256'],
                  'method_configuration': configuration,
                  'contexts': contexts, 'cases': cases, 'workload': len(workload),
                  'candidate_texts': len(states), 'eligible':sum(i['target']['status']=='selected' for i in workload), 'sampling': 'seeded simple random sample of all texts, including diagnostic failures',
                  'model_calls': sum(i.get('model_calls_used', 0) or 0 for i in workload),
                  'sampling_requested': spec.sample_size}
        with self.runs.connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='batches'").fetchone() and not db.execute('SELECT 1 FROM batches WHERE job_id=?',(spec.job_id,)).fetchone():
                raise ValueError('Source task was deleted; cannot create an audit')
            db.execute('INSERT OR IGNORE INTO audit_cohorts VALUES(?,?,?,?,?,NULL)',
                       (identity, json.dumps(spec.model_dump()), json.dumps(frozen), 'open', time.time()))
        return self.detail(identity)

    def verify_native_box(self,scene,packet):
        """Bind released supervision directly to the recorded raw native row."""
        import pyarrow.parquet as pq
        source=scene.source_components['lidar_box']
        path=checked_asset(raw_source_root(self.root,source),source['file'],source['sha256'])
        prefix='[LiDARBoxComponent]'
        coordinates=[f'{prefix}.box.center.{a}' for a in 'xyz']
        dimensions=[f'{prefix}.box.size.{a}' for a in 'xyz']
        heading=f'{prefix}.box.heading';category=f'{prefix}.type'
        rows=pq.read_table(path,columns=[*coordinates,*dimensions,heading,category],filters=[
            ('key.segment_context_name','=',packet.context),('key.frame_timestamp_micros','=',packet.frame_timestamp_micros),
            ('key.laser_object_id','=',packet.target_object_id)]).to_pylist()
        if len(rows)!=1:raise ValueError('Released native target key is missing or ambiguous in the raw source')
        row=rows[0];expected=[*packet.target.center_m,*packet.target.size_m,packet.target.heading_rad]
        values=[float(row[k]) for k in [*coordinates,*dimensions,heading]]
        label={1:'vehicle',2:'pedestrian',3:'sign',4:'cyclist'}.get(row[category],'unknown')
        if values!=expected or label!=packet.target.category:
            raise ValueError('Released box/category differs from the raw native target row')

    def private(self, cohort_id):
        with self.runs.connect() as db:
            row = db.execute('SELECT * FROM audit_cohorts WHERE cohort_id=?', (cohort_id,)).fetchone()
            answers = {r['case_id']: json.loads(r['answer']) for r in db.execute('SELECT * FROM audit_answers WHERE cohort_id=?', (cohort_id,))}
        if row is None:
            raise ValueError('Audit cohort not found')
        return {**dict(row), 'spec': json.loads(row['spec']), 'frozen': json.loads(row['frozen']), 'answers': answers,
                'report': json.loads(row['report']) if row['report'] else None}

    def detail(self, cohort_id):
        cohort = self.private(cohort_id)
        frozen = cohort['frozen']
        return {'cohort_id': cohort_id, 'status': cohort['status'], 'spec': cohort['spec'],
                'protocol': frozen['protocol'], 'contexts': frozen['contexts'], 'workload': frozen['workload'],
                'candidate_texts': frozen['candidate_texts'], 'sampled': len(frozen['cases']),
                'reviewed': len(cohort['answers']), 'cases': [{'case_id': c['case_id'], 'reviewed': c['case_id'] in cohort['answers']} for c in frozen['cases']],
                'report': cohort['report']}

    def case(self, cohort_id, case_id):
        cohort = self.private(cohort_id)
        case = next((c for c in cohort['frozen']['cases'] if c['case_id'] == case_id), None)
        if not case:
            raise ValueError('Audit case not found')
        # Do not serve target identity, generation predicates, target crops, API packet
        # URLs, target-centered point sampling, or correctness feedback before sealing.
        return case, cohort

    def localization(self, cohort_id, case_id):
        with self.runs.connect() as db:
            row = db.execute('SELECT selected_labels FROM audit_localizations WHERE cohort_id=? AND case_id=?',(cohort_id,case_id)).fetchone()
        return json.loads(row['selected_labels']) if row else None

    @serialized
    def lock_localization(self, cohort_id, case_id, labels):
        case,cohort = self.case(cohort_id,case_id)
        if cohort['status'] != 'open' or len(set(labels)) != len(labels) or set(labels)-set(case['mapping']):
            raise ValueError('Invalid or closed blind localization')
        labels=sorted(labels)
        with self.runs.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT status FROM audit_cohorts WHERE cohort_id=?',(cohort_id,)).fetchone()['status'] != 'open':
                raise ValueError('Audit was sealed concurrently')
            row=db.execute('SELECT selected_labels FROM audit_localizations WHERE cohort_id=? AND case_id=?',(cohort_id,case_id)).fetchone()
            if row and json.loads(row['selected_labels'])!=labels:
                raise ValueError('Confirmed localization cannot be changed')
            db.execute('INSERT OR IGNORE INTO audit_localizations VALUES(?,?,?,?)',(cohort_id,case_id,json.dumps(labels),time.time()))
        return {'locked':True,'selected_labels':labels}

    @serialized
    def submit(self, cohort_id, case_id, answer):
        case, cohort = self.case(cohort_id, case_id)
        if cohort['status'] != 'open':
            raise ValueError('Sealed reports cannot be edited')
        if set(answer.selected_labels) - set(case['mapping']):
            raise ValueError('Selection is outside this blind candidate vocabulary')
        if self.localization(cohort_id,case_id) != sorted(answer.selected_labels):
            raise ValueError('Confirm the localization selection before saving the audit')
        criterion = 'input_sufficient' if cohort['frozen']['protocol'] == PROTOCOL else 'geometry_sufficient'
        if getattr(answer, criterion) is None:
            raise ValueError(f'This frozen protocol requires {criterion}')
        encoded = answer.model_dump_json()
        with self.runs.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT status FROM audit_cohorts WHERE cohort_id=?', (cohort_id,)).fetchone()['status'] != 'open':
                raise ValueError('Audit was sealed concurrently')
            db.execute('INSERT OR REPLACE INTO audit_answers VALUES(?,?,?,?)', (cohort_id, case_id, encoded, time.time()))
            db.execute('INSERT INTO audit_answer_events(cohort_id,case_id,answer,at) VALUES(?,?,?,?)', (cohort_id,case_id,encoded,time.time()))
        return {'saved': True, 'scored_feedback_hidden_until_sealed': True}

    def compute(self, cohort):
        accepted, correct, diagnostic_accepted, diagnostic_n = 0, 0, 0, 0
        reviewed, exposed, rows = 0, 0, []
        for case in cohort['frozen']['cases']:
            answer = cohort['answers'].get(case['case_id'])
            if not answer:
                continue
            if answer['prior_target_exposure']:
                exposed += 1
                continue
            reviewed += 1
            matched = answer['selected_labels'] == [case['expected']]
            criterion = 'input_sufficient' if cohort['frozen']['protocol'] == PROTOCOL else 'geometry_sufficient'
            passed = matched and all(answer.get(k) is True for k in ['factual_correct',criterion,'alignment_correct'])
            correct += matched
            accepted += passed
            diagnostic_n += case['diagnostic_pass']
            diagnostic_accepted += passed and case['diagnostic_pass']
            rows.append({'case_id': case['case_id'], 'run_id': case['run_id'], 'correct_target_and_unique': matched, 'expression_sha256':hashlib.sha256(case['expression'].encode()).hexdigest(),
                         'diagnostic_pass':case['diagnostic_pass'],
                         'packet_id':case['packet']['packet_id'],'scene_sha256':case['packet']['scene']['manifest_sha256'],
                         'target_id':case['packet']['target_object_id'],'target_box':{k:case['packet']['target'][k] for k in ['center_m','size_m','heading_rad']},
                         'accepted': passed, 'factual_correct': answer['factual_correct'],
                         criterion: answer[criterion], 'alignment_correct': answer['alignment_correct']})
        acceptance = wilson(accepted, reviewed)
        risk = wilson(diagnostic_n - diagnostic_accepted, diagnostic_n)
        spec, frozen = cohort['spec'], cohort['frozen']
        full = len(cohort['answers']) == len(frozen['cases']) and exposed == 0
        gate = full and diagnostic_n >= spec['min_reviewed'] and risk['ci95'] is not None and risk['ci95'][1] <= spec['max_error_upper95']
        return {'protocol': frozen['protocol'], 'input_capabilities': frozen.get('input_capabilities', ['geometry']),
                'scope': 'human blind re-grounding among native GT candidates; not free-box 3DVG model accuracy',
                'sampled': len(frozen['cases']), 'reviewed_independent': reviewed, 'prior_exposure_excluded': exposed,
                'complete': full, 'human_joint_acceptance': acceptance, 'blind_unique_target_accuracy': wilson(correct,reviewed),
                'automatic_candidate_error': risk, 'calibration_gate_passed': gate,
                'policy': {'min_reviewed': spec['min_reviewed'], 'max_error_upper95': spec['max_error_upper95']},
                'contexts': frozen['contexts'], 'method_sha256': frozen['method_sha256'],
                'method_configuration': frozen.get('method_configuration'),
                'candidate_coverage': {'numerator': frozen['candidate_texts'], 'denominator': frozen['eligible'],
                                       'value': frozen['candidate_texts']/frozen['eligible'] if frozen['eligible'] else None},
                'model_calls': frozen['model_calls'], 'sampling': frozen['sampling'], 'cases': rows,
                'generalization_limit': 'Only the audited contexts and categories are observed; two local contexts cannot establish broad production reliability.'}

    @serialized
    def seal(self, cohort_id):
        cohort = self.private(cohort_id)
        if cohort['status'] == 'sealed':
            return self.detail(cohort_id)
        if len(cohort['answers']) != len(cohort['frozen']['cases']):
            raise ValueError('Every sampled case must be answered before sealing')
        for case in cohort['frozen']['cases']:
            if digest(self.runs.snapshot(case['run_id'])) != case['run_sha256']:
                raise ValueError('Frozen annotation version changed during the audit')
        with self.runs.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # Re-read answers under the seal transaction, preventing a last-moment edit.
            cohort['answers'] = {r['case_id']:json.loads(r['answer']) for r in db.execute('SELECT * FROM audit_answers WHERE cohort_id=?',(cohort_id,))}
            report = self.compute(cohort)
            db.execute("UPDATE audit_cohorts SET status='sealed',report=? WHERE cohort_id=? AND status='open'",(json.dumps(report),cohort_id))
        return self.detail(cohort_id)

    def list(self):
        with self.runs.connect() as db:
            ids = [r['cohort_id'] for r in db.execute('SELECT cohort_id FROM audit_cohorts ORDER BY created_at DESC')]
        return [self.detail(id_) for id_ in ids]

    @serialized
    def release(self, calibration_id, evaluation_id):
        calibration, evaluation = self.private(calibration_id), self.private(evaluation_id)
        if calibration['spec']['purpose'] != 'calibration' or evaluation['spec']['purpose'] != 'evaluation':
            raise ValueError('Release requires separately declared calibration and evaluation cohorts')
        for cohort in [calibration,evaluation]:
            if cohort['status'] != 'sealed' or not cohort['report']['calibration_gate_passed']:
                raise ValueError('Independent sealed audit does not satisfy the declared uncertainty bound')
            policy=cohort['report']['policy']
            if (policy['min_reviewed'] < RELEASE_MIN_REVIEWED or policy['max_error_upper95'] > RELEASE_MAX_ERROR_UPPER95
                    or cohort['spec'].get('text_version','final') != 'final'):
                raise ValueError('Training release requires final text and the fixed minimum 30 reviews / 10% error upper bound; exploratory policies cannot publish')
            validate_release_report(cohort['report'])
            if (cohort['report']['method_configuration'] != cohort['frozen'].get('method_configuration')
                    or cohort['report']['method_sha256'] != cohort['frozen']['method_sha256']):
                raise ValueError('Release report differs from its frozen annotation method')
        if set(calibration['frozen']['contexts']) & set(evaluation['frozen']['contexts']):
            raise ValueError('Calibration and evaluation contexts must be disjoint')
        if calibration['frozen']['method_sha256'] != evaluation['frozen']['method_sha256']:
            raise ValueError('Audit methods differ; cannot share a release certificate')
        if calibration['frozen']['method_configuration'] != evaluation['frozen']['method_configuration']:
            raise ValueError('Audit model or execution configurations differ; cannot share a release certificate')
        if calibration['frozen']['protocol'] != evaluation['frozen']['protocol']:
            raise ValueError('Audit input protocols differ; cannot share a release certificate')
        if calibration['report']['policy'] != evaluation['report']['policy']:
            raise ValueError('Calibration/evaluation release policies differ')
        # Conservative first release: only independently audited accepted calibration
        # cases, never the held-out evaluation cases or unreviewed extrapolations.
        approved = {r['case_id'] for r in calibration['report']['cases'] if r['accepted']}
        selected = [c for c in calibration['frozen']['cases'] if c['case_id'] in approved and c['diagnostic_pass']]
        from app.agent.harness import expression_policy_errors
        if any(expression_policy_errors(case['expression']) for case in selected):
            raise ValueError('描述不符合当前文本规则（句首大写、无具体距离、无最近或最远排名）；请重新生成并检查后再发布训练数据。')
        from app.quality.contrast import unsupported_ego_directions
        for case in selected:
            source = self.runs.snapshot(case['run_id'])
            ids = (source.get('attempts') or [{}])[-1].get('generated', {}).get('predicate_ids', [])
            if unsupported_ego_directions(EvidencePacket.model_validate(case['packet']).target, ids):
                raise ValueError('描述中的自车方位缺少完整原生框几何支持，请重新核验后再发布训练数据。')
        release_id = digest([calibration_id,evaluation_id,calibration['report'],evaluation['report']])[:24]
        folder = self.root/'runtime'/'releases'/release_id
        folder.mkdir(parents=True,exist_ok=True)
        report_refs = {}
        for name,cohort in [('calibration',calibration),('validation',evaluation)]:
            path = folder/(name+'.json')
            certificate=cohort['report'] if name=='calibration' else {k:v for k,v in cohort['report'].items() if k!='cases'}
            if name=='validation': certificate={**certificate,'case_gold_withheld_from_training_package':True,'full_report_sha256':digest(cohort['report'])}
            atomic_json(path,certificate)
            report_refs[name] = {'path':path.relative_to(self.root).as_posix(),'sha256':file_sha256(path)}
        from app.schemas.training_contracts import TrainingSample, TaskProfile, QualityCertificate, NativeBox
        records = []
        for case in selected:
            state = self.runs.snapshot(case['run_id'])
            from app.quality.sample_review import review_status
            if review_status(state) in {'rejected', 'discarded'}:
                raise ValueError('人工审核不通过或已废弃的样本不能发布。')
            if digest(state) != case['run_sha256'] or state.get('provider_scope') != 'real-model':
                raise ValueError('Release source annotation changed')
            packet = EvidencePacket.model_validate(case['packet'])
            scene_path = checked_asset(self.root,packet.scene.manifest_path,packet.scene.manifest_sha256)
            scene,arrays = read_scene(scene_path.parent)
            from app.quality.geometry import inside_oriented_box
            observed=int(inside_oriented_box(arrays['points'][:,:3],packet.target.center_m,packet.target.size_m,packet.target.heading_rad).sum())
            if (scene.context,scene.frame_timestamp_micros)!=(packet.context,packet.frame_timestamp_micros) or scene.point_count!=packet.scene.total_points or observed<=0 or observed!=packet.scene.target_points:
                raise ValueError('Released scene/target identity or point observability differs from audit')
            # Every essential landmark must also be visible in the actual LiDAR
            # input. A GT box or label count alone is not this measurement.
            predicates={p.predicate_id:p for p in packet.predicates}
            anchors={a['anchor_id']:a['object_id'] for tool in state.get('tool_results',[]) for a in tool.get('anchors',[])}
            objects={o.object_id:o for o in packet.candidates}
            used=(state.get('attempts') or [{}])[-1].get('generated',{}).get('predicate_ids',[])
            for id_ in used:
                rule=predicates.get(id_)
                if rule and rule.kind in {'relative_left','relative_right','ahead_of','behind_anchor','vehicle_heading','nearest_to_anchor'}:
                    anchor_id=rule.value.split(':')[-1]
                    landmark=objects.get(anchors.get(anchor_id))
                    if landmark is None or not landmark.geometry_valid or not inside_oriented_box(arrays['points'][:,:3],landmark.center_m,landmark.size_m,landmark.heading_rad).any():
                        raise ValueError('An essential landmark has no verified points in the released input')
            required={'lidar','lidar_pose','lidar_calibration','lidar_camera_projection','vehicle_pose','camera_image','camera_calibration','lidar_box'}
            if not required<=scene.source_components.keys() or scene.conversion.get('implementation')!='official-waymo-sdk':
                raise ValueError('Release requires complete official conversion and raw-source provenance')
            self.verify_native_box(scene,packet)
            for source in scene.source_components.values():
                checked_asset(raw_source_root(self.root,source),source['file'],source['sha256'])
            for name,hash_ in packet.provenance.get('artifact_sha256',{}).items():
                checked_asset(self.root/'runtime'/'evidence'/packet.packet_id,name,hash_)
            attempt = (state.get('attempts') or [{}])[-1]
            visual = attempt.get('generated', {}).get('visual_attributes', [])
            if visual and not attempt.get('appearance_check', {}).get('passed'):
                raise ValueError('Appearance claims require a saved image verification before release')
            capabilities = ('geometry', 'rgb') if visual or calibration['frozen']['protocol'] == PROTOCOL else ('geometry',)
            record = TrainingSample(sample_id=digest([case['run_sha256'],release_id])[:24],context=packet.context,
                frame_timestamp_micros=packet.frame_timestamp_micros,track_id=packet.target_object_id,category=packet.target.category,
                official_split=scene.official_split,scene={'path':packet.scene.manifest_path,'sha256':packet.scene.manifest_sha256},
                expression=case['expression'],task=TaskProfile(input_capabilities=capabilities),required_capabilities=capabilities,
                source_view=packet.source_view, source_camera_object_id=packet.source_camera_object_id,
                target_box=NativeBox(center_m=packet.target.center_m,size_m=packet.target.size_m,heading_rad=packet.target.heading_rad),
                annotation_run_id=case['run_id'],evidence_packet_id=packet.packet_id,
                quality=QualityCertificate(protocol='independent-blind-audited-v1',oracle_checks_passed=True,
                    counterfactual_checks_passed=(state.get('attempts') or [{}])[-1].get('counterfactual',{}).get('passed'),
                    source_alignment_verified=True,independent_risk_calibration=True,
                    calibration_report=report_refs['calibration'],validation_report=report_refs['validation']))
            records.append(record.model_dump(mode='json'))
        if not records:
            raise ValueError('No independently audited accepted calibration cases to release')
        output = folder/'samples.jsonl'
        content = ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records)
        temporary = output.with_suffix('.tmp'); temporary.write_text(content,encoding='utf-8'); temporary.replace(output)
        manifest = {'release_id':release_id,'protocol':calibration['frozen']['protocol'],'records':len(records),'run_ids':[c['run_id'] for c in selected],
                    'cohort_ids':{'calibration':calibration_id,'evaluation':evaluation_id},
                    'scope':'independently audited calibration cases only; evaluation contexts excluded from training',
                    'samples':{'path':output.relative_to(self.root).as_posix(),'sha256':file_sha256(output)},
                    'calibration':report_refs['calibration'],'validation':report_refs['validation'],
                    'evaluation_contexts_excluded':evaluation['frozen']['contexts']}
        atomic_json(folder/'manifest.json',manifest)
        with self.runs.connect() as db:
            db.execute('INSERT OR IGNORE INTO label_releases VALUES(?,?,?)',(release_id,json.dumps(manifest),time.time()))
        return manifest

    def job_metrics(self, job_id):
        # Never pool duplicate reviews, different protocols, exploratory arms,
        # or differently sampled cohorts into a fabricated common denominator.
        cohorts=[c for c in self.list() if c['spec']['job_id']==job_id and c['spec'].get('text_version','final')=='final']
        sealed=[c for c in cohorts if c['status']=='sealed']
        return {'cohorts':[{'cohort_id':c['cohort_id'],'purpose':c['spec']['purpose'],'status':c['status'],
            'reviewed':c['reviewed'],'sampled':c['sampled'],'report':c['report']} for c in cohorts],
            'scope':'separate frozen cohorts; no pooling repeated judgments',
            'acceptance':sealed[0]['report']['human_joint_acceptance'] if len(sealed)==1 else None}

    def compare_repair(self, first_id, final_id):
        first, final = self.private(first_id), self.private(final_id)
        if any(c['status']!='sealed' for c in (first,final)):
            raise ValueError('Both blind reports must be sealed')
        if first['spec'].get('text_version','final')!='first_draft' or final['spec'].get('text_version','final')!='final':
            raise ValueError('Compare the first draft against final text')
        if first['frozen']['method_sha256']!=final['frozen']['method_sha256'] or first['spec']['job_id']!=final['spec']['job_id']:
            raise ValueError('Repair comparison requires the same frozen job and method')
        if (not first['frozen'].get('method_configuration')
                or first['frozen']['method_configuration'] != final['frozen'].get('method_configuration')):
            raise ValueError('Repair comparison requires the same recorded model and execution configuration')
        if first['frozen']['protocol'] != final['frozen']['protocol']:
            raise ValueError('Repair comparison requires the same audit input protocol')
        a={r['run_id']:r for r in first['report']['cases']};b={r['run_id']:r for r in final['report']['cases']}
        if not a or set(a)!=set(b) or not first['report']['complete'] or not final['report']['complete']:
            raise ValueError('A paired comparison requires the same completely and independently reviewed targets')
        reviewers=lambda c:{v['reviewer'].strip().casefold() for v in c['answers'].values()}
        if reviewers(first)&reviewers(final):
            raise ValueError('Use different evaluators for the two text versions to avoid target carryover')
        transitions=Counter((a[id_]['accepted'],b[id_]['accepted']) for id_ in a)
        return {'scope':'paired human review of saved first/final texts; repair-only comparison, not a one-shot VLM baseline or downstream training gain',
            'first_cohort_id':first_id,'final_cohort_id':final_id,'method_sha256':first['frozen']['method_sha256'],
            'targets':len(a),'first_acceptance':first['report']['human_joint_acceptance'],
            'final_acceptance':final['report']['human_joint_acceptance'],
            'improved':transitions[(False,True)],'regressed':transitions[(True,False)],
            'both_accepted':transitions[(True,True)],'both_rejected':transitions[(False,False)],
            'acceptance_difference':(transitions[(False,True)]-transitions[(True,False)])/len(a)}


    @serialized
    def package(self, release_id):
        """Self-contained complete scene inputs, labels and audit certificates."""
        import zipfile
        with self.runs.connect() as db:
            row=db.execute('SELECT manifest FROM label_releases WHERE release_id=?',(release_id,)).fetchone()
        if not row:
            raise ValueError('Release not found')
        manifest=json.loads(row['manifest'])
        sample_ref=manifest['samples']
        sample_path=checked_asset(self.root,sample_ref['path'],sample_ref['sha256'])
        refs={sample_ref['path']:sample_ref['sha256']}
        for name in ['calibration','validation']:
            refs[manifest[name]['path']]=manifest[name]['sha256']
        for line in sample_path.read_text(encoding='utf-8').splitlines():
            record=json.loads(line);scene_ref=record['scene']
            from app.quality.sample_review import review_status
            if review_status(self.runs.snapshot(record['annotation_run_id'])) in {'rejected', 'discarded'}:
                raise ValueError('发布包中包含人工审核不通过或已废弃的版本，请重新审核并生成发布包。')
            scene_path=checked_asset(self.root,scene_ref['path'],scene_ref['sha256'])
            scene,_=read_scene(scene_path.parent)
            refs[scene_ref['path']]=scene_ref['sha256']
            for ref in [scene.points,scene.calibration,*(c.image for c in scene.cameras)]:
                relative=(scene_path.parent/ref.path).relative_to(self.root).as_posix()
                refs[relative]=ref.sha256
        folder=sample_path.parent
        output=folder/'training-package.zip';temporary=output.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
            for relative,sha in sorted(refs.items()):
                path=checked_asset(self.root,relative,sha)
                data=path.read_bytes()
                archive.writestr(relative,data)
            archive.writestr('release-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            archive.writestr('README.txt','Load samples relative to this extracted directory. Complete scene point clouds and five camera images are included. Evaluation case gold is withheld. Candidate proposals were independently audited; this is not a free-box 3DVG benchmark score.')
        temporary.replace(output)
        return output
