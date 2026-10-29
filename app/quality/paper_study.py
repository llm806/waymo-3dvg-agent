"""Prospective paired experiments. Human judgments never alter training labels."""
import csv
import io
import json
import random
import secrets
import time
from pathlib import Path

from pydantic import Field, model_validator

from app.schemas.contracts import Contract, EvidencePacket
from app.quality.research import digest, final_text, wilson

PROTOCOL = 'prospective-paired-single-frame-human-v1'
BASELINE = 'fixed-vlm-one-request-v1'


class StudySpec(Contract):
    request_id: str = Field(pattern=r'^[a-f0-9]{24}$')
    title: str = Field(min_length=1, max_length=100)
    sample_size: int = Field(default=200, ge=1, le=500)
    seed: int = Field(default=42, ge=0, le=2147483647)
    dataset_split: str = Field(default='validation', pattern=r'^(train|validation)$')
    reviewers: list[str] = Field(min_length=3, max_length=3)
    input_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_price_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def people(self):
        self.title = self.title.strip()
        self.reviewers = [r.strip() for r in self.reviewers]
        if not self.title or any(not r or len(r)>80 for r in self.reviewers) or len(set(self.reviewers))!=3:
            raise ValueError('请填写三个不同的审核者标识。前两人独立审核，第三人裁定。')
        return self


class Judgment(Contract):
    factual_correct: bool
    unique: bool
    input_sufficient: bool
    prior_target_exposure: bool
    notes: str = Field(default='', max_length=2000)
    elapsed_seconds: float = Field(ge=0, le=86400, allow_inf_nan=False)


def exists(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='paper_studies'").fetchone())


def reserved_for_other(runs, job, target, *, claim=False):
    with runs.connect() as db:
        if not exists(db): return False
        if claim:db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT s.job_id FROM paper_targets t JOIN paper_studies s ON s.study_id=t.study_id WHERE t.context=? AND t.object_id=?',
            (target['context'],target['object_id'])).fetchone()
        reserved=bool(row and row['job_id'] != job)
        if claim and not reserved:
            db.execute("UPDATE batch_items SET status='preparing' WHERE job_id=? AND json_extract(target,'$.target_id')=? AND status='pending' AND run_id IS NULL",
                (job,target['target_id']))
        return reserved


def study_for_job(runs, job):
    with runs.connect() as db:
        if not exists(db): return None
        row=db.execute('SELECT study_id FROM paper_studies WHERE job_id=?',(job,)).fetchone()
        return row['study_id'] if row else None


def baseline_messages(packet, root):
    # Exactly the initial target observation's user payload/images; no acquired
    # observations, later frames, generated drafts or Agent verification results.
    from app.agent.harness import appearance_messages
    messages=appearance_messages(packet)
    content=messages[1]['content']
    initial=json.loads(content[0]['text'])
    initial.pop('output_schema',None)
    initial.update(target=packet.target.model_dump(),candidates=[o.model_dump() for o in packet.candidates],
        source_view=packet.source_view,coordinate_frame='ego: x forward, y left, z up',
        output_instruction='Return one expression string in JSON. Do not return an observation record.')
    content[0]={'type':'text','text':json.dumps(initial)}
    return [{'role':'system','content':
        'Write one natural English referring expression identifying the designated target. '
        'Use only visible facts and the supplied native geometry. It must distinguish the target '
        'from the other objects and be understandable from the single source frame. '
        'Do not claim motion or invent attributes. Return JSON with expression only. '
        'If no grounded expression is possible, return an empty expression.'}, *messages[1:]]


class PaperStudyStore:
    def __init__(self, runs, root):
        self.runs,self.root=runs,Path(root)
        with runs.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS paper_studies(study_id TEXT PRIMARY KEY,job_id TEXT UNIQUE,spec TEXT NOT NULL,
                method_sha256 TEXT NOT NULL,status TEXT NOT NULL,created_at REAL NOT NULL,report TEXT);
            CREATE TABLE IF NOT EXISTS paper_targets(context TEXT NOT NULL,object_id TEXT NOT NULL,study_id TEXT NOT NULL,
                target_id TEXT NOT NULL,PRIMARY KEY(context,object_id));
            CREATE TABLE IF NOT EXISTS paper_trials(study_id TEXT NOT NULL,ordinal INTEGER NOT NULL,initial_packet TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,agent_runs TEXT NOT NULL,baseline_run TEXT,PRIMARY KEY(study_id,ordinal));
            CREATE TABLE IF NOT EXISTS paper_reviewers(token TEXT PRIMARY KEY,study_id TEXT NOT NULL,role INTEGER NOT NULL,name TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS paper_cases(case_id TEXT PRIMARY KEY,study_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
                method TEXT NOT NULL,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS paper_reviews(case_id TEXT NOT NULL,token TEXT NOT NULL,selection TEXT,answer TEXT,
                updated_at REAL NOT NULL,localization_seconds REAL NOT NULL DEFAULT 0,PRIMARY KEY(case_id,token));
            ''')
            if 'localization_seconds' not in {r['name'] for r in db.execute('PRAGMA table_info(paper_reviews)')}:
                db.execute('ALTER TABLE paper_reviews ADD COLUMN localization_seconds REAL NOT NULL DEFAULT 0')

    def private(self, sid):
        with self.runs.connect() as db:
            row=db.execute('SELECT * FROM paper_studies WHERE study_id=?',(sid,)).fetchone()
        if not row: raise ValueError('科研实验不存在。')
        return {**dict(row),'spec':json.loads(row['spec']),'report':json.loads(row['report']) if row['report'] else None}

    def create(self, spec, service):
        from app.agent.batch import BatchSpec,method_fingerprint
        from app.datasets.catalog import load_catalog
        sid=spec.request_id
        with self.runs.maintenance():
            with self.runs.connect() as db:
                old=db.execute('SELECT spec,job_id,status FROM paper_studies WHERE study_id=?',(sid,)).fetchone()
            if old:
                if json.loads(old['spec'])!=spec.model_dump(): raise ValueError('重复请求的实验设置不同。')
                if old['job_id']:return self.detail(sid,service)
            service.ensure_loaded_code()
            catalog=load_catalog()
            official='training' if spec.dataset_split=='train' else 'validation'
            if old:
                with self.runs.connect() as db:
                    selected_ids={r['target_id'] for r in db.execute('SELECT target_id FROM paper_targets WHERE study_id=?',(sid,))}
                chosen=[t for t in catalog['targets'] if t['target_id'] in selected_ids]
                if len(chosen)!=spec.sample_size:raise ValueError('中断创建时的冻结候选已变化。')
                job,_=service.create(BatchSpec(catalog_id=catalog['catalog_id'],dataset_split=spec.dataset_split,
                    target_ids=[t['target_id'] for t in chosen],prepare_only=False,desired_count=None,calls_per_target=12,max_repairs=2))
                with self.runs.connect() as db:
                    db.execute("UPDATE paper_studies SET job_id=?,status='ready' WHERE study_id=?",(job,sid))
                    db.executemany('INSERT INTO paper_reviewers VALUES(?,?,?,?)',[(secrets.token_hex(24),sid,i,name) for i,name in enumerate(spec.reviewers)])
                return self.detail(sid,service)
            # Reserve under the SQLite write lock. Preparing/running jobs and all
            # historical attempts are excluded; unattempted queue reserves may be claimed.
            with self.runs.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                unavailable={(r['context'],r['object_id']) for r in db.execute('SELECT context,object_id FROM paper_targets')}
                for row in db.execute('SELECT target,status,run_id,packet_id FROM batch_items'):
                    if row['run_id'] or row['packet_id'] or row['status'] not in {'pending','not_needed'}:
                        t=json.loads(row['target']);unavailable.add((t['context'],t['object_id']))
                # Include independent runs and deleted-task exclusion/accounting.
                unavailable.update((r['context'],r['object_id']) for r in db.execute("""SELECT
                    COALESCE(json_extract(snapshot,'$.current_packet.context'),json_extract(snapshot,'$.context')) AS context,
                    json_extract(snapshot,'$.current_packet.target_object_id') AS object_id FROM run_summaries"""))
                unavailable |= self.runs.excluded_target_keys()
                rows=[t for t in catalog['targets'] if t['status']=='selected' and t.get('official_split')==official
                    and (t['context'],t['object_id']) not in unavailable]
                rows.sort(key=lambda t:t['target_id']);random.Random(spec.seed).shuffle(rows)
                if len(rows)<spec.sample_size: raise ValueError(f'未处理的可用任务只有 {len(rows)} 个，请减少样本量或更换官方划分。')
                chosen=rows[:spec.sample_size]
                db.execute('INSERT INTO paper_studies VALUES(?,NULL,?,?,?, ?,NULL)',
                    (sid,json.dumps(spec.model_dump()),method_fingerprint(),'creating',time.time()))
                db.executemany('INSERT INTO paper_targets VALUES(?,?,?,?)',
                    [(t['context'],t['object_id'],sid,t['target_id']) for t in chosen])
            try:
                job,_=service.create(BatchSpec(catalog_id=catalog['catalog_id'],dataset_split=spec.dataset_split,
                    target_ids=[t['target_id'] for t in chosen],prepare_only=False,desired_count=None,calls_per_target=12,max_repairs=2))
            except Exception:
                with self.runs.connect() as db:
                    db.execute('DELETE FROM paper_targets WHERE study_id=?',(sid,));db.execute('DELETE FROM paper_studies WHERE study_id=?',(sid,))
                raise
            with self.runs.connect() as db:
                db.execute("UPDATE paper_studies SET job_id=?,status='ready' WHERE study_id=?",(job,sid))
                db.executemany('INSERT INTO paper_reviewers VALUES(?,?,?,?)',
                    [(secrets.token_hex(24),sid,i,name) for i,name in enumerate(spec.reviewers)])
            return self.detail(sid,service)

    def list(self, service):
        with self.runs.connect() as db: ids=[r['study_id'] for r in db.execute('SELECT study_id FROM paper_studies ORDER BY created_at DESC')]
        return [self.detail(sid,service) for sid in ids]

    def detail(self,sid,service):
        s=self.private(sid)
        with self.runs.connect() as db:
            reviewers=[dict(r) for r in db.execute('SELECT token,role,name FROM paper_reviewers WHERE study_id=? ORDER BY role',(sid,))]
            cases=db.execute('SELECT COUNT(*) FROM paper_cases WHERE study_id=?',(sid,)).fetchone()[0]
            for r in reviewers:
                r['localized']=db.execute('SELECT COUNT(*) FROM paper_reviews WHERE token=? AND selection IS NOT NULL',(r['token'],)).fetchone()[0]
                r['answered']=db.execute('SELECT COUNT(*) FROM paper_reviews WHERE token=? AND answer IS NOT NULL',(r['token'],)).fetchone()[0]
                r['url']='#research?review='+r.pop('token')
        batch=service.store.execution_progress(s['job_id']) if s['job_id'] else {}
        if s['job_id']:
            with self.runs.connect() as db:
                header=db.execute('SELECT status,control,reason FROM batches WHERE job_id=?',(s['job_id'],)).fetchone()
                batch.update(dict(header))
                batch['finished']=db.execute("SELECT COUNT(*) FROM batch_items WHERE job_id=? AND status NOT IN ('pending','running','preparing')",(s['job_id'],)).fetchone()[0]
        return {k:s[k] for k in ('study_id','job_id','spec','status','created_at','report')} | {
            'protocol':PROTOCOL,'baseline':BASELINE,'cases':cases,'reviewers':reviewers,
            'batch':batch,'method_sha256':s['method_sha256']}

    def ensure_method(self, job):
        from app.agent.batch import method_fingerprint
        sid=study_for_job(self.runs,job)
        if sid and self.private(sid)['method_sha256']!=method_fingerprint():
            raise ValueError('科研实验冻结后代码已变化；不能混合方法继续本实验。请保留本次记录并新建实验。')

    def bind(self,sid,ordinal,packet,run_id):
        from app.datasets.evidence import EVIDENCE_ROOT
        input_hash=digest(baseline_messages(packet,EVIDENCE_ROOT)[1:])
        with self.runs.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM paper_trials WHERE study_id=? AND ordinal=?',(sid,ordinal)).fetchone()
            if row:
                if row['input_sha256']!=input_hash: raise ValueError('配对实验的初始输入发生变化，已阻止继续。')
                runs=json.loads(row['agent_runs'])
                if run_id not in runs:runs.append(run_id)
                db.execute('UPDATE paper_trials SET agent_runs=? WHERE study_id=? AND ordinal=?',(json.dumps(runs),sid,ordinal))
            else:
                db.execute('INSERT INTO paper_trials VALUES(?,?,?,?,?,NULL)',
                    (sid,ordinal,packet.model_dump_json(),input_hash,json.dumps([run_id])))

    def baseline(self, execution, sid, ordinal, calls):
        from app.agent.agent_runtime import initial_run,run_agent
        with self.runs.connect() as db: trial=dict(db.execute('SELECT * FROM paper_trials WHERE study_id=? AND ordinal=?',(sid,ordinal)).fetchone())
        packet=EvidencePacket.model_validate_json(trial['initial_packet'])
        packet.task['research_baseline_policy']=BASELINE
        initial=initial_run(packet,calls,0,1,13)
        initial.update(batch_id=execution.job,research_study_id=sid,research_method='baseline')
        primary=self.runs.snapshot(json.loads(trial['agent_runs'])[-1])
        for key in ('binding_target','preparation_source_reselection'):
            if key in primary:initial[key]=primary[key]
        rid=trial['baseline_run'] or initial['run_id']
        if not trial['baseline_run']:
            self.runs.create(initial)
            with self.runs.connect() as db: db.execute('UPDATE paper_trials SET baseline_run=? WHERE study_id=? AND ordinal=?',(rid,sid,ordinal))
        if execution.cancelled():
            self.runs.request_control(rid,'pause')
        else:
            self.runs.request_control(rid,'resume')
        return run_agent(packet,calls,0,execution.worker.root,1,13,resume_run_id=rid,
            executor_owner=execution.owner,model_slots=execution.worker.model_slots)

    def costs(self, run_ids, spec):
        calls=0;prompt=completion=0;known=True;seconds=0
        for rid in dict.fromkeys(run_ids):
            state=self.runs.snapshot(rid);calls+=state.get('model_calls_used',0);seconds+=state.get('elapsed_active_seconds',0)
            results=list((self.runs.path.parent/rid/'model').glob('*/result.json'))
            if len(results)<state.get('model_calls_used',0):known=False
            for path in results:
                usage=json.loads(path.read_text(encoding='utf-8')).get('usage') or {}
                if not isinstance(usage.get('prompt_tokens'),int) or not isinstance(usage.get('completion_tokens'),int):known=False
                prompt+=usage.get('prompt_tokens') or 0;completion+=usage.get('completion_tokens') or 0
        prices=(spec['input_price_per_million'],spec['output_price_per_million'])
        money=(prompt*prices[0]+completion*prices[1])/1e6 if known and all(v is not None for v in prices) else None
        return {'calls':calls,'prompt_tokens':prompt if known else None,'completion_tokens':completion if known else None,
            'estimated_cost':money,'generation_seconds':seconds,'generation_human_interventions':0}

    def freeze(self,sid,service):
        with self.runs.maintenance():
            s=self.private(sid)
            if s['status'] in {'review','sealed'}:return self.detail(sid,service)
            self.ensure_method(s['job_id'])
            batch=service.store.detail(s['job_id'])
            if batch['status'] not in {'completed','blocked'} or batch['executor_active']:
                raise ValueError('请等待全部固定任务结束；暂停或排队中的实验不能进入盲审。')
            rows=[]
            with self.runs.connect() as db:
                trials={r['ordinal']:dict(r) for r in db.execute('SELECT * FROM paper_trials WHERE study_id=?',(sid,))}
            for item in batch['items']:
                trial=trials.get(item['ordinal'])
                for method in ('agent','baseline'):
                    runs=(json.loads(trial['agent_runs']) if method=='agent' else [trial['baseline_run']] if trial['baseline_run'] else []) if trial else []
                    states=[self.runs.snapshot(rid) for rid in runs]
                    if any(t.get('provider_scope')!='real-model' for t in states):
                        raise ValueError('本地模拟调用不能纳入正式科研盲审。')
                    if any(t['status'] in {'running','paused','stopped'} or t.get('inflight') and t['status']=='running' for t in states):
                        raise ValueError('存在未结束的配对请求，不能冻结盲审。')
                    state=states[-1] if states else {}
                    text=final_text(state)
                    accepted=bool(state.get('diagnostic_pass') if method=='agent' else state.get('research_baseline_delivered'))
                    case={'expression':text,'accepted':accepted,'status':state.get('status',item['status']),
                        'reason':state.get('stop_reason',item.get('reason')),'packet':state.get('current_packet'),
                        'run_ids':runs,'input_sha256':trial['input_sha256'] if trial else None,
                        'target_id':item['target']['target_id'],'context':item['target']['context'],
                        'cost':self.costs(runs,s['spec'])}
                    if text and state.get('current_packet'):
                        p=EvidencePacket.model_validate(state['current_packet'])
                        ids=sorted(o.object_id for o in p.candidates);random.SystemRandom().shuffle(ids)
                        case['mapping']={f'C{i+1:03d}':oid for i,oid in enumerate(ids)}
                        case['expected']=next(k for k,v in case['mapping'].items() if v==p.target_object_id)
                    rows.append((secrets.token_hex(12),sid,item['ordinal'],method,json.dumps(case)))
            if len(rows)!=s['spec']['sample_size']*2:raise ValueError('固定任务数不符，不能改变实验分母。')
            with self.runs.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                db.executemany('INSERT INTO paper_cases VALUES(?,?,?,?,?)',rows)
                db.execute("UPDATE paper_studies SET status='review' WHERE study_id=?",(sid,))
            return self.detail(sid,service)

    def reviewer(self,token):
        with self.runs.connect() as db:r=db.execute('SELECT * FROM paper_reviewers WHERE token=?',(token,)).fetchone()
        if not r:raise ValueError('审核链接无效。')
        return dict(r)

    def records(self,sid):
        with self.runs.connect() as db:
            cases=[{**dict(r),'payload':json.loads(r['payload'])} for r in db.execute('SELECT * FROM paper_cases WHERE study_id=?',(sid,))]
            reviews={(r['case_id'],r['token']):{'selection':json.loads(r['selection']) if r['selection'] is not None else None,
                'answer':json.loads(r['answer']) if r['answer'] else None,'localization_seconds':r['localization_seconds']} for r in db.execute('SELECT r.* FROM paper_reviews r JOIN paper_cases c ON c.case_id=r.case_id WHERE c.study_id=?',(sid,))}
            people=[dict(r) for r in db.execute('SELECT * FROM paper_reviewers WHERE study_id=? ORDER BY role',(sid,))]
        return cases,reviews,people

    @staticmethod
    def verdict(case,review):
        a=review.get('answer') or {}
        if not a or a.get('prior_target_exposure'):return None
        return review['selection']==[case['payload']['expected']] and all(a[k] for k in ('unique','factual_correct','input_sufficient'))

    @staticmethod
    def disagreement(case,first,second):
        fields=('unique','factual_correct','input_sufficient','prior_target_exposure')
        return first['selection']!=second['selection'] or any(first['answer'][k]!=second['answer'][k] for k in fields)

    def assigned(self,token):
        person=self.reviewer(token);s=self.private(person['study_id'])
        cases,reviews,people=self.records(s['study_id'])
        visible=[c for c in cases if c['payload']['expression']]
        complete=all(reviews.get((c['case_id'],p['token']),{}).get('answer') for c in visible for p in people[:2])
        if person['role']==2:
            visible=[c for c in visible if complete and self.disagreement(c,reviews[(c['case_id'],people[0]['token'])],reviews[(c['case_id'],people[1]['token'])])]
        # Block every target reveal until this evaluator has locked ALL
        # assigned localizations, preventing paired gold leakage into round two.
        localized=all(reviews.get((c['case_id'],token),{}).get('selection') is not None for c in visible)
        random.Random(int(digest(token),16)).shuffle(visible)
        return person,s,visible,reviews,localized,complete

    def inbox(self,token):
        person,s,cases,reviews,localized,complete=self.assigned(token)
        return {'title':s['spec']['title'],'reviewer':person['name'],'role':person['role'],'status':s['status'],
            'phase':'judgment' if localized else 'localization','waiting':person['role']==2 and not complete,
            'cases':[{'case_id':c['case_id'],'localized':reviews.get((c['case_id'],token),{}).get('selection') is not None,
                'answered':bool(reviews.get((c['case_id'],token),{}).get('answer'))} for c in cases]}

    def case(self,token,cid):
        person,s,cases,reviews,localized,_=self.assigned(token)
        c=next((c for c in cases if c['case_id']==cid),None)
        if c is None or s['status'] not in {'review','sealed'}:raise ValueError('当前审核者没有此条审核任务。')
        return c, reviews.get((cid,token),{}),localized

    def select(self,token,cid,labels,elapsed_seconds=0):
        with self.runs.maintenance():
            c,old,localized=self.case(token,cid)
            if self.private(c['study_id'])['status']=='sealed':raise ValueError('报告已封存。')
            if len(set(labels))!=len(labels) or not set(labels)<=c['payload']['mapping'].keys():raise ValueError('候选编号无效。')
            if old.get('selection') is not None:
                if old['selection']!=labels:raise ValueError('定位已锁定，不能在查看目标后修改。')
                return self.inbox(token)
            with self.runs.connect() as db:db.execute('INSERT INTO paper_reviews VALUES(?,?,?,NULL,?,?)',(cid,token,json.dumps(labels),time.time(),elapsed_seconds))
        return self.inbox(token)

    def answer(self,token,cid,answer):
        with self.runs.maintenance():
            c,old,localized=self.case(token,cid)
            if self.private(c['study_id'])['status']=='sealed':raise ValueError('报告已封存。')
            if not localized or old.get('selection') is None:raise ValueError('必须先锁定全部定位，才能核对目标及事实。')
            if old.get('answer') and old['answer']!=answer.model_dump():raise ValueError('已提交的独立判断不能覆写。')
            with self.runs.connect() as db:db.execute('UPDATE paper_reviews SET answer=?,updated_at=? WHERE case_id=? AND token=?',
                (json.dumps(answer.model_dump()),time.time(),cid,token))
        return self.inbox(token)

    def seal(self,sid,service):
        with self.runs.maintenance():
            s=self.private(sid)
            if s['status']=='sealed':return self.detail(sid,service)
            if s['status']!='review':raise ValueError('请先冻结生成结果并完成盲审。')
            cases,reviews,people=self.records(sid);rows=[];disputes=0
            for c in cases:
                data=c['payload'];judgments=[];effective=False;excluded=False
                if data['expression']:
                    for p in people[:2]:
                        review=reviews.get((c['case_id'],p['token']),{})
                        if not review.get('answer'):raise ValueError('两名审核者尚未完成全部独立判断。')
                        judgments.append({'reviewer':p['name'],**review})
                    conflict=self.disagreement(c,judgments[0],judgments[1]);disputes+=conflict
                    selected=judgments[0]
                    if conflict:
                        review=reviews.get((c['case_id'],people[2]['token']),{})
                        if not review.get('answer'):raise ValueError('分歧尚未由第三人裁定。')
                        selected={'reviewer':people[2]['name'],**review};judgments.append(selected)
                    excluded=any(j['answer']['prior_target_exposure'] for j in judgments)
                    effective=self.verdict(c,selected) if not excluded else None
                rows.append({'case_id':c['case_id'],'ordinal':c['ordinal'],'method':c['method'],
                    **{k:v for k,v in data.items() if k not in {'packet','mapping'}},
                    'judgments':judgments,'valid':effective,'exposure_excluded':excluded})
            metrics={}
            for method in ('agent','baseline'):
                items=[r for r in rows if r['method']==method];accepted=[r for r in items if r['accepted']]
                valid=sum(r['accepted'] and r['valid'] is True for r in items)
                excluded=sum(r['exposure_excluded'] for r in items);complete=excluded==0
                cost={key:sum(r['cost'][key] for r in items) if all(r['cost'][key] is not None for r in items) else None
                    for key in ('calls','prompt_tokens','completion_tokens','estimated_cost','generation_seconds','generation_human_interventions')}
                metrics[method]={'tasks':len(items),'accepted':len(accepted),'valid_automatically_completed':valid,
                    'automatic_completion_rate':wilson(len(accepted),len(items)),
                    'accepted_validity':wilson(valid,len(accepted)) if complete else None,
                    'automatic_valid_coverage':wilson(valid,len(items)) if complete else None,
                    'exposure_excluded':excluded,'cost':cost,
                    'cost_per_valid':{k:v/valid if complete and v is not None and valid else None for k,v in cost.items()},
                    'review_seconds':sum(j['answer']['elapsed_seconds']+j['localization_seconds'] for r in items for j in r['judgments'])}
            paired={o:{r['method']:bool(r['accepted'] and r['valid']) for r in rows if r['ordinal']==o} for o in {r['ordinal'] for r in rows}}
            delta=(sum(v['agent']-v['baseline'] for v in paired.values())/len(paired)) if not any(r['exposure_excluded'] for r in rows) else None
            # Paired context bootstrap acknowledges within-scene dependence.
            groups={}
            for r in rows:groups.setdefault(r['context'],set()).add(r['ordinal'])
            interval=None
            if delta is not None and len(groups)>=2:
                rng=random.Random(s['spec']['seed']);keys=list(groups);values=[]
                for _ in range(2000):
                    sampled=[o for key in rng.choices(keys,k=len(keys)) for o in groups[key]]
                    values.append(sum(paired[o]['agent']-paired[o]['baseline'] for o in sampled)/len(sampled))
                values.sort();interval=[values[49],values[1949]]
            report={'protocol':PROTOCOL,'study_id':sid,'spec':s['spec'],'method_sha256':s['method_sha256'],
                'generation_configuration':service.store.detail(s['job_id'])['spec'],
                'baseline_policy':BASELINE,'initial_input':'same prepared packet, native geometry and source RGB; Agent may acquire additional evidence within its frozen budget',
                'sampling_frame':'seeded random sample of unattempted selected tracks in the declared official split; common preparation failures remain in the denominator',
                'execution_order':'fixed baseline first, then Agent; no cross-method observation reuse',
                'methods':metrics,'paired_coverage_difference':delta,'paired_context_bootstrap_ci95':interval,
                'contexts':len(groups),'disagreements':disputes,'rows':rows,
                'limitations':['人工原生候选定位，不是自由3D框预测或独立认证。','同一审核者先锁定全部定位再看目标；配对文本可能存在顺序影响。',
                    '货币成本仅按填写单价估算；缺少usage或单价时不报告金额。','提前见过目标的记录不计独立有效率；此时主质量比例不发布。']}
            with self.runs.connect() as db:db.execute("UPDATE paper_studies SET status='sealed',report=? WHERE study_id=?",(json.dumps(report),sid))
            return self.detail(sid,service)

    def export_csv(self,sid):
        s=self.private(sid)
        if s['status']!='sealed':raise ValueError('完成双人盲审、裁定并封存后才能导出带方法标签的结果。')
        output=io.StringIO();fields=['ordinal','method','expression','accepted','valid','status','reason','input_sha256','run_ids','cost','judgments','exposure_excluded']
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for row in s['report']['rows']:
            record={k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(dict,list)) else row[k] for k in fields}
            for key,value in record.items():
                if isinstance(value,str) and value.startswith(('=','+','-','@','\t','\r')):record[key]="'"+value
            writer.writerow(record)
        return '\ufeff'+output.getvalue()
