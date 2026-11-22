import { defaultEvaluationTitle, evaluationTitle, serviceText, localeTag, localizeLabels, t, useLocale } from '../../shared/locale.ts'
import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { readJson, readPollingJson, errorMessage, categoryLabel } from '../../shared/annotationDisplay'
import { formatDuration, hasIncompleteTiming } from '../../shared/duration'
import { HelpHint } from '../../components/Help'
import TaskScene from '../results/TaskScene'
import type { TaskSceneData } from '../results/TaskScene'
import { useTaskDeletion } from '../tasks/TaskManager'
import './research.css'

type Rate = { numerator:number; denominator:number; value:number|null; ci95:number[]|null }
const qualityProtocol = 'agent-quality-metrics-v2'
type Quality = { automatic_completion_rate:Rate; text_accuracy?:Rate|null; blind_target_identification_accuracy?:Rate|null; accepted_validity?:Rate|null; random_sample_pass_rate?:Rate|null }
type Report = { quality?:Quality; methods?:{agent:Quality}; prior_exposure_count?:number }
type Study = { study_id:string;job_id:string;status:string;created_at:number;spec:{ title:string;sample_size:number;seed:number;dataset_split:string;reviewers:string[];review_sample_size?:number|null;review_all_deliveries?:boolean;quality_protocol?:string|null };cases:number;
  reviewers:{name:string;role:number;url:string;localized:number;answered:number}[];batch:{status:string;calls:number;finished:number;passed?:number;control?:string|null;reason?:string};report:Report|null;quality_preview?:Report|null;automatic_completion_rate:Rate|null }
type LocalizationResult = 'correct'|'incorrect'|'unmatched'
type Inbox = {title:string;reviewer:string;role:number;status:string;phase:string;waiting:boolean;quality_protocol?:string|null;cases:{case_id:string;localized:boolean;answered:boolean;localization_result?:LocalizationResult}[]}
type Answer = { factual_correct:boolean;unique?:boolean|null;input_sufficient?:boolean|null;prior_target_exposure?:boolean|null;notes:string;elapsed_seconds:number }
type BlindCase = Pick<TaskSceneData,'preview'|'candidates'|'cameras'> & {case_id:string;expression:string;selection:string[]|null;answer:Answer|null;revealed:boolean;expected:string|null;full_cloud:string;point_count:number}
const names:Record<string,string>=localizeLabels({ready:'待开始',running:'生成中',queued:'已排队',paused:'已暂停',stopped:'已停止',completed:'生成结束',blocked:'生成受阻',review:'独立盲审中',sealed:'已封存',creating:'建立中'})
function singleReviewComplete(study:Study){
  const delivered=study.automatic_completion_rate?.numerator??study.batch.passed
  if(study.status!=='review'||study.reviewers.length!==1||study.spec.review_sample_size==null||delivered==null)return false
  const required=study.spec.review_all_deliveries?delivered:Math.min(study.spec.review_sample_size,delivered)
  return study.reviewers[0].answered>=required
}
function studyStatus(study:Study, progress?:BatchProgress|null){
  if(singleReviewComplete(study))return t("审核完成 · 待生成报告")
  if(study.status!=='ready')return names[study.status]??study.status
  if(study.batch.status==='queued'&&progress?.queue_paused)return t("队列已暂停")
  if(study.batch.status==='running'){
    if(progress?.executor_active===false)return t("等待恢复")
    const control=progress?progress.control:study.batch.control
    if(control==='pause')return t("正在暂停")
    if(control==='stop')return t("正在停止")
  }
  if(['completed','paused','stopped','blocked'].includes(study.batch.status)&&progress?.executor_active)return t("正在保存")
  return names[study.batch.status]??study.batch.status
}
function selectedStudy(data:Study[],old:string,completedReview:string|null){
  if(data.some(s=>s.study_id===old))return old
  return (completedReview?data.find(s=>s.reviewers.some(r=>new URLSearchParams(r.url.split('?')[1]).get('review')===completedReview))?.study_id:undefined)??data[0]?.study_id??''
}
const number=(n:number|null|undefined)=>n==null?t("未记录"):n.toLocaleString(localeTag(),{maximumFractionDigits:2})
const elapsedSince=(started:number)=>(Date.now()-started)/1000
async function get<T>(path:string):Promise<T>{return readJson(await fetch('/api/research/'+path)) as Promise<T>}
async function post<T>(path:string,body:unknown):Promise<T>{return readJson(await fetch('/api/research/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})) as Promise<T>}

export default function ResearchWorkspace() {
  useLocale()

  const token=new URLSearchParams(window.location.hash.split('?')[1]).get('review')
  return token?<BlindReview key={token} token={token}/>:<StudyManager/>
}

function StudyManager(){
  useLocale()

  const [studies,setStudies]=useState<Study[]>([]),[selected,setSelected]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
  const [creating,setCreating]=useState(false),[loading,setLoading]=useState(true)
  const [loadError,setLoadError]=useState(''),[reload,setReload]=useState(0)
  const [draftTitle,setTitle]=useState<string|null>(null),[size,setSize]=useState('50'),[reviewSize,setReviewSize]=useState('20'),[split,setSplit]=useState('train')
  const title=draftTitle??evaluationTitle()
  const [reviewer,setReviewer]=useState('')
  const [requestId,setRequestId]=useState(()=>crypto.randomUUID().replaceAll('-','').slice(0,24))
  const completedReview=new URLSearchParams(window.location.hash.split('?')[1]).get('review_done')
  useEffect(()=>{
    if(busy)return
    const controller=new AbortController()
    let timer:number|undefined
    const load=async()=>{
      try{
        if(document.hidden)return
        const data=await readPollingJson('/api/research/studies',controller.signal) as {studies:Study[]}
        if(controller.signal.aborted)return
        setStudies(data.studies)
        setSelected(old=>selectedStudy(data.studies,old,completedReview))
        setLoadError('')
      }catch(e){
        if(!controller.signal.aborted)setLoadError(errorMessage(e))
      }finally{
        if(!controller.signal.aborted){
          if(!document.hidden)setLoading(false)
          timer=window.setTimeout(load,5000)
        }
      }
    }
    void load()
    return()=>{controller.abort();window.clearTimeout(timer)}
  },[busy,completedReview,reload])
  const study=studies.find(s=>s.study_id===selected)
  const legacy=!!study&&study.spec.quality_protocol!==qualityProtocol
  const targetCount=Number(size)
  const reviewCount=Number(reviewSize)
  const validTargetCount=Number.isInteger(targetCount)&&targetCount>=1&&targetCount<=500
  const generationPlan=validTargetCount
    ? t("固定抽取 {0} 个目标。该数量是自动完成率的分母。",targetCount)
    : t("固定目标数是自动完成率的分母。")
  const reviewPlan=Number.isInteger(reviewCount)&&reviewCount>=1&&reviewCount<=500
    ? t("从自动通过的描述中随机抽审最多 {0} 条，不足时全部审核。",validTargetCount?Math.min(targetCount,reviewCount):reviewCount)
    : t("从自动通过的描述中随机抽审，数量不足时全部审核。")
  const {progress,stale}=useEvaluationProgress(study,reload)
  const workflow=study?studyWorkflow(study,progress,stale):null
  const studyAction=workflow?.action
  const generationActive = progress?.executor_active || (progress?.status ?? study?.batch.status) === 'running'
  const deletion = useTaskDeletion({
    studyId: study?.study_id,
    isActive: id => id === study?.job_id && Boolean(generationActive),
    onDeleted: jobs => {
      setStudies(old => old.filter(item => !jobs.includes(item.job_id)))
      setReload(value => value + 1)
    },
  })
  async function create(e:FormEvent){e.preventDefault();setBusy(true);setError('')
    try{const data=await post<Study>('studies',{request_id:requestId,title:draftTitle??defaultEvaluationTitle,sample_size:Number(size),seed:parseInt(requestId.slice(0,7),16),dataset_split:split,reviewers:[reviewer],
      review_sample_size:Number(reviewSize),quality_protocol:qualityProtocol})
      setStudies(old=>[data,...old.filter(s=>s.study_id!==data.study_id)]);setSelected(data.study_id)
      setLoading(false);setCreating(false);setReload(v=>v+1);setRequestId(crypto.randomUUID().replaceAll('-','').slice(0,24))
    }catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  async function control(action:string){if(!study)return;
    setBusy(true);setError('');deletion.clearMessages()
    try{
      const result=await post<Study>(`studies/${study.study_id}/control`,{action})
      setStudies(old=>old.map(s=>s.study_id===result.study_id?result:s))
      setReload(v=>v+1)
    }catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  return <div className="paper-workspace">
    {error&&<p className="error" role="alert">{serviceText(error)}</p>}
    {deletion.error&&!deletion.review&&<p className="error" role="alert">{serviceText(deletion.error)}</p>}
    {deletion.feedback&&<p className="operation-feedback" role="status">{serviceText(deletion.feedback)}</p>}
    {deletion.dialog}
    {loadError&&<div className="error results-error" role="alert"><span>{t("评估记录暂未同步。")}{serviceText(loadError)}</span><button onClick={()=>setReload(v=>v+1)}>{t("重试读取")}</button></div>}
    <section className="paper-card paper-report" aria-labelledby="evaluation-metrics">
      <h2 id="evaluation-metrics" className="sr-only">{t("评估指标")}</h2>
      <div className="section-heading paper-report-toolbar">
        <div className="paper-study-picker">
          {studies.length>0?<label>
            <span>{t("评估记录")}</span>
            <select value={selected} onChange={e=>setSelected(e.target.value)} disabled={busy}>
              {studies.map(s=><option key={s.study_id} value={s.study_id}>{evaluationTitle(s.spec.title)} · {new Date(s.created_at*1000).toLocaleDateString(localeTag())}</option>)}
            </select>
          </label>:!loadError&&<p className="muted" role="status">{loading?t("正在读取…"):t("暂无评估记录")}</p>}
          {legacy&&<p className="muted">{t("历史评估 · 原统计口径")}</p>}
        </div>
        <div className="paper-actions">
          {study?.report&&<a className="button-link" href={'/api/research/studies/'+study.study_id+'/report.json'}>{t("下载报告")}</a>}
          <button className="primary" aria-expanded={creating} aria-controls={creating?'evaluation-plan':undefined} onClick={()=>setCreating(!creating)} disabled={busy}>{creating?t("收起设置"):t("新建评估")}</button>
        </div>
      </div>
      <StudyReport report={study?.report??study?.quality_preview??null} completion={study?.automatic_completion_rate??null} legacy={legacy}/>
    </section>
    {creating&&<form id="evaluation-plan" className="paper-card paper-create" aria-labelledby="evaluation-plan-title" onSubmit={create}>
      <h2 id="evaluation-plan-title">{t("评估设置")}</h2>
      <label>{t("评估名称")}<input required maxLength={100} value={title} onChange={e=>setTitle(e.target.value)}/></label>
      <div className="paper-plan-grid">
        <fieldset>
          <legend>{t("自动生成")}</legend>
          <div className="paper-plan-fields">
            <label>{t("评估数据集")}<select value={split} onChange={e=>setSplit(e.target.value)}><option value="train">{t("官方训练集")}</option><option value="validation">{t("官方验证集")}</option></select></label>
            <label>{t("固定目标数")}<input type="number" min={1} max={500} required value={size} aria-describedby="generation-plan" onChange={e=>setSize(e.target.value)}/></label>
          </div>
          <p id="generation-plan">{generationPlan}</p>
        </fieldset>
        <fieldset>
          <legend>{t("独立人工盲审")}</legend>
          <div className="paper-plan-fields">
            <label>{t("计划抽审数")}<input type="number" min={1} max={500} required value={reviewSize} aria-describedby="review-plan" onChange={e=>setReviewSize(e.target.value)}/></label>
            <label>{t("独立审核者")}<input required maxLength={80} placeholder={t("审核者标识")} value={reviewer} onChange={e=>setReviewer(e.target.value)}/></label>
          </div>
          <p id="review-plan">{reviewPlan} {t("同一批样本先盲选目标，再核对文本，据此计算人工盲选准确率和文本准确率。")}</p>
        </fieldset>
      </div>
      <div className="paper-form-footer"><p>{t("创建评估时建立专用标注任务并锁定目标。点击「开始生成」后调用模型。")}</p><button className="primary" disabled={busy}>{busy?t("正在创建…"):t("创建评估")}</button></div>
    </form>}
    {study&&workflow&&<>
        <section className="paper-card">
          <div className="section-heading"><div><h2><HelpHint label={t("评估进度")} detail={t("抽样随机种子：{0}", study.spec.seed)} /></h2><p className="section-caption">{study.spec.dataset_split==='validation'?t("官方验证集"):t("官方训练集")} · {t("{0} 个固定任务", study.spec.sample_size)}</p></div><span className="paper-status" role="status">{studyStatus(study,progress)}</span></div>
          <ol className="sample-steps study-steps" aria-label={t("评估流程")}>{[t("生成标注"),t("冻结样本"),t("独立盲审"),t("封存报告")].map((label,index)=><li key={label}
            aria-current={index===workflow.step?'step':undefined} className={index<workflow.step?'done':index===workflow.step?'current':''}><span aria-hidden="true">{index<workflow.step?'✓':index+1}</span>{label}</li>)}</ol>
          <EvaluationProgress study={study} progress={progress} stale={stale} />
          {workflow.message&&<p className="workflow-message" role="status">{workflow.message}</p>}
          {!workflow.reviewOpen&&<div className="paper-actions">
            {studyAction&&<button className="primary" disabled={busy} onClick={()=>void control(studyAction.kind)}>{studyAction.label}</button>}
            {workflow.cancelQueue&&<button disabled={busy} onClick={()=>void control('pause')}>{t("取消排队")}</button>}
            {study.status==='ready'&&study.job_id&&!study.report&&!study.cases&&
              <button className="danger text-button" disabled={busy||deletion.busy||generationActive}
                onClick={()=>void deletion.prepare([study.job_id])}>{t("删除评估")}</button>}
          </div>}
        </section>
        <section className="paper-card">
          <h2><HelpHint label={t("独立人工审核")} detail={t("先锁定全部盲测选择，再查看原生目标核对描述。")} /></h2>
          <div className="paper-reviewers">{study.reviewers.map(r=><div key={r.role}><strong>{r.name}</strong><span>{t("已定位 ")}{r.localized}{t(" · 已核对 ")}{r.answered}</span>{workflow.reviewOpen?<a className="button-link" href={r.url} target="_blank" rel="noreferrer">{t("打开盲审页")}</a>:<span className="muted">{t("待冻结")}</span>}</div>)}</div>
          {singleReviewComplete(study)&&<p className="muted" role="status">{t("审核已保存，可以生成报告。")}</p>}
          {study.status==='review'&&<button className="primary" disabled={busy||(!legacy&&!singleReviewComplete(study))} onClick={()=>void control('seal')}>{busy?t("正在处理…"):t("生成并封存报告")}</button>}
        </section>
    </>}
  </div>
}

type BatchProgress = {
  job_id:string; status:string; control:string|null; executor_active:boolean; counts:Record<string,number>;
  queue_paused?:boolean; queue_position?:number|null
  model_calls_reserved:number; timing?:{active_seconds:number|null;history_incomplete?:boolean;running?:boolean}
}

function studyWorkflow(study:Study, progress:BatchProgress|null, stale:boolean){
  const reviewOpen=study.status==='review'||study.status==='sealed'
  let step=study.status==='sealed'?4:study.status==='review'?singleReviewComplete(study)?3:2:0
  let action:{kind:'start'|'pause'|'freeze';label:string}|undefined
  let message=''
  let cancelQueue=false
  const {status}=study.batch
  const control=progress?progress.control:study.batch.control
  if(study.status==='ready'){
    const canFreeze=status==='completed'||status==='blocked'&&study.spec.quality_protocol!==qualityProtocol
    if(canFreeze){
      step=1
      if(progress?.executor_active===false&&!stale)action={kind:'freeze',label:t("冻结结果，进入盲审")}
      else message=progress?.executor_active?t("正在保存生成结果。"):t("正在核对执行器状态。")
    }else if(status==='queued'){
      cancelQueue=!!progress&&!stale
      if(progress?.queue_paused&&!stale)action={kind:'start',label:t("继续队列")}
      message=progress?.queue_paused?t("队列已暂停。"):progress?.queue_position!=null?t("当前排队第 {0} 位。", progress.queue_position):t("等待可用执行位。")
    }else if(status==='running'){
      if(progress?.executor_active===false&&!stale){
        action={kind:'start',label:t("继续生成")}
        message=t("执行器已退出，可从已保存的进度继续。")
      }else if(control){
        message=control==='stop'?t("正在保存当前步骤并停止。"):t("正在保存当前步骤并暂停。")
      }else if(progress?.executor_active&&!stale){
        action={kind:'pause',label:t("暂停实验")}
      }else if(!progress)message=t("正在同步执行器状态。")
    }else if(status==='ready'){
      action={kind:'start',label:t("开始生成")}
    }else if(['paused','stopped','blocked'].includes(status)){
      if(progress?.executor_active===false&&!stale){
        action={kind:'start',label:t("继续生成")}
        if(status==='blocked')message=t("生成受阻，恢复后可处理剩余目标。")
      }else message=progress?.executor_active?t("正在保存生成结果。"):t("正在核对执行器状态。")
    }
  }
  return {reviewOpen,step,action,message,cancelQueue}
}

function useEvaluationProgress(study:Study|undefined,reload:number){
  const [snapshot,setSnapshot]=useState<{data:BatchProgress;reload:number}|null>(null)
  const [failedJob,setFailedJob]=useState('')
  const jobId=study?.job_id
  const status=study?.batch.status
  useEffect(()=>{
    if(!jobId)return
    const controller=new AbortController()
    let timer:number|undefined
    const load=async()=>{
      try{
        if(!document.hidden){
          const data=await readPollingJson(`/api/batches/${jobId}/progress`,controller.signal) as BatchProgress
          if(data.status==='queued'){
            const index=await readPollingJson('/api/batches',controller.signal) as {
              jobs:{job_id:string;status:string;queue_paused?:boolean;queue_position?:number|null}[]
            }
            const queued=index.jobs.find(job=>job.job_id===jobId&&job.status==='queued')
            if(!queued){
              if(!controller.signal.aborted)setSnapshot(null)
              return
            }
            data.queue_paused=queued.queue_paused
            data.queue_position=queued.queue_position
          }
          if(!controller.signal.aborted){setSnapshot({data:{...data,job_id:jobId},reload});setFailedJob('')}
        }
      }catch{
        if(!controller.signal.aborted)setFailedJob(jobId)
      }finally{
        if(!controller.signal.aborted)timer=window.setTimeout(load,status==='running'||status==='queued'?5000:15000)
      }
    }
    void load()
    return()=>{controller.abort();window.clearTimeout(timer)}
  },[jobId,status,reload])
  const progress=snapshot?.reload===reload?snapshot.data:null
  return {
    progress:progress?.job_id===jobId&&progress?.status===status?progress:null,
    stale:!!jobId&&failedJob===jobId,
  }
}

function EvaluationProgress({study,progress:current,stale}:{study:Study;progress:BatchProgress|null;stale:boolean}){
  useLocale()

  const total=study.spec.sample_size
  const status=study.batch.status
  const counts=current?.counts
  const finished=counts?Object.entries(counts).reduce((n,[state,count])=>n+(['pending','preparing','running'].includes(state)?0:count),0):study.batch.finished
  const seconds=current?.timing?.active_seconds
  const running=status==='running'&&current?.executor_active&&!current.control&&!study.batch.control
  const eta=running&&!stale&&finished>=3&&seconds!=null&&!hasIncompleteTiming(current?.timing)
    ? formatDuration(seconds/finished*Math.max(0,total-finished))
    : null
  return <div className="paper-progress">
    <p><strong>{finished} / {total}</strong>{t(" 个目标已处理")}</p>
    <progress className="batch-progress" aria-label={t("评估处理进度")} max={total} value={finished}/>
    {counts&&<p className="muted">{running&&t("准备中 {0} · 标注中 {1} · 待处理 {2} · ", counts.preparing??0, counts.running??0, counts.pending??0)}{t("模型调用 {0} 次", current?.model_calls_reserved??0)}</p>}
    {seconds!=null&&<p className="muted">{t("累计运行 ")}{formatDuration(seconds)}{running&&t(" · 预计剩余 {0}", eta??t("待至少 3 个目标处理完毕后估算"))}</p>}
    {stale&&<p role="status">{t("进度暂未更新，等待重新连接。")}</p>}
  </div>
}

function StudyReport({report,completion,legacy}:{report:Report|null;completion:Rate|null;legacy:boolean}){
  useLocale()

  const quality=report?.quality??report?.methods?.agent
  const metrics=[
    {label:t("自动完成率"), rate:quality?.automatic_completion_rate??completion, detail:legacy?t("通过自动验证的交付 ÷ 全部固定任务"):t("无人工干预且通过自动核验的目标数 ÷ 固定目标数")},
    ...(legacy ? [
      {label:t("交付有效率"), rate:quality?.accepted_validity, detail:t("盲审有效交付 ÷ 全部交付")},
      {label:t("随机抽样通过率"), rate:quality?.random_sample_pass_rate, detail:t("盲审有效样本 ÷ 实际抽样数")},
    ] : [
      {label:t("人工盲选准确率"), rate:quality?.blind_target_identification_accuracy, detail:t("锁定的盲选命中数 ÷ 实际抽审数")},
      {label:t("文本准确率"), rate:quality?.text_accuracy, detail:t("属性与关系均正确的描述数 ÷ 实际抽审数")},
    ]),
  ]
  return <>
    <dl className="paper-stats">{metrics.map(({label,rate,detail})=>(
      <div className="paper-stat" key={label}>
        <dt>{label}</dt>
        <dd>
          <div className="paper-stat-value">
            {rate?.value==null
              ? <><span className="paper-stat-dash" aria-hidden="true">—</span><span>{t("未测量")}</span></>
              : <strong>{(rate.value*100).toFixed(1)}%</strong>}
          </div>
          <small className="paper-metric-definition">{detail}</small>
          <div className="paper-stat-details">
            {rate&&<small className="paper-ci">{rate.numerator} / {rate.denominator}{rate.ci95&&t(" · 95% Wilson 区间 {0}%–{1}%", (rate.ci95[0]*100).toFixed(1), (rate.ci95[1]*100).toFixed(1))}</small>}
          </div>
        </dd>
      </div>
    ))}</dl>
    {!!report?.prior_exposure_count&&<p className="notice" role="status">{t("有 {0} 条样本在盲测前已知目标，人工盲选准确率不发布。", report.prior_exposure_count)}</p>}
  </>
}

const localizationLabel=(result:LocalizationResult)=>result==='correct'?t("命中"):result==='incorrect'?t("选错目标"):t("未选目标")

function BlindReview({token}:{token:string}){
  useLocale()

  const [inbox,setInbox]=useState<Inbox|null>(null),[cid,setCid]=useState(''),[data,setData]=useState<BlindCase|null>(null)
  const [selection,setSelection]=useState<string[]>([]),[focus,setFocus]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('')
  const [answers,setAnswers]=useState<Record<string,string>>({}),[notes,setNotes]=useState(''),[started,setStarted]=useState(()=>Date.now())
  const [noMatch,setNoMatch]=useState(false),[loadVersion,setLoadVersion]=useState(0)
  const [inboxVersion,setInboxVersion]=useState(0)
  const [inspectCompleted,setInspectCompleted]=useState(false)
  const [wrongOnly,setWrongOnly]=useState(false)
  const [jumpIndex,setJumpIndex]=useState(''),[navigationError,setNavigationError]=useState('')
  const [savedBlind,setSavedBlind]=useState<{index:number;result:LocalizationResult}|null>(null)
  const reviewComplete=!!inbox?.cases.length&&inbox.cases.every(c=>c.answered)
  const singleTarget=inbox?.quality_protocol===qualityProtocol
  const entries=(inbox?.cases??[]).map((entry,index)=>({entry,index}))
  const isMiss=(entry:Inbox['cases'][number])=>entry.localization_result==='incorrect'||entry.localization_result==='unmatched'
  const misses=entries.filter(({entry})=>isMiss(entry))
  const visibleEntries=wrongOnly?entries.filter(({entry})=>entry.case_id===cid||isMiss(entry)):entries
  const position=visibleEntries.findIndex(({entry})=>entry.case_id===cid)
  const submitting=useRef(false)
  useEffect(()=>{let live=true;get<Inbox>(`review/${token}`).then(v=>{if(live){setInbox(v);setCid(v.cases.find(c=>v.phase==='localization'?!c.localized:!c.answered)?.case_id??v.cases[0]?.case_id??'')}}).catch(e=>{if(live)setError(errorMessage(e))});return()=>{live=false}},[token,inboxVersion])
  useEffect(()=>{let live=true
    if(cid)void get<BlindCase>(`review/${token}/cases/${cid}`).then(v=>{if(live){setData(v);setNoMatch(false);setError('');setFocus('');setStarted(Date.now());setSelection(v.selection??[]);setAnswers(v.answer?Object.fromEntries(Object.entries(v.answer).filter(([key,value])=>key!=='prior_target_exposure'&&typeof value==='boolean').map(([k,v])=>[k,String(v)])):{});setNotes(v.answer?.notes??'')}}).catch(e=>{if(live)setError(errorMessage(e))})
    return()=>{live=false}
  },[token,cid,inbox?.phase,loadVersion])
  const current=data?.case_id===cid&&data.revealed===(inbox?.phase==='judgment')
  const locked=data?.selection!=null||!!inbox?.cases.find(c=>c.case_id===cid)?.localized
  const selectionDisabled=!current||locked||busy||inbox?.status==='sealed'
  const judgmentDisabled=!current||busy||!!data?.answer||inbox?.status==='sealed'
  async function submit(kind:'localization'|'answer'){
    if(submitting.current||!current||inbox?.status==='sealed')return
    if(kind==='localization'&&(locked||data?.revealed||(!selection.length&&!noMatch)))return
    if(kind==='answer'&&(!data?.revealed||data.answer||fields.some(([key])=>!['true','false'].includes(answers[key]))))return
    submitting.current=true;setBusy(true);setError('')
    try{const v=await post<Inbox>(`review/${token}/cases/${cid}/${kind}`,kind==='localization'?{selected_labels:selection,elapsed_seconds:elapsedSince(started)}:{...Object.fromEntries(Object.entries(answers).map(([k,v])=>[k,v==='true'])),notes,elapsed_seconds:elapsedSince(started)})
      const result=v.cases.find(c=>c.case_id===cid)?.localization_result
      if(kind==='localization'&&result)setSavedBlind({index:entries.findIndex(({entry})=>entry.case_id===cid)+1,result})
      setData(null);setInbox(v);setCid(v.cases.find(c=>v.phase==='localization'?!c.localized:!c.answered)?.case_id??cid)
      setLoadVersion(old=>old+1)
    }catch(e){setError(errorMessage(e))}finally{submitting.current=false;setBusy(false)}
  }
  function choose(id:string){setFocus(id);if(!selectionDisabled&&!submitting.current){setNoMatch(false);setSelection(old=>singleTarget?[id]:old.includes(id)?old:[...old,id])}}
  function toggle(id:string){setFocus(id);if(!selectionDisabled&&!submitting.current){setNoMatch(false);setSelection(old=>old.includes(id)?old.filter(v=>v!==id):singleTarget?[id]:[...old,id])}}
  function openEntry(id:string){
    if(submitting.current)return
    setNavigationError('')
    if(id!==cid){setData(null);setCid(id);setError('');setJumpIndex('')}
  }
  function goToIndex(event:FormEvent){
    event.preventDefault()
    if(submitting.current)return
    const number=Number(jumpIndex)
    const target=/^\d+$/.test(jumpIndex.trim())&&visibleEntries.find(({index})=>index+1===number)
    if(!target){setNavigationError(t("请输入当前列表中的条目序号。"));return}
    openEntry(target.entry.case_id);setJumpIndex('')
  }
  function inspectMisses(checked:boolean){
    if(submitting.current)return
    setWrongOnly(checked)
    setNavigationError('')
    if(checked&&!misses.some(({entry})=>entry.case_id===cid)){
      openEntry(misses[0]?.entry.case_id??'')
    }
  }
  const fields=singleTarget
    ? [['factual_correct',t("针对框出的原生目标，描述中的全部属性和空间关系是否都与场景一致？无法确认时选“否”。")]]
    : [['factual_correct',t("表达中的事实是否都正确？")],['unique',t("表达是否只符合一个对象，没有其他可能对象？")],['input_sufficient',t("只凭这张原图与同帧点云，是否足以理解表达？")]]
  const blindStep=reviewComplete?2:inbox?.phase==='judgment'?1:0
  return <div className="paper-workspace paper-blind"><section className="paper-card paper-intro">
    <div className="section-heading"><div><h2>{inbox?evaluationTitle(inbox.title):t("审核任务")}</h2><p className="section-caption">{inbox?`${inbox.reviewer} · ${inbox.role===2?t("第三人裁定"):t("独立审核")}`:error?t("读取失败"):t("正在读取审核任务…")}</p></div></div>
    {inbox&&<ol className="sample-steps blind-steps" aria-label={t("独立审核流程")}>{[t("目标识别盲测"),t("文本事实核对")].map((label,index)=><li key={label}
      aria-current={index===blindStep?'step':undefined} className={index<blindStep?'done':index===blindStep?'current':''}><span aria-hidden="true">{index<blindStep?'✓':index+1}</span>{label}</li>)}</ol>}
    {inbox&&!inbox.waiting&&inbox.cases.length>0&&(!reviewComplete||inspectCompleted)&&<div className="paper-review-progress">
      <div className="paper-review-entries">
      <label>{t("审核条目")}<select disabled={busy} value={cid} onChange={e=>openEntry(e.target.value)}>
        {visibleEntries.map(({entry:c,index})=><option key={c.case_id} value={c.case_id}>{index+1} / {inbox.cases.length} · {c.localization_result
          ? localizationLabel(c.localization_result)
          : c.answered?t("已判断"):c.localized?t("已锁定定位"):t("待定位")}</option>)}
      </select></label>
      <nav className="paper-entry-nav" aria-label={t("审核条目导航")}>
        <button type="button" disabled={busy||position<=0} onClick={()=>openEntry(visibleEntries[position-1].entry.case_id)}>{t("上一条")}</button>
        <button type="button" disabled={busy||position<0||position>=visibleEntries.length-1} onClick={()=>openEntry(visibleEntries[position+1].entry.case_id)}>{t("下一条")}</button>
      </nav>
      <form className="paper-entry-jump" onSubmit={goToIndex}>
        <input type="text" inputMode="numeric" aria-label={t("条目序号")} placeholder={t("序号")} value={jumpIndex}
          aria-invalid={!!navigationError} disabled={busy} onChange={e=>{setJumpIndex(e.target.value);setNavigationError('')}}/>
        <button disabled={busy}>{t("跳转")}</button>
      </form>
      {singleTarget&&reviewComplete&&<label className="review-checkbox"><input type="checkbox" checked={wrongOnly} disabled={busy||(!wrongOnly&&!misses.length)}
        onChange={e=>inspectMisses(e.target.checked)}/>{t("只看未命中（{0}）",misses.length)}</label>}
      </div>
      <span>{t("已定位 ")}{inbox.cases.filter(c=>c.localized).length}{t(" · 已核对 ")}{inbox.cases.filter(c=>c.answered).length}</span>
      {navigationError&&<p className="error paper-entry-error" role="alert">{navigationError}</p>}
      {savedBlind&&inbox.phase==='judgment'&&!reviewComplete&&<p className="paper-selection-receipt" role="status">{t("第 {0} 条已保存 · {1}",savedBlind.index,localizationLabel(savedBlind.result))}</p>}
    </div>}
  </section>
    {error&&<div role="alert" className="error results-error"><span>{serviceText(error)}</span>{!inbox&&<button onClick={()=>{setError('');setInboxVersion(v=>v+1)}}>{t("重试读取审核任务")}</button>}</div>}
    {inbox?.waiting?<section className="paper-card paper-empty"><h3>{t("等待两名审核者完成")}</h3><p>{t("独立判断全部提交后，这里仅提供有分歧的表达。")}</p><button onClick={()=>window.location.reload()}>{t("刷新进度")}</button></section>:inbox&&!inbox.cases.length?<section className="paper-card paper-empty"><h3>{t("当前没有需要审核的表达")}</h3><p>{t("本入口没有待分配的条目。无输出任务仍计入自动完成率的分母。")}</p></section>:<>
    {reviewComplete&&<section className="paper-card paper-review-complete" role="status" aria-label={t("审核完成")}>
      <h3>{t("审核已完成 · ")}{inbox.cases.length} / {inbox.cases.length}</h3>
      <p>{t("全部独立判断已保存，审核完成。")}</p>
      <div className="paper-actions"><a className="primary paper-completion-link" href={`#research?review_done=${encodeURIComponent(token)}`}>{inbox.status==='sealed'?t("返回评估，查看报告"):singleTarget?t("返回评估，查看指标"):t("下一步：返回评估生成报告")}</a><button onClick={()=>setInspectCompleted(old=>!old)}>{inspectCompleted?t("收起已提交判断"):t("查看已提交判断")}</button></div>
    </section>}
    {(!reviewComplete||inspectCompleted)&&<>
    {cid&&!current&&<p role="status">{error?<button onClick={()=>{setError('');setLoadVersion(old=>old+1)}}>{t("重新加载当前条目")}</button>:t("正在加载审核条目…")}</p>}
    {current&&data&&<section className="paper-card"><blockquote className="paper-expression" lang="en">{data.expression}</blockquote>
      {(!data.revealed||!singleTarget)&&<p className="paper-note">{data.revealed?t("预期目标：{0}。你之前锁定的候选：{1}。请独立判断事实和唯一性。", data.expected, data.selection?.join('、')||t("没有匹配对象")):singleTarget?t("根据描述和场景选择一个目标。没有匹配对象或有多个合理对象时，选“无法唯一定位”。全部选择锁定后才会揭示目标。"):t("请选择所有符合表达的对象，可多选或明确提交空选。全部条目定位锁定前不会揭示目标。")}</p>}
      {singleTarget&&data.revealed&&inspectCompleted&&<p className="paper-note">{t("锁定选择 {0} · 原生目标 {1}",data.selection?.[0]??t("无法唯一定位"),data.expected)}</p>}
      {!data.revealed&&<div className="paper-localization-actions" aria-label={t("盲评定位确认")}>
        <div aria-live="polite"><strong>{locked?t("定位已锁定"):selection.length?singleTarget?t("已选 ")+selection[0]:t("已选 {0} 个对象", selection.length):noMatch&&singleTarget?t("无法唯一定位"):t("尚未选择匹配对象")}</strong><p className="muted">{locked?t("完成其余条目后进入事实核对。"):singleTarget?t("点击场景中的框或候选编号选择目标。"):t("点击框即可加入答案。点击下方编号可移除。")}</p>
          {savedBlind&&<p className="paper-selection-receipt">{t("第 {0} 条已保存 · {1}",savedBlind.index,localizationLabel(savedBlind.result))}</p>}
          <div className="paper-selected-objects">{selection.map(id=><button key={id} disabled={selectionDisabled} aria-label={t("移除所选对象 {0}", id)} onClick={()=>toggle(id)}>{id} ×</button>)}</div>
        </div>
        <div className="paper-actions"><label><input type="checkbox" aria-label={singleTarget?t("无法唯一定位"):t("没有匹配对象")} checked={noMatch} disabled={selectionDisabled||(!singleTarget&&selection.length>0)} onChange={e=>{setNoMatch(e.target.checked);if(singleTarget&&e.target.checked){setSelection([]);setFocus('')}}}/>{singleTarget?t("无法唯一定位"):t("没有匹配对象")}</label><button className="primary" disabled={selectionDisabled||(!selection.length&&!noMatch)} onClick={()=>void submit('localization')}>{busy?t("正在保存…"):locked?t("定位已锁定"):singleTarget?t("锁定选择，下一条"):noMatch?t("确认无匹配对象，下一条"):t("确认所选对象，下一条")}</button></div>
        {error&&<p role="alert" className="error">{serviceText(error)}</p>}
      </div>}
      <TaskScene key={`${cid}:${data.revealed?'revealed':'blind'}`} data={{...data,job_id:'',ordinal:0,revision:'',run_id:'',can_confirm:false,review:true,selected_candidate_id:null,confirmation:null,image_box_source:'official_lidar_box_projection'}} selected={data.revealed?(data.expected??''):focus} onSelect={data.revealed?undefined:choose} candidatePicker={false}/>
      {(!singleTarget||!data.revealed)&&<div className="paper-candidates">{data.candidates.map(c=><label key={c.candidate_id} className={selection.includes(c.candidate_id)?'checked':''}><input type={singleTarget?'radio':'checkbox'} name={singleTarget?'blind-target':undefined} disabled={locked||busy||inbox?.status==='sealed'} checked={selection.includes(c.candidate_id)} onChange={()=>singleTarget?choose(c.candidate_id):toggle(c.candidate_id)}/>{c.candidate_id}<small>{categoryLabel(c.category)}</small></label>)}</div>}
      <p className="muted">{t("完整点云 ")}{number(data.point_count)}{t(" 点 · ")}<a href={data.full_cloud}>{t("下载单帧点云")}</a></p>
      {data.revealed&&<form onSubmit={e=>{e.preventDefault();void submit('answer')}} className="paper-judgment"><h3>{t("独立事实判断")}</h3>{fields.map(([key,label])=><fieldset key={key} disabled={judgmentDisabled}><legend>{label}</legend>{['true','false'].map(value=><label key={value}><input type="radio" name={key} required value={value} checked={answers[key]===value} onChange={()=>setAnswers(old=>({...old,[key]:value}))}/>{value==='true'?t("是"):t("否")}</label>)}</fieldset>)}<label>{t("原因与备注")}<textarea maxLength={2000} value={notes} onChange={e=>setNotes(e.target.value)} disabled={judgmentDisabled} placeholder={t("指出错误事实、其他可能对象或输入缺失")}/></label><button className="primary" disabled={judgmentDisabled}>{data.answer?t("判断已保存"):busy?t("正在保存…"):t("提交独立判断，下一条")}</button></form>}
    </section>}</>}
    </>}
  </div>
}
