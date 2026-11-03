import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { readJson, errorMessage, categoryLabel } from '../../shared/annotationDisplay'
import TaskScene from '../results/TaskScene'
import type { TaskSceneData } from '../results/TaskScene'
import './research.css'

type Rate = { numerator:number; denominator:number; value:number|null; ci95:number[]|null }
type Quality = { automatic_completion_rate:Rate; accepted_validity:Rate|null; random_sample_pass_rate?:Rate|null }
type Report = { quality?:Quality; methods?:{agent:Quality} }
type Study = { study_id:string;job_id:string;status:string;created_at:number;spec:{ title:string;sample_size:number;seed:number;dataset_split:string;reviewers:string[] };cases:number;
  reviewers:{name:string;role:number;url:string;localized:number;answered:number}[];batch:{status:string;calls:number;finished:number;reason?:string};report:Report|null;automatic_completion_rate:Rate|null }
type Inbox = {title:string;reviewer:string;role:number;status:string;phase:string;waiting:boolean;cases:{case_id:string;localized:boolean;answered:boolean}[]}
type Answer = { factual_correct:boolean;unique:boolean;input_sufficient:boolean;prior_target_exposure:boolean;notes:string;elapsed_seconds:number }
type BlindCase = Pick<TaskSceneData,'preview'|'candidates'|'cameras'> & {case_id:string;expression:string;selection:string[]|null;answer:Answer|null;revealed:boolean;expected:string|null;full_cloud:string;point_count:number}
const names:Record<string,string>={ready:'待开始',running:'生成中',queued:'已排队',paused:'已暂停',stopped:'已停止',completed:'生成结束',blocked:'生成受阻',review:'独立盲审中',sealed:'已封存',creating:'建立中'}
const pct=(r:Rate|null)=>r?.value==null?'未测量':`${(r.value*100).toFixed(1)}%`
const number=(n:number|null|undefined)=>n==null?'未记录':n.toLocaleString('zh-CN',{maximumFractionDigits:2})
async function get<T>(path:string):Promise<T>{return readJson(await fetch('/api/research/'+path)) as Promise<T>}
async function post<T>(path:string,body:unknown):Promise<T>{return readJson(await fetch('/api/research/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})) as Promise<T>}

export default function ResearchWorkspace() {
  const token=new URLSearchParams(window.location.hash.split('?')[1]).get('review')
  return token?<BlindReview key={token} token={token}/>:<StudyManager/>
}

function StudyManager(){
  const [studies,setStudies]=useState<Study[]>([]),[selected,setSelected]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
  const [creating,setCreating]=useState(false),[loading,setLoading]=useState(true)
  const [title,setTitle]=useState('标注质量评估'),[size,setSize]=useState('200'),[reviewSize,setReviewSize]=useState('20'),[split,setSplit]=useState('validation')
  const [people,setPeople]=useState(['','',''])
  const [reviewAll,setReviewAll]=useState(false)
  const [requestId,setRequestId]=useState(()=>crypto.randomUUID().replaceAll('-','').slice(0,24))
  async function refresh(){const data=await get<{studies:Study[]}>('studies');setStudies(data.studies);setSelected(old=>old||data.studies[0]?.study_id||'')}
  useEffect(()=>{let live=true
    const load=()=>get<{studies:Study[]}>('studies').then(data=>{if(live){setStudies(data.studies);setSelected(old=>old||data.studies[0]?.study_id||'');setLoading(false)}}).catch(e=>{if(live){setError(errorMessage(e));setLoading(false)}})
    void load();const timer=window.setInterval(()=>{if(!document.hidden)void load()},5000)
    return()=>{live=false;clearInterval(timer)}
  },[])
  const study=studies.find(s=>s.study_id===selected)
  async function create(e:FormEvent){e.preventDefault();setBusy(true);setError('')
    try{const data=await post<Study>('studies',{request_id:requestId,title,sample_size:Number(size),seed:parseInt(requestId.slice(0,7),16),dataset_split:split,reviewers:people,
      review_sample_size:Number(reviewSize),review_all_deliveries:reviewAll})
      await refresh();setSelected(data.study_id);setCreating(false);setRequestId(crypto.randomUUID().replaceAll('-','').slice(0,24))
    }catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  async function control(action:string){if(!study)return;setBusy(true);setError('')
    try{await post(`studies/${study.study_id}/control`,{action});await refresh()}catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  return <div className="paper-workspace">
    <section className="paper-intro"><div><h2>质量评估</h2><p>运行固定批次，从自动交付的样本中随机抽取一批做人工盲审。</p></div><button className="primary" onClick={()=>setCreating(!creating)} disabled={busy}>{creating?'收起设置':'新建评估'}</button></section>
    {error&&<p className="error" role="alert">{error}</p>}
    {creating&&<form className="paper-card paper-create" onSubmit={create}><h3>设置评估</h3><div className="paper-form-grid">
      <label className="paper-wide">评估名称<input required maxLength={100} value={title} onChange={e=>setTitle(e.target.value)}/></label>
      <label>官方数据划分<select value={split} onChange={e=>setSplit(e.target.value)}><option value="validation">Validation 验证集</option><option value="train">Training 训练集</option></select></label>
      <label>随机任务数<input type="number" min={1} max={500} required value={size} onChange={e=>setSize(e.target.value)}/></label>
      <label>盲审抽样数<input type="number" min={1} max={500} required value={reviewSize} onChange={e=>setReviewSize(e.target.value)}/></label>
      <label><input type="checkbox" checked={reviewAll} onChange={e=>setReviewAll(e.target.checked)}/>审核全部交付样本，同时计算全量交付有效率</label>
      {people.map((person,i)=><label key={i}>{i===2?'第三人 · 分歧裁定':`独立审核者 ${i+1}`}<input required maxLength={80} placeholder="填写真实审核者标识" value={person} onChange={e=>setPeople(old=>old.map((v,n)=>n===i?e.target.value:v))}/></label>)}
    </div><p className="paper-note">样本不足时审核全部交付样本。点击开始后才调用模型，每个目标最多 12 次。</p>
    <button className="primary" disabled={busy}>{busy?'正在冻结…':'创建评估'}</button></form>}
    <div className="paper-layout"><aside className="paper-card paper-list"><h3>评估记录</h3>{loading?<p>正在读取…</p>:!studies.length?<p className="muted">还没有质量评估。</p>:studies.map(s=><button key={s.study_id} className={s.study_id===selected?'selected':''} onClick={()=>setSelected(s.study_id)}><strong>{s.spec.title}</strong><span>{s.spec.sample_size} 个任务 · {names[s.status==='ready'?s.batch.status:s.status]??s.status}</span><small>{new Date(s.created_at*1000).toLocaleDateString()}</small></button>)}</aside>
    <div className="paper-detail">{study?<>
      <section className="paper-card"><div className="paper-section-heading"><div><h3>{study.spec.title}</h3><p className="muted">{study.spec.dataset_split==='validation'?'官方验证集':'官方训练集'} · 固定 {study.spec.sample_size} 个任务 · 随机种子 {study.spec.seed}</p></div><span className="paper-status">{names[study.status==='ready'?study.batch.status:study.status]??study.status}</span></div>
        <p>已处理 {study.batch.finished??0} / {study.spec.sample_size} 个目标</p>
        {study.status!=='review'&&study.status!=='sealed'&&<div className="paper-actions"><button disabled={busy||['completed','blocked','running','queued'].includes(study.batch.status)} onClick={()=>void control('start')}>开始 / 继续生成</button><button disabled={busy||!['running','queued'].includes(study.batch.status)} onClick={()=>void control('pause')}>暂停实验</button><button className="primary" disabled={busy||!['completed','blocked'].includes(study.batch.status)} onClick={()=>void control('freeze')}>冻结结果，进入盲审</button></div>}
      </section>
      <section className="paper-card"><h3>独立审核入口</h3><p className="muted">每位审核者使用自己的链接。第一阶段选择所有可能对象，全部定位锁定后才显示预期目标；第二阶段核对事实。分歧出现后第三人才能开始裁定。</p><div className="paper-reviewers">{study.reviewers.map(r=><div key={r.role}><div><strong>{r.name}</strong><small>{r.role===2?'分歧裁定':`独立审核者 ${r.role+1}`}</small></div><span>定位 {r.localized} · 判断 {r.answered}</span>{study.status==='review'||study.status==='sealed'?<a href={r.url} target="_blank" rel="noreferrer">打开专属盲审页 ↗</a>:<span className="muted">等待生成结果冻结</span>}</div>)}</div>
        {study.status==='review'&&<button className="primary" disabled={busy} onClick={()=>void control('seal')}>完成裁定并封存报告</button>}</section>
      <section className="paper-card"><div className="paper-section-heading"><h3>评估指标</h3>{study.report&&<a href={`/api/research/studies/${study.study_id}/report.json`}>下载报告</a>}</div>
        <StudyReport report={study.report} completion={study.automatic_completion_rate}/></section>
    </>:<section className="paper-card paper-empty">新建评估后，即可在这里查看进度、盲审入口和指标。</section>}</div></div>
  </div>
}

function StudyReport({report,completion}:{report:Report|null;completion:Rate|null}){
  const quality=report?.quality??report?.methods?.agent
  const metrics=[
    {label:'自动完成率', rate:quality?.automatic_completion_rate??completion, detail:'自动交付数 ÷ 全部目标数'},
    {label:'交付有效率', rate:quality?.accepted_validity, detail:'盲审有效数 ÷ 全部交付数，需审核全部交付样本'},
    {label:'随机抽样通过率', rate:quality?.random_sample_pass_rate, detail:'抽中且盲审有效数 ÷ 实际抽中数'},
  ]
  return <div className="paper-stats">{metrics.map(({label,rate,detail})=><div key={label}>
    <strong>{pct(rate??null)}</strong><span>{label}{rate&&`（${rate.numerator} / ${rate.denominator}）`}</span><small>{detail}</small>
  </div>)}</div>
}

function BlindReview({token}:{token:string}){
  const [inbox,setInbox]=useState<Inbox|null>(null),[cid,setCid]=useState(''),[data,setData]=useState<BlindCase|null>(null)
  const [selection,setSelection]=useState<string[]>([]),[focus,setFocus]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState('')
  const [answers,setAnswers]=useState<Record<string,string>>({}),[notes,setNotes]=useState(''),[started,setStarted]=useState(()=>Date.now())
  useEffect(()=>{let live=true;get<Inbox>(`review/${token}`).then(v=>{if(live){setInbox(v);setCid(v.cases.find(c=>v.phase==='localization'?!c.localized:!c.answered)?.case_id??v.cases[0]?.case_id??'')}}).catch(e=>{if(live)setError(errorMessage(e))});return()=>{live=false}},[token])
  useEffect(()=>{let live=true
    if(cid)void get<BlindCase>(`review/${token}/cases/${cid}`).then(v=>{if(live){setData(v);setError('');setFocus('');setStarted(Date.now());setSelection(v.selection??[]);setAnswers(v.answer?Object.fromEntries(Object.entries(v.answer).filter(([,value])=>typeof value==='boolean').map(([k,v])=>[k,String(v)])):{});setNotes(v.answer?.notes??'')}}).catch(e=>{if(live)setError(errorMessage(e))})
    return()=>{live=false}
  },[token,cid,inbox?.phase])
  async function submit(kind:'localization'|'answer'){setBusy(true);setError('')
    try{const v=await post<Inbox>(`review/${token}/cases/${cid}/${kind}`,kind==='localization'?{selected_labels:selection,elapsed_seconds:(Date.now()-started)/1000}:{...Object.fromEntries(Object.entries(answers).map(([k,v])=>[k,v==='true'])),notes,elapsed_seconds:(Date.now()-started)/1000})
      setInbox(v);setCid(v.cases.find(c=>v.phase==='localization'?!c.localized:!c.answered)?.case_id??cid)
      if(v.cases.every(c=>c.answered))setData(await get<BlindCase>(`review/${token}/cases/${cid}`))
    }catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  const locked=data?.selection!=null
  function toggle(id:string){setFocus(id);if(!locked)setSelection(old=>old.includes(id)?old.filter(v=>v!==id):[...old,id])}
  const fields=[['factual_correct','表达中的事实是否都正确？'],['unique','表达是否只符合一个对象，没有其他可能对象？'],['input_sufficient','只凭这张原图与同帧点云，是否足以理解表达？'],['prior_target_exposure','本轮锁定定位之前，你是否已知道预期目标？']]
  return <div className="paper-workspace paper-blind"><section className="paper-intro"><div><span className="paper-eyebrow">INDEPENDENT BLIND REVIEW</span><h2>{inbox?.title??'独立人工盲审'}</h2><p>{inbox?`${inbox.reviewer} · ${inbox.role===2?'第三人裁定':'独立审核'} · ${inbox.phase==='localization'?'第一阶段：先找出所有可能对象':'第二阶段：核对预期目标及描述事实'}`:'正在读取审核任务…'}</p></div></section>
    {error&&<p role="alert" className="error">{error}</p>}
    {inbox?.waiting?<section className="paper-card paper-empty"><h3>等待两名审核者完成</h3><p>独立判断全部提交后，这里仅提供有分歧的表达。</p><button onClick={()=>window.location.reload()}>刷新进度</button></section>:inbox&&!inbox.cases.length?<section className="paper-card paper-empty"><h3>当前没有需要审核的表达</h3><p>可能没有生成文本，或两位审核者没有分歧。无输出任务仍保留在实验分母。</p></section>:<>
    {inbox&&<section className="paper-card paper-review-progress"><label>审核条目<select value={cid} onChange={e=>setCid(e.target.value)}>{inbox.cases.map((c,i)=><option key={c.case_id} value={c.case_id}>{i+1} / {inbox.cases.length} · {c.answered?'已判断':c.localized?'已锁定定位':'待定位'}</option>)}</select></label><span>已定位 {inbox.cases.filter(c=>c.localized).length} · 已判断 {inbox.cases.filter(c=>c.answered).length}</span></section>}
    {data?.case_id===cid&&<section className="paper-card"><blockquote className="paper-expression">{data.expression}</blockquote><p className="paper-note">{data.revealed?`预期目标：${data.expected}。你之前锁定的候选：${data.selection?.join('、')||'没有匹配对象'}。请独立判断事实和唯一性。`:'请选择所有符合表达的对象；可多选，也可明确提交空选。预期目标、生成方法和系统判断均已隐藏。全部条目定位锁定前不会揭示目标。'}</p>
      <TaskScene key={cid} data={{...data,job_id:'',ordinal:0,revision:'',run_id:'',can_confirm:false,review:true,selected_candidate_id:null,confirmation:null,image_box_source:'official_lidar_box_projection'}} selected={focus} onSelect={id=>{setFocus(id)}}/>
      <div className="paper-candidates">{data.candidates.map(c=><label key={c.candidate_id} className={selection.includes(c.candidate_id)?'checked':''}><input type="checkbox" disabled={locked||busy||inbox?.status==='sealed'} checked={selection.includes(c.candidate_id)} onChange={()=>toggle(c.candidate_id)}/>{c.candidate_id}<small>{categoryLabel(c.category)}</small></label>)}</div>
      <p className="muted">当前勾选：{selection.join('、')||'无'}。三维窗口只高亮你点选查看的对象，不自动勾选。</p><p className="muted">点云窗口为目标无关预览；完整点数 {number(data.point_count)}。<a href={data.full_cloud}>下载完整单帧点云</a></p>
      {!data.revealed?<><button className="primary" disabled={locked||busy||inbox?.status==='sealed'} onClick={()=>void submit('localization')}>{busy?'正在保存…':selection.length?'锁定所选对象，下一条':'确认无匹配对象，下一条'}</button>{locked&&<p>定位已锁定；完成其余条目后进入事实核对。</p>}</>:<form onSubmit={e=>{e.preventDefault();void submit('answer')}} className="paper-judgment"><h3>独立事实判断</h3>{fields.map(([key,label])=><fieldset key={key} disabled={!!data.answer||busy||inbox?.status==='sealed'}><legend>{label}</legend>{['true','false'].map(value=><label key={value}><input type="radio" name={key} required value={value} checked={answers[key]===value} onChange={()=>setAnswers(old=>({...old,[key]:value}))}/>{value==='true'?'是':'否'}</label>)}</fieldset>)}<label>原因与备注<textarea maxLength={2000} value={notes} onChange={e=>setNotes(e.target.value)} disabled={!!data.answer} placeholder="指出错误事实、其他可能对象或输入缺失"/></label><button className="primary" disabled={busy||!!data.answer||inbox?.status==='sealed'}>{data.answer?'判断已保存':busy?'正在保存…':'提交独立判断，下一条'}</button></form>}
    </section>}</>}
  </div>
}
