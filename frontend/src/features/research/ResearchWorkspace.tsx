import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { readJson, errorMessage, categoryLabel } from '../../shared/annotationDisplay'
import TaskScene from '../results/TaskScene'
import type { TaskSceneData } from '../results/TaskScene'
import './research.css'

type Rate = { numerator:number; denominator:number; value:number|null; ci95:number[]|null }
type Metrics = { tasks:number;accepted:number;valid_automatically_completed:number;accepted_validity:Rate|null;automatic_valid_coverage:Rate|null;automatic_completion_rate:Rate;
  exposure_excluded:number;cost:Record<string,number|null>;cost_per_valid:Record<string,number|null>;review_seconds:number }
type Report = { methods:Record<string,Metrics>;paired_coverage_difference:number|null;paired_context_bootstrap_ci95:number[]|null;disagreements:number;contexts:number;limitations:string[] }
type Study = { study_id:string;job_id:string;status:string;created_at:number;spec:{ title:string;sample_size:number;seed:number;dataset_split:string;reviewers:string[] };cases:number;
  reviewers:{name:string;role:number;url:string;localized:number;answered:number}[];batch:{status:string;calls:number;finished:number;reason?:string};report:Report|null }
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
  const [title,setTitle]=useState('Agent 与固定流程 VLM 对照'),[size,setSize]=useState('200'),[seed,setSeed]=useState('42'),[split,setSplit]=useState('validation')
  const [people,setPeople]=useState(['','','']),[inputPrice,setInputPrice]=useState(''),[outputPrice,setOutputPrice]=useState('')
  const [requestId,setRequestId]=useState(()=>crypto.randomUUID().replaceAll('-','').slice(0,24))
  async function refresh(){const data=await get<{studies:Study[]}>('studies');setStudies(data.studies);setSelected(old=>old||data.studies[0]?.study_id||'')}
  useEffect(()=>{let live=true
    const load=()=>get<{studies:Study[]}>('studies').then(data=>{if(live){setStudies(data.studies);setSelected(old=>old||data.studies[0]?.study_id||'');setLoading(false)}}).catch(e=>{if(live){setError(errorMessage(e));setLoading(false)}})
    void load();const timer=window.setInterval(()=>{if(!document.hidden)void load()},5000)
    return()=>{live=false;clearInterval(timer)}
  },[])
  const study=studies.find(s=>s.study_id===selected)
  async function create(e:FormEvent){e.preventDefault();setBusy(true);setError('')
    try{const data=await post<Study>('studies',{request_id:requestId,title,sample_size:Number(size),seed:Number(seed),dataset_split:split,reviewers:people,
      input_price_per_million:inputPrice===''?null:Number(inputPrice),output_price_per_million:outputPrice===''?null:Number(outputPrice)})
      await refresh();setSelected(data.study_id);setCreating(false);setRequestId(crypto.randomUUID().replaceAll('-','').slice(0,24))
    }catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  async function control(action:string){if(!study)return;setBusy(true);setError('')
    try{await post(`studies/${study.study_id}/control`,{action});await refresh()}catch(e){setError(errorMessage(e))}finally{setBusy(false)}
  }
  return <div className="paper-workspace">
    <section className="paper-intro"><div><span className="paper-eyebrow">PAPER EVIDENCE</span><h2>用独立盲审衡量真实质量</h2><p>同一批随机任务、同一模型与初始证据，比较完整 Agent 和固定流程 VLM。有效表达须同时满足事实正确、唯一定位、单帧可理解。</p></div><button className="primary" onClick={()=>setCreating(!creating)} disabled={busy}>{creating?'收起设置':'新建对照实验'}</button></section>
    <ol className="paper-steps"><li><b>01</b><span>冻结随机任务<small>失败不替换，分母不变</small></span></li><li><b>02</b><span>运行两种方法<small>保存请求、重试和拒绝</small></span></li><li><b>03</b><span>两人独立盲审<small>先定位，后核对事实</small></span></li><li><b>04</b><span>裁定与导出<small>逐条记录与论文指标</small></span></li></ol>
    {error&&<p className="error" role="alert">{error}</p>}
    {creating&&<form className="paper-card paper-create" onSubmit={create}><h3>预先登记实验</h3><div className="paper-form-grid">
      <label className="paper-wide">实验名称<input required maxLength={100} value={title} onChange={e=>setTitle(e.target.value)}/></label>
      <label>官方数据划分<select value={split} onChange={e=>setSplit(e.target.value)}><option value="validation">Validation 验证集</option><option value="train">Training 训练集</option></select></label>
      <label>随机任务数<input type="number" min={1} max={500} required value={size} onChange={e=>setSize(e.target.value)}/></label>
      <label>抽样随机种子<input type="number" min={0} max={2147483647} required value={seed} onChange={e=>setSeed(e.target.value)}/></label>
      {people.map((person,i)=><label key={i}>{i===2?'第三人 · 分歧裁定':`独立审核者 ${i+1}`}<input required maxLength={80} placeholder="填写真实审核者标识" value={person} onChange={e=>setPeople(old=>old.map((v,n)=>n===i?e.target.value:v))}/></label>)}
    </div><details><summary>可选：模型单价，用于估算货币成本</summary><div className="paper-form-grid"><label>每百万输入 token 单价<input type="number" min={0} step="any" value={inputPrice} onChange={e=>setInputPrice(e.target.value)}/></label><label>每百万输出 token 单价<input type="number" min={0} step="any" value={outputPrice} onChange={e=>setOutputPrice(e.target.value)}/></label></div><p className="muted">两项使用相同货币单位。估算不等于账单费用；未填写或 usage 不完整时只报告调用量。</p></details>
    <p className="paper-note">冻结当前模型与代码。每任务 Agent 最多 12 次，固定流程基线 1 次，共最多 {Number(size)*13||0} 次；基线额外调用单独留账。创建只抽样和预留，点击开始后才调用模型；执行器忙时按现有队列等待。</p>
    <button className="primary" disabled={busy}>{busy?'正在冻结…':'冻结任务并创建实验'}</button></form>}
    <div className="paper-layout"><aside className="paper-card paper-list"><h3>我的实验</h3>{loading?<p>正在读取…</p>:!studies.length?<p className="muted">还没有科研实验。创建实验后，进度与审核入口将保存在这里。</p>:studies.map(s=><button key={s.study_id} className={s.study_id===selected?'selected':''} onClick={()=>setSelected(s.study_id)}><strong>{s.spec.title}</strong><span>{s.spec.sample_size} 个任务 · {names[s.status==='ready'?s.batch.status:s.status]??s.status}</span><small>{new Date(s.created_at*1000).toLocaleDateString()}</small></button>)}</aside>
    <div className="paper-detail">{study?<>
      <section className="paper-card"><div className="paper-section-heading"><div><h3>{study.spec.title}</h3><p className="muted">{study.spec.dataset_split==='validation'?'官方验证集':'官方训练集'} · 固定 {study.spec.sample_size} 个任务 · 随机种子 {study.spec.seed}</p></div><span className="paper-status">{names[study.status==='ready'?study.batch.status:study.status]??study.status}</span></div>
        <div className="paper-stats"><div><strong>{study.batch.finished??0} / {study.spec.sample_size}</strong><span>已处理任务</span></div><div><strong>{number(study.batch.calls)}</strong><span>累计调用 · 含失败和基线</span></div><div><strong>{study.cases}</strong><span>冻结的逐方法结果</span></div></div>
        {study.status!=='review'&&study.status!=='sealed'&&<div className="paper-actions"><button disabled={busy||['completed','blocked','running','queued'].includes(study.batch.status)} onClick={()=>void control('start')}>开始 / 继续生成</button><button disabled={busy||!['running','queued'].includes(study.batch.status)} onClick={()=>void control('pause')}>暂停实验</button><button className="primary" disabled={busy||!['completed','blocked'].includes(study.batch.status)} onClick={()=>void control('freeze')}>冻结结果，进入盲审</button></div>}
        <p className="muted">基线按固定提示生成一次，不进行自主补看、修复或质量筛选。Agent 可补证并拒绝。准备失败、无输出和拒绝均保留在全部任务分母。</p>
      </section>
      <section className="paper-card"><h3>独立审核入口</h3><p className="muted">每位审核者使用自己的链接。第一阶段选择所有可能对象，全部定位锁定后才显示预期目标；第二阶段核对事实。分歧出现后第三人才能开始裁定。</p><div className="paper-reviewers">{study.reviewers.map(r=><div key={r.role}><div><strong>{r.name}</strong><small>{r.role===2?'分歧裁定':`独立审核者 ${r.role+1}`}</small></div><span>定位 {r.localized} · 判断 {r.answered}</span>{study.status==='review'||study.status==='sealed'?<a href={r.url} target="_blank" rel="noreferrer">打开专属盲审页 ↗</a>:<span className="muted">等待生成结果冻结</span>}</div>)}</div>
        {study.status==='review'&&<button className="primary" disabled={busy} onClick={()=>void control('seal')}>完成裁定并封存报告</button>}</section>
      <section className="paper-card"><div className="paper-section-heading"><h3>论文证据</h3>{study.report&&<div className="paper-actions"><a href={`/api/research/studies/${study.study_id}/report.csv`}>逐条结果 CSV</a><a href={`/api/research/studies/${study.study_id}/report.json`}>完整证据 JSON</a></div>}</div>
        {study.report?<StudyReport report={study.report}/>:<div className="paper-empty"><strong>等待独立人工判断</strong><p>这里将分别报告已接受表达的有效率，以及全部任务中有效且自动完成的比例。自动核验通过数不会代替人工结果。</p></div>}</section>
    </>:<section className="paper-card paper-empty"><strong>把质量论证落到逐条证据</strong><p>创建一个对照实验，冻结任务和方法后开始生成。这里会持续保存运行状态、独立审核进度与报告。</p><p>建议先用少量任务走通审核流程，再冻结约 200 个未处理任务开展正式初评。</p></section>}</div></div>
  </div>
}

function StudyReport({report}:{report:Report}){return <><div className="paper-table-wrap"><table><thead><tr><th>指标</th><th>完整 Agent</th><th>固定流程 VLM</th></tr></thead><tbody>{[
  ['已接受表达的有效率',(m:Metrics)=>pct(m.accepted_validity)],['全部任务的自动有效覆盖率',(m:Metrics)=>pct(m.automatic_valid_coverage)],
  ['自动完成率',(m:Metrics)=>pct(m.automatic_completion_rate)],['自动有效 / 全部任务',(m:Metrics)=>`${m.valid_automatically_completed} / ${m.tasks}`],
  ['总调用次数',(m:Metrics)=>number(m.cost.calls)],['每条自动有效表达的调用次数',(m:Metrics)=>number(m.cost_per_valid.calls)],
  ['估算总金额',(m:Metrics)=>number(m.cost.estimated_cost)],['每条自动有效表达的估算金额',(m:Metrics)=>number(m.cost_per_valid.estimated_cost)],
  ['生成中的人工介入',(m:Metrics)=>number(m.cost.generation_human_interventions)],['人工盲审时间（分钟）',(m:Metrics)=>number(m.review_seconds/60)],
  ['提前见过目标的排除记录',(m:Metrics)=>number(m.exposure_excluded)],
  ].map(([label,value])=><tr key={label as string}><th>{label as string}</th>{['agent','baseline'].map(method=><td key={method}>{(value as (m:Metrics)=>string)(report.methods[method])}</td>)}</tr>)}</tbody></table></div>
  <p>配对自动有效覆盖率差异：<strong>{report.paired_coverage_difference==null?'未测量':`${(report.paired_coverage_difference*100).toFixed(1)} 个百分点`}</strong>。{report.paired_context_bootstrap_ci95?`按场景重采样的 95% 区间：${report.paired_context_bootstrap_ci95.map(n=>(n*100).toFixed(1)).join(' 至 ')} 个百分点。`:'场景不足或存在提前暴露记录，未计算区间。'}</p><p className="muted">{report.disagreements} 条分歧 · {report.contexts} 个场景</p><details><summary>统计口径与适用范围</summary>{report.limitations.map(t=><p key={t}>{t}</p>)}</details></>}

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
