import { phases, reasonLabel } from '../../shared/annotationDisplay'
import { taskUrl } from '../../navigation'

export type ReviewRetryProgress = {
  ordinal:number; status:string; reason:string|null; requested_at:number|null
  packet_id:string|null; run_id:string|null; phase:string|null; preparation_stage:string|null
  calls:number; call_budget:number; model_pending:boolean; automatic_passed:boolean
  manual_approved?:boolean
  feedback_check?:{passed:boolean|null;historical?:boolean;blocking_errors?:string[];requirements:{kind:string;label:string;status:string;reason:string;required?:boolean}[]}|null
  previous_expression?:string|null; feedback?:string|null
  attempt_limit?:number
  attempts?:{expression:string;errors:string[];unsupported_claims?:string[];native_remaining:number|null;image_remaining:number|null;ground_rounds:number;binding_retries?:number}[]
}
export type ReworkState = {job_id:string;ordinal:number;item:ReviewRetryProgress|null;task?:TaskState;measured_at?:number}
type TaskState = {status:string; executor_active:boolean; control:string|null; reason:string|null; queue_paused?:boolean}
const checks = ['appearance','ground','ground_permuted','scene_ground','scene_ground_permuted','counterfactual']
const preparation = ['precheck_target','precheck_candidates','precheck_decision','recheck_source_attributes','recheck_feedback_landmarks','relook_candidates','observe_candidates']

function reworkPresentation(item:ReviewRetryProgress, task:TaskState) {
  const settled = !['pending','preparing','running'].includes(item.status)
  const step = item.automatic_passed ? 4 : item.status==='pending' ? 0
    : item.status==='preparing'||!item.phase||preparation.includes(item.phase) ? 1 : checks.includes(item.phase) ? 3 : 2
  if (item.automatic_passed && item.feedback_check?.passed===false) return item.feedback_check.blocking_errors?.length===0
    ? {step, state:'done', text:'文本核验通过 · 部分建议未采用'}
    : {step, state:'failed', text:'文本核验通过 · 反馈未满足'}
  if (item.automatic_passed) return {step, state:'done', text:'文本核验通过 · 反馈请人工确认'}
  if (settled) return {step, state:'failed', text:item.manual_approved?'自动未通过 · 已人工审批':'返工未通过'}
  if (task.control || ['paused','stopped','blocked'].includes(task.status)) return {step, state:'waiting',
    text:task.control ? '正在保存步骤，随后暂停或停止' : task.status==='blocked' ? '返工执行受阻' : task.status==='stopped' ? '返工已停止' : '返工已暂停'}
  if (task.status==='queued') return {step:0, state:'waiting', text:task.queue_paused ? '返工排队已暂停' : '返工排队等待执行'}
  if (!task.executor_active) return {step, state:'waiting', text:'返工已保存，等待启动原任务'}
  if (item.status==='pending') return {step, state:'waiting', text:'等待执行器安排返工'}
  return {step, state:'running', text:item.status==='preparing' ? '正在准备返工证据' : phases[item.phase??'']??'正在根据反馈返工'}
}

const errorLabels:Record<string,string> = {
  declared_expression_not_unique:'描述仍匹配多个或尚未排除的对象，不能唯一定位。',
  text_claims_disagree_with_declared_predicates:'实际文本与模型声明的定位条件不一致，需要同步改写文字和条件。',
  manual_review_rejected_text_repeated:'重复了已被人工否决的原描述。',
}

export default function ReworkProgress({items, task, measuredAt, preparationLabel=()=> '准备图像与点云', failureLabel=reasonLabel, jobId}: {
  items:ReviewRetryProgress[]; task:TaskState; measuredAt?:number
  preparationLabel?:(stage:unknown)=>string; jobId?:string
  failureLabel?:(value:string)=>string
}) {
  if (!items.length) return null
  return <section className="sample-rework rework-progress" aria-label="反馈修改进度">
    <div className="section-heading"><h2>修改进度</h2>
      {measuredAt&&<span className="muted">更新于 {new Date(measuredAt*1000).toLocaleTimeString('zh-CN',{hour12:false})}</span>}</div>
    {items.map(item=>{
      const view = reworkPresentation(item,task)
      const failure = item.reason || (view.state==='waiting' ? task.reason : null)
      const accepted = item.automatic_passed ? item.attempts?.at(-1) : undefined
      return <article key={item.ordinal} className={`rework-record ${view.state}`} aria-label={`样本 #${item.ordinal} 返工`}>
        <div className="section-heading"><h3>样本 #{item.ordinal}</h3><strong className="sample-stage" role="status">{view.text}</strong></div>
        <ol className="sample-steps" aria-label={`样本 #${item.ordinal} 返工步骤`}>{['等待执行','准备与观察','改写描述','自动核验'].map((label,index)=><li key={label}
          aria-current={index===view.step?'step':undefined} className={index<view.step?'done':index===view.step?view.state==='failed'?'failed':'current':''}>
          <span aria-hidden="true">{index<view.step?'✓':index+1}</span>{label}</li>)}</ol>
        {view.state==='running'&&<p className="muted" role="status">{item.status==='preparing'?preparationLabel(item.preparation_stage):item.model_pending?'模型请求处理中，正在等待返回。':'正在处理当前步骤。'}</p>}
        {failure&&!item.automatic_passed&&<p className={view.state==='failed'&&!item.manual_approved?'error':'notice'} role="status">{item.manual_approved?'当前版本已由人工审批纳入可用集，自动核验的原失败记录保留。':failureLabel(failure)}</p>}
        {view.state==='failed'&&!item.automatic_passed&&item.attempt_limit&&item.attempts&&item.attempts.length>=item.attempt_limit&&<p className="muted">已达到本轮最多 {item.attempt_limit} 版草稿（含首稿）；剩余调用额度不会自动增加改写次数。</p>}
        {view.state==='waiting'&&<p className="muted">{task.status==='queued'?'前序任务结束后接续；排队暂停时需先继续队列。':task.executor_active&&!task.control?'执行器会在有空位时处理这条返工。':'请在原任务中开始或继续；无需再次提交反馈。'} {jobId&&<a href={taskUrl(jobId)}>打开任务控制</a>}</p>}
        {accepted&&<div className="rework-feedback"><strong>修改后描述 · 自动核验通过</strong><p lang="en">{accepted.expression}</p></div>}
        {item.feedback&&<div className="rework-feedback"><strong>本次人工反馈</strong><p>{item.feedback}</p></div>}
        {item.feedback_check&&<div className="rework-feedback"><strong>反馈要求检查</strong>
          {item.feedback_check.requirements.map(requirement=><div key={`${requirement.kind}:${requirement.label}`} className={`feedback-requirement ${requirement.status}`}>
            <span>{requirement.label}{requirement.required===false?'（建议）':''}：{({covered:requirement.kind==='concision'?'表达检查通过':'已写入描述',missing:requirement.required===false?'未采用':'未落实',unsupported:'暂不支持核验',pending:'待检查',not_verified:'未完成核验'} as Record<string,string>)[requirement.status]??requirement.status}</span>
            {requirement.reason&&(requirement.status==='missing'
              ? <p className="muted">{requirement.reason}</p>
              : <details className="inline-help"><summary>说明</summary><p>{requirement.reason}</p></details>)}
          </div>)}
          {!item.feedback_check.requirements.length&&<p className="muted">这项反馈需人工确认是否落实。</p>}
          {item.feedback_check.historical&&<details className="inline-help"><summary>历史记录说明</summary><p>以上为补充解读，原核验结论、状态和调用记录保留。</p></details>}
        </div>}
        {item.previous_expression&&<details className="rework-previous"><summary>修改前的描述</summary><p lang="en">{item.previous_expression}</p></details>}
        {item.attempts?.map((attempt,index)=>accepted===attempt ? null : <details key={index} className="rework-attempt">
          <summary>{accepted?'历史草稿':'返工草稿'} {index+1}{attempt.errors.length?' · 未通过':item.automatic_passed&&index===item.attempts!.length-1?' · 自动核验通过':' · 核验中'}</summary>
          <p lang="en">{attempt.expression}</p>
          {!!attempt.errors.length&&<ul className="check-error-list">{[...new Set(attempt.errors)].map(error=><li key={error}>{errorLabels[error]??failureLabel(error)}</li>)}</ul>}
          {!!attempt.unsupported_claims?.length&&<p className="muted">尚无证据支持的文本片段：{attempt.unsupported_claims.join('；')}</p>}
          {attempt.errors.some(error=>['declared_expression_not_unique','parsed_expression_not_symbolically_unique','image_expression_not_unique'].includes(error))&&<p className="muted">尚未排除：{attempt.native_remaining??'数量未知的'} 个三维物体、{attempt.image_remaining??'数量未知的'} 个图像物体；其中可能包含未确认的对象。</p>}
          <p className="muted">已完成 {attempt.ground_rounds} 轮定位核验{attempt.binding_retries ? ` · 编号复核 ${attempt.binding_retries} 次` : ''}</p>
        </details>)}
        <div className="rework-record-footer"><span className="muted">{item.run_id?`本次运行模型调用 ${item.calls} / ${item.call_budget} 次`:'尚未进入模型调用阶段'}</span>
          {view.state==='failed'&&!item.automatic_passed&&<span className="muted">{item.manual_approved?'当前版本已人工审批，可纳入后续导出。':item.attempts?.length?'失败草稿保留供人工判断；人工审批通过后可纳入可用集。':'本轮未生成新草稿；返工前版本仅供查看，不能直接审批。'}</span>}</div>
      </article>
    })}
  </section>
}
