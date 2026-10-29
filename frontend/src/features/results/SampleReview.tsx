import { useEffect, useRef, useState } from 'react'
import { errorMessage, readJson, manualReviewLabels } from '../../shared/annotationDisplay'
import { sampleUrl } from '../../navigation'

export type ManualReview = {
  review_id: string; revision: string; verdict: 'accepted' | 'rejected' | 'discarded'; notes: string
  approval_policy?:string
  job_id?:string; ordinal?:number; run_id?:string
}

type Props = {
  jobId: string; ordinal: number; revision: string | null; canReview: boolean
  canRetry?: boolean; retryFailedRunId?: string
  decision: ManualReview | null; localizationCorrect: boolean; busy: boolean
  onBusy: (value: boolean) => void; onReviewed: (value: ManualReview) => void
  onOpenRetry: (jobId: string) => void
  onRetry?: (value: { job_id: string; ordinal: number; started: boolean; message?: string }) => void
  repairMode?: boolean; onClose?: () => void; taskGoalReached?: boolean
  automaticPassed?:boolean
  approvalInHeader?:boolean
}

export default function SampleReview({ jobId, ordinal, revision, canReview, canRetry = canReview, retryFailedRunId, decision,
  localizationCorrect, busy, onBusy, onReviewed, onOpenRetry, onRetry, repairMode = false, onClose, taskGoalReached = false, automaticPassed = true, approvalInHeader = false }: Props) {
  const [notes, setNotes] = useState(decision?.notes ?? '')
  const [error, setError] = useState('')
  const [submittingRetry, setSubmittingRetry] = useState(false)
  const [waiting, setWaiting] = useState(false)
  const [retry, setRetry] = useState<{ job_id: string; started: boolean; message?: string } | null>(null)
  const notesField = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { if (repairMode) notesField.current?.focus() }, [repairMode])
  useEffect(() => {
    if (!submittingRetry) return
    const timer = setTimeout(() => setWaiting(true), 15000)
    return () => clearTimeout(timer)
  }, [submittingRetry])

  async function record(verdict: ManualReview['verdict']) {
    return await readJson(await fetch(`/api/batches/${jobId}/samples/${ordinal}/review`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ revision, verdict, notes: notes.trim(), ...(verdict==='accepted'&&!automaticPassed?{approve_failed:true}:{}) }),
    })) as ManualReview
  }

  async function save(verdict: ManualReview['verdict']) {
    if (!revision) return
    onBusy(true); setError('')
    try {
      const value = await record(verdict)
      onReviewed(value)
    } catch (e) { setError(errorMessage(e)) } finally { onBusy(false) }
  }

  async function regenerate() {
    if (!revision || !canRetry || !notes.trim()) return
    if (taskGoalReached) { setError('本任务已达到目标条数，请先手动删除一条有效标注，腾出名额后再返工。'); return }
    setSubmittingRetry(true); setWaiting(false)
    onBusy(true); setError('')
    try {
      // The server checks the task's available slot before rejecting this version.
      const value = await readJson(await fetch(`/api/batches/${jobId}/samples/${ordinal}/retry`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ revision, notes: notes.trim(), ...(retryFailedRunId?{retry_failed_run_id:retryFailedRunId}:{}) }),
      }))
      setRetry(value)
      onRetry?.({ ...value, ordinal })
      if (value.manual_review) onReviewed(value.manual_review)
    } catch (e) { setError(errorMessage(e)) } finally { setSubmittingRetry(false); onBusy(false) }
  }

  const unavailable = busy || !revision || !canReview
  return <section className={`sample-review${repairMode ? ' sample-rework' : ''}`} aria-label={repairMode ? '样本反馈修改' : '人工审核结论'}>
    <div className="section-heading"><h3>{repairMode || approvalInHeader ? '反馈修改' : '人工审核'}</h3>
      <strong className={`review-status ${decision?.verdict ?? ''}`}>{decision ? manualReviewLabels[decision.verdict] : '待审核'}</strong>
      {onClose && <button className="text-button" disabled={busy} onClick={onClose}>收起</button>}
    </div>
    {repairMode && <p className="rework-guidance">说明哪里不准确，系统会重新观察并修改。</p>}
    <label>{repairMode || approvalInHeader ? '问题与修改建议' : '审核意见'}<textarea ref={notesField} value={notes} maxLength={2000} disabled={busy}
      placeholder="例如：应为白色轿车；删除不准确的前后关系。"
      onChange={e => setNotes(e.target.value)} /></label>
    {!repairMode && !localizationCorrect && <p className="muted">请在图像或点云中选择物体，在场景上方确认定位并提交审批；无法唯一定位时，可填写反馈返工或废弃当前版本。</p>}
    {!canReview && <p className="muted">{canRetry?'上轮返工已结束但未生成新描述，可以针对当前展示的旧描述再次提交反馈返工；旧版本不能直接审批。':'当前版本暂不可提交审核或返工，请查看下方返工状态。'}</p>}
    <div className="controls">
      {!repairMode && !approvalInHeader && <button className={!automaticPassed?'primary':undefined} disabled={unavailable || !localizationCorrect} onClick={() => void save('accepted')}>{automaticPassed?'审核通过':'人工审批通过并纳入可用集'}</button>}
      <button className="primary" disabled={busy || !revision || !canRetry || !notes.trim() || !!retry || taskGoalReached} onClick={() => void regenerate()}>{submittingRetry ? '正在提交返工…' : '提交反馈并修改'}</button>
    </div>
    {!repairMode&&!automaticPassed&&<p className="muted">人工审批将当前描述纳入可用集和后续导出；自动未通过的结论与原因保留。请结合完整图像和点云核对描述，并先正确确认物体定位。</p>}
    {submittingRetry && <p className="notice" role="status" aria-live="polite">{waiting
      ? '后台仍未返回返工确认，反馈可能已保存。请勿重复点击；可刷新当前样本查看返工状态。'
      : '正在保存反馈并安排原任务返工，后台确认后会显示结果…'}</p>}
    {taskGoalReached && <p className="notice">本任务已达到目标条数，该条暂不能返工；请先手动删除一条有效标注，腾出名额。</p>}
    <details className="inline-help rework-budget"><summary>修改规则与调用额度</summary><p>返工会保存“人工不通过”结论和反馈，并将反馈交给模型修正，为本次返工分配最多 12 次新调用额度，在当前任务内重试，通过后计入本任务合格数量。达到任务目标时须先手动删除一条有效标注，腾出名额。同一次反馈重复提交不重复加额，历史调用与费用保留。新描述仍须通过自动核验。</p></details>
    {retry && <div className="notice" role="status"><span>{retry.message ?? (retry.started ? '系统 已在原任务内重试；原样本及审核意见保留。' : '该样本已加入原任务等待重试。')}</span>
      <button disabled={busy} onClick={() => onOpenRetry(retry.job_id)}>查看样本返工</button>
      {!retry.started && <a href={sampleUrl(retry.job_id,ordinal)}>查看该样本进度</a>}
    </div>}
    {error && <p className="error" role="alert">{error}</p>}
  </section>
}
