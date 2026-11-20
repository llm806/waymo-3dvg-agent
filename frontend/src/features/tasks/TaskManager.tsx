import { serviceText, t, useLocale } from '../../shared/locale.ts'
import type { InterfaceMessage } from '../../shared/locale.ts'
import { useEffect, useRef, useState } from 'react'
import { errorMessage, readJson } from '../../shared/annotationDisplay'
import { taskName, taskDetails } from '../../shared/taskName'

type Task = { job_id: string; status: string; created_at: number; counts: Record<string, number>; executor_active: boolean; queue_position?: number | null }
type Review = { job_ids: string[]; requested: string[]; token: string; fromStudy?: string;
  return_to_pool: boolean; processed_tracks: number; candidate_tokens: { return: string; exclude: string };
  research_drafts?: { study_id: string; title: string; reserved_targets: number }[];
  shared_cleanup_deferred?: boolean;
  summary: { tasks: number; retained_tasks?: number; runs: number; model_calls: number; cohorts: number; releases: number; evidence: number; scenes: number };
  retained_shared: { runs: number; evidence: number; scenes: number } }
type DeletionProgress = { operation_id: string; status: 'running' | 'completed' | 'failed';
  stage: string; message: string; completed: number; total: number; files_checked: number; files_removed?: number;
  result: Review | { deleted_job_ids: string[]; cleanup_pending: boolean } | null }

export function useTaskDeletion({ onDeleted, isActive, studyId }: {
  onDeleted: (ids: string[]) => void;
  isActive: (id: string) => boolean;
  studyId?: string;
}) {
  const [review, setReview] = useState<Review | null>(null)
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [feedback, setFeedback] = useState<InterfaceMessage>('')
  const [pending, setPending] = useState(false)
  const [progress, setProgress] = useState<DeletionProgress | null>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const trigger = useRef<HTMLElement | null>(null)
  const reviewActive = review?.job_ids.some(isActive)
  useEffect(() => { if ((review || busy) && !dialog.current?.open) dialog.current?.showModal() }, [review, busy])
  function closeReview(restoreFocus = true) {
    dialog.current?.close()
    setReview(null); setError(''); setProgress(null)
    if (restoreFocus && trigger.current?.isConnected) trigger.current.focus()
  }

  async function operation(action: 'preview' | 'delete', values: Record<string, unknown>) {
    const response = await fetch('/api/batches/deletion/operations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...values, action }),
    })
    if (response.status === 404) throw new Error(t("后端尚未加载删除进度功能，请重启后端并刷新页面后再操作。"))
    let value = await readJson(response) as DeletionProgress
    setProgress(value)
    while (value.status === 'running') {
      await new Promise(resolve => setTimeout(resolve, 350))
      value = await readJson(await fetch(`/api/batches/deletion/operations/${value.operation_id}`)) as DeletionProgress
      setProgress(value)
    }
    if (value.status !== 'completed' || !value.result) throw new Error(value.message || t("删除操作未完成，请刷新任务列表后检查。"))
    return value.result
  }

  async function prepare(ids: string[]) {
    const fromStudy = review?.fromStudy ?? studyId
    if (!review) trigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setBusy(true); setError(''); setFeedback(''); setPending(false); setProgress(null)
    try {
      const value = await operation('preview', { job_ids: ids }) as Review
      if (typeof value.return_to_pool !== 'boolean' || !value.candidate_tokens?.return || !value.candidate_tokens.exclude) {
        throw new Error(t("后端尚未加载删除时的候选池选项，请重启后端服务并刷新页面后再删除。"))
      }
      if (fromStudy && (value.job_ids.length !== 1 || value.job_ids[0] !== ids[0] ||
          !value.research_drafts?.some(study => study.study_id === fromStudy))) {
        throw new Error(t("评估关联已变化，请刷新后重试。"))
      }
      const returnToPool = review?.return_to_pool ?? value.return_to_pool
      setReview({ ...value, requested: ids, fromStudy, return_to_pool: returnToPool,
        token: value.candidate_tokens[returnToPool ? 'return' : 'exclude'] })
    } catch (e) { setError(errorMessage(e)) } finally { setBusy(false) }
  }
  async function remove() {
    if (!review) return
    setBusy(true); setError(''); setProgress(null)
    try {
      const result = await operation('delete', { job_ids: review.requested, token: review.token,
        return_to_pool: review.return_to_pool }) as { deleted_job_ids: string[]; cleanup_pending: boolean }
      if (result.cleanup_pending) {
        setPending(true); setError("任务记录已删除，但部分文件仍被占用。关闭占用文件的程序后，点击“重试清理”。")
      } else {
        closeReview(Boolean(review.fromStudy)); setPending(false)
        setFeedback(review.fromStudy ? "已删除评估及其生成任务。" :
          ["已删除 {0} 个任务及其不再被使用的关联数据。", result.deleted_job_ids.length])
        onDeleted(result.deleted_job_ids)
      }
    } catch (e) { setError(errorMessage(e)) } finally { setBusy(false) }
  }

  const confirmation = (review || busy) && <dialog ref={dialog} className="delete-dialog" aria-labelledby="delete-task-title" aria-busy={busy} onCancel={e => { e.preventDefault(); if (!busy) closeReview() }}>
      {busy && <div className="deletion-progress" role="status" aria-live="polite">
        <p>{serviceText(progress?.message) || t("正在准备操作…")}{progress && progress.total > 0 && `（${progress.completed} / ${progress.total}）`}</p>
        {progress && progress.total > 0 ? <progress max={progress.total} value={progress.completed} aria-label={t("当前删除阶段进度")} /> : <progress aria-label={t("正在处理删除操作")} />}
        {!!progress?.files_checked && <p className="muted">{t("已检查 ")}{progress.files_checked.toLocaleString()}{t(" 个文件／目录；其他标注任务可继续保存结果。")}</p>}
        {!!progress?.files_removed && <p className="muted">{t("已清理 ")}{progress.files_removed.toLocaleString()}{t(" 个独占文件。")}</p>}
      </div>}
      {!review ? <><h2 id="delete-task-title">{t("正在核对删除范围…")}</h2><p role="status">{t("正在检查所选任务的关联记录和共享文件，完成后将显示确认信息。")}</p><p className="muted">{t("此时尚未删除任何数据。")}</p></> : <>
      <h2 id="delete-task-title">{review.fromStudy ? t("永久删除评估及其生成任务？") : t("永久删除 {0} 个任务？", review.summary.tasks)}</h2>
      <p>{review.fromStudy ? t("同时删除此评估专用的生成任务及其独占记录。") :
        t("仅删除勾选的任务，以及不再被使用的文本、执行记录、日志和图片／点云缓存。")}</p>
      {!!review.research_drafts?.length && <div className="notice">
        <p>{t("以下评估尚未进入盲审，将随任务删除并解除候选预留。")}</p>
        <ul>{review.research_drafts.map(study => <li key={study.study_id}>
          {t("「{0}」 · 预留 {1} 个目标", study.title, study.reserved_targets)}
        </li>)}</ul>
      </div>}
      <label><input type="checkbox" checked={review.return_to_pool} disabled={busy || pending} onChange={e => {
        const returnToPool = e.target.checked
        setReview({ ...review, return_to_pool: returnToPool, token: review.candidate_tokens[returnToPool ? 'return' : 'exclude'] })
      }} />{t("允许新任务再次标注这些已处理的物体")}</label>
      <p className="muted">{review.return_to_pool
        ? t("新任务可再次选择这些物体。但如果其他任务仍在处理它们、其他记录禁止再次选择，或剩余模型调用次数不够，仍不能选择。")
        : t("保留 {0} 个已处理物体的排除记录，后续新任务不会重新选择它们。", review.processed_tracks)}{t("未处理的候选物体不受此选项影响。")}{review.return_to_pool ? t("如果这些样本已导出，重新标注可能导致跨包版本冲突。") : ''}</p>
      {!!review.summary.retained_tasks && <p className="notice">{t("另有 ")}{review.summary.retained_tasks}{t(" 个关联任务会保留，其继承样本、必要过程记录和累计调用额度不受影响。")}</p>}
      <p className="muted">{t("关联数据：")}{review.summary.runs}{t(" 条执行记录 · ")}{review.summary.cohorts}{t(" 个盲评集合 · ")}{review.summary.releases}{t(" 个发布包 · ")}{review.summary.scenes}{t(" 帧点云。")}</p>
      {Object.values(review.retained_shared).some(n => n > 0) && <p className="notice">{t("仍被其他任务或结果引用的共用数据会保留。")}</p>}
      {review.shared_cleanup_deferred && <p className="notice">{t("本次按核对范围保留共享场景、证据、冻结方法和日志，其他标注任务可继续保存结果。")}</p>}
      <p>{t("删除不可恢复。原始数据与模型配置保留，已产生的 API 费用不会撤销；仍被其他任务使用的调用记录和统计会保留。已下载到项目外的文件需自行删除。")}</p>
      {error && <p className="error" role="alert">{serviceText(error)}</p>}
      {reviewActive && <p className="notice">{t("所选任务已开始执行，暂不能删除。其他任务不受影响。")}</p>}
      {busy && <p role="status">{t("正在处理关联记录与文件，完成后自动更新任务列表，请勿重复操作。")}</p>}
      <div className="controls"><button disabled={busy} onClick={() => closeReview()}>{pending ? t("稍后处理") : t("取消")}</button>{error && !pending && <button disabled={busy} onClick={() => void prepare(review.requested)}>{t("重新核对范围")}</button>}<button className="danger" disabled={busy || reviewActive} onClick={() => void remove()}>{busy ? t("正在删除…") : pending ? t("重试清理") : t("永久删除")}</button></div>
      </>}
    </dialog>
  return { busy, review, error, feedback, prepare, dialog: confirmation,
    clearMessages: () => { setError(''); setFeedback('') } }
}

export default function TaskManager({ jobs, loading = false, loadError = '', onReload, onBusyChange, current, activeJobId, labels, onOpen, onDeleted }: {
  jobs: Task[]; current?: string; activeJobId?: string; labels: Record<string, string>;
  loading?: boolean; loadError?: string; onReload?: () => void;
  onBusyChange?: (busy: boolean) => void;
  onOpen: (id: string) => void; onDeleted: (ids: string[]) => void;
}) {
  useLocale()

  const [checked, setChecked] = useState<string[]>([])
  const [managing, setManaging] = useState(false)
  const manageButton = useRef<HTMLButtonElement>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const selectAll = useRef<HTMLInputElement>(null)
  const returnFocusAfterDelete = useRef(false)
  const isActive = (job: Task) => job.executor_active || job.job_id === activeJobId
  const deletable = jobs.filter(job => !isActive(job))
  const selected = checked.filter(id => deletable.some(j => j.job_id === id))
  const currentJob = jobs.find(job => job.job_id === current) ?? jobs.find(job => job.job_id === activeJobId) ?? jobs[0]
  const statusLabel = (job: Task) => job.status === 'queued'
    ? t("排队第 {0} 位", job.queue_position ?? '—')
    : labels[job.status] ?? job.status
  const { busy, review, error, feedback, prepare, dialog, clearMessages } = useTaskDeletion({
    isActive: id => jobs.some(job => job.job_id === id && isActive(job)),
    onDeleted: ids => {
      returnFocusAfterDelete.current = true
      setChecked([]); setManaging(false)
      onDeleted(ids)
    },
  })
  useEffect(() => { onBusyChange?.(busy) }, [busy, onBusyChange])
  useEffect(() => {
    if (selectAll.current) selectAll.current.indeterminate = selected.length > 0 && selected.length < deletable.length
  }, [selected.length, deletable.length, managing])
  useEffect(() => {
    if (returnFocusAfterDelete.current && !review && !managing) {
      returnFocusAfterDelete.current = false
      const target = manageButton.current ?? heading.current
      target?.focus()
    }
  }, [review, managing, jobs.length])
  return <section className="panel task-manager" aria-labelledby="task-list-heading">
    <div className="task-manager-header">
      <h2 id="task-list-heading" ref={heading} tabIndex={-1}>{t("任务列表 ")}<span className="count-badge">{!jobs.length && (loading || loadError) ? '—' : jobs.length}</span></h2>
      {currentJob && !managing && <div className="task-picker">
        {jobs.length === 1 ? <>
          <span>{t("当前任务")}</span>
          <strong>{taskName(currentJob)}</strong>
          {currentJob.status === 'queued' && <span className="status-badge status-queued">{statusLabel(currentJob)}</span>}
        </> : <>
          <label htmlFor="annotation-task-picker">{t("当前任务")}</label>
          <select id="annotation-task-picker" value={currentJob.job_id} disabled={busy} onChange={e => onOpen(e.target.value)}>
            {jobs.map(job => {
              const { scope, createdAt } = taskDetails(job)
              return <option key={job.job_id} value={job.job_id}>{scope} · {statusLabel(job)} · {createdAt}</option>
            })}
          </select>
        </>}
      </div>}
      {jobs.length > 0 && <button ref={manageButton} className="task-manage-toggle" type="button" disabled={busy} aria-expanded={managing} onClick={() => { setManaging(!managing); setChecked([]); clearMessages() }}>{managing ? t("完成") : t("管理")}</button>}
    </div>
    {managing && jobs.length > 0 && <div className="task-actions"><div className="task-selection-summary"><label><input ref={selectAll} type="checkbox" aria-label={t("选择全部任务")} title={t("仅选择未在执行的任务")} checked={deletable.length > 0 && selected.length === deletable.length} disabled={busy || !deletable.length} onChange={e => setChecked(e.target.checked ? deletable.map(j => j.job_id) : [])} />{t("全选")}</label><span>{t("已选 ")}{selected.length}{t(" 项")}</span></div>
      <button className="danger" disabled={busy || !selected.length} onClick={() => void prepare(selected)}>{busy && !review ? t("正在核对删除范围…") : t("删除所选任务")}</button></div>}
    {jobs.some(isActive) && managing && <p className="muted">{t("执行中的任务不可删除，其他任务可正常删除。")}</p>}
    {error && !review && <p className="error" role="alert">{serviceText(error)}</p>}
    {feedback && <p role="status" className="operation-feedback">{serviceText(feedback)}</p>}
    {loadError && <p className="error" role="alert">{t("任务列表读取失败：")}{serviceText(loadError)}。{onReload && <button className="text-button" onClick={onReload}>{t("重新读取")}</button>}</p>}
    {!jobs.length && loading && <p className="task-list-empty" role="status">{t("正在读取已有任务…")}</p>}
    {!jobs.length && !loading && !loadError && <p className="task-list-empty">{t("暂无任务。")}</p>}
    {managing && jobs.length > 0 && <ul className="task-list" aria-label={t("标注任务列表")}>{jobs.map(job => {
      const { scope, createdAt } = taskDetails(job)
      return <li key={job.job_id}>
        <label className="task-management-row">
          <input type="checkbox" aria-label={t("选择任务 {0}", taskName(job))} checked={selected.includes(job.job_id)} disabled={busy || isActive(job)} onChange={e => setChecked(e.target.checked ? [...selected, job.job_id] : selected.filter(id => id !== job.job_id))} />
          <strong>{scope}</strong>
          <time dateTime={new Date(job.created_at * 1000).toISOString()}>{createdAt}</time>
          <span className={`status-badge status-${job.status}`}>{statusLabel(job)}</span>
        </label>
      </li>
    })}</ul>}
    {dialog}
  </section>
}
