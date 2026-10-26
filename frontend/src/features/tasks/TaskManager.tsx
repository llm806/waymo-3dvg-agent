import { useEffect, useRef, useState } from 'react'
import { errorMessage, readJson } from '../../shared/annotationDisplay'
import { taskName, taskDetails } from '../../shared/taskName'

type Task = { job_id: string; status: string; created_at: number; counts: Record<string, number>; executor_active: boolean; queue_position?: number | null }
type Review = { job_ids: string[]; requested: string[]; token: string;
  return_to_pool: boolean; processed_tracks: number; candidate_tokens: { return: string; exclude: string };
  shared_cleanup_deferred?: boolean;
  summary: { tasks: number; retained_tasks?: number; runs: number; model_calls: number; cohorts: number; releases: number; evidence: number; scenes: number };
  retained_shared: { runs: number; evidence: number; scenes: number } }
type DeletionProgress = { operation_id: string; status: 'running' | 'completed' | 'failed';
  stage: string; message: string; completed: number; total: number; files_checked: number; files_removed?: number;
  result: Review | { deleted_job_ids: string[]; cleanup_pending: boolean } | null }

export default function TaskManager({ jobs, loading = false, loadError = '', onReload, onBusyChange, current, activeJobId, labels, onOpen, onDeleted }: {
  jobs: Task[]; current?: string; activeJobId?: string; labels: Record<string, string>;
  loading?: boolean; loadError?: string; onReload?: () => void;
  onBusyChange?: (busy: boolean) => void;
  onOpen: (id: string) => void; onDeleted: (ids: string[]) => void;
}) {
  const [checked, setChecked] = useState<string[]>([])
  const [review, setReview] = useState<Review | null>(null)
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [feedback, setFeedback] = useState('')
  const [pending, setPending] = useState(false)
  const [progress, setProgress] = useState<DeletionProgress | null>(null)
  const [managing, setManaging] = useState(false)
  const dialog = useRef<HTMLDialogElement>(null)
  const trigger = useRef<HTMLElement | null>(null)
  const manageButton = useRef<HTMLButtonElement>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const selectAll = useRef<HTMLInputElement>(null)
  const returnFocusAfterDelete = useRef(false)
  const isActive = (job: Task) => job.executor_active || job.job_id === activeJobId
  const deletable = jobs.filter(job => !isActive(job))
  const selected = checked.filter(id => deletable.some(j => j.job_id === id))
  const reviewActive = !!review?.job_ids.some(id => jobs.some(job => job.job_id === id && isActive(job)))
  useEffect(() => { if ((review || busy) && !dialog.current?.open) dialog.current?.showModal() }, [review, busy])
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
  function closeReview(restoreFocus = true) {
    dialog.current?.close()
    setReview(null); setError(''); setProgress(null)
    if (restoreFocus && trigger.current?.isConnected) trigger.current.focus()
  }

  async function operation(action: 'preview' | 'delete', values: Record<string, unknown>) {
    const response = await fetch('/api/batches/deletion/operations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...values, action }),
    })
    if (response.status === 404) throw new Error('后端尚未加载删除进度功能，请重启后端并刷新页面后再操作。')
    let value = await readJson(response) as DeletionProgress
    setProgress(value)
    while (value.status === 'running') {
      await new Promise(resolve => setTimeout(resolve, 350))
      value = await readJson(await fetch(`/api/batches/deletion/operations/${value.operation_id}`)) as DeletionProgress
      setProgress(value)
    }
    if (value.status !== 'completed' || !value.result) throw new Error(value.message || '删除操作未完成，请刷新任务列表后检查。')
    return value.result
  }

  async function prepare(ids: string[]) {
    if (!review) trigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setBusy(true); setError(''); setFeedback(''); setPending(false); setProgress(null)
    try {
      const value = await operation('preview', { job_ids: ids }) as Review
      if (typeof value.return_to_pool !== 'boolean' || !value.candidate_tokens?.return || !value.candidate_tokens.exclude) {
        throw new Error('后端尚未加载删除时的候选池选项，请重启后端服务并刷新页面后再删除。')
      }
      const returnToPool = review?.return_to_pool ?? value.return_to_pool
      setReview({ ...value, requested: ids, return_to_pool: returnToPool,
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
        setPending(true); setError('任务记录已删除，但部分文件仍被占用。关闭占用文件的程序后，点击“重试清理”。')
      } else {
        returnFocusAfterDelete.current = true
        setChecked([]); setManaging(false); closeReview(false); setPending(false)
        setFeedback(`已删除 ${result.deleted_job_ids.length} 个任务及其不再被使用的关联数据。`)
        onDeleted(result.deleted_job_ids)
      }
    } catch (e) { setError(errorMessage(e)) } finally { setBusy(false) }
  }

  return <div className="task-manager">
    <div className="section-heading"><h2 ref={heading} tabIndex={-1}>任务列表 <span className="count-badge">{!jobs.length && (loading || loadError) ? '—' : jobs.length}</span></h2>
      {jobs.length > 0 && <button ref={manageButton} className="text-button" disabled={busy} aria-pressed={managing} onClick={() => { setManaging(!managing); setChecked([]); setError(''); setFeedback('') }}>{managing ? '完成' : '管理'}</button>}
    </div>
    {managing && jobs.length > 0 && <div className="task-actions"><div className="task-selection-summary"><label><input ref={selectAll} type="checkbox" aria-label="选择全部任务" title="仅选择未在执行的任务" checked={deletable.length > 0 && selected.length === deletable.length} disabled={busy || !deletable.length} onChange={e => setChecked(e.target.checked ? deletable.map(j => j.job_id) : [])} />全选</label><span>已选 {selected.length} 项</span></div>
      <button className="danger" disabled={busy || !selected.length} onClick={() => void prepare(selected)}>{busy && !review ? '正在核对删除范围…' : '删除所选任务'}</button></div>}
    {jobs.some(isActive) && managing && <p className="muted">执行中的任务不可删除，其他任务可正常删除。</p>}
    {error && !review && <p className="error" role="alert">{error}</p>}
    {feedback && <p role="status" className="operation-feedback">{feedback}</p>}
    {loadError && <p className="error" role="alert">任务列表读取失败：{loadError}。{onReload && <button className="text-button" onClick={onReload}>重新读取</button>}</p>}
    {!jobs.length && loading && <p className="task-list-empty" role="status">正在读取已有任务…</p>}
    {!jobs.length && !loading && !loadError && <p className="task-list-empty">暂无任务。<br />在上方选择数量并创建任务。</p>}
    {jobs.length > 0 && <ul className="task-list" aria-label="标注任务列表">{jobs.map(job => <li key={job.job_id} className={`${current === job.job_id ? 'selected-task' : ''}${managing ? ' managing-task' : ''}`}>
      {managing && <input type="checkbox" aria-label={`选择任务 ${taskName(job)}`} checked={selected.includes(job.job_id)} disabled={busy || isActive(job)} onChange={e => setChecked(e.target.checked ? [...selected, job.job_id] : selected.filter(id => id !== job.job_id))} />}
      <button className="task-open" disabled={busy} title={taskName(job)} aria-current={current === job.job_id ? 'true' : undefined} aria-label={`查看任务 ${taskName(job)}`} onClick={() => onOpen(job.job_id)}>
        <strong className="task-card-title">{taskDetails(job).scope}</strong>
        <time dateTime={new Date(job.created_at * 1000).toISOString()}>{taskDetails(job).createdAt}</time>
        <span className="task-card-footer"><span className={`status-badge status-${job.status}`}>{job.status==='queued' ? `排队第 ${job.queue_position ?? '—'} 位` : labels[job.status] ?? job.status}</span>{current === job.job_id && <small>当前任务</small>}</span>
      </button>
    </li>)}</ul>}
    {(review || busy) && <dialog ref={dialog} className="delete-dialog" aria-labelledby="delete-task-title" aria-busy={busy} onCancel={e => { e.preventDefault(); if (!busy) closeReview() }}>
      {busy && <div className="deletion-progress" role="status" aria-live="polite">
        <p>{progress?.message || '正在准备操作…'}{progress && progress.total > 0 && `（${progress.completed} / ${progress.total}）`}</p>
        {progress && progress.total > 0 ? <progress max={progress.total} value={progress.completed} aria-label="当前删除阶段进度" /> : <progress aria-label="正在处理删除操作" />}
        {!!progress?.files_checked && <p className="muted">已检查 {progress.files_checked.toLocaleString()} 个文件／目录；其他标注任务可继续保存结果。</p>}
        {!!progress?.files_removed && <p className="muted">已清理 {progress.files_removed.toLocaleString()} 个独占文件。</p>}
      </div>}
      {!review ? <><h2 id="delete-task-title">正在核对删除范围…</h2><p role="status">正在检查所选任务的关联记录和共享文件，完成后将显示确认信息。</p><p className="muted">此时尚未删除任何数据。</p></> : <>
      <h2 id="delete-task-title">永久删除 {review.summary.tasks} 个任务？</h2>
      <p>仅删除勾选的任务，以及不再被使用的文本、执行记录、日志和图片／点云缓存。</p>
      <label><input type="checkbox" checked={review.return_to_pool} disabled={busy || pending} onChange={e => {
        const returnToPool = e.target.checked
        setReview({ ...review, return_to_pool: returnToPool, token: review.candidate_tokens[returnToPool ? 'return' : 'exclude'] })
      }} />将已处理物体重新加入候选池</label>
      <p className="muted">{review.return_to_pool
        ? '允许重新选择这些物体；仍被其他任务或记录排除、或累计额度不足的物体不会重新开放。'
        : `保留 ${review.processed_tracks} 个已处理物体的排除记录，后续新任务不会重新选择它们。`}
        未处理的候选物体不受此选项影响。{review.return_to_pool ? '如果这些样本已导出，重新标注可能导致跨包版本冲突。' : ''}</p>
      {!!review.summary.retained_tasks && <p className="notice">另有 {review.summary.retained_tasks} 个关联任务会保留，其继承样本、必要过程记录和累计调用额度不受影响。</p>}
      <p className="muted">关联数据：{review.summary.runs} 条执行记录 · {review.summary.cohorts} 个盲评集合 · {review.summary.releases} 个发布包 · {review.summary.scenes} 帧点云。</p>
      {Object.values(review.retained_shared).some(n => n > 0) && <p className="notice">仍被其他任务或结果引用的共用数据会保留。</p>}
      {review.shared_cleanup_deferred && <p className="notice">本次按核对范围保留共享场景、证据、冻结方法和日志，其他标注任务可继续保存结果。</p>}
      <p>删除不可恢复。原始数据与模型配置保留，已产生的 API 费用不会撤销；仍被其他任务使用的调用记录和统计会保留。已下载到项目外的文件需自行删除。</p>
      {error && <p className="error" role="alert">{error}</p>}
      {reviewActive && <p className="notice">所选任务已开始执行，暂不能删除。其他任务不受影响。</p>}
      {busy && <p role="status">正在处理关联记录与文件，完成后自动更新任务列表，请勿重复操作。</p>}
      <div className="controls"><button disabled={busy} onClick={() => closeReview()}>{pending ? '稍后处理' : '取消'}</button>{error && !pending && <button disabled={busy} onClick={() => void prepare(review.requested)}>重新核对范围</button>}<button className="danger" disabled={busy || reviewActive} onClick={() => void remove()}>{busy ? '正在删除…' : pending ? '重试清理' : '永久删除'}</button></div>
      </>}
    </dialog>}
  </div>
}
