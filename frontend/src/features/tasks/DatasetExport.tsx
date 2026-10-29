import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { taskDetails, taskName, exportTaskSelection } from '../../shared/taskName'
import { errorMessage, readJson } from '../../shared/annotationDisplay'

type Task = { spec?: { dataset_split?: 'train' | 'validation' }; status: string; executor_active: boolean; job_id: string; created_at: number; counts: Record<string, number>; selection_progress?: { requested: number | null } }
type ExportJob = { export_id: string; status: string; stage: string; message: string; completed: number; total: number; bytes_written: number; bytes_sent: number; created_at: number; updated_at: number }
const activeStates = ['running', 'ready', 'downloading', 'cancelling']
const exportApi = '/api/batches/export/jobs'
function size(bytes: number) { return `${(bytes / 1024 / 1024).toFixed(1)} MB` }

function unfinishedReason(job: Task): string | undefined {
  if (job.status === 'completed' && job.executor_active === false) return undefined
  if (job.executor_active || ['running', 'preparing'].includes(job.status)) return '任务仍在执行，请等待处理结束后导出。'
  const state: Record<string, string> = { ready: '待启动', paused: '已暂停', stopped: '已停止', blocked: '执行受阻' }
  return `任务尚未完成${state[job.status] ? `（${state[job.status]}）` : ''}，暂不可导出。`
}

function ControlHint({ id, reason, children, onExplain }: { id: string; reason?: string; children: ReactNode; onExplain: (reason: string) => void }) {
  return <span className={`dataset-export-control-hint${reason ? ' is-disabled' : ''}`} tabIndex={reason ? 0 : undefined}
    aria-describedby={reason ? id : undefined} onClick={() => { if (reason) onExplain(reason) }}
    onKeyDown={e => { if (reason && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onExplain(reason) } }}>
    {children}{reason && <span id={id} className="dataset-export-tooltip" role="tooltip">{reason}</span>}
  </span>
}

export default function DatasetExport({ jobs, current }: { jobs: Task[]; current?: string }) {
  const [scope, setScope] = useState<'selected' | 'all'>('selected')
  const [checked, setChecked] = useState<string[]>([])
  const [filename, setFilename] = useState('')
  const [feedback, setFeedback] = useState('')
  const [exportJob, setExportJob] = useState<ExportJob | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [now, setNow] = useState(Date.now())
  const submitLock = useRef(false)
  const downloadLink = useRef<HTMLAnchorElement>(null)
  const downloaded = useRef('')
  const disclosure = useRef<HTMLDetailsElement>(null)
  const active = !!exportJob && activeStates.includes(exportJob.status)
  const busy = loading || submitting || active
  const exportId = exportJob?.export_id
  useEffect(() => { if (active && disclosure.current) disclosure.current.open = true }, [active])

  useEffect(() => {
    let disposed = false
    readJsonResponse(`${exportApi}/current`).then(value => { if (!disposed) setExportJob(value) })
      .catch(error => { if (!disposed) setFeedback(errorMessage(error)) })
      .finally(() => { if (!disposed) setLoading(false) })
    return () => { disposed = true }
  }, [])

  useEffect(() => {
    if (!exportId || !active) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const response = await fetch(`${exportApi}/${exportId}`, { cache: 'no-store', signal: AbortSignal.timeout(10000) })
        if (response.status === 404) {
          if (!disposed) {
            setExportJob(null)
            setFeedback('导出任务已失效（服务可能已重启），请重新导出。')
          }
          return
        }
        const value = await readJson(response)
        if (!disposed) { setExportJob(value); setFeedback(''); setNow(Date.now()) }
      } catch (error) {
        if (!disposed) { setFeedback(`暂时无法读取导出进度：${errorMessage(error)} 正在重试。`); setNow(Date.now()) }
      }
      if (!disposed) timer = setTimeout(poll, 1000)
    }
    void poll()
    return () => { disposed = true; clearTimeout(timer) }
  }, [exportId, active])

  useEffect(() => {
    if (exportJob?.status === 'ready' && downloaded.current !== exportJob.export_id) {
      downloaded.current = exportJob.export_id
      downloadLink.current?.click()
    }
  }, [exportJob])

  async function readJsonResponse(url: string) { return readJson(await fetch(url, { cache: 'no-store', signal: AbortSignal.timeout(10000) })) }

  async function startExport(split: string) {
    if (submitLock.current || busy) return
    submitLock.current = true
    setSubmitting(true); setFeedback('')
    try {
      const value = await readJson(await fetch(exportApi, { method: 'POST', signal: AbortSignal.timeout(10000), headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ split, export_scope: scope, job_ids: scope === 'selected' ? selected : null, filename }) }))
      setExportJob(value); setNow(Date.now())
    } catch (error) {
      setFeedback(errorMessage(error))
      // A lost creation response may still have started an export. Recover it
      // before allowing another request to consume resources.
      try { setExportJob(await readJsonResponse(`${exportApi}/current`)) } catch { /* Keep the actionable request error. */ }
    } finally { submitLock.current = false; setSubmitting(false) }
  }

  async function cancelExport() {
    if (!exportJob || cancelling || !window.confirm('确定终止本次导出？将停止打包或传输，并清理本次服务器临时 ZIP。原始数据和标注记录会保留；浏览器本地下载文件请在下载列表中处理。')) return
    setCancelling(true); setFeedback('')
    try {
      setExportJob(await readJson(await fetch(`${exportApi}/${exportJob.export_id}/cancel`, { method: 'POST', signal: AbortSignal.timeout(10000) })))
    } catch (error) { setFeedback(`终止请求未确认：${errorMessage(error)} 请重试。`) }
    finally { setCancelling(false) }
  }
  const eligible = jobs.filter(job => !unfinishedReason(job))
  const selected = checked.filter(id => eligible.some(job => job.job_id === id))
  const selection = exportTaskSelection(scope === 'all' ? eligible : eligible.filter(job => selected.includes(job.job_id)))
  const currentJob = jobs.find(job => job.job_id === current)
  const currentReason = currentJob ? unfinishedReason(currentJob) : undefined
  const validName = /^[\p{L}\p{N}_ -]*$/u.test(filename)
  const ready = !busy && !selection.error && validName && (scope === 'all' ? eligible.length > 0 : selected.length > 0 && selected.length <= 1000)
  const taskCount = scope === 'all' ? eligible.length : selected.length
  const unavailable = !jobs.length ? '暂无任务，完成标注后即可在这里导出。'
    : !eligible.length ? '暂无已完成任务，请等待任务处理结束且执行器退出后导出。'
    : !taskCount ? '请选择要打包的任务。' : taskCount > 1000 && scope === 'selected' ? '每包最多选择 1000 个任务。' : ''
  const blockedReason = loading ? '正在读取导出状态…' : submitting ? '正在提交导出请求…' : active ? '本次导出正在处理，请查看下方进度。' : selection.error || unavailable || (!validName ? '压缩包名称无效，请先修改名称。' : '')
  function chooseTasks(ids: string[]) { setChecked(ids); setFeedback('') }
  return <details ref={disclosure} className="panel dataset-export export-disclosure" aria-labelledby="dataset-export-heading">
    <summary className="dataset-export-heading"><h2 id="dataset-export-heading">导出数据集</h2><span className="dataset-export-format">Waymo-3DRef · ZIP</span></summary>
    <form onChange={() => setFeedback('')} onSubmit={e => {
      e.preventDefault()
      if (!ready) { setFeedback(blockedReason); return }
      void startExport(selection.split)
    }}>
      <div className="dataset-export-grid">
        <fieldset className="dataset-export-step dataset-export-selection" disabled={busy}>
          <legend><span className="dataset-export-step-number" aria-hidden="true">1</span>选择任务</legend>
          <div className="dataset-export-scope" role="group" aria-label="导出范围">
            <label className={`${scope === 'selected' ? 'is-selected' : ''}${!eligible.length ? ' is-disabled' : ''}`} title={!eligible.length ? '暂无已完成任务，等待任务处理结束后可选择导出范围。' : '点击后勾选需要打包的任务，可多选。'}><input type="radio" name="export_scope" disabled={!eligible.length} checked={scope === 'selected'} onChange={() => setScope('selected')} value="selected" />按任务选择</label>
            <label className={`${scope === 'all' ? 'is-selected' : ''}${!eligible.length ? ' is-disabled' : ''}`} title={!eligible.length ? '暂无已完成任务，等待任务处理结束后可选择导出范围。' : '点击后汇总全部已完成任务中的成功标注，自动跳过未完成任务。'}><input type="radio" name="export_scope" disabled={!eligible.length} checked={scope === 'all'} onChange={() => setScope('all')} value="all" />全部已完成任务 <span>{eligible.length}</span></label>
          </div>
          {!jobs.length ? <div className="dataset-export-empty"><strong>暂无任务可导出</strong><p>先创建并完成标注任务，再选择需要打包的数据。</p></div>
            : scope === 'selected' ? <>
              <div className="dataset-export-toolbar"><span aria-live="polite">已选 <strong>{selected.length}</strong> / {eligible.length} 个可导出任务</span>
                <div><ControlHint id="dataset-select-all-reason" reason={!eligible.length ? '暂无已完成任务可选。' : selected.length === eligible.length ? '已选中全部已完成任务。' : undefined} onExplain={setFeedback}>
                  <button type="button" title="选中全部已完成任务" disabled={selected.length === eligible.length} onClick={() => chooseTasks(eligible.map(job => job.job_id))}>全选</button></ControlHint>
                  <ControlHint id="dataset-clear-reason" reason={!selected.length ? '还没有选中的任务，无需清空。' : undefined} onExplain={setFeedback}>
                    <button type="button" title="取消所有已选任务" disabled={!selected.length} onClick={() => chooseTasks([])}>清空</button></ControlHint>
                  {current && currentJob && <ControlHint id="dataset-current-reason" reason={currentReason || (selected.length === 1 && selected[0] === current ? '当前任务已单独选中。' : undefined)} onExplain={setFeedback}>
                    <button type="button" title="仅选中当前正在查看的任务" disabled={!!currentReason || (selected.length === 1 && selected[0] === current)} onClick={() => chooseTasks([current])}>仅当前任务</button></ControlHint>}</div>
              </div>
              <p className="dataset-export-hint">仅已完成且执行器已退出的任务可选；未完成任务置灰。</p>
              <div className="dataset-export-task-list">
                {jobs.map(job => {
                  const reason = unfinishedReason(job)
                  return <label key={job.job_id} className={reason ? 'is-disabled' : selected.includes(job.job_id) ? 'is-selected' : ''}
                    title={reason || (selected.includes(job.job_id) ? '已选中，点击卡片任意位置取消。' : '点击卡片任意位置，加入本次导出。')}>
                    <input type="checkbox" disabled={!!reason} checked={selected.includes(job.job_id)} aria-label={`导出任务 ${taskName(job)}`}
                      onChange={e => { if (!reason) setChecked(previous => e.target.checked ? [...previous, job.job_id] : previous.filter(id => id !== job.job_id)) }} />
                    <span><strong>{job.spec?.dataset_split === 'validation' ? '' : '训练集 · '}{taskDetails(job).scope}</strong><time dateTime={new Date(job.created_at * 1000).toISOString()}>{taskDetails(job).createdAt}</time>
                      {reason && <small className="dataset-export-task-reason">{reason}</small>}</span>
                    <small>{reason ? '暂不可导出' : job.job_id === current ? '当前任务' : '已完成'}</small>
                  </label>
                })}
              </div>
              {selected.map(id => <input key={id} type="hidden" name="job_ids" value={id} />)}
              {selected.length > 1000 && <p className="error" role="alert">每包最多选择 1000 个任务，请减少数量。</p>}
            </> : <div className="dataset-export-all"><strong>{eligible.length} 个已完成任务</strong><p>只汇总已完成且执行器已退出任务中的成功标注，自动跳过未完成任务。需要分包上传时，请切换为“按任务选择”。</p></div>}
        </fieldset>
        <div className="dataset-export-settings">
          <fieldset className="dataset-export-step" disabled={busy}>
            <legend><span className="dataset-export-step-number" aria-hidden="true">2</span>命名压缩包 <small>可选</small></legend>
            <label className={`dataset-export-filename${!validName ? ' is-invalid' : ''}${!eligible.length ? ' is-disabled' : ''}`} title={!eligible.length ? '暂无已完成任务，暂不能填写压缩包名称。' : '可输入自定义名称，例如 trainset1，无需输入 .zip。'}>
              <span className="sr-only">压缩包名称</span>
              <input name="filename" value={filename} maxLength={80} disabled={!eligible.length}
                placeholder="例如 trainset1" onChange={e => setFilename(e.target.value)} aria-invalid={!validName} aria-describedby="dataset-filename-help" />
              <span aria-hidden="true">.zip</span>
            </label>
            <p id="dataset-filename-help" className="dataset-export-hint">{!eligible.length ? '暂无已完成任务，压缩包名称暂不可编辑。' : '留空使用默认名称，自动添加 .zip 后缀。'}</p>
            {!validName && <p className="error" role="alert">名称仅支持中文、字母、数字、空格、下划线和短横线，无需输入 .zip。</p>}
          </fieldset>
          <fieldset className="dataset-export-step dataset-export-download">
            <legend><span className="dataset-export-step-number" aria-hidden="true">3</span>下载数据集</legend>
            {selection.error && <p className="error" role="alert">{selection.error}</p>}
            <p id="dataset-download-status" className={`dataset-export-summary ${ready ? 'is-ready' : 'is-unavailable'}`} role="status"><span aria-hidden="true">{ready ? '✓' : '○'}</span>{blockedReason || `已就绪，可导出 ${taskCount} 个任务中的成功标注。`}</p>
            <div className="dataset-export-actions">
              <ControlHint id="dataset-train-reason" reason={!ready ? blockedReason : undefined} onExplain={setFeedback}>
                <button className="primary" type="submit" aria-describedby="dataset-download-status" disabled={!ready}>导出{selection.split === 'validation' ? '验证集' : '训练集'} ZIP ↓</button></ControlHint>
            </div>
            {exportJob && <div className={`dataset-export-progress${exportJob.status === 'failed' ? ' is-failed' : ''}`} role="group" aria-label="导出进度">
              <div className="dataset-export-progress-heading"><strong>{({ selecting: '检查导出范围', packaging: '打包数据', finalizing: '完成压缩包', ready: '打包完成', downloading: '传输至浏览器', finished: '导出结果' } as Record<string, string>)[exportJob.stage] || '正在导出'}</strong>
                {active && <span>已用时 {Math.max(0, Math.floor((now / 1000 - exportJob.created_at) / 60))} 分 {Math.max(0, Math.floor(now / 1000 - exportJob.created_at) % 60)} 秒</span>}</div>
              {active && <progress aria-label="当前导出阶段进度" max={exportJob.total || 1} value={exportJob.status === 'ready' ? 1 : exportJob.total ? exportJob.completed : undefined} />}
              <p role="status" aria-live="polite">{exportJob.message}</p>
              <div className="dataset-export-progress-details">
                {active && exportJob.total > 0 && <span>{exportJob.stage === 'downloading' ? `${size(exportJob.bytes_sent)} / ${size(exportJob.bytes_written)}` : `${exportJob.completed} / ${exportJob.total} ${exportJob.stage === 'selecting' ? '个任务已检查' : '条样本已打包'}`}</span>}
                {exportJob.bytes_written > 0 && exportJob.stage !== 'downloading' && <span>ZIP 大小 {size(exportJob.bytes_written)}</span>}
              </div>
              {exportJob.status === 'ready' && <a ref={downloadLink} href={`${exportApi}/${exportJob.export_id}/dataset.zip`} target="dataset-export-download">若未自动下载，点击下载 ZIP ↓</a>}
              {active && <button type="button" className="dataset-export-cancel" disabled={cancelling || exportJob.status === 'cancelling'} onClick={() => void cancelExport()}>{cancelling || exportJob.status === 'cancelling' ? '正在终止并清理…' : '终止导出'}</button>}
              {active && <p className="dataset-export-hint">进度按当前阶段实际完成量更新。刷新页面可恢复查看；打包完成后自动交给浏览器下载。</p>}
            </div>}
            {feedback && <p className="dataset-export-feedback" role="status">{feedback}</p>}
            <p className="dataset-export-hint">导出类型及默认文件名跟随任务的数据集，训练集和验证集须分开导出；格式与质量规则相同。</p>
          </fieldset>
        </div>
      </div>
    </form>
    <iframe name="dataset-export-download" title="数据集下载" hidden />
    <details className="dataset-export-help">
      <summary title="点击展开或收起数据内容与合并命令">包含哪些数据，如何合并分包？<span className="dataset-help-open" aria-hidden="true">点击展开</span><span className="dataset-help-close" aria-hidden="true">点击收起</span></summary>
      <ul><li>包含自动核验通过或当前版本已人工审批的描述、完整点云、来源原图、标定、三维标注框及加载脚本；拒绝、废弃或无文本的记录不导出。</li>
        <li>压缩包使用统一精简格式，只保留训练数据和必要身份信息。从任一分包取出 merge.py 和 loader.py 放在同一目录；合并工具可直接读取 ZIP，无需逐包解压。</li>
        <li>合并训练分包：<code>python merge.py --output trainset trainset1.zip trainset2.zip</code>；合并验证分包：<code>python merge.py --output valset valset1.zip valset2.zip</code>。以后追加新包仍使用同一命令，已有数据会保留。</li>
        <li>同一样本在多个包中出现时，只保留一份；共用的图像和点云也会合并保存。如果同一样本的内容不一致，或训练集与验证集包含同一段驾驶记录，合并会停止并提示原因。</li>
        <li>保留原始数据的官方训练 / 验证集标识，不自动划分数据；自动校验通过不代表独立人工审核通过。</li></ul>
    </details>
  </details>
}
