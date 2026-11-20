import { serviceText, localizeLabels, t, useLocale } from '../../shared/locale.ts'
import type { InterfaceMessage } from '../../shared/locale.ts'
import { useEffect, useMemo, useState } from 'react'
import { categoryLabel, errorMessage, recordStatus, phases, providerDiagnosisLabel, readJson, readPollingJson, reasonLabel, isAnnotationResult, isProcessingRecord } from '../../shared/annotationDisplay'
import AlignmentInspection from '../scene/AlignmentInspection'
import TaskManager from './TaskManager'
import TaskMetrics from './TaskMetrics'
import DatasetExport from './DatasetExport'
import { formatTiming, hasIncompleteTiming } from '../../shared/duration'
import type { ItemTiming, TaskMetricsData, TaskTiming } from './TaskMetrics'
import { parseAnnotationCount } from '../../shared/taskName'
import type { TaskContinuation } from '../../shared/taskName'
import { cameraViewLabel } from '../../shared/cameraViews'
import type { ManualReview } from '../results/SampleReview'
import { taskLocation } from '../../navigation'
import { HelpHint, TermHelp } from '../../components/Help'

type Target = { official_split?: 'training' | 'validation'; target_id: string; ordinal: number; context: string; object_id: string; category: string; status: string; timestamp: number | null; frame_exclusion_counts: Record<string, number>; recommendation_rank?: number }
type Catalog = { catalog_id: string | null; summary: { tracks?: number; selected?: number; excluded?: number; contexts?: number; pending_pairing?: number }; targets: Target[]; total: number }
type Item = { manual_approved?:boolean; manual_review?: ManualReview | null; display_reason?: string; timing?: ItemTiming; ordinal: number; target: Target; display_target?: Target; status: string; run_status?: string; reason: string | null; packet_id: string | null; run_id: string | null; expression?: string | null; phase?: string; model_calls_used?: number; provider_diagnosis?: string; failure_diagnostic?: { http_status?: number; provider_error_code?: string }; retryable_parameter_failure?: boolean; retryable_preparation_failure?: boolean; retryable_connection_failure?: boolean; retryable_visibility_failure?: boolean; admission_status?: string; source_view?: string | null; source_view_status?: string; result_label?: string | null; legacy_rules?: boolean; visibility_observation?: { visibility: string | null; visibility_reason: string | null } | null }
type Job = { queue_position?: number | null; queue_paused?: boolean; continuation?: TaskContinuation | null; job_id: string; status: string; created_at: number; reason: string | null; executor_active: boolean; control: string | null; spec: { dataset_split?: 'train' | 'validation'; prepare_only: boolean; max_model_calls: number | null; catalog_id: string; desired_count?: number | null }; counts: Record<string, number>; selection_progress?: { requested: number | null } }
type JobDetail = Job & { resume?: { mode: 'resume' | 'recover' | 'unavailable'; message: string }; model_calls_reserved: number; selection_progress?: { requested: number | null; diagnostic_passed: number; attempted: number; candidate_pool: number }; timing?: TaskTiming; measured_at?: number; items: Item[]; events: { seq: number; kind: string; at: number; payload: Record<string, unknown> }[] }
type Progress = Pick<JobDetail, 'job_id' | 'status' | 'control' | 'reason' | 'executor_active' | 'counts' | 'model_calls_reserved' | 'selection_progress' | 'timing' | 'measured_at'> & { active_item_timings: Record<string, ItemTiming> }
const labels: Record<string, string> = localizeLabels({ discarded: '已删除', rejected: '人工不通过', ready: '待启动', pending: '待处理', selected: '有可用帧', excluded: '不符合标注条件', preparing: '准备数据', prepared: '数据已准备', running: '执行中', paused: '已暂停', stopped: '已停止', completed: '处理结束', blocked: '执行受阻', calibration_hold: '自动核验通过', abstained: '未采纳', needs_evidence: '未通过核验', failed: '执行失败', budget_skipped: '预算不足', reused: '复用结果' })
labels.manual_approved = '人工审批通过'
labels.not_needed = '未尝试 · 保留候选'
labels.queued = '排队中'
const completionReasons: Record<string, string> = localizeLabels({
  description_goal_reached: '已达到期望通过数量，剩余候选未调用模型。描述仍待独立评估。',
  selection_budget_exhausted: '剩余额度不足以完成下一目标的完整检查，已结束；未放宽质量要求。',
  candidate_pool_exhausted: '本任务内可自动处理的候选已用尽，已结束；失败记录与原因保留。返工或重试需按原任务规则处理。',
})
const reason = (value: string, diagnosis?: string, providerErrorCode?: string) => {
  if (value === 'description_goal_reached') return t("已达到期望通过数量，该候选未执行。")
  if (value === 'selection_budget_exhausted') return t("剩余调用额度不足以完成下一目标的完整检查，该候选未执行。")
  if (value === 'candidate_pool_exhausted') return t("本次候选池已用尽，未降低质量要求补足数量。")
  if (value === 'target_already_attempted') return t("此物体已有尝试记录，自动补选已改选其它目标。")
  if (value === 'repeated_request_rejections' || value === 'repeated_response_shape_failures') return reasonLabel(value)
  const providerMessage = providerDiagnosisLabel(diagnosis, providerErrorCode)
  if (providerMessage) return providerMessage
  if (value === 'target_spatially_ambiguous') return t("预检查发现目标与其它物体缺少可靠的空间区分线索，已提前跳过，未转换点云或调用模型。")
  if (value === 'preparation:official_conversion_failed:wsl_init_scope_stopping') return t("WSL 的 init.scope 正在停止，继续转换会被系统强制终止。请先保存 WSL / Docker 中的工作，再手动执行 wsl --shutdown；恢复环境后点击继续标注。当前物体未调用模型。")
  if (value === 'preparation:official_conversion_failed:wsl_environment_check_failed') return t("无法确认 WSL 转换环境状态，已停止数据准备。请检查本地准备日志并恢复 WSL 环境。当前物体未调用模型。")
  if (value === 'preparation:official_conversion_failed:4294967295') return t("WSL 未能启动转换程序，请检查本地准备日志并恢复 WSL 环境，再继续处理任务。当前物体未调用模型。")
  if (value.startsWith('preparation:official_conversion_failed')) {
    const code = value.slice('preparation:official_conversion_failed'.length).replace(/^:/, '')
    return code ? t("官方点云转换进程异常退出（{0}），当前物体未调用模型。详细原因已保存在本地准备日志中。", code === '124' ? t("转换超时") : code)
      : t("官方点云转换进程异常退出。请检查本地准备日志，恢复环境后继续标注。")
  }
  if (value === 'preparation:official_conversion_timeout') return t("点云转换超时，已停止本次准备。请检查本地准备日志。")
  if (value.startsWith('preparation:catalog_source_changed:')) return t("原始数据与物体目录不一致，请重新扫描物体目录。")
  return ({ no_target_lidar_points: t("目标无雷达点"), converted_target_points_missing: t("所选帧转换后的完整点云中，原生目标框内没有点，尚未进入文本生成。图像清晰不代表这一帧的点云证据完整。"), development_attribute_review_failed: t("开发复核发现描述含未经图像支持的属性，已保留为未通过候选，不纳入训练样本"), no_valid_projected_view: t("缺少有效视角"), target_evidence_not_observable: t("所选帧未满足图像身份或点云证据要求，未进入文本生成"), preparation_failed: t("数据准备失败"), child_run_blocked: t("目标执行受阻"), insufficient_complete_validation_budget: t("预算不足以完成检查") } as Record<string,string>)[value] ?? reasonLabel(value)
}
const preparationStage = (stage: unknown) => {
  if (typeof stage !== 'string') return ''
  if (stage === 'checking_converter_environment') return t("检查本地点云转换环境")
  const lidar = /^converting_lidar:([1-5]):([12])$/.exec(stage)
  if (lidar) return t("转换点云：雷达 {0} / 5，第 {1} 次回波", lidar[1], lidar[2])
  return ({ checking_sources: t("校验原始数据"), checking_distinguishability: t("检查图像与点云候选"), finding_alternate_frame: t("查看同一物体的其他帧"), starting_converter: t("首次启动点云转换环境"), reusing_converter: t("复用已加载的转换环境"), retrying_converter: t("转换进程中断，重新启动后重试一次"), loading_converter: t("加载数据读取组件"), loading_sdk: t("首次加载官方 Waymo SDK"), reusing_sdk: t("复用官方 Waymo SDK"), checking_scene_sources: t("校验当前帧来源"), reading_lidar: t("读取当前帧雷达数据"), reading_lidar_pose: t("读取当前帧雷达运动补偿数据"), reading_lidar_projections: t("读取当前帧雷达相机投影"), saving_points: t("保存完整点云"), reading_images: t("读取同帧图片"), saving_scene: t("保存当前帧"), scene_ready: t("点云与图片转换完成"), validating_scene: t("校验点云与图片"), building_evidence: t("准备目标框与标注依据") } as Record<string,string>)[stage] ?? t("准备当前物体数据")
}
const maxTargets = 2000
const settledStatuses = new Set(['prepared', 'calibration_hold', 'abstained', 'needs_evidence', 'failed', 'excluded', 'budget_skipped', 'reused'])

function LiveTiming({ item, task, observedAt, active }: { item?: ItemTiming; task?: TaskTiming; observedAt: number; active: boolean }) {
  useLocale()

  const [clock, setClock] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active || !task?.running) return
    const timer = setInterval(() => setClock(Date.now() / 1000), 1000)
    return () => clearInterval(timer)
  }, [active, task?.running])
  const extra = active ? Math.max(0, clock - observedAt) : 0
  if(!item&&!task)return null
  const timingNote = t("含数据准备与检查，暂停时不计时")
  return <div className="task-timing" aria-label={t("标注用时")}>
    {item&&<span>{task?t("本条用时 "):<HelpHint label={t("本条用时")} detail={timingNote}/>}<strong>{formatTiming(item, extra)}</strong></span>}
    {task&&<span><HelpHint label={t("累计运行用时")} detail={timingNote}/><strong>{formatTiming(task, extra)}</strong></span>}
    {hasIncompleteTiming(task) && <small role="status">{t("历史计时不完整，缺失部分未计入已记录用时。")}</small>}
  </div>
}

export default function BatchWorkspace({ onOpenPacket, active = true }: { onOpenPacket: (id: string, runId?: string) => void; active?: boolean }) {
  useLocale()

  const [readiness,setReadiness]=useState<{annotation_ready:boolean;supports_candidate_replacement?:boolean;supports_unlimited_batches?:boolean;supports_vision_model_checks?:boolean;attempted_target_ids:string[];replacement_unavailable_target_ids?:string[];checks:{name:string;ready:boolean;detail:string}[]}|null>(null)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [jobs, setJobs] = useState<Job[]>([]), [chosen, setChosen] = useState(()=>taskLocation().jobId)
  const [jobsLoaded, setJobsLoaded] = useState(false), [jobsError, setJobsError] = useState('')
  const [readinessError, setReadinessError] = useState('')
  const [snapshot, setDetail] = useState<JobDetail | null>(null)
  const [metricsResult, setMetricsResult] = useState<{ job_id: string; metrics?: TaskMetricsData; error?: string; pending?: string } | null>(null)
  const [metricsVersion, setMetricsVersion] = useState(0)
  const [maintenanceBusy, setMaintenanceBusy] = useState(false)
  const [activeJob, setActiveJob] = useState<string | null>(null)
  const [countText, setCountText] = useState('5')
  const [countTouched, setCountTouched] = useState(false)
  const count = parseAnnotationCount(countText)
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [syncError, setSyncError] = useState(''), [feedback, setFeedback] = useState<InterfaceMessage>('')
  const [resultPage, setResultPage] = useState(0), [resultFilter, setResultFilter] = useState('')
  const [resultScope, setResultScope] = useState('results')
  const [pollVersion, setPollVersion] = useState(0)
  const [observedAt, setObservedAt] = useState(0)
  const [recordSelection, setRecordSelection] = useState<{ jobId: string; ordinal: number } | null>(null)
  const [datasetSplit, setDatasetSplit] = useState<'train' | 'validation'>('train')
  const [pageVisible, setPageVisible] = useState(() => typeof document === 'undefined' || document.visibilityState !== 'hidden')
  const [progress, setProgress] = useState<Progress | null>(null)
  const [progressError, setProgressError] = useState('')
  const [progressObservedAt, setProgressObservedAt] = useState(0)
  const [progressUnavailable, setProgressUnavailable] = useState(false)
  const [metricsRequestedJob, setMetricsRequestedJob] = useState<string | null>(null)
  const progressJob = (chosen && jobs.some(j => j.job_id === chosen) ? chosen : '') || snapshot?.job_id || activeJob || jobs[0]?.job_id
  const live = progress?.job_id === progressJob && (progress?.measured_at ?? 0) >= (snapshot?.measured_at ?? 0) ? progress : null
  const detail = useMemo(() => {
    if (snapshot) return live ? { ...snapshot, ...live } : snapshot
    const job = jobs.find(j => j.job_id === live?.job_id)
    return job && live ? { ...job, ...live, items: [], events: [] } as JobDetail : null
  }, [snapshot, live, jobs])
  const metrics = metricsResult?.job_id === detail?.job_id ? metricsResult?.metrics : null
  const metricsError = metricsResult?.job_id === detail?.job_id ? metricsResult?.error : ''
  const metricsPending = metricsResult?.job_id === detail?.job_id ? metricsResult?.pending : ''
  const pollActive = active && pageVisible
  useEffect(()=>{
    const follow = ()=>{
      if (!window.location.hash.startsWith('#annotate?')) return
      const target=taskLocation()
      if (!target.jobId) return
      setChosen(target.jobId);setDetail(null);setProgress(null);setMetricsResult(null)
      setRecordSelection(target.ordinal?{jobId:target.jobId,ordinal:target.ordinal}:null)
      setPollVersion(v=>v+1)
    }
    window.addEventListener('hashchange',follow)
    return()=>window.removeEventListener('hashchange',follow)
  },[])
  useEffect(() => {
    const update = () => setPageVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', update)
    update()
    return () => document.removeEventListener('visibilitychange', update)
  }, [])
  useEffect(() => {
    if (!active) return
    const c = new AbortController()
    void (async () => {
      try {
        let value: Catalog | null = null
        const targets: Target[] = []
        for (let offset = 0; ; offset += 10000) {
          const page = await readPollingJson('/api/batches/catalog?compact=true&limit=10000&offset=' + offset, c.signal, 60000) as Catalog
          if (value && value.catalog_id !== page.catalog_id) throw new Error(t("目录版本变化，请刷新。"))
          value = page; targets.push(...page.targets)
          if (targets.length >= page.total) break
        }
        if (!c.signal.aborted) setCatalog({ ...value!, targets })
      } catch (e) { if (!c.signal.aborted) setError(e instanceof Error ? e.message : String(e)) }
    })()
    return () => c.abort()
  }, [active])
  useEffect(() => {
    if (maintenanceBusy || !pollActive) return
    const c = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function pollReadiness() {
      try {
        const status = await readPollingJson('/api/system/status', c.signal, 60000)
        if (!c.signal.aborted) { setReadiness(status); setReadinessError('') }
      } catch (e) { if (!c.signal.aborted) setReadinessError(errorMessage(e)) }
      finally { if (!c.signal.aborted) timer = setTimeout(() => void pollReadiness(), 30000) }
    }
    void pollReadiness()
    return () => { c.abort(); clearTimeout(timer) }
  }, [pollVersion, maintenanceBusy, pollActive])
  useEffect(() => {
    if (maintenanceBusy || !pollActive || busy) return
    const c = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    let intervalMs=10000
    async function poll() {
      let indexLoaded = false
      try {
        const index = await readPollingJson('/api/batches', c.signal, 45000) as { jobs: Job[]; active_batch_id: string | null }
        if (c.signal.aborted) return
        indexLoaded = true
        setJobsLoaded(true); setJobsError('')
        intervalMs=index.active_batch_id || index.jobs.some(job => job.executor_active || ['running', 'queued'].includes(job.status)) ? 10000 : 30000
        setJobs(index.jobs); setActiveJob(index.active_batch_id || index.jobs.find(j => j.executor_active)?.job_id || null)
        const id = (chosen && index.jobs.some(j => j.job_id === chosen) ? chosen : '') || index.active_batch_id || index.jobs[0]?.job_id
        const next = id ? await readPollingJson('/api/batches/' + id + '?compact=true', c.signal, 45000) : null
        if (!c.signal.aborted) { setDetail(next); setObservedAt(Date.now() / 1000); setSyncError('') }
      } catch (e) {
        if (!c.signal.aborted) {
          setSyncError(errorMessage(e))
          if (!indexLoaded) setJobsError(errorMessage(e))
        }
      }
      finally { if (!c.signal.aborted) timer = setTimeout(() => void poll(), intervalMs) }
    }
    void poll()
    return () => { c.abort(); clearTimeout(timer) }
  }, [chosen, pollVersion, maintenanceBusy, pollActive, busy])
  useEffect(() => {
    if (!progressJob || progressUnavailable || maintenanceBusy || !pollActive || busy) return
    const c = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function pollProgress() {
      let interval = 2000
      try {
        const value = await readPollingJson(`/api/batches/${progressJob}/progress`, c.signal) as Progress
        if (c.signal.aborted) return
        setProgress(value); setProgressError(''); setProgressObservedAt(Date.now() / 1000)
        if (!value.executor_active && !['running', 'queued'].includes(value.status)) interval = 30000
      } catch (e) {
        if (!c.signal.aborted) {
          if ((e as { status?: number })?.status === 404) {
            setProgressUnavailable(true); setProgressError(''); return
          }
          setProgressError(errorMessage(e))
        }
        interval = 5000
      } finally {
        if (!c.signal.aborted) timer = setTimeout(() => void pollProgress(), interval)
      }
    }
    void pollProgress()
    return () => { c.abort(); clearTimeout(timer) }
  }, [progressJob, progressUnavailable, pollVersion, maintenanceBusy, pollActive, busy])
  const metricsJob = detail && !detail.executor_active && ['completed', 'blocked'].includes(detail.status) ? detail.job_id : null
  function requestMetrics(jobId: string) {
    setMetricsResult({ job_id: jobId, pending: "正在汇总历史调用记录，大任务可能需要数分钟。任务列表、标注结果与导出仍可使用。" })
    setMetricsRequestedJob(jobId)
    setMetricsVersion(v => v + 1)
  }
  useEffect(() => {
    if (!metricsJob || metricsRequestedJob !== metricsJob) return
    const c = new AbortController()
    const deadline = Date.now() + 15 * 60 * 1000
    let timer: ReturnType<typeof setTimeout>
    async function loadMetrics() {
      if (c.signal.aborted) return
      if (Date.now() >= deadline) {
        setMetricsResult({ job_id: metricsJob!, error: "统计等待已超过 15 分钟，请稍后重试；已保存的标注不受影响。" })
        return
      }
      try {
        const report = await readPollingJson(`/api/batches/${metricsJob}/report`, c.signal, deadline - Date.now())
        if (!c.signal.aborted) setMetricsResult({ job_id: metricsJob!, metrics: report.core_metrics })
      } catch (e) {
        if (c.signal.aborted) return
        if ((e as { status?: number })?.status === 409 && errorMessage(e).startsWith("正在读取任务统计")) {
          setMetricsResult({ job_id: metricsJob!, pending: "已有统计正在计算，正在等待并自动重试。任务列表、标注结果与导出仍可使用。" })
          timer = setTimeout(() => void loadMetrics(), 5000)
        } else {
          setMetricsResult({ job_id: metricsJob!, error: Date.now() >= deadline
            ? "统计等待已超过 15 分钟，请稍后重试；已保存的标注不受影响。" : errorMessage(e) })
        }
      }
    }
    void loadMetrics()
    // A user-requested report is a single long read. Display polling, exports
    // and tab visibility must not abort it and leave the server working alone.
    return () => { c.abort(); clearTimeout(timer) }
  }, [metricsJob, metricsRequestedJob, metricsVersion])
  const splitTargets = useMemo(() => catalog?.targets.filter(t => (t.official_split ?? 'training') === (datasetSplit === 'train' ? 'training' : 'validation')) ?? [], [catalog, datasetSplit])
  const available = useMemo(() => {
    const unavailableIds = new Set(readiness?.replacement_unavailable_target_ids ?? readiness?.attempted_target_ids ?? [])
    return splitTargets.filter(t => t.status === 'selected' && !unavailableIds.has(t.target_id))
      .sort((a,b) => (a.recommendation_rank ?? a.ordinal) - (b.recommendation_rank ?? b.ordinal)) ?? []
  }, [splitTargets, readiness])
  let selected: Target[] = [], scopeError = ''
  if (count === null) scopeError = countText.trim() ? t("请输入大于 0 的整数。") : t("请输入期望标注条数。")
  else if (count > maxTargets) scopeError = t("单次最多填写 {0} 条描述，请分批创建。", maxTargets)
  else {
    selected = available.slice(0, count)
    if (catalog?.catalog_id && readiness && selected.length < count) scopeError = available.length
      ? t("当前只有 {0} 个可用候选，请将期望标注条数设为 {1} 以内。", available.length, available.length)
      : t("暂无尚未尝试的待验证候选，请查看已有结果。")
  }
  async function create() {
    setProgress(null)
    setBusy(true); setError(''); setFeedback('')
    try {
      if (scopeError || !selected.length) throw new Error(scopeError || t("没有可处理的目标。"))
      if (!readiness?.supports_candidate_replacement) throw new Error(t("后端尚未加载自动补选功能，请重启服务后再创建任务。"))
      if (!readiness.supports_unlimited_batches) throw new Error(t("后端尚未加载取消任务总调用上限的修改，请重启服务后再创建任务。"))
      if (!readiness.supports_vision_model_checks) throw new Error(t("后端尚未加载模型图像能力检查，请重启服务后再创建任务。"))
      const response = await readJson(await fetch('/api/batches', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ catalog_id: catalog!.catalog_id, dataset_split: datasetSplit, target_ids: available.map(t => t.target_id), desired_count: selected.length, prepare_only: false }) }))
      setChosen(response.job_id); setDetail(await readJson(await fetch('/api/batches/' + response.job_id + '?compact=true')))
      setMetricsResult(null); setResultPage(0); setResultFilter(''); setResultScope('results')
      setFeedback(response.reused_job ? "已打开相同配置的已有任务。" : "任务已创建。点击“开始自动标注”执行。")
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }
  async function control(command: 'start' | 'pause' | 'stop' | 'resume') {
    setProgress(null)
    setBusy(true); setError(''); setFeedback('')
    try {
      const receipt = await readJson(await fetch('/api/batches/' + detail!.job_id + (command === 'start' ? '/start' : '/control'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: command === 'start' ? undefined : JSON.stringify({ command }) }))
      setPollVersion(v => v + 1)
      setFeedback(receipt.queued ? ["已加入队列，当前第 {0} 位。前面的任务完成后自动开始。", receipt.queue_position] : '')
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }
  async function deleteSample(item: Item) {
    if (!detail || busy || !item.run_id || !item.packet_id || !item.expression) return
    if (!window.confirm(t("确认删除标注 #{0}？\n\n{1}\n\n删除后会扣减有效标注进度，并从后续导出中排除。原始数据和费用记录保留。", item.ordinal, item.expression))) return
    const jobId = detail.job_id
    setProgress(null)
    setBusy(true); setError(''); setFeedback('')
    try {
      const { progress, jobs: updatedJobs, ...decision } = await readJson(await fetch(`/api/batches/${jobId}/samples/${item.ordinal}`, {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: item.run_id, packet_id: item.packet_id, expression: item.expression }),
      })) as ManualReview & { progress: { passed: number }; jobs: Job[] }
      setJobs(updatedJobs)
      setDetail(current => current?.job_id === jobId ? { ...current,
        counts: updatedJobs.find(job => job.job_id === jobId)?.counts ?? current.counts,
        selection_progress: current.selection_progress ? { ...current.selection_progress, diagnostic_passed: progress.passed } : current.selection_progress,
        items: current.items.map(row =>
        row.run_id === item.run_id ? { ...row, manual_review: decision } : row) } : current)
      setMetricsResult(null); setPollVersion(v => v + 1)
      setFeedback(["已删除 #{0}，已从有效结果和后续导出中排除。", item.ordinal])
    } catch (e) { setError(errorMessage(e)) } finally { setBusy(false) }
  }
  const recordItems = detail?.items.filter(isProcessingRecord) ?? []
  const processedCounts = recordItems.reduce<Record<string, number>>((counts, item) => {
    const status = item.manual_approved ? 'manual_approved' : ['rejected','discarded'].includes(item.manual_review?.verdict ?? '') ? item.manual_review!.verdict : item.status
    counts[status] = (counts[status] ?? 0) + 1
    return counts
  }, {})
  const activeItem = recordItems.find(i => ['preparing','running'].includes(i.status))
  const timedItem = activeItem ?? recordItems.at(-1)
  const itemTiming = live && timedItem ? live.active_item_timings[String(timedItem.ordinal)]
    ?? (timedItem.timing ? { ...timedItem.timing, running: false } : undefined) : timedItem?.timing
  const parallelCount = live ? Object.keys(live.active_item_timings).length : detail?.items.filter(i => i.timing?.running).length ?? 0
  const latestItemEvent = detail?.events.findLast(e => Number.isInteger(e.payload.ordinal))
  const followingItem = detail?.status === 'ready' ? undefined : activeItem
    ?? recordItems.find(i => i.ordinal === latestItemEvent?.payload.ordinal)
    ?? recordItems.at(-1)
  const manualItem = recordSelection?.jobId === detail?.job_id
    ? recordItems.find(i => i.ordinal === recordSelection?.ordinal) : undefined
  const viewingItem = manualItem ?? followingItem
  const viewingIndex = recordItems.findIndex(i => i.ordinal === viewingItem?.ordinal)
  function viewRecord(index: number) {
    const item = recordItems[index]
    if (detail && item) setRecordSelection({ jobId: detail.job_id, ordinal: item.ordinal })
  }
  function openProcess(item: Item) {
    if (item.expression && item.packet_id) onOpenPacket(item.packet_id, item.run_id ?? undefined)
    else {
      viewRecord(recordItems.findIndex(i => i.ordinal === item.ordinal))
      document.querySelector(t("[aria-label=\"标注物体记录\"]"))?.scrollIntoView({ block: 'start', behavior: 'smooth' })
    }
  }
  const preparationEvent = detail?.events.findLast(e => e.kind === 'preparation_progress' && e.payload.ordinal === viewingItem?.ordinal)
  const preparationText = preparationStage(preparationEvent?.payload.stage)
  const preparationStarted = detail?.events.findLast(e => e.kind === 'preparation_started' && e.payload.ordinal === viewingItem?.ordinal)?.at
  const preparationSeconds = preparationStarted ? Math.max(0, Math.floor(observedAt - preparationStarted)) : 0
  const sceneWaitingTitle = viewingItem && ['pending','ready'].includes(viewingItem.status) ? t("此物体尚未处理")
    : viewingItem?.status === 'not_needed' ? t("此候选未尝试")
    : viewingItem?.status === 'failed' ? t("数据准备未完成")
    : viewingItem?.status === 'budget_skipped' || viewingItem?.status === 'excluded' ? t("已跳过此物体")
    : detail && ['paused','stopped'].includes(detail.status) ? t("场景准备已暂停")
    : detail && ['completed','blocked'].includes(detail.status) ? t("暂无可显示的场景") : t("正在准备场景")
  const sceneWaitingMessage = viewingItem && ['pending','ready'].includes(viewingItem.status) ? t("本条记录还没有生成点云、图像和描述。")
    : viewingItem?.status === 'not_needed' ? t("本任务未为此候选准备场景或调用模型；具体原因见处理记录。")
    : viewingItem?.status === 'excluded' ? t("此物体未通过预检查，没有本次标注场景。")
    : viewingItem?.status === 'failed' ? t("当前物体的数据准备失败，暂无可显示的点云与图像。")
    : viewingItem?.status === 'budget_skipped' ? t("当前物体因调用额度不足未执行标注，没有本次场景预览。")
    : detail && ['paused', 'stopped'].includes(detail.status) ? t("当前物体的数据尚未准备完成，继续任务后会自动显示。")
    : detail && ['completed', 'blocked'].includes(detail.status) ? t("当前物体没有可显示的点云与图像，请查看处理原因。")
    : t("{0}{1}。同帧数据直接复用；新帧仅转换一次，完成后自动显示。", preparationText || t("正在准备当前物体的点云与图像"), preparationStarted ? t(" · 已等待 {0} 秒", preparationSeconds) : '')
  const results = detail?.items.filter(i => detail.spec.prepare_only ? i.status === 'prepared' : isAnnotationResult(i)) ?? []
  const displayedRecords = resultScope === 'history' ? detail?.items ?? [] : results
  const items = displayedRecords.filter(i => !resultFilter || i.status === resultFilter)
  const visiblePage = Math.max(0, Math.min(resultPage, Math.ceil(items.length / 25) - 1))
  const progressHealthy = !!live && !progressError
  const disabled = busy || !!syncError && !progressHealthy
  const terminal = detail && ['completed','blocked'].includes(detail.status)
  const settled = detail?.items.filter(i => settledStatuses.has(i.status)).length ?? 0
  const failureItem = detail?.items.find(i => i.status === 'failed')
  let failureReason = detail && ['blocked','paused'].includes(detail.status)
    ? (detail.reason === 'preparation_failed' || detail.reason === 'child_run_blocked'
      ? failureItem?.reason ?? detail.reason : detail.reason) : null
  if (detail?.status === 'paused' && detail.reason?.startsWith('preparation:official_conversion_failed:')) failureReason = detail.reason
  const recoverable = !!detail?.items.some(i => i.retryable_parameter_failure || i.retryable_preparation_failure || i.retryable_connection_failure || (i.retryable_visibility_failure && i.legacy_rules))
  const goalReached = !!detail?.spec.desired_count && (detail.selection_progress?.diagnostic_passed ?? 0) >= detail.spec.desired_count
  const hasPending = !!detail?.items.some(i => ['pending','budget_skipped'].includes(i.status)
    || i.status === 'not_needed' && i.reason === 'description_goal_reached' && !i.run_id)
  const hasContinuationBudget = !!detail && (detail.spec.prepare_only || detail.spec.max_model_calls == null || detail.model_calls_reserved + 4 <= detail.spec.max_model_calls)
  const canContinue = terminal && !goalReached && hasContinuationBudget && (hasPending || recoverable)
  const controlPending = !!detail?.executor_active && !!detail.control
  let taskAction: {command:'start'|'pause'|'resume'; label:string}|undefined
  if(detail&&!controlPending){
    const resumeLabel=detail.spec.prepare_only?t("继续数据准备"):t("继续标注")
    if(detail.status==='ready')taskAction={command:'start',
      label:activeJob&&activeJob!==detail.job_id?t("加入队列"):detail.continuation?resumeLabel:detail.spec.prepare_only?t("开始数据准备"):t("开始自动标注")}
    else if(detail.status==='queued'&&detail.queue_paused)taskAction={command:'resume',label:t("继续队列")}
    else if(detail.status==='running'&&detail.executor_active)taskAction={command:'pause',label:detail.spec.prepare_only?t("暂停数据准备"):t("暂停标注")}
    else if(['paused','stopped','running'].includes(detail.status)||canContinue)taskAction={command:'resume',label:resumeLabel}
  }
  const primaryTaskAction=taskAction
  const taskStatus=detail&&(
    controlPending?(detail.control==='stop'?t("正在停止"):t("正在暂停")):
    detail.status==='queued'&&detail.queue_paused?t("队列已暂停"):
    detail.status==='running'&&!detail.executor_active?t("等待恢复"):
    labels[detail.status]??detail.status)
  const sampleFinished = !!viewingItem && ['prepared','calibration_hold','reused'].includes(viewingItem.status)
  const sampleStage = sampleFinished ? 3 : viewingItem?.status === 'preparing' ? 0
    : ['appearance','ground','ground_permuted','scene_ground','scene_ground_permuted','counterfactual'].includes(viewingItem?.phase ?? '') ? 2
    : ['generate','decide','execute'].includes(viewingItem?.phase ?? '') ? 1 : 0
  const sampleFailed = !!viewingItem && ['failed','abstained','excluded','needs_evidence','blocked'].includes(viewingItem.status)
  const sampleExecuting = detail?.status === 'running' && detail.executor_active && !detail.control && !syncError
    && !!viewingItem && ['preparing','running'].includes(viewingItem.status)
  const goal = detail?.spec.desired_count
  const progressValue = goal ? detail?.selection_progress?.diagnostic_passed ?? 0 : settled
  const progressTotal = goal || detail?.items.length || 1
  const progressPercent = Math.round(progressValue / progressTotal * 100)
  const countHelp = readiness && catalog && available.length > 0
    ? t("可填写 1–{0}，未通过时自动补选其他物体。", Math.min(maxTargets, available.length)) : ''
  const inventoryHelp = t("同一驾驶片段内按物体去重。系统会自动补选，直至达到通过条数或候选用尽。训练与验证沿用 Waymo 官方划分。")
  const viewedStatus = viewingItem && detail ? recordStatus(viewingItem, detail) : null
  const viewedReason = viewingItem?.reason ? reason(viewingItem.display_reason ?? viewingItem.reason, viewingItem.provider_diagnosis, viewingItem.failure_diagnostic?.provider_error_code) : ''
  return <div className="batch-workspace">
    {!detail && error && <p className="error" role="alert">{serviceText(error)}</p>}
    {syncError && <p className={progressHealthy ? 'notice' : 'error'} role="status">{progressHealthy ? t("结果列表同步暂时中断，进度仍在更新：") : t("状态同步失败：")}{serviceText(syncError)}<button className="text-button" onClick={() => setPollVersion(v => v + 1)}>{t("立即重试")}</button></p>}
    {progressError && <p className="notice" role="status">{t("进度同步暂时中断：")}{serviceText(progressError)}<button className="text-button" onClick={() => setPollVersion(v => v + 1)}>{t("立即重试")}</button></p>}
    {readinessError && <p className="error" role="alert">{t("系统状态读取失败：")}{serviceText(readinessError)}{t("。已有任务仍可查看。")}</p>}
    <form className="panel setup-panel" aria-labelledby="setup-heading" noValidate onSubmit={e => { e.preventDefault(); setCountTouched(true); if (!busy && !syncError && !readinessError && readiness?.annotation_ready && readiness.supports_candidate_replacement && readiness.supports_unlimited_batches && readiness.supports_vision_model_checks && catalog?.catalog_id && !scopeError && selected.length) void create() }}>
      <div className="section-heading"><h2 id="setup-heading">{t("新建标注任务")}</h2></div>
      <div className="setup-layout">
        <div className="setup-fields">
          <label className="dataset-task-choice"><span>{t("标注数据集")}</span><select aria-label={t("标注数据集")} value={datasetSplit} disabled={busy} onChange={e => { setDatasetSplit(e.target.value as 'train' | 'validation'); setCountTouched(false); setError('') }}><option value="train">{t("官方训练集")}</option><option value="validation">{t("官方验证集")}</option></select></label>
          <div className="count-field">
            <div className="field-title"><label htmlFor="annotation-count">{t("期望通过条数")}</label>{countHelp&&<HelpHint label={t("期望通过条数")} detail={countHelp} iconOnly/>}</div>
            <div className="setup-action">
              <span className="number-input"><input id="annotation-count" aria-label={t("期望标注条数")} aria-describedby={`target-count-help${countTouched && scopeError ? ' target-count-error' : ''}`} aria-invalid={countTouched && !!scopeError} type="text" inputMode="numeric" autoComplete="off" placeholder={t("填写数量")} value={countText} onChange={e => { setCountText(e.target.value); setCountTouched(false) }} onBlur={() => { setCountTouched(true); if (count !== null) setCountText(String(count)) }} /><span>{t("条描述")}</span></span>
              <button className="primary" type="submit" disabled={busy || !!syncError || !!readinessError || !readiness?.annotation_ready || !readiness.supports_candidate_replacement || !readiness.supports_unlimited_batches || !readiness.supports_vision_model_checks || !catalog?.catalog_id || !!scopeError || !selected.length}>{busy ? t("处理中…") : t("创建任务")}</button>
            </div>
            <small id="target-count-help" className={countHelp?'sr-only':undefined}>{countHelp||(readiness && catalog ? t("暂无可用候选，请查看已有任务的处理结果。") : t("正在读取可选物体…"))}</small>
          </div>
        </div>
        <section className="setup-inventory" aria-labelledby="setup-inventory-heading">
          <h3 id="setup-inventory-heading"><HelpHint label={t("候选概况")} detail={[inventoryHelp, catalog?.summary.pending_pairing?t("另有 {0} 个物体尚未找到对应图像，暂不能标注。", catalog.summary.pending_pairing):''].filter(Boolean).join(' ')} /></h3>
          <dl className="setup-stats" aria-label={t("数据与候选范围")}>
            <div className="available-stat"><dt><TermHelp term="availableObjects" /></dt><dd>{readiness && catalog ? available.length.toLocaleString('en-US') : '—'}</dd></div>
            <div><dt><TermHelp term="eligibleObjects" /></dt><dd>{catalog ? splitTargets.filter(t => t.status === 'selected').length.toLocaleString('en-US') : '—'}</dd></div>
            <div><dt><TermHelp term="datasetObjects" label={t("数据集物体数")} /></dt><dd>{catalog ? splitTargets.length.toLocaleString('en-US') : '—'}</dd></div>
            <div><dt><TermHelp term="context" label={t("驾驶片段")} /></dt><dd>{catalog ? new Set(splitTargets.map(t => t.context)).size : '—'}</dd></div>
          </dl>
        </section>
      </div>
      {countTouched && scopeError && <p id="target-count-error" className="error" role="alert">{scopeError}</p>}
      {readiness && !readiness.supports_candidate_replacement && <p className="notice" role="status">{t("后端尚未加载自动补选功能，请由你重启服务后再创建任务。已有任务仍可查看。")}</p>}
      {readiness?.supports_candidate_replacement && !readiness.supports_unlimited_batches && <p className="notice" role="status">{t("后端尚未加载取消任务总调用上限的修改，请由你重启服务后再创建任务。")}</p>}
      {readiness?.supports_unlimited_batches && !readiness.supports_vision_model_checks && <p className="notice" role="status">{t("后端尚未加载模型图像能力检查，请由你重启服务后再创建任务。")}</p>}
      {readiness && !readiness.annotation_ready && <p className="error" role="alert">{t("暂无法创建任务：")}{readiness.checks.filter(c => !c.ready).map(c => t("{0}缺失，{1}", serviceText(c.name), serviceText(c.detail))).join(t('；'))}</p>}
    </form>

    <TaskManager jobs={jobs} loading={!jobsLoaded && !jobsError} loadError={jobsError} onReload={() => setPollVersion(v => v + 1)} onBusyChange={setMaintenanceBusy} current={chosen || detail?.job_id} activeJobId={activeJob ?? undefined} labels={labels}
      onOpen={id => { if (id === (chosen || detail?.job_id)) return; setChosen(id); setDetail(null); setMetricsResult(null); setResultPage(0); setResultFilter(''); setResultScope('results'); setFeedback('') }}
      onDeleted={ids => { setJobs(previous => previous.filter(j => !ids.includes(j.job_id))); if ((detail && ids.includes(detail.job_id)) || ids.includes(chosen)) { setChosen(''); setDetail(null); setMetricsResult(null); setResultPage(0); setResultFilter(''); setResultScope('results'); setFeedback('') } setPollVersion(v => v + 1) }} />
    {(detail || feedback || (jobs.length > 0 && !syncError)) && <div className="task-workspace">
        {feedback && <p role="status" className="operation-feedback">{serviceText(feedback)}</p>}
        {!detail ? jobs.length > 0 && !syncError && <p className="task-loading muted" role="status">{t("正在读取任务…")}</p> : <>
          <section className="panel progress-panel" aria-labelledby="progress-heading">
            <div className="section-heading progress-heading">
              <h2 id="progress-heading">{t("标注进度")}</h2>
              <div className="progress-actions">
                <span role="status" className={`status-badge status-${detail.status}`}>{taskStatus}</span>
                {primaryTaskAction && <button className="primary" disabled={disabled || (primaryTaskAction.command!=='pause' && (detail.executor_active || detail.resume?.mode==='unavailable'))} onClick={() => void control(primaryTaskAction.command)}>{primaryTaskAction.label}</button>}
                {detail.status==='queued'&&<button disabled={disabled} onClick={() => void control('pause')}>{t("取消排队")}</button>}
                {!terminal&&!controlPending&&['running','paused'].includes(detail.status)&&<button disabled={disabled} onClick={() => void control('stop')}>{t("停止任务")}</button>}
              </div>
            </div>
            {detail.status === 'ready' && !detail.spec.prepare_only && <p className="progress-action-note muted">{t("开始后将调用模型并产生 API 费用。")}</p>}
            {controlPending && <p className="progress-action-note muted">{t("正在保存当前步骤，随后")}{detail.control==='stop'?t("停止"):t("暂停")}。</p>}
            <div className="progress-numbers"><div><strong>{progressValue}</strong><span> / {progressTotal} {goal ? t("条可用描述") : t("个已处理")}</span></div><span>{progressPercent}%</span></div>
            <progress className="batch-progress" aria-label={goal ? t("描述数量进度") : t("目标处理进度")} value={progressValue} max={progressTotal} />
            <LiveTiming item={itemTiming} task={detail.timing} observedAt={live ? progressObservedAt : observedAt} active={pollActive && !progressError} />
            {detail.executor_active && parallelCount > 1 && <p className="muted">{t("正在并行处理 ")}{parallelCount}{t(" 个目标")}</p>}
            {detail.status === 'completed' && detail.reason && (detail.reason !== 'description_goal_reached' || !goalReached) && completionReasons[detail.reason] && <p className="notice" role="status">{completionReasons[detail.reason]}</p>}
            <div className="progress-footnote">
              {goal && <span>{t("已尝试 ")}{detail.selection_progress?.attempted ?? 0}{t(" 个物体")}</span>}
              <span>{Object.entries(goal ? live ? Object.fromEntries(Object.entries(detail.counts).filter(([s]) => !['pending','not_needed','budget_skipped'].includes(s))) : processedCounts : detail.counts).map(([s,n]) => `${labels[s] ?? s} ${n}`).join(' · ')}</span>
              <span>{t("模型调用 ")}{detail.model_calls_reserved}{detail.spec.max_model_calls != null ? ` / ${detail.spec.max_model_calls}` : ''}{t(" 次")}</span>
            </div>
            {detail.spec.prepare_only && <p className="muted">{t("本任务仅准备数据，不生成文本或调用模型。")}</p>}
            {detail.status === 'queued' && <p className="notice" role="status">{t("排队第 ")}{detail.queue_position ?? '—'}{t(" 位。")}{detail.queue_paused ? t("队列已暂停，点击“继续队列”后恢复自动调度。") : t("前面的任务完成后自动开始，排队期间不调用模型。")}</p>}
            {detail.status === 'paused' && detail.reason && detail.reason !== 'repeated_response_shape_failures' && !detail.executor_active && <p className="notice">{t("执行异常已暂停，其它目标的进度已保存。排查失败记录后可继续。")}</p>}
            {detail.resume && detail.resume.mode !== 'resume' && <p className="notice" role="status">{serviceText(detail.resume.message)}</p>}
            {error && <p className="error" role="alert">{serviceText(error)}</p>}
            {failureReason && <div className="task-alert" role="status"><strong>{t("任务执行中断")}</strong><p>{reason(failureReason, failureItem?.provider_diagnosis, failureItem?.failure_diagnostic?.provider_error_code)}</p>{failureReason !== 'repeated_response_shape_failures' && failureItem?.failure_diagnostic?.http_status && <p>HTTP {failureItem.failure_diagnostic.http_status}{failureItem.failure_diagnostic.provider_error_code && ` · ${failureItem.failure_diagnostic.provider_error_code}`}</p>}</div>}
            {activeJob && activeJob !== detail.job_id && <p className="notice">{t("另一任务正在执行，可从上方任务列表切换查看。")}</p>}
            {detail.status === 'running' && !detail.executor_active && <p className="notice">{t("执行器已中断，点击“继续标注”恢复已保存进度。")}</p>}
          </section>

          {viewingItem && viewedStatus && <section className="panel live-sample" aria-label={t("标注物体记录")} aria-busy={!!syncError}>
            <div className="section-heading"><div><h2>{sampleExecuting ? (viewingItem.phase?.startsWith('precheck_') ? t("正在预检查") : t("正在标注")) : t("标注物体记录")}</h2><p className="section-caption">{categoryLabel(viewingItem.target.category)} · #{viewingItem.ordinal}{viewingItem.source_view&&<> · {t("来源相机 · ")}{cameraViewLabel(viewingItem.source_view)}</>}</p></div><span className={`record-state record-state-${viewedStatus.tone}`} role="status">{viewedReason&&isAnnotationResult(viewingItem)?<HelpHint label={viewedStatus.label} detail={viewedReason}/>:viewedStatus.label}</span></div>
            {sampleExecuting && viewingItem.phase && <p className="section-caption">{t("当前步骤：")}{phases[viewingItem.phase] ?? t("正在处理")}</p>}
            {viewingItem.legacy_rules&&<p className="notice">{t("此记录保留其完成时的标注规则与校验结果。")}</p>}
            <nav className="record-navigation" aria-label={t("切换标注物体记录")}>
              <button disabled={viewingIndex <= 0} onClick={() => viewRecord(viewingIndex - 1)}>{t("上一个")}</button>
              <label>{t("处理记录")}<select aria-label={t("按索引跳转物体记录")} value={viewingItem.ordinal} onChange={e => setRecordSelection({jobId: detail.job_id, ordinal: Number(e.target.value)})}>{recordItems.map((item, index) => <option key={item.ordinal} value={item.ordinal}>{t("第 {0} 个 · #{1}", index + 1, item.ordinal)} · {categoryLabel(item.target.category)}</option>)}</select></label>
              <button disabled={viewingIndex >= recordItems.length - 1} onClick={() => viewRecord(viewingIndex + 1)}>{t("下一个")}</button>
              {detail.status === 'running' && <button className="text-button" aria-pressed={!manualItem} disabled={!manualItem} onClick={() => setRecordSelection(null)}>{manualItem ? t("跟随标注进度") : t("正在跟随进度")}</button>}
            </nav>
            <ol className="sample-steps" aria-label={t("当前物体处理步骤")}>{[t("准备与预检查"),t("生成描述"),t("检查描述"),t("保存结果")].map((step,index) => <li key={step} aria-current={index === sampleStage ? 'step' : undefined} className={sampleFinished || index < sampleStage ? 'done' : index === sampleStage ? sampleFailed ? 'failed' : 'current' : ''}><span aria-hidden="true">{sampleFinished || index < sampleStage ? '✓' : index + 1}</span>{step}</li>)}</ol>
            {viewedReason && !isAnnotationResult(viewingItem) && <p className="notice" role="status">{viewedReason}</p>}
            {viewingItem.failure_diagnostic?.http_status && <p className="notice">HTTP {viewingItem.failure_diagnostic.http_status}{viewingItem.failure_diagnostic.provider_error_code && ` · ${viewingItem.failure_diagnostic.provider_error_code}`}</p>}
            {viewingItem.reason === 'precheck_target_occluded_or_uncertain' && viewingItem.visibility_observation?.visibility_reason && <p className="muted">{t("来源视角的模型判断：")}<span lang="en">{viewingItem.visibility_observation.visibility_reason}</span></p>}
            {!syncError && viewingItem.expression && <div className="live-description"><span>{viewingItem.manual_review?.verdict === 'discarded' ? t("已删除的描述 · 不用于训练") : viewingItem.manual_review?.verdict === 'rejected' ? t("人工不通过的描述 · 不用于训练") : ['needs_evidence','abstained','blocked'].includes(viewingItem.status) ? t("未通过的候选描述 · 不用于训练") : t("当前候选描述")}</span><p className="expression" lang="en">{viewingItem.expression}</p></div>}
            {syncError ? <p className="notice" role="status">{t("进度同步中断，已暂停场景展示；恢复后自动跟随当前物体。")}</p> : viewingItem.packet_id && (viewingItem.admission_status === 'passed' || viewingItem.expression || manualItem)
              ? <AlignmentInspection key={detail.job_id + ':' + viewingItem.ordinal + ':' + viewingItem.packet_id + ':' + viewingItem.source_view} packetId={viewingItem.packet_id} preferredSourceView={viewingItem.source_view} liveTarget={viewingItem.display_target ?? viewingItem.target} embedded />
              : ['provider_account_rejected','provider_model_rejected'].includes(viewingItem.provider_diagnosis ?? '') ? null
              : viewingItem.packet_id ? <div className="scene-placeholder" role="status"><strong>{sampleExecuting ? t("正在确认来源视角中的目标身份") : t("候选尚未获得可靠描述，已跳过")}</strong><p>{sampleExecuting ? t("确认能辨认所选物体，并寻找可区分它的描述线索。局部遮挡不会直接否决标注。") : t("未将此候选作为标注样本；处理记录和原因已保留。")}</p></div>
              : <div className="scene-placeholder" role="status">{sampleExecuting&&<span className="loading-dot" aria-hidden="true" />}<strong>{sceneWaitingTitle}</strong><p>{sceneWaitingMessage}</p></div>}
          </section>}

          <section className="panel results-panel" aria-labelledby="results-heading">
            <div className="section-heading"><h2 id="results-heading">{t("标注结果")}</h2></div>
            <div className="filter-row"><label><span className="sr-only">{t("记录范围")}</span><select aria-label={t("记录范围")} value={resultScope} onChange={e => { setResultScope(e.target.value); setResultFilter(''); setResultPage(0) }}><option value="results">{detail.spec.prepare_only ? t("已准备数据") : t("通过自动检查的描述")}</option><option value="history">{t("处理记录（含待处理候选）")}</option></select></label><label><span className="sr-only">{t("处理状态")}</span><select aria-label={t("处理状态")} value={resultFilter} onChange={e => { setResultFilter(e.target.value); setResultPage(0) }}><option value="">{t("全部状态")}</option>{Array.from(new Set(displayedRecords.map(i => i.status))).map(s => <option key={s} value={s}>{labels[s] ?? s}</option>)}</select></label><span>{t("共 ")}{items.length}{t(" 条")}</span></div>
            <div className="table-wrap"><table className="result-table"><thead><tr><th>{t("物体")}</th><th>{t("候选描述")}</th><th>{t("来源视角")}</th><th>{t("状态")}</th><th><span className="sr-only">{t("操作")}</span></th></tr></thead><tbody>{items.slice(visiblePage * 25,(visiblePage + 1) * 25).map(i => <tr key={i.ordinal}><td><strong>{categoryLabel(i.target.category)}</strong><small className="object-number">#{i.ordinal}</small></td><td lang={i.expression ? 'en' : 'zh'}>{i.expression ?? <span className="placeholder-text">{i.status === 'pending' || i.status === 'ready' ? t("待生成") : t("未生成描述")}</span>}</td><td>{cameraViewLabel(i.source_view)}{i.source_view_status === 'candidate' && t("（候选）")}</td><td><span title={i.reason ? reason(i.reason, i.provider_diagnosis, i.failure_diagnostic?.provider_error_code) : undefined} className={`record-state record-state-${recordStatus(i,detail).tone}`}>{recordStatus(i,detail).label}</span>{i.legacy_rules&&<small className="legacy-rule-label">{t("旧规则记录")}</small>}</td><td><div className="record-actions">{i.packet_id && <button className="row-action" onClick={() => openProcess(i)}>{t("查看过程 →")}</button>}{i.expression && i.run_id && i.packet_id && <button className="row-action delete-record" aria-label={t("删除标注 #{0}", i.ordinal)} disabled={busy || ['running','paused','stopped'].includes(i.run_status ?? i.status) || i.manual_review?.verdict === 'discarded'} onClick={() => void deleteSample(i)}>{i.manual_review?.verdict === 'discarded' ? t("已删除") : t("删除")}</button>}</div></td></tr>)}</tbody></table></div>
            {!items.length && <p className="empty">{!snapshot ? t("正在加载处理记录，进度已实时同步…") : resultScope === 'history' ? t("当前状态下没有处理记录。") : detail.spec.prepare_only ? t("暂无已准备数据。") : t("暂无符合当前筛选条件的通过描述；失败和待处理候选可在“处理记录”中查看。")}</p>}
            {items.length > 25 && <div className="pagination"><button disabled={!visiblePage} onClick={() => setResultPage(visiblePage - 1)}>{t("上一页")}</button><span>{visiblePage + 1} / {Math.ceil(items.length / 25)}</span><button disabled={(visiblePage + 1) * 25 >= items.length} onClick={() => setResultPage(visiblePage + 1)}>{t("下一页")}</button></div>}
          </section>
          {metricsJob && !detail.spec.prepare_only && !metrics && !metricsError && <section className="panel" aria-label={t("任务统计入口")}><div className="section-heading metrics-heading"><h2>{t("任务统计")}</h2><button disabled={metricsRequestedJob===metricsJob} onClick={()=>requestMetrics(metricsJob)}>{metricsRequestedJob===metricsJob?t("正在计算统计…"):t("查看任务统计")}</button></div>{metricsPending&&<p className="muted" role="status">{serviceText(metricsPending)}</p>}</section>}
          {metrics && !detail.spec.prepare_only && <TaskMetrics metrics={metrics} jobId={detail.job_id} />} 
          {metricsError && <p className="notice" role="status">{t("统计读取失败：")}{serviceText(metricsError)}{t(" 任务列表与标注结果仍可使用。")}<button className="text-button" onClick={() => requestMetrics(detail.job_id)}>{t("重试统计")}</button></p>}
        </>}
    </div>}
    {!!jobs.length&&<DatasetExport jobs={jobs} current={chosen || detail?.job_id} />}
  </div>
}
