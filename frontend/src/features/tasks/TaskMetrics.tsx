import { categoryLabel, reasonLabel } from '../../shared/annotationDisplay'
import { formatDuration, formatTiming, hasIncompleteTiming } from '../../shared/duration'

export type ItemTiming = { active_seconds: number | null; recorded_active_seconds?: number | null; history_incomplete?: boolean; running: boolean; complete: boolean }
export type TaskTiming = ItemTiming & { status: string; wall_seconds: number | null; wall_running: boolean }
type Rate = { numerator: number; denominator: number; value: number | null; ci95?: [number, number] | null }
type Distribution = { count: number; mean: number | null; p50: number | null; p95: number | null }
export type TaskMetricsData = {
  schema_version?: string
  timing: TaskTiming
  counts: { pool: number; attempted: number; settled: number; texts: number; passed: number; new_passed: number; reused: number; unattempted: number; automatic_passed?:number; human_approved?:number }
  coverage: { pool_utilization: Rate; dataset_utilization: Rate; eligible_catalog_utilization: Rate; attempt_coverage: Rate; pass_per_settled: Rate }
  speed: { item_active_seconds: Distribution; passed_item_active_seconds: Distribution; mean_active_seconds_per_new_pass: number | null; new_passes_per_active_hour: number | null; new_passes_per_wall_hour: number | null }
  quality: { first_draft_pass: Rate; first_draft_incomplete: number; repair_success: Rate; counterfactuals: { complete_per_workload: Rate; pass_per_complete: Rate } }
  candidate_coverage: Rate
  cost: { model_calls: number; reserved_model_calls: number; tokens: number; tokens_complete: boolean; model_seconds: number; model_seconds_complete: boolean; calls_per_new_pass: Rate; tokens_per_new_pass: number | null }
  independent_acceptance: Rate | null
  independent_audits: { cohort_id: string; purpose: string; status: string; reviewed: number; sampled: number; report: { human_joint_acceptance: Rate; blind_unique_target_accuracy: Rate } | null }[]
  failure_reasons: { reason: string; count: number }[]
  categories: { category: string; pool: number; attempted: number; settled: number; passed: number; pass_per_settled: Rate }[]
}

const percent = (rate: Rate | null) => rate?.value == null ? '未测量' : `${(rate.value * 100).toFixed(1)}%`
const fraction = (rate: Rate) => `${rate.numerator} / ${rate.denominator}`
const number = (value: number | null) => value == null ? '未测量' : value.toLocaleString(undefined, { maximumFractionDigits: 2 })
const interval = (rate: Rate) => rate.ci95 ? `95% CI ${(rate.ci95[0] * 100).toFixed(1)}%–${(rate.ci95[1] * 100).toFixed(1)}%` : '未测量置信区间'

export default function TaskMetrics({ metrics: m, jobId }: { metrics: TaskMetricsData; jobId: string }) {
  if (m.schema_version !== 'task-metrics-v1') return <p className="notice">重启后端后可查看新版任务指标；旧任务缺失的计时不会补算。</p>
  const latency = m.speed.item_active_seconds
  return <section className="panel task-metrics" aria-label="任务统计">
    <div className="section-heading"><div><h2>任务统计</h2><p className="section-caption">仅统计当前任务，同一物体只算一次。</p></div><a className="button-link" href={`/api/batches/${jobId}/report?download=true`}>导出统计 JSON ↓</a></div>
    <p className="muted">候选 {m.counts.pool} · 已尝试 {m.counts.attempted} · 已结束 {m.counts.settled} · 当前可用 {m.counts.passed} · 未尝试 {m.counts.unattempted}{m.counts.reused ? ` · 复用 ${m.counts.reused}` : ''}</p>
    {m.counts.human_approved!=null&&<p className="muted">自动核验通过 {m.counts.automatic_passed??0} · 人工审批纳入 {m.counts.human_approved}；人工审批不计入自动核验通过率。</p>}
    <div className="metric-grid">
      <div><span>任务候选利用率</span><strong>{percent(m.coverage.pool_utilization)}</strong><small>通过 / 本任务候选：{fraction(m.coverage.pool_utilization)}</small></div>
      <div><span>数据集标注覆盖率</span><strong>{percent(m.coverage.dataset_utilization)}</strong><small>通过描述 / 数据集物体总数：{fraction(m.coverage.dataset_utilization)}</small></div>
      <div><span>尝试通过率</span><strong>{percent(m.coverage.pass_per_settled)}</strong><small>通过 / 已结束尝试：{fraction(m.coverage.pass_per_settled)}</small></div>
      <div><span>任务墙钟用时</span><strong>{formatDuration(m.timing.wall_seconds)}</strong><small>从首次启动至结束，包含暂停</small></div>
      <div><span>累计运行用时</span><strong>{formatTiming(m.timing)}</strong><small>包含准备、失败与核验，排除暂停</small></div>
      <div><span>平均每条通过描述耗时</span><strong>{formatDuration(m.speed.mean_active_seconds_per_new_pass)}</strong><small>累计运行用时 / 新产出 {m.counts.new_passed} 条</small></div>
      <div><span>处理耗时 P50 / P95</span><strong>{formatDuration(latency.p50)} / {formatDuration(latency.p95)}</strong><small>{latency.count} 条完整计时，含失败尝试；均值 {formatDuration(latency.mean)}</small></div>
      <div><span>通过描述吞吐量</span><strong>{number(m.speed.new_passes_per_active_hour)} 条/时</strong><small>按运行用时；按墙钟 {number(m.speed.new_passes_per_wall_hour)} 条/时</small></div>
    </div>
    {hasIncompleteTiming(m.timing) && <p className="notice">历史计时不完整，缺失部分未计入已记录用时；相关速度指标无法完整测量。</p>}
    <details><summary>质量与模型用量</summary><p className="muted">自动检查衡量内部一致性；首稿、修订与改词检查按执行记录统计，标注准确性由独立质量评估衡量。</p>
      <div className="metric-grid">
        <div><span>文本生成覆盖率</span><strong>{percent(m.candidate_coverage)}</strong><small>产生描述 / 已尝试：{fraction(m.candidate_coverage)}</small></div>
        <div><span>首稿自动检查通过率</span><strong>{percent(m.quality.first_draft_pass)}</strong><small>{fraction(m.quality.first_draft_pass)}；检查未完成 {m.quality.first_draft_incomplete} 条</small></div>
        <div><span>修订恢复率</span><strong>{percent(m.quality.repair_success)}</strong><small>最终通过 / 首稿失败：{fraction(m.quality.repair_success)}</small></div>
        <div><span>改词对照检查通过率</span><strong>{percent(m.quality.counterfactuals.pass_per_complete)}</strong><small>{fraction(m.quality.counterfactuals.pass_per_complete)}；已完成检查 / 执行记录：{fraction(m.quality.counterfactuals.complete_per_workload)}</small></div>
        <div><span>模型调用</span><strong>{m.cost.reserved_model_calls} 次</strong><small>已找到调用记录 {m.cost.model_calls} 条；每条新通过描述 {number(m.cost.calls_per_new_pass.value)} 次</small></div>
        <div><span>模型调用耗时合计</span><strong>{formatDuration(m.cost.model_seconds)}</strong><small>{m.cost.model_seconds_complete ? '调用时长记录完整' : '仅包含已记录时长，尚不完整'}</small></div>
        <div><span>Token 用量</span><strong>{m.cost.tokens.toLocaleString()}</strong><small>{m.cost.tokens_complete ? '用量记录完整' : '仅包含服务返回的已知用量，尚不完整'}</small></div>
        <div><span>每条新通过描述 Token</span><strong>{number(m.cost.tokens_per_new_pass)}</strong><small>包括未通过尝试的消耗</small></div>
      </div>
      <h3>独立质量评估</h3>
      {!m.independent_audits.length && <p className="muted">尚未进行独立质量评估，准确率与人工接受率未测量。</p>}
      {m.independent_audits.map(a => <p key={a.cohort_id}>{a.purpose} · {a.reviewed} / {a.sampled} 已评：{a.status === 'sealed' && a.report ? <>唯一目标定位 {percent(a.report.blind_unique_target_accuracy)}（{fraction(a.report.blind_unique_target_accuracy)}，{interval(a.report.blind_unique_target_accuracy)}）；联合接受率 {percent(a.report.human_joint_acceptance)}（{fraction(a.report.human_joint_acceptance)}，{interval(a.report.human_joint_acceptance)}）</> : '尚未封存，暂无最终指标'}</p>)}
      <a href="#results?mode=audit">查看或进行独立评估 →</a>
    </details>
    <details><summary>类别覆盖与未通过原因</summary>
      <div className="table-wrap"><table className="result-table"><thead><tr><th>类别</th><th>候选</th><th>已尝试 / 已结束</th><th>通过</th><th>尝试通过率</th></tr></thead><tbody>{m.categories.map(c => <tr key={c.category}><td>{categoryLabel(c.category)}</td><td>{c.pool}</td><td>{c.attempted} / {c.settled}</td><td>{c.passed}</td><td>{percent(c.pass_per_settled)}</td></tr>)}</tbody></table></div>
      {m.failure_reasons.length ? <ul>{m.failure_reasons.map(f => <li key={f.reason}>{reasonLabel(f.reason)}：{f.count}</li>)}</ul> : <p className="muted">已结束的尝试中暂无未通过记录。</p>}
    </details>
    <p className="muted">分子与分母、计时口径、模型配置和方法哈希随指标导出。未测量值不会以 0 代替。</p>
  </section>
}
