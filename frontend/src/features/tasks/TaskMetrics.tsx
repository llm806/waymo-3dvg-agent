import { t, useLocale } from '../../shared/locale.ts'
import { categoryLabel, reasonLabel } from '../../shared/annotationDisplay'
import { formatDuration, formatTiming, hasIncompleteTiming } from '../../shared/duration'
import { HelpHint } from '../../components/Help'

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

const percent = (rate: Rate | null) => rate?.value == null ? t("未测量") : `${(rate.value * 100).toFixed(1)}%`
const fraction = (rate: Rate) => `${rate.numerator} / ${rate.denominator}`
const number = (value: number | null) => value == null ? t("未测量") : value.toLocaleString(undefined, { maximumFractionDigits: 2 })
const interval = (rate: Rate) => rate.ci95 ? `95% CI ${(rate.ci95[0] * 100).toFixed(1)}%–${(rate.ci95[1] * 100).toFixed(1)}%` : t("未测量置信区间")

export default function TaskMetrics({ metrics: m, jobId }: { metrics: TaskMetricsData; jobId: string }) {
  useLocale()

  if (m.schema_version !== 'task-metrics-v1') return <p className="notice">{t("重启后端后可查看新版任务指标；旧任务缺失的计时不会补算。")}</p>
  const latency = m.speed.item_active_seconds
  const metrics = [
    [t("任务候选利用率"), percent(m.coverage.pool_utilization), t("通过 / 本任务候选：") + fraction(m.coverage.pool_utilization)],
    [t("数据集标注覆盖率"), percent(m.coverage.dataset_utilization), t("通过描述 / 数据集物体总数：") + fraction(m.coverage.dataset_utilization)],
    [t("尝试通过率"), percent(m.coverage.pass_per_settled), t("通过 / 已结束尝试：") + fraction(m.coverage.pass_per_settled)],
    [t("任务墙钟用时"), formatDuration(m.timing.wall_seconds), t("从首次启动至结束，包含暂停")],
    [t("累计运行用时"), formatTiming(m.timing), t("包含准备、失败与核验，排除暂停")],
    [t("平均每条通过描述耗时"), formatDuration(m.speed.mean_active_seconds_per_new_pass), t("累计运行用时 / 新产出 ") + m.counts.new_passed + t(" 条")],
    [t("处理耗时 P50 / P95"), `${formatDuration(latency.p50)} / ${formatDuration(latency.p95)}`, latency.count + t(" 条完整计时，含失败尝试；均值 ") + formatDuration(latency.mean)],
    [t("通过描述吞吐量"), number(m.speed.new_passes_per_active_hour) + t(" 条/时"), t("按运行用时；按墙钟 ") + number(m.speed.new_passes_per_wall_hour) + t(" 条/时")],
  ]
  return <section className="panel task-metrics" aria-label={t("任务统计")}>
    <div className="section-heading"><h2><HelpHint label={t("任务统计")} detail={`${t("仅统计当前任务，同一物体只算一次。")} ${t("分子与分母、计时口径、模型配置和方法哈希随指标导出。未测量值不会以 0 代替。")}`} /></h2><a className="button-link" href={`/api/batches/${jobId}/report?download=true`}>{t("导出统计 JSON ↓")}</a></div>
    <p className="muted">{t("候选 ")}{m.counts.pool}{t(" · 已尝试 ")}{m.counts.attempted}{t(" · 已结束 ")}{m.counts.settled}{t(" · 当前可用 ")}{m.counts.passed}{t(" · 未尝试 ")}{m.counts.unattempted}{m.counts.reused ? t(" · 复用 {0}", m.counts.reused) : ''}</p>
    {m.counts.human_approved!=null&&<p className="muted">{t("自动核验通过 ")}{m.counts.automatic_passed??0}{t(" · 人工审批纳入 ")}{m.counts.human_approved}{t("；人工审批不计入自动核验通过率。")}</p>}
    <div className="metric-grid">{metrics.map(([label, value, detail])=><div key={label}><HelpHint label={label} detail={detail}/><strong>{value}</strong></div>)}</div>
    {hasIncompleteTiming(m.timing) && <p className="notice">{t("历史计时不完整，缺失部分未计入已记录用时；相关速度指标无法完整测量。")}</p>}
    <details><summary>{t("质量与模型用量")}</summary><p className="muted">{t("自动检查衡量内部一致性；首稿、修订与改词检查按执行记录统计，标注准确性由独立质量评估衡量。")}</p>
      <div className="metric-grid">
        <div><span>{t("文本生成覆盖率")}</span><strong>{percent(m.candidate_coverage)}</strong><small>{t("产生描述 / 已尝试：")}{fraction(m.candidate_coverage)}</small></div>
        <div><span>{t("首稿自动检查通过率")}</span><strong>{percent(m.quality.first_draft_pass)}</strong><small>{fraction(m.quality.first_draft_pass)}{t("；检查未完成 ")}{m.quality.first_draft_incomplete}{t(" 条")}</small></div>
        <div><span>{t("修订恢复率")}</span><strong>{percent(m.quality.repair_success)}</strong><small>{t("最终通过 / 首稿失败：")}{fraction(m.quality.repair_success)}</small></div>
        <div><span>{t("改词对照检查通过率")}</span><strong>{percent(m.quality.counterfactuals.pass_per_complete)}</strong><small>{fraction(m.quality.counterfactuals.pass_per_complete)}{t("；已完成检查 / 执行记录：")}{fraction(m.quality.counterfactuals.complete_per_workload)}</small></div>
        <div><span>{t("模型调用")}</span><strong>{m.cost.reserved_model_calls}{t(" 次")}</strong><small>{t("已找到调用记录 ")}{m.cost.model_calls}{t(" 条；每条新通过描述 ")}{number(m.cost.calls_per_new_pass.value)}{t(" 次")}</small></div>
        <div><span>{t("模型调用耗时合计")}</span><strong>{formatDuration(m.cost.model_seconds)}</strong><small>{m.cost.model_seconds_complete ? t("调用时长记录完整") : t("仅包含已记录时长，尚不完整")}</small></div>
        <div><span>{t("Token 用量")}</span><strong>{m.cost.tokens.toLocaleString()}</strong><small>{m.cost.tokens_complete ? t("用量记录完整") : t("仅包含服务返回的已知用量，尚不完整")}</small></div>
        <div><span>{t("每条新通过描述 Token")}</span><strong>{number(m.cost.tokens_per_new_pass)}</strong><small>{t("包括未通过尝试的消耗")}</small></div>
      </div>
      <h3>{t("独立质量评估")}</h3>
      {!m.independent_audits.length && <p className="muted">{t("尚未进行独立质量评估，准确率与人工接受率未测量。")}</p>}
      {m.independent_audits.map(a => <p key={a.cohort_id}>{a.purpose} · {a.reviewed} / {a.sampled}{t(" 已评：")}{a.status === 'sealed' && a.report ? <>{t("唯一目标定位 ")}{percent(a.report.blind_unique_target_accuracy)}（{fraction(a.report.blind_unique_target_accuracy)}，{interval(a.report.blind_unique_target_accuracy)}{t("）；联合接受率 ")}{percent(a.report.human_joint_acceptance)}（{fraction(a.report.human_joint_acceptance)}，{interval(a.report.human_joint_acceptance)}）</> : t("尚未封存，暂无最终指标")}</p>)}
      <a href="#research">{t("打开质量评估 →")}</a>
    </details>
    <details><summary>{t("类别覆盖与未通过原因")}</summary>
      <div className="table-wrap"><table className="metrics-table"><thead><tr><th scope="col">{t("类别")}</th><th scope="col">{t("候选")}</th><th scope="col">{t("已尝试 / 已结束")}</th><th scope="col">{t("通过")}</th><th scope="col">{t("尝试通过率")}</th></tr></thead><tbody>{m.categories.map(c => <tr key={c.category}><th scope="row">{categoryLabel(c.category)}</th><td>{c.pool}</td><td>{c.attempted} / {c.settled}</td><td>{c.passed}</td><td>{percent(c.pass_per_settled)}</td></tr>)}</tbody></table></div>
      {m.failure_reasons.length ? <ul>{m.failure_reasons.map(f => <li key={f.reason}>{reasonLabel(f.reason)}：{f.count}</li>)}</ul> : <p className="muted">{t("已结束的尝试中暂无未通过记录。")}</p>}
    </details>
  </section>
}
