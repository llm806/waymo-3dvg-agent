import { t } from './locale.ts'
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return t("未测量")
  const total = Math.max(0, Math.floor(seconds))
  if (total < 60) return t("{0} 秒", total)
  if (total < 3600) return t("{0} 分 {1} 秒", Math.floor(total / 60), total % 60)
  return t("{0} 时 {1} 分 {2} 秒", Math.floor(total / 3600), Math.floor(total % 3600 / 60), total % 60)
}

type Timing = { active_seconds: number | null; recorded_active_seconds?: number | null; history_incomplete?: boolean; running?: boolean; status?: string }

export function hasIncompleteTiming(timing?: Timing): boolean {
  return !!timing && (timing.history_incomplete === true || timing.status === 'incomplete'
    || timing.running === true && timing.active_seconds == null)
}

export function formatTiming(timing?: Timing, extra = 0): string {
  const seconds = timing?.active_seconds ?? timing?.recorded_active_seconds
  const label = formatDuration(seconds == null ? null : seconds + (timing?.running ? Math.max(0, extra) : 0))
  return seconds != null && hasIncompleteTiming(timing) ? t("已记录 {0}", label) : label
}
