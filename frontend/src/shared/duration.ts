export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '未测量'
  const total = Math.max(0, Math.floor(seconds))
  if (total < 60) return `${total} 秒`
  if (total < 3600) return `${Math.floor(total / 60)} 分 ${total % 60} 秒`
  return `${Math.floor(total / 3600)} 时 ${Math.floor(total % 3600 / 60)} 分 ${total % 60} 秒`
}

type Timing = { active_seconds: number | null; recorded_active_seconds?: number | null; history_incomplete?: boolean; running?: boolean; status?: string }

export function hasIncompleteTiming(timing?: Timing): boolean {
  return !!timing && (timing.history_incomplete === true || timing.status === 'incomplete'
    || timing.running === true && timing.active_seconds == null)
}

export function formatTiming(timing?: Timing, extra = 0): string {
  const seconds = timing?.active_seconds ?? timing?.recorded_active_seconds
  const label = formatDuration(seconds == null ? null : seconds + (timing?.running ? Math.max(0, extra) : 0))
  return seconds != null && hasIncompleteTiming(timing) ? `已记录 ${label}` : label
}
