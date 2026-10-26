export type TaskContinuation = { source_job_id: string; inherited_passed: number }
type DatasetTask = { spec?: { dataset_split?: 'train' | 'validation' } }
type NamedTask = DatasetTask & { created_at: number; counts: Record<string, number>; selection_progress?: { requested: number | null }; continuation?: TaskContinuation | null }

export function exportTaskSelection(tasks: DatasetTask[]): { split: 'train' | 'validation'; error: string } {
  const splits = new Set(tasks.map(task => task.spec?.dataset_split ?? 'train'))
  return { split: splits.has('validation') ? 'validation' : 'train',
    error: splits.size > 1 ? '训练集和验证集任务不能一起导出，请分别选择并分开导出。' : '' }
}

export function parseAnnotationCount(text: string): number | null {
  const value = text.trim()
  if (!/^\d+$/.test(value)) return null
  const count = Number(value)
  return Number.isSafeInteger(count) && count > 0 ? count : null
}

const createdTime = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
})

export function taskDetails(task: NamedTask): { createdAt: string; scope: string } {
  const prefix = task.spec?.dataset_split === 'validation' ? '验证集 · ' : ''
  const parts = Object.fromEntries(createdTime.formatToParts(new Date(task.created_at * 1000)).map(p => [p.type, p.value]))
  const createdAt = `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`
  if (task.selection_progress?.requested) {
    return { createdAt, scope: `${prefix}目标 ${task.selection_progress.requested} 条描述` }
  }
  const count = Object.values(task.counts).reduce((sum, n) => sum + n, 0)
  return { createdAt, scope: `${prefix}${count} 个物体` }
}

export function taskName(task: NamedTask): string {
  const { createdAt, scope } = taskDetails(task)
  return `${createdAt} · ${scope}`
}
