export const pages = [
  { id: 'annotate', title: '自动标注' },
  { id: 'results', title: '结果与质量' },
] as const
export type WorkspaceView = typeof pages[number]['id']
export function currentView(): WorkspaceView {
  const id = window.location.hash.slice(1).split('?')[0]
  if (pages.some(p => p.id === id)) return id as WorkspaceView
  return id === 'workbench' || id.startsWith('records/') || id.startsWith('sample/') ? 'results' : 'annotate'
}
export function navigate(view: WorkspaceView) { window.location.hash = view }
export function openResult(packet: string, run?: string) {
  window.location.hash = 'results?' + new URLSearchParams({ packet, run:run??'data' })
}
export function taskLocation() {
  const params = new URLSearchParams(typeof window === 'undefined' ? '' : window.location?.hash?.split('?')[1])
  const ordinal = Number(params.get('ordinal'))
  return { jobId: params.get('job') ?? '', ordinal: Number.isSafeInteger(ordinal) && ordinal > 0 ? ordinal : null }
}
export function taskUrl(jobId: string, ordinal?: number) {
  return '#annotate?' + new URLSearchParams({ job:jobId, ...(ordinal ? { ordinal:String(ordinal) } : {}) })
}
export function openTask(jobId: string, ordinal?: number) { window.location.hash = taskUrl(jobId, ordinal) }
export function sampleUrl(jobId:string, ordinal:number) {
  return '#results?' + new URLSearchParams({job:jobId,ordinal:String(ordinal)})
}
