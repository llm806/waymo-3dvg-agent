// Tab-local browsing preferences only; never save approval or execution state.
export const resultViewKey = 'waymo.results.view.v1'
export type ResultView = {
  jobId:string; ordinal:number; filter:'all'|'passed'|'failed'|'pending'|'unknown'
  page:number; listScroll:number; pageScroll:number; review:boolean; queue:number[]|null
}
const integer = (value:unknown, fallback=0) => Number.isSafeInteger(value) && Number(value)>=0 ? Number(value) : fallback
export function initialResultView(hash:string):ResultView & {restored:boolean} {
  const empty:ResultView={jobId:'',ordinal:0,filter:'all',page:0,listScroll:0,pageScroll:0,review:false,queue:null}
  let saved:ResultView|null=null
  try {
    const raw=JSON.parse(window.sessionStorage.getItem(resultViewKey)??'null')
    if(raw&&typeof raw.jobId==='string'&&raw.jobId.length<=200){
      saved={jobId:raw.jobId,ordinal:integer(raw.ordinal),
        filter:['all','passed','failed','pending','unknown'].includes(raw.filter)?raw.filter:'all',
        page:integer(raw.page),listScroll:integer(raw.listScroll),pageScroll:integer(raw.pageScroll),
        review:raw.review===true,queue:Array.isArray(raw.queue)&&raw.queue.length<=1000&&raw.queue.every((id:unknown)=>integer(id)>0)?raw.queue:null}
    }
  } catch { /* Storage can be blocked or stale; ordinary browsing still works. */ }
  const params=new URLSearchParams(hash.split('?')[1])
  const job=params.get('job')??'',ordinal=integer(Number(params.get('ordinal')))
  // An explicit different sample link wins over the tab's previous selection.
  if(params.has('packet')||params.has('run')||(job&&saved?.jobId!==job)||(ordinal&&saved?.ordinal!==ordinal))
    return {...empty,jobId:job,ordinal,restored:false}
  return saved?{...saved,restored:true}:{...empty,jobId:job,ordinal,restored:false}
}
export function saveResultView(view:ResultView) {
  try { window.sessionStorage.setItem(resultViewKey,JSON.stringify(view)) } catch { /* Optional browser storage. */ }
}
