import { serviceText, localizeLabels, t, useLocale } from '../../shared/locale.ts'
import type { InterfaceMessage } from '../../shared/locale.ts'
import { useEffect, useMemo, useRef, useState } from 'react'
import TaskScene from './TaskScene'
import { HelpHint } from '../../components/Help'
import { taskName } from '../../shared/taskName'
import { cameraViewLabel } from '../../shared/cameraViews'
import type { Confirmation, TaskSceneData } from './TaskScene'
import { objectName, categoryLabel, errorMessage, readJson, readPollingJson, manualReviewLabels, automaticCheckLabels, automaticCheckState, sampleFilterLabels, matchesSampleFilter } from '../../shared/annotationDisplay'
import SampleReview from './SampleReview'
import type { ManualReview } from './SampleReview'
import { sampleUrl } from '../../navigation'
import ReworkProgress from './ReworkProgress'
import { initialResultView, saveResultView } from './resultViewState'
import type { ResultView } from './resultViewState'
import type { ReworkState } from './ReworkProgress'

type Job = { job_id: string; created_at: number; status: string; counts: Record<string,number> }
type Sample = { ordinal: number; object_id?: string | null; category: string; expression: string; run_id: string; packet_id: string; status: string; automatic_passed?: boolean; manual_approved?:boolean; usable?:boolean; viewing_previous?:boolean; reviewed: boolean; source_view?: string | null; legacy_rules?: boolean; revision: string | null; can_review: boolean; can_retry?: boolean; retry_failed_run_id?: string; manual_review: ManualReview | null }
type CheckFilter = ResultView['filter']
type Samples = { task_goal?: number | null; job_id: string; status: string; samples: Sample[]; total_objects: number; created_at: number; review_summary: { unreviewed: number; accepted: number; rejected: number; discarded: number; usable: number; precheck_failed: number } }
const jobLabels: Record<string,string> = localizeLabels({ ready:'待启动', queued:'排队中', running:'执行中', paused:'已暂停', completed:'已完成', stopped:'已停止', blocked:'执行受阻' })
const pageSize = 30
const emptySamples: Sample[] = []
function fromUrl() { const p=new URLSearchParams(window.location.hash.split('?')[1]);return {job:p.get('job')??'',packet:p.get('packet')??'',run:p.get('run')??'',ordinal:Number(p.get('ordinal'))||0} }

export default function TaskSamples() {
  useLocale()

  const restoredView=initialResultView(window.location.hash)
  const [review,setReview]=useState(restoredView.review)
  const [jobs,setJobs]=useState<Job[]>([]),[jobId,setJobId]=useState(restoredView.jobId),[loaded,setLoaded]=useState(false)
  const [samples,setSamples]=useState<Samples|null>(null),[requestedOrdinal,setOrdinal]=useState(restoredView.ordinal)
  const [scene,setScene]=useState<TaskSceneData|null>(null),[selected,setSelected]=useState('')
  const [confirmation,setConfirmation]=useState<Confirmation|null>(null),[busy,setBusy]=useState(false)
  const [error,setError]=useState(''),[sceneError,setSceneError]=useState('')
  const [refresh,setRefresh]=useState(0)
  const [queue,setQueue]=useState<number[]|null>(restoredView.queue),[drawCount,setDrawCount]=useState('10')
  const [unreviewed,setUnreviewed]=useState(true),[jumpIndex,setJumpIndex]=useState('1')
  const [selectionError,setSelectionError]=useState<InterfaceMessage>('')
  const [sceneRequest,setSceneRequest]=useState(0)
  const [listPage,setListPage]=useState(restoredView.page),[loading,setLoading]=useState(false)
  const [jobsError,setJobsError]=useState('')
  const [checkFilter,setCheckFilter]=useState<CheckFilter>(restoredView.filter)
  const [reworkTarget,setReworkTarget]=useState('')
  const [sampleActionError,setSampleActionError]=useState('')
  const [retryReceipt,setRetryReceipt]=useState<{job_id:string;ordinal:number;message?:string}|null>(null)
  const sampleCache=useRef(new Map<string,Samples>())
  const sceneCache=useRef(new Map<string,TaskSceneData>())
  const submittingReview=useRef(false)
  const linkResolved=useRef(false)
  const [rework,setRework]=useState<ReworkState|null>(null),[reworkError,setReworkError]=useState('')
  const refreshedRework=useRef('')
  const reviewGeneration=useRef(0)
  const sceneArea=useRef<HTMLDivElement|null>(null)
  const approvalArea=useRef<HTMLElement|null>(null)
  const [approving,setApproving]=useState(false)
  const [actionFeedback,setActionFeedback]=useState<{jobId:string;ordinal:number;revision:string|null;error:boolean;message:InterfaceMessage}|null>(null)
  const sampleArea=useRef<HTMLDivElement|null>(null)
  const scrollToSample=useRef(Boolean(fromUrl().ordinal)&&!restoredView.restored)
  const [sampleIdInput,setSampleIdInput]=useState('')
  const [sampleSearchError,setSampleSearchError]=useState<InterfaceMessage>('')
  const pageTarget=useRef<{job:string;ordinal:number}|null>(restoredView.restored?null:fromUrl())
  const listArea=useRef<HTMLOListElement|null>(null)
  const restoreScroll=useRef(restoredView.restored?{...restoredView,listDone:false}:null)
  const savedView=useRef<ResultView|null>(null)
  const [selectionOpen,setSelectionOpen]=useState(!!restoredView.queue)
  const urlTarget=fromUrl()
  const rows=samples?.job_id===jobId?samples.samples:emptySamples
  const linkedOrdinal=urlTarget.job===jobId&&urlTarget.ordinal===requestedOrdinal&&!rows.some(s=>s.ordinal===requestedOrdinal)?requestedOrdinal:0
  const listedRows=useMemo(()=>{if(!queue)return rows;const byOrdinal=new Map(rows.map(s=>[s.ordinal,s]));return queue.map(n=>byOrdinal.get(n)).filter((s):s is Sample=>!!s)},[rows,queue])
  const visibleRows=useMemo(()=>listedRows.filter(s=>matchesSampleFilter(s,checkFilter)),[listedRows,checkFilter])
  const navigationJob=useRef(jobId)
  const navigationRows=useRef(visibleRows)
  useEffect(()=>{navigationJob.current=jobId;navigationRows.current=visibleRows},[jobId,visibleRows])
  // Selection belongs to the detail pane. A retry/approval can move the selected
  // sample out of the list filter without replacing either selection or filter.
  const ordinal=linkedOrdinal|| (rows.some(s=>s.ordinal===requestedOrdinal)?requestedOrdinal:visibleRows[0]?.ordinal??0)
  function focusSample(id:string,n:number){
    restoreScroll.current=null
    scrollToSample.current=true
    pageTarget.current={job:id,ordinal:n}
    if(id!==navigationJob.current){setQueue(null);setCheckFilter('all');setListPage(0)}
    const position=(id===navigationJob.current?navigationRows.current:sampleCache.current.get(id)?.samples??[]).findIndex(s=>s.ordinal===n)
    if(position>=0){setListPage(Math.floor(position/pageSize));pageTarget.current=null}
    setJobId(id);setOrdinal(n);setReview(false)
    setScene(null);setSelected('');setConfirmation(null);setSceneError('');setSelectionError('');setSampleActionError('')
  }
  useEffect(()=>{
    const follow=()=>{
      const location=fromUrl()
      if(!window.location.hash.startsWith('#results?')||!location.job||!location.ordinal)return
      focusSample(location.job,location.ordinal)
      setRefresh(v=>v+1)
    }
    window.addEventListener('hashchange',follow)
    return()=>window.removeEventListener('hashchange',follow)
  },[])
  useEffect(()=>{
    const c=new AbortController();let timer:ReturnType<typeof setTimeout>
    function schedule(){timer=setTimeout(()=>{if(document.hidden)schedule();else void poll()},10000)}
    async function poll(){try{
      const value=await readPollingJson('/api/batches',c.signal)
      if(c.signal.aborted)return
      setJobs(value.jobs);setLoaded(true);setJobsError('')
      {
        const url=fromUrl();let desired=value.jobs.find((j:Job)=>j.job_id===url.job)?.job_id
        if(!linkResolved.current&&!desired&&(url.packet||url.run)) {
          for(const j of value.jobs as Job[]){const list=sampleCache.current.get(j.job_id)??await fetch(`/api/batches/${j.job_id}/samples`,{signal:c.signal}).then(readJson) as Samples;sampleCache.current.set(j.job_id,list);if(list.samples.some(s=>s.packet_id===url.packet||s.run_id===url.run)){desired=j.job_id;break}}
        }
        if(!c.signal.aborted){linkResolved.current=true;setJobId(current=>value.jobs.some((j:Job)=>j.job_id===current)?current:desired??value.jobs.find((j:Job)=>(j.counts.calibration_hold??0)+(j.counts.auto_pass??0)+(j.counts.reused??0)+(j.counts.manual_approved??0)>0)?.job_id??value.jobs[0]?.job_id??'')}
      }
    }catch(e){if(!c.signal.aborted){setJobsError(errorMessage(e));setLoaded(true)}}finally{if(!c.signal.aborted)schedule()}}
    void poll();return()=>{c.abort();clearTimeout(timer)}
  },[refresh])
  useEffect(()=>{
    if(!jobId)return
    const c=new AbortController();let timer:ReturnType<typeof setTimeout>
    let intervalMs=30000
    function schedule(){timer=setTimeout(()=>{if(document.hidden)schedule();else void poll()},intervalMs)}
    async function poll(){try{
      setLoading(true)
      const generation=reviewGeneration.current
      const value=await readPollingJson(`/api/batches/${jobId}/samples`,c.signal,30000) as Samples
      if(c.signal.aborted)return
      // Relinking a retry removes its old result. Keep its submitting form
      // mounted until the request returns so its feedback cannot disappear.
      if(submittingReview.current||generation!==reviewGeneration.current)return
      sampleCache.current.delete(jobId);sampleCache.current.set(jobId,value)
      if(sampleCache.current.size>4)sampleCache.current.delete(sampleCache.current.keys().next().value!)
      setSamples(current=>JSON.stringify(current)===JSON.stringify(value)?current:value);setError('')
      intervalMs=['running','ready','queued'].includes(value.status)?5000:30000
      setOrdinal(current=>{const link=fromUrl();if(link.job===jobId&&link.ordinal===current)return current;if(value.samples.some(s=>s.ordinal===current))return current;const url=fromUrl();return value.samples.find(s=>s.run_id===url.run||s.packet_id===url.packet)?.ordinal??value.samples[0]?.ordinal??0})
    }catch(e){if(!c.signal.aborted)setError(errorMessage(e))}finally{if(!c.signal.aborted){setLoading(false);schedule()}}}
    void poll();return()=>{c.abort();clearTimeout(timer)}
  },[jobId,refresh])
  useEffect(()=>{
    const target=pageTarget.current
    if(!target||target.job!==jobId||samples?.job_id!==jobId)return
    const position=visibleRows.findIndex(s=>s.ordinal===target.ordinal)
    if(position>=0)setListPage(Math.floor(position/pageSize))
    if(rows.some(s=>s.ordinal===target.ordinal))pageTarget.current=null
  },[jobId,samples,rows,visibleRows])
  const sample=rows.find(s=>s.ordinal===ordinal)
  const sampleVersion=sample?JSON.stringify([sample.expression,sample.packet_id,sample.run_id,sample.status,sample.revision]):''
  useEffect(()=>{
    if(!jobId||!ordinal)return
    const c=new AbortController();let timer:ReturnType<typeof setTimeout>
    let interval=5000, fetching=false
    async function poll(){
      if(fetching||c.signal.aborted)return
      clearTimeout(timer)
      if(document.hidden){timer=setTimeout(()=>void poll(),interval);return}
      fetching=true
      try{
        const generation=reviewGeneration.current
        const value=await readPollingJson(`/api/batches/${jobId}/samples/${ordinal}/rework`,c.signal) as ReworkState
        if(c.signal.aborted)return
        if(submittingReview.current||generation!==reviewGeneration.current)return
        if(value.job_id!==jobId||value.ordinal!==ordinal)throw new Error(t("返工记录与所选样本不一致。"))
        setRework(value);setReworkError('')
        const latest=value.item?.attempts?.at(-1)?.expression
        const version=JSON.stringify([jobId,ordinal,value.item?.run_id,latest,value.item?.status])
        if(latest&&(value.item?.run_id!==sample?.run_id||latest!==sample?.expression)&&refreshedRework.current!==version){
          refreshedRework.current=version;setRefresh(v=>v+1)
        }
        interval=value.item&&['pending','preparing','running'].includes(value.item.status)?3000:30000
      }catch(e){if(!c.signal.aborted)setReworkError(errorMessage(e))}
      finally{fetching=false;if(!c.signal.aborted)timer=setTimeout(()=>void poll(),interval)}
    }
    const visible=()=>{if(!document.hidden){clearTimeout(timer);void poll()}}
    document.addEventListener('visibilitychange',visible)
    void poll()
    return()=>{c.abort();clearTimeout(timer);document.removeEventListener('visibilitychange',visible)}
  },[jobId,ordinal,refresh,sample?.run_id,sample?.expression])
  useEffect(()=>{
    if(!jobId||!ordinal||!sampleVersion)return
    const c=new AbortController()
    const cacheKey=JSON.stringify([jobId,ordinal,review,sampleVersion])
    const cached=sceneCache.current.get(cacheKey)
    function display(v:TaskSceneData){setScene(v);setSceneError('');setSelected(review?v.confirmation?.selected_candidate_id??'':v.selected_candidate_id??'');setConfirmation(review?v.confirmation:null)}
    if(cached){display(cached);return}
    void fetch(`/api/batches/${jobId}/samples/${ordinal}/scene?review=${review}`,{signal:c.signal}).then(readJson).then((v:TaskSceneData)=>{
      if(c.signal.aborted)return
      if(v.job_id!==jobId||v.ordinal!==ordinal)throw new Error(t("返回的场景与所选任务样本不一致。"))
      sceneCache.current.set(cacheKey,v)
      if(sceneCache.current.size>4)sceneCache.current.delete(sceneCache.current.keys().next().value!)
      display(v)
    }).catch(e=>{if(!c.signal.aborted)setSceneError(errorMessage(e))})
    return()=>c.abort()
  },[jobId,ordinal,review,sampleVersion,sceneRequest])
  function openSample(id:string,n:number){
    focusSample(id,n)
    window.location.hash=sampleUrl(id,n)
    setRefresh(v=>v+1)
  }
  useEffect(()=>{
    if(scrollToSample.current&&sampleArea.current){
      scrollToSample.current=false
      sampleArea.current.scrollIntoView({block:'start',behavior:'smooth'})
    }
  },[jobId,ordinal,samples,refresh])
  function toggleReview(){setReview(v=>!v);setScene(null);setSelected('');setConfirmation(null);setSceneError('')}
  function chooseSample(n:number){pageTarget.current=null;setSampleActionError('');setOrdinal(n);if(n===ordinal)return;setScene(null);setSelected('');setConfirmation(null);setSceneError('')}
  function chooseJob(id:string){restoreScroll.current=null;pageTarget.current=null;setSampleActionError('');const cached=sampleCache.current.get(id);setJobId(id);setSamples(cached??null);setOrdinal(cached?.samples[0]?.ordinal??0);setScene(null);setSelected('');setConfirmation(null);setSceneError('');setQueue(null);setCheckFilter('all');setListPage(0);setReview(false);setSelectionError('');setSampleIdInput('');setSampleSearchError('');setError('')}
  async function drawSamples(){
    const count=Number(drawCount)
    if(!Number.isSafeInteger(count)||count<1||count>1000){setSelectionError("抽样数量请输入 1～1000 的整数。");return}
    setBusy(true);setSelectionError('')
    try{
      const value=await readJson(await fetch(`/api/batches/${jobId}/samples/random?count=${count}&unreviewed=${unreviewed}`)) as {ordinals:number[]}
      if(!value.ordinals.length){setSelectionError("没有符合条件的样本；可以取消“仅未审核”或切换任务。");return}
      setQueue(value.ordinals);setCheckFilter('all');setListPage(0);setReview(true);chooseSample(value.ordinals[0]);setScene(null);setSelected('');setConfirmation(null);setSceneRequest(v=>v+1)
    }catch(e){setSelectionError(errorMessage(e))}finally{setBusy(false)}
  }
  function jumpToSampleId(){
    const query=sampleIdInput.trim()
    const number=/^#?\s*(\d+)$/.exec(query)
    const ordinal=number?Number(number[1]):0
    const validNumber=Number.isSafeInteger(ordinal)&&ordinal>0
    let found=validNumber&&(rows.some(s=>s.ordinal===ordinal)||(retryReceipt?.job_id===jobId&&retryReceipt.ordinal===ordinal))?ordinal:undefined
    if(found===undefined&&query&&!query.startsWith('#')){
      const matching=rows.filter(s=>s.object_id===query)
      if(matching.length>1){
        setSampleSearchError("该物体 ID 对应多条样本，请用样本编号查找。")
        return
      }
      found=matching[0]?.ordinal
    }
    if(found===undefined){
      setSampleSearchError(!query||query.startsWith('#')&&!validNumber
        ?"请输入样本编号或 Waymo 物体 ID。"
        :validNumber?["当前任务没有样本 #{0}，请核对编号或切换任务。",ordinal]
        :"当前任务没有匹配的样本。")
      return
    }
    setSampleSearchError('');setReworkTarget('');openSample(jobId,found)
  }
  function jumpToIndex(){
    const index=Number(jumpIndex),rows=visibleRows
    if(!Number.isSafeInteger(index)||index<1||index>rows.length){setSelectionError(["请输入 1～{0} 的列表索引。", rows.length]);return}
    setSelectionError('');setListPage(Math.floor((index-1)/pageSize));setReview(true);chooseSample(rows[index-1].ordinal);setScene(null);setSelected('');setConfirmation(null);setSceneRequest(v=>v+1)
  }
  function reviewed(value:ManualReview){
    const reviewedJob=value.job_id??jobId, reviewedOrdinal=value.ordinal??ordinal
    sceneCache.current.clear();sampleCache.current.delete(jobId)
    setSamples(current=>{
      if(!current||current.job_id!==reviewedJob)return current
      const updated=current.samples.map(s=>s.ordinal===reviewedOrdinal&&(!value.run_id||value.run_id===s.run_id)&&s.revision===value.revision?{...s,manual_review:value,
        manual_approved:value.verdict==='accepted'&&value.approval_policy==='localized-human-approval-v1',
        usable:(s.automatic_passed||value.verdict==='accepted'&&value.approval_policy==='localized-human-approval-v1')&&!['rejected','discarded'].includes(value.verdict)}:s)
      const review_summary={...current.review_summary}
      for(const key of ['unreviewed','accepted','rejected','discarded'] as const)
        review_summary[key]=updated.filter(s=>(s.manual_review?.verdict??'unreviewed')===key).length
      review_summary.usable=updated.filter(s=>(s.automatic_passed||s.manual_approved)&&!['rejected','discarded'].includes(s.manual_review?.verdict??'')).length
      return {...current,samples:updated,review_summary}
    })
    setRework(current=>current?.job_id===reviewedJob&&current.ordinal===reviewedOrdinal&&current.item&&(!value.run_id||current.item.run_id===value.run_id)
      ?{...current,item:{...current.item,manual_approved:value.verdict==='accepted'&&value.approval_policy==='localized-human-approval-v1'}}:current)
    setActionFeedback({jobId:reviewedJob,ordinal:reviewedOrdinal,revision:value.revision,error:false,
      message:value.verdict==='accepted'?["样本 #{0} 已{1}，已纳入当前可用集。", reviewedOrdinal, value.approval_policy?"人工审批通过":"审核通过"]:value.verdict==='discarded'?["样本 #{0} 已废弃，已从当前可用集排除。", reviewedOrdinal]:["样本 #{0} 的审核意见已保存。", reviewedOrdinal]})
    setRefresh(v=>v+1)
  }
  function reviewBusy(value:boolean){reviewGeneration.current+=1;submittingReview.current=value;setBusy(value)}
  function beginApproval(){
    if(busy||!sample?.can_review)return
    setReworkTarget('')
    if(!review)toggleReview()
    setActionFeedback({jobId,ordinal,revision:sample.revision,error:false,
      message:"已进入人工审批：先在下方图像或点云中选择物体，再确认定位，最后提交审批。当前尚未保存审批结论。"})
    approvalArea.current?.scrollIntoView({block:'center',behavior:'smooth'})
  }
  async function approveSample(){
    if(submittingReview.current||busy||!sample?.can_review||!sample.revision||!feedback?.correct)return
    const target={jobId,ordinal:sample.ordinal,revision:sample.revision,runId:sample.run_id}
    reviewBusy(true);setApproving(true)
    setActionFeedback({...target,error:false,message:"正在保存人工审批，请稍候…"})
    try{
      const value=await readJson(await fetch(`/api/batches/${target.jobId}/samples/${target.ordinal}/review`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({revision:target.revision,verdict:'accepted',notes:sample.manual_review?.notes??'',
          ...(!sample.automatic_passed?{approve_failed:true}:{})}),
      })) as ManualReview
      if(value.verdict!=='accepted'||value.revision!==target.revision||(value.run_id&&value.run_id!==target.runId))throw new Error(t("审批返回的版本与当前样本不一致，请刷新核对。"))
      reviewed({...value,job_id:target.jobId,ordinal:target.ordinal})
    }catch(e){setActionFeedback({...target,error:true,message:["审批未确认成功：{0}", errorMessage(e)]})}
    finally{setApproving(false);reviewBusy(false);setRefresh(v=>v+1)}
  }
  function retryRequested(value:{job_id:string;ordinal:number;message?:string}) {
    setRetryReceipt(value)
    setReworkTarget('');openSample(value.job_id,value.ordinal)
  }
  function openRetry(id:string){openSample(id,ordinal)}
  async function discardSample() {
    if (!sample || busy || !sample.can_review || sample.manual_review?.verdict === 'discarded') return
    reviewBusy(true); setSampleActionError('')
    try {
      const value = await readJson(await fetch(`/api/batches/${jobId}/samples/${sample.ordinal}`, {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: sample.run_id, packet_id: sample.packet_id, expression: sample.expression }),
      })) as ManualReview & { jobs?: Job[] }
      reviewed(value)
      if (value.jobs) setJobs(value.jobs)
    } catch (e) { setSampleActionError(errorMessage(e)) } finally { reviewBusy(false) }
  }
  async function confirm(){
    if(!scene||!selected||busy||submittingReview.current)return
    reviewBusy(true);setSceneError('')
    try{
      const value=await readJson(await fetch(`/api/batches/${jobId}/samples/${ordinal}/confirm`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:scene.revision,selected_candidate_id:selected})})) as Confirmation
      sceneCache.current.clear();setConfirmation(value)
      setActionFeedback({jobId,ordinal,revision:scene.revision,error:!value.correct,
        message:value.correct?"定位已确认正确。请点击“审批通过并纳入可用集”保存最终审批。":"定位未通过，请在图像或点云中重新选择物体，审批尚未保存。"})
    }catch(e){setActionFeedback({jobId,ordinal,revision:scene.revision,error:true,message:["定位确认失败：{0}", errorMessage(e)]})}
    finally{reviewBusy(false);setRefresh(v=>v+1)}
  }
  const currentScene=scene?.job_id===jobId&&scene.ordinal===ordinal&&scene.expression===sample?.expression&&scene.run_id===sample?.run_id&&scene.revision===sample?.revision&&scene.review===review?scene:null
  useEffect(()=>{
    if(!loaded||samples?.job_id!==jobId||!jobId)return
    const restoring=restoreScroll.current
    const value:ResultView={jobId,ordinal,filter:checkFilter,page:Math.max(0,Math.min(listPage,Math.ceil(visibleRows.length/pageSize)-1)),review,queue,
      listScroll:Math.round(restoring&&!restoring.listDone?restoring.listScroll:listArea.current?.scrollTop??0),
      pageScroll:Math.round(restoring?.pageScroll??window.scrollY??0)}
    savedView.current=value;saveResultView(value)
    // replaceState updates refresh/deep-link identity without firing hashchange
    // and resetting the filter or the random-review queue.
    window.history?.replaceState(null,'',ordinal?sampleUrl(jobId,ordinal):'#results?'+new URLSearchParams({job:jobId}))
  },[jobId,ordinal,checkFilter,listPage,review,queue,loaded,samples,visibleRows.length])
  useEffect(()=>{
    const restore=restoreScroll.current
    if(!restore||restore.jobId!==jobId||samples?.job_id!==jobId)return
    if(!restore.listDone&&listArea.current){restore.listDone=true;listArea.current.scrollTop=restore.listScroll}
    // Wait for the actual scene layout before restoring document scroll.
    if(currentScene||sceneError||(!sample&&rows.length)){
      window.scrollTo?.({top:restore.pageScroll,behavior:'instant'})
      restoreScroll.current=null
    }
  },[jobId,samples,currentScene,sceneError,sample,rows.length])
  useEffect(()=>{
    const scroll=()=>{if(savedView.current&&!restoreScroll.current)savedView.current.pageScroll=Math.round(window.scrollY??0)}
    const flush=()=>{scroll();if(savedView.current)saveResultView(savedView.current)}
    window.addEventListener('scroll',scroll,{passive:true});window.addEventListener('pagehide',flush)
    return()=>{flush();window.removeEventListener('scroll',scroll);window.removeEventListener('pagehide',flush)}
  },[])
  const feedback=currentScene&&confirmation?.selected_candidate_id===selected?confirmation:null
  const index=visibleRows.findIndex(s=>s.ordinal===ordinal)
  const positionLabel=queue?t("抽样队列"):checkFilter==='all'?t("列表索引"):t("筛选结果")
  const page=Math.max(0,Math.min(listPage,Math.ceil(visibleRows.length/pageSize)-1))
  const pageRows=visibleRows.slice(page*pageSize,(page+1)*pageSize)
  function stepSample(offset:number){const next=index+offset;if(next<0||next>=visibleRows.length)return;setListPage(Math.floor(next/pageSize));chooseSample(visibleRows[next].ordinal)}
  function reloadResults(){sceneCache.current.clear();setSceneRequest(v=>v+1);setRefresh(v=>v+1)}
  function filterChecks(value:CheckFilter){pageTarget.current=null;if(value!=='all'&&!listedRows.some(s=>s.ordinal===requestedOrdinal&&matchesSampleFilter(s,value)))setOrdinal(0);setCheckFilter(value);setListPage(0);setSceneError('');setSelectionError('')}
  const checkState=sample?automaticCheckState(sample):null
  const checkLabel=sample?.manual_approved?t("人工审批通过 · 当前可用"):checkState?automaticCheckLabels[checkState]:''
  const checkNote=sample?.manual_approved?'':checkState==='failed'?t("确认定位后可人工审批，也可反馈修改或废弃。"):
    checkState==='pending'?t("核验完成前不可导出。"):checkState==='unknown'?t("状态暂不可读，请刷新。"):
    checkState==='passed'&&sample?.manual_review?.verdict==='rejected'?t("已排除，不纳入导出。"):''
  const feedbackAvailable=checkState==='passed'||checkState==='failed'
  const reworkOpen=feedbackAvailable&&reworkTarget===`${jobId}:${ordinal}`
  const summary=samples?.job_id===jobId?samples.review_summary:null
  const currentRework=rework?.job_id===jobId&&rework.ordinal===ordinal?rework:null
  const alignPanels=!!sample&&!review&&!reworkOpen&&!currentRework?.item
  const currentAction=actionFeedback?.jobId===jobId&&actionFeedback.ordinal===ordinal&&actionFeedback.revision===sample?.revision?actionFeedback:null
  const approved=sample?.manual_review?.verdict==='accepted'&&(sample.automatic_passed||sample.manual_approved)
  const approvalStep=approved?3:feedback?.correct?2:selected?1:0
  useEffect(()=>{
    if(review)approvalArea.current?.scrollIntoView({block:'center',behavior:'smooth'})
  },[review])
  return <div className="task-results">
    {(error||jobsError)&&<div className="error results-error" role="alert"><span>{t("读取失败：")}{serviceText(error||jobsError)}</span><button disabled={busy||loading} onClick={reloadResults}>{t("重新读取")}</button></div>}
    {!loaded?<div className="panel results-loading" role="status"><span className="loading-dot"/>{t("正在读取标注任务…")}</div>:!jobs.length?<section className="panel results-empty-state"><h2>{jobsError?t("任务暂时无法读取"):t("还没有标注任务")}</h2>{!jobsError&&<a className="button-link" href="#annotate">{t("新建标注任务")}</a>}</section>:<>
      <section className="panel results-overview" aria-label={t("结果任务列表")}>
        <div className="results-taskbar">
          <label className="results-job-select"><span>{t("标注任务")}</span><select aria-label={t("选择标注任务")} value={jobId} disabled={busy} onChange={e=>chooseJob(e.target.value)}>{jobs.map(j=><option key={j.job_id} value={j.job_id}>{taskName(j)} · {jobLabels[j.status]??j.status}</option>)}</select></label>
          <div className="results-task-actions">
            <button className="quiet-button" disabled={busy||loading} onClick={reloadResults}>{loading?t("同步中…"):t("刷新结果")}</button>
            {!!rows.length&&<button className="quiet-button" aria-expanded={selectionOpen} aria-controls="review-selection" onClick={()=>setSelectionOpen(open=>!open)}>{t("抽样与快速查找")}</button>}
          </div>
        </div>
        <dl className="review-metrics">
          {([['usable',t("可用样本")],['unreviewed',t("待审核")],['accepted',t("人工通过")],['rejected',t("不通过")],['discarded',t("已废弃")]] as const).map(([key,label])=><div key={key} className={`review-metric ${key}`}><dt>{label}</dt><dd>{summary?.[key]??'—'}</dd></div>)}
          <div className="review-metric">
            <dt>{t("预检查未通过")}</dt>
            <dd>{summary?.precheck_failed??'—'}</dd>
          </div>
        </dl>
        {!!rows.length&&<div id="review-selection" className="review-selection" role="region" aria-label={t("审核样本选择")} hidden={!selectionOpen}>
          <div className="review-toolbar">
            <form onSubmit={e=>{e.preventDefault();void drawSamples()}}><label htmlFor="review-count">{t("抽样数量")}</label><input id="review-count" type="number" min="1" max="1000" value={drawCount} disabled={busy} onChange={e=>setDrawCount(e.target.value)}/><label className="review-checkbox"><input type="checkbox" checked={unreviewed} disabled={busy} onChange={e=>setUnreviewed(e.target.checked)}/>{t("仅未审核")}</label><button className="primary" disabled={busy}>{t("随机抽样")}</button></form>
            <form onSubmit={e=>{e.preventDefault();jumpToIndex()}}><label htmlFor="review-index">{t("列表位置 ")}<span className="muted">1–{visibleRows.length}</span></label><input id="review-index" type="number" min="1" max={visibleRows.length} value={jumpIndex} disabled={busy} onChange={e=>setJumpIndex(e.target.value)}/><button disabled={busy}>{t("跳转")}</button></form>
            {queue&&<button className="text-button" disabled={busy} onClick={()=>{setQueue(null);setListPage(Math.floor(rows.findIndex(s=>s.ordinal===ordinal)/pageSize))}}>{t("退出抽样")}</button>}
          </div>
          {selectionError&&<p className="error" role="alert">{serviceText(selectionError)}</p>}
        </div>}
      </section>
      {retryReceipt?.job_id===jobId&&<div className="notice" role="status"><span>{t("样本 #")}{retryReceipt.ordinal}：{(retryReceipt.message?serviceText(retryReceipt.message):undefined)??t("返工已受理，新描述仍须通过自动核验。")}</span> <a href={sampleUrl(jobId,retryReceipt.ordinal)} onClick={event=>{if(event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();openSample(jobId,retryReceipt.ordinal)}}>{t("查看该样本")}</a><button className="text-button" onClick={()=>setRetryReceipt(null)}>{t("关闭提示")}</button></div>}
      {!samples||samples.job_id!==jobId?<section className="panel results-loading" role="status">{!error&&<span className="loading-dot"/>}{error?t("样本读取未完成"):t("正在读取该任务的样本…")}</section>:!samples.samples.length&&!linkedOrdinal?<section className="panel results-empty-state"><h2>{t("该任务还没有生成文本样本")}</h2></section>:<div className={`sample-browser-grid${alignPanels?' sample-browser-aligned':''}`}>
        <aside className="sample-list panel" aria-label={t("当前任务样本")}>
          <div className="sample-list-header">
          <h3 className="sample-list-heading">{queue?t("本轮抽样"):t("标注样本")}</h3>
          <form className="sample-list-search" role="search" onSubmit={event=>{event.preventDefault();jumpToSampleId()}}>
            <input id="review-sample-id" type="search" aria-label={t("样本编号或物体 ID")} aria-invalid={!!sampleSearchError}
              aria-describedby={sampleSearchError?'review-sample-search-error':undefined} placeholder={t("编号 / 物体 ID")}
              value={sampleIdInput} disabled={busy} onChange={event=>{setSampleIdInput(event.target.value);setSampleSearchError('')}} />
            <button disabled={busy}>{t("查找")}</button>
          </form>
          {sampleSearchError&&<p id="review-sample-search-error" className="error" role="alert">{serviceText(sampleSearchError)}</p>}
          <select className="sample-check-filter" aria-label={t("样本状态筛选")} value={checkFilter} disabled={busy} onChange={e=>filterChecks(e.target.value as CheckFilter)}>{(Object.keys(sampleFilterLabels) as CheckFilter[]).map(key=><option key={key} value={key}>{sampleFilterLabels[key]}（{listedRows.filter(s=>matchesSampleFilter(s,key)).length}）</option>)}</select>
          </div>
          <ol ref={listArea} onScroll={event=>{if(savedView.current&&(!restoreScroll.current||restoreScroll.current.listDone)){savedView.current.listScroll=Math.round(event.currentTarget.scrollTop);saveResultView(savedView.current)}}} start={page*pageSize+1}>
            {pageRows.map(s=>{
              const state=automaticCheckState(s)
              const reviewState=s.manual_approved?t("人工审批通过"):s.manual_review?manualReviewLabels[s.manual_review.verdict]:s.reviewed?t("已定位 · 人工待审"):''
              return <li key={s.ordinal}><button disabled={busy} aria-current={s.ordinal===ordinal?'true':undefined} onClick={()=>chooseSample(s.ordinal)}>
                <span className="sample-list-row-title">
                  <strong>#{s.ordinal} · {categoryLabel(s.category)}</strong>
                  <span className={`sample-check-marker ${state}`} title={automaticCheckLabels[state]} aria-hidden="true">{{passed:'✓',failed:'!',pending:'…',unknown:'?'}[state]}</span>
                  <span className="sr-only">{automaticCheckLabels[state]}</span>
                </span>
                <p lang="en">{s.expression}</p>
                {reviewState?<small className={`review-status ${s.manual_review?.verdict??''}`}>{reviewState}</small>:<span className="sr-only">{t("人工待审")}</span>}
              </button></li>
            })}
          </ol>
          {visibleRows.length>pageSize&&<nav className="sample-list-pages" aria-label={t("样本列表分页")}><button disabled={busy||page===0} onClick={()=>setListPage(page-1)} aria-label={t("上一页样本")}>←</button><span>{page+1} / {Math.ceil(visibleRows.length/pageSize)}</span><button disabled={busy||(page+1)*pageSize>=visibleRows.length} onClick={()=>setListPage(page+1)} aria-label={t("下一页样本")}>→</button></nav>}
        </aside>
        <div className="sample-main" ref={sampleArea}>{!sample?<section className="panel sample-filter-empty">{linkedOrdinal?<><h2>{t("样本 #")}{linkedOrdinal}</h2><p>{t("该样本暂未出现在当前文本列表中，可查看返工进度或刷新。返工期间新描述可能尚未生成。")}</p>{currentRework?.item&&currentRework.task&&<ReworkProgress items={[currentRework.item]} task={currentRework.task} measuredAt={currentRework.measured_at} jobId={jobId}/>}<button disabled={loading} onClick={reloadResults}>{t("刷新该样本")}</button>{reworkError&&<p className="error" role="alert">{serviceText(reworkError)}</p>}</>:<><h2>{t("没有符合筛选条件的描述")}</h2><button onClick={()=>filterChecks('all')}>{t("查看全部描述")}</button></>}</section>:<section className="panel sample-content">
          <div className="section-heading">
            <div className="sample-identity"><h2>#{sample.ordinal} · {categoryLabel(sample.category)}</h2>{!review&&sample.source_view&&<span className="source-view-label">{t("来源相机 · ")}{cameraViewLabel(sample.source_view)}</span>}</div>
            <div className="sample-header-meta">
              {checkLabel&&!checkNote&&<span className="record-state record-state-passed">{checkLabel}</span>}
              {sample.legacy_rules&&<HelpHint label={t("旧规则记录")} detail={t("此描述采用较早的规则，可反馈修改。")} />}
            </div>
          </div>
          {index<0&&<p className="notice">{t("该样本不在当前“")}{sampleFilterLabels[checkFilter]}{t("”列表中，详情仍保留；左侧筛选未改变。")}</p>}
          <p className="expression" lang="en">{sample.expression}</p>
          {checkNote&&<div className={`sample-check-notice ${checkState}`}><strong>{checkLabel}</strong><span>{checkNote}</span></div>}
          {sample.viewing_previous&&<p className="notice">{t("返工尚未生成新描述，当前展示返工前版本及其对应原图和点云；此版本不能作为有效样本使用。")}</p>}
          {sceneError&&<div className="error results-error" role="alert"><span>{serviceText(sceneError)}</span><button onClick={()=>{setSceneError('');setSceneRequest(v=>v+1)}}>{t("重新加载场景")}</button></div>}
          <div className="sample-toolbar">
            <div className="sample-actions">
              <button className={review||reworkOpen?undefined:'primary'} disabled={busy||!currentScene} onClick={()=>{setReworkTarget('');if(review)toggleReview();else if(checkState==='failed'&&!sample.manual_approved)beginApproval();else toggleReview()}}>{review?t("结束定位检查"):t("检查物体定位")}</button>
              {feedbackAvailable&&<button className="quiet-button" disabled={busy} aria-expanded={reworkOpen} onClick={()=>{setReworkTarget(reworkOpen?'':`${jobId}:${ordinal}`)}}>{reworkOpen?t("收起反馈"):t("反馈修改")}</button>}
              <button className="danger quiet-button" aria-label={t("废弃样本 #{0}", sample.ordinal)} disabled={busy||!sample.can_review||sample.manual_review?.verdict==='discarded'} onClick={()=>void discardSample()}>{sample.manual_review?.verdict==='discarded'?t("已废弃"):t("废弃样本")}</button>
            </div>
            {index>=0&&visibleRows.length>1&&<nav className="sample-pagination" aria-label={positionLabel}>
              <button className="quiet-button" disabled={busy||index<=0} onClick={()=>stepSample(-1)}>{t("上一条")}</button>
              <span title={positionLabel}>{index+1} / {visibleRows.length}</span>
              <button className="quiet-button" disabled={busy||index>=visibleRows.length-1} onClick={()=>stepSample(1)}>{t("下一条")}</button>
            </nav>}
          </div>
          {reworkOpen&&<SampleReview key={`${jobId}:${ordinal}:${sample.retry_failed_run_id ?? sample.run_id}:repair`} jobId={jobId} ordinal={ordinal} revision={sample.revision} canReview={sample.can_review} canRetry={sample.can_retry ?? sample.can_review} retryFailedRunId={sample.retry_failed_run_id} automaticPassed={sample.automatic_passed}
            decision={sample.manual_review} localizationCorrect={false} busy={busy} onBusy={reviewBusy} onReviewed={reviewed} onRetry={retryRequested} onOpenRetry={openRetry} taskGoalReached={samples?.task_goal != null && samples.review_summary.usable >= samples.task_goal} repairMode onClose={()=>setReworkTarget('')}/>}
          {sampleActionError&&<p className="error" role="alert">{serviceText(sampleActionError)}</p>}
          {sample.manual_review?.verdict==='discarded'&&<p className="notice">{t("此版本已从有效样本和后续导出中排除，原始数据、标注历史和费用记录保留。")}</p>}
          {review&&!sample.can_review&&<p className="notice">{t("当前只展示返工前的旧版本，可检查图像与点云；本轮生成新草稿后，才能确认定位并提交人工审批。")}</p>}
          {review&&!reworkOpen&&sample.can_review&&<section ref={approvalArea} className={`sample-approval-flow${approved?' approved':''}`} aria-label={t("人工审批步骤")} aria-busy={busy}>
            <h3>{t("人工审批")}</h3>
            <ol className="sample-steps approval-steps" aria-label={t("审批步骤")}>{[t("选择物体"),t("确认定位"),t("提交审批")].map((label,index)=><li key={label}
              aria-current={index===approvalStep?'step':undefined} className={index<approvalStep?'done':index===approvalStep?'current':''}><span aria-hidden="true">{index<approvalStep?'✓':index+1}</span>{label}</li>)}</ol>
            <div className="approval-flow-actions"><span>{approved?t("审批已保存，当前版本已纳入可用集。"):!currentScene?t("正在加载定位场景…"):feedback?.correct?t("定位已确认，可以提交审批。"):selected?t("已选 {0}，请确认定位。", objectName(selected)):t("请在图像或点云中选择一个物体。")}</span>
              {!approved&&<>{!feedback?.correct?<button className="primary" disabled={busy||!selected||!currentScene?.can_confirm} onClick={()=>void confirm()}>{busy&&!approving?t("正在确认定位…"):t("确认所选物体")}</button>:<button className="primary" disabled={busy||!sample.can_review} onClick={()=>void approveSample()}>{approving?t("正在保存审批…"):t("审批通过并纳入可用集")}</button>}
              <button disabled={busy||!currentScene} onClick={()=>sceneArea.current?.scrollIntoView({block:'center',behavior:'smooth'})}>{t("查看图像与点云")}</button></>}
              {approved&&<span className="approval-saved">{t("已审批通过 · 当前可用 ")}{summary?.usable??'—'}{t(" 条")}</span>}
            </div>
            {currentAction&&<p className={`approval-action-feedback ${currentAction.error?'error':'success'}`} role={currentAction.error?'alert':'status'} aria-live="polite">{serviceText(currentAction.message)}</p>}
            {!approved&&!currentAction&&feedback&&!feedback.correct&&<p className="error" role="status">{t("定位不正确，请重新选择物体。")}</p>}
          </section>}
          {!review&&currentAction&&<p className={`approval-action-feedback ${currentAction.error?'error':'success'}`} role={currentAction.error?'alert':'status'}>{serviceText(currentAction.message)}</p>}
          <div ref={sceneArea}>{currentScene?<TaskScene key={`${jobId}:${ordinal}:${currentScene.revision}:${review}`} data={currentScene} selected={selected} compact onSelect={review?(id=>{if(!busy){setSelected(id);setConfirmation(null);setActionFeedback(null)}}):undefined}/>:!sceneError&&<div className="scene-loading results-scene-loading" role="status"><span className="loading-dot"/>{t("正在加载图像与点云…")}</div>}</div>
          {review&&sample&&!reworkOpen&&<SampleReview key={`${jobId}:${ordinal}:${sample.retry_failed_run_id ?? sample.run_id}`} jobId={jobId} ordinal={ordinal} revision={sample.revision} canReview={sample.can_review} canRetry={sample.can_retry ?? sample.can_review} retryFailedRunId={sample.retry_failed_run_id} automaticPassed={sample.automatic_passed}
            decision={sample.manual_review} localizationCorrect={feedback?.correct===true} busy={busy} onBusy={reviewBusy} onReviewed={reviewed} onRetry={retryRequested} onOpenRetry={openRetry} taskGoalReached={samples?.task_goal != null && samples.review_summary.usable >= samples.task_goal} approvalInHeader/>}
          {currentRework?.item&&currentRework.task&&<ReworkProgress items={[currentRework.item]} task={currentRework.task} measuredAt={currentRework.measured_at} jobId={jobId}/>}
          {reworkError&&<p className="notice" role="status">{t("返工详情读取失败：")}{serviceText(reworkError)} <button className="text-button" onClick={()=>setRefresh(v=>v+1)}>{t("重试读取返工")}</button></p>}
        </section>}
        </div>
      </div>}
    </>}
  </div>
}
