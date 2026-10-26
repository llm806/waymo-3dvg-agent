import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { setImmediate } from 'node:timers/promises'
import { readJson, readPollingJson, errorMessage, isAnnotationResult, isProcessingRecord, providerDiagnosisLabel, categoryLabel, reasonLabel, phases, automaticCheckState, automaticCheckLabels } from '../src/shared/annotationDisplay.ts'
import { taskName, taskDetails, parseAnnotationCount, exportTaskSelection } from '../src/shared/taskName.ts'
import { formatDuration, formatTiming, hasIncompleteTiming } from '../src/shared/duration.ts'
import { taskLocation, taskUrl, openTask, sampleUrl } from '../src/navigation.ts'
import * as resultViewState from '../src/features/results/resultViewState.ts'
import { renderToStaticMarkup } from 'react-dom/server'

const require = createRequire(import.meta.url)
const ts = require('typescript')
const compiled = ts.transpileModule(readFileSync(new URL('../src/features/tasks/TaskManager.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText

// Run the component's event handlers with local hook/DOM substitutes. No server,
// real deletion request, extra runtime dependency or model call is needed.
function manager(outcome, taskCount = 2, activeSecond = false) {
  const original = { fetch: globalThis.fetch, document: globalThis.document, HTMLElement: globalThis.HTMLElement }
  class Element {
    isConnected = true
    open = false
    focus() { this.focused = true }
    showModal() { this.open = true }
    close() { this.open = false }
  }
  globalThis.HTMLElement = Element
  globalThis.document = { activeElement: new Element() }
  const deletionRequests = []
  let currentAction = 'preview', polls = 0
  globalThis.fetch = async (url, options) => {
    const body = options?.body ? JSON.parse(options.body) : undefined
    deletionRequests.push({url, body})
    if (outcome==='old_backend') return new Response(JSON.stringify({detail:'Not Found'}),{status:404})
    if (body) currentAction = body.action
    const result = currentAction === 'preview'
      ? { requested: ['a'], job_ids: ['a'], token: 'exclude-token', return_to_pool: false, processed_tracks: 2,
        candidate_tokens: { return: 'return-token', exclude: 'exclude-token' },
        summary: { tasks: 1, retained_tasks: taskCount - 1 }, retained_shared: {} }
      : { deleted_job_ids: ['a'], cleanup_pending: outcome === 'pending' }
    const running = outcome === 'progress' && (!body ? polls++ === 0 : true)
    const failed = outcome === 'error' && currentAction === 'delete'
    return new Response(JSON.stringify({operation_id:'operation',status:failed?'failed':running?'running':'completed',
      stage:running?'references':'finished',message:failed?'Test deletion failed':'正在核对执行记录',
      completed:running?20:1,total:running?100:1,files_checked:200,result:running||failed?null:result}))
  }
  let index = 0, effects = [], tree
  const slots = [], targets = new Map()
  const hooks = {
    useState(initial) {
      const key = index++
      if (!(key in slots)) slots[key] = initial
      return [slots[key], value => { slots[key] = value }]
    },
    useRef(initial) {
      const key = index++
      if (!(key in slots)) slots[key] = { current: initial }
      return slots[key]
    },
    useEffect(effect) { effects.push(effect) },
  }
  const exports = {}
  new Function('require', 'exports', compiled)(name => {
    if (name === 'react') return hooks
    if (name === 'react/jsx-runtime') return require(name)
    if (name.endsWith('/annotationDisplay')) return { readJson, errorMessage }
    if (name.endsWith('/taskName')) return { taskName, taskDetails, exportTaskSelection }
    throw new Error(`Unexpected import: ${name}`)
  }, exports)
  const props = {
    jobs: ['a', 'b'].slice(0, taskCount).map(job_id => ({ job_id, status: activeSecond && job_id === 'b' ? 'running' : 'stopped',
      executor_active: activeSecond && job_id === 'b', created_at: 0, counts: {} })),
    activeJobId: activeSecond ? 'b' : undefined, labels: {}, onOpen() {},
    onDeleted(ids) { props.jobs = props.jobs.filter(job => !ids.includes(job.job_id)) },
  }
  function walk(node) {
    return Array.isArray(node) ? node.flatMap(walk) : node && typeof node === 'object'
      ? [node, ...walk(node.props?.children)] : []
  }
  const nodes = () => walk(tree)
  function render() {
    index = 0; effects = []
    tree = exports.default(props)
    for (const node of nodes()) {
      const ref = node.props?.ref
      if (!ref) continue
      if (!targets.has(ref)) targets.set(ref, new Element())
      ref.current = targets.get(ref)
    }
    for (const [ref, target] of targets) {
      if (!nodes().some(node => node.props?.ref === ref)) { ref.current = null; target.isConnected = false }
    }
    effects.forEach(effect => effect())
  }
  const button = text => nodes().find(node => node.type === 'button' && node.props.children === text)
  render()
  button('管理').props.onClick(); render()
  nodes().find(node => node.type === 'input' && node.props['aria-label']?.startsWith('选择任务 ')).props.onChange({ target: { checked: true } })
  render()
  return { render, button, nodes, deletionRequests, restore: () => Object.assign(globalThis, original), async preview() {
    button('删除所选任务').props.onClick(); render()
    assert.ok(nodes().some(node => node.type === 'dialog' && node.props['aria-busy']))
    assert.ok(nodes().some(node => node.type === 'h2' && node.props.children === '正在核对删除范围…'))
    assert.equal(button('永久删除'), undefined)
    await setImmediate(); render()
  } }
}

test('task quantities accept editable integers and count only passed descriptions', async () => {
  const taskManager = {}
  new Function('require', 'exports', compiled)(name => {
    if (name.endsWith('/annotationDisplay')) return { readJson, errorMessage }
    if (name.endsWith('/taskName')) return { taskName, taskDetails, exportTaskSelection }
    return require(name)
  }, taskManager)
  const listProps = { jobs: [], labels: {}, onOpen() {}, onDeleted() {}, onReload() {} }
  for (const state of [{ loading: true }, { loadError: 'Service unavailable' }]) {
    const html = renderToStaticMarkup(require('react').createElement(taskManager.default, { ...listProps, ...state }))
    assert.doesNotMatch(html, /暂无任务/)
    assert.match(html, state.loading ? /正在读取已有任务/ : /任务列表读取失败.*重新读取/)
  }
  assert.match(renderToStaticMarkup(require('react').createElement(taskManager.default, listProps)), /暂无任务/)
  assert.equal(parseAnnotationCount(''), null)
  assert.equal(parseAnnotationCount('020'), 20)
  for (const value of ['0', '-1', '1.5', '1e2', 'abc', '9007199254740993']) assert.equal(parseAnnotationCount(value), null)
  const task = { created_at:0, counts:{pending:156}, selection_progress:{requested:20} }
  assert.equal(taskDetails(task).scope, '目标 20 条描述')
  const validationTask = { ...task, spec: { dataset_split: 'validation' } }
  assert.equal(taskDetails(validationTask).scope, '验证集 · 目标 20 条描述')
  assert.deepEqual(exportTaskSelection([task]), { split: 'train', error: '' })
  assert.deepEqual(exportTaskSelection([validationTask]), { split: 'validation', error: '' })
  assert.match(exportTaskSelection([task, validationTask]).error, /不能一起导出.*分开导出/)
  assert.match(taskName(task), /目标 20 条描述$/)
  const successor={created_at:0,counts:{reused:60,pending:140},selection_progress:{requested:200},
    continuation:{source_job_id:'source-job',inherited_passed:60}}
  assert.equal(taskDetails(successor).scope,'目标 200 条描述')
  assert.equal(taskName(successor), taskName({...successor,continuation:null}))
  assert.equal(taskDetails({...successor,counts:{reused:60,calibration_hold:80}}).scope,taskDetails(successor).scope)
  assert.equal(taskDetails({...successor,continuation:{source_job_id:'next-source',inherited_passed:100}}).scope,'目标 200 条描述')
  const continuedList=renderToStaticMarkup(require('react').createElement(taskManager.default,{...listProps,
    jobs:[{...successor,job_id:'successor',status:'ready',executor_active:false}]}))
  assert.match(continuedList,/目标 200 条描述/)
  assert.doesNotMatch(continuedList,/续作|目标新增|继承|合计目标/)
  const queuedList=renderToStaticMarkup(require('react').createElement(taskManager.default,{...listProps,
    jobs:[{...successor,job_id:'queued',status:'queued',queue_position:2,executor_active:false}]}))
  assert.match(queuedList,/排队第 2 位/)
  assert.doesNotMatch(queuedList,/执行中/)
  const records = Array.from({length:156}, () => ({status:'pending'}))
  records[0] = {status:'calibration_hold', expression:'The silver sedan.'}
  records[1] = {status:'needs_evidence', expression:'Rejected draft.'}
  records[2] = {status:'running', expression:'Unchecked draft.'}
  assert.equal(records.filter(isAnnotationResult).length, 1)
  assert.equal(records.filter(isProcessingRecord).length, 3)
  assert.equal(isProcessingRecord({status:'preparing'}), true)
  assert.equal(isProcessingRecord({status:'not_needed'}), false)
  assert.equal(isProcessingRecord({status:'budget_skipped'}), false)
  assert.equal(isProcessingRecord({status:'pending',packet_id:'prepared-packet'}), true)
  assert.equal(isAnnotationResult({status:'reused', run_status:'needs_evidence', expression:'Rejected.'}), false)
  assert.equal(automaticCheckState({automatic_passed:true,status:'reused',can_review:true}),'passed')
  assert.equal(automaticCheckState({automatic_passed:false,status:'needs_evidence',can_review:true,manual_review:{verdict:'accepted'}}),'failed')
  assert.equal(automaticCheckState({automatic_passed:false,status:'running',can_review:false}),'pending')
  assert.equal(automaticCheckState({status:'calibration_hold',can_review:true}),'unknown')
  assert.equal(automaticCheckLabels.failed,'自动核验未通过')
  assert.match(providerDiagnosisLabel('vision_model_text_only'), /模型仅支持文本/)
  assert.match(reasonLabel('precheck_visible_but_ambiguous'), /已确认可见.*未能完成消歧/)
  const metricCode = ts.transpileModule(readFileSync(new URL('../src/features/tasks/TaskMetrics.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  const component = {}
  new Function('require', 'exports', metricCode)(name => {
    if (name.endsWith('/annotationDisplay')) return { categoryLabel, reasonLabel }
    if (name.endsWith('/duration')) return { formatDuration, formatTiming, hasIncompleteTiming }
    return require(name)
  }, component)
  const rate = { numerator:1, denominator:2, value:.5 }
  const metrics = { schema_version:'task-metrics-v1', timing:{active_seconds:40,wall_seconds:122,status:'complete'},
    counts:{pool:2,attempted:2,settled:2,passed:1,new_passed:1,reused:0,unattempted:0},
    coverage:{pool_utilization:rate,dataset_utilization:rate,pass_per_settled:rate}, candidate_coverage:rate,
    speed:{item_active_seconds:{count:2,p50:17.5,p95:24.25,mean:17.5},mean_active_seconds_per_new_pass:40,new_passes_per_active_hour:90,new_passes_per_wall_hour:29.5},
    quality:{first_draft_pass:rate,first_draft_incomplete:0,repair_success:rate,counterfactuals:{pass_per_complete:rate,complete_per_workload:rate}},
    cost:{reserved_model_calls:4,model_calls:4,model_seconds:20,model_seconds_complete:true,tokens:12,tokens_complete:false,calls_per_new_pass:{value:4},tokens_per_new_pass:null},
    independent_audits:[],categories:[],failure_reasons:[] }
  const html = renderToStaticMarkup(component.default({metrics,jobId:'test'}))
  assert.match(html, /任务候选利用率/)
  assert.match(html, /处理耗时 P50/)
  assert.match(html, /122|2 分 2 秒/)
  assert.match(html, /尚未进行独立盲评/)
  assert.match(html, /已知用量，尚不完整/)
  assert.match(html, /report\?download=true/)
  assert.equal(formatDuration(null), '未测量')
  assert.equal(formatDuration(0), '0 秒')
  const partialTiming = {active_seconds:null,recorded_active_seconds:18,history_incomplete:true,running:true,status:'running'}
  assert.equal(formatTiming(partialTiming,10), '已记录 28 秒')
  assert.equal(formatTiming({...partialTiming,running:false},1000), '已记录 18 秒')
  assert.equal(formatTiming({active_seconds:40,running:false}), '40 秒')
  assert.equal(formatTiming({active_seconds:null,recorded_active_seconds:0,history_incomplete:true}), '已记录 0 秒')
  assert.equal(formatTiming({active_seconds:null,history_incomplete:true}), '未测量')
  assert.ok(hasIncompleteTiming({active_seconds:null,running:true}))
  const partialHtml = renderToStaticMarkup(component.default({metrics:{...metrics,timing:partialTiming},jobId:'test'}))
  assert.match(partialHtml, /已记录 18 秒/)
  assert.match(partialHtml, /历史计时不完整，缺失部分未计入已记录用时/)
  const exportCode = ts.transpileModule(readFileSync(new URL('../src/features/tasks/DatasetExport.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  let slotIndex = 0
  const states = [], exportComponent = {}, exportEffects = [], exportCleanups = []
  new Function('require', 'exports', exportCode)(name => {
    if (name === 'react') return { useState(initial) {
      const key = slotIndex++
      if (!(key in states)) states[key] = initial
      return [states[key], next => { states[key] = typeof next === 'function' ? next(states[key]) : next }]
    }, useRef(initial) {
      const key=slotIndex++
      if (!(key in states)) states[key]={current:initial}
      return states[key]
    }, useEffect(effect,deps) {
      const key=slotIndex++
      if (!(key in states) || deps.some((value,index)=>value!==states[key][index])) {
        states[key]=deps;exportEffects.push(effect)
      }
    } }
    if (name.endsWith('/annotationDisplay')) return {readJson,errorMessage}
    if (name.endsWith('/taskName')) return { taskName, taskDetails, exportTaskSelection }
    return require(name)
  }, exportComponent)
  let exportJobs = ['a','b'].map(job_id => ({job_id,created_at:0,counts:{},status:'completed',executor_active:false}))
  const flatten = node => Array.isArray(node) ? node.flatMap(flatten) : node && typeof node === 'object'
    ? [node,...flatten(node.props?.children)] : []
  const exportNodes = () => { slotIndex=0;return flatten(exportComponent.default({jobs:exportJobs,current:'b'})) }
  const flushExportEffects = () => { for (const effect of exportEffects.splice(0)) { const cleanup=effect();if (cleanup) exportCleanups.push(cleanup) } }
  const originalExportFetch = globalThis.fetch
  globalThis.fetch = async () => new Response('null')
  exportNodes();flushExportEffects();await setImmediate()
  globalThis.fetch = originalExportFetch
  let formNodes = exportNodes()
  assert.equal(formNodes.find(n=>n.props?.id==='dataset-train-reason').props.reason,'请选择要打包的任务。')
  const blockedHint = formNodes.find(n=>n.props?.id==='dataset-train-reason')
  const hintTree = blockedHint.type(blockedHint.props)
  assert.equal(hintTree.props.tabIndex,0)
  assert.ok(flatten(hintTree).some(n=>n.props?.role==='tooltip' && n.props.children==='请选择要打包的任务。'))
  hintTree.props.onClick()
  assert.equal(exportNodes().find(n=>n.props?.className==='dataset-export-feedback').props.children,'请选择要打包的任务。')
  let explainedByKeyboard=false
  hintTree.props.onKeyDown({key:'Enter',preventDefault(){explainedByKeyboard=true}})
  assert.ok(explainedByKeyboard)
  assert.ok(formNodes.filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  let prevented = false
  formNodes.find(n=>n.type==='form').props.onSubmit({preventDefault(){prevented=true}})
  assert.ok(prevented)
  formNodes.find(n=>n.type==='button' && n.props.children==='仅当前任务').props.onClick()
  formNodes = exportNodes()
  assert.deepEqual(formNodes.filter(n=>n.type==='input' && n.props.name==='job_ids').map(n=>n.props.value),['b'])
  assert.equal(formNodes.find(n=>n.props?.id==='dataset-train-reason').props.reason,undefined)
  assert.ok(formNodes.filter(n=>n.type==='button' && n.props.type==='submit').every(n=>!n.props.disabled))
  exportJobs = exportJobs.map(job => ({...job,spec:{dataset_split:job.job_id==='a'?'validation':'train'}}))
  exportNodes().find(n=>n.type==='button' && n.props.children==='全选').props.onClick()
  assert.match(exportNodes().find(n=>n.props?.id==='dataset-train-reason').props.reason,/分开导出/)
  assert.ok(exportNodes().some(n=>n.props?.role==='alert' && /分开导出/.test(n.props.children)))
  assert.ok(exportNodes().filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  exportNodes().find(n=>n.type==='button' && n.props.children==='仅当前任务').props.onClick()
  exportJobs = exportJobs.map(job => ({...job,spec:{dataset_split:'train'}}))
  formNodes=exportNodes()
  formNodes.find(n=>n.type==='input' && n.props.name==='filename').props.onChange({target:{value:'trainset1'}})
  assert.equal(exportNodes().find(n=>n.type==='input' && n.props.name==='filename').props.value,'trainset1')
  for (const [status,executor_active] of [['running',true],['paused',false],['ready',false],['stopped',false],['blocked',false],['completed',true]]) {
    exportJobs = exportJobs.map(job=>job.job_id==='b'?{...job,status,executor_active}:job)
    formNodes = exportNodes()
    assert.ok(formNodes.some(n=>n.type==='input' && n.props.type==='checkbox' && n.props.disabled))
    assert.ok(formNodes.find(n=>n.type==='button' && n.props.children==='仅当前任务').props.disabled)
    assert.equal(formNodes.filter(n=>n.type==='input' && n.props.name==='job_ids').length,0)
    assert.ok(formNodes.filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  }
  formNodes.find(n=>n.type==='button' && n.props.children==='全选').props.onClick()
  assert.deepEqual(exportNodes().filter(n=>n.type==='input' && n.props.name==='job_ids').map(n=>n.props.value),['a'])
  exportNodes().find(n=>n.type==='input' && n.props.type==='radio' && n.props.value==='all').props.onChange()
  assert.match(exportNodes().find(n=>n.props?.id==='dataset-download-status').props.children.join(''),/1 个任务/)
  exportNodes().find(n=>n.type==='input' && n.props.type==='radio' && n.props.value==='selected').props.onChange()
  exportJobs = exportJobs.map(job=>({...job,status:'completed',executor_active:false}))
  exportNodes().find(n=>n.type==='button' && n.props.children==='仅当前任务').props.onClick()
  exportJobs = exportJobs.filter(j=>j.job_id!=='b')
  assert.ok(exportNodes().filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  exportJobs = []
  formNodes = exportNodes()
  assert.match(formNodes.find(n=>n.props?.id==='dataset-train-reason').props.reason,/暂无任务/)
  assert.ok(formNodes.find(n=>n.type==='input' && n.props.name==='filename').props.disabled)
  formNodes.find(n=>n.type==='input' && n.props.type==='radio' && n.props.value==='all').props.onChange()
  assert.ok(exportNodes().filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  exportJobs = [{job_id:'a',created_at:0,counts:{},status:'completed',executor_active:false}]
  formNodes = exportNodes()
  assert.equal(formNodes.filter(n=>n.type==='input' && n.props.name==='job_ids').length,0)
  formNodes.find(n=>n.type==='input' && n.props.name==='filename').props.onChange({target:{value:'../outside'}})
  assert.ok(exportNodes().filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
  assert.match(exportNodes().find(n=>n.props?.id==='dataset-train-reason').props.reason,/名称无效/)
  // Exercise real submit/cancel handlers without a server or native download.
  const originals = { fetch:globalThis.fetch, window:globalThis.window, setTimeout:globalThis.setTimeout, clearTimeout:globalThis.clearTimeout }
  const calls=[], timers=new Map()
  let confirm=false, timerId=0
  let serverJob={export_id:'e'.repeat(24),status:'running',stage:'packaging',message:'正在写入完整点云',completed:1,total:4,bytes_written:1234,bytes_sent:0,created_at:Date.now()/1000,updated_at:Date.now()/1000}
  globalThis.window={confirm:()=>confirm}
  globalThis.setTimeout=callback=>{timers.set(++timerId,callback);return timerId}
  globalThis.clearTimeout=id=>timers.delete(id)
  globalThis.fetch=async (url,options)=>{
    calls.push({url,options})
    if (url.endsWith('/cancel')) serverJob={...serverJob,status:'cancelling',message:'正在清理…'}
    return new Response(JSON.stringify(serverJob))
  }
  try {
    exportJobs = exportJobs.map(job=>({...job,spec:{dataset_split:'validation'}}))
    exportNodes().find(n=>n.type==='input' && n.props.name==='filename').props.onChange({target:{value:'trainset1'}})
    const submit = exportNodes().find(n=>n.type==='form').props.onSubmit
    const event={preventDefault(){},nativeEvent:{submitter:{value:'train'}}}
    submit(event);submit(event)
    assert.equal(calls.filter(c=>c.options?.method==='POST').length,1)
    assert.equal(JSON.parse(calls[0].options.body).export_scope,'all')
    assert.equal(JSON.parse(calls[0].options.body).split,'validation')
    await setImmediate();exportNodes();flushExportEffects();await setImmediate()
    let nodes=exportNodes()
    assert.ok(nodes.filter(n=>n.type==='button' && n.props.type==='submit').every(n=>n.props.disabled))
    const progress=nodes.find(n=>n.type==='progress')
    assert.equal(progress.props.value,1);assert.equal(progress.props.max,4)
    await nodes.find(n=>n.type==='button' && n.props.children==='终止导出').props.onClick()
    assert.equal(calls.filter(c=>c.url.endsWith('/cancel')).length,0)
    confirm=true
    nodes.find(n=>n.type==='button' && n.props.children==='终止导出').props.onClick()
    await setImmediate();nodes=exportNodes()
    assert.ok(nodes.find(n=>n.type==='button' && n.props.children==='正在终止并清理…').props.disabled)
    assert.equal(calls.filter(c=>c.url.endsWith('/cancel')).length,1)
    serverJob={...serverJob,status:'cancelled',stage:'finished',message:'已终止导出，临时文件已清理。'}
    const poll=timers.values().next().value;timers.clear();await poll()
    nodes=exportNodes();flushExportEffects()
    assert.equal(nodes.some(n=>n.type==='progress'),false)
    assert.ok(nodes.filter(n=>n.type==='button' && n.props.type==='submit').every(n=>!n.props.disabled))
    // Download handoff uses a native link and the same job; never buffers a
    // multi-gigabyte archive in a browser Blob.
    serverJob={...serverJob,status:'running',stage:'packaging'}
    submit(event);await setImmediate()
    serverJob={...serverJob,status:'ready',stage:'ready'}
    const running=exportNodes();flushExportEffects();await setImmediate()
    let clicked=0
    const readyNodes=exportNodes()
    readyNodes.find(n=>n.type==='a' && n.props.target==='dataset-export-download').props.ref.current={click(){clicked++}}
    flushExportEffects()
    assert.equal(clicked,1)
    assert.ok(running.some(n=>n.props?.className?.startsWith('dataset-export-progress')))
  } finally {
    for (const cleanup of exportCleanups) cleanup()
    Object.assign(globalThis,originals)
  }
})

test('deletion and rework preserve user decisions when follow-up operations fail', async () => {
  const workspaceCode=ts.transpileModule(readFileSync(new URL('../src/features/tasks/BatchWorkspace.tsx',import.meta.url),'utf8'),{
    compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX},
  }).outputText
  const walk=node=>Array.isArray(node)?node.flatMap(walk):node&&typeof node==='object'?[node,...walk(node.props?.children)]:[]
  // Visibility changes cancel display polling only. Returning to the tab
  // refreshes immediately; no start/pause/control request is ever sent.
  {
    const saved={window:globalThis.window,document:globalThis.document,fetch:globalThis.fetch,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout}
    const listeners=new Map(),timers=new Map(),requests=[],slots=[],effects=[],pending=[],component={}
    let job=null, holdDetail=false, detailSignal=null, tree, reportRequests=0, finishReport, reportSignal
    let slot=0,serial=0
    globalThis.document={visibilityState:'visible',addEventListener(name,fn){listeners.set(name,fn)},removeEventListener(name){listeners.delete(name)}}
    globalThis.window={location:{hash:''},addEventListener(name,fn){listeners.set('window:'+name,fn)},removeEventListener(name){listeners.delete('window:'+name)}}
    globalThis.setTimeout=(fn,ms)=>{const id=++serial;timers.set(id,{fn,ms});return id}
    globalThis.clearTimeout=id=>timers.delete(id)
    globalThis.fetch=async (url,options)=>{
      requests.push({url,options})
      assert.equal(options?.method,undefined)
      if (holdDetail && url==='/api/batches/live?compact=true') {
        detailSignal=options.signal
        return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}))
      }
      if (url.endsWith('/report')) {
        reportRequests++
        if (reportRequests===1) return new Response(JSON.stringify({detail:'正在读取任务统计，请等待本次统计结束后再试。'}),{status:409})
        reportSignal=options.signal
        return new Promise(resolve=>{finishReport=()=>resolve(new Response(JSON.stringify({core_metrics:{schema_version:'task-metrics-test'}})))})
      }
      const result=url.startsWith('/api/batches/catalog')?{catalog_id:'a'.repeat(24),summary:{},targets:[],total:0}
        :url==='/api/system/status'?{annotation_ready:true,checks:[],attempted_target_ids:[]}
        :url.endsWith('/report')?{core_metrics:{schema_version:'task-metrics-test'}}
        :url.endsWith('?compact=true')?{...job,measured_at:20,items:[],events:[],model_calls_reserved:0,selection_progress:{requested:1000,diagnostic_passed:1000,attempted:4533,candidate_pool:4533}}
        :url.endsWith('/progress')?{...job,counts:{calibration_hold:965},model_calls_reserved:32649,measured_at:10,
          selection_progress:{requested:1000,diagnostic_passed:965,attempted:4427,candidate_pool:4533},active_item_timings:{},
          review_retries:job.job_id==='retry-job'?[{ordinal:12,status:'pending',calls:0,call_budget:12}]:[]}
        :{jobs:job?[job]:[],active_batch_id:job?.job_id??null}
      return new Response(JSON.stringify(result))
    }
    const same=(a,b)=>a&&b&&a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]))
    new Function('require','exports',workspaceCode)(name=>{
      if(name==='react')return {
        useState(initial){const key=slot++;if(!(key in slots))slots[key]=typeof initial==='function'?initial():initial
          return [slots[key],v=>{slots[key]=typeof v==='function'?v(slots[key]):v}]},
        useMemo(fn,deps){const key=slot++;if(!same(slots[key]?.deps,deps))slots[key]={deps,value:fn()};return slots[key].value},
        useEffect(fn,deps){const key=slot++;if(!same(effects[key]?.deps,deps))pending.push(()=>{
          effects[key]?.cleanup?.();effects[key]={deps,cleanup:fn()}
        })},
      }
      if(name.endsWith('/annotationDisplay'))return {readJson,readPollingJson,errorMessage,isAnnotationResult,isProcessingRecord,categoryLabel,reasonLabel,providerDiagnosisLabel,phases:{}}
      if(name.endsWith('/taskName'))return {taskName,parseAnnotationCount}
      if(name.endsWith('/duration'))return {formatDuration,formatTiming,hasIncompleteTiming}
      if(name.endsWith('/navigation'))return {taskLocation,taskUrl,openTask,sampleUrl}
      if(name==='react/jsx-runtime')return require(name)
      return {default:()=>null}
    },component)
    const render=()=>{slot=0;tree=component.default({active:true,onOpenPacket(){}});pending.splice(0).forEach(fn=>fn())}
    try {
      render();await setImmediate();render()
      assert.equal(requests.filter(r=>r.url==='/api/system/status').length,1)
      assert.equal(requests.filter(r=>r.url==='/api/batches').length,1)
      assert.equal(timers.size,2)
      const initial=requests.length
      globalThis.document.visibilityState='hidden';listeners.get('visibilitychange')();render()
      assert.equal(timers.size,0)
      assert.ok(requests.filter(r=>!r.url.includes('/catalog')).every(r=>r.options.cache==='no-store'))
      await setImmediate();assert.equal(requests.length,initial)
      globalThis.document.visibilityState='visible';listeners.get('visibilitychange')();render()
      await setImmediate();render()
      assert.equal(requests.filter(r=>r.url==='/api/system/status').length,2)
      assert.equal(requests.filter(r=>r.url==='/api/batches').length,2)
      assert.equal(requests.filter(r=>r.url.includes('/catalog')).length,1)
      // The full result read can hang while the independently scheduled live
      // projection renders the saved counters on the initial visit.
      job={job_id:'live',status:'running',created_at:0,counts:{},executor_active:true,control:null,reason:null,
        spec:{prepare_only:false,desired_count:1000,max_model_calls:null,catalog_id:'a'.repeat(24)}}
      holdDetail=true
      assert.ok([...timers.values()].every(t=>t.ms===30000))
      const entry=[...timers].reverse().find(([,t])=>t.ms===30000)
      timers.delete(entry[0]);entry[1].fn();await setImmediate();render()
      await setImmediate();render()
      assert.ok(walk(tree).some(n=>n.type==='progress'&&n.props.value===965))
      assert.ok(requests.some(r=>r.url==='/api/batches/live/progress'))
      assert.equal(detailSignal.aborted,false)
      globalThis.document.visibilityState='hidden';listeners.get('visibilitychange')();render()
      assert.equal(detailSignal.aborted,true)
      await setImmediate()
      // Opening an already-completed task must not auto-scan every model call.
      holdDetail=false;job={...job,status:'completed',executor_active:false}
      globalThis.document.visibilityState='visible';listeners.get('visibilitychange')();render()
      await setImmediate();render();await setImmediate();render()
      assert.equal(requests.filter(r=>r.url.endsWith('/report')).length,0)
      const metricsButton=walk(tree).find(n=>n.type==='button'&&n.props.children==='查看任务统计')
      assert.ok(metricsButton)
      metricsButton.props.onClick();render();await setImmediate();render()
      assert.equal(requests.filter(r=>r.url.endsWith('/report')).length,1)
      assert.ok(requests.filter(r=>r.url.endsWith('/report')).every(r=>r.options.cache==='no-store'))
      assert.ok(walk(tree).some(n=>n.props?.role==='status'&&n.props.children?.includes('自动重试')))
      assert.equal(walk(tree).some(n=>n.type==='button'&&n.props.children==='重试统计'),false)
      const retry=[...timers].find(([,t])=>t.ms===5000)
      assert.ok(retry)
      timers.delete(retry[0]);retry[1].fn();await setImmediate();render()
      assert.equal(reportRequests,2)
      assert.ok([...timers.values()].some(t=>t.ms>60000))
      // Hiding/returning and refreshing the task list cannot abandon a long
      // statistics read or send a duplicate that gets rejected as busy.
      globalThis.document.visibilityState='hidden';listeners.get('visibilitychange')();render()
      assert.equal(reportSignal.aborted,false)
      globalThis.document.visibilityState='visible';listeners.get('visibilitychange')();render()
      await setImmediate();render()
      assert.equal(reportRequests,2)
      assert.equal(reportSignal.aborted,false)
      finishReport();await setImmediate();render()
      assert.ok(walk(tree).some(n=>n.props?.metrics?.schema_version==='task-metrics-test'))
      // Task controls still follow deep links; rework is shown in Results.
      job={...job,job_id:'retry-job',status:'ready'}
      openTask(job.job_id,12)
      assert.deepEqual(taskLocation(),{jobId:'retry-job',ordinal:12})
      listeners.get('window:hashchange')();render();await setImmediate();render();await setImmediate();render()
      assert.ok(requests.some(r=>r.url==='/api/batches/retry-job/progress'))
      assert.equal(walk(tree).some(n=>n.props?.items?.[0]?.ordinal===12),false)
    } finally {
      effects.forEach(effect=>effect?.cleanup?.())
      assert.equal(timers.size,0);assert.equal(listeners.size,0)
      Object.assign(globalThis,saved)
    }
  }
  for (const mode of ['recover','resume','failure','retry','delete','delete_failure','delete_cancel','restart_required']) {
    const originalFetch=globalThis.fetch, originalWindow=globalThis.window, states=[], calls=[], component={}, confirmations=[]
    globalThis.window={confirm:text=>{confirmations.push(text);return mode!=='delete_cancel'}}
    let slot=0, rejected=false
    const paused={job_id:'paused-job',status:'paused',created_at:0,counts:{calibration_hold:60},reason:'RunBusy',executor_active:false,control:null,
      spec:{prepare_only:false,max_model_calls:null,desired_count:200,catalog_id:'a'.repeat(24)},model_calls_reserved:500,
      selection_progress:{requested:200,diagnostic_passed:60,attempted:95,candidate_pool:300},items:[],events:[],
      resume:{mode:mode==='resume'?'resume':'recover',message:'系统已更新，已完成的标注会保留。点击“继续标注”处理剩余数量。'}}
    if (mode==='restart_required') paused.resume={mode:'unavailable',message:'后端运行期间代码已更新，请先重启后端服务。'}
    if (mode.startsWith('delete')) paused.items=[{ordinal:5059,run_id:'a'.repeat(24),packet_id:'b'.repeat(24),
      expression:'The visible red vehicle in the left-side view.',status:'calibration_hold',legacy_rules:true,
      target:{category:'vehicle',context:'same-context',object_id:'native-object',timestamp:123},manual_review:null}]
    new Function('require','exports',workspaceCode)(name=>{
      if (name==='react') return {useEffect(){},useMemo:fn=>fn(),useState(initial){
        const key=slot++
        if (!(key in states)) states[key]=key===7?paused:typeof initial==='function'?initial():initial
        return [states[key],value=>{states[key]=typeof value==='function'?value(states[key]):value}]
      }}
      if (name.endsWith('/annotationDisplay')) return {readJson,readPollingJson,errorMessage,isAnnotationResult,isProcessingRecord,categoryLabel,reasonLabel,providerDiagnosisLabel,phases:{}}
      if (name.endsWith('/taskName')) return {taskName,parseAnnotationCount}
      if (name.endsWith('/duration')) return {formatDuration,formatTiming,hasIncompleteTiming}
      if (name.endsWith('/navigation')) return {taskLocation,taskUrl,openTask,sampleUrl}
      if (name.endsWith('/cameraViews')) return {cameraViewLabel:()=>''}
      if (name==='react/jsx-runtime') return require(name)
      return {default:()=>null}
    },component)
    const render=()=>{slot=0;return walk(component.default({active:false,onOpenPacket(){}}))}
    globalThis.fetch=async (url,options)=>{
      calls.push({url,options})
      const failed=mode==='failure' || mode==='retry' && !rejected || mode==='delete_failure'
      rejected=true
      return new Response(JSON.stringify(failed?{detail:'环境未就绪，原任务和已有标注已保留。'}:mode==='delete'
        ?{verdict:'discarded',revision:'c'.repeat(64),review_id:'d'.repeat(64),notes:'从标注结果列表删除。',
          progress:{passed:59},jobs:[{...paused,counts:{calibration_hold:59,discarded:1}}]}:{job_id:paused.job_id}),{status:failed?409:200})
    }
    try {
      let nodes=render()
      if (mode==='restart_required') {
        assert.equal(nodes.find(n=>n.type==='button' && n.props.children==='继续标注').props.disabled,true)
        assert.equal(calls.length,0)
        continue
      }
      if (mode.startsWith('delete')) {
        assert.ok(nodes.some(n=>n.props?.className==='legacy-rule-label' && n.props.children==='旧规则记录'))
        nodes.find(n=>n.type==='button' && n.props['aria-label']==='删除标注 #5059').props.onClick()
        await setImmediate();nodes=render()
        assert.equal(confirmations.length,1)
        assert.match(confirmations[0],/确认删除标注 #5059/)
        assert.match(confirmations[0],/The visible red vehicle/)
        if (mode==='delete_cancel') {
          assert.equal(calls.length,0)
          assert.equal(states[7].selection_progress.diagnostic_passed,60)
          assert.ok(nodes.some(n=>n.props?.className==='legacy-rule-label'))
          continue
        }
        assert.equal(calls[0].url,'/api/batches/paused-job/samples/5059')
        assert.equal(calls[0].options.method,'DELETE')
        assert.deepEqual(JSON.parse(calls[0].options.body),{run_id:'a'.repeat(24),packet_id:'b'.repeat(24),
          expression:'The visible red vehicle in the left-side view.'})
        assert.equal(nodes.some(n=>n.type==='button' && n.props['aria-label']==='删除标注 #5059'),mode==='delete_failure')
        assert.equal(states[7].items[0].manual_review?.verdict,mode==='delete'?'discarded':undefined)
        assert.equal(states[7].selection_progress.diagnostic_passed,mode==='delete'?59:60)
        assert.equal(states[7].counts.calibration_hold,mode==='delete'?59:60)
        assert.ok(nodes.some(n=>n.type==='progress' && n.props.value===(mode==='delete'?59:60)))
        continue
      }
      nodes.find(n=>n.type==='button' && n.props.children==='继续标注').props.onClick()
      await setImmediate();nodes=render()
      assert.equal(calls[0].url,'/api/batches/paused-job/control')
      assert.deepEqual(JSON.parse(calls[0].options.body),{command:'resume'})
      assert.ok(nodes.some(n=>n.type==='progress' && n.props.value===60 && n.props.max===200))
      assert.ok(nodes.some(n=>n.props?.className==='section-caption' && typeof n.props.children==='string' && /目标 200 条描述$/.test(n.props.children)))
      if (mode==='failure' || mode==='retry') {
        assert.ok(nodes.some(n=>n.props?.role==='alert' && n.props.children.includes('已有标注已保留')))
        if (mode==='retry') {
          nodes.find(n=>n.type==='button' && n.props.children==='继续标注').props.onClick()
          await setImmediate();nodes=render()
          assert.equal(calls.length,2)
          assert.equal(calls[1].url,'/api/batches/paused-job/control')
        }
      }
      assert.ok(calls.every(c=>!c.url.endsWith('/continue') && c.options.method==='POST'))
    } finally {globalThis.fetch=originalFetch;if(originalWindow===undefined)delete globalThis.window;else globalThis.window=originalWindow}
  }
  for (const [outcome, taskCount] of [['success',1], ['success',2], ['error',2], ['pending',2], ['cancel',2]]) {
    const ui = manager(outcome, taskCount)
    try {
      await ui.preview()
      const notice = JSON.stringify(ui.nodes().filter(node => node.type === 'p').map(node => node.props.children))
      assert.match(notice, /仅删除勾选的任务/)
      if (taskCount > 1) assert.match(notice, /其继承样本、必要过程记录和累计调用额度不受影响/)
      assert.doesNotMatch(notice, /将一并删除/)
      ui.button(outcome === 'cancel' ? '取消' : '永久删除').props.onClick()
      if (outcome !== 'cancel') {
        ui.render()
        assert.ok(ui.button('正在删除…')?.props.disabled)
        assert.ok(ui.nodes().some(node => node.type === 'p' && node.props.role === 'status'))
      }
      await setImmediate(); ui.render()
      if (outcome === 'success') {
        assert.equal(ui.nodes().filter(node => node.type === 'input').length, 0)
        assert.equal(ui.nodes().filter(node => node.type === 'dialog').length, 0)
        assert.equal(!!ui.button('管理'), taskCount > 1)
        const target = taskCount > 1 ? ui.button('管理') : ui.nodes().find(node => node.type === 'h2')
        assert.equal(target.props.ref.current.focused, true)
      } else {
        assert.ok(ui.button('完成'))
        assert.ok(ui.nodes().some(node => node.type === 'input' && node.props['aria-label']?.startsWith('选择任务 ') && node.props.checked))
        assert.equal(ui.nodes().some(node => node.type === 'dialog'), outcome !== 'cancel')
      }
    } finally { ui.restore() }
  }
  const concurrent = manager('success', 2, true)
  try {
    const choices = () => concurrent.nodes().filter(node => node.type === 'input' && node.props['aria-label']?.startsWith('选择任务 '))
    assert.equal(choices()[0].props.disabled, false)
    assert.equal(choices()[1].props.disabled, true)
    const all = concurrent.nodes().find(node => node.type === 'input' && node.props['aria-label'] === '选择全部任务')
    assert.equal(all.props.disabled, false)
    all.props.onChange({target:{checked:true}});concurrent.render()
    assert.equal(choices()[0].props.checked, true)
    assert.equal(choices()[1].props.checked, false)
    await concurrent.preview()
    const policy = concurrent.nodes().find(node => node.type === 'input' && !node.props['aria-label'])
    assert.equal(policy.props.checked, false)
    policy.props.onChange({target:{checked:true}}); concurrent.render()
    assert.equal(concurrent.button('永久删除').props.disabled, false)
    concurrent.button('永久删除').props.onClick();await setImmediate();concurrent.render()
    assert.deepEqual(concurrent.deletionRequests.at(-1).body,
      {job_ids:['a'],token:'return-token',return_to_pool:true,action:'delete'})
    assert.equal(choices().length, 0) // Manager closes; the running task remains.
    assert.equal(concurrent.nodes().filter(node => node.type === 'li').length, 1)
  } finally { concurrent.restore() }
  const progress = manager('progress')
  try {
    await progress.preview()
    assert.equal(progress.button('永久删除'),undefined)
    assert.ok(progress.nodes().some(n=>n.type==='progress' && n.props.value===20 && n.props.max===100))
    await new Promise(resolve=>setTimeout(resolve,800)); progress.render()
    assert.ok(progress.button('永久删除'))
    assert.ok(progress.deletionRequests.some(r=>r.url.endsWith('/operation') && !r.body))
  } finally { progress.restore() }
  const outdated = manager('old_backend')
  try {
    await outdated.preview()
    assert.equal(outdated.button('永久删除'),undefined)
    assert.ok(outdated.nodes().some(n=>n.props?.role==='alert' && /后端尚未加载删除进度功能/.test(n.props.children)))
  } finally { outdated.restore() }
  // Rework sends feedback atomically to the original task. A full task must
  // reject the action without first changing the existing review decision.
  const reviewCode = ts.transpileModule(readFileSync(new URL('../src/features/results/SampleReview.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  const flatten = node => Array.isArray(node) ? node.flatMap(flatten) : node && typeof node === 'object'
    ? [node, ...flatten(node.props?.children)] : []
  for (const outcome of ['success','retry_error','already_saved','goal_reached','empty_failed_round']) {
    const originalFetch = globalThis.fetch, calls = [], states = [], component = {}
    let slotIndex = 0, opened = '', receipt = null
    const feedback = { review_id:'feedback-id', revision:'current-revision', verdict:'rejected', notes:'描述不能唯一定位，请补充可确认的区分线索。' }
    const props = { jobId:'source-job', ordinal:12, revision:feedback.revision, canReview:outcome!=='empty_failed_round',
      ...(outcome==='empty_failed_round'?{canRetry:true,retryFailedRunId:'failed-round'}:{}),
      taskGoalReached:outcome==='goal_reached', decision:outcome==='already_saved'?feedback:null, localizationCorrect:false, busy:false, repairMode:true,
      onBusy(value) { props.busy=value }, onReviewed(value) { props.decision=value }, onOpenRetry(id) { opened=id }, onRetry(value) { receipt=value } }
    new Function('require','exports',reviewCode)(name => {
      if (name==='react') return {
        useState(initial) { const index=slotIndex++;if (!(index in states)) states[index]=initial;return [states[index],value=>{states[index]=value}] },
        useRef() { return {current:null} }, useEffect() {},
      }
      if (name.endsWith('/annotationDisplay')) return {readJson,errorMessage,manualReviewLabels:{rejected:'审核不通过'}}
      if (name.endsWith('/navigation')) return {taskLocation,taskUrl,openTask,sampleUrl}
      return require(name)
    }, component)
    const nodes = () => { slotIndex=0;return flatten(component.default(props)) }
    const button = label => nodes().find(n=>n.type==='button' && n.props.children===label)
    globalThis.fetch = async (url, options) => {
      calls.push({url,body:JSON.parse(options.body)})
      const isReview = url.endsWith('/review'), failed = outcome===(isReview?'review_error':'retry_error')
      return new Response(JSON.stringify(failed?{detail:isReview?'反馈保存失败':'该物体累计剩余额度不足'}:isReview?feedback:{job_id:'source-job',started:true,manual_review:feedback}),{status:failed?409:200})
    }
    try {
      assert.equal(button('审核通过'),undefined)
      assert.equal(button('仅保存不通过'),undefined)
      assert.equal(button('废弃该样本'),undefined)
      if (!props.decision) assert.ok(button('保存反馈并让 Agent 返工').props.disabled)
      nodes().find(n=>n.type==='textarea').props.onChange({target:{value:`  ${feedback.notes}  `}})
      assert.equal(button('保存反馈并让 Agent 返工').props.disabled,outcome==='goal_reached')
      if (outcome==='goal_reached') {
        assert.ok(nodes().some(n=>n.props?.className==='notice' && /先手动删除/.test(n.props.children)))
        assert.equal(calls.length,0); assert.equal(props.decision,null); continue
      }
      button('保存反馈并让 Agent 返工').props.onClick()
      assert.ok(props.busy)
      assert.ok(button('正在提交返工…').props.disabled)
      assert.ok(nodes().some(n=>n.props?.role==='status' && /正在保存反馈/.test(n.props.children)))
      await setImmediate()
      assert.equal(props.busy,false)
      assert.deepEqual(calls.map(c=>c.url.split('/').at(-1)),['retry'])
      if (outcome==='retry_error') assert.equal(props.decision,null)
      else assert.equal(props.decision.review_id,feedback.review_id)
      assert.equal(receipt?.ordinal??null,outcome==='retry_error'?null:12)
      if (outcome==='review_error'||outcome==='retry_error') {
        assert.ok(nodes().some(n=>n.props?.role==='alert'))
        assert.equal(button('查看样本返工'),undefined)
      } else {
        assert.deepEqual(calls.at(-1).body,{revision:feedback.revision,notes:feedback.notes,
          ...(outcome==='empty_failed_round'?{retry_failed_run_id:'failed-round'}:{})})
        assert.ok(button('保存反馈并让 Agent 返工').props.disabled)
        button('查看样本返工').props.onClick()
        assert.equal(opened,'source-job')
      }
      if (outcome!=='already_saved') assert.equal(calls[0].body.notes,feedback.notes)
    } finally { globalThis.fetch=originalFetch }
  }
  // Human approval is explicit and stays disabled until localization succeeds.
  {
    const originalFetch=globalThis.fetch, states=[], component={}, requests=[]
    let slot=0, reviewed=null
    const props={jobId:'source-job',ordinal:12,revision:'current-revision',canReview:true,decision:null,
      automaticPassed:false,localizationCorrect:false,busy:false,onBusy(value){props.busy=value},
      onReviewed(value){reviewed=value},onOpenRetry(){}}
    new Function('require','exports',reviewCode)(name=>{
      if(name==='react')return {useState(initial){const i=slot++;if(!(i in states))states[i]=initial;return [states[i],v=>states[i]=v]},useRef(){return {current:null}},useEffect(){}}
      if(name.endsWith('/annotationDisplay'))return {readJson,errorMessage,manualReviewLabels:{accepted:'审核通过'}}
      if(name.endsWith('/navigation'))return {sampleUrl}
      return require(name)
    },component)
    const nodes=()=>{slot=0;return flatten(component.default(props))}
    const approve=()=>nodes().find(n=>n.type==='button'&&n.props.children==='人工审批通过并纳入可用集')
    globalThis.fetch=async(url,options)=>{
      requests.push({url,body:JSON.parse(options.body)})
      return new Response(JSON.stringify({verdict:'accepted',approval_policy:'localized-human-approval-v1'}))
    }
    try{
      assert.ok(approve().props.disabled)
      props.localizationCorrect=true
      assert.equal(approve().props.disabled,false)
      approve().props.onClick();await setImmediate()
      assert.deepEqual(requests,[{url:'/api/batches/source-job/samples/12/review',body:{revision:'current-revision',verdict:'accepted',notes:'',approve_failed:true}}])
      assert.equal(reviewed.approval_policy,'localized-human-approval-v1')
    }finally{globalThis.fetch=originalFetch}
  }
  const reworkCode=ts.transpileModule(readFileSync(new URL('../src/features/results/ReworkProgress.tsx',import.meta.url),'utf8'),{
    compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX},
  }).outputText
  const rework={}
  new Function('require','exports',reworkCode)(name=>name.endsWith('/annotationDisplay')?{phases,reasonLabel}:name.endsWith('/navigation')?{taskUrl}:require(name),rework)
  const base={ordinal:12,status:'running',phase:'generate',model_pending:true,automatic_passed:false,calls:2,call_budget:12,run_id:'run',packet_id:'packet'}
  for(const row of [
    {status:'pending',phase:null,task:'ready',text:'等待启动原任务'},
    {status:'pending',task:'queued',text:'排队等待执行'},
    {status:'preparing',preparation_stage:'validating_scene',text:'正在准备返工证据'},
    {text:'生成目标描述'},
    {phase:'ground_permuted',text:'换一种输入顺序再次检查'},
    {task:'paused',text:'返工已暂停'},
    {status:'needs_evidence',reason:'target_spatially_ambiguous',text:'返工未通过'},
    {status:'calibration_hold',automatic_passed:true,text:'返工完成'},
  ]){
    const props={items:[{...base,...row}],task:{status:row.task??'running',executor_active:!row.task,control:null,reason:null},
      preparationLabel:()=> '校验点云与图片',onResult(){}}
    const html=renderToStaticMarkup(require('react').createElement(rework.default,props))
    assert.ok(html.includes(row.text),html)
    assert.match(html,/本次运行模型调用 2 \/ 12 次/)
    if(row.task||row.status==='needs_evidence'||row.automatic_passed)assert.doesNotMatch(html,/模型请求处理中/)
    if(row.phase==='ground_permuted')assert.match(html,/自动核验/)
  }
  const failure=renderToStaticMarkup(require('react').createElement(rework.default,{
    items:[{...base,status:'needs_evidence',reason:'description_validation_failed',previous_expression:'The original pickup.',
      feedback:'请补充区分信息。',attempt_limit:1,attempts:[{expression:'The silver pickup.',errors:['declared_expression_not_unique'],native_remaining:6,image_remaining:7,ground_rounds:0}]}],
    task:{status:'completed',executor_active:false,control:null,reason:null},
  }))
  for(const text of ['The original pickup.','The silver pickup.','请补充区分信息。','6','7','不能唯一定位','最多 1 版草稿'])assert.ok(failure.includes(text),text)
  const mismatch=renderToStaticMarkup(require('react').createElement(rework.default,{
    items:[{...base,status:'needs_evidence',reason:'description_validation_failed',attempts:[{
      expression:'The sedan near the red car in the left-side view.',
      errors:['text_claims_disagree_with_declared_predicates','unsupported_text_claim','image_visibility_scope_not_explicit',
        'parsed_expression_not_symbolically_unique','image_expression_not_unique','image_order_scope_not_explicit',
        'failed_draft_text_repeated','scene_grounding_unsupported_claims','scene_grounding_ambiguous',
        'scene_grounding_not_unique_target','scene_grounding_invalid_candidate_ids','scene_grounding_identity_unverified'],
      unsupported_claims:['near the red car'],native_remaining:31,image_remaining:70,ground_rounds:1}]}],
    task:{status:'completed',executor_active:false,control:null,reason:null},
  }))
  assert.doesNotMatch(mismatch,/本次执行已结束或遇到问题/)
  for(const text of ['同步改写','比较范围','3D 竞争对象','图像竞争对象','尚无证据支持的文本片段','near the red car','31','70'])assert.ok(mismatch.includes(text),text)
})

test('sample discard is available outside localization and protects the displayed version', async () => {
  const compiled = ts.transpileModule(readFileSync(new URL('../src/features/results/TaskSamples.tsx', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  const flatten = node => Array.isArray(node) ? node.flatMap(flatten) : node && typeof node === 'object'
    ? [node, ...flatten(node.props?.children)] : []
  // Browse a filtered, paginated list, then create a fresh component just as
  // a document reload would. Real handlers/effects must preserve the selection.
  {
    const original={window:globalThis.window,document:globalThis.document,fetch:globalThis.fetch,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout}
    const storage=new Map(),job={job_id:'remembered-job',status:'completed',counts:{needs_evidence:60}}
    const samples=Array.from({length:61},(_,i)=>({ordinal:100+i,category:'vehicle',expression:`Vehicle ${i}.`,run_id:`run-${i}`,packet_id:`packet-${i}`,
      revision:`revision-${i}`,status:i?'needs_evidence':'calibration_hold',automatic_passed:!i,can_review:true,manual_review:null}))
    const list={job_id:job.job_id,status:'completed',samples,review_summary:{unreviewed:61,usable:1,accepted:0,rejected:0,discarded:0}}
    const requests=[]
    function mount(hash){
      let index=0,serial=0,tree
      const slots=[],effects=[],pending=[],listeners=new Map(),timers=new Map(),component={}
      const listDom={scrollTop:0},area={scrollIntoView(){}}
      globalThis.window={location:{hash},scrollY:0,
        sessionStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
        history:{replaceState(_state,_title,url){globalThis.window.location.hash=url}},
        scrollTo({top}){globalThis.window.scrollY=top},
        addEventListener:(n,f)=>listeners.set(n,f),removeEventListener:n=>listeners.delete(n)}
      globalThis.document={hidden:false,addEventListener(){},removeEventListener(){}}
      globalThis.setTimeout=f=>{timers.set(++serial,f);return serial};globalThis.clearTimeout=id=>timers.delete(id)
      globalThis.fetch=async(url,options)=>{
        requests.push({url,method:options?.method})
        const ordinal=Number(url.match(/samples\/(\d+)/)?.[1]),sample=samples.find(s=>s.ordinal===ordinal)
        const result=url==='/api/batches'?{jobs:[job]}:url.endsWith('/samples')?list:url.endsWith('/rework')?{job_id:job.job_id,ordinal,item:null}:
          {...sample,job_id:job.job_id,ordinal,review:url.endsWith('true'),selected_candidate_id:'C001',confirmation:null}
        return new Response(JSON.stringify(result))
      }
      const same=(a,b)=>a&&b&&a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]))
      new Function('require','exports',compiled)(name=>{
        if(name==='react')return {
          useState(initial){const i=index++;if(!(i in slots))slots[i]=typeof initial==='function'?initial():initial;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v}]},
          useRef(initial){const i=index++;if(!(i in slots))slots[i]={current:initial};return slots[i]},useMemo(fn){return fn()},
          useEffect(fn,deps){const i=index++;if(!same(effects[i]?.deps,deps))pending.push(()=>{effects[i]?.cleanup?.();effects[i]={deps,cleanup:fn()}})},
        }
        if(name==='./resultViewState')return resultViewState
        if(name.endsWith('/annotationDisplay'))return {readJson,readPollingJson,errorMessage,automaticCheckState,automaticCheckLabels,categoryLabel,objectName:id=>id,manualReviewLabels:{}}
        if(name.endsWith('/taskName'))return {taskName:()=> '原任务'}
        if(name.endsWith('/cameraViews'))return {cameraViewLabel:()=> '前方'}
        if(name.endsWith('/navigation'))return {sampleUrl}
        if(name==='./TaskScene'||name==='./SampleReview'||name==='./ReworkProgress')return {default:()=>null}
        return require(name)
      },component)
      const render=()=>{
        index=0;tree=component.default();const nodes=flatten(tree)
        for(const n of nodes)if(n.props?.ref)n.props.ref.current=n.type==='ol'?listDom:area
        pending.splice(0).forEach(fn=>fn());return nodes
      }
      return {render,listDom,listeners,async settle(){for(let i=0;i<7;i++){render();await setImmediate()}return render()},
        close(){effects.forEach(e=>e?.cleanup?.())}}
    }
    let ui
    try{
      ui=mount('#results');let nodes=await ui.settle()
      nodes.find(n=>n.props?.['aria-label']==='自动核验筛选').props.onChange({target:{value:'failed'}})
      nodes=ui.render();nodes.find(n=>n.props?.['aria-label']==='下一页样本').props.onClick()
      nodes=ui.render();nodes.find(n=>n.type==='button'&&flatten(n.props.children).some(c=>c.type==='strong'&&c.props.children?.includes(145))).props.onClick()
      nodes=await ui.settle();ui.listDom.scrollTop=365
      nodes.find(n=>n.type==='ol'&&n.props.onScroll).props.onScroll({currentTarget:ui.listDom})
      globalThis.window.scrollY=420;ui.listeners.get('pagehide')()
      const href=globalThis.window.location.hash
      assert.equal(href,sampleUrl(job.job_id,145))
      const saved=JSON.parse(storage.get(resultViewState.resultViewKey))
      assert.deepEqual([saved.ordinal,saved.filter,saved.page,saved.listScroll,saved.pageScroll],[145,'failed',1,365,420])
      ui.close();ui=mount(href);nodes=await ui.settle()
      assert.equal(nodes.find(n=>n.props?.['aria-label']==='自动核验筛选').props.value,'failed')
      assert.ok(nodes.find(n=>n.type==='button'&&n.props['aria-current']==='true'&&flatten(n.props.children).some(c=>c.type==='strong'&&c.props.children?.includes(145))))
      assert.equal(nodes.find(n=>n.type==='ol'&&n.props.onScroll).props.start,31)
      assert.equal(ui.listDom.scrollTop,365);assert.equal(globalThis.window.scrollY,420)
      assert.equal(resultViewState.initialResultView(sampleUrl(job.job_id,102)).restored,false)
      assert.equal(resultViewState.initialResultView('#results').ordinal,145)
      assert.ok(requests.every(r=>!r.method)) // Restoring never reviews or starts work.
      ui.close();ui=null
      storage.set(resultViewState.resultViewKey,'broken JSON')
      assert.equal(resultViewState.initialResultView('#results').filter,'all')
      globalThis.window.sessionStorage={getItem(){throw Error('blocked')},setItem(){throw Error('blocked')}}
      assert.equal(resultViewState.initialResultView('#results').restored,false)
      assert.doesNotThrow(()=>resultViewState.saveResultView(saved))
    }finally{ui?.close();Object.assign(globalThis,original)}
  }
  for (const mode of ['ordinary','review','failed_check','stale','unfinished','discarded','navigation']) {
    const originalFetch=globalThis.fetch, originalWindow=globalThis.window, calls=[], component={}
    globalThis.window={location:{hash:''}}
    const job={job_id:'source-job',created_at:0,status:'completed',counts:{calibration_hold:1}}
    const decision={review_id:'discard-receipt',revision:'current-revision',verdict:'discarded',notes:'从标注结果列表删除。'}
    const sample={ordinal:12,category:'vehicle',expression:'The white van.',run_id:'run-id',packet_id:'packet-id',
      revision:'current-revision',status:'calibration_hold',automatic_passed:mode!=='failed_check',reviewed:false,
      can_review:mode!=='unfinished',manual_review:mode==='discarded'?decision:null}
    const list={job_id:job.job_id,created_at:0,status:'completed',total_objects:1,samples:[sample],
      review_summary:{unreviewed:1,accepted:0,rejected:0,discarded:0,usable:1}}
    if(mode==='navigation')list.samples.push(...Array.from({length:35},(_,i)=>({...sample,ordinal:1000+i})),{...sample,ordinal:4104})
    const states=[mode==='review',[job],job.job_id,true,list,12,null,'',null,false]
    let index=0
    new Function('require','exports',compiled)(name => {
      if (name==='react') return {
        useState(initial) { const slot=index++;if (!(slot in states)) states[slot]=initial;return [states[slot],value=>{states[slot]=typeof value==='function'?value(states[slot]):value}] },
        useRef(initial) { return {current:initial} }, useEffect() {}, useMemo(fn) { return fn() },
      }
      if (name.endsWith('/annotationDisplay')) return {readJson,errorMessage,automaticCheckState,automaticCheckLabels,categoryLabel,
        objectName:value=>value,manualReviewLabels:{discarded:'已废弃'}}
      if (name.endsWith('/taskName')) return {taskName:()=> '原任务'}
      if (name.endsWith('/cameraViews')) return {cameraViewLabel:()=> '前方'}
      if (name.endsWith('/navigation')) return {taskLocation,taskUrl,openTask,sampleUrl}
      if(name==='./resultViewState')return resultViewState
      if (name==='./TaskScene'||name==='./SampleReview'||name==='./ReworkProgress') return {default:()=>null}
      return require(name)
    },component)
    const nodes=()=>{index=0;return flatten(component.default())}
    const discard=()=>nodes().find(n=>n.type==='button'&&n.props['aria-label']==='废弃样本 #12')
    globalThis.fetch=async (url,options)=>{
      calls.push({url,method:options.method,body:JSON.parse(options.body)})
      return new Response(JSON.stringify(mode==='stale'?{detail:'描述版本已变化，请刷新。'}:decision),{status:mode==='stale'?409:200})
    }
    try {
      if(mode==='navigation'){
        const input=()=>nodes().find(n=>n.type==='input'&&n.props.id==='review-sample-id')
        const submit=()=>nodes().find(n=>n.type==='form'&&flatten(n.props.children).some(c=>c.props?.id==='review-sample-id')).props.onSubmit({preventDefault(){}})
        input().props.onChange({target:{value:'#4104'}});submit()
        assert.equal(states[5],4104)
        assert.equal(globalThis.window.location.hash,sampleUrl(job.job_id,4104))
        assert.ok(nodes().some(n=>n.type==='h2'&&n.props.children?.includes(4104)))
        for(const value of ['#9999','1.5','-2','']){
          input().props.onChange({target:{value}});submit()
          assert.equal(states[5],4104) // Invalid IDs never become list indexes.
          assert.ok(nodes().some(n=>n.props?.className==='error'&&/样本|编号/.test(n.props.children)))
        }
        nodes().find(n=>n.type==='button'&&n.props.children==='填写反馈／Agent 返工').props.onClick()
        nodes().find(n=>n.props?.onRetry).props.onRetry({job_id:job.job_id,ordinal:4104,started:true})
        // Selection can change without changing the existing hash. Clicking
        // the SAME receipt URL must still select the intended object.
        assert.ok(nodes().find(n=>n.type==='button'&&n.props['aria-current']==='true'&&flatten(n.props.children).some(c=>c.type==='strong'&&c.props.children?.includes(4104))))
        nodes().find(n=>n.type==='button'&&n.props['aria-label']==='上一页样本').props.onClick()
        nodes().find(n=>n.type==='button'&&flatten(n.props.children).some(c=>c.type==='strong'&&c.props.children?.includes(12))).props.onClick()
        assert.equal(states[5],12)
        const link=nodes().find(n=>n.type==='a'&&n.props.children==='查看该样本')
        let prevented=false
        link.props.onClick({button:0,preventDefault(){prevented=true}})
        assert.ok(prevented);assert.equal(states[5],4104)
        assert.equal(calls.length,0)
        states[4]={...list,samples:[sample]}
        assert.ok(nodes().some(n=>n.type==='h2'&&n.props.children?.includes(4104))) // Missing retry result stays pinned.
        assert.equal(nodes().some(n=>n.type==='p'&&n.props.className==='expression'),false)
        continue
      }
      assert.ok(discard())
      assert.equal(discard().props.disabled,['unfinished','discarded'].includes(mode))
      assert.equal(nodes().some(n=>n.type==='button'&&n.props.children==='仅保存不通过'),false)
      if (mode==='ordinary'||mode==='review') {
        const entry=nodes().find(n=>n.type==='button'&&n.props.children==='填写反馈／Agent 返工')
        assert.ok(entry && !entry.props.disabled)
        assert.equal(nodes().some(n=>n.type==='button'&&n.props.children==='开始人工审批'),false)
        entry.props.onClick()
        const forms=nodes().filter(n=>n.props?.onRetry)
        assert.equal(forms.length,1) // No duplicate form while localization is open.
        assert.equal(forms[0].props.repairMode,true)
        assert.equal(forms[0].props.automaticPassed,true)
        assert.equal(forms[0].props.canRetry,true)
        assert.equal(forms[0].props.revision,sample.revision)
        assert.equal(calls.length,0) // Opening feedback never changes the sample.
        forms[0].props.onClose()
        assert.equal(nodes().some(n=>n.props?.repairMode),false)
      }
      if(mode==='failed_check'){
        nodes().find(n=>n.type==='button'&&n.props.children==='开始人工审批').props.onClick()
        assert.equal(states[0],true)
        assert.ok(nodes().some(n=>n.props?.role==='status'&&/尚未保存审批/.test(n.props.children)))
        assert.ok(nodes().find(n=>n.type==='button'&&n.props.children==='确认所选物体').props.disabled)
        assert.equal(calls.length,0) // Entering the flow does not silently approve.
        states[0]=false
        nodes().find(n=>n.type==='button'&&n.props.children==='填写反馈／Agent 返工').props.onClick()
        const review=nodes().find(n=>n.props?.onRetry)
        review.props.onRetry({job_id:job.job_id,ordinal:12,started:true})
        assert.equal(globalThis.window.location.hash,sampleUrl(job.job_id,12))
        assert.equal(calls.length,0) // Navigation never starts the task twice.
      }
      if (discard().props.disabled) { discard().props.onClick();assert.equal(calls.length,0);continue }
      discard().props.onClick();await setImmediate()
      assert.deepEqual(calls,[{url:'/api/batches/source-job/samples/12',method:'DELETE',
        body:{run_id:'run-id',packet_id:'packet-id',expression:'The white van.'}}])
      assert.equal(states[0],mode==='review') // Does not enter localization first.
      if (mode==='stale') {
        assert.equal(states[4].samples[0].manual_review,null)
        assert.equal(states[4].review_summary.usable,1)
        assert.ok(nodes().some(n=>n.props?.role==='alert'&&/版本已变化/.test(n.props.children)))
      } else {
        assert.equal(states[4].samples[0].manual_review.verdict,'discarded')
        assert.equal(states[4].review_summary.usable,0)
        assert.equal(states[4].review_summary.discarded,1)
        assert.equal(discard().props.children,'已废弃')
        assert.ok(discard().props.disabled)
      }
    } finally { globalThis.fetch=originalFetch;globalThis.window=originalWindow }
  }
  // A pre-save poll may return after the POST finishes and before React's
  // effect cleanup. It must never undo the saved decision or its counters.
  for(const outcome of ['success','error']){
    const saved={fetch:globalThis.fetch,window:globalThis.window,document:globalThis.document,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout}
    const timers=new Map(),listeners=new Map(),effects=[],pending=[],component={},requests=[]
    let index=0,serial=0,resolveList,resolveApproval,tree
    const job={job_id:'approval-job',created_at:0,status:'completed',counts:{needs_evidence:1}}
    const sample={ordinal:12,category:'vehicle',expression:'The silver pickup.',run_id:'run',packet_id:'packet',
      revision:'revision',status:'needs_evidence',automatic_passed:false,can_review:true,manual_review:null}
    const oldList={job_id:job.job_id,created_at:0,status:'completed',total_objects:1,samples:[sample],
      review_summary:{unreviewed:1,accepted:0,rejected:0,discarded:0,usable:0}}
    const confirmation={correct:true,selected_candidate_id:'C001',expected_candidate_id:'C001'}
    const scene={job_id:job.job_id,ordinal:12,expression:sample.expression,run_id:'run',revision:'revision',
      review:true,can_confirm:true,confirmation,selected_candidate_id:null}
    const states=[true,[job],job.job_id,true,oldList,12,scene,'C001',confirmation,false]
    globalThis.window={location:{hash:''},addEventListener(n,fn){listeners.set('w:'+n,fn)},removeEventListener(n){listeners.delete('w:'+n)}}
    globalThis.document={hidden:false,addEventListener(n,fn){listeners.set(n,fn)},removeEventListener(n){listeners.delete(n)}}
    globalThis.setTimeout=(fn,ms)=>{const id=++serial;timers.set(id,{fn,ms});return id}
    globalThis.clearTimeout=id=>timers.delete(id)
    const same=(a,b)=>a&&b&&a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]))
    new Function('require','exports',compiled)(name=>{
      if(name==='react')return {
        useState(initial){const i=index++;if(!(i in states))states[i]=typeof initial==='function'?initial():initial;return [states[i],v=>{states[i]=typeof v==='function'?v(states[i]):v}]},
        useRef(initial){const i=index++;if(!(i in states))states[i]={current:initial};return states[i]},
        useMemo(fn){return fn()},
        useEffect(fn,deps){const i=index++;if(!same(effects[i]?.deps,deps))pending.push(()=>{effects[i]?.cleanup?.();effects[i]={deps,cleanup:fn()}})},
      }
      if(name.endsWith('/annotationDisplay'))return {readJson,readPollingJson,errorMessage,automaticCheckState,automaticCheckLabels,categoryLabel,objectName:id=>id,manualReviewLabels:{accepted:'审核通过'}}
      if(name.endsWith('/taskName'))return {taskName:()=> '原任务'}
      if(name.endsWith('/cameraViews'))return {cameraViewLabel:()=> '前方'}
      if(name.endsWith('/navigation'))return {sampleUrl}
      if(name==='./resultViewState')return resultViewState
      if(name==='./TaskScene'||name==='./SampleReview'||name==='./ReworkProgress')return {default:()=>null}
      return require(name)
    },component)
    globalThis.fetch=async(url,options)=>{
      requests.push({url,method:options?.method,body:options?.body?JSON.parse(options.body):undefined})
      if(url.endsWith('/review'))return await new Promise(resolve=>resolveApproval=resolve)
      if(url.endsWith('/samples'))return await new Promise(resolve=>resolveList=resolve)
      return new Response(JSON.stringify(url==='/api/batches'?{jobs:[job]}:url.includes('/scene?')?scene:{job_id:job.job_id,ordinal:12,item:null}))
    }
    const render=()=>{index=0;tree=component.default();pending.splice(0).forEach(fn=>fn());return flatten(tree)}
    try{
      render();await setImmediate()
      const submit=render().find(n=>n.type==='button'&&n.props.children==='审批通过并纳入可用集')
      assert.ok(submit&&!submit.props.disabled)
      submit.props.onClick();submit.props.onClick()
      assert.equal(requests.filter(r=>r.url.endsWith('/review')).length,1)
      assert.ok(render().some(n=>n.type==='button'&&n.props.children==='正在保存审批…'&&n.props.disabled))
      const decision={job_id:job.job_id,ordinal:12,run_id:'run',revision:'revision',verdict:'accepted',
        approval_policy:'localized-human-approval-v1',notes:''}
      resolveApproval(new Response(JSON.stringify(outcome==='error'?{detail:'保存失败，请重试。'}:decision),{status:outcome==='error'?409:200}))
      await setImmediate()
      assert.equal(states[4].review_summary.usable,outcome==='success'?1:0)
      resolveList(new Response(JSON.stringify(oldList)));await setImmediate()
      assert.equal(states[4].review_summary.usable,outcome==='success'?1:0)
      const visible=render()
      if(outcome==='success'){
        assert.equal(states[4].review_summary.accepted,1)
        assert.equal(states[4].review_summary.unreviewed,0)
        assert.equal(states[4].samples[0].manual_approved,true)
        assert.ok(visible.some(n=>n.props?.role==='status'&&/已纳入当前可用集/.test(n.props.children)))
        assert.equal(visible.some(n=>n.type==='button'&&n.props.children==='审批通过并纳入可用集'),false)
      }else{
        assert.equal(states[4].samples[0].manual_review,null)
        assert.ok(visible.some(n=>n.props?.role==='alert'&&/保存失败/.test(n.props.children)))
      }
    }finally{
      effects.forEach(e=>e?.cleanup?.());await setImmediate()
      Object.assign(globalThis,saved)
    }
  }
})

test('polling bounds stalled bodies, releases timers and propagates cancellation', async () => {
  const saved={fetch:globalThis.fetch,setTimeout:globalThis.setTimeout,clearTimeout:globalThis.clearTimeout}
  const timers=new Map();let serial=0, readSignal, reading=false
  globalThis.setTimeout=fn=>{const id=++serial;timers.set(id,fn);return id}
  globalThis.clearTimeout=id=>timers.delete(id)
  globalThis.fetch=async (url,options)=>{
    readSignal=options.signal
    assert.equal(options.cache,'no-store')
    return {ok:true,text:()=>{reading=true;return new Promise((resolve,reject)=>{
      options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true})
    })}}
  }
  try {
    const first=readPollingJson('/stalled',new AbortController().signal)
    const timeoutCheck=assert.rejects(first,/读取超时/)
    await setImmediate();assert.equal(reading,true)
    timers.values().next().value();await timeoutCheck
    assert.equal(readSignal.aborted,true);assert.equal(timers.size,0)
    const owner=new AbortController(),second=readPollingJson('/cancelled',owner.signal)
    const cancelCheck=assert.rejects(second,{name:'AbortError'})
    await setImmediate();owner.abort();await cancelCheck
    assert.equal(readSignal.aborted,true);assert.equal(timers.size,0)
  } finally {Object.assign(globalThis,saved)}
})
