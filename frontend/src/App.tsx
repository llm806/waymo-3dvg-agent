import { useEffect, useState } from 'react'
import './App.css'
import './workspace.css'
import TaskSamples from './features/results/TaskSamples'
import BatchWorkspace from './features/tasks/BatchWorkspace'
import ResearchWorkspace from './features/research/ResearchWorkspace'
import { currentView, openResult, pages } from './navigation'

export default function App() {
  const [view, setView] = useState(currentView)
  const [location,setLocation]=useState(window.location.hash)
  const blindReview=view==='research'&&new URLSearchParams(location.split('?')[1]).has('review')
  useEffect(() => {
    const change = () => { setView(currentView()); setLocation(window.location.hash); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  return <div className="research-app">
    <header className="research-header"><a className="research-brand" href={blindReview?location:'#annotate'}><span className="brand-symbol" aria-hidden="true">3D</span><span>Waymo <strong>{blindReview?'独立盲审':'标注工作台'}</strong></span></a>{!blindReview&&<nav className="research-navigation" aria-label="主导航">{pages.map(p => <a key={p.id} href={'#' + p.id} aria-current={view === p.id ? 'page' : undefined}>{p.title}</a>)}</nav>}<span className="workspace-label"><i aria-hidden="true" />本地工作区</span></header>
    <main className={view==='results'?'results-page':undefined}>
      <header className="research-title"><h1>{pages.find(p => p.id === view)!.title}</h1><span>{view==='annotate'?'创建任务 · 查看进度':view==='research'?'同任务对照 · 独立盲审 · 论文证据':'查看描述 · 核对目标'}</span></header>
      {!blindReview&&<div hidden={view !== 'annotate'}><BatchWorkspace active={view === 'annotate'} onOpenPacket={openResult} /></div>}
      {view === 'results' && <TaskSamples />}
      {view === 'research' && <ResearchWorkspace key={location} />}
    </main>
  </div>
}
