import { useEffect, useState } from 'react'
import './App.css'
import TaskSamples from './features/results/TaskSamples'
import BatchWorkspace from './features/tasks/BatchWorkspace'
import { currentView, openResult, pages } from './navigation'

export default function App() {
  const [view, setView] = useState(currentView)
  useEffect(() => {
    const change = () => { setView(currentView()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  return <div className="research-app">
    <header className="research-header"><a className="research-brand" href="#annotate"><span className="brand-symbol" aria-hidden="true">3D</span><span>Waymo <strong>标注工作台</strong></span></a><span className="workspace-label"><i aria-hidden="true" />本地工作区</span></header>
    <main className={view==='results'?'results-page':undefined}>
      <nav className="research-navigation" aria-label="主导航">{pages.map(p => <a key={p.id} href={'#' + p.id} aria-current={view === p.id ? 'page' : undefined}>{p.title}</a>)}</nav>
      <header className="research-title"><h1>{pages.find(p => p.id === view)!.title}</h1>{view === 'annotate' && <p>填写期望标注条数，自动选择物体并生成描述。点云、图像与处理结果随进度同步展示。</p>}</header>
      <div hidden={view !== 'annotate'}><BatchWorkspace active={view === 'annotate'} onOpenPacket={openResult} /></div>
      {view === 'results' && <TaskSamples />}
      <footer className="research-footer">样本与人工检查记录按标注任务保存。</footer>
    </main>
  </div>
}
