import { localeTag, setLocale, t, useLocale } from './shared/locale.ts'
import { useEffect, useState } from 'react'
import './App.css'
import './workspace.css'
import TaskSamples from './features/results/TaskSamples'
import BatchWorkspace from './features/tasks/BatchWorkspace'
import ResearchWorkspace from './features/research/ResearchWorkspace'
import { currentView, openResult, pages } from './navigation'

export default function App() {
  const locale = useLocale()
  const [view, setView] = useState(currentView)
  const [location,setLocation]=useState(window.location.hash)
  const blindReview=view==='research'&&new URLSearchParams(location.split('?')[1]).has('review')
  useEffect(() => { document.documentElement.lang = localeTag() }, [locale])
  useEffect(() => {
    const change = () => { setView(currentView()); setLocation(window.location.hash); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  return <div className="research-app">
    <header className="research-header">
      <div className="workspace-container research-header-inner">
        {blindReview?<span className="research-brand">Waymo 3DVG</span>:<a className="research-brand" href="#annotate">Waymo 3DVG</a>}
        {!blindReview&&<nav className="research-navigation" aria-label={t("主导航")}>{pages.map(p => <a key={p.id} href={'#' + p.id} aria-current={view === p.id ? 'page' : undefined}>{t(p.title)}</a>)}</nav>}
        <button className="language-toggle" type="button" lang={locale === 'zh' ? 'en' : 'zh-CN'} onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}>{locale === 'zh' ? 'English' : '中文'}</button>
      </div>
    </header>
    <main className="workspace-container">
      <header className="research-title">
        <h1>{blindReview?t("独立人工盲审"):t(pages.find(p => p.id === view)!.title)}</h1>
      </header>
      {!blindReview&&<div hidden={view !== 'annotate'}><BatchWorkspace active={view === 'annotate'} onOpenPacket={openResult} /></div>}
      {view === 'results' && <TaskSamples />}
      {view === 'research' && <ResearchWorkspace key={location} />}
    </main>
  </div>
}
