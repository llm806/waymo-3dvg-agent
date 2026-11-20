import { t, useLocale } from '../shared/locale.ts'
import { useEffect, useId, useRef, useState } from 'react'

import { terms } from '../shared/helpContent'

export function TermHelp({ term, label }: { term: keyof typeof terms; label?: string }) {
  useLocale()
  const [name, explanation] = terms[term].map(text=>t(text))
  return <HelpHint label={label ?? name} detail={explanation} />
}

export function HelpHint({ label, detail, iconOnly = false }: { label: string; detail: string; iconOnly?: boolean }) {
  const id = useId()
  const anchor = useRef<HTMLSpanElement>(null)
  const [hint, setHint] = useState<{ mode: 'hover' | 'focus' | 'pinned'; offset: number; above: boolean } | null>(null)
  const open = hint !== null

  useEffect(() => {
    if (!open) return
    const dismiss = (event: KeyboardEvent) => { if (event.key === 'Escape') setHint(null) }
    const outside = (event: PointerEvent) => { if (!anchor.current?.contains(event.target as Node)) setHint(null) }
    document.addEventListener('keydown', dismiss)
    document.addEventListener('pointerdown', outside)
    return () => {
      document.removeEventListener('keydown', dismiss)
      document.removeEventListener('pointerdown', outside)
    }
  }, [open])

  function show(mode: 'hover' | 'focus' | 'pinned') {
    const rect = anchor.current?.getBoundingClientRect()
    if (!rect) return
    const viewport = document.documentElement.clientWidth
    const width = Math.min(304, viewport - 24)
    const offset = Math.max(12, Math.min(rect.left, viewport - width - 12)) - rect.left
    const above = window.innerHeight - rect.bottom < Math.min(180, rect.top)
    setHint({ mode, offset, above })
  }

  return <span ref={anchor} className="term-help"
    onPointerEnter={event => { if (event.pointerType !== 'touch' && !hint) show('hover') }}
    onPointerLeave={() => { if (hint?.mode === 'hover') setHint(null) }}>
    <button className={`term-trigger${iconOnly ? ' term-icon-trigger' : ''}`} type="button"
      aria-label={iconOnly ? t("解释：") + label : undefined} aria-describedby={id} aria-expanded={open}
      onFocus={event => { if (event.currentTarget.matches(':focus-visible')) show('focus') }}
      onBlur={() => { if (hint?.mode !== 'hover') setHint(null) }}
      onClick={() => { if (hint?.mode === 'pinned') setHint(null); else show('pinned') }}>
      {!iconOnly && label}<span className="term-info-icon" aria-hidden="true">ⓘ</span>
    </button>
    <span id={id} hidden={!open} className={`term-explanation${hint?.above ? ' above' : ''}`} style={{ left: hint?.offset }} role="tooltip">{detail}</span>
  </span>
}

