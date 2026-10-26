import { useId, useState } from 'react'

import { terms } from '../shared/helpContent'

export function TermHelp({ term, label }: { term: keyof typeof terms; label?: string }) {
  const [name, explanation] = terms[term]
  const [open, setOpen] = useState(false)
  const [alignRight, setAlignRight] = useState(false)
  const id = useId()
  return <span className="term-help"><button className="term-trigger" type="button" title={explanation} aria-label={'解释：' + (label ?? name)} aria-expanded={open} aria-controls={id} aria-describedby={id} onClick={e => { setAlignRight(e.currentTarget.getBoundingClientRect().left + 300 > window.innerWidth - 20); setOpen(v => !v) }} onBlur={() => setOpen(false)} onKeyDown={e => { if (e.key === 'Escape') setOpen(false) }}>{label ?? name}<span aria-hidden="true"> ⓘ</span></button><span id={id} hidden={!open} className={'term-explanation' + (alignRight ? ' align-right' : '')} role="note">{explanation}</span></span>
}

