import { useSyncExternalStore } from 'react'
import { english, serviceTemplates } from './messages.en.ts'

export type Locale = 'zh' | 'en'
export type InterfaceMessage = string | [message: string, ...values: (string | number | null | undefined)[]]
const storageKey = 'waymo-3dvg-language'
const listeners = new Set<() => void>()

function preferredLocale(): Locale {
  try { return typeof window !== 'undefined' && window.localStorage.getItem(storageKey) === 'en' ? 'en' : 'zh' }
  catch { return 'zh' }
}

let locale = preferredLocale()
const getLocale = () => locale
function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export function useLocale() {
  return useSyncExternalStore(subscribe, getLocale, getLocale)
}

export function setLocale(next: Locale) {
  if (locale === next) return
  locale = next
  try { if (typeof window !== 'undefined') window.localStorage.setItem(storageKey, next) }
  catch { /* The current tab can still switch languages. */ }
  for (const listener of listeners) listener()
}

export function localeTag() {
  return locale === 'en' ? 'en-US' : 'zh-CN'
}

export function t(message: string, ...values: unknown[]) {
  const key = message.trim()
  const translated = locale === 'en' && key && Object.hasOwn(english, key)
    ? message.replace(key, () => english[key])
    : message
  return values.length ? translated.replace(/\{(\d+)\}/g, (match, index) =>
    Number(index) < values.length ? String(values[Number(index)]) : match) : translated
}

export const defaultEvaluationTitle = '标注质量评估'

export function evaluationTitle(title: string = defaultEvaluationTitle) {
  // Older evaluations stored the default title in the language used at creation.
  return title === defaultEvaluationTitle || title === english[defaultEvaluationTitle]
    ? t(defaultEvaluationTitle) : title
}

const servicePatterns = serviceTemplates.map(message => ({
  message,
  pattern: new RegExp('^' + message.split(/\{\d+\}/g).map(part =>
    part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('(.+?)') + '$'),
}))

export function serviceText(value: InterfaceMessage | null | undefined): string {
  if (Array.isArray(value)) {
    const [message, ...values] = value
    return t(message, ...values.map(value => typeof value === 'string' ? serviceText(value) : value))
  }
  if (!value || locale === 'zh') return value ?? ''
  const direct = t(value)
  if (direct !== value) return direct
  for (const {message, pattern} of servicePatterns) {
    const match = pattern.exec(value)
    if (match) return t(message, ...match.slice(1))
  }
  return value
}

// Status lookups remain keyed by the API values. Translate their labels when read.
export function localizeLabels<T extends Record<string, string>>(labels: T): T {
  return new Proxy(labels, {
    get(target, key, receiver) {
      const value = Reflect.get(target, key, receiver)
      return typeof value === 'string' ? t(value) : value
    },
  })
}
