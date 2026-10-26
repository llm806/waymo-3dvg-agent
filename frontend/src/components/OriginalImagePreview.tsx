import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'

export default function OriginalImagePreview({ src, label, children }: { src: string; label: string; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [open, setOpen] = useState(false)
  const [failed, setFailed] = useState(false)
  useEffect(() => { if (open) dialog.current?.showModal() }, [open])

  return <>
    <button type="button" className="image-preview-trigger" aria-label={`放大查看${label}原图`} title="点击查看无标注原图"
      onClick={() => { setFailed(false); setOpen(true) }}>{children}</button>
    {open && createPortal(<dialog ref={dialog} className="original-image-dialog" aria-label={`${label}原图预览`}
      onClose={() => setOpen(false)} onClick={event => { if (event.target === event.currentTarget) dialog.current?.close() }}>
      <div className="original-image-heading"><strong>{label} · 原图</strong><button type="button" autoFocus aria-label="关闭原图预览" onClick={() => dialog.current?.close()}>关闭</button></div>
      {failed ? <p role="alert">原图加载失败，请关闭后重试。</p> : <img src={src} alt={`${label}原始图像，无标注框`} onError={() => setFailed(true)} />}
    </dialog>, document.body)}
  </>
}
