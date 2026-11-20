import { localeTag, t, useLocale } from '../shared/locale.ts'
import { useEffect, useRef, useState } from 'react'
import { HelpHint } from './Help'
import { cameraPixel } from './cameraView'
import type { CameraView } from './cameraView'
import { pickBox } from './scenePicking'
import type { ScreenBox } from './scenePicking'

type Vector = number[]
export type Preview = { points: Vector[]; target_points: Vector[]; target_corners: Vector[];
  candidate_boxes: { object_id: string; corners: Vector[]; category?: string }[]; sampling: string }

export function PointCloudHeading({ label, preview, selectable = false }: { label: string; preview: Preview; selectable?: boolean }) {
  useLocale()
  const count = (preview.points.length + preview.target_points.length).toLocaleString(localeTag())
  const detail = [t("拖动或使用方向键旋转，使用滑条缩放。"), t("俯视时，三维框的上下边会重合。"),
    ...(selectable ? [t("点击候选框选择物体。")] : []), t("当前预览含 {0} 个采样点。视图操作仅影响显示，训练输入使用完整点云。", count)].join(' ')
  return <h3><HelpHint label={label} detail={detail} /></h3>
}

const colors = {
  points: '#748791',
  target: '#7548a7',
  candidate: '#38835d',
  projection: '#287da2',
  targetProjection: '#a86a22',
  label: '#324750',
}

export default function PointCloudView({ preview, cameraView, cameraObjectIds, cameraOverlay, selectedObjectId, blind = false, highlightTarget = true, initialFocus = !blind, simple = false, canvasHeight = 520, onSelectObject }: { preview: Preview; cameraView?: CameraView | null; cameraObjectIds?: string[]; selectedObjectId?: string; blind?: boolean; highlightTarget?: boolean; initialFocus?: boolean; simple?: boolean; canvasHeight?: number; onSelectObject?: (id: string) => void;
  cameraOverlay?: { points: Vector[]; target_points: Vector[] } }) {
  const locale = useLocale()

  const canvas = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; y: number; startX: number; startY: number; moved: boolean } | null>(null)
  const hitBoxes = useRef<ScreenBox[]>([])
  const [cameraMode, setCameraMode] = useState(!!cameraView)
  const [yaw, setYaw] = useState(-0.6)
  const [pitch, setPitch] = useState(0.65)
  const [zoom, setZoom] = useState(1)
  const [focus, setFocus] = useState(initialFocus)
  const [boxes, setBoxes] = useState(true)
  // Derive the view selection from the actual camera angle, including manual rotation.
  const topView = !cameraMode && Math.abs(pitch - Math.PI / 2) < 1e-6

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (!ctx) return
    const { width, height } = ctx.canvas
    ctx.clearRect(0, 0, width, height)
    const focusedCorners = preview.candidate_boxes.find(b => b.object_id === selectedObjectId)?.corners ?? preview.target_corners
    const origin = focus && focusedCorners.length === 8 ? [0, 1, 2].map(i => focusedCorners.reduce((s, p) => s + p[i], 0) / 8) : [0, 0, 0]
    // Fit real 3D geometry to the viewport; zoom is relative to this fit.
    const fitPoints = focus && focusedCorners.length === 8 ? focusedCorners : [...preview.points, ...preview.target_corners, ...preview.candidate_boxes.flatMap(b => b.corners)]
    const radius = fitPoints.reduce((r, p) => Math.max(r, Math.hypot(...p.map((v, i) => v - origin[i]))), focus ? 1.5 : 10)
    const scale = Math.min(width, height) * .42 / radius * zoom
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch)
    const cameraCorners = cameraMode && cameraView && focus ? focusedCorners.map(p => cameraPixel(p, cameraView)).filter((p): p is number[] => p !== null) : []
    const center = cameraCorners.length ? [0, 1].map(i => cameraCorners.reduce((sum, p) => sum+p[i], 0)/cameraCorners.length) : [cameraView ? cameraView.width/2 : 0, cameraView ? cameraView.height/2 : 0]
    const cameraSpan = cameraCorners.length ? Math.max(...cameraCorners.map(p => Math.max(Math.abs(p[0]-center[0]), Math.abs(p[1]-center[1]))))*2 : Math.max(cameraView?.width ?? 1, (cameraView?.height ?? 1)*width/height)
    const cameraScale = cameraCorners.length ? Math.min(width, height)*.75/Math.max(cameraSpan, 20)*zoom : width/cameraSpan*zoom
    const project = (p: Vector) => {
      if (cameraMode && cameraView) {
        const uv = cameraPixel(p, cameraView)
        return uv ? [width/2+(uv[0]-center[0])*cameraScale, height/2+(uv[1]-center[1])*cameraScale] : [NaN, NaN]
      }
      const x = p[0] - origin[0], y = p[1] - origin[1], z = p[2] - origin[2]
      const horizontal = x * cy - y * sy, depth = x * sy + y * cy
      return [width / 2 + horizontal * scale, height / 2 + (-depth * sp - z * cp) * scale]
    }
    const drawPoints = (points: Vector[], color: string, size: number) => {
      ctx.fillStyle = color
      for (const p of points) { const [x, y] = project(p); if (x >= 0 && x < width && y >= 0 && y < height) ctx.fillRect(x, y, size, size) }
    }
    drawPoints(preview.points, colors.points, 1.4)
    drawPoints(preview.target_points, highlightTarget ? colors.target : colors.points, highlightTarget ? 2 : 1.4)
    if (cameraOverlay) {
      drawPoints(cameraOverlay.points, colors.projection, 2)
      drawPoints(cameraOverlay.target_points, colors.targetProjection, 3)
    }
    const drawBox = (corners: Vector[], color: string) => {
      if (corners.length !== 8) return
      ctx.strokeStyle = color; ctx.lineWidth = color === colors.target ? 2.6 : 1.2
      for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) {
        const j = i ^ bit; if (j < i) continue
        const a = project(corners[i]), b = project(corners[j])
        if (![...a, ...b].every(Number.isFinite)) continue
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke()
      }
    }
    const visibleBoxes = cameraMode && cameraObjectIds ? preview.candidate_boxes.filter(b=>cameraObjectIds.includes(b.object_id)) : preview.candidate_boxes
    hitBoxes.current = boxes ? visibleBoxes.map(box => ({ id: box.object_id, points: box.corners.map(project) })).filter(b => b.points.every(p => p.every(Number.isFinite))) : []
    if (boxes) for (const box of visibleBoxes) {
      const index = preview.candidate_boxes.findIndex(b => b.object_id === box.object_id)
      drawBox(box.corners, highlightTarget && box.object_id === selectedObjectId ? colors.target : colors.candidate)
      if (box.object_id === selectedObjectId || (!simple && (blind || !focus))) {
        const center = [0,1,2].map(i => box.corners.reduce((sum,p) => sum+p[i],0)/8)
        const [x,y] = project(center)
        ctx.fillStyle = colors.label; ctx.font = '13px sans-serif'
        if (x >= 0 && x <= width && y >= 0 && y <= height) ctx.fillText(blind ? simple ? t("物体 {0}", Number(box.object_id.slice(1))) : box.object_id : t("物体 {0}{1}", index+1, box.object_id===selectedObjectId?t(" · 当前查看"):''),x+6,y-6)
      }
    }
    const selectedBox = visibleBoxes.find(box => box.object_id === selectedObjectId)
    if (boxes && selectedBox && highlightTarget) drawBox(selectedBox.corners, colors.target)
    if (!blind && highlightTarget) {
      drawBox(preview.target_corners, colors.target)
      if (preview.target_corners.length === 8) {
        const center = [0, 1, 2].map(i => preview.target_corners.reduce((sum, p) => sum + p[i], 0) / 8)
        const [x, y] = project(center)
        ctx.fillStyle = colors.target; ctx.font = 'bold 15px sans-serif'
        if (x >= 0 && x <= width && y >= 0 && y <= height) ctx.fillText(selectedObjectId ? t("当前查看") : t("标注目标"), x + 7, y + 16)
      }
    }
    // A direction compass stays visible when focusing a distant target.
    // It rotates with the view and is not a metric scale or an ego position marker.
    const compass = (p: Vector) => cameraMode && cameraView ? [p.reduce((sum,v,i)=>sum+v*cameraView.right[i],0), -p.reduce((sum,v,i)=>sum+v*cameraView.up[i],0)] : [(p[0]*cy-p[1]*sy), -(p[0]*sy+p[1]*cy)*sp-p[2]*cp]
    const reference = compass([0, 0, 0])
    const start = [68, height - 65]
    for (const [point, color, text] of [[[8, 0, 0], colors.candidate, t("x 前")], [[0, 8, 0], colors.projection, t("y 左")], [[0, 0, 8], colors.target, t("z 上")]] as [Vector, string, string][]) {
      const projected = compass(point)
      const end = [start[0]+(projected[0]-reference[0])*5,start[1]+(projected[1]-reference[1])*5]
      if (Math.hypot(end[0]-start[0],end[1]-start[1]) < 5) { ctx.fillStyle = color; ctx.fillText(text+t(" · 纵深"), start[0]-40, start[1]+25); continue }
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(start[0], start[1]); ctx.lineTo(end[0], end[1]); ctx.stroke()
      ctx.font = '14px sans-serif'; ctx.fillText(text, end[0] + 6, end[1])
    }
  }, [preview, cameraView, cameraObjectIds, cameraMode, yaw, pitch, zoom, focus, boxes, cameraOverlay, selectedObjectId, blind, simple, highlightTarget, locale, canvasHeight])

  return <section className={simple ? "point-view point-view-simple" : "point-view"} aria-label={blind ? t("三维点云与匿名候选框") : highlightTarget ? t("三维点云与目标框") : t("三维点云与场景真值框")}>
    <p className="sr-only" role="status">{t("当前视图：")}{focus ? t("聚焦目标") : t("完整场景")} · {cameraMode ? t("相机视角") : topView ? t("俯视") : t("三维视角")}{t("。显示范围与观察视角可分别选择。")}</p>
    <canvas ref={canvas} width={960} height={canvasHeight} tabIndex={0} aria-label={blind ? t("盲评点云：匿名候选框，不突出真实目标") : t("点云预览：拖动旋转，紫色为标注目标框")}
      onPointerDown={e => { drag.current = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false }; e.currentTarget.setPointerCapture(e.pointerId) }}
      onPointerUp={e => {
        const gesture = drag.current; drag.current = null
        if (gesture && !gesture.moved && onSelectObject) {
          const { width, height } = e.currentTarget
          const rect = e.currentTarget.getBoundingClientRect()
          const id = pickBox(hitBoxes.current, [(e.clientX-rect.left)*width/rect.width,(e.clientY-rect.top)*height/rect.height], 6*width/rect.width)
          if (id) onSelectObject(id)
        }
      }} onPointerCancel={() => { drag.current = null }}
      onPointerMove={e => { const gesture = drag.current; if (gesture) {
        const moved = gesture.moved || Math.hypot(e.clientX-gesture.startX,e.clientY-gesture.startY)>4
        if (moved) { setCameraMode(false); setYaw(v=>v+(e.clientX-gesture.x)*0.008); setPitch(v=>Math.max(0.05,Math.min(Math.PI/2,v+(e.clientY-gesture.y)*0.008))) }
        drag.current = {...gesture,x:e.clientX,y:e.clientY,moved}
      } }}
      onKeyDown={e => { if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) { e.preventDefault(); setCameraMode(false); if (e.key === 'ArrowLeft') setYaw(v => v - .1); if (e.key === 'ArrowRight') setYaw(v => v + .1); if (e.key === 'ArrowUp') setPitch(v => Math.min(Math.PI / 2, v + .1)); if (e.key === 'ArrowDown') setPitch(v => Math.max(.05, v - .1)) } }} />
    <div className="controls point-controls">
      <div className="point-control-group" role="group" aria-label={t("显示范围")}><span>{t("显示范围")}</span>
        <button type="button" disabled={!highlightTarget || blind && !selectedObjectId} aria-pressed={focus} title={blind ? t("以你选看的匿名候选框为中心适配视图，保留当前观察角度") : t("以紫色 3D 真值框为中心适配视图，保留当前观察角度")} onClick={() => { setFocus(true); setZoom(1) }}>{blind ? t("聚焦选择") : t("聚焦物体")}</button>
        <button type="button" aria-pressed={!focus} title={t("以自车为中心适配完整场景，保留当前观察角度")} onClick={() => { setFocus(false); setZoom(1) }}>{t("完整场景")}</button>
      </div>
      <div className="point-control-group" role="group" aria-label={t("观察视角")}><span>{t("观察视角")}</span>
        {cameraView && <button type="button" aria-pressed={cameraMode} onClick={() => { setCameraMode(true); setZoom(1) }}>{t("相机视角")}</button>}
        <button type="button" aria-pressed={!cameraMode && !topView} title={t("恢复倾斜观察角度，可看清 3D 框的高度；保留当前显示范围")} onClick={() => { setCameraMode(false); setYaw(-0.6); setPitch(0.65) }}>{t("三维视角")}</button>
        <button type="button" aria-pressed={topView} title={t("从上方看：3D 框的上下表面会重合；保留当前显示范围")} onClick={() => { setCameraMode(false); setYaw(Math.PI/2); setPitch(Math.PI / 2) }}>{t("俯视")}</button>
      </div>
    </div>
    <div className="controls point-secondary-controls"><label>{t("缩放 ")}<input aria-label={t("点云缩放")} type="range" min="0.1" max="20" step="0.1" value={zoom} onChange={e => setZoom(Number(e.target.value))} />{zoom.toFixed(1)}×</label><label><input type="checkbox" checked={boxes} onChange={e => setBoxes(e.target.checked)} />{blind ? t("候选物体 3D 框") : t("其它物体 3D 框")}</label></div>
  </section>
}
