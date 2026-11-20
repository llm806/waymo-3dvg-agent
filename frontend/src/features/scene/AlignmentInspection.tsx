import { localeTag, serviceText, t, useLocale } from '../../shared/locale.ts'
import { memo, useEffect, useState } from 'react'
import PointCloudView, { PointCloudHeading } from '../../components/PointCloudView'
import OriginalImagePreview from '../../components/OriginalImagePreview'
import type { CameraView } from '../../components/cameraView'
import type { Preview } from '../../components/PointCloudView'
import { HelpHint, TermHelp } from '../../components/Help'
import { categoryLabel, readJson } from '../../shared/annotationDisplay'
import SampleLinks from './SampleLinks'
import { openResult } from '../../navigation'
import { SourcePath } from '../../components/SourcePath'

type View = { camera_name: number; camera_label: string; width: number; height: number; projected_box_xyxy: number[] | null;
  image_sha256_matches: boolean; dimensions_match: boolean; rectangle_in_image: boolean | null;
  projection_entries: number; out_of_bounds_entries: number; target_projected_entries: number;
  target_entries_in_rectangle: number | null; target_in_rectangle_fraction: number | null; pose_minus_frame_ms: number | null;
  scene_points: number[][]; target_points: number[][];
  trigger_time_s: number | null; readout_done_time_s: number | null; pixels: number[][]; target_pixels: number[][]; source_image: string }
type Inspection = { status: string; checked_at: string; independent_alignment_verified: boolean; integrity_verified: boolean; identity_verified: boolean;
  scene_objects: {object_id:string;category:string;converted_points:number;observable:boolean;distinguishable:boolean;annotation_target:boolean;projected_boxes:Record<string,number[]>}[];
  packet_id: string; context: string; frame_timestamp_micros: number; target_object_id: string; default_camera_name: number;
  target_pair_confirmed?: boolean; target_camera_names?: number[];
  coordinate_frame: string; point_count: number; target_points: number; native_label_target_points: number | null;
  point_count_matches_packet: boolean; target_point_count_matches_packet: boolean;
  sensor_returns: Record<string, number>; preview: Preview; views: View[]; limitations: string[];
  sources: Record<string, { file: string; sha256: string }>;
  camera_label_sources: Record<string, { file: string; sha256: string }>;
  image_box_source?: string;
  all_cameras: { view_axes?: CameraView | null; projected_boxes?: {object_id:string; category:string; xyxy:number[]}[]; camera_name: number; width: number; height: number; image: string;
    native_boxes: { camera_object_id: string; category: string; xyxy: number[]; selected_target: boolean; association_status: string; laser_object_ids?: string[] }[] }[] }

function AlignmentInspection({ packetId, sourcesOnly = false, embedded = false, preferredObjectId, preferredSourceView, liveTarget }: { packetId: string; preferredObjectId?: string; preferredSourceView?: string | null; sourcesOnly?: boolean; embedded?: boolean; liveTarget?: { context: string; timestamp: number | null; object_id: string } }) {
  useLocale()

  const live = !!liveTarget
  const [data, setData] = useState<Inspection | null>(null)
  const [error, setError] = useState<{ packetId: string; message: string } | null>(null)
  const [imageStatus, setImageStatus] = useState<{ image: string; loaded: boolean } | null>(null)
  const [points, setPoints] = useState(false)
  const [nativeBoxes, setNativeBoxes] = useState(true)
  const [projection, setProjection] = useState(false)
  const [camera, setCamera] = useState(0)
  const [objectSelection,setObjectSelection] = useState({ id:'', preferred:preferredObjectId })
  const objectId = objectSelection.preferred===preferredObjectId?objectSelection.id:preferredObjectId
  const setObjectId = (id:string) => setObjectSelection({id,preferred:preferredObjectId})
  const [related,setRelated] = useState<{packet_id:string;object_id:string}[]>([])
  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        const r = await fetch('/api/evidence/' + packetId + '/inspection', { signal: controller.signal })
        const value = await readJson(r) as Inspection
        if (value.packet_id !== packetId) throw new Error(t("图像与点云返回了不同样本"))
        if (!controller.signal.aborted) {
          setData(value); setError(null)
          if (!live) {
            const index = await readJson(await fetch('/api/evidence',{signal:controller.signal}))
            if(!controller.signal.aborted) setRelated(index.packets.filter((p:{context:string;timestamp:number})=>p.context===value.context&&p.timestamp===value.frame_timestamp_micros))
          }
        }
      } catch (e) { if (!controller.signal.aborted) setError({ packetId, message: e instanceof Error ? e.message : t("读取失败") }) }
    })()
    return () => controller.abort()
  }, [packetId, live])
  if (error?.packetId === packetId) return <p className="error" role="alert">{serviceText(error.message)}</p>
  if (!data || data.packet_id !== packetId) return <p role="status">{t("校验场景文件与数据关联，读取官方像素投影…")}</p>
  if (liveTarget && (data.context !== liveTarget.context || data.frame_timestamp_micros !== liveTarget.timestamp || data.target_object_id !== liveTarget.object_id)) return <p className="error" role="alert">{t("当前画面与正在处理的物体或帧不一致，已停止展示。")}</p>
  if (liveTarget && (!data.integrity_verified || !data.identity_verified)) return <p className="error" role="alert">{t("当前场景的数据或物体身份校验未通过，已停止展示。")}</p>
  if (sourcesOnly) return <><h3>{t("场景点云转换源文件")}</h3><div className="table-wrap source-files"><table><thead><tr><th>{t("组件")}</th><th>{t("具体文件")}</th><th>{t("文件校验详情")}</th></tr></thead><tbody>{Object.entries(data.sources).map(([key, source]) => <tr key={key}><td>{key}</td><td><code>{source.file}</code></td><td><code>{source.sha256}</code></td></tr>)}</tbody></table></div><details><summary>{t("五个传感器与双回波点数")}</summary><pre>{JSON.stringify(data.sensor_returns, null, 2)}</pre></details></>
  const projectedMode = data.image_box_source === 'official_lidar_box_projection'
  const imageBoxLabel = projectedMode ? t("三维标注投影") : t("图像二维标注")
  const cameras = data.all_cameras.map(c => ({...c, native_boxes: projectedMode ? (c.projected_boxes ?? []).map(b => ({camera_object_id:b.object_id, category:b.category, xyxy:b.xyxy,
    selected_target:b.object_id===data.target_object_id, association_status:'official_projection', laser_object_ids:[b.object_id]})) : c.native_boxes}))
  const inspected = data.scene_objects?.find(o=>o.object_id===(liveTarget ? data.target_object_id : objectId||data.target_object_id))
  const preferredCamera = ({ FRONT:1, FRONT_LEFT:2, FRONT_RIGHT:3, SIDE_LEFT:4, SIDE_RIGHT:5 } as Record<string,number>)[preferredSourceView ?? '']
  const targetCamera = liveTarget && cameras.find(c => c.camera_name === (preferredCamera ?? data.default_camera_name) && c.native_boxes.some(b => b.selected_target))
    || (liveTarget && cameras.find(c => c.native_boxes.some(b => b.selected_target)))
  const defaultCamera = targetCamera ? targetCamera.camera_name : inspected&&!inspected.annotation_target?Number(Object.keys(inspected.projected_boxes)[0]??data.default_camera_name):data.default_camera_name
  const selected = cameras.find(c => c.camera_name === camera) ?? cameras.find(c => c.camera_name === defaultCamera) ?? cameras[0]
  const view = data.views.find(v => v.camera_name === selected?.camera_name)
  const displayedBoxes = (selected?.native_boxes ?? []).map(b => ({ ...b,
    selected_target: inspected ? b.association_status !== 'ambiguous' && b.category === inspected.category
      && b.laser_object_ids?.length === 1 && b.laser_object_ids[0] === inspected.object_id : b.selected_target }))
  const linkedTarget = displayedBoxes.some(b => b.selected_target)
  const inspectingTarget = !inspected || inspected.object_id===data.target_object_id
  // A different inspected object must not retain the annotation target's points or box highlight.
  const preview = inspectingTarget ? data.preview : { ...data.preview,
    points: [...data.preview.points, ...data.preview.target_points], target_points: [],
    target_corners: data.preview.candidate_boxes.find(b => b.object_id === inspected.object_id)?.corners ?? [] }
  const rect = inspected?.projected_boxes[String(selected?.camera_name)] ?? (inspectingTarget?view?.projected_box_xyxy:null)
  const hasPixels = !!view && view.projection_entries > view.out_of_bounds_entries
  const imageLoaded = imageStatus?.image === selected?.image && imageStatus?.loaded === true
  const pairedHighlight = !!linkedTarget && imageLoaded && nativeBoxes
  const imageFailed = imageStatus?.image === selected?.image && imageStatus?.loaded === false
  const labels: Record<number, string> = { 1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT' }
  const cameraNames: Record<number, string> = { 1: t("前方"), 2: t("左前方"), 3: t("右前方"), 4: t("左侧"), 5: t("右侧") }
  const compact = embedded && !!liveTarget
  const cameraSelect = selected && <select aria-label={t("对齐相机")} value={selected.camera_name} onChange={e => setCamera(Number(e.target.value))}>{cameras.map(c => <option key={c.camera_name} value={c.camera_name}>{cameraNames[c.camera_name] ?? t("相机 {0}", c.camera_name)}{!compact && ` · ${labels[c.camera_name] ?? c.camera_name}`}</option>)}</select>
  const legend = <div className="scene-legend"><span><i className="box-green" />{t("数据标注框")}</span>{pairedHighlight&&<span><i className="box-purple" />{inspectingTarget?t("描述目标"):t("当前查看物体")}</span>}{!liveTarget&&projection&&rect&&<span><i className="box-orange" />{t("3D 投影参考")}</span>}</div>
  return <section className={embedded ? `embedded-alignment${compact ? ' scene-compact' : ''}` : undefined} aria-label={liveTarget ? t("当前物体的点云与图像") : t("图像与点云对齐诊断")} data-packet-id={data.packet_id} data-frame-timestamp={data.frame_timestamp_micros} data-target-id={data.target_object_id}>

    {!liveTarget && data.scene_objects?.length>0&&<div className="scene-object-choice"><label>{t("场景中物体")}<select aria-label={t("查看场景物体")} value={inspected?.object_id??data.target_object_id} onChange={e=>{setObjectId(e.target.value);setCamera(0)}}>{data.scene_objects.map((o,i)=><option key={o.object_id} value={o.object_id}>{t("物体 ")}{i+1} · {categoryLabel(o.category)} · {o.annotation_target?t("标注目标"):o.observable?o.distinguishable?t("具备区分线索"):t("需补关系"):t("无可用点／视角")}</option>)}</select></label>{inspected&&<span>{inspected.converted_points}{t(" 个实测框内点 · ")}{Object.keys(inspected.projected_boxes).length}{t(" 个投影视角")}{related.find(p=>p.object_id===inspected.object_id&&p.packet_id!==packetId)&&<button onClick={()=>openResult(related.find(p=>p.object_id===inspected.object_id&&p.packet_id!==packetId)!.packet_id)}>{t("打开此物体的标注记录")}</button>}</span>}</div>}
    {!compact && legend}
    <div className="scene-comparison">
      <section className="scene-pane">
        <div className="scene-pane-heading"><PointCloudHeading label={t("三维点云")} preview={preview} />{compact && legend}</div>
        <PointCloudView key={`${selected?.camera_name}-${pairedHighlight ? 'paired' : 'scene'}`} cameraView={selected?.view_axes} cameraObjectIds={selected?.projected_boxes?.map(b=>b.object_id)} preview={preview} initialFocus={pairedHighlight} highlightTarget={pairedHighlight} selectedObjectId={liveTarget ? undefined : inspected?.object_id} cameraOverlay={points && hasPixels && view ? { points: view.scene_points, target_points: inspectingTarget ? view.target_points : [] } : undefined} />
      </section>
      <section className="scene-pane">
        <div className="scene-pane-heading"><h3><HelpHint label={t("相机图像")} detail={t("图像框为{0}。点击图像可查看原图。", imageBoxLabel)} /></h3>{compact && cameraSelect}{!liveTarget && <TermHelp term="cameraBox" label={t("框的区别")} />}</div>
        <div>
          <div className="scene-image-frame">
            {selected && !imageLoaded && !imageFailed && <p className="scene-image-message" role="status">{t("读取当前帧图像…")}</p>}
            {imageFailed && <p className="scene-image-message error" role="alert">{t("当前相机图像加载失败，请切换相机或刷新。")}</p>}
            {selected && <OriginalImagePreview key={selected.image} src={selected.image} label={cameraNames[selected.camera_name] ?? t("相机")}>
              <svg style={{ visibility: imageLoaded ? "visible" : "hidden" }} viewBox={`0 0 ${selected.width} ${selected.height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={(labels[selected.camera_name] ?? selected.camera_name) + t(" 图像与") + imageBoxLabel}>
                <svg width={selected.width} height={selected.height} overflow="hidden">
                  <image href={selected.image} width={selected.width} height={selected.height} onLoad={() => setImageStatus({ image: selected.image, loaded: true })} onError={() => setImageStatus({ image: selected.image, loaded: false })} />
                  {points && view && <g fill="#4bc5ff" opacity="0.6">{view.pixels.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="2" />)}</g>}
                  {points && inspectingTarget && view && <g fill="#ffd373">{view.target_pixels.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3" />)}</g>}
                  {nativeBoxes && [...displayedBoxes].sort((a,b)=>Number(a.selected_target)-Number(b.selected_target)).map(b => <rect key={b.camera_object_id} x={b.xyxy[0]} y={b.xyxy[1]} width={b.xyxy[2] - b.xyxy[0]} height={b.xyxy[3] - b.xyxy[1]} fill="none" stroke={b.selected_target ? '#b47cff' : '#46d48c'} strokeWidth={b.selected_target ? 6 : 3}><title>{categoryLabel(b.category)} · {imageBoxLabel} · {b.camera_object_id}{b.selected_target ? (b.association_status === 'official_projection' ? t(" · 与原生3D框同一物体ID") : b.association_status === 'point_supported' ? t(" · 实测点对应推导关联（非官方 ID关联）") : t(" · 官方关联所选目标")) : ''}</title></rect>)}
                  {!liveTarget && projection && rect && <rect x={rect[0]} y={rect[1]} width={rect[2] - rect[0]} height={rect[3] - rect[1]} fill="none" stroke="#ffbe55" strokeWidth="5" strokeDasharray="14 8"><title>{t("当前查看物体的3D框投影；不是原生相机2D真值")}</title></rect>}
                </svg>
              </svg>
            </OriginalImagePreview>}
            {!selected && <p className="scene-image-message">{t("该帧没有保存相机图像。")}</p>}
          </div>
          {selected && !compact && <div className="viewer-toolbar"><label className="camera-control">{t("相机")}{cameraSelect}</label>
            {!liveTarget && <><label className="projection-toggle" title={!hasPixels ? t("所选相机没有有效像素对应，不能启用叠加") : t("在点云和图像中同时高亮同一组官方对应点")}><input aria-label={t("显示官方点云像素对应")} type="checkbox" checked={points && hasPixels} onChange={e => setPoints(e.target.checked)} disabled={!hasPixels} />{t("联动对应点 ")}</label><TermHelp term="projection" label={t("说明")} /></>}</div>}
          {!liveTarget && !hasPixels && <p className="muted">{t("此相机没有有效像素对应，联动叠加不可用。")}</p>}
          {!linkedTarget && <p className="association-note">{cameras.some(c=>c.native_boxes.some(b=>b.selected_target))
            ? <>{t("此相机没有目标紫框。")}{targetCamera&&<button className="text-button" onClick={()=>setCamera(targetCamera.camera_name)}>{t("查看目标相机")}</button>}</>
            : t("此记录未确认图像关联，仅显示场景真值，不作为标注目标。")}</p>}
          {points && view && <p className="muted correspondence-legend">{t("蓝点：场景对应")}{inspectingTarget ? t(" · 黄点：标注目标对应") : ''}{t("；与点云同组联动。")}</p>}
          {!liveTarget && <div className="viewer-help" role="group" aria-label={t("图层设置")}><label className="projection-toggle"><input type="checkbox" checked={nativeBoxes} onChange={e => setNativeBoxes(e.target.checked)} />{t("显示")}{imageBoxLabel}（{displayedBoxes.length}{t(" 个）")}</label>{!projectedMode && <label className="projection-toggle" title={!rect ? t("此相机没有该目标的投影矩形") : t("来自 projected_lidar_box，不是 camera_box 原生2D真值")}><input type="checkbox" checked={projection && !!rect} onChange={e => setProjection(e.target.checked)} disabled={!rect} />{t("显示当前查看物体的 3D 框投影（参考）")}</label>}{!rect && <p className="muted">{t("此相机没有当前查看物体的投影记录，投影开关不可用。")}</p>}</div>}
        </div>
      </section>
    </div>
    {!liveTarget && <><div className="scene-check-summary"><span>{t("数据与身份：")}{data.integrity_verified && data.identity_verified ? t("一致") : t("存在异常")}</span><span className="quality-pending">{t("独立对齐质量：")}{data.independent_alignment_verified ? t("已验证") : t("待验证")}</span><TermHelp term="projection" label={t("检查范围")} /></div>
    <details className="panel scene-diagnostics"><summary>{t("数据关联与对齐检查")}<span className="muted">{t("文件、主键、统计与时间差")}</span></summary>
    <div className="linked-scene-identity"><span><TermHelp term="frame" label={t("帧时间")} /> {data.frame_timestamp_micros} μs</span><span><TermHelp term="track" label={t("目标 ID")} /> <code>{data.target_object_id}</code></span><span>{t("来源片段：")}<code>{data.context}</code></span></div>
    <SourcePath file={data.sources.lidar?.file} label={t("点云来源")} /><SourcePath file={data.sources.camera_image?.file} label={t("图像来源")} />
    <p className="muted">{data.coordinate_frame}{t(" · 完整场景 ")}{data.point_count.toLocaleString()}{t(" 点；目标框内 ")}{data.target_points.toLocaleString()}{t(" 点。预览采样不改变完整数据。")}</p>
    <div className="gate-grid"><div>{t("文件内容与场景身份")}<strong>{data.integrity_verified && data.identity_verified ? t("通过") : t("未通过")}</strong><TermHelp term="hash" label={t("检查范围")} /></div><div>{t("点数与目标证据记录")}<strong>{data.point_count_matches_packet && data.target_point_count_matches_packet ? t("一致") : t("不一致")}</strong><TermHelp term="evidence" /></div><div className="pending-gate">{t("独立对齐质量验证")}<strong>{data.independent_alignment_verified ? t("已验证") : t("尚未验证")}</strong><TermHelp term="projection" label={t("为何前两项通过仍未验证？")} /></div></div>
    {view && inspectingTarget && <details className="projection-statistics"><summary>{t("当前相机对应统计与时间差")}</summary><p className="muted">{t("以下“矩形”指目标 3D 标签的二维投影矩形，不是原生相机 2D 真值框。场景有效对应 ")}{view.projection_entries - view.out_of_bounds_entries}{t(" 条。")}</p><dl className="metadata-grid"><div><dt>{t("图像内容 / 尺寸 / 投影矩形边界")}</dt><dd>{view.image_sha256_matches ? t("一致") : t("不一致")} / {view.dimensions_match ? t("一致") : t("不一致")} / {view.rectangle_in_image === null ? t("无目标投影矩形") : view.rectangle_in_image ? t("图像内") : t("部分超出图像，显示裁切")}</dd></div><div><dt>{t("目标像素对应 / 投影矩形内（未采样统计）")}</dt><dd>{view.target_projected_entries} / {view.target_entries_in_rectangle ?? t("无矩形")} · {view.target_in_rectangle_fraction === null ? (rect ? t("无有效目标像素对应") : t("无目标投影矩形")) : (view.target_in_rectangle_fraction * 100).toFixed(1) + '%'}</dd></div><div><dt>{t("相机 Pose 与 LiDAR 帧时间差")}</dt><dd>{view.pose_minus_frame_ms?.toFixed(2) ?? t("未知")} ms</dd></div><div><dt>{t("官方投影记录 / 越界记录")}</dt><dd>{view.projection_entries.toLocaleString()} / {view.out_of_bounds_entries}</dd></div></dl><p>{t("投影矩形：")}{rect?.map(v => v.toFixed(2)).join(', ') ?? t("无目标投影矩形")} px</p><p>{t("曝光触发 ")}{view.trigger_time_s ?? t("未知")}{t(" s / 读出结束 ")}{view.readout_done_time_s ?? t("未知")} s</p></details>}
    <SampleLinks key={packetId} packetId={packetId} camera={selected?.camera_name ?? 1} />
    <details><summary>{t("图像框与关联的来源")}</summary>{Object.entries(data.camera_label_sources).map(([name, source]) => <p key={name}><code>{name}</code>：<code>{source.file}</code><code>{source.sha256}</code></p>)}</details>
    <details><summary>{t("对齐诊断范围与限制")}</summary><p>{t("诊断快照 ")}{new Date(data.checked_at).toLocaleString(localeTag())}{t("；按场景文件内容识别数据版本。原生标签目标点数 ")}{data.native_label_target_points?.toLocaleString(localeTag()) ?? t("未知")}{t("；该统计与转换后框内点数口径可能不同。")}</p><ul>{data.limitations.map(s => <li key={s}>{serviceText(s)}</li>)}</ul></details>
    </details></>}
  </section>
}

// Polls replace item objects even when their scene identity is unchanged.
// Compare the complete set of props used by this viewer, including presence
// of liveTarget. Its own camera/layer state still renders normally.
export default memo(AlignmentInspection, (a,b) => a.packetId === b.packetId
  && a.sourcesOnly === b.sourcesOnly && a.embedded === b.embedded
  && a.preferredObjectId === b.preferredObjectId && a.preferredSourceView === b.preferredSourceView
  && !!a.liveTarget === !!b.liveTarget && a.liveTarget?.context === b.liveTarget?.context
  && a.liveTarget?.timestamp === b.liveTarget?.timestamp && a.liveTarget?.object_id === b.liveTarget?.object_id)

