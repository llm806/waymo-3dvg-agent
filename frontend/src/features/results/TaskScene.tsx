import { localizeLabels, t, useLocale } from '../../shared/locale.ts'
import { useState } from 'react'
import type { CSSProperties } from 'react'
import PointCloudView, { PointCloudHeading } from '../../components/PointCloudView'
import OriginalImagePreview from '../../components/OriginalImagePreview'
import { HelpHint } from '../../components/Help'
import type { CameraView } from '../../components/cameraView'
import type { Preview } from '../../components/PointCloudView'
import { categoryLabel, objectName } from '../../shared/annotationDisplay'

export type Confirmation = { correct: boolean; selected_candidate_id: string; expected_candidate_id: string; attempt: number; first_attempt_correct: boolean }
export type TaskSceneData = { job_id: string; ordinal: number; revision: string; expression: string; run_id: string; can_confirm: boolean; review: boolean;
  source_view?: string | null; image_box_source?: string;
  selected_candidate_id: string | null; confirmation: Confirmation | null; preview: Preview;
  image_evidence?: { recommended_camera: number | null; point_projection_cameras: number[]; lidar_points: number | null } | null;
  candidates: { candidate_id: string; category: string }[];
  cameras: { view_axes?: CameraView | null; projected_boxes?: {box_id:string;candidate_id:string|null;category:string;xyxy:number[]}[]; camera_name: number; width: number; height: number; image: string;
    native_boxes: { box_id: string; candidate_id: string | null; category: string; xyxy: number[] }[] }[] }
const cameraNames: Record<number,string> = localizeLabels({1:'前方',2:'左前方',3:'右前方',4:'左侧',5:'右侧'})

export default function TaskScene({ data, selected, onSelect, candidatePicker = true, compact = false }: {
  data: TaskSceneData; selected: string; onSelect?: (id: string) => void; candidatePicker?: boolean; compact?: boolean
}) {
  useLocale()

  const projectedMode = data.image_box_source === 'official_lidar_box_projection'
  const imageBoxLabel = projectedMode ? t("三维标注在图像中的投影") : t("图像二维标注框")
  const cameras = data.cameras.map(c=>({...c,native_boxes:projectedMode?(c.projected_boxes??[]):c.native_boxes}))
  const [camera, setCamera] = useState(() => onSelect ? 1 : cameras.find(c=>c.camera_name===data.image_evidence?.recommended_camera&&c.native_boxes.some(b=>b.candidate_id===selected))?.camera_name ?? cameras.find(c=>c.native_boxes.some(b=>b.candidate_id===selected))?.camera_name ?? 1)
  const [loaded, setLoaded] = useState(''), [failed, setFailed] = useState(''), [unlinked, setUnlinked] = useState('')
  const image = cameras.find(c=>c.camera_name===camera) ?? cameras[0]
  const imageAspect = compact && image && image.width > 0 && image.height > 0 ? image.width / image.height : null
  const linkedCameras = cameras.filter(c=>c.native_boxes.some(b=>b.candidate_id===selected))
  function choose(id: string) {
    if (!onSelect) return
    setUnlinked(''); onSelect(id)
    if (!image?.native_boxes.some(b=>b.candidate_id===id)) {
      const matching = cameras.find(c=>c.native_boxes.some(b=>b.candidate_id===id))
      if (matching) setCamera(matching.camera_name)
    }
  }
  const legend = <div className="scene-legend"><span><i className="box-green" />{t("数据标注框")}</span><span><i className="box-purple" />{onSelect?t("当前选择"):t("描述目标")}</span></div>
  const cameraSelect = image && <select aria-label={t("图像相机")} value={image.camera_name} onChange={e=>{setCamera(Number(e.target.value));setUnlinked('')}}>
    {cameras.map(c=><option key={c.camera_name} value={c.camera_name}>{cameraNames[c.camera_name]??t("相机 {0}", c.camera_name)}</option>)}
  </select>
  return <div className={`task-scene${compact?' scene-compact':''}`}>
    {!compact&&legend}
    <div className="scene-comparison" style={imageAspect ? {'--scene-aspect-ratio': imageAspect} as CSSProperties : undefined}>
      <section className="scene-pane">
        <div className="scene-pane-heading"><PointCloudHeading label={t("三维点云")} preview={data.preview} selectable={!!onSelect} /></div>
        <div>
          <PointCloudView key={image?.camera_name} cameraView={image?.view_axes} cameraObjectIds={image?.projected_boxes?.map(b=>b.candidate_id).filter((id): id is string => !!id)} preview={data.preview} blind simple canvasHeight={imageAspect ? Math.round(960 / imageAspect) : undefined} initialFocus={!onSelect} selectedObjectId={selected || undefined} onSelectObject={onSelect ? choose : undefined} />
        </div>
      </section>
      <section className="scene-pane">
        <div className="scene-pane-heading">
          <h3><HelpHint label={t("相机图像")} detail={[t("图像框为{0}。点击图像可查看原图。", imageBoxLabel), onSelect?t("点击有三维关联的候选框可选择物体。"):''].filter(Boolean).join(' ')} /></h3>
          {compact&&cameraSelect}
        </div>
        <div>
          {image ? <>
            <div className="scene-image-frame">
              <OriginalImagePreview key={image.image} src={image.image} label={cameraNames[image.camera_name]??t("相机 {0}", image.camera_name)}><img src={image.image} alt={t("与点云对应的同帧原始相机图像")} onLoad={()=>setLoaded(image.image)} onError={()=>setFailed(image.image)} /></OriginalImagePreview>
              {loaded===image.image && <svg viewBox={`0 0 ${image.width} ${image.height}`} preserveAspectRatio="xMidYMid meet" aria-label={imageBoxLabel} className={`native-image-boxes${onSelect ? ' selectable' : ''}`}>
                <svg width={image.width} height={image.height} overflow="hidden">
                  {[...image.native_boxes].sort((a,b)=>Number(a.candidate_id===selected)-Number(b.candidate_id===selected)).map(box=>{
                    const active = !!selected && box.candidate_id===selected
                    const activate = () => { if (!onSelect) return; if(box.candidate_id)choose(box.candidate_id);else setUnlinked(box.box_id) }
                    return <g key={box.box_id}><rect x={box.xyxy[0]} y={box.xyxy[1]} width={box.xyxy[2]-box.xyxy[0]} height={box.xyxy[3]-box.xyxy[1]} fill="transparent" stroke={active?'#b47cff':'#46d48c'} strokeWidth={active?3:1.5} vectorEffect="non-scaling-stroke"
                      role={onSelect?'button':undefined} tabIndex={onSelect?0:undefined} aria-label={`${box.candidate_id?objectName(box.candidate_id):t("无唯一三维关联")} · ${categoryLabel(box.category)}`} aria-pressed={onSelect?active:undefined}
                      onClick={activate} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}}} style={{cursor:onSelect?'pointer':'default'}}><title>{box.candidate_id?objectName(box.candidate_id):t("此原生 2D 框缺少已确认的唯一 3D 关联")}</title></rect>
                      {active&&<text x={Math.max(0,box.xyxy[0]+4)} y={Math.max(20,box.xyxy[1]-8)} fill="#b47cff" fontSize={Math.max(20,image.width/65)} fontWeight="600">{objectName(box.candidate_id!)}</text>}</g>
                  })}
                </svg>
              </svg>}
            </div>
            {compact&&legend}
            {!compact&&<div className="viewer-toolbar"><label className="camera-control">{t("相机")}{cameraSelect}</label></div>}
            {failed===image.image&&loaded!==image.image&&<p role="alert" className="error">{t("图像加载失败，请重新打开样本后重试。")}</p>}
            {unlinked && <p role="status" className="notice">{t("这个原生 2D 框没有已确认的唯一 3D 关联，无法据此判断是哪一个 3D 物体。请在点云中选择。")}</p>}
            {!!selected&&!image.native_boxes.some(b=>b.candidate_id===selected)&&<p className="scene-pane-help">{linkedCameras.length
              ? t("此相机没有所选物体的对应框；可在{0}查看关联框。", linkedCameras.map(c=>cameraNames[c.camera_name]).join('、'))
              : t("当前帧所有相机均无所选物体的唯一原生 2D 关联框，切换相机也无法补齐关联。缺少关联不代表物体不在图像中；绿色 2D 框与紫色 3D 框不能据此自动对应。")}</p>}

          </> : <><div className="scene-image-frame"><p className="scene-image-message">{t("该帧没有保存相机图像。")}</p></div>{compact&&legend}</>}
        </div>
      </section>
    </div>
    {onSelect&&candidatePicker&&<details className="scene-keyboard-choice"><summary>{t("按物体编号选择")}</summary><div>{data.candidates.map(c=><button key={c.candidate_id} aria-pressed={selected===c.candidate_id} onClick={()=>choose(c.candidate_id)}>{objectName(c.candidate_id)} · {categoryLabel(c.category)}</button>)}</div></details>}
  </div>
}
