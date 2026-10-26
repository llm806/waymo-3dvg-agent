import { useState } from 'react'
import PointCloudView from '../../components/PointCloudView'
import OriginalImagePreview from '../../components/OriginalImagePreview'
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
const cameraNames: Record<number,string> = {1:'前方',2:'左前方',3:'右前方',4:'左侧',5:'右侧'}

export default function TaskScene({ data, selected, onSelect }: { data: TaskSceneData; selected: string; onSelect?: (id: string) => void }) {
  const projectedMode = data.image_box_source === 'official_lidar_box_projection'
  const imageBoxLabel = projectedMode ? '3D 真值官方投影' : '原生 2D 框'
  const cameras = data.cameras.map(c=>({...c,native_boxes:projectedMode?(c.projected_boxes??[]):c.native_boxes}))
  const [camera, setCamera] = useState(() => onSelect ? 1 : cameras.find(c=>c.camera_name===data.image_evidence?.recommended_camera&&c.native_boxes.some(b=>b.candidate_id===selected))?.camera_name ?? cameras.find(c=>c.native_boxes.some(b=>b.candidate_id===selected))?.camera_name ?? 1)
  const [loaded, setLoaded] = useState(''), [failed, setFailed] = useState(''), [unlinked, setUnlinked] = useState('')
  const image = cameras.find(c=>c.camera_name===camera) ?? cameras[0]
  const linkedCameras = cameras.filter(c=>c.native_boxes.some(b=>b.candidate_id===selected))
  function choose(id: string) {
    if (!onSelect) return
    setUnlinked(''); onSelect(id)
    if (!image?.native_boxes.some(b=>b.candidate_id===id)) {
      const matching = cameras.find(c=>c.native_boxes.some(b=>b.candidate_id===id))
      if (matching) setCamera(matching.camera_name)
    }
  }
  return <div className="task-scene">
    <div className="scene-legend"><span><i className="box-green" />原生真值框</span><span><i className="box-purple" />{onSelect?'当前选择':'文本对应物体'}</span><span className="muted">点云：原生 3D 框 · 图像：{imageBoxLabel}</span></div>
    <div className="task-scene-grid">
      <section className="scene-pane"><h3>点云场景</h3><PointCloudView key={image?.camera_name} cameraView={image?.view_axes} cameraObjectIds={image?.projected_boxes?.map(b=>b.candidate_id).filter((id): id is string => !!id)} preview={data.preview} blind simple initialFocus={!onSelect} selectedObjectId={selected || undefined} onSelectObject={onSelect ? choose : undefined} /><p className="scene-pane-help">{onSelect?'点击框选择物体；':'紫框对应当前文本；'}拖动旋转，滑动缩放条调整大小。</p></section>
      <section className="scene-pane"><div className="scene-camera-heading"><h3>图像场景</h3><label><span className="sr-only">图像相机</span><select aria-label="图像相机" value={image?.camera_name ?? 1} onChange={e=>{setCamera(Number(e.target.value));setUnlinked('')}}>{cameras.map(c=><option key={c.camera_name} value={c.camera_name}>{cameraNames[c.camera_name]??`相机 ${c.camera_name}`}</option>)}</select></label></div>
        {image ? <><div className="native-image">
          <OriginalImagePreview key={image.image} src={image.image} label={cameraNames[image.camera_name]??`相机 ${image.camera_name}`}><img src={image.image} alt="与点云对应的同帧原始相机图像" onLoad={()=>setLoaded(image.image)} onError={()=>setFailed(image.image)} /></OriginalImagePreview>
          {loaded===image.image && <svg viewBox={`0 0 ${image.width} ${image.height}`} aria-label={imageBoxLabel} className={`native-image-boxes${onSelect ? ' selectable' : ''}`}>
            {[...image.native_boxes].sort((a,b)=>Number(a.candidate_id===selected)-Number(b.candidate_id===selected)).map(box=>{
              const active = !!selected && box.candidate_id===selected
              const activate = () => { if (!onSelect) return; if(box.candidate_id)choose(box.candidate_id);else setUnlinked(box.box_id) }
              return <g key={box.box_id}><rect x={box.xyxy[0]} y={box.xyxy[1]} width={box.xyxy[2]-box.xyxy[0]} height={box.xyxy[3]-box.xyxy[1]} fill="transparent" stroke={active?'#b47cff':'#46d48c'} strokeWidth={active?3:1.5} vectorEffect="non-scaling-stroke"
                role={onSelect?'button':undefined} tabIndex={onSelect?0:undefined} aria-label={`${box.candidate_id?objectName(box.candidate_id):'无唯一三维关联'} · ${categoryLabel(box.category)}`} aria-pressed={onSelect?active:undefined}
                onClick={activate} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}}} style={{cursor:onSelect?'pointer':'default'}}><title>{box.candidate_id?objectName(box.candidate_id):'此原生 2D 框缺少已确认的唯一 3D 关联'}</title></rect>
                {active&&<text x={Math.max(0,box.xyxy[0]+4)} y={Math.max(20,box.xyxy[1]-8)} fill="#b47cff" fontSize={Math.max(20,image.width/65)} fontWeight="600">{objectName(box.candidate_id!)}</text>}</g>
            })}
          </svg>}
        </div>{failed===image.image&&loaded!==image.image&&<p role="alert" className="error">图像加载失败，请重新打开样本后重试。</p>}
          {unlinked && <p role="status" className="notice">这个原生 2D 框没有已确认的唯一 3D 关联，无法据此判断是哪一个 3D 物体。请在点云中选择。</p>}
          {!!selected&&!image.native_boxes.some(b=>b.candidate_id===selected)&&<p className="scene-pane-help">{linkedCameras.length
            ? `此相机没有所选物体的对应框；可在${linkedCameras.map(c=>cameraNames[c.camera_name]).join('、')}查看关联框。`
            : '当前帧所有相机均无所选物体的唯一原生 2D 关联框，切换相机也无法补齐关联。缺少关联不代表物体不在图像中；绿色 2D 框与紫色 3D 框不能据此自动对应。'}</p>}

          {!selected&&!unlinked&&<p className="scene-pane-help">{onSelect?'点击绿色框选择物体；有确认关联时，两侧会同步紫色高亮。':`绿色框：${imageBoxLabel}；紫框与点云中的物体使用同一身份。`}</p>}
        </> : <p className="empty">该帧没有保存相机图像。</p>}
      </section>
    </div>
    {onSelect&&<details className="scene-keyboard-choice"><summary>按物体编号选择</summary><div>{data.candidates.map(c=><button key={c.candidate_id} aria-pressed={selected===c.candidate_id} onClick={()=>choose(c.candidate_id)}>{objectName(c.candidate_id)} · {categoryLabel(c.category)}</button>)}</div></details>}
  </div>
}
