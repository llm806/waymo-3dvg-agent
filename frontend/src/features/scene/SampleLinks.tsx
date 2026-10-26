import { useEffect, useState } from 'react'
import { readJson } from '../../shared/annotationDisplay'
import { componentInfo } from '../../shared/helpContent'

type Link = { component: string; status: string; file: string | null; scope: string | null;
  rows_in_scope: number | null; target_rows: number | null; camera_counts: Record<string, number>;
  target_camera_counts: Record<string, number>;
  laser_counts: Record<string, number>; key_columns: string[]; source_check: string; sha256?: string }
type Links = { packet_id: string; context: string; frame_timestamp_micros: number; target_object_id: string; components: Link[] }
const statuses: Record<string, string> = { linked: '已匹配记录', empty: '此范围无记录', missing: '文件缺失', unreadable: '文件读取失败', invalid_keys: '关联键异常', source_mismatch: '来源已改变' }

export default function SampleLinks({ packetId, camera }: { packetId: string; camera: number }) {
  const [data, setData] = useState<Links | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        const value = await readJson(await fetch(`/api/evidence/${packetId}/links`, { signal: controller.signal })) as Links
        if (value.packet_id !== packetId) throw new Error('数据关联返回了不同样本')
        if (!controller.signal.aborted) setData(value)
      } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : '关联读取失败') }
    })()
    return () => controller.abort()
  }, [packetId])
  if (error) return <p className="error" role="alert">13 组件关联：{error}</p>
  if (!data || data.packet_id !== packetId) return <p role="status">正在按片段与帧主键关联 13 个数据组件…</p>
  const available = data.components.filter(c => ['linked', 'empty'].includes(c.status)).length
  const problems = data.components.filter(c => !['linked', 'empty'].includes(c.status))
  return <details className="sample-links"><summary>当前样本的数据关联 · {available} / {data.components.length} 个组件文件可用{problems.length ? ` · ${problems.length} 项异常` : ''}</summary>
    <p className="muted">13 个文件夹是信息表，按主键关联，不按文件行号拼接。标定按驾驶片段取值，其它表按当前帧取值；框表再按目标 ID 或官方跨模态关联确认。当前相机列随相机切换联动，空关联不会被推测补齐。</p>
    {problems.length > 0 && <p className="error">{problems.map(p => `${componentInfo[p.component]?.[0] ?? p.component}：${statuses[p.status]}`).join('；')}</p>}
    <div className="table-wrap source-files"><table><thead><tr><th>组件 / 用途</th><th>范围与匹配记录</th><th>当前相机 / 目标键</th><th>实际文件与主键</th></tr></thead><tbody>{data.components.map(c => <tr key={c.component}><td><strong>{componentInfo[c.component]?.[0] ?? c.component}</strong><br /><code>{c.component}</code><p className="muted">{componentInfo[c.component]?.[1]}</p></td><td>{statuses[c.status] ?? c.status}<br />{c.scope === 'context' ? '驾驶片段级（不按帧重复）' : c.scope === 'frame' ? '当前帧' : '范围未读取'} · {c.rows_in_scope ?? '未知'} 条<br /><small>{c.source_check === 'matches_recorded_sources' ? '来源指纹与原证据一致' : c.source_check === 'current_source_only' ? '当前文件快照，未参与原证据构建' : c.source_check === 'mismatch' ? '指纹不一致' : '无已记录来源'}</small></td><td>{c.key_columns.includes('key.camera_name') ? `相机 ${camera}：${c.camera_counts[String(camera)] ?? 0} 条` : c.key_columns.includes('key.laser_name') ? `雷达：${Object.keys(c.laser_counts).length} 个` : '不按相机划分'}<br />{c.target_rows !== null ? `全帧目标键：${c.target_rows} 条` : '非三维目标键表'}{c.target_rows !== null && c.key_columns.includes('key.camera_name') && <><br />当前相机目标键：{c.target_camera_counts[String(camera)] ?? 0} 条</>}</td><td><code>{c.file ?? '无文件'}</code><details><summary>关联主键与来源指纹</summary><pre>{c.key_columns.join('\n')}</pre><code>{c.sha256 ?? '未读取'}</code></details></td></tr>)}</tbody></table></div>
  </details>
}
