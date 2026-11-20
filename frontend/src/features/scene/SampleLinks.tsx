import { serviceText, localizeLabels, t, useLocale } from '../../shared/locale.ts'
import { useEffect, useState } from 'react'
import { readJson } from '../../shared/annotationDisplay'
import { componentInfo } from '../../shared/helpContent'

type Link = { component: string; status: string; file: string | null; scope: string | null;
  rows_in_scope: number | null; target_rows: number | null; camera_counts: Record<string, number>;
  target_camera_counts: Record<string, number>;
  laser_counts: Record<string, number>; key_columns: string[]; source_check: string; sha256?: string }
type Links = { packet_id: string; context: string; frame_timestamp_micros: number; target_object_id: string; components: Link[] }
const statuses: Record<string, string> = localizeLabels({ linked: '已匹配记录', empty: '此范围无记录', missing: '文件缺失', unreadable: '文件读取失败', invalid_keys: '关联键异常', source_mismatch: '来源已改变' })

export default function SampleLinks({ packetId, camera }: { packetId: string; camera: number }) {
  useLocale()

  const [data, setData] = useState<Links | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        const value = await readJson(await fetch(`/api/evidence/${packetId}/links`, { signal: controller.signal })) as Links
        if (value.packet_id !== packetId) throw new Error(t("数据关联返回了不同样本"))
        if (!controller.signal.aborted) setData(value)
      } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : t("关联读取失败")) }
    })()
    return () => controller.abort()
  }, [packetId])
  if (error) return <p className="error" role="alert">{t("13 组件关联：")}{serviceText(error)}</p>
  if (!data || data.packet_id !== packetId) return <p role="status">{t("正在按片段与帧主键关联 13 个数据组件…")}</p>
  const available = data.components.filter(c => ['linked', 'empty'].includes(c.status)).length
  const problems = data.components.filter(c => !['linked', 'empty'].includes(c.status))
  return <details className="sample-links"><summary>{t("当前样本的数据关联 · ")}{available} / {data.components.length}{t(" 个组件文件可用")}{problems.length ? t(" · {0} 项异常", problems.length) : ''}</summary>
    <p className="muted">{t("13 个文件夹是信息表，按主键关联，不按文件行号拼接。标定按驾驶片段取值，其它表按当前帧取值；框表再按目标 ID 或官方跨模态关联确认。当前相机列随相机切换联动，空关联不会被推测补齐。")}</p>
    {problems.length > 0 && <p className="error">{problems.map(p => `${t(componentInfo[p.component]?.[0] ?? p.component)}：${statuses[p.status]}`).join('；')}</p>}
    <div className="table-wrap source-files"><table><thead><tr><th>{t("组件 / 用途")}</th><th>{t("范围与匹配记录")}</th><th>{t("当前相机 / 目标键")}</th><th>{t("实际文件与主键")}</th></tr></thead><tbody>{data.components.map(c => <tr key={c.component}><td><strong>{t(componentInfo[c.component]?.[0] ?? c.component)}</strong><br /><code>{c.component}</code><p className="muted">{t(componentInfo[c.component]?.[1] ?? "")}</p></td><td>{statuses[c.status] ?? c.status}<br />{c.scope === 'context' ? t("驾驶片段级（不按帧重复）") : c.scope === 'frame' ? t("当前帧") : t("范围未读取")} · {c.rows_in_scope ?? t("未知")}{t(" 条")}<br /><small>{c.source_check === 'matches_recorded_sources' ? t("来源指纹与原证据一致") : c.source_check === 'current_source_only' ? t("当前文件快照，未参与原证据构建") : c.source_check === 'mismatch' ? t("指纹不一致") : t("无已记录来源")}</small></td><td>{c.key_columns.includes('key.camera_name') ? t("相机 {0}：{1} 条", camera, c.camera_counts[String(camera)] ?? 0) : c.key_columns.includes('key.laser_name') ? t("雷达：{0} 个", Object.keys(c.laser_counts).length) : t("不按相机划分")}<br />{c.target_rows !== null ? t("全帧目标键：{0} 条", c.target_rows) : t("非三维目标键表")}{c.target_rows !== null && c.key_columns.includes('key.camera_name') && <><br />{t("当前相机目标键：")}{c.target_camera_counts[String(camera)] ?? 0}{t(" 条")}</>}</td><td><code>{c.file ?? t("无文件")}</code><details><summary>{t("关联主键与来源指纹")}</summary><pre>{c.key_columns.join('\n')}</pre><code>{c.sha256 ?? t("未读取")}</code></details></td></tr>)}</tbody></table></div>
  </details>
}
