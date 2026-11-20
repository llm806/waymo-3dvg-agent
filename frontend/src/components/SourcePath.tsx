import { t, useLocale } from '../shared/locale.ts'
import { sourcePath } from '../shared/annotationDisplay'

export function SourcePath({ file, label = t("目标真值来源文件") }: { file: string | null | undefined; label?: string }) {
  useLocale()

  return <div className="source-path"><span>{label}</span><code>{sourcePath(file)}</code></div>
}
