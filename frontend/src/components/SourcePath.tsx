import { sourcePath } from '../shared/annotationDisplay'

export function SourcePath({ file, label = '目标真值来源文件' }: { file: string | null | undefined; label?: string }) {
  return <div className="source-path"><span>{label}</span><code>{sourcePath(file)}</code></div>
}
