const names: Record<string, string> = {
  FRONT: '前方', FRONT_LEFT: '左前方', FRONT_RIGHT: '右前方', SIDE_LEFT: '左侧', SIDE_RIGHT: '右侧',
}

export function cameraViewLabel(view?: string | null): string {
  return view ? names[view] ?? view : '未记录视角'
}
