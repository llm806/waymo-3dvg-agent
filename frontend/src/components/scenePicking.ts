export type ScreenBox = { id: string; points: number[][] }
function cross(a: number[], b: number[], c: number[]) { return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]) }
function hull(points: number[][]) {
  const sorted = points.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1])
  const lower: number[][] = [], upper: number[][] = []
  for (const p of sorted) { while(lower.length>1&&cross(lower.at(-2)!,lower.at(-1)!,p)<=0)lower.pop(); lower.push(p) }
  for (const p of sorted.slice().reverse()) { while(upper.length>1&&cross(upper.at(-2)!,upper.at(-1)!,p)<=0)upper.pop(); upper.push(p) }
  return [...lower.slice(0,-1),...upper.slice(0,-1)]
}
function edgeDistance(p: number[], a: number[], b: number[]) {
  const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy
  const t=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0
  return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)
}
export function pickBox(boxes: ScreenBox[], point: number[], tolerance=6): string | undefined {
  const matches = boxes.flatMap(box=>{
    const polygon=hull(box.points)
    if(polygon.length<2)return []
    const distance=Math.min(...polygon.map((p,i)=>edgeDistance(point,p,polygon[(i+1)%polygon.length])))
    const inside=polygon.length>=3&&polygon.every((p,i)=>cross(p,polygon[(i+1)%polygon.length],point)>=-0.01)
    if(!inside&&distance>tolerance)return []
    const area=Math.abs(polygon.reduce((sum,p,i)=>{const q=polygon[(i+1)%polygon.length];return sum+p[0]*q[1]-q[0]*p[1]},0))/2
    return [{id:box.id,inside,area,distance}]
  })
  // Prefer an enclosing box, then the smaller visible footprint for overlapping boxes.
  matches.sort((a,b)=>Number(b.inside)-Number(a.inside)||(a.inside?a.area-b.area:a.distance-b.distance)||a.id.localeCompare(b.id))
  return matches[0]?.id
}
