export type CameraView = { right: number[]; up: number[]; forward: number[]; origin: number[];
  intrinsic: number[]; width: number; height: number }

// Official camera axes: x forward, y left, z up. Image right is -y.
// Static calibrated preview; rolling-shutter motion is not re-estimated here.
export function cameraPixel(point: number[], camera: CameraView): number[] | null {
  const delta = point.map((v, i) => v - camera.origin[i])
  const dot = (axis: number[]) => axis.reduce((sum, v, i) => sum + v * delta[i], 0)
  const depth = dot(camera.forward)
  if (depth <= .1) return null
  const x = dot(camera.right) / depth, y = -dot(camera.up) / depth
  // Outside the view: avoid extrapolating the lens polynomial near the camera plane.
  if (Math.abs(x) > 2 || Math.abs(y) > 2) return null
  const [fu, fv, cu, cv, k1, k2, p1, p2, k3] = camera.intrinsic
  const r2 = x*x + y*y, radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
  return [fu*(x*radial + 2*p1*x*y + p2*(r2+2*x*x))+cu,
    fv*(y*radial + p1*(r2+2*y*y) + 2*p2*x*y)+cv]
}
