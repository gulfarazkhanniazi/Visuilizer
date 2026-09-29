/**
 * Which pixels lie on the floor plane, by depth.
 *
 * The segmenter's "table" region often spills over the rug around a coffee
 * table, and its "chair" region covers the floor between the legs. A table
 * top or a seat stands well above the floor; a rug lies on it. With the floor
 * plane known (Y = 0 in the scanner's world frame), a pixel is on the floor
 * when its observed depth matches where its viewing ray meets that plane.
 */
import { dirToWorld } from './pointcloud.js';
import { bilinear } from './math.js';

/** Relative depth error of every pixel against the floor plane (NaN above the horizon). */
export function floorPlaneError(geo, RW, RH, w, h) {
  const { cam, frame, grid } = geo;
  const out = new Float32Array(RW * RH).fill(NaN);
  const H = frame.camHeight;
  for (let y = 0; y < RH; y++) {
    const v = ((y + 0.5) * h) / RH;
    for (let x = 0; x < RW; x++) {
      const u = ((x + 0.5) * w) / RW;
      const d = dirToWorld(frame, [(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1]);
      if (d[1] >= -1e-6) continue;                 // at or above the horizon
      const t = -H / d[1];                          // camera depth of the floor hit
      if (!(t > 0.2 && t < 15)) continue;
      const z = bilinear(grid.Z, grid.GW, grid.GH, ((x + 0.5) * grid.GW) / RW - 0.5, ((y + 0.5) * grid.GH) / RH - 0.5);
      out[y * RW + x] = Math.abs(z - t) / t;
    }
  }
  return out;
}
