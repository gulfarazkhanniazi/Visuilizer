/**
 * SAM 2 refinement of furniture silhouettes against walls (quality=high).
 *
 * Coarse -> uncertain -> precise: SegFormer says roughly where the sofa is;
 * its boundary with the wall is where it is least sure; SAM 2, prompted with
 * a box around each such object, draws that boundary precisely. SAM only
 * gets the objects that actually touch a wall, never the whole image, and
 * its answer is only allowed to move pixels inside a narrow band around the
 * existing boundary -- a confident SAM mask sharpens an edge, it does not
 * get to invent or delete furniture.
 */
import { refineWithSam } from './depth.js';

const MAX_BOXES = 8;
const MIN_SCORE = 0.7;

export async function refineOccluders(imagePath, seg) {
  const { RW, RH, w, h } = seg;
  const wall = seg.crisp?.wall;
  if (!wall) return { applied: false, reason: 'no wall mask' };
  const n = RW * RH;
  const surf = (i) => wall[i] || seg.crisp.floor?.[i] || seg.crisp.ceiling?.[i];
  // Object pixels: anything no surface claims.
  const obj = new Uint8Array(n);
  for (let i = 0; i < n; i++) obj[i] = surf(i) ? 0 : 1;

  // Connected object components that touch the wall.
  const label = new Int32Array(n).fill(-1);
  const comps = [];
  for (let s = 0; s < n; s++) {
    if (!obj[s] || label[s] >= 0) continue;
    const id = comps.length; const stack = [s]; label[s] = id;
    let x0 = RW; let y0 = RH; let x1 = 0; let y1 = 0; let size = 0; let touches = 0;
    while (stack.length) {
      const p = stack.pop(); size++;
      const x = p % RW; const y = (p / RW) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (const q of [p - 1, p + 1, p - RW, p + RW]) {
        if (q < 0 || q >= n || (q === p - 1 && x === 0) || (q === p + 1 && x === RW - 1)) continue;
        if (wall[q]) touches++;
        if (obj[q] && label[q] < 0) { label[q] = id; stack.push(q); }
      }
    }
    comps.push({ id, size, touches, box: [x0, y0, x1, y1] });
  }
  const picked = comps
    .filter((c) => c.touches > 20 && c.size > n * 0.003 && c.size < n * 0.4)
    .sort((a, b) => b.touches - a.touches)
    .slice(0, MAX_BOXES);
  if (!picked.length) return { applied: false, reason: 'no furniture touches a wall' };

  const sx = w / RW; const sy = h / RH;
  const pad = Math.round(RW * 0.03);
  const boxes = picked.map((c) => [
    Math.max(0, (c.box[0] - pad) * sx), Math.max(0, (c.box[1] - pad) * sy),
    Math.min(w, (c.box[2] + pad) * sx), Math.min(h, (c.box[3] + pad) * sy),
  ]);
  const t = Date.now();
  const masks = await refineWithSam(imagePath, boxes, RW);
  if (!masks) return { applied: false, reason: 'SAM unavailable' };

  // Band around the current object boundary: the only pixels SAM may move.
  const band = Math.max(3, Math.round(RW * 0.012));
  const dist = distanceToBoundary(obj, RW, RH, band);
  let toObject = 0; let toWall = 0; let used = 0;
  masks.forEach((m, k) => {
    if ((m.score ?? 0) < MIN_SCORE) return;
    used++;
    const c = picked[k];
    const [bx0, by0, bx1, by1] = [c.box[0] - pad, c.box[1] - pad, c.box[2] + pad, c.box[3] + pad];
    for (let y = Math.max(0, by0); y <= Math.min(RH - 1, by1); y++) {
      const my = Math.min(m.height - 1, Math.floor(((y + 0.5) * m.height) / RH));
      for (let x = Math.max(0, bx0); x <= Math.min(RW - 1, bx1); x++) {
        const i = y * RW + x;
        if (dist[i] > band) continue;
        const mx = Math.min(m.width - 1, Math.floor(((x + 0.5) * m.width) / RW));
        const sam = m.data[my * m.width + mx] > 127;
        if (sam && wall[i]) { wall[i] = 0; toObject++; }
        else if (!sam && obj[i] && label[i] === c.id && hasWallNear(wall, RW, RH, x, y, band)) { wall[i] = 255; toWall++; }
      }
    }
  });
  return {
    applied: used > 0, boxes: boxes.length, confident: used, pixelsToObject: toObject, pixelsToWall: toWall, ms: Date.now() - t,
  };
}

/** Chebyshev distance to the nearest object/non-object transition, capped. */
function distanceToBoundary(obj, W, H, cap) {
  const d = new Uint16Array(W * H).fill(cap + 1);
  const queue = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const b = (x > 0 && obj[i - 1] !== obj[i]) || (x < W - 1 && obj[i + 1] !== obj[i])
        || (y > 0 && obj[i - W] !== obj[i]) || (y < H - 1 && obj[i + W] !== obj[i]);
      if (b) { d[i] = 0; queue.push(i); }
    }
  }
  for (let q = 0; q < queue.length; q++) {
    const p = queue[q]; const x = p % W; const y = (p / W) | 0;
    if (d[p] >= cap) continue;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx; const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const j = yy * W + xx;
        if (d[j] > d[p] + 1) { d[j] = d[p] + 1; queue.push(j); }
      }
    }
  }
  return d;
}

function hasWallNear(wall, W, H, x, y, r) {
  for (let dy = -r; dy <= r; dy += 2) {
    for (let dx = -r; dx <= r; dx += 2) {
      const xx = x + dx; const yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H && wall[yy * W + xx]) return true;
    }
  }
  return false;
}
