/**
 * Automatic surface detection.
 *
 * Two halves:
 *
 *  1. WHERE the surfaces are -- a semantic segmentation (SegFormer trained on
 *     ADE20K) labels every pixel wall / floor / ceiling / sofa / curtain / ...
 *     The floor class already excludes the furniture standing on it, so the
 *     class map *is* the coverage mask; we only have to trace it into the
 *     editable polygons the rest of the app speaks.
 *
 *  2. HOW they sit in space -- segmentation says nothing about geometry, and a
 *     mask alone cannot lay a tile. For a level camera (which is how interior
 *     photographs are shot, to keep verticals vertical) the horizon runs
 *     through the principal point, and then the floor plane's homography is
 *     fully determined by the focal length and the camera height:
 *
 *         u = cx + f * X / Z          Z = f * h / (v - cy)
 *         v = cy + f * h / Z          X = (u - cx) * h / (v - cy)
 *
 *     Note X depends only on h, not on f: the lateral scale is set by camera
 *     height alone. So if the ceiling is visible we can solve h outright from
 *     the ratio of the floor and ceiling junctions against an assumed room
 *     height, and the metric scale comes out right rather than guessed.
 *
 *     Each wall is then found by un-projecting its own floor junction onto that
 *     solved floor plane, which gives the wall's base line in world space -- and
 *     a vertical plane through a known base line is a complete wall.
 */
import sharp from 'sharp';
import path from 'node:path';
import { DATA_DIR } from './db.js';
import {
  refineClassMaps, resampleMask, toLuma, upscaleMask, dilate,
} from './refine.js';
import { verticalCorners } from './corners.js';
import {
  runGeometryScan, publicWall, publicCorner, publicRejected, CONFIDENCE_BANDS, GEOMETRY_VERSION,
} from './scanner/run.js';
import { floorPlaneError } from './scanner/floor.js';

/** Shape of the `scan` block in a detection result. */
export const SCAN_API_VERSION = 'scan-1';

// ADE20K classes we care about. Everything else is simply not a surface.
const SURFACE_CLASSES = {
  floor: { key: 'floor', label: 'Floor', surface: 'floor' },
  wall: { key: 'wall', label: 'Wall', surface: 'wall' },
  ceiling: { key: 'ceiling', label: 'Ceiling', surface: 'ceiling' },
};

// Traced at this width, then scaled back up: smoother contours, far fewer
// points, and the tracing cost stops depending on the photo's resolution.
const TRACE_W = 720;

// Where the class maps are snapped to the photograph's edges. Wide enough that
// a skirting board and a chair leg are several pixels across -- below about
// 800 the guide has no edge left to snap to -- and no wider, because the
// filter's cost is per pixel and nothing downstream traces finer than this.
const REFINE_W = 1024;

// The pseudo-class every non-surface region is merged into. Named so it cannot
// collide with an ADE20K label.
const OTHER_KEY = '__occluder';

let segmenterPromise = null;
// Which model actually loaded (SEG_MODEL, or the fallback): reported with
// every scan, because a different segmenter changes the output.
let loadedSegModel = null;

/**
 * Which SegFormer to segment with.
 *
 * B0 is the smallest of the family and it shows: on an ordinary living room it
 * misses the pendant lamps entirely and paints wall straight over them, and it
 * puts the edge of a rug several centimetres out. B4 finds the lamps, holds the
 * rug boundary, and scores about 50 mIoU on ADE20K against B0's 37 -- and since
 * every mask in this app is downstream of that number, it is the single biggest
 * lever on quality there is.
 *
 * It is not a free win, which is why B0 stays the default: on the same living
 * room B4 under-segments the rug (4.6% of frame against B0's 6.6%, and the rug
 * really is nearer the larger figure), so more of it ends up tiled -- and it
 * costs about twice the inference time and a much larger one-off download.
 * Set SEG_MODEL=Xenova/segformer-b4-finetuned-ade-512-512 to trade that for
 * the better object detection.
 */
const SEG_MODEL = process.env.SEG_MODEL || 'Xenova/segformer-b0-finetuned-ade-512-512';
const SEG_FALLBACK = 'Xenova/segformer-b0-finetuned-ade-512-512';

async function loadModel(id) {
  const { pipeline, env } = await import('@huggingface/transformers');
  env.allowLocalModels = false;
  // Keep the downloaded weights out of node_modules: writing there makes
  // node --watch restart the server in the middle of the first request.
  env.cacheDir = path.join(DATA_DIR, 'models');
  // q8 weights: materially less memory for no visible difference in the
  // masks, which matters because this runs alongside the user's browser.
  return pipeline('image-segmentation', id, { dtype: 'q8' });
}

/**
 * Load the model once and keep it warm; the first call pays the download.
 *
 * SEG_MODEL selects a different one. Xenova/segformer-b4-finetuned-ade-512-512
 * scores about 50 mIoU on ADE20K against B0's 37 and finds the pendant lamps
 * B0 paints over -- but it under-segments a rug that B0 gets right, and costs
 * about twice the inference time, so B0 remains the default.
 */
async function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = (async () => {
      try {
        const m = await loadModel(SEG_MODEL);
        loadedSegModel = SEG_MODEL;
        return m;
      } catch (e) {
        if (SEG_MODEL === SEG_FALLBACK) throw e;
        console.error(`  detect   ${SEG_MODEL} failed to load (${e.message}); using ${SEG_FALLBACK}`);
        const m = await loadModel(SEG_FALLBACK);
        loadedSegModel = SEG_FALLBACK;
        return m;
      }
    })();
  }
  return segmenterPromise;
}

export async function warmUp() {
  try { await getSegmenter(); return true; } catch { return false; }
}

/* ------------------------------------------------------------- contours -- */

/** Nearest-neighbour downsample of a byte mask to `tw` wide. */
function downsample(data, w, h, tw) {
  const th = Math.max(1, Math.round((h / w) * tw));
  const out = new Uint8Array(tw * th);
  for (let y = 0; y < th; y++) {
    const sy = Math.min(h - 1, Math.floor((y * h) / th));
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(w - 1, Math.floor((x * w) / tw));
      out[y * tw + x] = data[sy * w + sx] > 127 ? 1 : 0;
    }
  }
  return { data: out, width: tw, height: th };
}

/** 4-connected labelling. `target` picks foreground (1) or background (0). */
function components(bin, w, h, target) {
  const label = new Int32Array(w * h).fill(-1);
  const out = [];
  const stack = [];
  for (let i = 0; i < w * h; i++) {
    if (bin[i] !== target || label[i] !== -1) continue;
    const id = out.length;
    let size = 0;
    let touchesBorder = false;
    stack.push(i);
    label[i] = id;
    while (stack.length) {
      const p = stack.pop();
      size++;
      const x = p % w;
      const y = (p / w) | 0;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) touchesBorder = true;
      if (x > 0 && bin[p - 1] === target && label[p - 1] === -1) { label[p - 1] = id; stack.push(p - 1); }
      if (x < w - 1 && bin[p + 1] === target && label[p + 1] === -1) { label[p + 1] = id; stack.push(p + 1); }
      if (y > 0 && bin[p - w] === target && label[p - w] === -1) { label[p - w] = id; stack.push(p - w); }
      if (y < h - 1 && bin[p + w] === target && label[p + w] === -1) { label[p + w] = id; stack.push(p + w); }
    }
    out.push({ id, size, seed: i, touchesBorder });
  }
  return { label, list: out };
}

/** Moore-neighbour boundary trace of one labelled component. */
function traceComponent(label, w, h, id, seed) {
  const inside = (x, y) => x >= 0 && y >= 0 && x < w && y < h && label[y * w + x] === id;
  const start = [seed % w, (seed / w) | 0];
  const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

  const pts = [];
  let cur = start;
  let dir = 6;                      // came from "up"
  const limit = w * h * 4;
  let guard = 0;

  do {
    pts.push([cur[0], cur[1]]);
    let found = false;
    for (let k = 0; k < 8; k++) {
      const d = (dir + 6 + k) % 8;  // start looking back-left of travel
      const nx = cur[0] + dirs[d][0];
      const ny = cur[1] + dirs[d][1];
      if (inside(nx, ny)) { cur = [nx, ny]; dir = d; found = true; break; }
    }
    if (!found) break;
    guard++;
  } while ((cur[0] !== start[0] || cur[1] !== start[1]) && guard < limit);

  return pts;
}

/** Ramer-Douglas-Peucker, iterative so a long contour cannot blow the stack. */
function simplify(points, epsilon) {
  if (points.length < 3) return points;
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];

  while (stack.length) {
    const [a, b] = stack.pop();
    if (b <= a + 1) continue;
    const [ax, ay] = points[a];
    const [bx, by] = points[b];
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    let best = -1;
    let bestD = epsilon;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = points[i];
      const d = Math.abs(dy * px - dx * py + bx * ay - by * ax) / len;
      if (d > bestD) { bestD = d; best = i; }
    }
    if (best > 0) {
      keep[best] = 1;
      stack.push([a, best], [best, b]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

/**
 * Binary morphology on the traced-resolution mask.
 *
 * `pass` of 1 dilates, 0 erodes. A 4-neighbourhood is enough here: the point
 * is not to reshape anything, only to take the pepper off a segmentation edge
 * before it becomes polygon vertices.
 */
function morph(bin, w, h, grow) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let acc = bin[i];
      for (let k = 0; k < 4; k++) {
        const nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
        const ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);
        const v = (nx < 0 || ny < 0 || nx >= w || ny >= h) ? bin[i] : bin[ny * w + nx];
        acc = grow ? Math.max(acc, v) : Math.min(acc, v);
      }
      out[i] = acc;
    }
  }
  return out;
}

/**
 * Morphological cleanup. Off by default, and kept only for the cases that
 * still want it.
 *
 * It predates the guided filter. Back when the masks came straight from a
 * 512x512 class map they were speckled and needed it; now they are snapped to
 * the photograph's own edges before they get here, and opening -- which
 * rounds off every convex corner -- actively destroys the precision that
 * bought. On a dining chair it lifts the floor mask up over the seat.
 *
 * Open, and by default do not close.
 *
 * A class map straight out of the segmenter is speckled: stray pixels of wall
 * inside the sofa, stray pixels of sofa inside the wall. Traced as-is every
 * speck becomes its own polygon or a spike on someone else's. Opening -- erode
 * then dilate -- removes them, and nothing else here does.
 *
 * Closing was doing the opposite job and doing damage with it. Its purpose is
 * to fill pinholes, but a pinhole and a chair leg look identical to it, so it
 * bridged the gap around every thin object standing on the floor and swallowed
 * it into the mask. Small holes are better dealt with where the polygons are
 * traced, which can ask whether an object accounts for the hole before cutting
 * it out; morphology cannot ask anything.
 */
function despeckle(bin, w, h, open = 1, close = 0) {
  let m = bin;
  for (let i = 0; i < open; i++) m = morph(m, w, h, 0);
  for (let i = 0; i < open + close; i++) m = morph(m, w, h, 1);
  for (let i = 0; i < close; i++) m = morph(m, w, h, 0);
  return m;
}

/**
 * Simplify, then keep simplifying until the contour is small enough to edit.
 *
 * Every point becomes a draggable handle in the Studio, so a 400-vertex
 * outline is not a better mask -- it is an uneditable one. Doubling epsilon
 * converges quickly and always terminates.
 */
function simplifyCapped(points, epsilon, maxPoints = 110) {
  let eps = epsilon;
  let out = simplify(points, eps);
  for (let i = 0; i < 8 && out.length > maxPoints; i++) {
    eps *= 1.7;
    out = simplify(points, eps);
  }
  return out;
}

/**
 * Turn a raster mask into editable polygons: the outer contour of each
 * sizeable blob as `add`, and any enclosed background as `subtract`.
 */
function maskToPolygons(data, w, h, {
  minAreaFrac = 0.004, epsilon = 1.1, clean = 0, close = 0, occluder = null,
} = {}) {
  const small = downsample(data, w, h, Math.min(TRACE_W, w));
  if (clean > 0 || close > 0) small.data = despeckle(small.data, small.width, small.height, clean, close);
  const sx = w / small.width;
  const sy = h / small.height;
  const total = small.width * small.height;

  const fg = components(small.data, small.width, small.height, 1);
  const polys = [];

  for (const c of fg.list) {
    if (c.size / total < minAreaFrac) continue;
    const pts = simplifyCapped(
      traceComponent(fg.label, small.width, small.height, c.id, c.seed),
      epsilon,
    );
    if (pts.length < 3) continue;
    polys.push({
      mode: 'add',
      points: pts.map(([x, y]) => [round1((x + 0.5) * sx), round1((y + 0.5) * sy)]),
    });
  }
  if (!polys.length) return [];

  // Background blobs that do not reach the border are holes -- a picture on
  // the wall, a rug on the floor -- and must be cut back out.
  //
  // But only if something is actually there. A hole no object accounts for is
  // the segmenter having been unsure, not a thing in the room, and cutting it
  // out leaves an untiled island in the middle of a wall for no reason. So a
  // hole is kept as a cut-out when the occluder map explains enough of it, and
  // otherwise it is simply left filled -- which it already is, because the
  // outer contour encloses it.
  const bg = components(small.data, small.width, small.height, 0);
  const holes = { cut: 0, filled: 0 };
  for (const c of bg.list) {
    if (c.touchesBorder) continue;
    if (c.size / total < minAreaFrac * 0.5) continue;

    if (occluder) {
      let explained = 0;
      let seen = 0;
      for (let y = 0; y < small.height; y++) {
        for (let x = 0; x < small.width; x++) {
          if (bg.label[y * small.width + x] !== c.id) continue;
          seen++;
          const fx = Math.min(w - 1, Math.floor((x + 0.5) * sx));
          const fy = Math.min(h - 1, Math.floor((y + 0.5) * sy));
          if (occluder[fy * w + fx]) explained++;
        }
      }
      if (seen && explained / seen < 0.35) { holes.filled++; continue; }
    }

    const pts = simplifyCapped(
      traceComponent(bg.label, small.width, small.height, c.id, c.seed),
      epsilon,
      70,
    );
    if (pts.length < 3) continue;
    holes.cut++;
    polys.push({
      mode: 'subtract',
      points: pts.map(([x, y]) => [round1((x + 0.5) * sx), round1((y + 0.5) * sy)]),
    });
  }
  polys.holes = holes;
  return polys;
}

const round1 = (n) => Math.round(n * 10) / 10;

/**
 * Objects labelled cabinet/wardrobe/shelf that are really the wall surface:
 * a whole region whose pixels mostly lie flush on one wall's plane (wood
 * panelling reads as "cabinet"). A real cabinet stands proud of the wall,
 * with at most an edge near it, and is left out. Returns, per wall index, a
 * mask of the approved regions plus a small margin (the band the occluder
 * map adds around objects), or undefined.
 */
function findFlushPanels(candidates, flush, W, H, { minFlush = 0.7, margin = 6 } = {}) {
  const out = [];
  if (!candidates || !flush?.length) return out;
  const seen = new Uint8Array(W * H);
  for (let s = 0; s < W * H; s++) {
    if (candidates[s] < 0.5 || seen[s]) continue;
    const pix = []; const stack = [s]; seen[s] = 1;
    while (stack.length) {
      const p = stack.pop(); pix.push(p);
      const x = p % W;
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W]) {
        if (q < 0 || q >= W * H || seen[q] || candidates[q] < 0.5) continue;
        seen[q] = 1; stack.push(q);
      }
    }
    if (pix.length < W * H * 0.001) continue;
    let bestK = -1; let best = 0;
    flush.forEach((f, k) => {
      let c = 0;
      for (const p of pix) if (f[p]) c++;
      if (c > best) { best = c; bestK = k; }
    });
    if (bestK < 0 || best / pix.length < minFlush) continue;
    out[bestK] ??= new Uint8Array(W * H);
    const m = new Uint8Array(W * H);
    for (const p of pix) m[p] = 1;
    const grown = dilate(m, W, H, margin);
    for (let i = 0; i < W * H; i++) if (grown[i]) out[bestK][i] = 255;
  }
  return out;
}

/**
 * The floor, rebuilt once the floor plane is known.
 *
 * The segmenter's labels spill: "coffee table" over the rug around and behind
 * the table, "chair" over the floor between the legs. Cutting those out
 * leaves bare floor round every piece of furniture. Depth settles it: a table
 * top or a seat stands well above the floor, a rug lies on it. So a pixel
 * labelled as furniture whose depth puts it on the floor plane is floor; only
 * furniture above the floor is cut (tables with their outline, so a glass top
 * is still cut whole). Tables get a looser tolerance -- depth smears a table's
 * height a little onto the rug beside it, and a table top 30+ cm up is far
 * outside it. Chairs and other furniture are cut whole, as they always were.
 */
function refineFloorWithDepth(seg, geo, { tableTol = 0.07, furnitureTol = 0.035, edgeStep = 0.04 } = {}) {
  const { RW, RH, w, h, crisp } = seg;
  const { before, furniture, tables } = seg.floorParts;
  const n = RW * RH;
  const err = floorPlaneError(geo, RW, RH, w, h);
  const floor = Uint8Array.from(before);
  const realTables = new Float32Array(n);
  const hard = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (crisp.wall?.[i] || crisp.ceiling?.[i]) continue;
    const e = err[i];
    const isTable = !!tables && tables[i] >= 0.5;
    // Chairs, beds, stools...: cut out whole, as before. Their legs are too
    // thin for depth to see or for a traced outline to keep as holes, so
    // reclaiming the floor between them would tile over the legs.
    if (furniture[i] >= 0.01) { hard[i] = 1; continue; }
    // Table-labelled pixels above the floor are the table. Those on the
    // floor plane may be rug under a spilled label -- or a table's foot,
    // which depth cannot tell from the floor it stands on -- so they are left
    // for the edge-checked growth below to decide.
    if (isTable && !(e <= tableTol)) { realTables[i] = 1; hard[i] = 1; }
  }
  const occluder = dilate(hard, RW, RH, 3);
  for (let i = 0; i < n; i++) if (occluder[i]) floor[i] = 0;
  const inTable = new Uint8Array(n);
  if (tables) cutTableHulls(floor, realTables, RW, RH, inTable);
  dropEnclosedFloor(floor, occluder, RW, RH);

  // Finally, grow the floor into every neighbouring pixel the depth puts
  // firmly on the floor plane, whatever the segmenter called it: floor it
  // labelled "sofa" along a sofa's base, "windowpane" by a window, the margin
  // cut around furniture, the gap between a table and the unit behind it.
  // Only through connected pixels, never into wall or ceiling, and only at
  // the tight tolerance. Monocular depth blurs thin things -- a chair leg
  // reads at the depth of the floor behind it -- so the growth also stops at
  // a clear edge in the photograph: floor mislabelled as furniture looks like
  // the floor beside it; a leg does not.
  const luma = seg.guide;
  const queue = [];
  for (let i = 0; i < n; i++) if (floor[i]) queue.push(i);
  for (let q = 0; q < queue.length; q++) {
    const p = queue[q];
    const x = p % RW;
    for (const j of [x > 0 ? p - 1 : -1, x < RW - 1 ? p + 1 : -1, p - RW, p + RW]) {
      if (j < 0 || j >= n || floor[j] || crisp.wall?.[j] || crisp.ceiling?.[j]) continue;
      if (occluder[j] && furniture[j] >= 0.01) continue;   // chairs etc. stay cut out
      if (inTable[j]) continue;                            // inside a table's outline
      const isTableLabel = !!tables && tables[j] >= 0.5;
      if (!(err[j] <= (isTableLabel ? tableTol : furnitureTol))) continue;
      if (luma && Math.abs(luma[j] - luma[p]) > edgeStep) continue;
      floor[j] = 255;
      queue.push(j);
    }
  }
  return floor;
}

/** Clear object pixels from a wall's painted area (in place). */
function removeObjects(mask, objects) {
  if (objects) for (let i = 0; i < mask.length; i++) if (objects[i]) mask[i] = 0;
  return mask;
}

/**
 * Cut each table's convex outline out of the floor. A table top is convex,
 * and a glass one is labelled only where its frame is, so its pixels alone
 * leave the floor showing through the top.
 */
function cutTableHulls(floor, tables, W, H, inside = null) {
  const seen = new Uint8Array(W * H);
  for (let s = 0; s < W * H; s++) {
    if (tables[s] < 0.5 || seen[s]) continue;
    const pts = []; const stack = [s]; seen[s] = 1;
    while (stack.length) {
      const p = stack.pop(); const x = p % W; const y = (p / W) | 0;
      pts.push([x, y]);
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W]) {
        if (q < 0 || q >= W * H || seen[q] || tables[q] < 0.5) continue;
        seen[q] = 1; stack.push(q);
      }
    }
    if (pts.length < W * H * 0.001) continue;
    const hull = convexHull(pts);
    let x0 = W; let x1 = 0; let y0 = H; let y1 = 0;
    for (const [x, y] of hull) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (!insideConvex(hull, x, y)) continue;
        floor[y * W + x] = 0;
        if (inside) inside[y * W + x] = 1;
      }
    }
  }
}

function convexHull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = []; const hi = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (hi.length >= 2 && cr(hi[hi.length - 2], hi[hi.length - 1], q) <= 0) hi.pop(); hi.push(q); }
  return lo.slice(0, -1).concat(hi.slice(0, -1));
}

function insideConvex(hull, x, y) {
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i]; const b = hull[(i + 1) % hull.length];
    if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) return false;
  }
  return true;
}

/**
 * Remove floor seen only through or under furniture: a patch of floor cut off
 * from the main floor whose border is mostly furniture -- the rug under a
 * glass table top. Tiling it would put the floor design on the table.
 */
function dropEnclosedFloor(floor, furniture, W, H) {
  const label = new Int32Array(W * H).fill(-1);
  const comps = [];
  for (let s = 0; s < W * H; s++) {
    if (!floor[s] || label[s] >= 0) continue;
    const id = comps.length; const pix = []; const stack = [s]; label[s] = id;
    while (stack.length) {
      const p = stack.pop(); pix.push(p);
      const x = p % W;
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W]) {
        if (q < 0 || q >= W * H || !floor[q] || label[q] >= 0) continue;
        label[q] = id; stack.push(q);
      }
    }
    comps.push(pix);
  }
  if (comps.length < 2) return;
  const main = comps.reduce((a, b) => (b.length > a.length ? b : a));
  for (const pix of comps) {
    if (pix === main) continue;
    const id = label[pix[0]];
    let ring = 0; let furn = 0;
    for (const p of pix) {
      const x = p % W;
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W]) {
        if (q < 0 || q >= W * H || label[q] === id) continue;
        ring++;
        if (furniture[q]) furn++;
      }
    }
    if (ring && furn / ring >= 0.6) for (const p of pix) floor[p] = 0;
  }
}

/** Do segments p1-p2 and p3-p4 properly cross (touching endpoints do not count)? */
function segmentsCross(p1, p2, p3, p4) {
  const o = (a, b, c) => Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
  return o(p1, p2, p3) * o(p1, p2, p4) < 0 && o(p3, p4, p1) * o(p3, p4, p2) < 0;
}

export function selfIntersects(pts) {
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const a = pts[i]; const b = pts[(i + 1) % n];
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;          // adjacent through the wrap
      if (segmentsCross(a, b, pts[j], pts[(j + 1) % n])) return true;
    }
  }
  return false;
}

/**
 * Remove self-intersections with 2-opt moves: when edges (a,b) and (c,d)
 * cross, reversing the run b..c replaces them with (a,c) and (b,d), which do
 * not. Every move shortens the perimeter, so it terminates; nothing is
 * deleted, so the outline stays where the mask is. Simplified traces cross
 * themselves only at pinches, which one or two moves undo.
 */
export function untangle(points, maxMoves = 200) {
  const pts = points.slice();
  const n = pts.length;
  for (let move = 0; move < maxMoves; move++) {
    let found = false;
    for (let i = 0; i < n && !found; i++) {
      for (let j = i + 2; j < n && !found; j++) {
        if (i === 0 && j === n - 1) continue;
        if (segmentsCross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n])) {
          let a = i + 1; let b = j;
          while (a < b) { [pts[a], pts[b]] = [pts[b], pts[a]]; a++; b--; }
          found = true;
        }
      }
    }
    if (!found) break;
  }
  return pts;
}

export function polygonArea(pts) {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i]; const [x2, y2] = pts[(i + 1) % pts.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

/**
 * Every polygon handed to the Studio must be a valid simple polygon inside
 * the image with real area. Traced contours almost always are; one that is
 * not is repaired by re-simplifying (which removes the pinch that made it
 * cross itself) and dropped if that fails -- never shipped broken.
 */
export function validatePolygons(polys, w, h) {
  const notes = [];
  const out = [];
  const minArea = w * h * 1e-4;
  for (const p of polys) {
    let pts = p.points.map(([x, y]) => [Math.min(w, Math.max(0, x)), Math.min(h, Math.max(0, y))]);
    if (pts.length < 3) { notes.push(`${p.mode} polygon dropped: fewer than 3 points`); continue; }
    // Untangle before measuring: a figure-8's signed areas cancel out.
    const crossed = selfIntersects(pts);
    if (crossed) pts = untangle(pts);
    if (selfIntersects(pts)) { notes.push(`${p.mode} polygon dropped: self-intersecting and not repairable`); continue; }
    if (polygonArea(pts) < minArea) { notes.push(`${p.mode} polygon dropped: no area`); continue; }
    if (crossed) notes.push(`${p.mode} polygon repaired (self-intersection)`);
    out.push({ ...p, points: pts.map(([x, y]) => [round1(x), round1(y)]) });
  }
  const res = out;
  res.holes = polys.holes;
  return { polygons: res, notes };
}

/* ------------------------------------------------------------- geometry -- */

/**
 * Camera model for a level shot: horizon through the principal point.
 * Solves the camera height from the ceiling if it is visible, because the
 * lateral metric scale depends on height alone.
 */
function solveCamera(masks, w, h, { roomHeight, hfov }) {
  const cx = w / 2;
  const cy = h / 2;
  const f = (w / 2) / Math.tan((hfov * Math.PI) / 360);

  let height = 1.5;
  let source = 'assumed';

  const floor = masks.floor;
  const ceiling = masks.ceiling;
  if (floor && ceiling) {
    // Per column, how far the floor sits below the horizon vs the ceiling
    // above it. Their ratio fixes where the camera sits in the room height.
    const ratios = [];
    for (let x = 0; x < w; x += Math.max(1, Math.floor(w / 160))) {
      let vFloor = -1;
      for (let y = h - 1; y > cy; y--) if (floor[y * w + x] > 127) { vFloor = y; break; }
      let vCeil = -1;
      for (let y = 0; y < cy; y++) if (ceiling[y * w + x] > 127) { vCeil = y; break; }
      if (vFloor < 0 || vCeil < 0) continue;
      const A = vFloor - cy;          // f*h/Z
      const B = cy - vCeil;           // f*(H-h)/Z
      if (A > 4 && B > 4) ratios.push(A / (A + B));
    }
    if (ratios.length > 8) {
      ratios.sort((a, b) => a - b);
      const med = ratios[Math.floor(ratios.length / 2)];
      const solved = roomHeight * med;
      if (solved > 0.7 && solved < 2.4) { height = solved; source = 'ceiling'; }
    }
  }
  return { f, cx, cy, height, source };
}

/** Image point on the floor plane -> world (X, Z) in metres. */
function floorPoint(cam, u, v) {
  const d = v - cam.cy;
  if (d <= 1e-3) return null;                 // at or above the horizon
  const Z = (cam.f * cam.height) / d;
  const X = ((u - cam.cx) * cam.height) / d;
  return [X, Z];
}

/** Image point on the ceiling plane (Y = H) -> world (X, Z) in metres. */
function ceilingPoint(cam, u, v, H) {
  const d = cam.cy - v;
  if (d <= 1e-3) return null;                 // at or below the horizon
  const Z = (cam.f * (H - cam.height)) / d;
  const X = ((u - cam.cx) * (H - cam.height)) / d;
  return [X, Z];
}

/** World point -> image. */
function project(cam, X, Y, Z) {
  if (Z <= 1e-4) return null;
  return [cam.cx + (cam.f * X) / Z, cam.cy + (cam.f * (cam.height - Y)) / Z];
}

/** Vertical image extent of a mask, and its horizontal span. */
function maskExtent(mask, w, h) {
  let minY = h; let maxY = -1; let minX = w; let maxX = -1; let count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] <= 127) continue;
      count++;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
    }
  }
  return count ? { minX, maxX, minY, maxY, count } : null;
}

/**
 * A floor or ceiling rectangle spanning what is actually visible.
 *
 * The two are mirror images about the horizon and must be un-projected onto
 * their own plane: a ceiling read with the floor's equation comes back tens of
 * metres across, because the sign of (v - cy) is inverted and the height above
 * the camera is not the height below it.
 */
function horizontalQuad(cam, mask, w, h, y0) {
  const ext = maskExtent(mask, w, h);
  if (!ext) return null;

  const ceiling = y0 > 0.01;
  if (ceiling && !(y0 > cam.height + 0.2)) return null;
  const unproject = ceiling
    ? (u, v) => ceilingPoint(cam, u, v, y0)
    : (u, v) => floorPoint(cam, u, v);

  // Stay well away from the horizon. Right at it depth goes to infinity, and
  // even near it the un-projection is so ill-conditioned that a pixel of noise
  // in the mask edge moves the far corner by metres -- which is how a 4 m room
  // ends up with a 9 m floor quad. The far edge is the one nearest the
  // horizon, which is the bottom of a ceiling and the top of a floor.
  const margin = Math.max(10, h * 0.045);
  let vFar;
  let vNear;
  if (ceiling) {
    vFar = Math.min(ext.maxY, cam.cy - margin);
    vNear = Math.min(vFar - 4, ext.minY);
    if (!(vFar > 0) || !(vNear < vFar)) return null;
  } else {
    vFar = Math.max(ext.minY, cam.cy + margin);
    vNear = Math.max(vFar + 4, ext.maxY);
  }

  const far = unproject(cam.cx, vFar);
  const near = unproject(cam.cx, vNear);
  if (!far || !near) return null;
  const Zfar = far[1];
  const Znear = near[1];

  // Widen to cover the mask horizontally at the near edge, where it is widest.
  const left = unproject(ext.minX, vNear);
  const right = unproject(ext.maxX, vNear);
  const halfW = Math.max(0.5, Math.max(Math.abs(left?.[0] ?? 1), Math.abs(right?.[0] ?? 1)));

  const depth = Math.abs(Zfar - Znear);
  // Ceilings only. The same refusal the walls make -- a surface this size is a
  // bad solve, not a big room -- but applying it to floors would be a change
  // to behaviour that has been working, so it is scoped to the path it was
  // written for.
  if (ceiling && (halfW * 2 > 14 || depth > 14 || depth < 0.3)) return null;

  const corners = [
    project(cam, -halfW, y0, Zfar),
    project(cam, halfW, y0, Zfar),
    project(cam, halfW, y0, Znear),
    project(cam, -halfW, y0, Znear),
  ];
  if (corners.some((c) => !c)) return null;

  return {
    quad: corners.map(([x, y]) => [round1(x), round1(y)]),
    realSize: { w: round2(halfW * 2), h: round2(depth) },
  };
}

/**
 * Collect wall/floor and wall/ceiling junction points, un-projected to world.
 *
 * A wall is vertical, so both junctions lie on the very same vertical plane and
 * constrain the same base line in XZ. Skirting is hidden behind furniture far
 * more often than cornice is, so pooling the two is what makes this work in
 * furnished rooms.
 */
function collectJunctions(cam, mask, w, h, floorMask, ceilingMask, roomHeight) {
  const samples = [];
  const stepX = Math.max(1, Math.floor(w / 260));
  // Depth blows up towards the horizon, so a junction has to sit a clear
  // distance from it before its un-projection means anything.
  const minDrop = Math.max(8, h * 0.045);

  for (let x = 0; x < w; x += stepX) {
    let vBase = -1;
    for (let y = h - 1; y >= 0; y--) if (mask[y * w + x] > 127) { vBase = y; break; }
    if (vBase < 0) continue;

    // Only a wall pixel with floor directly beneath it is a real junction.
    // Without this test the bottom of a sofa reads as the skirting and drags
    // the fitted wall metres out of place.
    if (vBase - cam.cy >= minDrop && floorMask) {
      let touchesFloor = false;
      for (let dy = 1; dy <= 6 && vBase + dy < h; dy++) {
        if (floorMask[(vBase + dy) * w + x] > 127) { touchesFloor = true; break; }
      }
      if (touchesFloor) {
        const q = floorPoint(cam, x, vBase);
        if (q && q[1] > 0.5 && q[1] < 14) samples.push({ u: x, X: q[0], Z: q[1], src: 'floor' });
      }
    }

    if (ceilingMask) {
      let vTop = -1;
      for (let y = 0; y < h; y++) if (mask[y * w + x] > 127) { vTop = y; break; }
      if (vTop >= 0 && cam.cy - vTop >= minDrop) {
        let touchesCeiling = false;
        for (let dy = 1; dy <= 6 && vTop - dy >= 0; dy++) {
          if (ceilingMask[(vTop - dy) * w + x] > 127) { touchesCeiling = true; break; }
        }
        if (touchesCeiling) {
          const q = ceilingPoint(cam, x, vTop, roomHeight);
          if (q && q[1] > 0.5 && q[1] < 14) samples.push({ u: x, X: q[0], Z: q[1], src: 'ceiling' });
        }
      }
    }
  }
  return samples;
}



/** Turn one straight piece of the junction run into a wall surface. */
function planeFromRun(cam, origin, dir, inliers, mask, w, h, roomHeight) {

  const ss = inliers.map((s) => (s.X - origin[0]) * dir[0] + (s.Z - origin[1]) * dir[1]);
  ss.sort((a, b) => a - b);
  const s0 = ss[Math.floor(ss.length * 0.02)];
  const s1 = ss[Math.floor(ss.length * 0.98)];
  if (!(s1 - s0 > 0.3)) return { reason: `run too short (${(s1 - s0).toFixed(2)}m)` };

  const us = inliers.map((s) => s.u).sort((a, b) => a - b);
  const uMin = us[0];
  const uMax = us[us.length - 1];

  // If the cornice was in the fit, the wall runs the full room height.
  // Otherwise derive it: the top of the wall mask sits on the same world
  // vertical line as the column's base point, so its height follows directly.
  let height = roomHeight;
  if (!inliers.some((s) => s.src === 'ceiling')) {
    const mid = inliers[Math.floor(inliers.length / 2)];
    let vTop = 0;
    for (let y = 0; y < h; y++) if (mask[y * w + mid.u] > 127) { vTop = y; break; }
    height = cam.height - ((vTop - cam.cy) * mid.Z) / cam.f;
    // Same bound the fronto-parallel fallback uses. A measured wall taller
    // than this in an ordinary interior is a bad fit, not a tall room, and
    // believing it scales every tile on that wall down to match.
    if (!(height > 1.2 && height < 3.6)) height = roomHeight;
  }

  const at = (sv, t) => project(cam, origin[0] + dir[0] * sv, t, origin[1] + dir[1] * sv);
  const corners = [at(s0, height), at(s1, height), at(s1, 0), at(s0, 0)];
  if (corners.some((c) => !c)) return { reason: 'corner behind camera' };

  const realW = s1 - s0;
  // A wall wider than this is a bad fit, not a big room; better to drop the
  // surface and let the author draw it than to ship nonsense geometry. Nine
  // metres is already generous for one wall of a photographed interior -- the
  // old fourteen let a line fitted across two different walls through.
  if (realW > 9 || realW < 0.4) return { reason: `implausible width (${realW.toFixed(1)}m)` };

  return {
    quad: corners.map(([x, y]) => [round1(x), round1(y)]),
    realSize: { w: round2(realW), h: round2(height) },
    uMin,
    uMax,
    count: inliers.length,
    // The plane itself, kept so pixels can be tested against it rather than
    // against a vertical cut through the image.
    origin,
    dir,
    s0,
    s1,
  };
}


/**
 * Last resort for a wall with no visible junction at all: assume it faces the
 * camera, at the depth where the floor runs out.
 *
 * This is not a wild guess -- the junction is hidden precisely when furniture
 * is pushed flat against the wall, which is also when the shot is square-on to
 * it. The floor's far edge is that wall's base, so un-projecting it gives the
 * depth, and a fronto-parallel plane there is the wall.
 */
function frontoParallelWall(cam, mask, w, h, floorMask, roomHeight) {
  if (!floorMask) return null;

  // Where the floor runs out, looking only at columns this wall occupies.
  const depths = [];
  const stepX = Math.max(1, Math.floor(w / 200));
  for (let x = 0; x < w; x += stepX) {
    let hasWall = false;
    for (let y = 0; y < h; y++) if (mask[y * w + x] > 127) { hasWall = true; break; }
    if (!hasWall) continue;
    let vFar = -1;
    for (let y = 0; y < h; y++) if (floorMask[y * w + x] > 127) { vFar = y; break; }
    if (vFar < 0 || vFar - cam.cy < Math.max(6, h * 0.02)) continue;
    const q = floorPoint(cam, x, vFar);
    if (q && q[1] > 0.5 && q[1] < 16) depths.push(q[1]);
  }
  if (depths.length < 6) return null;
  depths.sort((a, b) => a - b);
  const Z = depths[Math.floor(depths.length / 2)];

  const ext = maskExtent(mask, w, h);
  if (!ext) return null;
  const X0 = ((ext.minX - cam.cx) * Z) / cam.f;
  const X1 = ((ext.maxX - cam.cx) * Z) / cam.f;
  // If the wall runs off the top of the frame its true height is unknowable,
  // so fall back to the assumed room height rather than reading the crop as
  // a five-metre wall and scaling every tile down to match.
  let height = roomHeight;
  if (ext.minY > 2) {
    const measured = cam.height - ((ext.minY - cam.cy) * Z) / cam.f;
    if (measured > 1.2 && measured < 3.6) height = measured;
  }

  const at = (X, Y) => project(cam, X, Y, Z);
  const corners = [at(X0, height), at(X1, height), at(X1, 0), at(X0, 0)];
  if (corners.some((c) => !c)) return null;

  const realW = Math.abs(X1 - X0);
  if (realW > 14 || realW < 0.4) return null;

  return {
    quad: corners.map(([x, y]) => [round1(x), round1(y)]),
    realSize: { w: round2(realW), h: round2(height) },
    uMin: ext.minX,
    uMax: ext.maxX,
    count: depths.length,
    assumed: true,
    // Constant depth, so the base line runs along X at Z.
    origin: [0, Z],
    dir: [1, 0],
    s0: Math.min(X0, X1),
    s1: Math.max(X0, X1),
  };
}

/**
 * Prefix sums of the junction points, so the best straight-line fit through
 * any range is O(1) instead of O(range).
 *
 * The segmentation below tries every possible corner position at every level,
 * which is only affordable because of this.
 */
function junctionSums(samples) {
  const n = samples.length;
  const sx = new Float64Array(n + 1);
  const sz = new Float64Array(n + 1);
  const sxx = new Float64Array(n + 1);
  const szz = new Float64Array(n + 1);
  const sxz = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const { X, Z } = samples[i];
    sx[i + 1] = sx[i] + X;
    sz[i + 1] = sz[i] + Z;
    sxx[i + 1] = sxx[i] + X * X;
    szz[i + 1] = szz[i] + Z * Z;
    sxz[i + 1] = sxz[i] + X * Z;
  }
  return { sx, sz, sxx, szz, sxz };
}

/**
 * Total-least-squares line through samples [a..b].
 *
 * Total least squares rather than a regression of one axis on the other,
 * because a wall's base line can point any way in the ground plane: fitting Z
 * on X blows up for a side wall and fitting X on Z blows up for a back one.
 * The principal axis of the covariance has no preferred direction.
 *
 * `err` is the sum of squared perpendicular distances, which for a 2x2
 * covariance is exactly its smaller eigenvalue -- so the quality of a fit
 * comes out of the same arithmetic as its direction, with nothing to iterate.
 */
function fitRange(P, a, b) {
  const n = b - a + 1;
  const mx = (P.sx[b + 1] - P.sx[a]) / n;
  const mz = (P.sz[b + 1] - P.sz[a]) / n;
  const cxx = (P.sxx[b + 1] - P.sxx[a]) - n * mx * mx;
  const czz = (P.szz[b + 1] - P.szz[a]) - n * mz * mz;
  const cxz = (P.sxz[b + 1] - P.sxz[a]) - n * mx * mz;

  const tr = cxx + czz;
  const det = cxx * czz - cxz * cxz;
  const disc = Math.max(0, tr * tr - 4 * det);
  const lo = (tr - Math.sqrt(disc)) / 2;

  const theta = 0.5 * Math.atan2(2 * cxz, cxx - czz);
  return {
    origin: [mx, mz],
    dir: [Math.cos(theta), Math.sin(theta)],
    err: Math.max(0, lo),
    rms: Math.sqrt(Math.max(0, lo) / n),
    n,
  };
}


/**
 * Cut the junction run into straight pieces -- one per wall.
 *
 * Sequential RANSAC was the wrong tool. It looks for the globally dominant
 * line and discards the points agreeing with it, which ignores the one thing
 * always true of walls in a photograph: from a single viewpoint they are
 * contiguous in image columns and they meet at corners. A line fitted from the
 * leftovers of one wall and the start of the next is plausible to RANSAC and
 * geometric nonsense in the room.
 *
 * Read left to right, the wall/floor junction is a piecewise-linear path
 * across the ground plane and the corners are its breakpoints. So: split at
 * the breakpoint that best explains the run, then merge back neighbours that
 * turn out to be collinear after all.
 */
function segmentRuns(samples, { tol = 0.06, minSamples = 12, minSpanPx = 40, minWidth = 0.2, gain = 0.6 } = {}) {
  const n = samples.length;
  if (n < minSamples) return [];
  const P = junctionSums(samples);
  const span = (a, b) => samples[b].u - samples[a].u;

  /** How long a run is along its own fitted line, in metres. */
  const runWidth = (a, b) => {
    const f = fitRange(P, a, b);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = a; i <= b; i++) {
      const t = (samples[i].X - f.origin[0]) * f.dir[0] + (samples[i].Z - f.origin[1]) * f.dir[1];
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
    return hi - lo;
  };

  const runs = [];
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const whole = fitRange(P, a, b);

    // A corner is not always a turn. Two walls can be near enough parallel and
    // still be different planes -- a chimney breast, a window wall set back
    // from the one beside it -- and the junction line steps rather than bends.
    // Testing the angle misses exactly those, so what has to improve is the
    // fit: split only when one line genuinely fails to explain the run and two
    // do far better.
    let cut = -1;
    if (whole.rms > tol && b - a + 1 >= minSamples * 2) {
      let bestErr = whole.err * gain;
      for (let k = a + minSamples - 1; k <= b - minSamples; k++) {
        // Both sides have to be wide enough in the photograph to be a wall
        // somebody could point at.
        if (span(a, k) < minSpanPx || span(k + 1, b) < minSpanPx) continue;
        const err = fitRange(P, a, k).err + fitRange(P, k + 1, b).err;
        if (err < bestErr) { bestErr = err; cut = k; }
      }
    }
    if (cut >= 0) stack.push([a, cut], [cut + 1, b]);
    else runs.push({ a, b });
  }
  runs.sort((p, q) => p.a - q.a);

  // Merge neighbours that turned out to be collinear. Split-and-merge can cut
  // early on a stretch that later evidence straightens out, and without this a
  // wall with one bad patch comes back as two surfaces meeting at a corner
  // that is not there.
  let merged = true;
  while (merged && runs.length > 1) {
    merged = false;
    for (let i = 0; i < runs.length - 1; i++) {
      const joint = fitRange(P, runs[i].a, runs[i + 1].b);
      if (joint.rms <= tol) {
        runs.splice(i, 2, { a: runs[i].a, b: runs[i + 1].b });
        merged = true;
        break;
      }
    }
  }

  // Absorb anything too narrow to be a wall in its own right into whichever
  // neighbour explains it better. Dropping slivers instead would leave their
  // columns owned by nothing -- a bare stripe of photograph down the join --
  // and shipping them produces two half-metre "walls" where the room has one.
  let absorbed = true;
  while (absorbed && runs.length > 1) {
    absorbed = false;
    for (let i = 0; i < runs.length; i++) {
      // Both tests matter, and the metric one does the work: a run can occupy
      // plenty of columns and still be a metre of wall seen at a glancing
      // angle, which is a fragment of something, not a wall of its own.
      if (span(runs[i].a, runs[i].b) >= minSpanPx
          && runWidth(runs[i].a, runs[i].b) >= minWidth) continue;
      const left = i > 0 ? fitRange(P, runs[i - 1].a, runs[i].b).rms : Infinity;
      const right = i < runs.length - 1 ? fitRange(P, runs[i].a, runs[i + 1].b).rms : Infinity;
      if (left === Infinity && right === Infinity) break;
      if (left <= right) runs.splice(i - 1, 2, { a: runs[i - 1].a, b: runs[i].b });
      else runs.splice(i, 2, { a: runs[i].a, b: runs[i + 1].b });
      absorbed = true;
      break;
    }
  }

  return runs.filter((r) => r.b - r.a + 1 >= minSamples).map((r) => ({ ...r, fit: fitRange(P, r.a, r.b) }));
}

/**
 * Throw away junction samples that disagree with their neighbours.
 *
 * A single bad un-projection -- the bottom edge of a rug read as skirting --
 * drags a breakpoint metres out of place, and the segmentation has no way to
 * tell that point from a real corner. Comparing each sample against the median
 * of the ones beside it in the image costs nothing and removes exactly that.
 */
function dropOutliers(samples, { window = 6, limit = 0.5 } = {}) {
  if (samples.length < window * 2) return samples;
  const median = (arr) => {
    const s = [...arr].sort((p, q) => p - q);
    return s[Math.floor(s.length / 2)];
  };
  const out = [];
  for (let i = 0; i < samples.length; i++) {
    const lo = Math.max(0, i - window);
    const hi = Math.min(samples.length - 1, i + window);
    const xs = [];
    const zs = [];
    for (let k = lo; k <= hi; k++) { xs.push(samples[k].X); zs.push(samples[k].Z); }
    if (Math.hypot(samples[i].X - median(xs), samples[i].Z - median(zs)) <= limit) out.push(samples[i]);
  }
  return out;
}

/**
 * Cut runs at visible corners the junction data could not find.
 *
 * Only where it could not: a run the piecewise fit already explains well is
 * one straight wall, and a vertical line crossing it is a door frame, a
 * radiator pipe or the edge of a wardrobe -- splitting there would invent a
 * corner the room does not have. So a corner is only allowed to divide a run
 * whose fit is poor, which is precisely the signature of two walls averaged
 * into one.
 */
function splitAtCorners(samples, runs, corners, { minSamples = 3, minSpanPx = 5 } = {}) {
  const out = [];
  const P = junctionSums(samples);
  for (const run of runs) {
    let pieces = [run];
    for (const corner of corners) {
      const next = [];
      for (const piece of pieces) {
        const uLo = samples[piece.a].u;
        const uHi = samples[piece.b].u;
        if (corner.u <= uLo || corner.u >= uHi
            || corner.u - uLo < minSpanPx || uHi - corner.u < minSpanPx) {
          next.push(piece);
          continue;
        }
          let k = piece.a;
          while (k < piece.b && samples[k + 1].u <= corner.u) k += 1;
          if (k - piece.a + 1 < minSamples || piece.b - k < minSamples) {
            next.push(piece);
            continue;
          }
          next.push({ a: piece.a, b: k, cut: corner.u }, { a: k + 1, b: piece.b, cut: corner.u });
        }
        pieces = next;
      }
    out.push(...pieces);
  }
  return out.map((r) => ({ ...r, fit: r.fit ?? null }));
}

/**
 * Split one wall region into its separate planes.
 *
 * Segmentation gives a single "wall" class, so the left, back and right walls
 * of an ordinary room arrive as one region. Their junction points, read left
 * to right, trace a piecewise-linear path across the ground plane: each
 * straight piece is a wall and each breakpoint is a corner.
 */
function wallPlanes(cam, mask, w, h, floorMask, ceilingMask, roomHeight, why = {}, corners = []) {
  const raw = collectJunctions(cam, mask, w, h, floorMask, ceilingMask, roomHeight);
  why.samples = raw.length;

  const fallback = () => {
    const fp = frontoParallelWall(cam, mask, w, h, floorMask, roomHeight);
    if (fp) {
      why.reason = 'no junction visible - assumed square-on to the camera';
      why.assumed = true;
      return [fp];
    }
    why.reason = why.reason ?? 'too few junction samples';
    return [];
  };

  if (raw.length < 12) return fallback();

  // Left to right across the photograph, which is the order the piecewise fit
  // reads them in, and outliers removed first so one bad un-projection cannot
  // invent a corner.
  const samples = dropOutliers([...raw].sort((p, q) => p.u - q.u));
  const minSpanPx = Math.max(10, w * 0.02);
  let runs = segmentRuns(samples, { minSpanPx });
  why.runs = runs.length;

  // Then let the photograph have its say. A run the junction data fits badly
  // is one where the evidence for a corner was never there to read -- the sofa
  // was in front of it -- and that is exactly the case a visible vertical
  // corner settles.
  if (corners.length) {
    const before = runs.length;
    runs = splitAtCorners(samples, runs, corners);
    why.cornerSplits = runs.length - before;
  }

  const minSpan = w * 0.01;
  const planes = [];
  for (const run of runs) {
    const inliers = samples.slice(run.a, run.b + 1);
    const fit = run.fit ?? fitRange(junctionSums(samples), run.a, run.b);
    const plane = planeFromRun(cam, fit.origin, fit.dir, inliers, mask, w, h, roomHeight);

    if (!plane.quad) {
      (why.dropped ??= []).push(plane.reason);
    } else if (plane.uMax - plane.uMin < minSpan) {
      (why.dropped ??= []).push(`spans only ${Math.round(plane.uMax - plane.uMin)}px of the frame`);
    } else {
      (why.accepted ??= []).push(
        `u ${Math.round(plane.uMin)}..${Math.round(plane.uMax)} `
        + `${plane.realSize.w}x${plane.realSize.h}m dir=[${plane.dir.map((v) => v.toFixed(2))}] `
        + `n=${plane.count} rms=${fit.rms.toFixed(3)}m`,
      );
      planes.push(plane);
    }
  }

  if (!planes.length) return fallback();
  planes.sort((a, b) => (a.uMin + a.uMax) - (b.uMin + b.uMax));
  return planes;
}

/**
 * Decide which plane every wall pixel actually belongs to.
 *
 * The obvious split -- cut the wall region into vertical column bands, one per
 * plane -- is wrong wherever a corner is not vertical in the image, which is
 * most of the time: the corner between a side wall and the back wall runs at
 * an angle, and a straight cut hands a wedge of one wall to the other. The
 * wedge then gets a homography that does not describe it, and its tiles come
 * out at the wrong scale and angle.
 *
 * Each plane is a known finite rectangle in world space, so the honest test is
 * the one the panorama renderer already uses: cast the ray through the pixel,
 * intersect it with every plane, and keep the nearest hit that lands inside
 * that plane's extent. The corner falls out of the geometry, exactly where it
 * belongs, and a wall broken in two by a wardrobe still ends up as one surface.
 */
function assignWallPixels(cam, planes, mask, w, h) {
  const out = planes.map(() => new Uint8Array(w * h));
  const counts = new Array(planes.length).fill(0);

  // Precompute each plane's normal and its offset, in the XZ ground plane.
  const geo = planes.map((p) => {
    const dir = p.dir ?? [1, 0];
    const origin = p.origin ?? [0, 1];
    const n = [-dir[1], dir[0]];
    return {
      dir,
      origin,
      n,
      d: n[0] * origin[0] + n[1] * origin[1],
      s0: p.s0 ?? -1e9,
      s1: p.s1 ?? 1e9,
      height: p.realSize?.h ?? 2.7,
    };
  });

  // A little slack: segmentation edges are ragged, and a pixel a few
  // centimetres past the top of a fitted wall is still that wall.
  const sPad = 0.12;
  const yPad = 0.25;

  for (let y = 0; y < h; y++) {
    const row = y * w;
    const b = (y - cam.cy) / cam.f;
    for (let x = 0; x < w; x++) {
      if (mask[row + x] <= 127) continue;
      const a = (x - cam.cx) / cam.f;

      let best = -1;
      let bestZ = Infinity;
      for (let i = 0; i < geo.length; i++) {
        const g = geo[i];
        const denom = g.n[0] * a + g.n[1];
        if (Math.abs(denom) < 1e-6) continue;
        const Z = g.d / denom;
        if (!(Z > 0.2) || Z >= bestZ) continue;

        const X = a * Z;
        const s = (X - g.origin[0]) * g.dir[0] + (Z - g.origin[1]) * g.dir[1];
        const Y = cam.height - b * Z;

        bestZ = Z;
        best = i;
      }
      if (best >= 0) {
        out[best][row + x] = 255;
        counts[best]++;
      }
    }
  }
  return { masks: out, counts };
}

/**
 * Name a wall the way someone standing in the room would.
 *
 * "Wall 1 / Wall 2 / Wall 3" is accurate and useless. A wall whose base line
 * runs across the view is the one being looked at; one running away from the
 * camera is to the left or the right, decided by which side of the optical
 * axis it sits on.
 */
function nameWall(plane, cam) {
  const dir = plane.dir ?? [1, 0];
  const uMin = plane.uMin ?? 0;
  const uMax = plane.uMax ?? 0;

  // Square-on to the camera is what makes a wall a back wall, wherever it
  // happens to sit in the frame. Requiring it to also straddle the optical
  // axis was wrong: once corners divide a back wall into two, neither half
  // contains the centre any more and both get called side walls, which is not
  // what anybody standing in the room would say.
  const squareOn = Math.abs(dir[0]) > 2.5 * Math.abs(dir[1]);
  if (squareOn) return { key: 'back_wall', label: 'Back Wall' };

  // Ambiguous orientation: fall back to straddling the axis.
  if (Math.abs(dir[0]) >= Math.abs(dir[1]) && uMin <= cam.cx && uMax >= cam.cx) {
    return { key: 'back_wall', label: 'Back Wall' };
  }

  return (uMin + uMax) / 2 < cam.cx
    ? { key: 'left_wall', label: 'Left Wall' }
    : { key: 'right_wall', label: 'Right Wall' };
}

/**
 * Name a wall from its 3D orientation. With a real plane there is no need to
 * guess from where it sits in the frame: a wall facing the camera (normal
 * within 35 degrees of the view axis) is a back wall, and otherwise the side
 * its normal points to says which wall it is -- a wall facing right is the
 * left wall. Two chimney-breast faces are both back walls, which is what
 * someone in the room would call them.
 */
export function nameWallFromPlane(plane) {
  const facing = (Math.acos(Math.max(-1, Math.min(1, -plane.nz))) * 180) / Math.PI;
  if (facing <= 35) return { key: 'back_wall', label: 'Back Wall' };
  return plane.nx > 0
    ? { key: 'left_wall', label: 'Left Wall' }
    : { key: 'right_wall', label: 'Right Wall' };
}

const norm2 = ([x, y]) => {
  const l = Math.hypot(x, y) || 1;
  return [x / l, y / l];
};
const round2 = (n) => Math.round(n * 100) / 100;

/* ---------------------------------------------------------------- main --- */

/**
 * Detect the floor, walls and ceiling in a room photograph.
 * Returns an objectList in exactly the format the Studio edits, so anything
 * here can be adjusted by hand afterwards.
 */
/**
 * Segment the photograph and snap the class maps to its edges.
 *
 * Returns the crisp class masks both at the refinement resolution (where the
 * 3D scanner reads them) and at photo resolution (where polygons are traced),
 * plus the soft object maps the occlusion reasoning needs.
 */
export async function segmentAndRefine(imagePath) {
  const segmenter = await getSegmenter();
  const results = await segmenter(imagePath);
  const meta = await sharp(imagePath).metadata();
  const w = meta.width;
  const h = meta.height;

  // --- refine the class maps against the photograph ------------------------
  // The segmenter's output is right about what is where and wrong by several
  // pixels about where it stops; blown up to photo resolution that becomes a
  // wobbling band along every boundary. The photograph knows where the ceiling
  // line and the edge of the sofa are, so it is used as the guide.
  const RW = Math.min(REFINE_W, w);
  const RH = Math.max(1, Math.round((h / w) * RW));

  const soft = {};
  // Everything that is not a surface -- sofa, curtain, painting, lamp, rug,
  // window -- merged into one competitor. It is what lets a surface be decided
  // by comparison rather than by an absolute score: a wall pixel the segmenter
  // was only half sure about still beats "nothing is here", but never beats
  // the curtain hanging in front of it.
  const other = new Float32Array(RW * RH);
  let hasOther = false;

  const wallObjects = new Float32Array(RW * RH);
  const floorObjects = new Float32Array(RW * RH);
  // Floor furniture other than tables, kept apart so that, once depth is
  // known, the floor can be rebuilt (see refineFloorWithDepth).
  const furnitureNoTables = new Float32Array(RW * RH);
  let hasWallObjects = false;
  let hasFloorObjects = false;

  const resampleMaskAdd = (img, out, dw, dh) => {
    const sw = img.width, sh = img.height;
    const src = img.data;
    for (let y = 0; y < dh; y++) {
      const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
      for (let x = 0; x < dw; x++) {
        const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
        out[y * dw + x] += src[sy * sw + sx] / 255;
      }
    }
  };

  const WALL_OBJECT_LABELS = new Set([
    'painting, picture', 'windowpane, window', 'door, double door', 
    'cabinet', 'mirror', 'sconce', 'radiator', 'fireplace, hearth, open fireplace',
    'television, television receiver, television set, tv, tv set'
  ]);

  const FLOOR_OBJECT_LABELS = new Set([
    'bed', 'sofa, couch, lounge', 'chair', 'table', 'coffee table', 
    'stool', 'cushion', 'ottoman, pouf, pouffe, puff, hassock',
  ]);

  // A rug or mat lies flat on the floor, so it is floor: the floor surface
  // covers it rather than being cut around it.
  // Matched on the first name of the label: the segmenter returns 'rug' in
  // some versions and 'rug, carpet, carpeting' in others.
  const FLOOR_LABELS = new Set(['rug', 'carpet']);
  // Tables: cut out of the floor by their convex outline, because a glass
  // top shows the floor or rug through it and the segmenter labels only the
  // frame as table.
  const TABLE_LABELS = new Set(['table', 'coffee table', 'desk']);
  const tableObjects = new Float32Array(RW * RH);
  let hasTables = false;

  // Every object the segmenter names -- plant, window, curtain, painting,
  // wall hanging, lamp, TV, furniture -- whatever its label. Used ONLY to
  // leave objects out of each wall's painted area at the very end; wall and
  // corner detection never see it.
  const anyObjects = new Float32Array(RW * RH);
  let hasAnyObjects = false;
  // Box furniture that, when it lies flush on a wall, is the wall: the
  // segmenter calls wood panelling "cabinet". A real cabinet or wardrobe
  // stands well proud of the wall and is still left unpainted.
  const FLUSH_WALL_LABELS = new Set(['cabinet', 'wardrobe', 'chest of drawers', 'shelf', 'bookcase', 'buffet']);
  const flushCandidates = new Float32Array(RW * RH);
  let hasFlushCandidates = false;

  for (const r of results) {
    const key = r.label.toLowerCase();
    if (!SURFACE_CLASSES[key] && !FLOOR_LABELS.has(key.split(',')[0].trim())) {
      resampleMaskAdd(r.mask, anyObjects, RW, RH);
      hasAnyObjects = true;
    }
    if (FLUSH_WALL_LABELS.has(key.split(',')[0].trim())) {
      resampleMaskAdd(r.mask, flushCandidates, RW, RH);
      hasFlushCandidates = true;
    }
    if (SURFACE_CLASSES[key]) {
      soft[key] ??= new Float32Array(RW * RH);
      resampleMask(r.mask, soft[key], RW, RH);
    } else if (TABLE_LABELS.has(key.split(',')[0].trim())) {
      // Tables also stay floor occluders (below), and their outline is kept
      // separately so a glass top can be cut out whole.
      resampleMaskAdd(r.mask, tableObjects, RW, RH);
      resampleMaskAdd(r.mask, floorObjects, RW, RH);
      hasFloorObjects = true;
      hasTables = true;
    } else if (FLOOR_LABELS.has(key.split(',')[0].trim())) {
      soft.floor ??= new Float32Array(RW * RH);
      resampleMask(r.mask, soft.floor, RW, RH);
    } else {
      let isWallObj = WALL_OBJECT_LABELS.has(key);
      let isFloorObj = FLOOR_OBJECT_LABELS.has(key);
      
      if (isWallObj) {
        resampleMaskAdd(r.mask, wallObjects, RW, RH);
        hasWallObjects = true;
      }
      if (isFloorObj) {
        resampleMaskAdd(r.mask, floorObjects, RW, RH);
        resampleMaskAdd(r.mask, furnitureNoTables, RW, RH);
        hasFloorObjects = true;
      }
    }
  }

  const masks = {};
  let refineInfo = null;
  let occluder = null;
  let occluderFull = null;
  let occluderWallFull = null;
  let floorBeforeCuts = null;
  // Kept past the refinement block: the corner search wants the photograph and
  // the wall mask at the same, already-computed, working resolution.
  let guideRef = null;
  let wallRef = null;
  let crispRef = null;
  if (Object.keys(soft).length) {
    const rgb = await sharp(imagePath).removeAlpha().resize(RW, RH, { fit: 'fill' }).raw().toBuffer();
    const guide = toLuma(rgb, RW * RH);
    guideRef = guide;
    const combinedOther = new Float32Array(RW * RH);
    if (hasWallObjects || hasFloorObjects) {
      for (let i = 0; i < RW * RH; i++) {
        combinedOther[i] = Math.max(wallObjects[i], floorObjects[i]);
      }
    }
    const competitors = (hasWallObjects || hasFloorObjects) ? { ...soft, [OTHER_KEY]: combinedOther } : soft;
    const { masks: crisp, radius } = refineClassMaps(guide, competitors, RW, RH, {
      drop: [OTHER_KEY],
    });

    let occluderFloor = null;
    let occluderWall = null;
    
    if (hasWallObjects || hasFloorObjects) {
      const hardWall = new Uint8Array(RW * RH);
      const hardFloor = new Uint8Array(RW * RH);
      for (let i = 0; i < RW * RH; i++) {
        if (wallObjects[i] >= 0.1) hardWall[i] = 1;
        if (floorObjects[i] >= 0.01) hardFloor[i] = 1;
      }
      occluderWall = dilate(hardWall, RW, RH, 3);
      occluderFloor = dilate(hardFloor, RW, RH, 3);
      occluder = occluderWall; // keep a reference for upscale

      for (const key of Object.keys(crisp)) {
        if (key === 'floor') continue; // Handled separately, below
        const m = crisp[key];
        for (let i = 0; i < RW * RH; i++) if (occluderWall[i]) m[i] = 0;
      }
      // Furniture standing on the floor (sofa, table, chair...) is cut out of
      // the floor, so no tile is drawn on a table top -- a glass table over a
      // rug otherwise reads as floor. Rugs are floor and are not in this set.
      if (crisp.floor && hasFloorObjects) {
        floorBeforeCuts = Uint8Array.from(crisp.floor);
        for (let i = 0; i < RW * RH; i++) if (occluderFloor[i]) crisp.floor[i] = 0;
        if (hasTables) cutTableHulls(crisp.floor, tableObjects, RW, RH);
        dropEnclosedFloor(crisp.floor, occluderFloor, RW, RH);
      }
    }

    // Object map for the walls' painted areas (full resolution).
    if (hasAnyObjects) {
      const hard = new Uint8Array(RW * RH);
      for (let i = 0; i < RW * RH; i++) if (anyObjects[i] >= 0.5 || occluderWall?.[i]) hard[i] = 255;
      occluderWallFull = upscaleMask(dilate(hard, RW, RH, 2), RW, RH, w, h);
    }

    refineInfo = { width: RW, height: RH, radius, occluderClass: (hasWallObjects || hasFloorObjects) };
    wallRef = crisp.wall ?? null;
    crispRef = crisp;
    for (const key of Object.keys(crisp)) {
      masks[key] = upscaleMask(crisp[key], RW, RH, w, h);
    }
    if (occluder) occluderFull = upscaleMask(occluder, RW, RH, w, h);
  }

  return {
    w, h, RW, RH,
    labels: results.map((r) => r.label),
    masks,
    crisp: crispRef,
    guide: guideRef,
    wall: wallRef,
    objects: {
      wall: hasWallObjects ? wallObjects : null,
      floor: hasFloorObjects ? floorObjects : null,
    },
    occluderFull,
    occluderWallFull,
    flushCandidates: hasFlushCandidates ? flushCandidates : null,
    floorParts: floorBeforeCuts
      ? { before: floorBeforeCuts, furniture: furnitureNoTables, tables: hasTables ? tableObjects : null }
      : null,
    refineInfo,
  };
}

/**
 * Detect the floor, walls and ceiling in a room photograph.
 * Returns an objectList in exactly the format the Studio edits, so anything
 * here can be adjusted by hand afterwards.
 *
 * The 3D scanner (./scanner) runs first; the junction-based fit below is what
 * it falls back to when no depth is available or the 3D stage cannot find a
 * wall, so a scan never gets worse than it was before depth existed.
 */
export async function autoDetectSurfaces(imagePath, opts = {}) {
  const { roomHeight = 2.7, hfov = 70, includeCeiling = false, trace = {} } = opts;
  const t0 = Date.now();
  const seg = await segmentAndRefine(imagePath);
  const tSeg = Date.now() - t0;
  const {
    w, h, masks, occluderFull, refineInfo,
  } = seg;
  // Objects on walls: removed from each wall's painted area only.
  const wallOccluderFull = seg.occluderWallFull ?? occluderFull;
  const guideRef = seg.guide;
  const wallRef = seg.wall;
  const RW = seg.RW;
  const RH = seg.RH;
  const diagnostics = {};

  const cam = solveCamera(masks, w, h, { roomHeight, hfov });
  const objectList = [];

  // --- 3D scan -------------------------------------------------------------
  // Depth -> point cloud -> planes -> corners -> raycast. When it cannot run
  // (no depth source, no wall plane passes validation) the junction scanner
  // below does the walls exactly as before, and the reason is reported.
  const scannerMode = opts.scanner ?? process.env.SCANNER ?? 'auto';
  const quality = opts.quality ?? 'balanced';
  const debugDir = opts.debug === true
    ? path.join(DATA_DIR, 'debug', `${path.basename(imagePath).replace(/\.[^.]+$/, '')}-${Date.now()}`)
    : (typeof opts.debug === 'string' ? opts.debug : (process.env.SCAN_DEBUG === '1' ? path.join(DATA_DIR, 'debug', `${path.basename(imagePath).replace(/\.[^.]+$/, '')}-${Date.now()}`) : null));
  let geoRun = null;
  if (scannerMode !== 'legacy' && masks.wall) {
    geoRun = await runGeometryScan(imagePath, seg, {
      quality,
      roomHeight,
      // Only a caller-supplied field of view overrides the estimate.
      hfov: opts.hfovExplicit ? hfov : undefined,
      hfovSource: opts.hfovSource ?? undefined,
      debugDir,
      levelCam: cam,
      depthModel: opts.depthModel,
    });
    if (!geoRun.ok) {
      diagnostics.scannerFallback = `${geoRun.stage}: ${geoRun.reason}`;
      if (geoRun.stack) console.error('[scan]', geoRun.stack);
    }
  }
  const geoOk = !!geoRun?.ok;
  const geo3d = geoOk ? geoRun.geo : null;

  // --- floor ---------------------------------------------------------------
  // With depth, furniture is only cut out of the floor where it actually
  // stands above the floor plane. Runs after the walls, and changes nothing
  // but the floor's mask.
  if (geoOk && seg.floorParts && seg.crisp) {
    const refined = refineFloorWithDepth(seg, geo3d);
    if (refined) masks.floor = upscaleMask(refined, RW, RH, w, h);
  }
  if (masks.floor) {
    const geo = horizontalQuad(cam, masks.floor, w, h, 0);
    const polys = maskToPolygons(masks.floor, w, h, { occluder: occluderFull, ...trace });
    if (geo && polys.length) {
      objectList.push({
        name: 'floor',
        label: 'Floor',
        product_surface: 'floor',
        isMain: true,
        order: 0,
        quad: geo.quad,
        realSize: geo.realSize,
        mask: { feather: 0.8, polygons: polys },
        auto: true,
        defaults: {
          tileSize: { w: 600, h: 600 },
          layout: 'grid',
          grout: { size: 2, color: '#c9c9c4' },
        },
      });
    }
  }

  // --- walls (3D) ------------------------------------------------------------
  // One surface per validated wall plane, its mask being exactly the pixels
  // the raycast gave it. Masks are exclusive, so a corner pixel belongs to one
  // wall only.
  const scanWalls = [];
  if (geoOk) {
    const { walls: gw, masks: gm } = geo3d;
    const minPixels = Math.max(500 * ((RW * RH) / (w * h)), Math.round(RW * RH * 0.0025));
    const flushPanels = findFlushPanels(seg.flushCandidates, gm.flush, RW, RH);
    const found = [];
    for (const wl of gw) {
      const counts = gm.counts[wl.index];
      if (counts.visible < minPixels || !wl.quad) {
        (diagnostics.wallsRejected ??= []).push(
          `${wl.id}: ${!wl.quad ? 'quad does not project' : `only ${counts.visible} px visible after raycast`} (plane kept in scan.walls)`,
        );
        scanWalls.push(publicWall(wl, geo3d, { selectable: false }));
        continue;
      }
      // Panelling the segmenter mislabelled as a cabinet, lying flush on this
      // wall's plane, is painted with the wall -- decided per whole object.
      const panel = flushPanels[wl.index];
      const full = upscaleMask(gm.visible[wl.index], RW, RH, w, h);
      if (panel) {
        const panelFull = upscaleMask(panel, RW, RH, w, h);
        for (let i = 0; i < full.length; i++) {
          if (panelFull[i]) full[i] = 255;
          else if (seg.occluderWallFull?.[i]) full[i] = 0;
        }
      } else {
        removeObjects(full, seg.occluderWallFull);
      }
      const checked = validatePolygons(maskToPolygons(full, w, h, { occluder: wallOccluderFull, ...trace }), w, h);
      if (!checked.polygons.some((p) => p.mode === 'add')) {
        (diagnostics.wallsRejected ??= []).push(`${wl.id}: no valid polygon (${checked.notes.join('; ')})`);
        scanWalls.push(publicWall(wl, geo3d, { selectable: false }));
        continue;
      }
      let occludedPolys = [];
      if (counts.occluded > minPixels / 4) {
        occludedPolys = validatePolygons(
          maskToPolygons(upscaleMask(gm.occluded[wl.index], RW, RH, w, h), w, h, { minAreaFrac: 0.002 }),
          w, h,
        ).polygons.filter((p) => p.mode === 'add');
      }
      found.push({ wl, polys: checked.polygons, occludedPolys, notes: checked.notes });
    }

    const used = new Map();
    for (const f of found) {
      const named = nameWallFromPlane(f.wl.plane);
      const n = (used.get(named.key) ?? 0) + 1;
      used.set(named.key, n);
      const name = n > 1 ? `${named.key}_${n}` : named.key;
      const pub = publicWall(f.wl, geo3d, {
        objectName: name,
        selectable: true,
        polygon2D: f.polys.filter((p) => p.mode === 'add').map((p) => p.points),
        visibleMask: { polygons: f.polys },
        occludedMask: { polygons: f.occludedPolys },
        polygonNotes: f.notes.length ? f.notes : undefined,
      });
      scanWalls.push(pub);
      objectList.push({
        name,
        label: n > 1 ? `${named.label} ${n}` : named.label,
        product_surface: 'wall',
        order: objectList.length,
        quad: f.wl.quad,
        realSize: f.wl.realSize,
        mask: { feather: 0.8, polygons: f.polys },
        auto: true,
        assumed: false,
        scanId: f.wl.id,
        confidence: pub.confidence,
        geometry: {
          plane: pub.plane,
          normal: pub.normal,
          polygon3D: pub.polygon3D,
          corners: pub.corners,
          occluded: f.occludedPolys,
        },
        defaults: {
          tileSize: { w: 300, h: 600 },
          layout: 'brick',
          grout: { size: 2, color: '#f5f5f2' },
        },
      });
    }
    diagnostics.wallsFitted = gw.length;
    diagnostics.raycast = gm.stats;
  }

  // --- walls (junction fallback) ----------------------------------------------
  // The wall class arrives as one region covering every wall in the room, and
  // often broken into several pieces by whatever furniture stands in front of
  // it. Both are handled the same way: fit the planes over the whole class at
  // once, then decide per pixel which plane each one belongs to.
  if (!geoOk && masks.wall) {
    // Vertical corners, in full-resolution columns. Independent evidence of a
    // wall boundary, and the only kind available where furniture hides the
    // junction the piecewise fit reads.
    let corners = [];
    if (guideRef && wallRef) {
      const scale = w / RW;
      corners = verticalCorners(guideRef, wallRef, RW, RH)
        .map((c) => ({ ...c, u: Math.round(c.u * scale) }));
      if (corners.length) {
        diagnostics.corners = corners.map(
          (c) => `u ${c.u} continuity ${c.continuity.toFixed(2)} contrast ${c.contrast.toFixed(3)}`,
        );
      }
    }

    const small = downsample(masks.wall, w, h, Math.min(TRACE_W, w));
    const comps = components(small.data, small.width, small.height, 1);
    const total = small.width * small.height;

    // Keep every piece big enough to be a wall rather than a mis-labelled
    // sliver, and merge them: a wall cut in two by a wardrobe is still one
    // wall, and fitting its halves separately gives two disagreeing planes.
    const keep = new Set(
      comps.list.filter((c) => c.size / total > 0.004).map((c) => c.id),
    );
    const wall = new Uint8Array(w * h);
    const sxr = small.width / w;
    const syr = small.height / h;
    for (let y = 0; y < h; y++) {
      const sy = Math.min(small.height - 1, Math.floor(y * syr));
      for (let x = 0; x < w; x++) {
        const sx = Math.min(small.width - 1, Math.floor(x * sxr));
        if (keep.has(comps.label[sy * small.width + sx])) wall[y * w + x] = 255;
      }
    }

    const why = {};
    const planes = wallPlanes(cam, wall, w, h, masks.floor, masks.ceiling, roomHeight, why, corners);

    diagnostics.wallComponents = comps.list.length;
    diagnostics.wallPieces = keep.size;
    diagnostics.wallsFitted = planes.length;
    if (why.reason) diagnostics.wallNote = why.reason;
    if (why.samples !== undefined) diagnostics.junctionSamples = why.samples;
    if (why.runs !== undefined) diagnostics.runs = why.runs;
    if (why.cornerSplits) diagnostics.cornerSplits = why.cornerSplits;
    if (why.dropped) diagnostics.wallsRejected = why.dropped;
    if (why.accepted) diagnostics.wallsAccepted = why.accepted;

    const found = [];
    if (planes.length === 1) {
      // Nothing to divide up: the single fitted plane owns the whole region.
      const polys = maskToPolygons(removeObjects(Uint8Array.from(wall), seg.occluderWallFull), w, h, { occluder: wallOccluderFull, ...trace });
      if (polys.length) found.push({ plane: planes[0], polys });
    } else if (planes.length > 1) {
      const { masks: perPlane, counts } = assignWallPixels(cam, planes, wall, w, h);

      // Whatever the geometry could not place -- a pixel just past the top of
      // every fitted wall, say -- falls back to the old column-band rule, so
      // the corner between two walls never leaves a bare stripe of photo.
      const bounds = [0];
      for (let k = 1; k < planes.length; k++) {
        const mid = Math.round((planes[k - 1].uMax + planes[k].uMin) / 2);
        bounds.push(Math.max(bounds[k - 1] + 1, Math.min(w - 1, mid)));
      }
      bounds.push(w);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
          if (wall[row + x] <= 127) continue;
          if (perPlane.some((m) => m[row + x] > 127)) continue;
          for (let k = 0; k < planes.length; k++) {
            if (x >= bounds[k] && x < bounds[k + 1]) { perPlane[k][row + x] = 255; counts[k]++; break; }
          }
        }
      }

      const minPixels = Math.max(500, Math.round(w * h * 0.0025));
      planes.forEach((plane, i) => {
        if (counts[i] < minPixels) {
          (diagnostics.wallsRejected ??= []).push(
            `plane ${i + 1}: only ${counts[i]} px of the photo resolve onto it`,
          );
          return;
        }
        const polys = maskToPolygons(removeObjects(perPlane[i], seg.occluderWallFull), w, h, { occluder: wallOccluderFull, ...trace });
        if (polys.length) found.push({ plane, polys });
        else {
          // Geometrically fine, but nothing of it is visible -- a curtain or a
          // wardrobe covers the whole span. Worth saying rather than dropping
          // in silence, because "four planes, three walls" otherwise looks
          // like a bug.
          (diagnostics.wallsRejected ??= []).push(
            `plane ${i + 1} (${plane.realSize.w}m): no wall visible, it is covered`,
          );
        }
      });
    }

    // Left to right across the photo, so the list reads the way the room does.
    found.sort((a, b) => (a.plane.uMin + a.plane.uMax) - (b.plane.uMin + b.plane.uMax));

    const used = new Map();
    for (const f of found) {
      const named = nameWall(f.plane, cam);
      const n = (used.get(named.key) ?? 0) + 1;
      used.set(named.key, n);
      objectList.push({
        name: n > 1 ? `${named.key}_${n}` : named.key,
        label: n > 1 ? `${named.label} ${n}` : named.label,
        product_surface: 'wall',
        order: objectList.length,
        quad: f.plane.quad,
        realSize: f.plane.realSize,
        mask: { feather: 0.8, polygons: f.polys },
        auto: true,
        assumed: !!f.plane.assumed,
        defaults: {
          tileSize: { w: 300, h: 600 },
          layout: 'brick',
          grout: { size: 2, color: '#f5f5f2' },
        },
      });
    }
  }

  // --- ceiling -------------------------------------------------------------
  if (includeCeiling && masks.ceiling) {
    const geo = geo3d?.ceiling?.quad ? geo3d.ceiling : horizontalQuad(cam, masks.ceiling, w, h, roomHeight);
    const polys = maskToPolygons(masks.ceiling, w, h, { occluder: occluderFull, ...trace });
    if (geo && polys.length) {
      objectList.push({
        name: 'ceiling',
        label: 'Ceiling',
        product_surface: 'ceiling',
        order: objectList.length,
        quad: geo.quad,
        realSize: geo.realSize,
        mask: { feather: 0.8, polygons: polys },
        auto: true,
        defaults: {
          tileSize: { w: 600, h: 600 },
          layout: 'grid',
          grout: { size: 2, color: '#f5f5f2' },
        },
      });
    }
  }

  if (refineInfo) diagnostics.refined = refineInfo;

  const gcam = geo3d?.camera;
  const timings = { segmentation: tSeg, ...(geoRun?.timings ?? {}), total: Date.now() - t0 };
  const scan = {
    version: SCAN_API_VERSION,
    scanner: geoOk ? 'geometry3d' : 'legacy',
    quality,
    fallbackReason: geoOk ? undefined : (diagnostics.scannerFallback ?? (scannerMode === 'legacy' ? 'legacy scanner requested' : 'no wall class detected')),
    walls: scanWalls,
    corners: geoOk ? geo3d.corners.map(publicCorner) : [],
    rejectedCorners: geoOk ? geo3d.rejectedCorners.map(publicRejected) : [],
    floor: geo3d?.floor ? { plane: geo3d.floor.plane, polygon3D: geo3d.floor.polygon3D } : null,
    ceiling: geo3d?.ceiling ? { plane: geo3d.ceiling.plane, height: geo3d.ceiling.height, polygon3D: geo3d.ceiling.polygon3D } : null,
    camera: gcam ?? null,
    confidenceBands: CONFIDENCE_BANDS,
    metadata: {
      processingTimeMs: timings.total,
      timings,
      depth: geoRun?.depth ?? null,
      modelVersions: {
        segmentationModel: loadedSegModel,
        depthModel: geoRun?.depth?.modelVersions?.depthModel ?? null,
        depthLicense: geoRun?.depth?.modelVersions?.depthLicense ?? null,
        geometryVersion: GEOMETRY_VERSION,
        cvService: geoRun?.depth?.modelVersions?.cvService ?? null,
      },
      debugDir: geoRun?.debugFiles ? debugDir : null,
    },
  };

  return {
    objectList,
    diagnostics,
    camera: gcam ? {
      focal: round2(gcam.focalPx),
      height: round2(gcam.height),
      heightSource: gcam.heightSource,
      hfov: round2(gcam.hfov),
      roomHeight,
      focalSource: gcam.focalSource,
      pitchDeg: gcam.pitchDeg,
      rollDeg: gcam.rollDeg,
      scaleSource: gcam.scaleSource,
    } : {
      focal: round2(cam.f),
      height: round2(cam.height),
      heightSource: cam.source,
      hfov,
      roomHeight,
    },
    scan,
    detected: Object.keys(masks),
  };
}
