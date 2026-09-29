/**
 * Wall planes from the gravity-aligned point cloud.
 *
 * A wall is a vertical plane, which in the world frame is a line in the ground
 * (XZ) plane: nx*X + nz*Z = o, with (nx, nz) the unit normal facing the
 * camera. Verticality is a physical prior, not a Manhattan one -- nothing here
 * assumes walls meet at right angles.
 *
 *   1. directions   histogram of horizontal normal angles over wall pixels;
 *                   each peak is a wall orientation. Any angle is allowed, so
 *                   45 and 135 degree corners are found the same way as 90.
 *   2. offsets      for each direction, histogram of n.p; each peak is a
 *                   distinct parallel plane (the back wall and the front of a
 *                   chimney breast share a direction and differ here)
 *   3. refinement   MSAC 2-point RANSAC in XZ, then total least squares
 *   4. membership   every wall pixel joins the plane it fits best -- by
 *                   distance AND normal -- or none
 *   5. regions      connected components; coplanar pieces are one wall only if
 *                   nothing but furniture separates them
 *   6. merge        similar planes that are adjacent become one wall
 *   7. validation   residuals, normal consistency, inlier ratio, semantic
 *                   consistency; a plane failing them is rejected, not shipped
 *
 * Colour plays no part in any of it: two walls painted identically are two
 * walls because their normals differ.
 */
import { CLS } from './pointcloud.js';
import { PLANES } from './config.js';
import { rng, percentile, median, deg, rad, clamp } from './math.js';

const TAU = Math.PI * 2;
const tolAt = (dist) => PLANES.tolAbs + PLANES.tolRel * dist;
const wrap = (a) => ((a % TAU) + TAU) % TAU;
const angDiff = (a, b) => {
  const d = Math.abs(wrap(a) - wrap(b));
  return Math.min(d, TAU - d);
};

/** Total least squares line through points in XZ, oriented to face the camera. */
export function fitVertical(W, idx) {
  let mx = 0; let mz = 0;
  for (const i of idx) { mx += W.X[i]; mz += W.Z[i]; }
  mx /= idx.length; mz /= idx.length;
  let cxx = 0; let czz = 0; let cxz = 0;
  for (const i of idx) {
    const dx = W.X[i] - mx; const dz = W.Z[i] - mz;
    cxx += dx * dx; czz += dz * dz; cxz += dx * dz;
  }
  // Direction of the line = principal axis; normal = perpendicular to it.
  const theta = 0.5 * Math.atan2(2 * cxz, cxx - czz);
  let nx = -Math.sin(theta); let nz = Math.cos(theta);
  let o = nx * mx + nz * mz;
  if (o > 0) { nx = -nx; nz = -nz; o = -o; }
  return { nx, nz, o };
}

const residual = (pl, W, i) => pl.nx * W.X[i] + pl.nz * W.Z[i] - pl.o;
const horizDist = (W, i) => Math.hypot(W.X[i], W.Z[i]);

function msacVertical(W, idx, rand, iters) {
  let best = null;
  const stride = idx.length > 3000 ? Math.ceil(idx.length / 3000) : 1;
  for (let it = 0; it < iters; it++) {
    const a = idx[Math.floor(rand() * idx.length)];
    const b = idx[Math.floor(rand() * idx.length)];
    const dx = W.X[b] - W.X[a]; const dz = W.Z[b] - W.Z[a];
    const len = Math.hypot(dx, dz);
    if (len < 0.05) continue;
    let nx = -dz / len; let nz = dx / len;
    let o = nx * W.X[a] + nz * W.Z[a];
    if (o > 0) { nx = -nx; nz = -nz; o = -o; }
    const pl = { nx, nz, o };
    let cost = 0;
    for (let k = 0; k < idx.length; k += stride) {
      const i = idx[k];
      const T = tolAt(horizDist(W, i));
      const r = residual(pl, W, i);
      cost += Math.min(r * r, T * T) / (T * T);
    }
    if (!best || cost < best.cost) best = { ...pl, cost };
  }
  if (!best) return null;
  let pl = best;
  for (let k = 0; k < 3; k++) {
    const inl = idx.filter((i) => Math.abs(residual(pl, W, i)) < tolAt(horizDist(W, i)));
    if (inl.length < 10) break;
    pl = fitVertical(W, inl);
  }
  return pl;
}

function histogramPeaks(values, weights, { lo, hi, bin, circular, smooth, minFrac, minSep }) {
  const nb = Math.max(1, Math.ceil((hi - lo) / bin));
  const hist = new Float64Array(nb);
  let total = 0;
  for (let k = 0; k < values.length; k++) {
    let b = Math.floor((values[k] - lo) / bin);
    if (circular) b = ((b % nb) + nb) % nb;
    if (b < 0 || b >= nb) continue;
    const wt = weights ? weights[k] : 1;
    hist[b] += wt; total += wt;
  }
  const sm = new Float64Array(nb);
  for (let b = 0; b < nb; b++) {
    let s = 0; let ws = 0;
    for (let d = -smooth; d <= smooth; d++) {
      let j = b + d;
      if (circular) j = ((j % nb) + nb) % nb;
      else if (j < 0 || j >= nb) continue;
      const wk = smooth + 1 - Math.abs(d);
      s += hist[j] * wk; ws += wk;
    }
    sm[b] = s / ws;
  }
  const peaks = [];
  for (let b = 0; b < nb; b++) {
    const l = circular ? sm[(b - 1 + nb) % nb] : (b > 0 ? sm[b - 1] : -1);
    const r = circular ? sm[(b + 1) % nb] : (b < nb - 1 ? sm[b + 1] : -1);
    if (sm[b] >= l && sm[b] > r) peaks.push({ at: lo + (b + 0.5) * bin, mass: sm[b] * (2 * smooth + 1) });
  }
  peaks.sort((p, q) => q.mass - p.mass);
  const kept = [];
  for (const p of peaks) {
    if (p.mass < minFrac * total) break;
    const dist = (a, b) => (circular ? angDiff(a, b) : Math.abs(a - b));
    if (kept.some((k) => dist(k.at, p.at) < minSep)) continue;
    kept.push(p);
  }
  return kept;
}

/** 8-connected components of pixels carrying `label === id`. */
function components(label, GW, GH, id) {
  const seen = new Uint8Array(GW * GH);
  const out = [];
  for (let s = 0; s < label.length; s++) {
    if (label[s] !== id || seen[s]) continue;
    const pix = [];
    const stack = [s];
    seen[s] = 1;
    while (stack.length) {
      const p = stack.pop();
      pix.push(p);
      const x = p % GW; const y = (p / GW) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx; const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= GW || ny >= GH) continue;
          const q = ny * GW + nx;
          if (label[q] === id && !seen[q]) { seen[q] = 1; stack.push(q); }
        }
      }
    }
    out.push(pix);
  }
  return out;
}

/**
 * Are two regions separated by another wall plane (=> separate walls), or only
 * by furniture, openings and unlabelled pixels (=> one wall seen in pieces)?
 * Walks every row both regions occupy and classifies the pixels between them.
 */
function foreignGapFraction(a, b, groupOf, GW, selfIds) {
  const rowsA = new Map(); const rowsB = new Map();
  const note = (rows, p) => {
    const y = (p / GW) | 0; const x = p % GW;
    const r = rows.get(y);
    if (!r) rows.set(y, [x, x]); else { if (x < r[0]) r[0] = x; if (x > r[1]) r[1] = x; }
  };
  for (const p of a) note(rowsA, p);
  for (const p of b) note(rowsB, p);
  let gap = 0; let foreign = 0;
  for (const [y, ra] of rowsA) {
    const rb = rowsB.get(y);
    if (!rb) continue;
    const [x0, x1] = ra[1] < rb[0] ? [ra[1] + 1, rb[0] - 1] : rb[1] < ra[0] ? [rb[1] + 1, ra[0] - 1] : [1, 0];
    for (let x = x0; x <= x1; x++) {
      gap++;
      const g = groupOf[y * GW + x];
      if (g >= 0 && !selfIds.has(g)) foreign++;
    }
  }
  return gap ? foreign / gap : 0;
}

/**
 * Extract, validate and label wall planes.
 * Returns walls (with grid pixel lists) and a grid label map (-1 = none).
 */
export function extractWalls(grid, W, { seed = 1234 } = {}) {
  const { GW, GH, cls, interior } = grid;
  const n = GW * GH;
  const rand = rng(seed);
  const maxTilt = Math.sin(rad(PLANES.maxWallTiltDeg));

  const wallIdx = [];
  const seedIdx = []; const seedAng = [];
  for (let i = 0; i < n; i++) {
    if (cls[i] !== CLS.WALL || !W.valid[i]) continue;
    wallIdx.push(i);
    if (!interior[i] || !W.ok[i] || Math.abs(W.nY[i]) > maxTilt) continue;
    seedIdx.push(i);
    seedAng.push(wrap(Math.atan2(W.nZ[i], W.nX[i])));
  }
  const diag = { wallPoints: wallIdx.length, seedPoints: seedIdx.length, candidates: [], rejected: [] };
  if (seedIdx.length < Math.max(80, n * PLANES.minWallPointsFrac)) {
    diag.reason = 'too few wall points with reliable normals';
    return { walls: [], label: new Int16Array(n).fill(-1), diag };
  }

  // 1. directions ---------------------------------------------------------
  const dirPeaks = histogramPeaks(seedAng, null, {
    lo: 0, hi: TAU, bin: rad(PLANES.angleBinDeg), circular: true,
    smooth: PLANES.angleSmoothBins, minFrac: PLANES.minPeakFrac, minSep: rad(PLANES.minPeakSepDeg),
  });

  // 2 + 3. offsets per direction, then refinement --------------------------
  let candidates = [];
  for (const dp of dirPeaks) {
    const members = []; const offs = [];
    for (let k = 0; k < seedIdx.length; k++) {
      if (angDiff(seedAng[k], dp.at) > rad(15)) continue;
      const i = seedIdx[k];
      members.push(i);
      // log |offset| so the bin width is relative, like the depth error.
      offs.push(Math.log(Math.max(0.05, -(Math.cos(dp.at) * W.X[i] + Math.sin(dp.at) * W.Z[i]))));
    }
    if (members.length < 40) continue;
    // Fine offset bins: a chimney breast 25 cm proud of a wall 5 m away is a
    // 5% difference, and it must come out as its own peak. Spurious peaks
    // from depth drift along one wall are merged back below, by fit quality.
    const offPeaks = histogramPeaks(offs, null, {
      lo: Math.log(0.05), hi: Math.log(40), bin: PLANES.offsetBinLog, circular: false,
      smooth: PLANES.offsetSmoothBins, minFrac: PLANES.minOffsetPeakFrac, minSep: PLANES.minOffsetSepLog,
    });
    for (const op of offPeaks) {
      const o0 = -Math.exp(op.at);
      // Window: never reach past halfway to a neighbouring parallel peak, or
      // RANSAC on both windows converges on the same, larger plane.
      let half = Math.max(0.2, 0.07 * -o0);
      for (const other of offPeaks) {
        if (other === op) continue;
        half = Math.min(half, 0.5 * Math.abs(Math.exp(other.at) - Math.exp(op.at)));
      }
      const win = members.filter((i) => {
        const r = Math.cos(dp.at) * W.X[i] + Math.sin(dp.at) * W.Z[i] - o0;
        return Math.abs(r) < half;
      });
      if (win.length < 30) continue;
      const pl = msacVertical(W, win, rand, PLANES.ransacIters);
      if (!pl) continue;
      candidates.push({ ...pl, support: win.length });
    }
  }
  // Candidates from different peaks can land on the same plane.
  candidates.sort((p, q) => q.support - p.support);
  candidates = candidates.filter((c, k) => !candidates.slice(0, k).some((d) => samePlane(c, d)));
  diag.candidates = candidates.map((c) => describePlane(c));

  // 4. membership ---------------------------------------------------------
  const assign = (planes) => {
    const lab = new Int16Array(n).fill(-1);
    const cosTol = Math.cos(rad(PLANES.normalTolDeg));
    for (const i of wallIdx) {
      const dist = horizDist(W, i);
      const T = tolAt(dist);
      let best = -1; let bestR = Infinity;
      for (let k = 0; k < planes.length; k++) {
        const pl = planes[k];
        const r = Math.abs(residual(pl, W, i));
        if (r > T) continue;
        if (W.ok[i]) {
          const h = Math.hypot(W.nX[i], W.nZ[i]) || 1;
          if ((W.nX[i] * pl.nx + W.nZ[i] * pl.nz) / h < cosTol) continue;
        }
        if (r < bestR) { bestR = r; best = k; }
      }
      lab[i] = best;
    }
    return lab;
  };
  let label = assign(candidates);

  // 5. regions --------------------------------------------------------------
  const minComp = Math.max(25, n * PLANES.minComponentFrac);
  let groups = [];
  candidates.forEach((pl, k) => {
    for (const pix of components(label, GW, GH, k)) {
      if (pix.length >= minComp) groups.push({ plane: pl, pixels: pix, planeIds: new Set([k]) });
    }
  });
  const groupMap = () => {
    const g = new Int16Array(n).fill(-1);
    groups.forEach((gr, k) => { for (const p of gr.pixels) g[p] = k; });
    return g;
  };

  // 5 + 6. merge coplanar pieces separated only by furniture, and similar
  // planes that touch -- but never across another wall.
  let changed = true;
  while (changed) {
    changed = false;
    const gmap = groupMap();
    outer: for (let a = 0; a < groups.length; a++) {
      for (let b = a + 1; b < groups.length; b++) {
        const A = groups[a]; const B = groups[b];
        const same = [...A.planeIds].some((id) => B.planeIds.has(id));
        if (!same && !parallel(A.plane, B.plane)) continue;
        const foreign = foreignGapFraction(A.pixels, B.pixels, gmap, GW, new Set([a, b]));
        if (foreign > PLANES.maxForeignGapFrac) continue;
        const pixels = A.pixels.concat(B.pixels);
        const inl = pixels.filter((p) => W.valid[p]);
        const union = fitVertical(W, inl);
        // Geometric continuity: one plane must explain the union about as
        // well as the two separate planes explain their halves. Depth drift
        // along a single wall passes; a real step (chimney breast, recess)
        // does not, however similar the two offsets are.
        if (!same) {
          const mA = medianRelResidual(W, A.pixels, A.plane);
          const mB = medianRelResidual(W, B.pixels, B.plane);
          const mU = medianRelResidual(W, pixels, union);
          if (mU > PLANES.mergeResidualGain * Math.max(mA, mB, PLANES.mergeResidualFloor)) continue;
        }
        groups.splice(b, 1);
        groups[a] = { plane: union, pixels, planeIds: new Set([...A.planeIds, ...B.planeIds]) };
        changed = true;
        break outer;
      }
    }
  }

  // 5b. profile split -------------------------------------------------------
  // A histogram sees offsets, not where they are. Along each wall, the
  // per-column median of the wall's 3D position traces its base line left to
  // right; a step (chimney breast) or bend (a corner the normals smeared)
  // shows up there as a breakpoint even when it is only a few percent of the
  // distance. Columns, because a vertical wall occupies whole columns.
  const gmapFinal = groupMap();
  const pieces = [];
  groups.forEach((g, k) => {
    const split = profileSplit(grid, W, g, k, gmapFinal, wallIdx);
    if (split.length > 1) (diag.profileSplits ??= []).push(split.map((s) => ({ cols: s.cols, plane: describePlane(s.plane) })));
    for (const s of split) pieces.push(s);
  });

  // Refit each wall on its own pixels, then re-assign once against the final
  // planes so membership reflects them rather than the first guesses.
  const finalPlanes = pieces.map((g) => g.plane);
  label = assign(finalPlanes);
  const walls = [];
  finalPlanes.forEach((pl, k) => {
    const comps = components(label, GW, GH, k).filter((c) => c.length >= minComp);
    if (!comps.length) return;
    // Keep the pieces this group actually had; a plane's infinite extension
    // picking up a stray patch across the room is not part of this wall.
    const own = pieces[k].own;
    const pixels = [];
    for (const c of comps) if (c.some((p) => own.has(p))) pixels.push(...c);
    if (pixels.length >= minComp) walls.push({ plane: fitVertical(W, pixels), pixels });
  });

  // Final label map from the surviving walls only.
  label = new Int16Array(n).fill(-1);
  walls.forEach((wl, k) => { for (const p of wl.pixels) label[p] = k; });

  // 7. validation -----------------------------------------------------------
  const accepted = [];
  for (const wl of walls) {
    const v = validateWall(grid, W, wl, label);
    wl.stats = v;
    if (v.reject) diag.rejected.push({ plane: describePlane(wl.plane), reason: v.reject, stats: v });
    else accepted.push(wl);
  }
  label = new Int16Array(n).fill(-1);
  accepted.forEach((wl, k) => { for (const p of wl.pixels) label[p] = k; });
  diag.accepted = accepted.length;
  return { walls: accepted, label, diag };
}

/**
 * Split one wall group along its column profile. Returns one or more
 * { plane, own:Set, cols:[c0,c1] }.
 */
function profileSplit(grid, W, group, gid, gmap, wallIdx) {
  const { GW } = grid;
  let c0 = GW; let c1 = -1;
  for (const p of group.pixels) { const x = p % GW; if (x < c0) c0 = x; if (x > c1) c1 = x; }
  const whole = () => [{ plane: fitVertical(W, group.pixels), own: new Set(group.pixels), cols: [c0, c1] }];
  // Every wall-class pixel in those columns not claimed by another wall.
  const byCol = new Map();
  for (const i of wallIdx) {
    const x = i % GW;
    if (x < c0 || x > c1) continue;
    const g = gmap[i];
    if (g >= 0 && g !== gid) continue;
    if (!byCol.has(x)) byCol.set(x, []);
    byCol.get(x).push(i);
  }
  const cols = [...byCol.keys()].sort((a, b) => a - b).filter((x) => byCol.get(x).length >= 3);
  const minCols = Math.max(4, Math.round(GW * PLANES.splitMinColsFrac));
  if (cols.length < 2 * minCols) return whole();
  const prof = cols.map((x) => {
    const pix = byCol.get(x);
    return { x, X: median(pix.map((i) => W.X[i])), Z: median(pix.map((i) => W.Z[i])) };
  });

  // Prefix sums: best-fit line through any range of the profile in O(1).
  const m = prof.length;
  const S = { x: [0], z: [0], xx: [0], zz: [0], xz: [0] };
  for (let k = 0; k < m; k++) {
    const { X, Z } = prof[k];
    S.x.push(S.x[k] + X); S.z.push(S.z[k] + Z); S.xx.push(S.xx[k] + X * X); S.zz.push(S.zz[k] + Z * Z); S.xz.push(S.xz[k] + X * Z);
  }
  const fit = (a, b) => {
    const n = b - a + 1;
    const mx = (S.x[b + 1] - S.x[a]) / n; const mz = (S.z[b + 1] - S.z[a]) / n;
    const cxx = S.xx[b + 1] - S.xx[a] - n * mx * mx;
    const czz = S.zz[b + 1] - S.zz[a] - n * mz * mz;
    const cxz = S.xz[b + 1] - S.xz[a] - n * mx * mz;
    const tr = cxx + czz; const det = cxx * czz - cxz * cxz;
    const err = Math.max(0, (tr - Math.sqrt(Math.max(0, tr * tr - 4 * det))) / 2);
    const theta = 0.5 * Math.atan2(2 * cxz, cxx - czz);
    return { err, n, mx, mz, dir: [Math.cos(theta), Math.sin(theta)], dist: Math.hypot(mx, mz) };
  };
  const lineAt = (f, X, Z) => {
    // signed distance of (X,Z) from line f
    const nx = -f.dir[1]; const nz = f.dir[0];
    return nx * (X - f.mx) + nz * (Z - f.mz);
  };

  const ranges = [];
  const stack = [[0, m - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const all = fit(a, b);
    const relRms = Math.sqrt(all.err / all.n) / Math.max(0.5, all.dist);
    let cut = -1;
    if (relRms > PLANES.splitRelRms && b - a + 1 >= 2 * minCols) {
      let best = all.err * PLANES.splitGain;
      for (let k = a + minCols - 1; k <= b - minCols; k++) {
        const e = fit(a, k).err + fit(k + 1, b).err;
        if (e < best) { best = e; cut = k; }
      }
    }
    if (cut >= 0) {
      // Accept only a physical difference: a bend or a step, not bowing.
      const L = fit(a, cut); const R = fit(cut + 1, b);
      const bend = deg(Math.acos(Math.min(1, Math.abs(L.dir[0] * R.dir[0] + L.dir[1] * R.dir[1]))));
      const pB = prof[cut]; const pA = prof[cut + 1];
      const mid = [(pB.X + pA.X) / 2, (pB.Z + pA.Z) / 2];
      const step = Math.abs(lineAt(L, ...mid) - lineAt(R, ...mid));
      const dist = Math.hypot(...mid);
      if (bend >= PLANES.splitMinBendDeg || step >= Math.max(PLANES.splitMinStep, PLANES.splitMinStepRel * dist)) {
        stack.push([a, cut], [cut + 1, b]);
        continue;
      }
    }
    ranges.push([a, b]);
  }
  if (ranges.length < 2) return whole();
  ranges.sort((p, q) => p[0] - q[0]);
  return ranges.map(([a, b]) => {
    const x0 = prof[a].x; const x1 = prof[b].x;
    const pix = [];
    for (const x of cols) if (x >= x0 && x <= x1) pix.push(...byCol.get(x));
    return { plane: fitVertical(W, pix), own: new Set(pix), cols: [x0, x1] };
  });
}

export function parallel(a, b) {
  return a.nx * b.nx + a.nz * b.nz >= Math.cos(rad(PLANES.mergeAngleDeg));
}

/** Near-duplicate candidates: parallel and within half an inlier tolerance. */
function samePlane(a, b) {
  return parallel(a, b) && Math.abs(a.o - b.o) < 0.5 * tolAt(Math.max(Math.abs(a.o), Math.abs(b.o)));
}

function medianRelResidual(W, pixels, pl) {
  const r = [];
  const stride = pixels.length > 4000 ? Math.ceil(pixels.length / 4000) : 1;
  for (let k = 0; k < pixels.length; k += stride) {
    const p = pixels[k];
    if (!W.valid[p]) continue;
    r.push(Math.abs(residual(pl, W, p)) / Math.max(0.3, horizDist(W, p)));
  }
  return r.length ? median(r) : 0;
}

function describePlane(p) {
  return { normal: [+(p.nx.toFixed(3)), +(p.nz.toFixed(3))], offset: +p.o.toFixed(3), azimuthDeg: +deg(Math.atan2(p.nz, p.nx)).toFixed(1) };
}

/**
 * The checks every wall plane must pass. Residuals are relative to distance
 * because monocular depth error is: 5 cm is a poor fit at 1 m and an
 * excellent one at 8 m.
 */
export function validateWall(grid, W, wall, label) {
  const { GW, GH, cls } = grid;
  const pl = wall.plane;
  const res = []; let nOk = 0; let nSum = 0; let tilt = 0; let tiltN = 0;
  for (const p of wall.pixels) {
    const d = horizDist(W, p);
    res.push(Math.abs(residual(pl, W, p)) / Math.max(0.3, d));
    if (W.ok[p]) {
      const h = Math.hypot(W.nX[p], W.nZ[p]);
      nSum += Math.max(0, (W.nX[p] * pl.nx + W.nZ[p] * pl.nz)); nOk++;
      tilt += Math.abs(W.nY[p]) / Math.max(1e-6, Math.hypot(h, W.nY[p])); tiltN++;
    }
  }
  // Semantic consistency: how much of the wall's neighbourhood the segmenter
  // also calls wall (rather than floor, ceiling or furniture).
  let semWall = 0; let semAll = 0;
  for (let k = 0; k < wall.pixels.length; k += 3) {
    const p = wall.pixels[k];
    const x = p % GW; const y = (p / GW) | 0;
    for (let dy = -2; dy <= 2; dy += 2) {
      for (let dx = -2; dx <= 2; dx += 2) {
        const xx = x + dx; const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= GW || yy >= GH) continue;
        semAll++;
        const c = cls[yy * GW + xx];
        if (c === CLS.WALL || c === CLS.WALL_OBJECT) semWall++;
      }
    }
  }
  // Inlier ratio: of all wall-class pixels inside the wall's image footprint
  // (its rows, between its left and right edge), how many actually fit it.
  let foot = 0; let inl = 0;
  const rows = new Map();
  for (const p of wall.pixels) {
    const y = (p / GW) | 0; const x = p % GW;
    const r = rows.get(y);
    if (!r) rows.set(y, [x, x]); else { if (x < r[0]) r[0] = x; if (x > r[1]) r[1] = x; }
  }
  const self = label[wall.pixels[0]];
  for (const [y, [x0, x1]] of rows) {
    for (let x = x0; x <= x1; x++) {
      const i = y * GW + x;
      if (cls[i] !== CLS.WALL || !W.valid[i]) continue;
      const l = label[i];
      if (l >= 0 && l !== self) continue;          // belongs to a neighbour
      foot++;
      if (l === self) inl++;
    }
  }

  const stats = {
    points: wall.pixels.length,
    meanRelResidual: res.reduce((s, v) => s + v, 0) / res.length,
    medianRelResidual: median(res),
    p95RelResidual: percentile(res, 0.95),
    normalConsistency: nOk ? nSum / nOk : 0,
    verticalityDeg: tiltN ? deg(Math.asin(clamp(tilt / tiltN, 0, 1))) : null,
    inlierRatio: foot ? inl / foot : 0,
    segmentationConsistency: semAll ? semWall / semAll : 0,
  };
  if (stats.medianRelResidual > PLANES.maxMedianRelResidual) stats.reject = `median residual ${(stats.medianRelResidual * 100).toFixed(1)}% of distance`;
  else if (stats.p95RelResidual > PLANES.maxP95RelResidual) stats.reject = `95th percentile residual ${(stats.p95RelResidual * 100).toFixed(1)}% of distance`;
  else if (nOk > 20 && stats.normalConsistency < PLANES.minNormalConsistency) stats.reject = `normals inconsistent (${stats.normalConsistency.toFixed(2)})`;
  else if (stats.inlierRatio < PLANES.minInlierRatio) stats.reject = `only ${(stats.inlierRatio * 100).toFixed(0)}% of its footprint fits it`;

  // Confidence: every factor in 0..1, combined geometrically so one bad
  // factor cannot be averaged away by good ones.
  const fResid = clamp(1 - stats.medianRelResidual / PLANES.maxMedianRelResidual, 0, 1) * 0.6 + 0.4;
  const fNorm = clamp((stats.normalConsistency - 0.6) / 0.35, 0, 1);
  const fSupport = clamp(Math.log10(stats.points / 40) / 2, 0, 1);
  const fSem = clamp(stats.segmentationConsistency, 0, 1);
  const fInl = clamp(stats.inlierRatio, 0, 1);
  stats.confidence = +((fResid * fNorm * fSupport * fSem * fInl) ** (1 / 5)).toFixed(3);
  return stats;
}
