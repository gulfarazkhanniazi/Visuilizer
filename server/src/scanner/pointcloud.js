/**
 * Depth map -> gravity-aligned 3D point cloud with surface normals.
 *
 * Frames
 *   camera   x right, y down, z forward (the optical axis), origin at the lens
 *   world    X right, Y up, Z forward-horizontal, origin on the floor directly
 *            below the camera. This is the frame the legacy scanner used for a
 *            level camera; here it is recovered for a tilted one too, from the
 *            floor plane's normal.
 *
 * Everything is computed on a coarse "geometry grid" (256-512 columns). Planes
 * need thousands of points, not millions, and monocular depth carries no more
 * detail than that anyway.
 */
import {
  dot, cross, unit, deg, rad, clamp, fitPlaneLS, rng, percentile, median, bilinear,
} from './math.js';
import { CAMERA, DEPTH, PLANES } from './config.js';

export const CLS = {
  OTHER: 0, WALL: 1, FLOOR: 2, CEILING: 3, WALL_OBJECT: 4, FLOOR_OBJECT: 5,
};

/** Sample the refinement-resolution maps down to the geometry grid. */
export function buildGrid({ w, h, RW, RH, masks, objects, depth, gridW }) {
  const GW = Math.min(gridW, RW);
  const GH = Math.max(8, Math.round((RH / RW) * GW));
  const n = GW * GH;
  const cls = new Uint8Array(n);
  const Z = new Float32Array(n);
  const conf = new Float32Array(n).fill(1);

  for (let gy = 0; gy < GH; gy++) {
    const ry = Math.min(RH - 1, Math.floor(((gy + 0.5) * RH) / GH));
    const dy = ((gy + 0.5) * depth.height) / GH - 0.5;
    for (let gx = 0; gx < GW; gx++) {
      const rx = Math.min(RW - 1, Math.floor(((gx + 0.5) * RW) / GW));
      const ri = ry * RW + rx;
      const i = gy * GW + gx;
      if (masks.wall?.[ri]) cls[i] = CLS.WALL;
      else if (masks.floor?.[ri]) cls[i] = CLS.FLOOR;
      else if (masks.ceiling?.[ri]) cls[i] = CLS.CEILING;
      else if (objects.wall?.[ri] >= 0.3) cls[i] = CLS.WALL_OBJECT;
      else if (objects.floor?.[ri] >= 0.3) cls[i] = CLS.FLOOR_OBJECT;
      const dx = ((gx + 0.5) * depth.width) / GW - 0.5;
      Z[i] = bilinear(depth.data, depth.width, depth.height, dx, dy);
      if (depth.confidence) conf[i] = bilinear(depth.confidence, depth.width, depth.height, dx, dy) / 255;
    }
  }
  // Pixels whose 4-neighbours all share their class: away from semantic
  // boundaries, where both the class and the depth are most trustworthy.
  const interior = new Uint8Array(n);
  for (let gy = 1; gy < GH - 1; gy++) {
    for (let gx = 1; gx < GW - 1; gx++) {
      const i = gy * GW + gx;
      const c = cls[i];
      interior[i] = cls[i - 1] === c && cls[i + 1] === c && cls[i - GW] === c && cls[i + GW] === c ? 1 : 0;
    }
  }
  return { GW, GH, w, h, sx: w / GW, sy: h / GH, cls, Z, conf, interior };
}

/**
 * Relative (affine-invariant) disparity -> metric depth, by aligning it to the
 * floor. For a level camera the floor's inverse depth is exactly linear in the
 * image row, 1/Z = (v - cy) / (f h), so a robust line fit of disparity against
 * (v - cy) over floor pixels recovers the model's unknown scale and shift.
 */
export function alignRelativeDepth(grid, { f, cy, camHeight }) {
  const xs = []; const ys = [];
  for (let gy = 0; gy < grid.GH; gy++) {
    const v = (gy + 0.5) * grid.sy;
    if (v - cy < grid.h * 0.06) continue;
    for (let gx = 0; gx < grid.GW; gx += 2) {
      const i = gy * grid.GW + gx;
      if (grid.cls[i] === CLS.FLOOR && grid.interior[i]) { xs.push(v - cy); ys.push(grid.Z[i]); }
    }
  }
  if (xs.length < 60) return { ok: false, reason: 'too little floor to align relative depth' };
  // RANSAC line, then least squares on inliers.
  const rand = rng(7);
  let best = null;
  const spread = percentile(ys, 0.9) - percentile(ys, 0.1) || 1;
  const tol = spread * 0.04;
  for (let it = 0; it < 200; it++) {
    const a = Math.floor(rand() * xs.length);
    const b = Math.floor(rand() * xs.length);
    if (Math.abs(xs[a] - xs[b]) < 1e-3) continue;
    const s = (ys[b] - ys[a]) / (xs[b] - xs[a]);
    const c = ys[a] - s * xs[a];
    let score = 0;
    for (let k = 0; k < xs.length; k++) score += Math.abs(ys[k] - (s * xs[k] + c)) < tol ? 1 : 0;
    if (!best || score > best.score) best = { s, c, score };
  }
  if (!best || best.s <= 0) return { ok: false, reason: 'disparity does not increase towards the camera on the floor' };
  let sxx = 0; let sx = 0; let sy = 0; let sxy = 0; let m = 0;
  for (let k = 0; k < xs.length; k++) {
    if (Math.abs(ys[k] - (best.s * xs[k] + best.c)) >= tol) continue;
    sxx += xs[k] * xs[k]; sx += xs[k]; sy += ys[k]; sxy += xs[k] * ys[k]; m++;
  }
  const s = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  const c = (sy - s * sx) / m;
  if (!(s > 0)) return { ok: false, reason: 'degenerate floor alignment' };
  // disparity r = s (v - cy) + c on the floor  =>  1/Z = (r - c) / (s f h)
  for (let i = 0; i < grid.Z.length; i++) {
    const inv = (grid.Z[i] - c) / (s * f * camHeight);
    grid.Z[i] = inv > 1 / DEPTH.maxDepth ? 1 / inv : DEPTH.maxDepth;
  }
  return { ok: true, inlierFrac: m / xs.length };
}

/**
 * Depth discontinuities. Inverse depth is affine in the image over any plane,
 * so its second difference is zero on every flat surface, small across a
 * crease (a room corner) and large across an occlusion boundary. That makes it
 * a far better silhouette detector than the gradient of Z itself, which is
 * large on any floor seen near the horizon.
 */
export function depthEdges(grid) {
  const { GW, GH, Z } = grid;
  const jump = new Float32Array(GW * GH);
  const crease = new Float32Array(GW * GH);
  const inv = (i) => 1 / Math.max(Z[i], 1e-3);
  for (let gy = 1; gy < GH - 1; gy++) {
    for (let gx = 1; gx < GW - 1; gx++) {
      const i = gy * GW + gx;
      const w0 = inv(i);
      const ddx = Math.abs(inv(i - 1) - 2 * w0 + inv(i + 1)) / w0;
      const ddy = Math.abs(inv(i - GW) - 2 * w0 + inv(i + GW)) / w0;
      const jx = Math.abs(inv(i + 1) - inv(i - 1)) / (2 * w0);
      const jy = Math.abs(inv(i + GW) - inv(i - GW)) / (2 * w0);
      crease[i] = Math.max(ddx, ddy);
      jump[i] = Math.max(jx, jy);
    }
  }
  return { jump, crease };
}

/** Back-project the grid through a pinhole: camera-frame points. */
export function backProject(grid, cam) {
  const n = grid.GW * grid.GH;
  const x = new Float32Array(n); const y = new Float32Array(n); const z = new Float32Array(n);
  const valid = new Uint8Array(n);
  for (let gy = 0; gy < grid.GH; gy++) {
    const b = ((gy + 0.5) * grid.sy - cam.cy) / cam.f;
    for (let gx = 0; gx < grid.GW; gx++) {
      const i = gy * grid.GW + gx;
      const Zi = grid.Z[i];
      const a = ((gx + 0.5) * grid.sx - cam.cx) / cam.f;
      x[i] = a * Zi; y[i] = b * Zi; z[i] = Zi;
      valid[i] = Zi > DEPTH.minDepth && Zi < DEPTH.maxDepth && Number.isFinite(Zi) ? 1 : 0;
    }
  }
  return { x, y, z, valid };
}

/**
 * Surface normals from the point cloud: cross product of central differences,
 * skipping any stencil that straddles a depth discontinuity, then a small
 * validity-weighted box smooth. Oriented to face the camera.
 */
export function computeNormals(grid, P, edges, { step = 2, smooth = 1 } = {}) {
  const { GW, GH } = grid;
  const n = GW * GH;
  let nx = new Float32Array(n); let ny = new Float32Array(n); let nz = new Float32Array(n);
  const ok = new Uint8Array(n);
  const jumpTol = DEPTH.maxRelGradient;
  for (let gy = step; gy < GH - step; gy++) {
    for (let gx = step; gx < GW - step; gx++) {
      const i = gy * GW + gx;
      if (!P.valid[i]) continue;
      const l = i - step; const r = i + step; const u = i - step * GW; const d = i + step * GW;
      if (!P.valid[l] || !P.valid[r] || !P.valid[u] || !P.valid[d]) continue;
      let blocked = false;
      for (let k = -step; k <= step && !blocked; k++) {
        if (edges.jump[i + k] > jumpTol || edges.jump[i + k * GW] > jumpTol) blocked = true;
      }
      if (blocked) continue;
      const ax = P.x[r] - P.x[l]; const ay = P.y[r] - P.y[l]; const az = P.z[r] - P.z[l];
      const bx = P.x[d] - P.x[u]; const by = P.y[d] - P.y[u]; const bz = P.z[d] - P.z[u];
      let cx = ay * bz - az * by; let cy = az * bx - ax * bz; let cz = ax * by - ay * bx;
      const len = Math.hypot(cx, cy, cz);
      if (!(len > 0)) continue;
      cx /= len; cy /= len; cz /= len;
      if (cx * P.x[i] + cy * P.y[i] + cz * P.z[i] > 0) { cx = -cx; cy = -cy; cz = -cz; }
      nx[i] = cx; ny[i] = cy; nz[i] = cz; ok[i] = 1;
    }
  }
  for (let pass = 0; pass < smooth; pass++) {
    const sx = new Float32Array(n); const sy = new Float32Array(n); const sz = new Float32Array(n);
    for (let gy = 1; gy < GH - 1; gy++) {
      for (let gx = 1; gx < GW - 1; gx++) {
        const i = gy * GW + gx;
        if (!ok[i]) continue;
        let ax = 0; let ay = 0; let az = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const j = i + dy * GW + dx;
            // Only average with neighbours on the same surface.
            if (!ok[j] || nx[i] * nx[j] + ny[i] * ny[j] + nz[i] * nz[j] < 0.9) continue;
            ax += nx[j]; ay += ny[j]; az += nz[j];
          }
        }
        const len = Math.hypot(ax, ay, az) || 1;
        sx[i] = ax / len; sy[i] = ay / len; sz[i] = az / len;
      }
    }
    nx = sx; ny = sy; nz = sz;
  }
  return { nx, ny, nz, ok };
}

const planeTol = (Z) => PLANES.tolAbs + PLANES.tolRel * Z;

/**
 * Robust plane through points of one semantic class, with its normal
 * constrained to within `maxDeg` of `prior`. MSAC: sample three points, score
 * by truncated squared residual, refit by least squares on the inliers.
 */
export function ransacPlane(P, idx, { prior, maxDeg = 45, iters = 220, seed = 1 } = {}) {
  if (idx.length < 30) return null;
  const rand = rng(seed);
  const cosMax = Math.cos(rad(maxDeg));
  let best = null;
  for (let it = 0; it < iters; it++) {
    const a = idx[Math.floor(rand() * idx.length)];
    const b = idx[Math.floor(rand() * idx.length)];
    const c = idx[Math.floor(rand() * idx.length)];
    const u = [P.x[b] - P.x[a], P.y[b] - P.y[a], P.z[b] - P.z[a]];
    const v = [P.x[c] - P.x[a], P.y[c] - P.y[a], P.z[c] - P.z[a]];
    let nrm = cross(u, v);
    const len = Math.hypot(...nrm);
    if (len < 1e-6) continue;
    nrm = [nrm[0] / len, nrm[1] / len, nrm[2] / len];
    if (prior) {
      const c0 = dot(nrm, prior);
      if (c0 < 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
      if (Math.abs(c0) < cosMax) continue;
    }
    const d = -(nrm[0] * P.x[a] + nrm[1] * P.y[a] + nrm[2] * P.z[a]);
    let cost = 0;
    const stride = idx.length > 4000 ? Math.ceil(idx.length / 4000) : 1;
    for (let k = 0; k < idx.length; k += stride) {
      const i = idx[k];
      const T = planeTol(P.z[i]);
      const r = nrm[0] * P.x[i] + nrm[1] * P.y[i] + nrm[2] * P.z[i] + d;
      cost += Math.min(r * r, T * T) / (T * T);
    }
    if (!best || cost < best.cost) best = { n: nrm, d, cost };
  }
  if (!best) return null;
  let plane = best;
  for (let iter = 0; iter < 3; iter++) {
    const inl = [];
    for (const i of idx) {
      const r = plane.n[0] * P.x[i] + plane.n[1] * P.y[i] + plane.n[2] * P.z[i] + plane.d;
      if (Math.abs(r) < planeTol(P.z[i])) inl.push(i);
    }
    if (inl.length < 20) break;
    const ls = fitPlaneLS(P, inl);
    if (!ls) break;
    let nn = ls.n;
    if (prior && dot(nn, prior) < 0) nn = [-nn[0], -nn[1], -nn[2]];
    plane = { n: nn, d: -(nn[0] * ls.centroid[0] + nn[1] * ls.centroid[1] + nn[2] * ls.centroid[2]), inliers: inl };
  }
  if (!plane.inliers) return null;
  const res = plane.inliers.map((i) => Math.abs(plane.n[0] * P.x[i] + plane.n[1] * P.y[i] + plane.n[2] * P.z[i] + plane.d) / P.z[i]);
  return {
    n: plane.n,
    d: plane.d,
    inliers: plane.inliers,
    inlierRatio: plane.inliers.length / idx.length,
    medianRelResidual: median(res),
    p95RelResidual: percentile(res, 0.95),
  };
}

function classIndices(grid, P, cls, { interiorOnly = true, stride = 1 } = {}) {
  const out = [];
  for (let i = 0; i < grid.cls.length; i += stride) {
    if (grid.cls[i] === cls && P.valid[i] && (!interiorOnly || grid.interior[i])) out.push(i);
  }
  return out;
}

/**
 * Floor and ceiling planes, and from them the gravity direction.
 *
 * Gravity comes from the floor normal where there is floor, the ceiling
 * normal otherwise, the cross product of two differently-oriented walls after
 * that, and a level camera only as the last resort.
 */
export function horizontalPlanes(grid, P) {
  const down = [0, 1, 0];
  const floorIdx = classIndices(grid, P, CLS.FLOOR);
  const ceilIdx = classIndices(grid, P, CLS.CEILING);
  const minPts = Math.max(60, grid.GW * grid.GH * 0.01);
  let floor = null;
  let ceiling = null;
  if (floorIdx.length >= minPts) {
    const pl = ransacPlane(P, floorIdx, { prior: [0, -1, 0], maxDeg: 40, seed: 11 });
    if (pl && pl.inlierRatio > 0.35) floor = pl;
  }
  if (ceilIdx.length >= minPts) {
    const pl = ransacPlane(P, ceilIdx, { prior: down, maxDeg: 40, seed: 13 });
    if (pl && pl.inlierRatio > 0.35) ceiling = pl;
  }
  return { floor, ceiling, floorPoints: floorIdx.length, ceilingPoints: ceilIdx.length };
}

/**
 * Build the camera->world rotation from an up vector (in camera coordinates).
 * Returns the basis and the pitch/roll it implies.
 */
export function worldFrame(up, camHeight) {
  const zc = [0, 0, 1];
  const fwd = unit([zc[0] - dot(zc, up) * up[0], zc[1] - dot(zc, up) * up[1], zc[2] - dot(zc, up) * up[2]]);
  const xhat = unit(cross(fwd, up));
  const pitch = deg(Math.asin(clamp(dot(zc, up), -1, 1)));
  const roll = deg(Math.asin(clamp(dot([1, 0, 0], up), -1, 1)));
  return { up, fwd, xhat, camHeight, pitch, roll };
}

/** Camera-frame point -> world. */
export function toWorld(frame, x, y, z) {
  const p = [x, y, z];
  return [dot(frame.xhat, p), dot(frame.up, p) + frame.camHeight, dot(frame.fwd, p)];
}

/** Camera-frame direction -> world direction. */
export function dirToWorld(frame, v) {
  return [dot(frame.xhat, v), dot(frame.up, v), dot(frame.fwd, v)];
}

/** World point -> camera frame. */
export function toCamera(frame, X, Y, Z) {
  const { xhat, up, fwd } = frame;
  const yy = Y - frame.camHeight;
  return [
    X * xhat[0] + yy * up[0] + Z * fwd[0],
    X * xhat[1] + yy * up[1] + Z * fwd[1],
    X * xhat[2] + yy * up[2] + Z * fwd[2],
  ];
}

/** World point -> image pixel (full-resolution photo coordinates). */
export function projectWorld(cam, frame, X, Y, Z, minZ = 1e-3) {
  const p = toCamera(frame, X, Y, Z);
  if (p[2] <= minZ) return null;
  return [cam.cx + (cam.f * p[0]) / p[2], cam.cy + (cam.f * p[1]) / p[2]];
}

/** World-frame points and normals for the whole grid. */
export function worldCloud(frame, P, N) {
  const n = P.x.length;
  const X = new Float32Array(n); const Y = new Float32Array(n); const Zw = new Float32Array(n);
  const nX = new Float32Array(n); const nY = new Float32Array(n); const nZ = new Float32Array(n);
  const { xhat, up, fwd, camHeight } = frame;
  for (let i = 0; i < n; i++) {
    const x = P.x[i]; const y = P.y[i]; const z = P.z[i];
    X[i] = xhat[0] * x + xhat[1] * y + xhat[2] * z;
    Y[i] = up[0] * x + up[1] * y + up[2] * z + camHeight;
    Zw[i] = fwd[0] * x + fwd[1] * y + fwd[2] * z;
    if (N.ok[i]) {
      const a = N.nx[i]; const b = N.ny[i]; const c = N.nz[i];
      nX[i] = xhat[0] * a + xhat[1] * b + xhat[2] * c;
      nY[i] = up[0] * a + up[1] * b + up[2] * c;
      nZ[i] = fwd[0] * a + fwd[1] * b + fwd[2] * c;
    }
  }
  return { X, Y, Z: Zw, nX, nY, nZ, ok: N.ok, valid: P.valid, depth: P.z };
}

/**
 * Walls are perpendicular to the floor. Given only depth, a wrong focal length
 * shears the point cloud and the walls stop being vertical, so the focal
 * length that makes wall normals most nearly horizontal -- relative to the
 * floor recovered at that same focal length -- is the camera's.
 *
 * Evaluated on a subsampled grid so the whole search costs tens of
 * milliseconds; golden-section refinement around the best coarse sample.
 */
export function selfCalibrateFocal(grid, edges, { w, cx, cy }) {
  const floorStride = 2;
  const cost = (hfov) => {
    const f = (w / 2) / Math.tan(rad(hfov) / 2);
    const cam = { f, cx, cy };
    const P = backProject(grid, cam);
    const floorIdx = classIndices(grid, P, CLS.FLOOR, { stride: floorStride });
    if (floorIdx.length < 50) return null;
    const pl = ransacPlane(P, floorIdx, { prior: [0, -1, 0], maxDeg: 40, iters: 80, seed: 3 });
    if (!pl) return null;
    const N = computeNormals(grid, P, edges, { step: 2, smooth: 0 });
    const vals = [];
    for (let i = 0; i < grid.cls.length; i += 3) {
      if (grid.cls[i] !== CLS.WALL || !grid.interior[i] || !N.ok[i]) continue;
      const c = Math.abs(N.nx[i] * pl.n[0] + N.ny[i] * pl.n[1] + N.nz[i] * pl.n[2]);
      vals.push(c);
    }
    if (vals.length < 50) return null;
    // Median, not mean: noisy normals near corners should not steer it.
    return median(vals);
  };

  const [lo, hi] = CAMERA.hfovRange;
  const samples = [];
  for (let k = 0; k <= 12; k++) {
    const hf = lo + ((hi - lo) * k) / 12;
    const c = cost(hf);
    if (c !== null) samples.push({ hf, c });
  }
  if (samples.length < 5) return null;
  samples.sort((p, q) => p.c - q.c);
  let a = Math.max(lo, samples[0].hf - (hi - lo) / 12);
  let b = Math.min(hi, samples[0].hf + (hi - lo) / 12);
  const g = (Math.sqrt(5) - 1) / 2;
  let c1 = b - g * (b - a); let c2 = a + g * (b - a);
  let f1 = cost(c1) ?? Infinity; let f2 = cost(c2) ?? Infinity;
  for (let it = 0; it < 10; it++) {
    if (f1 < f2) { b = c2; c2 = c1; f2 = f1; c1 = b - g * (b - a); f1 = cost(c1) ?? Infinity; }
    else { a = c1; c1 = c2; f1 = f2; c2 = a + g * (b - a); f2 = cost(c2) ?? Infinity; }
  }
  const hfov = (a + b) / 2;
  const best = cost(hfov);
  const worst = samples[samples.length - 1].c;
  // A flat cost curve means the geometry does not constrain the focal length
  // (e.g. a single wall seen square-on): report that rather than a guess.
  const contrast = worst > 0 ? 1 - best / worst : 0;
  return { hfov, verticalityResidual: best, contrast, curve: samples.sort((p, q) => p.hf - q.hf) };
}
