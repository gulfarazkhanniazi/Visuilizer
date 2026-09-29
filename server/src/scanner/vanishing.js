/**
 * Vanishing points from line segments, and the focal length they imply.
 *
 * Why not from depth: monocular depth networks compress depth non-linearly,
 * and a point cloud built from compressed depth has its angles squeezed -- a
 * search for the focal length that makes walls perpendicular to the floor
 * then drifts to the widest lens it is allowed (measured: ~105 deg on photos
 * Depth Pro puts at 55-70). Line segments do not depend on depth at all.
 *
 * Two vanishing points of perpendicular world directions, measured from the
 * principal point, satisfy  v1 . v2 = -f^2. That holds for the vertical
 * against any horizontal direction, and for two horizontal directions when
 * the walls they come from meet at a right angle -- a Manhattan assumption,
 * so the result is used as a prior and cross-checked, never as ground truth.
 */
import { rng } from './math.js';

function toLine(s, cx, cy, S) {
  // Homogeneous line through two points, in centred, scaled coordinates.
  const x1 = (s[0] - cx) / S; const y1 = (s[1] - cy) / S;
  const x2 = (s[2] - cx) / S; const y2 = (s[3] - cy) / S;
  const l = [y1 - y2, x2 - x1, x1 * y2 - x2 * y1];
  const n = Math.hypot(l[0], l[1]) || 1;
  return {
    l: [l[0] / n, l[1] / n, l[2] / n],
    mid: [(x1 + x2) / 2, (y1 + y2) / 2],
    dir: [(x2 - x1), (y2 - y1)],
    len: Math.hypot(x2 - x1, y2 - y1),
    angle: Math.atan2(y2 - y1, x2 - x1),
  };
}

const crossH = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Angle between a segment and the line joining its midpoint to the VP. */
function vpAngle(seg, vp) {
  let dx; let dy;
  if (Math.abs(vp[2]) < 1e-9) { dx = vp[0]; dy = vp[1]; } else { dx = vp[0] / vp[2] - seg.mid[0]; dy = vp[1] / vp[2] - seg.mid[1]; }
  const a = Math.hypot(dx, dy); const b = Math.hypot(seg.dir[0], seg.dir[1]);
  if (!a || !b) return Math.PI / 2;
  const c = Math.abs(dx * seg.dir[0] + dy * seg.dir[1]) / (a * b);
  return Math.acos(Math.min(1, c));
}

function ransacVp(segs, rand, { iters = 400, tolDeg = 1.5, filter = null } = {}) {
  const pool = filter ? segs.filter(filter) : segs;
  if (pool.length < 4) return null;
  const tol = (tolDeg * Math.PI) / 180;
  let best = null;
  for (let it = 0; it < iters; it++) {
    const a = pool[Math.floor(rand() * pool.length)];
    const b = pool[Math.floor(rand() * pool.length)];
    if (a === b) continue;
    const vp = crossH(a.l, b.l);
    const n = Math.hypot(...vp);
    if (!n) continue;
    const v = vp.map((x) => x / n);
    let score = 0;
    for (const s of pool) if (vpAngle(s, v) < tol) score += s.len;
    if (!best || score > best.score) best = { v, score };
  }
  if (!best) return null;
  // Refine: least squares of l.v over inliers = smallest eigenvector of sum l l^T.
  const inl = pool.filter((s) => vpAngle(s, best.v) < tol);
  const M = [0, 0, 0, 0, 0, 0];
  for (const s of inl) {
    const [a, b, c] = s.l; const wt = s.len;
    M[0] += wt * a * a; M[1] += wt * a * b; M[2] += wt * a * c; M[3] += wt * b * b; M[4] += wt * b * c; M[5] += wt * c * c;
  }
  return { v: smallestEig(M) ?? best.v, inliers: inl, support: inl.reduce((s, x) => s + x.len, 0) };
}

function smallestEig(m) {
  // Power iteration on (tr*I - M) finds the smallest eigenvector of M.
  const tr = m[0] + m[3] + m[5];
  const A = [[tr - m[0], -m[1], -m[2]], [-m[1], tr - m[3], -m[4]], [-m[2], -m[4], tr - m[5]]];
  let v = [0.3, 0.5, 0.8];
  for (let k = 0; k < 60; k++) {
    const nv = [0, 1, 2].map((r) => A[r][0] * v[0] + A[r][1] * v[1] + A[r][2] * v[2]);
    const n = Math.hypot(...nv);
    if (!n) return null;
    v = nv.map((x) => x / n);
  }
  return v;
}

/**
 * Detect up to three vanishing points and estimate the focal length.
 * `segments` are [x1, y1, x2, y2] in photo pixels.
 */
export function vanishingPoints(segments, w, h, { hfovRange = [38, 105] } = {}) {
  const cx = w / 2; const cy = h / 2; const S = Math.max(w, h);
  const segs = segments.map((s) => toLine(s, cx, cy, S)).filter((s) => s.len > 0.02);
  const out = { vps: [], focalPx: null, method: null, segments: segs.length };
  if (segs.length < 12) return out;
  const rand = rng(99);

  // Vertical first: near-vertical segments in the image.
  const nearVertical = (s) => Math.abs(Math.abs(s.angle) - Math.PI / 2) < (25 * Math.PI) / 180;
  let vert = ransacVp(segs, rand, { filter: nearVertical });
  // A vertical VP sits far above or below the image (or at infinity). One
  // near the horizon row is the floor lines running straight ahead, which
  // are near-vertical in the image too -- not verticals.
  if (vert && Math.abs(vert.v[2]) > 1e-6 && Math.abs((vert.v[1] / vert.v[2]) * S) < 1.5 * h) vert = null;
  if (vert && vert.support < 0.3) vert = null;
  const used = new Set(vert?.inliers ?? []);
  if (vert) out.vps.push({ kind: 'vertical', ...pack(vert, S, cx, cy) });
  // Horizontal VPs share one horizon row. The best-supported one anchors it
  // (and must be within reach of the centre for a plausible pitch); a VP off
  // that row was fitted to a mix of unrelated lines.
  let horizonY = null;
  const vy = (vp) => (vp.v[1] / vp.v[2]) * S;
  const onHorizon = (vp) => {
    if (Math.abs(vp.v[2]) < 1e-6) return true;
    if (horizonY === null) {
      if (Math.abs(vy(vp)) > 0.35 * h) return false;
      horizonY = vy(vp);
      return true;
    }
    return Math.abs(vy(vp) - horizonY) < 0.1 * h;
  };

  // Horizontal directions from what is left, strongest first.
  let rest = segs.filter((s) => !used.has(s) && !nearVertical(s));
  const horiz = [];
  for (let k = 0; k < 3 && rest.length >= 6; k++) {
    const vp = ransacVp(rest, rand, {});
    if (!vp || vp.support < 0.25) break;
    if (onHorizon(vp)) horiz.push(vp);
    const inl = new Set(vp.inliers);
    rest = rest.filter((s) => !inl.has(s));
  }
  for (const vp of horiz) out.vps.push({ kind: 'horizontal', ...pack(vp, S, cx, cy) });

  // Candidate focal lengths from perpendicular pairs.
  const fMin = (w / 2) / Math.tan((hfovRange[1] * Math.PI) / 360);
  const fMax = (w / 2) / Math.tan((hfovRange[0] * Math.PI) / 360);
  const cands = [];
  const finite = (vp) => Math.abs(vp.v[2]) > 1e-6;
  const pt = (vp) => [(vp.v[0] / vp.v[2]) * S, (vp.v[1] / vp.v[2]) * S];
  for (let a = 0; a < horiz.length; a++) {
    for (let b = a + 1; b < horiz.length; b++) {
      if (!finite(horiz[a]) || !finite(horiz[b])) continue;
      const p = pt(horiz[a]); const q = pt(horiz[b]);
      const f2 = -(p[0] * q[0] + p[1] * q[1]);
      if (f2 > 0) {
        const f = Math.sqrt(f2);
        if (f > fMin && f < fMax) cands.push({ f, weight: Math.min(horiz[a].support, horiz[b].support), pair: 'horizontal-horizontal' });
      }
    }
    if (vert && finite(vert) && finite(horiz[a])) {
      const p = pt(vert); const q = pt(horiz[a]);
      const f2 = -(p[0] * q[0] + p[1] * q[1]);
      // A near-infinite vertical VP (level camera) makes this pair
      // ill-conditioned: only use it when the camera is visibly tilted.
      if (f2 > 0 && Math.hypot(...p) < 20 * S) {
        const f = Math.sqrt(f2);
        if (f > fMin && f < fMax) cands.push({ f, weight: Math.min(vert.support, horiz[a].support), pair: 'vertical-horizontal' });
      }
    }
  }
  out.candidates = cands.map((c) => ({ ...c, hfov: (2 * Math.atan(w / 2 / c.f) * 180) / Math.PI }));
  if (cands.length) {
    // Weighted median of every perpendicular pair's estimate.
    cands.sort((p, q) => p.f - q.f);
    const total = cands.reduce((a, c) => a + c.weight, 0);
    let acc = 0; let pick = cands[0];
    for (const c of cands) { acc += c.weight; if (acc >= total / 2) { pick = c; break; } }
    out.focalPx = pick.f;
    out.method = cands.length > 1 ? `${pick.pair}, median of ${cands.length}` : pick.pair;
    out.hfov = (2 * Math.atan(w / 2 / pick.f) * 180) / Math.PI;
  }
  return out;
}

function pack(vp, S, cx, cy) {
  const finite = Math.abs(vp.v[2]) > 1e-6;
  return {
    point: finite ? [cx + (vp.v[0] / vp.v[2]) * S, cy + (vp.v[1] / vp.v[2]) * S] : null,
    direction: finite ? null : [vp.v[0], vp.v[1]],
    support: +vp.support.toFixed(3),
    inliers: vp.inliers.length,
  };
}
