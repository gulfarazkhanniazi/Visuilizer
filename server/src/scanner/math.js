/**
 * The small amount of linear algebra the scanner needs, on plain arrays.
 */

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const unit = (a) => {
  const l = norm(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const deg = (r) => (r * 180) / Math.PI;
export const rad = (d) => (d * Math.PI) / 180;
export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const round = (v, k = 3) => {
  const m = 10 ** k;
  return Math.round(v * m) / m;
};

/** Angle between two directions, in degrees, ignoring sign if `unsigned`. */
export function angleBetween(a, b, unsigned = false) {
  let c = dot(unit(a), unit(b));
  if (unsigned) c = Math.abs(c);
  return deg(Math.acos(clamp(c, -1, 1)));
}

/**
 * Eigen-decomposition of a symmetric 3x3 matrix (Jacobi). Returns values in
 * ascending order with matching unit vectors. Used for least-squares plane
 * fits: the normal is the eigenvector of the smallest eigenvalue of the
 * scatter matrix.
 */
export function eigSym3(m) {
  const a = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-15) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const out = [0, 1, 2].map((i) => ({ value: a[i][i], vector: [v[0][i], v[1][i], v[2][i]] }));
  return out.sort((p, q) => p.value - q.value);
}

/**
 * Least-squares plane through weighted points (xs, ys, zs indexable by idx).
 * Returns { n, d } with n unit and n.p + d = 0.
 */
export function fitPlaneLS(P, idx, weights = null) {
  let sw = 0;
  let mx = 0; let my = 0; let mz = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const wt = weights ? weights[k] : 1;
    sw += wt;
    mx += wt * P.x[i]; my += wt * P.y[i]; mz += wt * P.z[i];
  }
  if (!(sw > 0)) return null;
  mx /= sw; my /= sw; mz /= sw;
  let xx = 0; let xy = 0; let xz = 0; let yy = 0; let yz = 0; let zz = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const wt = weights ? weights[k] : 1;
    const dx = P.x[i] - mx; const dy = P.y[i] - my; const dz = P.z[i] - mz;
    xx += wt * dx * dx; xy += wt * dx * dy; xz += wt * dx * dz;
    yy += wt * dy * dy; yz += wt * dy * dz; zz += wt * dz * dz;
  }
  const e = eigSym3([xx, xy, xz, yy, yz, zz]);
  const n = unit(e[0].vector);
  return { n, d: -(n[0] * mx + n[1] * my + n[2] * mz), centroid: [mx, my, mz], flatness: e[0].value / (e[1].value || 1) };
}

/** Deterministic PRNG (mulberry32): scans must be repeatable. */
export function rng(seed = 0x9e3779b9) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** k-th percentile (0..1) of a numeric array, without mutating it. */
export function percentile(arr, q) {
  if (!arr.length) return NaN;
  const s = Float64Array.from(arr).sort();
  const pos = clamp(q, 0, 1) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(s.length - 1, lo + 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function median(arr) { return percentile(arr, 0.5); }

/** Bilinear sample of a float grid at continuous coordinates. */
export function bilinear(data, w, h, x, y) {
  const fx = clamp(x, 0, w - 1);
  const fy = clamp(y, 0, h - 1);
  const x0 = Math.floor(fx); const y0 = Math.floor(fy);
  const x1 = Math.min(w - 1, x0 + 1); const y1 = Math.min(h - 1, y0 + 1);
  const ax = fx - x0; const ay = fy - y0;
  return (data[y0 * w + x0] * (1 - ax) + data[y0 * w + x1] * ax) * (1 - ay)
    + (data[y1 * w + x0] * (1 - ax) + data[y1 * w + x1] * ax) * ay;
}
