/**
 * Scoring a scan against synthetic ground truth.
 *
 *   walls     IoU, precision, recall, boundary F1 per matched wall;
 *             incorrect merges and splits; count
 *   corners   localisation error (px and fraction of width), false positives,
 *             false negatives, angle error
 *   geometry  plane normal error (deg)
 */
const deg = (r) => (r * 180) / Math.PI;

export function scoreScan(scene, result, { cornerTolFrac = 0.04, boundaryTolPx = 2 } = {}) {
  const { w, h, wallId } = scene;
  const n = w * h;
  const det = result.masks.visible;
  const gtWalls = scene.walls;
  const gtCount = gtWalls.map((_, k) => countWhere(wallId, (v) => v === k));

  // Overlap matrix gt x detected.
  const overlap = gtWalls.map(() => new Array(det.length).fill(0));
  const detCount = det.map(() => 0);
  for (let i = 0; i < n; i++) {
    const g = wallId[i];
    for (let k = 0; k < det.length; k++) {
      if (!det[k][i]) continue;
      detCount[k]++;
      if (g >= 0) overlap[g][k]++;
    }
  }
  // Greedy matching by IoU.
  const pairs = [];
  gtWalls.forEach((_, g) => det.forEach((__, k) => {
    const inter = overlap[g][k];
    const iou = inter / (gtCount[g] + detCount[k] - inter || 1);
    pairs.push({ g, k, iou, inter });
  }));
  pairs.sort((a, b) => b.iou - a.iou);
  const gUsed = new Set(); const kUsed = new Set(); const matches = [];
  for (const p of pairs) {
    if (p.iou < 0.3 || gUsed.has(p.g) || kUsed.has(p.k)) continue;
    gUsed.add(p.g); kUsed.add(p.k); matches.push(p);
  }
  // Only GT walls with a real visible share count towards the expectation.
  const visibleGt = gtWalls.map((_, g) => g).filter((g) => gtCount[g] > n * 0.01);

  const wallScores = matches.map(({ g, k, iou, inter }) => {
    const precision = inter / (detCount[k] || 1);
    const recall = inter / (gtCount[g] || 1);
    const gtMask = new Uint8Array(n); for (let i = 0; i < n; i++) gtMask[i] = wallId[i] === g ? 1 : 0;
    const bf1 = boundaryF1(gtMask, det[k], w, h, boundaryTolPx);
    const wl = result.walls[k];
    // The scanner's world frame follows the camera heading; rotate the true
    // normal by the camera's yaw into that frame before comparing.
    const yaw = ((scene.camera.yaw ?? 0) * Math.PI) / 180;
    const g0 = gtWalls[g].normal;
    const gn = [g0[0] * Math.cos(yaw) - g0[2] * Math.sin(yaw), 0, g0[0] * Math.sin(yaw) + g0[2] * Math.cos(yaw)];
    // Detected normal in world frame (X, Z); compare horizontally.
    const dn = [wl.plane.nx, wl.plane.nz];
    const normalErr = deg(Math.acos(Math.min(1, Math.abs(gn[0] * dn[0] + gn[2] * dn[1]) / Math.hypot(gn[0], gn[2]))));
    return { gt: gtWalls[g].id, det: wl.id, iou, precision, recall, boundaryF1: bf1, normalErrDeg: normalErr };
  });

  // Merges: a detected wall holding >25% of each of two GT walls.
  let merges = 0; let splits = 0;
  det.forEach((_, k) => {
    const big = gtWalls.filter((__, g) => overlap[g][k] > 0.25 * gtCount[g] && gtCount[g] > n * 0.01).length;
    if (big > 1) merges += big - 1;
  });
  gtWalls.forEach((_, g) => {
    if (gtCount[g] < n * 0.01) return;
    const big = det.filter((__, k) => overlap[g][k] > 0.2 * gtCount[g]).length;
    if (big > 1) splits += big - 1;
  });

  // Corners: compare the detected projected line against the true one, at
  // the rows where both are in frame.
  // A corner is expected when both its walls are visible and it is in frame.
  const visibleId = new Set(visibleGt.map((g) => gtWalls[g].id));
  const gtC = scene.corners.filter((c) => c.seg2D[0] && c.seg2D[1] && inFrame(c, w, h)
    && c.walls.every((id) => visibleId.has(id)));
  const detC = result.corners;
  const used = new Set();
  const cornerScores = [];
  let fn = 0;
  for (const c of gtC) {
    let best = null;
    detC.forEach((d, k) => {
      if (used.has(k)) return;
      const e = lineDistance(c.seg2D, d.segment2D, h);
      if (e !== null && (!best || e < best.err)) best = { k, err: e, d };
    });
    if (best && best.err <= cornerTolFrac * w) {
      used.add(best.k);
      cornerScores.push({
        gt: c.walls.join('|'),
        errPx: best.err,
        errNorm: best.err / w,
        angleErrDeg: c.parallel || best.d.angle === null ? null : Math.abs(angleOf(best.d) - c.angle),
        visible: best.d.visible,
        inferred: best.d.inferred,
      });
    } else fn++;
  }
  const fp = detC.length - used.size;

  const mean = (arr, key) => (arr.length ? arr.reduce((s, x) => s + x[key], 0) / arr.length : null);
  return {
    walls: { expected: visibleGt.length, detected: det.length, matched: matches.length, merges, splits },
    wallScores,
    meanIoU: mean(wallScores, 'iou'),
    meanBoundaryF1: mean(wallScores, 'boundaryF1'),
    maxNormalErrDeg: wallScores.length ? Math.max(...wallScores.map((s) => s.normalErrDeg)) : null,
    corners: { expected: gtC.length, detected: detC.length, falsePositives: fp, falseNegatives: fn },
    cornerScores,
    meanCornerErrPx: mean(cornerScores, 'errPx'),
  };
}

function angleOf(d) { return d.concave === false ? 360 - d.angle : d.angle; }

function inFrame(c, w, h) {
  const [a, b] = c.seg2D;
  for (let k = 0; k <= 10; k++) {
    const x = a[0] + ((b[0] - a[0]) * k) / 10; const y = a[1] + ((b[1] - a[1]) * k) / 10;
    if (x > 0.02 * w && x < 0.98 * w && y > 0 && y < h) return true;
  }
  return false;
}

/** Mean horizontal distance between two image segments over their shared rows in frame. */
function lineDistance(s1, s2, h) {
  const xAt = (s, y) => {
    const [a, b] = s;
    if (Math.abs(b[1] - a[1]) < 1e-6) return null;
    const t = (y - a[1]) / (b[1] - a[1]);
    return a[0] + (b[0] - a[0]) * t;
  };
  const yLo = Math.max(0, Math.min(s1[0][1], s1[1][1]), Math.min(s2[0][1], s2[1][1]));
  const yHi = Math.min(h, Math.max(s1[0][1], s1[1][1]), Math.max(s2[0][1], s2[1][1]));
  if (!(yHi > yLo)) return null;
  let sum = 0; let k = 0;
  for (let y = yLo; y <= yHi; y += (yHi - yLo) / 10 || 1) {
    const a = xAt(s1, y); const b = xAt(s2, y);
    if (a === null || b === null) continue;
    sum += Math.abs(a - b); k++;
  }
  return k ? sum / k : null;
}

function countWhere(arr, fn) { let c = 0; for (let i = 0; i < arr.length; i++) if (fn(arr[i])) c++; return c; }

function boundary(mask, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w]) out[i] = 1;
    }
  }
  return out;
}

/** Boundary F1 with a pixel tolerance (Perazzi et al.). */
export function boundaryF1(gt, pred, w, h, tol) {
  const bg = boundary(gt, w, h); const bp = boundary(pred, w, h);
  const near = (b, x, y) => {
    for (let dy = -tol; dy <= tol; dy++) {
      for (let dx = -tol; dx <= tol; dx++) {
        const xx = x + dx; const yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && b[yy * w + xx]) return true;
      }
    }
    return false;
  };
  let tp = 0; let np = 0; let tr = 0; let ng = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (bp[i]) { np++; if (near(bg, x, y)) tp++; }
      if (bg[i]) { ng++; if (near(bp, x, y)) tr++; }
    }
  }
  const P = np ? tp / np : 0; const R = ng ? tr / ng : 0;
  return P + R ? (2 * P * R) / (P + R) : 0;
}
