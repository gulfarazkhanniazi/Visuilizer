/**
 * From validated planes to finished walls: corners, extents, quads.
 *
 * A corner is where two wall planes intersect. For vertical walls that is a
 * vertical 3D line, found by solving two line equations in the ground plane --
 * geometry first. It is then projected into the photograph and checked
 * against independent evidence:
 *
 *   planeIntersection  does the observed boundary between the two walls' pixels
 *                      run along the projected line?
 *   depth              do two planes explain the depth either side of the line
 *                      better than one?
 *   normals            do surface normals change across it?
 *   lines              is there a straight image line (LSD) along it?
 *   rgb                is there a brightness step / edge along it?
 *   segmentation       is there wall on both sides (not a wall/object edge)?
 *   roomGeometry       plausible angle, and do the walls' visible extents
 *                      actually end near it?
 *
 * The evidence is combined with configurable weights; geometry carries most
 * of them, so an RGB edge on its own -- a window frame, a shadow -- can never
 * make a corner. RGB-only candidates (the old column detector) are still
 * evaluated, so the debug output can show WHY a window frame was rejected.
 *
 * A corner hidden behind furniture keeps its geometric position and is
 * reported as inferred, never as observed.
 */
import { CLS, projectWorld, toCamera, dirToWorld } from './pointcloud.js';
import {
  CORNERS, ASSIGN, PLANES, confidenceLabel,
} from './config.js';
import {
  clamp, deg, percentile, median, bilinear, round,
} from './math.js';
import { verticalCorners } from '../corners.js';
import { fitVertical, validateWall } from './planes.js';

const r3 = (v) => round(v, 3);

export function buildWalls(ctx, ex) {
  const {
    grid, W, cam, frame, ceilingY, roomHeight, input, Q,
  } = ctx;
  const { GW, GH } = grid;

  // --- per-wall geometry, corners, and geometry-aware merging ------------------
  // Two adjacent walls stay separate only if a corner between them is
  // validated. When the candidate is rejected on evidence and the planes are
  // close to parallel, the split was depth noise: merge, refit, re-evaluate.
  let planesIn = ex.walls.slice();
  let exLabel = ex.label;
  let walls; let label; let candidates;
  const merges = [];
  for (let pass = 0; pass < 6; pass++) {
    walls = planesIn.map((wl, k) => describeWall(ctx, wl, k, exLabel));
    walls.sort((a, b) => a.uCenter - b.uCenter);
    // Stable IDs: left to right across the photograph.
    walls.forEach((wl, k) => { wl.id = `wall_${String(k + 1).padStart(2, '0')}`; wl.index = k; });
    label = new Int16Array(GW * GH).fill(-1);
    walls.forEach((wl) => { for (const p of wl.pixels) label[p] = wl.index; });
    candidates = adjacentPairs(grid, label, walls).map(([i, j]) => evaluatePair(ctx, walls[i], walls[j], label));
    const mergeable = candidates
      .filter((c) => c.stacked || (c.rejected && c.evidence && c.normalAngle !== undefined && c.normalAngle < CORNERS.mergeWithoutCornerDeg))
      .sort((a, b) => a.score - b.score);
    if (!mergeable.length) break;
    // Only if one plane explains both: parallel planes a step apart (a
    // chimney breast) are two walls even when their corner is not proven.
    // Parallel planes stacked one above the other are one wall regardless:
    // no vertical line can separate them.
    const m = mergeable.find((c) => {
      if (c.stacked) return true;
      const px = c.wallA.pixels.concat(c.wallB.pixels);
      const u = relResidual(W, px, fitVertical(W, px));
      return u <= PLANES.mergeResidualGain * Math.max(
        relResidual(W, c.wallA.pixels, c.wallA.plane), relResidual(W, c.wallB.pixels, c.wallB.plane), PLANES.mergeResidualFloor,
      );
    });
    if (!m) break;
    const pixels = m.wallA.pixels.concat(m.wallB.pixels);
    const merged = { plane: fitVertical(W, pixels), pixels };
    const rest = walls.filter((wl) => wl !== m.wallA && wl !== m.wallB).map((wl) => ({ plane: wl.plane, pixels: wl.pixels, stats: wl.stats }));
    planesIn = [...rest, merged];
    exLabel = new Int16Array(GW * GH).fill(-1);
    planesIn.forEach((wl, k) => { for (const p of wl.pixels) exLabel[p] = k; });
    merged.stats = validateWall(grid, W, merged, exLabel);
    merges.push(`${m.wallA.id}+${m.wallB.id}: corner rejected (score ${m.score.toFixed(2)}), planes ${m.normalAngle.toFixed(1)} deg apart`);
  }
  if (merges.length) ctx.diag.mergedWithoutCorner = merges;
  // Grid label map in the final order; the raycast reads it as a prior.
  grid.label = label;

  const corners = [];
  const rejected = [];
  for (const c of candidates) {
    if (c.rejected) rejected.push(c);
    else corners.push(c);
  }
  // One corner per wall end: if two accepted corners claim the same end of a
  // wall, keep the stronger.
  corners.sort((a, b) => b.score - a.score);
  const endTaken = new Map();
  const kept = [];
  for (const c of corners) {
    const ka = `${c.wallA.index}:R`; const kb = `${c.wallB.index}:L`;
    if (endTaken.has(ka) || endTaken.has(kb)) {
      rejected.push({ ...c, rejected: true, reason: `a stronger corner already ends ${endTaken.has(ka) ? c.wallA.id : c.wallB.id} on that side` });
      continue;
    }
    endTaken.set(ka, c); endTaken.set(kb, c);
    kept.push(c);
  }
  kept.sort((a, b) => a.position2D[0] - b.position2D[0]);
  kept.forEach((c, k) => { c.id = `corner_${String(k + 1).padStart(2, '0')}`; });

  // RGB-only candidates: the old detector's columns, checked against the
  // planes. A column with the same wall on both sides is a frame, a shadow or
  // a panel edge, and is rejected with that reason on record.
  if (Q.corners === 'full' && input.luma && input.masks.wall) {
    for (const rc of rgbCandidates(ctx, label, walls, kept)) rejected.push(rc);
  }

  // --- extents: snap to corners, extend to the frame edge ---------------------
  for (const wl of walls) {
    const right = endTaken.get(`${wl.index}:R`);
    const left = endTaken.get(`${wl.index}:L`);
    wl.sMin = wl.sVis[0]; wl.sMax = wl.sVis[1];
    // Snapping may trim a wall back to its corner by the inside tolerance,
    // never remove most of what is visible: that would mean the corner is
    // not where the geometry says, and the extent keeps what was seen.
    const width = wl.sVis[1] - wl.sVis[0];
    const keeps = (a, b) => b - a >= 0.6 * width;
    if (left) wl.cornerLeft = left.id;
    if (right) wl.cornerRight = right.id;
    if (left && keeps(left.sB, wl.sMax)) wl.sMin = left.sB;
    else if (left) left.snapRefused = (left.snapRefused ?? []).concat(wl.id);
    else if (wl.touchesLeft) wl.sMin = Math.min(wl.sMin, frustumExtent(ctx, wl, -1));
    if (right && keeps(wl.sMin, right.sA)) wl.sMax = right.sA;
    else if (right) right.snapRefused = (right.snapRefused ?? []).concat(wl.id);
    else if (wl.touchesRight) wl.sMax = Math.max(wl.sMax, frustumExtent(ctx, wl, +1));
    wl.hiddenLength = r3(Math.max(0, wl.sVis[0] - wl.sMin) + Math.max(0, wl.sMax - wl.sVis[1]));
    finishWall(ctx, wl);
  }

  // Wall-floor / wall-ceiling boundaries: the plane intersection, validated
  // against where the segmentation says the boundary is.
  for (const wl of walls) {
    wl.boundaries = {
      floor: boundaryAgreement(ctx, wl, 0, CLS.FLOOR),
      ceiling: ceilingY ? boundaryAgreement(ctx, wl, wl.top, CLS.CEILING) : null,
    };
  }

  const floor = horizontalSurface(ctx, CLS.FLOOR, 0);
  const ceilingRect = ceilingY ? horizontalSurface(ctx, CLS.CEILING, ceilingY) : null;
  return {
    walls,
    corners: kept,
    rejected,
    floor,
    ceiling: ceilingY ? {
      plane: { a: 0, b: 1, c: 0, d: -r3(ceilingY) }, height: r3(ceilingY), quad: ceilingRect?.quad, realSize: ceilingRect?.realSize, polygon3D: ceilingRect?.polygon3D,
    } : null,
    label,
  };
}

/** Median point-to-plane residual as a fraction of distance. */
function relResidual(W, pixels, pl) {
  const r = [];
  const stride = pixels.length > 3000 ? Math.ceil(pixels.length / 3000) : 1;
  for (let k = 0; k < pixels.length; k += stride) {
    const p = pixels[k];
    r.push(Math.abs(pl.nx * W.X[p] + pl.nz * W.Z[p] - pl.o) / Math.max(0.3, Math.hypot(W.X[p], W.Z[p])));
  }
  return median(r);
}

/** Base description of one wall plane: orientation, visible extent, height. */
function describeWall(ctx, wl, k, exLabel) {
  const { grid, W, frame, cam, ceilingY, roomHeight } = ctx;
  const { GW } = grid;
  const pl = wl.plane;
  // Tangent along the wall, oriented so that s increases left to right in the
  // image: take the wall's median point, step along +t, see where it projects.
  const xs = []; const zs = []; const us = []; const ys = [];
  let touchesLeft = false; let touchesRight = false; let touchesTop = false;
  const colTop = new Map();
  for (const p of wl.pixels) {
    xs.push(W.X[p]); zs.push(W.Z[p]); ys.push(W.Y[p]);
    const gx = p % GW; const gy = (p / GW) | 0;
    us.push((gx + 0.5) * grid.sx);
    if (gx <= 1) touchesLeft = true;
    if (gx >= GW - 2) touchesRight = true;
    if (gy <= 1) touchesTop = true;
    if (!colTop.has(gx) || gy < colTop.get(gx)) colTop.set(gx, gy);
  }
  // What is directly above the wall's top edge, column by column: ceiling or
  // the frame edge (the wall runs to the ceiling), or another wall (this is a
  // panel or chimney breast standing in front of it, and stops short).
  let upCeil = 0; let upWall = 0;
  for (const [gx, gy] of colTop) {
    if (gy <= 1) { upCeil++; continue; }
    for (let d = 1; d <= 3 && gy - d >= 0; d++) {
      const q = (gy - d) * GW + gx;
      if (grid.cls[q] === CLS.CEILING) { upCeil++; break; }
      if (exLabel[q] >= 0 && exLabel[q] !== k) { upWall++; break; }
    }
  }
  const stopsBelowWall = upWall > 2 * upCeil && upWall > 5;
  const mX = median(xs); const mZ = median(zs);
  let t = [-pl.nz, pl.nx];
  const a = projectWorld(cam, frame, mX, 1, mZ);
  const b = projectWorld(cam, frame, mX + t[0] * 0.1, 1, mZ + t[1] * 0.1);
  if (a && b && b[0] < a[0]) t = [-t[0], -t[1]];
  const s = wl.pixels.map((p) => t[0] * W.X[p] + t[1] * W.Z[p]);
  const sVis = [percentile(s, 0.01), percentile(s, 0.99)];

  let top;
  let topSource;
  if (stopsBelowWall) { top = clamp(percentile(ys, 0.99), 0.5, ceilingY ?? 4.5); topSource = 'measured (another wall continues above it)'; }
  else if (ceilingY) { top = ceilingY; topSource = 'ceiling plane'; }
  else if (touchesTop) {
    // The wall leaves the top of the frame, so its height is unknown -- but
    // the part that IS visible must be inside its extent: take whichever is
    // higher of the room-height prior and where the plane exits the frame.
    top = Math.max(roomHeight, frameTopHeight(ctx, pl, t, [percentile(xs, 0.05), percentile(zs, 0.05)], [percentile(xs, 0.95), percentile(zs, 0.95)]));
    topSource = 'room-height prior, extended to the top of the frame (wall runs out of frame)';
  }
  else { top = clamp(percentile(ys, 0.99), 1.8, 4.5); topSource = 'measured'; }

  return {
    plane: pl,
    pixels: wl.pixels,
    stats: wl.stats,
    t,
    sVis,
    top,
    topSource,
    uCenter: median(us),
    uRange: [percentile(us, 0.01), percentile(us, 0.99)],
    touchesLeft,
    touchesRight,
    touchesTop,
    k,
  };
}

/** Height at which a wall's plane meets the top edge of the frame (max over its span). */
function frameTopHeight(ctx, pl, t, pA, pB) {
  const { cam, frame } = ctx;
  let best = 0;
  for (let k = 0; k <= 8; k++) {
    const X = pA[0] + ((pB[0] - pA[0]) * k) / 8; const Z = pA[1] + ((pB[1] - pA[1]) * k) / 8;
    const p = projectWorld(cam, frame, X, 0, Z);
    if (!p) continue;
    // Ray through (u, 0): intersect with this wall's plane, read its height.
    const d = dirToWorld(frame, [(p[0] - cam.cx) / cam.f, (0 - cam.cy) / cam.f, 1]);
    const den = pl.nx * d[0] + pl.nz * d[2];
    if (Math.abs(den) < 1e-9) continue;
    const tt = pl.o / den;
    if (tt > 0) best = Math.max(best, frame.camHeight + d[1] * tt);
  }
  return Math.min(best + 0.05, 6);
}

/**
 * Pairs of walls that could share a corner: neighbours in left-to-right
 * order, plus any two whose pixels touch in the image.
 */
function adjacentPairs(grid, label, walls) {
  const { GW, GH } = grid;
  const pairs = new Set();
  // Every pair close together in the image, not only consecutive ones: a
  // sliver plane between two walls must not hide the corner they share.
  const W = GW * grid.sx;
  // Neighbours always -- however far apart their visible parts, furniture
  // may hide what lies between -- plus any pair close together in the image.
  for (let k = 0; k + 1 < walls.length; k++) pairs.add(`${k},${k + 1}`);
  // Also any two walls with only slivers (small planes: a window reveal, a
  // panel's return) between them in left-to-right order.
  for (let a = 0; a < walls.length; a++) {
    for (let b = a + 1; b < walls.length; b++) {
      const small = 0.2 * Math.min(walls[a].pixels.length, walls[b].pixels.length);
      const between = walls.slice(a + 1, b);
      if (walls[b].uRange[0] - walls[a].uRange[1] < 0.15 * W || between.every((m) => m.pixels.length < small)) pairs.add(`${a},${b}`);
    }
  }
  const r = Math.max(2, Math.round(GW * 0.02));
  for (let y = 0; y < GH; y += 2) {
    for (let x = 0; x < GW - r; x++) {
      const a = label[y * GW + x];
      if (a < 0) continue;
      for (let d = 1; d <= r; d++) {
        const b = label[y * GW + x + d];
        if (b >= 0 && b !== a) { pairs.add(a < b ? `${a},${b}` : `${b},${a}`); break; }
      }
    }
  }
  return [...pairs].map((s) => s.split(',').map(Number));
}

/** Evaluate one candidate corner between walls A (left) and B (right). */
function evaluatePair(ctx, A, B, label) {
  const { cam, frame } = ctx;
  const base = { wallA: A, wallB: B };
  const cosN = A.plane.nx * B.plane.nx + A.plane.nz * B.plane.nz;
  const normalAngle = deg(Math.acos(clamp(cosN, -1, 1)));

  let xz; let type;
  if (normalAngle >= CORNERS.minAngleDeg) {
    // Intersection of the two base lines.
    const det = A.plane.nx * B.plane.nz - A.plane.nz * B.plane.nx;
    xz = [
      (A.plane.o * B.plane.nz - A.plane.nz * B.plane.o) / det,
      (A.plane.nx * B.plane.o - A.plane.o * B.plane.nx) / det,
    ];
    type = 'intersection';
  } else {
    // Parallel planes: a step (chimney breast, recess). There is no
    // intersection; the corner is where the boundary between their pixels is.
    const bnd = observedBoundary(ctx, A, B, label);
    if (!bnd) return reject(base, 'parallel planes with no shared boundary in the image');
    const wrong = stepWrongSide(ctx.grid, A, B, bnd.u);
    if (wrong > CORNERS.stepMaxWrongSide) {
      return { ...reject(base, `parallel planes stacked one above the other (${(100 * wrong).toFixed(0)}% of their pixels on the wrong side of the step), not side by side`), stacked: true, normalAngle };
    }
    xz = null;
    type = 'step';
    base.stepBoundaryU = bnd.u;
  }

  let sA; let sB;
  if (type === 'intersection') {
    const camZ = toCamera(frame, xz[0], 1, xz[1])[2];
    if (camZ < 0.2) return reject(base, 'plane intersection is behind the camera');
    sA = A.t[0] * xz[0] + A.t[1] * xz[1];
    sB = B.t[0] * xz[0] + B.t[1] * xz[1];
  } else {
    // Where the boundary's viewing ray meets each plane.
    sA = sAtImageU(ctx, A, base.stepBoundaryU);
    sB = sAtImageU(ctx, B, base.stepBoundaryU);
    if (sA === null || sB === null) return reject(base, 'step boundary does not meet both planes');
    xz = [A.t[0] * sA + A.plane.nx * A.plane.o, A.t[1] * sA + A.plane.nz * A.plane.o];
  }

  // Extent consistency: the corner must lie at A's right end and B's left
  // end -- past them (hidden) by a bounded amount, not deep inside either.
  const extA = sA - A.sVis[1];
  const extB = B.sVis[0] - sB;
  const widthA = A.sVis[1] - A.sVis[0]; const widthB = B.sVis[1] - B.sVis[0];
  const maxExtA = Math.max(CORNERS.maxExtension, CORNERS.maxExtensionFrac * widthA);
  const maxExtB = Math.max(CORNERS.maxExtension, CORNERS.maxExtensionFrac * widthB);
  const inside = Math.max(CORNERS.insideTolerance, 0.08 * Math.min(widthA, widthB));
  if (type === 'intersection') {
    if (extA < -inside) return reject(base, `plane intersection lies ${(-extA).toFixed(2)} m inside ${A.id}'s visible surface`);
    if (extB < -inside) return reject(base, `plane intersection lies ${(-extB).toFixed(2)} m inside ${B.id}'s visible surface`);
    if (extA > maxExtA || extB > maxExtB) return reject(base, `walls do not meet within ${Math.max(maxExtA, maxExtB).toFixed(1)} m of their visible ends`);
  }

  const top = Math.min(A.top, B.top);
  const p0 = projectWorld(cam, frame, xz[0], 0, xz[1]);
  const p1 = projectWorld(cam, frame, xz[0], top, xz[1]);
  if (!p0 || !p1) return reject(base, 'corner does not project into the image');

  const zCorner = toCamera(frame, xz[0], top / 2, xz[1])[2];
  let stepJumpRel = null;
  if (type === 'step') {
    // Depth of both planes along the viewing ray through the boundary.
    const tA = rayDepthAtU(ctx, A, base.stepBoundaryU); const tB = rayDepthAtU(ctx, B, base.stepBoundaryU);
    if (tA && tB) stepJumpRel = Math.abs(tA - tB) / Math.min(tA, tB);
  }
  const ev = cornerEvidence(ctx, A, B, label, p0, p1, type, {
    extA, extB, inside, maxExtA, maxExtB, normalAngle, zCorner, stepJumpRel,
  });
  const w = CORNERS.weights;
  let num = 0; let den = 0;
  for (const [k, v] of Object.entries(ev.scores)) {
    if (v === null || v === undefined) continue;
    num += w[k] * v; den += w[k];
  }
  const score = den ? num / den : 0;
  // Surface angle of the corner. Normals face the camera, so for a concave
  // (inside) corner the angle between the walls is 180 - angle(normals).
  const angle = type === 'step' ? 180 : 180 - normalAngle;
  const concave = type === 'step' ? null : isConcave(A, B);
  const visible = ev.coverage >= 0.3 && ev.hiddenFrac < CORNERS.occludedFrac;

  const out = {
    ...base,
    type,
    normalAngle,
    xz,
    sA,
    sB,
    score,
    angle: type === 'step' ? null : r3(concave ? angle : 360 - angle),
    concave,
    visible,
    inferred: !visible,
    evidence: Object.fromEntries(Object.entries(ev.scores).map(([k, v]) => [k, v === null ? null : r3(v)])),
    reason: ev.reason,
    coverage: r3(ev.coverage),
    occludedFrac: r3(ev.occludedFrac),
    segment2D: [p0.map(r3), p1.map(r3)],
    position2D: visibleMidpoint(ctx, p0, p1).map(r3),
    position3D: [r3(xz[0]), r3(top / 2), r3(xz[1])],
    segment3D: [[r3(xz[0]), 0, r3(xz[1])], [r3(xz[0]), r3(top), r3(xz[1])]],
    hiddenExtension: { [A.id]: r3(Math.max(0, extA)), [B.id]: r3(Math.max(0, extB)) },
  };
  if (score < CORNERS.accept) return { ...out, rejected: true, reason: { ...ev.reason, summary: `evidence score ${score.toFixed(2)} below ${CORNERS.accept}` } };
  // An observed corner must be seen by the geometry, not only the image: a
  // crease or step in depth, or a change of surface normal. Image edges,
  // segmentation and room priors can support a corner, never make one.
  const geoSupport = Math.max(ev.scores.depth ?? 0, ev.scores.normals ?? 0);
  if (visible && geoSupport < CORNERS.minGeometricSupport) {
    return { ...out, rejected: true, reason: { ...ev.reason, summary: `no geometric support (depth/normal evidence ${geoSupport.toFixed(2)} < ${CORNERS.minGeometricSupport})` } };
  }
  // An inferred corner rests on geometry alone: say so in its confidence.
  out.confidence = r3(visible ? score : score * 0.85);
  out.confidenceLabel = confidenceLabel(out.confidence);
  return out;
}

function reject(base, reason) {
  return { ...base, rejected: true, reason, score: 0 };
}

/** Is the corner between A and B an inside (concave) corner? */
function isConcave(A, B) {
  // Each wall's centre lies on the camera side of the other's plane.
  const cA = centre(A); const cB = centre(B);
  const sideA = B.plane.nx * cA[0] + B.plane.nz * cA[1] - B.plane.o;
  const sideB = A.plane.nx * cB[0] + A.plane.nz * cB[1] - A.plane.o;
  return sideA > 0 && sideB > 0;
}
function centre(wl) {
  const s = (wl.sVis[0] + wl.sVis[1]) / 2;
  return [wl.t[0] * s + wl.plane.nx * wl.plane.o, wl.t[1] * s + wl.plane.nz * wl.plane.o];
}

/** Camera depth at which the mid-height viewing ray through column u meets a wall's plane. */
function rayDepthAtU(ctx, wl, u) {
  const { cam, frame } = ctx;
  const d = dirToWorld(frame, [(u - cam.cx) / cam.f, 0, 1]);
  const den = wl.plane.nx * d[0] + wl.plane.nz * d[2];
  if (Math.abs(den) < 1e-9) return null;
  const tt = wl.plane.o / den;
  return tt > 0 ? tt : null;
}

/** s along a wall where the viewing ray through image column u (mid-height) meets it. */
function sAtImageU(ctx, wl, u) {
  const { cam, frame } = ctx;
  const v = cam.cy;
  const d = dirToWorld(frame, [(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1]);
  const den = wl.plane.nx * d[0] + wl.plane.nz * d[2];
  if (Math.abs(den) < 1e-9) return null;
  const tt = wl.plane.o / den;
  if (tt <= 0) return null;
  return wl.t[0] * d[0] * tt + wl.t[1] * d[2] * tt;
}

/**
 * Share of A's and B's pixels on the wrong side of a vertical step line at
 * image column u (A belongs left of it, B right), counted only in the rows the
 * other wall also occupies, so a header running above a recess does not count.
 */
function stepWrongSide(grid, A, B, u) {
  const { GW } = grid;
  const ub = u / grid.sx;
  const rows = (wl) => {
    let y0 = Infinity; let y1 = -Infinity;
    for (const p of wl.pixels) { const y = (p / GW) | 0; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return [y0, y1];
  };
  const [a0, a1] = rows(A); const [b0, b1] = rows(B);
  let wrong = 0; let n = 0;
  for (const p of B.pixels) {
    const y = (p / GW) | 0;
    if (y < a0 || y > a1) continue;
    n++; if ((p % GW) + 0.5 < ub) wrong++;
  }
  for (const p of A.pixels) {
    const y = (p / GW) | 0;
    if (y < b0 || y > b1) continue;
    n++; if ((p % GW) + 0.5 > ub) wrong++;
  }
  return n ? wrong / n : 0;
}

/** Mean image column of the boundary between A's and B's pixels. */
function observedBoundary(ctx, A, B, label) {
  const { grid } = ctx;
  const { GW, GH } = grid;
  const xs = [];
  const reach = Math.max(3, Math.round(GW * 0.03));
  for (let y = 0; y < GH; y++) {
    for (let x = 0; x < GW - 1; x++) {
      if (label[y * GW + x] !== A.index) continue;
      for (let d = 1; d <= reach && x + d < GW; d++) {
        const l = label[y * GW + x + d];
        if (l === B.index) { xs.push(x + d / 2); break; }
        if (l === A.index) break;
      }
    }
  }
  if (xs.length < 5) return null;
  return { u: (median(xs) + 0.5) * grid.sx, n: xs.length };
}

/** The point of the projected corner segment to report: middle of its in-frame part. */
function visibleMidpoint(ctx, p0, p1) {
  const { w, h } = ctx.input;
  const pts = [];
  for (let k = 0; k <= 20; k++) {
    const x = p0[0] + ((p1[0] - p0[0]) * k) / 20;
    const y = p0[1] + ((p1[1] - p0[1]) * k) / 20;
    if (x >= 0 && y >= 0 && x < w && y < h) pts.push([x, y]);
  }
  if (!pts.length) return [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
  return pts[Math.floor(pts.length / 2)];
}

/**
 * All the evidence for one projected corner line.
 */
function cornerEvidence(ctx, A, B, label, p0, p1, type, geo) {
  const {
    grid, W, input, edges,
  } = ctx;
  const { GW, GH } = grid;
  const { w, h } = input;
  const K = CORNERS.samples;
  const bw = Math.max(2, Math.round(CORNERS.band * GW));
  const inner = Math.max(2, Math.round(CORNERS.depthBandInner * GW));
  const outer = Math.max(inner + 3, Math.round(CORNERS.depthBandOuter * GW));
  const sigma = Math.max(1, CORNERS.boundarySigma * GW);
  // Segment in grid coordinates, perpendicular pointing from A towards B.
  const g0 = [p0[0] / grid.sx - 0.5, p0[1] / grid.sy - 0.5];
  const g1 = [p1[0] / grid.sx - 0.5, p1[1] / grid.sy - 0.5];
  const dx = g1[0] - g0[0]; const dy = g1[1] - g0[1];
  const len = Math.hypot(dx, dy) || 1;
  let perp = [-dy / len, dx / len];
  if (perp[0] < 0) perp = [-perp[0], -perp[1]];
  const at = (x, y) => {
    const xi = Math.round(x); const yi = Math.round(y);
    return xi >= 0 && yi >= 0 && xi < GW && yi < GH ? yi * GW + xi : -1;
  };

  let observed = 0; let boundaryScore = 0; let inFrame = 0; let occluded = 0; let bothWall = 0;
  let hiddenByNearer = 0; let visibleN = 0;
  const sideA = []; const sideB = []; let jump = 0; let jumpN = 0;
  // A sample is hidden when something that is not wall stands in front of
  // the corner there. Hidden samples say nothing about the corner, so they
  // are excluded from the image evidence rather than counted against it.
  const hiddenAt = (f) => {
    const c = at(g0[0] + dx * f, g0[1] + dy * f);
    return c >= 0 && grid.cls[c] !== CLS.WALL && grid.Z[c] < geo.zCorner * (1 - 0.06);
  };
  for (let k = 0; k < K; k++) {
    const f = (k + 0.5) / K;
    const cx = g0[0] + dx * f; const cy = g0[1] + dy * f;
    const c = at(cx, cy);
    if (c < 0) continue;
    inFrame++;
    const cls = grid.cls[c];
    if (cls !== CLS.WALL) occluded++;
    if (hiddenAt(f)) { hiddenByNearer++; continue; }
    visibleN++;
    // Boundary: last A pixel before first B pixel along the perpendicular.
    let lastA = null; let firstB = null;
    for (let s = -3 * bw; s <= 3 * bw; s++) {
      const i = at(cx + perp[0] * s, cy + perp[1] * s);
      if (i < 0) continue;
      if (label[i] === A.index && firstB === null) lastA = s;
      if (label[i] === B.index && lastA !== null && firstB === null) firstB = s;
    }
    if (lastA !== null && firstB !== null) {
      observed++;
      const d = (lastA + firstB) / 2;
      boundaryScore += Math.exp(-((d / sigma) ** 2));
    }
    const ia = at(cx - perp[0] * bw, cy - perp[1] * bw);
    const ib = at(cx + perp[0] * bw, cy + perp[1] * bw);
    const isWallish = (i) => i >= 0 && (grid.cls[i] === CLS.WALL || grid.cls[i] === CLS.WALL_OBJECT);
    if (isWallish(ia) && isWallish(ib)) bothWall++;
    // Depth/normal samples skip the transition zone right at the line --
    // monocular depth smears a step or crease over several pixels there --
    // and read the two surfaces a little further out on each side.
    for (let s = inner; s <= outer; s++) {
      const a = at(cx - perp[0] * s, cy - perp[1] * s);
      const b = at(cx + perp[0] * s, cy + perp[1] * s);
      if (a >= 0 && grid.cls[a] === CLS.WALL && W.valid[a]) sideA.push(a);
      if (b >= 0 && grid.cls[b] === CLS.WALL && W.valid[b]) sideB.push(b);
    }
    jump += edges.jump[c]; jumpN++;
  }
  const coverage = inFrame ? observed / inFrame : 0;
  const occludedFrac = inFrame ? occluded / inFrame : 1;

  const scores = {};
  const reason = {};
  // Plane intersection vs observed boundary.
  scores.planeIntersection = observed >= 3 ? boundaryScore / observed : null;
  reason.planeIntersection = scores.planeIntersection === null ? 'boundary not observed (hidden or out of frame)' : scores.planeIntersection > 0.5;

  // Depth: two planes vs one across the line.
  if (sideA.length >= 8 && sideB.length >= 8) {
    const relRes = (pl, idx) => idx.reduce((s, i) => s + Math.abs(pl.nx * W.X[i] + pl.nz * W.Z[i] - pl.o) / Math.max(0.3, Math.hypot(W.X[i], W.Z[i])), 0) / idx.length;
    const two = (relRes(A.plane, sideA) * sideA.length + relRes(B.plane, sideB) * sideB.length) / (sideA.length + sideB.length);
    const one = relRes(fitVertical(W, sideA.concat(sideB)), sideA.concat(sideB));
    scores.depth = one > 1e-4 ? clamp((one - two) / one, 0, 1) : 0;
    if (type === 'step') {
      // Parallel planes: a single tilted plane passes close to two short
      // offset strips, so two-vs-one is weak evidence. The step itself is:
      // the depth jump the two planes predict along the boundary ray, which
      // the observed depth either side must confirm.
      const jumpRel = geo.stepJumpRel ?? 0;
      const zA = median(sideA.map((i) => grid.Z[i])); const zB = median(sideB.map((i) => grid.Z[i]));
      const observedRel = Math.abs(zA - zB) / Math.max(1e-3, Math.min(zA, zB));
      const confirmed = observedRel >= 0.5 * jumpRel ? 1 : observedRel / Math.max(1e-6, 0.5 * jumpRel);
      scores.depth = Math.max(scores.depth, clamp((jumpRel - 0.01) / 0.03, 0, 1) * confirmed);
    }
  } else {
    scores.depth = null;
  }
  reason.depthDiscontinuity = scores.depth === null ? 'too little visible wall beside the line' : scores.depth > 0.4;

  // Normals: angle between the mean normals either side.
  const meanN = (idx) => {
    let x = 0; let z = 0; let n = 0;
    for (const i of idx) if (W.ok[i]) { x += W.nX[i]; z += W.nZ[i]; n++; }
    return n >= 5 ? [x / n, z / n] : null;
  };
  const nA = meanN(sideA); const nB = meanN(sideB);
  if (nA && nB) {
    const c = (nA[0] * nB[0] + nA[1] * nB[1]) / ((Math.hypot(...nA) * Math.hypot(...nB)) || 1);
    const ang = deg(Math.acos(clamp(c, -1, 1)));
    scores.normals = type === 'step' ? null : clamp((ang - 4) / 20, 0, 1);
    reason.normalChange = scores.normals === null ? 'parallel planes (step)' : +ang.toFixed(1);
  } else {
    scores.normals = null;
    reason.normalChange = 'too few reliable normals beside the line';
  }

  // Image evidence at refinement resolution.
  const img = visibleN >= 3 ? imageEvidence(ctx, p0, p1, (f) => !hiddenAt(f)) : { rgb: null, lines: null };
  scores.rgb = img.rgb;
  scores.lines = img.lines;
  reason.rgbEdge = img.rgb === null ? 'not in frame' : +img.rgb.toFixed(2);
  reason.lineSupport = img.lines === null ? 'no line data' : img.lines > 0.3;

  scores.segmentation = visibleN >= 3 ? bothWall / visibleN : null;
  reason.segmentationBoundary = scores.segmentation === null ? 'hidden or not in frame' : scores.segmentation > 0.5;

  // Occlusion: for a corner that is not observed, is that because something
  // nearer stands in front of it along its whole visible height? If not, the
  // corner should have been seen, and its absence is evidence against it.
  const hiddenFrac = inFrame ? hiddenByNearer / inFrame : 0;
  scores.occlusion = coverage < 0.3 && inFrame ? hiddenByNearer / Math.max(1, inFrame - observed) : null;
  reason.occlusion = scores.occlusion === null ? 'corner observed' : +scores.occlusion.toFixed(2);

  // Room geometry: a mild Manhattan prior on the angle, and how cleanly the
  // walls' visible ends meet the corner.
  const a = type === 'step' ? 90 : 180 - geo.normalAngle;
  const anglePrior = type === 'step' ? 0.8 : Math.abs(a - 90) <= 10 ? 1 : Math.abs(a - 90) <= 50 ? 1 - 0.4 * ((Math.abs(a - 90) - 10) / 40) : 0.4;
  // Extending a wall to reach its corner costs nothing when the corner is
  // hidden (that is exactly where walls continue unseen), and a little
  // otherwise; running past a visible end into the other wall costs a lot.
  const pen = (e, max) => (e < 0 ? clamp(1 + e / geo.inside, 0, 1)
    : hiddenFrac > 0.5 ? 1 : clamp(1 - 0.5 * (e / max), 0.3, 1));
  const ends = type === 'step' ? 1 : pen(geo.extA, geo.maxExtA) * pen(geo.extB, geo.maxExtB);
  scores.roomGeometry = anglePrior * ends;
  reason.roomGeometry = { anglePrior: +anglePrior.toFixed(2), endsMeet: +ends.toFixed(2) };

  return { scores, reason, coverage, occludedFrac, hiddenFrac };
}

/** Brightness step / edge strength and LSD line support along a segment. */
function imageEvidence(ctx, p0, p1, visibleAt = () => true) {
  const { input } = ctx;
  const { luma, RW, RH, w, h } = input;
  if (!luma) return { rgb: null, lines: null };
  const sx = RW / w; const sy = RH / h;
  const a = [p0[0] * sx, p0[1] * sy]; const b = [p1[0] * sx, p1[1] * sy];
  const dx = b[0] - a[0]; const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const perp = [-dy / len, dx / len];
  const K = 64;
  let step = 0; let edge = 0; let n = 0; let coherent = 0;
  for (let k = 0; k < K; k++) {
    const f = (k + 0.5) / K;
    const x = a[0] + dx * f; const y = a[1] + dy * f;
    if (x < 3 || y < 3 || x > RW - 4 || y > RH - 4 || !visibleAt(f)) continue;
    n++;
    let L = 0; let R = 0;
    for (let d = 2; d <= 6; d++) {
      L += bilinear(luma, RW, RH, x - perp[0] * d, y - perp[1] * d);
      R += bilinear(luma, RW, RH, x + perp[0] * d, y + perp[1] * d);
    }
    step += Math.abs(L - R) / 5;
    let best = 0;
    for (let d = -2; d <= 2; d++) {
      const g = Math.abs(bilinear(luma, RW, RH, x + perp[0] * (d + 1), y + perp[1] * (d + 1))
        - bilinear(luma, RW, RH, x + perp[0] * (d - 1), y + perp[1] * (d - 1))) / 2;
      if (g > best) best = g;
    }
    edge += best;
    if (best > 0.02) coherent++;
  }
  if (!n) return { rgb: null, lines: null };
  const rgb = clamp(0.5 * (step / n) / 0.05 + 0.5 * (edge / n) / 0.04, 0, 1);

  // LSD line support: share of the segment covered by a near-collinear line.
  let lines = null;
  const segs = input.depth?.lines;
  if (segs?.length) {
    const L = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) || 1;
    const u = [(p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L];
    const tol = Math.max(3, 0.008 * w);
    const covered = [];
    for (const s of segs) {
      const sdx = s[2] - s[0]; const sdy = s[3] - s[1];
      const sl = Math.hypot(sdx, sdy) || 1;
      if (Math.abs(sdx * u[0] + sdy * u[1]) / sl < Math.cos((4 * Math.PI) / 180)) continue;
      const d1 = Math.abs((s[0] - p0[0]) * -u[1] + (s[1] - p0[1]) * u[0]);
      const d2 = Math.abs((s[2] - p0[0]) * -u[1] + (s[3] - p0[1]) * u[0]);
      if (d1 > tol || d2 > tol) continue;
      const t1 = (s[0] - p0[0]) * u[0] + (s[1] - p0[1]) * u[1];
      const t2 = (s[2] - p0[0]) * u[0] + (s[3] - p0[1]) * u[1];
      covered.push([Math.max(0, Math.min(t1, t2)), Math.min(L, Math.max(t1, t2))]);
    }
    covered.sort((p, q) => p[0] - q[0]);
    let tot = 0; let end = -Infinity;
    for (const [s0, s1] of covered) {
      if (s1 <= end) continue;
      tot += s1 - Math.max(s0, end); end = s1;
    }
    // Relative to the in-frame part of the segment.
    lines = clamp(tot / (L * clamp(n / K, 0.05, 1)), 0, 1);
  } else {
    lines = coherent / n;
  }
  return { rgb, lines };
}

/**
 * The old column detector's candidates, judged by the planes. It found
 * strong vertical edges; the planes say whether a wall actually changes there.
 */
function rgbCandidates(ctx, label, walls, accepted) {
  const { input, grid } = ctx;
  const { RW, RH, w } = input;
  const mask = new Uint8Array(RW * RH);
  for (let i = 0; i < mask.length; i++) mask[i] = input.masks.wall[i] ? 1 : 0;
  let cols;
  try { cols = verticalCorners(input.luma, mask, RW, RH); } catch { return []; }
  const out = [];
  for (const c of cols) {
    const u = (c.u + 0.5) * (w / RW);
    const near = accepted.find((k) => Math.abs(k.position2D[0] - u) < 0.03 * w);
    if (near) {
      near.rgbColumnSupport = true;
      continue;
    }
    const gx = Math.round(u / grid.sx - 0.5);
    const d = Math.max(2, Math.round(grid.GW * 0.02));
    const tally = (x0, x1) => {
      const m = new Map();
      for (let y = 0; y < grid.GH; y++) {
        for (let x = Math.max(0, x0); x <= Math.min(grid.GW - 1, x1); x++) {
          const l = label[y * grid.GW + x];
          if (l >= 0) m.set(l, (m.get(l) ?? 0) + 1);
        }
      }
      return [...m.entries()].sort((p, q) => q[1] - p[1])[0]?.[0] ?? -1;
    };
    const left = tally(gx - 3 * d, gx - 1); const right = tally(gx + 1, gx + 3 * d);
    let reason;
    if (left >= 0 && left === right) reason = `No supporting plane intersection: ${walls[left].id} continues on both sides (frame, panel edge, shadow or object edge)`;
    else if (left < 0 || right < 0) reason = 'No wall plane on one side of the edge';
    else reason = `Different walls either side, but their planes do not meet here (nearest validated corner elsewhere)`;
    out.push({
      source: 'rgb-column',
      rejected: true,
      reason,
      position2D: [r3(u), r3(input.h / 2)],
      evidence: { rgb: r3(Math.min(1, c.contrast / 0.05)), continuity: r3(c.continuity) },
    });
  }
  return out;
}

/** Furthest s at which the wall is still inside the horizontal field of view. */
function frustumExtent(ctx, wl, dirSign) {
  const { cam, frame } = ctx;
  const u = dirSign < 0 ? -0.02 * cam.w : 1.02 * cam.w;
  const s = sAtImageU(ctx, wl, u);
  if (s === null) return dirSign < 0 ? wl.sVis[0] - 0.5 : wl.sVis[1] + 0.5;
  // Never extend more than a few metres past what is visible.
  return dirSign < 0 ? Math.max(s, wl.sVis[0] - 4) : Math.min(s, wl.sVis[1] + 4);
}

/** Quad, real size, plane equations and 3D polygon for a wall with final extents. */
function finishWall(ctx, wl) {
  const { cam, frame } = ctx;
  const pt = (s) => [wl.t[0] * s + wl.plane.nx * wl.plane.o, wl.t[1] * s + wl.plane.nz * wl.plane.o];
  // A side wall can run past the camera; keep the quad's corners in front of
  // it, or the homography the renderer builds from them degenerates.
  const zOk = (s) => {
    const [X, Z] = pt(s);
    return toCamera(frame, X, 0, Z)[2] >= ASSIGN.minWallZ && toCamera(frame, X, wl.top, Z)[2] >= ASSIGN.minWallZ;
  };
  const mid = (wl.sVis[0] + wl.sVis[1]) / 2;
  const pull = (s) => {
    if (zOk(s)) return s;
    let lo = mid; let hi = s;
    for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; if (zOk(m)) lo = m; else hi = m; }
    return lo;
  };
  wl.sMin = pull(wl.sMin); wl.sMax = pull(wl.sMax);
  const [x0, z0] = pt(wl.sMin); const [x1, z1] = pt(wl.sMax);
  const corners3D = [[x0, wl.top, z0], [x1, wl.top, z1], [x1, 0, z1], [x0, 0, z0]];
  const quad = corners3D.map(([X, Y, Z]) => projectWorld(cam, frame, X, Y, Z));
  wl.quad = quad.every(Boolean) ? quad.map(([x, y]) => [round(x, 1), round(y, 1)]) : null;
  wl.polygon3D = corners3D.map((p) => p.map(r3));
  wl.realSize = { w: round(wl.sMax - wl.sMin, 2), h: round(wl.top, 2) };
  // Plane in the world frame: nx X + 0 Y + nz Z - o = 0; in the camera frame
  // its normal is nx*xhat + nz*fwd.
  wl.planeWorld = { a: r3(wl.plane.nx), b: 0, c: r3(wl.plane.nz), d: r3(-wl.plane.o) };
  const nc = [0, 1, 2].map((k) => wl.plane.nx * frame.xhat[k] + wl.plane.nz * frame.fwd[k]);
  wl.planeCamera = { a: r3(nc[0]), b: r3(nc[1]), c: r3(nc[2]), d: r3(-wl.plane.o) };
  wl.normal = [r3(wl.plane.nx), 0, r3(wl.plane.nz)];
  // Legacy-compatible description (base line in XZ), used by nameWall.
  wl.dir = wl.t;
  wl.origin = pt(0);
}

/**
 * How well the projected wall/floor (or wall/ceiling) line agrees with the
 * segmentation's boundary, where that boundary is visible.
 */
function boundaryAgreement(ctx, wl, Y, otherCls) {
  const { grid, cam, frame } = ctx;
  const pt = (s) => [wl.t[0] * s + wl.plane.nx * wl.plane.o, wl.t[1] * s + wl.plane.nz * wl.plane.o];
  const dists = []; let samples = 0;
  const sgn = otherCls === CLS.FLOOR ? 1 : -1;
  for (let k = 0; k <= 40; k++) {
    const s = wl.sMin + ((wl.sMax - wl.sMin) * k) / 40;
    const [X, Z] = pt(s);
    const p = projectWorld(cam, frame, X, Y, Z);
    if (!p) continue;
    const gx = Math.round(p[0] / grid.sx - 0.5); const gy = p[1] / grid.sy - 0.5;
    if (gx < 0 || gx >= grid.GW || gy < 0 || gy >= grid.GH) continue;
    samples++;
    // Search the column for a wall pixel directly adjoining the other class.
    let best = null;
    const reach = Math.round(grid.GH * 0.08);
    for (let d = -reach; d <= reach; d++) {
      const y = Math.round(gy + d);
      const y2 = y + sgn;
      if (y < 0 || y2 < 0 || y >= grid.GH || y2 >= grid.GH) continue;
      if (grid.cls[y * grid.GW + gx] === CLS.WALL && grid.cls[y2 * grid.GW + gx] === otherCls) {
        if (best === null || Math.abs(d) < Math.abs(best)) best = d;
      }
    }
    if (best !== null) dists.push(Math.abs(best) * grid.sy);
  }
  if (!samples) return null;
  return {
    observedFraction: r3(dists.length / samples),
    medianErrorPx: dists.length ? r3(median(dists)) : null,
    medianErrorNorm: dists.length ? r3(median(dists) / cam.w) : null,
    source: 'plane intersection, validated against segmentation',
  };
}

/**
 * Floor (Y = 0) or ceiling (Y = ceiling height) rectangle on its plane,
 * covering what is visible of it -- built on the same plane the walls meet,
 * so floor and walls share the skirting line exactly.
 */
function horizontalSurface(ctx, cls, Y) {
  const { grid, W, cam, frame } = ctx;
  const xs = []; const zs = [];
  for (let i = 0; i < grid.cls.length; i++) {
    if (grid.cls[i] !== cls || !W.valid[i]) continue;
    xs.push(W.X[i]); zs.push(W.Z[i]);
  }
  if (xs.length < 50) return null;
  const x0 = percentile(xs, 0.02); const x1 = percentile(xs, 0.98);
  let zNear = Math.max(0.5, percentile(zs, 0.02)); const zFar = Math.min(12, percentile(zs, 0.98));
  // Near edge in front of the camera by a margin, far edge clear of it.
  for (let k = 0; k < 20 && toCamera(frame, x0, Y, zNear)[2] < 0.4; k++) zNear += 0.1;
  if (!(zFar - zNear > 0.3)) return null;
  const corners = [[x0, Y, zFar], [x1, Y, zFar], [x1, Y, zNear], [x0, Y, zNear]];
  const quad = corners.map(([X, Y, Z]) => projectWorld(cam, frame, X, Y, Z));
  if (!quad.every(Boolean)) return null;
  return {
    quad: quad.map(([x, y]) => [round(x, 1), round(y, 1)]),
    realSize: { w: round(x1 - x0, 2), h: round(zFar - zNear, 2) },
    plane: { a: 0, b: 1, c: 0, d: -r3(Y) },
    polygon3D: corners.map((p) => p.map(r3)),
  };
}
