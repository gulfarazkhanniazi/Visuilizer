/**
 * Glue between the photograph and the pure geometry: fetch depth, run
 * scanGeometry, write debug images, and shape the public result.
 *
 * Every failure is caught and returned as { ok: false, stage, reason } so the
 * caller can fall back to the junction scanner; nothing here throws into the
 * request.
 */
import path from 'node:path';
import { estimateDepth } from './depth.js';
import { scanGeometry, GEOMETRY_VERSION } from './index.js';
import { QUALITY, confidenceLabel, CONFIDENCE_BANDS } from './config.js';
import { DebugWriter, PALETTE } from './debug.js';
import { refineOccluders } from './sam.js';

const r3 = (v) => Math.round(v * 1000) / 1000;

export async function runGeometryScan(imagePath, seg, {
  quality = 'balanced', roomHeight = 2.7, hfov, hfovSource, debugDir = null, levelCam = null, depthModel,
} = {}) {
  const timings = {};
  const q = QUALITY[quality] ? quality : 'balanced';
  let t = Date.now();
  const why = {};
  let depth;
  try {
    depth = await estimateDepth(imagePath, { quality: q, lines: QUALITY[q].lines, depthModel }, why);
  } catch (e) {
    return { ok: false, stage: 'depth', reason: `depth failure: ${e.message}` };
  }
  timings.depth = Date.now() - t;
  if (!depth) return { ok: false, stage: 'depth', reason: `no depth source available (${(why.depthErrors ?? []).join('; ')})` };

  const dbg = debugDir ? new DebugWriter(debugDir, { photo: imagePath, w: seg.w, h: seg.h }) : null;
  if (dbg) await dbg.original();

  // quality=high: SAM 2 sharpens furniture/wall boundaries before geometry.
  let sam = null;
  if (QUALITY[q].sam && seg.crisp?.wall) {
    t = Date.now();
    try { sam = await refineOccluders(imagePath, seg); } catch (e) { sam = { applied: false, reason: e.message }; }
    timings.sam = Date.now() - t;
  }

  t = Date.now();
  let geo;
  try {
    geo = scanGeometry({
      w: seg.w,
      h: seg.h,
      RW: seg.RW,
      RH: seg.RH,
      masks: seg.crisp ?? {},
      objects: seg.objects ?? {},
      luma: seg.guide,
      depth,
      levelCam,
    }, { quality: q, roomHeight, hfov, hfovSource, debugWriter: dbg });
  } catch (e) {
    return { ok: false, stage: 'geometry', reason: `geometry failure: ${e.message}`, stack: e.stack, depth: depthMeta(depth) };
  }
  timings.geometry = Date.now() - t;
  Object.assign(timings, prefix('geometry.', geo.timings));
  if (depth.serviceTimings) Object.assign(timings, prefix('cv.', depth.serviceTimings));

  if (!geo.ok) {
    if (dbg) await dbg.flush();
    return { ok: false, stage: 'geometry', reason: geo.reason, depth: depthMeta(depth), diag: geo.diag, timings };
  }
  if (sam) geo.diag.sam = sam;
  if (dbg) await writeDebug(dbg, geo, depth, seg);
  return {
    ok: true, geo, depth: depthMeta(depth), timings, sam, debugFiles: dbg ? await dbg.flush() : null,
  };
}

function prefix(p, o) {
  return Object.fromEntries(Object.entries(o ?? {}).map(([k, v]) => [p + k, v]));
}

function depthMeta(d) {
  return {
    source: d.source, metric: d.metric, width: d.width, height: d.height, device: d.device, cached: d.cached, modelVersions: d.modelVersions,
  };
}

/** Public, JSON-safe description of one wall. */
export function publicWall(wl, geo, extra = {}) {
  const counts = geo.masks.counts[wl.index];
  const st = wl.stats;
  const corners = [wl.cornerLeft, wl.cornerRight].filter(Boolean);
  // Confidence: the plane's own, tempered by how much of the region the
  // raycast could actually give it.
  const conf = r3(st.confidence);
  return {
    id: wl.id,
    plane: wl.planeWorld,
    planeCamera: wl.planeCamera,
    normal: wl.normal,
    polygon3D: wl.polygon3D,
    quad: wl.quad,
    realSize: wl.realSize,
    corners,
    confidence: conf,
    confidenceLabel: confidenceLabel(conf),
    visiblePixels: counts.visible,
    occludedPixels: counts.occluded,
    hiddenLength: wl.hiddenLength,
    height: { value: r3(wl.top), source: wl.topSource },
    boundaries: wl.boundaries,
    validation: {
      points: st.points,
      meanRelResidual: r3(st.meanRelResidual),
      medianRelResidual: r3(st.medianRelResidual),
      p95RelResidual: r3(st.p95RelResidual),
      inlierRatio: r3(st.inlierRatio),
      normalConsistency: r3(st.normalConsistency),
      verticalityDeg: st.verticalityDeg === null ? null : r3(st.verticalityDeg),
      segmentationConsistency: r3(st.segmentationConsistency),
    },
    ...extra,
  };
}

export function publicCorner(c) {
  return {
    id: c.id,
    wallA: c.wallA.id,
    wallB: c.wallB.id,
    type: c.type,
    position2D: c.position2D,
    position3D: c.position3D,
    segment2D: c.segment2D,
    segment3D: c.segment3D,
    angle: c.angle,
    concave: c.concave,
    visible: c.visible,
    inferred: c.inferred,
    confidence: c.confidence,
    confidenceLabel: c.confidenceLabel,
    score: r3(c.score),
    evidence: c.evidence,
    reason: c.reason,
    hiddenExtension: c.hiddenExtension,
    rgbColumnSupport: !!c.rgbColumnSupport,
  };
}

export function publicRejected(c) {
  return {
    rejected: true,
    source: c.source ?? 'plane-pair',
    wallA: c.wallA?.id ?? null,
    wallB: c.wallB?.id ?? null,
    reason: typeof c.reason === 'string' ? c.reason : c.reason?.summary,
    details: typeof c.reason === 'object' ? c.reason : undefined,
    score: c.score !== undefined ? r3(c.score) : undefined,
    evidence: c.evidence,
    position2D: c.position2D,
    segment2D: c.segment2D,
  };
}

export { CONFIDENCE_BANDS, GEOMETRY_VERSION };

/* ------------------------------------------------------------ debug -------- */

async function writeDebug(dbg, geo, depth, seg) {
  const {
    grid, walls, corners, rejectedCorners, masks,
  } = geo;
  const col = (k) => `rgb(${PALETTE[k % PALETTE.length].join(',')})`;
  const label = new Int16Array(grid.GW * grid.GH).fill(-1);
  walls.forEach((wl) => { for (const p of wl.pixels) label[p] = wl.index; });

  // 06: top-down cloud with final extents and corners.
  const ext = walls.map((wl) => ({
    id: wl.id,
    p0: [wl.t[0] * wl.sVis[0] + wl.plane.nx * wl.plane.o, wl.t[1] * wl.sVis[0] + wl.plane.nz * wl.plane.o],
    p1: [wl.t[0] * wl.sVis[1] + wl.plane.nx * wl.plane.o, wl.t[1] * wl.sVis[1] + wl.plane.nz * wl.plane.o],
  }));
  dbg.pointCloud(grid, geo.worldCloud, label, walls, { extents: ext, corners: [] });
  await dbg.labels(grid, label, '07-plane-detection.png');

  // 08: line segments and vanishing points.
  const vps = geo.diag.intrinsics?.vanishingPoints?.vps ?? [];
  await dbg.overlay('08-line-detection.png', (s) => {
    const parts = (depth.lines ?? []).map((L) => `<line x1="${L[0] * s}" y1="${L[1] * s}" x2="${L[2] * s}" y2="${L[3] * s}" stroke="#ffee00" stroke-width="1.5"/>`);
    for (const v of vps) {
      if (!v.point) continue;
      parts.push(`<circle cx="${v.point[0] * s}" cy="${v.point[1] * s}" r="7" fill="none" stroke="${v.kind === 'vertical' ? '#00e5ff' : '#ff4d4d'}" stroke-width="3"/>`);
    }
    parts.push(`<text x="8" y="20" fill="#fff" font-size="15" font-family="sans-serif" stroke="#000" stroke-width="3" paint-order="stroke">${(depth.lines ?? []).length} segments; hfov ${geo.camera.hfov} (${geo.camera.focalSource})</text>`);
    return parts.join('');
  });

  // 09: every candidate, accepted or rejected, with its score.
  const text = (x, y, s, str, fill) => `<text x="${x}" y="${y}" fill="${fill}" font-size="13" font-family="sans-serif" stroke="#000" stroke-width="3" paint-order="stroke">${str}</text>`;
  await dbg.overlay('09-corner-candidates.png', (s) => {
    const parts = [];
    for (const c of rejectedCorners) {
      if (c.segment2D) {
        const [a, b] = c.segment2D;
        parts.push(`<line x1="${a[0] * s}" y1="${a[1] * s}" x2="${b[0] * s}" y2="${b[1] * s}" stroke="#ff3030" stroke-width="2" stroke-dasharray="6 4"/>`);
        parts.push(text(((a[0] + b[0]) / 2) * s + 4, ((a[1] + b[1]) / 2) * s, s, `✗ ${(c.score ?? 0).toFixed(2)}`, '#ff8080'));
      } else if (c.position2D) {
        const [x] = c.position2D;
        parts.push(`<line x1="${x * s}" y1="0" x2="${x * s}" y2="${seg.h * s}" stroke="#ff9900" stroke-width="2" stroke-dasharray="3 5"/>`);
        parts.push(text(x * s + 4, 40, s, `✗ rgb-only`, '#ffbb66'));
      }
    }
    for (const c of corners) {
      const [a, b] = c.segment2D;
      parts.push(`<line x1="${a[0] * s}" y1="${a[1] * s}" x2="${b[0] * s}" y2="${b[1] * s}" stroke="#00ff66" stroke-width="3"/>`);
      parts.push(text(c.position2D[0] * s + 5, c.position2D[1] * s, s, `✓ ${c.score.toFixed(2)}`, '#80ffaa'));
    }
    return parts.join('');
  });

  // 10: validated corners with their evidence.
  await dbg.overlay('10-validated-corners.png', (s) => {
    const parts = [];
    for (const c of corners) {
      const [a, b] = c.segment2D;
      const colr = c.inferred ? '#ff00ff' : '#00ff66';
      parts.push(`<line x1="${a[0] * s}" y1="${a[1] * s}" x2="${b[0] * s}" y2="${b[1] * s}" stroke="${colr}" stroke-width="4" ${c.inferred ? 'stroke-dasharray="8 5"' : ''}/>`);
      const lines = [`${c.id} ${c.type}${c.angle ? ` ${Math.round(c.angle)}°` : ''} ${c.visible ? 'visible' : 'INFERRED'} ${c.confidence}`,
        ...Object.entries(c.evidence).map(([k, v]) => `${k}: ${v === null ? '—' : v}`)];
      lines.forEach((ln, k) => parts.push(text(c.position2D[0] * s + 6, c.position2D[1] * s - 50 + k * 15, s, ln, colr)));
    }
    return parts.join('');
  });

  // 11: wall quads (the geometry the renderer will use).
  await dbg.overlay('11-wall-polygons.png', (s) => walls.map((wl) => {
    if (!wl.quad) return '';
    const d = wl.quad.map(([x, y]) => `${x * s},${y * s}`).join(' ');
    const cx = wl.quad.reduce((a, p) => a + p[0], 0) / 4; const cy = wl.quad.reduce((a, p) => a + p[1], 0) / 4;
    return `<polygon points="${d}" fill="${col(wl.index)}" fill-opacity="0.25" stroke="${col(wl.index)}" stroke-width="3"/>${text(cx * s, cy * s, s, `${wl.id} ${wl.realSize.w}×${wl.realSize.h} m`, '#fff')}`;
  }).join(''));

  // 12: final visible masks, occluded hatched.
  await dbg.masks('12-final-wall-mask.png', masks.visible, masks.width, masks.height, { dim: masks.occluded });

  // 13: raycast result: owner per pixel, unassigned wall pixels in white.
  const own = masks.owner; const RW = masks.width; const RH = masks.height;
  const unassigned = new Uint8Array(RW * RH);
  for (let i = 0; i < RW * RH; i++) if (seg.crisp?.wall?.[i] && own[i] < 0) unassigned[i] = 255;
  await dbg.masks('13-raycast-result.png', [...masks.visible, unassigned], RW, RH);

  // 14: room model, top-down: final extents (incl. hidden parts) and corners.
  const fin = walls.map((wl) => ({
    id: wl.id,
    p0: [wl.t[0] * wl.sMin + wl.plane.nx * wl.plane.o, wl.t[1] * wl.sMin + wl.plane.nz * wl.plane.o],
    p1: [wl.t[0] * wl.sMax + wl.plane.nx * wl.plane.o, wl.t[1] * wl.sMax + wl.plane.nz * wl.plane.o],
  }));
  dbg.pointCloud(grid, geo.worldCloud, label, walls, {
    name: '14-final-room-model.png', extents: fin, corners: corners.map((c) => ({ xz: c.xz, inferred: c.inferred })),
  });

  dbg.writeJson('scan.json', {
    camera: geo.camera,
    walls: walls.map((wl) => publicWall(wl, geo)),
    corners: corners.map(publicCorner),
    rejectedCorners: rejectedCorners.map(publicRejected),
    raycast: masks.stats,
    diag: geo.diag,
    timings: geo.timings,
  });
}
