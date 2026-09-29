/**
 * Depth-model evaluation on real photographs, without ground truth.
 *
 * A depth map is only as good as the geometry it produces, and indoor
 * geometry has checkable invariants:
 *   planarity     floor and walls are planes: median point-to-plane residual
 *                 as a fraction of distance
 *   verticality   walls are perpendicular to the floor: median tilt of wall
 *                 normals, measured against the floor the same depth produced
 *   scale         the raw metric camera height (before any prior is applied)
 *                 should be a person's eye level, ~1.0-1.7 m
 *   separation    how many walls survive validation
 *   cost          inference time
 *
 *   node scripts/eval-depth.js [--models da2-metric-indoor-small,depth-pro]
 * Needs the CV service. Uses the photo's own field of view estimate for all
 * models so they are compared on depth alone.
 */
// Never open the real database from a dev script.
import '../test/helpers/env.js';
import fs from 'node:fs';
import path from 'node:path';
import { segmentAndRefine } from '../src/autodetect.js';
import { estimateDepth } from '../src/scanner/depth.js';
import { prepareGeometry } from '../src/scanner/index.js';
import { CLS } from '../src/scanner/pointcloud.js';
import { median } from '../src/scanner/math.js';

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const models = opt('--models', 'da2-metric-indoor-small,da2-metric-indoor-base,depth-pro').split(',');
const IMAGES = path.resolve(opt('--images', '../test-images'));
const files = fs.readdirSync(IMAGES).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort();

const rows = [];
for (const f of files) {
  const file = path.join(IMAGES, f);
  const seg = await segmentAndRefine(file);
  for (const model of models) {
    const t = Date.now();
    const depth = await estimateDepth(file, { depthModel: model, allowOnnx: false });
    const ms = Date.now() - t;
    if (!depth) { rows.push({ image: f, model, error: 'no depth' }); continue; }
    const ctx = prepareGeometry({
      w: seg.w, h: seg.h, RW: seg.RW, RH: seg.RH, masks: seg.crisp, objects: seg.objects, luma: seg.guide, depth,
    }, { quality: 'balanced', hfov: 70, hfovSource: 'fixed for evaluation' });
    const c = ctx.ok ? ctx : ctx.context;
    if (!c) { rows.push({ image: f, model, error: ctx.reason }); continue; }
    const walls = c.ex.walls;
    // Verticality over every interior wall pixel, not only accepted walls.
    const tilt = [];
    for (let i = 0; i < c.grid.cls.length; i++) {
      if (c.grid.cls[i] !== CLS.WALL || !c.grid.interior[i] || !c.W.ok[i]) continue;
      tilt.push((Math.asin(Math.min(1, Math.abs(c.W.nY[i]))) * 180) / Math.PI);
    }
    rows.push({
      image: f.slice(0, 22),
      model,
      ms,
      rawCamHeight: c.hp.floor ? +(c.hp.floor.d).toFixed(2) : null,
      floorResidualPct: c.hp.floor ? +(c.hp.floor.medianRelResidual * 100).toFixed(2) : null,
      wallResidualPct: walls.length ? +(median(walls.map((wl) => wl.stats.medianRelResidual)) * 100).toFixed(2) : null,
      wallTiltDeg: tilt.length ? +median(tilt).toFixed(1) : null,
      walls: walls.length,
      rejected: c.ex.diag.rejected.length,
      scaleFactor: c.diag.scale.factor,
    });
  }
}
console.table(rows);
const summary = {};
for (const m of models) {
  const r = rows.filter((x) => x.model === m && !x.error);
  const avg = (k) => +(r.reduce((s, x) => s + (x[k] ?? 0), 0) / Math.max(1, r.filter((x) => x[k] !== null).length)).toFixed(3);
  summary[m] = {
    images: r.length,
    medianMs: median(r.map((x) => x.ms)),
    floorResidualPct: avg('floorResidualPct'),
    wallResidualPct: avg('wallResidualPct'),
    wallTiltDeg: avg('wallTiltDeg'),
    meanAbsCamHeightErrorFrom1p35: +(r.reduce((s, x) => s + Math.abs((x.rawCamHeight ?? 1.35) - 1.35), 0) / Math.max(1, r.length)).toFixed(2),
  };
}
console.table(summary);
if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify({ rows, summary }, null, 1));
