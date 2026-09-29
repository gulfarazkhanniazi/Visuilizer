/**
 * Stage check: segmentation -> depth -> point cloud -> planes, with debug
 * images, without the later stages.
 *   node scripts/probe-planes.js [--only name] [--depth-model depth-pro]
 */
// Never open the real database from a dev script.
import '../test/helpers/env.js';
import fs from 'node:fs';
import path from 'node:path';
import { segmentAndRefine } from '../src/autodetect.js';
import { estimateDepth } from '../src/scanner/depth.js';
import { prepareGeometry } from '../src/scanner/index.js';
import { DebugWriter } from '../src/scanner/debug.js';
import { deg } from '../src/scanner/math.js';

const args = process.argv.slice(2);
const opt = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
const only = opt('--only');
const IMAGES = path.resolve('../test-images');
const OUT = path.resolve(opt('--out') ?? './test/reports/probe');
fs.mkdirSync(OUT, { recursive: true });

for (const f of fs.readdirSync(IMAGES).filter((x) => /\.(jpe?g|png)$/i.test(x)).sort()) {
  if (only && !f.includes(only)) continue;
  const file = path.join(IMAGES, f);
  const stem = f.replace(/\.[^.]+$/, '').replace(/[^a-z0-9]+/gi, '_').slice(0, 30);
  const seg = await segmentAndRefine(file);
  const depth = await estimateDepth(file, { quality: 'balanced', depthModel: opt('--depth-model') ?? undefined });
  const dbg = new DebugWriter(path.join(OUT, stem), { photo: file, w: seg.w, h: seg.h });
  const ctx = prepareGeometry({
    w: seg.w, h: seg.h, RW: seg.RW, RH: seg.RH, masks: seg.crisp, objects: seg.objects, luma: seg.guide, depth,
  }, { quality: 'balanced', debugWriter: dbg });
  const c = ctx.ok ? ctx : ctx.context;
  if (!c) { console.log(f, 'FAILED', ctx.reason); continue; }
  const ex = c.ex;
  dbg.pointCloud(c.grid, c.W, ex.label, ex.walls);
  await dbg.labels(c.grid, ex.label, '07-plane-detection.png');
  await dbg.flush();
  const cam = c.camera;
  console.log(`${f.slice(0, 30).padEnd(30)} hfov=${cam.hfov.toFixed(1)} (${cam.focalSource}) camH=${cam.height.toFixed(2)} ceil=${cam.ceilingHeight ?? '-'} `
    + `pitch=${cam.pitchDeg} roll=${cam.rollDeg} scale=${c.diag.scale.factor} (${c.diag.scale.source}) walls=${ex.walls.length} rej=${ex.diag.rejected.length}`);
  for (const wl of ex.walls) {
    const s = wl.stats;
    console.log(`    az=${deg(Math.atan2(wl.plane.nz, wl.plane.nx)).toFixed(0)} o=${wl.plane.o.toFixed(2)} n=${s.points} med=${(s.medianRelResidual * 100).toFixed(1)}% p95=${(s.p95RelResidual * 100).toFixed(1)}% nc=${s.normalConsistency.toFixed(2)} vert=${s.verticalityDeg?.toFixed(1)} inl=${s.inlierRatio.toFixed(2)} sem=${s.segmentationConsistency.toFixed(2)} conf=${s.confidence}`);
  }
  for (const r of ex.diag.rejected) console.log(`    REJECTED az=${r.plane.azimuthDeg} o=${r.plane.offset}: ${r.reason}`);
}
