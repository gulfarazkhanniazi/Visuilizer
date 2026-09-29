/**
 * Run the 3D scanner over every synthetic scene and print the metrics.
 *   node scripts/synthetic-report.js [--only name] [--debug] [--json out.json]
 */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { SCENES } from '../test/synthetic/scenes.js';
import { renderScene, corruptDepth, scanInput } from '../test/synthetic/scene.js';
import { scoreScan } from '../test/synthetic/metrics.js';
import { scanGeometry } from '../src/scanner/index.js';
import { DebugWriter } from '../src/scanner/debug.js';

const args = process.argv.slice(2);
const opt = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : null);
const only = opt('--only');
const OUT = path.resolve('./test/reports/synthetic');
const rows = {};

export async function runScene(name, def, { debug = false } = {}) {
  const scene = renderScene(def.spec);
  const depth = corruptDepth(scene, def.corrupt ?? {});
  let dbg = null;
  if (debug) {
    fs.mkdirSync(OUT, { recursive: true });
    const png = path.join(OUT, `${name}.png`);
    const rgb = Buffer.alloc(scene.w * scene.h * 3);
    for (let i = 0; i < scene.w * scene.h; i++) rgb.fill(Math.round(scene.luma[i] * 255), i * 3, i * 3 + 3);
    await sharp(rgb, { raw: { width: scene.w, height: scene.h, channels: 3 } }).png().toFile(png);
    dbg = new DebugWriter(path.join(OUT, name), { photo: png, w: scene.w, h: scene.h });
  }
  const t = performance.now();
  const res = scanGeometry(scanInput(scene, depth), { quality: def.quality ?? 'balanced', debugWriter: dbg, roomHeight: def.spec.roomHeight ?? 2.7 });
  const ms = performance.now() - t;
  if (dbg) await dbg.flush();
  return { scene, res, ms, score: res.ok ? scoreScan(scene, res) : null };
}

if (process.argv[1] && process.argv[1].endsWith('synthetic-report.js')) {
  for (const [name, def] of Object.entries(SCENES)) {
    if (only && !name.includes(only)) continue;
    const { res, ms, score } = await runScene(name, def, { debug: args.includes('--debug') });
    if (!res.ok) { console.log(`${name.padEnd(14)} FAILED: ${res.reason}`); rows[name] = { failed: res.reason }; continue; }
    const s = score;
    rows[name] = { ms: Math.round(ms), ...s, camera: res.camera };
    console.log(`${name.padEnd(14)} ${String(Math.round(ms)).padStart(4)}ms walls ${s.walls.detected}/${s.walls.expected} (merge ${s.walls.merges} split ${s.walls.splits}) `
      + `IoU ${s.meanIoU?.toFixed(3)} BF1 ${s.meanBoundaryF1?.toFixed(3)} nErr ${s.maxNormalErrDeg?.toFixed(1)}° | corners ${s.corners.detected}/${s.corners.expected} FP ${s.corners.falsePositives} FN ${s.corners.falseNegatives} `
      + `err ${s.meanCornerErrPx?.toFixed(1) ?? '-'}px ${s.cornerScores.map((c) => `${c.inferred ? 'inf' : 'vis'}${c.angleErrDeg !== null ? `/${c.angleErrDeg.toFixed(1)}°` : ''}`).join(',')} `
      + `| camH ${res.camera.height} pitch ${res.camera.pitchDeg}`);
  }
  if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify(rows, null, 1));
}
