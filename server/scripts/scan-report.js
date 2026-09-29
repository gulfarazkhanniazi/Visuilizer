/**
 * Run surface detection over a folder of photographs and write, per photo, the
 * full result as JSON and an overlay PNG with every surface drawn in its own
 * colour. Used to snapshot the scanner before a change and compare after it.
 *
 *   node scripts/scan-report.js [--images ../test-images] [--out ./test/reports/run]
 *                               [--quality balanced] [--debug] [--legacy]
 *
 * Runs the detector in-process (not through the worker), so a stack trace
 * lands here rather than in a child's stderr.
 */
// Never open the real database from a dev script.
import '../test/helpers/env.js';
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { autoDetectSurfaces } from '../src/autodetect.js';

const args = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : dflt;
};
const flag = (name) => args.includes(`--${name}`);

const IMAGES = path.resolve(arg('images', '../test-images'));
const OUT = path.resolve(arg('out', './test/reports/latest'));
const quality = arg('quality', 'balanced');
const only = arg('only', null);

const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990'];

function overlaySvg(result, w, h) {
  const parts = [];
  result.objectList.forEach((o, i) => {
    const col = o.product_surface === 'floor' ? '#ffe119' : PALETTE[i % PALETTE.length];
    for (const p of o.mask?.polygons ?? []) {
      const d = p.points.map(([x, y]) => `${x},${y}`).join(' ');
      parts.push(p.mode === 'add'
        ? `<polygon points="${d}" fill="${col}" fill-opacity="0.38" stroke="${col}" stroke-width="2"/>`
        : `<polygon points="${d}" fill="#000" fill-opacity="0.45" stroke="#000" stroke-width="1"/>`);
    }
    if (o.quad) {
      const d = o.quad.map(([x, y]) => `${x},${y}`).join(' ');
      parts.push(`<polygon points="${d}" fill="none" stroke="${col}" stroke-width="2" stroke-dasharray="8 5"/>`);
    }
  });
  for (const c of result.scan?.corners ?? []) {
    const [x, y] = c.position2D;
    const seg = c.segment2D;
    if (seg) {
      parts.push(`<line x1="${seg[0][0]}" y1="${seg[0][1]}" x2="${seg[1][0]}" y2="${seg[1][1]}" stroke="${c.inferred ? '#ff00ff' : '#00ff66'}" stroke-width="3" ${c.inferred ? 'stroke-dasharray="6 4"' : ''}/>`);
    }
    parts.push(`<circle cx="${x}" cy="${y}" r="6" fill="${c.inferred ? '#ff00ff' : '#00ff66'}" stroke="#000"/>`);
  }
  // Labels last so they sit on top.
  result.objectList.forEach((o, i) => {
    const pts = (o.mask?.polygons ?? []).filter((p) => p.mode === 'add').flatMap((p) => p.points);
    if (!pts.length) return;
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const label = `${o.label}${o.scanId ? ` (${o.scanId})` : ''}`;
    parts.push(`<text x="${cx}" y="${cy}" font-family="sans-serif" font-size="${Math.max(12, w / 45)}" fill="#fff" stroke="#000" stroke-width="3" paint-order="stroke" text-anchor="middle">${label}</text>`);
  });
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">${parts.join('')}</svg>`);
}

function summarise(result) {
  return {
    scanner: result.scan?.scanner ?? 'legacy',
    surfaces: result.objectList.map((o) => ({
      name: o.name,
      label: o.label,
      surface: o.product_surface,
      realSize: o.realSize,
      polygons: o.mask?.polygons?.length ?? 0,
      confidence: o.confidence,
    })),
    walls: result.objectList.filter((o) => o.product_surface === 'wall').length,
    corners: result.scan?.corners?.map((c) => ({
      id: c.id, wallA: c.wallA, wallB: c.wallB, position2D: c.position2D,
      angle: c.angle, visible: c.visible, inferred: c.inferred, confidence: c.confidence,
    })),
    camera: result.camera,
    timings: result.scan?.metadata?.timings,
  };
}

fs.mkdirSync(OUT, { recursive: true });
const files = fs.readdirSync(IMAGES).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)).sort()
  .filter((f) => !only || f.includes(only));
const index = {};

for (const f of files) {
  const file = path.join(IMAGES, f);
  const stem = f.replace(/\.[^.]+$/, '').replace(/[^a-z0-9]+/gi, '_').slice(0, 40);
  const t0 = Date.now();
  let result;
  try {
    result = await autoDetectSurfaces(file, {
      quality,
      debug: flag('debug') ? path.join(OUT, `${stem}.debug`) : false,
      scanner: flag('legacy') ? 'legacy' : undefined,
    });
  } catch (e) {
    console.error(`${f}: FAILED ${e.stack}`);
    index[f] = { error: e.message };
    continue;
  }
  const ms = Date.now() - t0;
  const meta = await sharp(file).metadata();
  await sharp(file)
    .composite([{ input: overlaySvg(result, meta.width, meta.height) }])
    .png()
    .toFile(path.join(OUT, `${stem}.png`));
  fs.writeFileSync(path.join(OUT, `${stem}.json`), JSON.stringify(result, null, 1));
  index[f] = { ms, ...summarise(result) };
  const s = index[f];
  console.log(`${f.slice(0, 40).padEnd(40)} ${String(ms).padStart(6)}ms  scanner=${s.scanner}  walls=${s.walls}  corners=${s.corners?.length ?? '-'}  `
    + s.surfaces.map((x) => `${x.name}[${x.realSize?.w}x${x.realSize?.h}]`).join(' '));
}
fs.writeFileSync(path.join(OUT, 'index.json'), JSON.stringify(index, null, 1));
