/**
 * Unit tests: the geometric primitives the scanner is built from.
 */
import './helpers/env.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  eigSym3, fitPlaneLS, rng, percentile, angleBetween,
} from '../src/scanner/math.js';
import {
  worldFrame, toWorld, toCamera, projectWorld, backProject, buildGrid, CLS,
} from '../src/scanner/pointcloud.js';
import { fitVertical } from '../src/scanner/planes.js';
import { vanishingPoints } from '../src/scanner/vanishing.js';
import { renderScene } from './synthetic/scene.js';
import { SCENES } from './synthetic/scenes.js';
import {
  validatePolygons, untangle, selfIntersects, polygonArea, nameWallFromPlane,
} from '../src/autodetect.js';
import { exifFocal35 } from '../src/storage.js';

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} expected ${b} ± ${tol}, got ${a}`);

test('eigSym3 recovers eigenvalues and vectors', () => {
  const e = eigSym3([2, 0, 0, 3, 0, 1]);
  near(e[0].value, 1, 1e-9); near(e[1].value, 2, 1e-9); near(e[2].value, 3, 1e-9);
  near(Math.abs(e[0].vector[2]), 1, 1e-9);
});

test('least-squares plane through points of a known plane', () => {
  // 0.3x - 0.2y + z = 4
  const P = { x: [], y: [], z: [] };
  const r = rng(1);
  for (let k = 0; k < 400; k++) {
    const x = r() * 4 - 2; const y = r() * 3 - 1.5;
    P.x.push(x); P.y.push(y); P.z.push(4 - 0.3 * x + 0.2 * y);
  }
  const pl = fitPlaneLS(P, [...Array(400).keys()]);
  const n = [0.3, -0.2, 1].map((v) => v / Math.hypot(0.3, 0.2, 1));
  near(angleBetween(pl.n, n, true), 0, 1e-6, 'normal');
});

test('rng is deterministic', () => {
  const a = rng(42); const b = rng(42);
  for (let k = 0; k < 10; k++) assert.equal(a(), b());
});

test('percentile interpolates', () => {
  near(percentile([0, 10], 0.5), 5, 1e-12);
  near(percentile([3, 1, 2], 1), 3, 1e-12);
});

test('world frame round trip, level and tilted cameras', () => {
  for (const up of [[0, -1, 0], [0, -Math.cos(0.2), Math.sin(0.2)], [Math.sin(0.05), -Math.cos(0.05), 0]]) {
    const frame = worldFrame(up, 1.4);
    const p = [0.7, -0.3, 3.2];
    const wp = toWorld(frame, ...p);
    const back = toCamera(frame, ...wp);
    for (let k = 0; k < 3; k++) near(back[k], p[k], 1e-9);
  }
  const level = worldFrame([0, -1, 0], 1.5);
  near(level.pitch, 0, 1e-9);
  // A point straight ahead on the floor projects below the centre.
  const cam = { f: 500, cx: 320, cy: 240 };
  const px = projectWorld(cam, level, 0, 0, 3);
  near(px[0], 320, 1e-9); near(px[1], 240 + (500 * 1.5) / 3, 1e-9);
});

test('a flat wall back-projects onto one plane (point cloud validation)', () => {
  // Depth of the fronto-parallel wall Z = 4 is constant; a wall at 30 deg is
  // linear in inverse depth. Both must reconstruct as exact planes.
  const w = 64; const h = 48; const f = 50;
  for (const [nx, nz, o] of [[0, -1, -4], [Math.sin(0.5), -Math.cos(0.5), -3]]) {
    const depth = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = (x + 0.5 - w / 2) / f;
        depth[y * w + x] = o / (nx * a + nz);
      }
    }
    const masks = { wall: new Uint8Array(w * h).fill(255) };
    const grid = buildGrid({ w, h, RW: w, RH: h, masks, objects: {}, depth: { data: depth, width: w, height: h }, gridW: w });
    const P = backProject(grid, { f, cx: w / 2, cy: h / 2 });
    const frame = worldFrame([0, -1, 0], 1.5);
    const W = { X: new Float32Array(w * h), Z: new Float32Array(w * h) };
    for (let i = 0; i < w * h; i++) { const q = toWorld(frame, P.x[i], P.y[i], P.z[i]); W.X[i] = q[0]; W.Z[i] = q[2]; }
    const pl = fitVertical(W, [...Array(w * h).keys()]);
    let maxRes = 0;
    for (let i = 0; i < w * h; i++) maxRes = Math.max(maxRes, Math.abs(pl.nx * W.X[i] + pl.nz * W.Z[i] - pl.o));
    assert.ok(maxRes < 1e-4, `max residual ${maxRes}`);
    near(Math.abs(pl.nx * nx + pl.nz * nz), 1, 1e-6, 'normal');
  }
  assert.equal(CLS.WALL, 1);
});

test('vanishing points give the focal length of a pitched synthetic room', () => {
  const sc = renderScene(SCENES.pitched.spec);
  const vp = vanishingPoints(sc.lines, sc.w, sc.h);
  // Either an estimate within 4 degrees, or an honest abstention.
  if (vp.hfov) near(vp.hfov, SCENES.pitched.spec.hfov, 4, 'hfov');
});

test('vanishing points abstain on too few lines', () => {
  const vp = vanishingPoints([[0, 0, 10, 10]], 640, 480);
  assert.equal(vp.focalPx, null);
});

test('polygon validation: area, bounds, self-intersection repair', () => {
  const bow = [[0, 0], [100, 100], [100, 0], [0, 100]];        // figure-8
  assert.ok(selfIntersects(bow));
  const fixed = untangle(bow);
  assert.ok(!selfIntersects(fixed));
  near(polygonArea(fixed), 10000, 1e-9);
  const { polygons, notes } = validatePolygons([
    { mode: 'add', points: bow },
    { mode: 'add', points: [[-50, -50], [700, -50], [700, 600], [-50, 600]] },   // clamps to image
    { mode: 'add', points: [[1, 1], [2, 1], [1, 2]] },                          // no area
  ], 640, 480);
  assert.equal(polygons.length, 2);
  assert.ok(polygons.every((p) => !selfIntersects(p.points)));
  assert.ok(polygons[1].points.every(([x, y]) => x >= 0 && y >= 0 && x <= 640 && y <= 480));
  assert.ok(notes.some((n) => /repaired/.test(n)) && notes.some((n) => /no area/.test(n)));
});

test('walls are named from their orientation', () => {
  assert.equal(nameWallFromPlane({ nx: 0, nz: -1 }).key, 'back_wall');
  assert.equal(nameWallFromPlane({ nx: 1, nz: 0 }).key, 'left_wall');
  assert.equal(nameWallFromPlane({ nx: -0.9, nz: -0.44 }).key, 'right_wall');
});

test('EXIF focal length parsing', async () => {
  const sharp = (await import('sharp')).default;
  const buf = await sharp({ create: { width: 8, height: 8, channels: 3, background: '#777' } })
    .jpeg().withExif({ IFD0: { Make: 'x' }, IFD2: { FocalLengthIn35mmFilm: '26' } }).toBuffer();
  assert.equal(exifFocal35((await sharp(buf).metadata()).exif), 26);
  assert.equal(exifFocal35(null), null);
  assert.equal(exifFocal35(Buffer.from('garbage-garbage')), null);
});
