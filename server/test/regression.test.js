/**
 * Regression on real photographs (test-images/), against manually verified
 * expectations -- and the frontend's own selection code run on the result.
 *
 * Needs the segmentation model and the CV service (python3 cv-service/server.py);
 * skipped otherwise, because without depth the scanner falls back by design.
 */
import './helpers/env.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { autoDetectSurfaces } from '../src/autodetect.js';
import { pointInMask } from '../../web/src/engine/masks.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMAGES = path.resolve(__dirname, '..', '..', 'test-images');
const EXPECT = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'real-expectations.json'), 'utf8'));

let serviceUp = false;
try { serviceUp = (await fetch(`${process.env.CV_SERVICE_URL || 'http://127.0.0.1:5179'}/health`)).ok; } catch { /* down */ }
const hasModel = fs.existsSync(path.join(__dirname, '..', 'data', 'models', 'Xenova', 'segformer-b0-finetuned-ade-512-512'));
const skip = !serviceUp ? 'CV service not running' : !hasModel ? 'segmentation model not cached' : false;

/** The Visualizer's surfaceAt: search back to front, first mask containing the point. */
const surfaceAt = (objectList, x, y) => {
  for (let i = objectList.length - 1; i >= 0; i--) if (pointInMask(objectList[i].mask, x, y)) return objectList[i].name;
  return null;
};

for (const [file, exp] of Object.entries(EXPECT)) {
  if (file.startsWith('_')) continue;
  test(`real photo: ${exp.scene}`, { skip }, async () => {
    const res = await autoDetectSurfaces(path.join(IMAGES, file), { quality: 'balanced' });
    assert.equal(res.scan.scanner, 'geometry3d', res.scan.fallbackReason);
    const walls = res.objectList.filter((o) => o.product_surface === 'wall');
    const names = walls.map((w) => w.name);
    assert.ok(walls.length >= exp.walls[0] && walls.length <= exp.walls[1], `walls ${names.join(',')} (expected ${exp.walls.join('-')})`);
    for (const n of exp.mustInclude) assert.ok(names.includes(n), `missing ${n} in ${names.join(',')}`);
    const nc = res.scan.corners.length;
    assert.ok(nc >= exp.corners[0] && nc <= exp.corners[1], `corners ${nc} (expected ${exp.corners.join('-')})`);

    // Selection, exactly as the Visualizer does it: a click inside a wall
    // selects that wall and nothing else.
    const meta = res.scan.walls.filter((w) => w.selectable);
    for (const wall of walls) {
      const pts = wall.mask.polygons.filter((p) => p.mode === 'add').flatMap((p) => p.points);
      const xs = pts.map((p) => p[0]); const ys = pts.map((p) => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      let inside = 0; let selected = 0; let alsoOther = 0;
      for (let k = 0; k < 900; k++) {
        const x = x0 + ((k % 30) + 0.5) * ((x1 - x0) / 30); const y = y0 + (Math.floor(k / 30) + 0.5) * ((y1 - y0) / 30);
        if (!pointInMask(wall.mask, x, y)) continue;
        inside++;
        if (surfaceAt(res.objectList, x, y) === wall.name) selected++;
        if (walls.some((o) => o !== wall && pointInMask(o.mask, x, y))) alsoOther++;
      }
      assert.ok(inside > 0, `${wall.name} has no interior`);
      assert.ok(selected / inside >= 0.97, `${wall.name}: only ${(100 * selected / inside).toFixed(1)}% of clicks select it`);
      // Traced outlines of adjacent walls may touch; they must not overlap.
      assert.ok(alsoOther / inside <= 0.01, `${wall.name}: ${(100 * alsoOther / inside).toFixed(1)}% of it also lies in another wall`);
      assert.ok(meta.some((m) => m.objectName === wall.name), `${wall.name} missing from scan.walls`);
    }
    // Stable IDs: the same photo scanned twice gives the same walls and corners.
    const again = await autoDetectSurfaces(path.join(IMAGES, file), { quality: 'balanced' });
    assert.deepEqual(again.objectList.map((o) => [o.name, o.scanId]), res.objectList.map((o) => [o.name, o.scanId]));
    assert.deepEqual(again.scan.corners.map((c) => c.id), res.scan.corners.map((c) => c.id));
  });
}
