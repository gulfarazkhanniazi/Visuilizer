/**
 * API integration: the real Express server, a throwaway database, a real
 * photograph through upload -> auto-detect, exactly as the frontend calls it.
 *
 * Verifies the response keeps the contract the frontend reads (objectList
 * entries with name/label/product_surface/quad/realSize/mask.polygons, and
 * camera.height for the Studio's toast) and carries the new scan block.
 *
 * Needs the segmentation model (cached in server/data/models); runs whether
 * or not the CV service is up -- without it the scan must fall back, not fail.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SERVER = path.resolve(__dirname, '..');
const PHOTO = path.resolve(SERVER, '..', 'test-images', 'put-together-a-perfect-guest-room-1976987-hero-223e3e8f697e4b13b62ad4fe898d492d.jpg');
const hasModel = fs.existsSync(path.join(SERVER, 'data', 'models', 'Xenova', 'segformer-b0-finetuned-ade-512-512'));

async function startServer(env) {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'scan-api-'));
  // Reuse the cached segmentation model rather than downloading it again.
  fs.symlinkSync(path.join(SERVER, 'data', 'models'), path.join(dataDir, 'models'));
  const port = 20000 + Math.floor(Math.random() * 20000);
  const child = spawn(process.execPath, ['src/index.js'], {
    cwd: SERVER, env: { ...process.env, PORT: String(port), DATA_DIR: dataDir, ...env }, stdio: ['ignore', 'pipe', 'pipe'],
  });
  let log = '';
  child.stdout.on('data', (b) => { log += b; });
  child.stderr.on('data', (b) => { log += b; });
  const base = `http://127.0.0.1:${port}/api`;
  for (let k = 0; k < 100; k++) {
    try { if ((await fetch(`${base}/health`)).ok) break; } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  return {
    base, child, dataDir, log: () => log,
    stop: () => { child.kill(); fs.rmSync(dataDir, { recursive: true, force: true }); },
  };
}

async function uploadAndDetect(base, body = {}) {
  const fd = new FormData();
  fd.append('photo', new Blob([fs.readFileSync(PHOTO)], { type: 'image/jpeg' }), 'room.jpg');
  fd.append('isCustom', '1');
  fd.append('owner', 'visitor-test');
  const up = await fetch(`${base}/rooms/upload`, { method: 'POST', body: fd });
  assert.equal(up.status, 201);
  const room = await up.json();
  const res = await fetch(`${base}/rooms/${room.id}/auto-detect`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-visitor': 'visitor-test' },
    body: JSON.stringify(body),
  });
  return { room, res, json: await res.json() };
}

function assertFrontendContract(json) {
  assert.ok(json.room && Array.isArray(json.room.objectList));
  assert.ok(json.room.objectList.length >= 2, 'floor and at least one wall');
  for (const o of json.room.objectList) {
    assert.equal(typeof o.name, 'string');
    assert.equal(typeof o.label, 'string');
    assert.ok(['floor', 'wall', 'ceiling'].includes(o.product_surface));
    assert.equal(o.quad.length, 4);
    for (const p of o.quad) assert.ok(p.every(Number.isFinite));
    assert.ok(o.realSize.w > 0 && o.realSize.h > 0);
    assert.ok(o.mask.polygons.length > 0);
    for (const poly of o.mask.polygons) {
      assert.ok(['add', 'subtract'].includes(poly.mode));
      assert.ok(poly.points.length >= 3);
    }
  }
  // Names are unique: the frontend selects surfaces by name.
  const names = json.room.objectList.map((o) => o.name);
  assert.equal(new Set(names).size, names.length);
  assert.equal(typeof json.camera.height, 'number');
  assert.ok(json.scan && Array.isArray(json.scan.walls) && Array.isArray(json.scan.corners));
  assert.ok(json.scan.metadata.modelVersions.segmentationModel);
  assert.ok(json.scan.metadata.modelVersions.geometryVersion);
}

test('upload -> auto-detect keeps the frontend contract (CV service off: controlled fallback)', { skip: !hasModel && 'segmentation model not cached' }, async () => {
  // Force both depth sources off: the scan must fall back to the junction
  // scanner and still succeed, saying why.
  const srv = await startServer({ CV_SERVICE: 'off' });
  try {
    const { res, json } = await uploadAndDetect(srv.base);
    assert.equal(res.status, 200, srv.log());
    assertFrontendContract(json);
    assert.equal(json.scan.scanner, 'legacy');
    assert.match(json.scan.fallbackReason, /depth/);
  } finally { srv.stop(); }
});

test('upload -> auto-detect with the 3D scanner (needs the CV service)', { skip: !hasModel && 'segmentation model not cached' }, async (t) => {
  let up = false;
  try { up = (await fetch(`${process.env.CV_SERVICE_URL || 'http://127.0.0.1:5179'}/health`)).ok; } catch { /* down */ }
  if (!up) { t.skip('CV service not running'); return; }
  const srv = await startServer({});
  try {
    const { res, json } = await uploadAndDetect(srv.base, { quality: 'balanced' });
    assert.equal(res.status, 200, srv.log());
    assertFrontendContract(json);
    assert.equal(json.scan.scanner, 'geometry3d');
    const walls = json.room.objectList.filter((o) => o.product_surface === 'wall');
    // The guest room: left, back and right walls.
    assert.equal(walls.length, 3, walls.map((w) => w.name).join(','));
    assert.deepEqual(walls.map((w) => w.name).sort(), ['back_wall', 'left_wall', 'right_wall']);
    for (const w of walls) {
      assert.match(w.scanId, /^wall_\d\d$/);
      assert.ok(w.confidence > 0 && w.confidence <= 1);
      assert.ok(w.geometry.plane && w.geometry.polygon3D.length === 4);
    }
    assert.equal(json.scan.corners.length, 2);
    for (const c of json.scan.corners) {
      assert.match(c.id, /^corner_\d\d$/);
      for (const k of ['wallA', 'wallB', 'position2D', 'position3D', 'angle', 'visible', 'inferred', 'confidence', 'evidence']) assert.ok(k in c, k);
    }
    // Persisted with the room.
    const saved = await (await fetch(`${srv.base}/rooms/${json.room.id}`)).json();
    assert.equal(saved.settings.scan.scanner, 'geometry3d');
    assert.equal(saved.objectList.length, json.room.objectList.length);
  } finally { srv.stop(); }
});

test('auto-detect on a missing room is a 404, not a crash', async () => {
  const srv = await startServer({ CV_SERVICE: 'off' });
  try {
    const res = await fetch(`${srv.base}/rooms/nope/auto-detect`, { method: 'POST' });
    assert.equal(res.status, 404);
    assert.ok((await fetch(`${srv.base}/health`)).ok, 'server still up');
  } finally { srv.stop(); }
});
