/**
 * Scanner tests against synthetic rooms with exact ground truth.
 *
 * Every case in the spec's test list that can be rendered is here, scored
 * with measured metrics rather than eyeballed:
 *   walls    count, incorrect merges/splits, IoU, boundary F1, normal error
 *   corners  false positives/negatives, localisation error, angle error,
 *            visible vs inferred
 *   camera   gravity (pitch) and metric scale recovery
 * The depth fed in is corrupted like a monocular network's output.
 */
import './helpers/env.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENES } from './synthetic/scenes.js';
import { renderScene, corruptDepth, scanInput } from './synthetic/scene.js';
import { scoreScan } from './synthetic/metrics.js';
import { scanGeometry } from '../src/scanner/index.js';

// Acceptance thresholds. Pixel errors are for 480 px wide renders.
const T = {
  minIoU: 0.95,
  minBoundaryF1: 0.95,
  maxCornerErrFrac: 0.006,      // 0.6% of image width ≈ 3 px
  maxAngleErrDeg: 8,
  maxNormalErrDeg: 8,
};

const run = (def, extra = {}) => {
  const scene = renderScene(def.spec);
  const depth = corruptDepth(scene, def.corrupt ?? {});
  const res = scanGeometry(scanInput(scene, depth, extra), { quality: 'balanced', roomHeight: def.spec.roomHeight ?? 2.7 });
  return { scene, res, score: res.ok ? scoreScan(scene, res) : null };
};

for (const [name, def] of Object.entries(SCENES)) {
  test(`synthetic: ${name}`, () => {
    const { res, score } = run(def);
    assert.ok(res.ok, `scan failed: ${res.reason}`);
    const s = score;
    assert.equal(s.walls.merges, 0, 'incorrect merges');
    assert.equal(s.walls.splits, 0, 'incorrect splits');
    assert.equal(s.walls.detected, s.walls.expected, 'wall count');
    assert.ok(s.meanIoU >= T.minIoU, `IoU ${s.meanIoU}`);
    assert.ok(s.meanBoundaryF1 >= T.minBoundaryF1, `boundary F1 ${s.meanBoundaryF1}`);
    assert.ok(s.maxNormalErrDeg <= T.maxNormalErrDeg, `normal error ${s.maxNormalErrDeg}`);
    assert.equal(s.corners.falsePositives, 0, 'false corners');
    assert.equal(s.corners.falseNegatives, 0, 'missed corners');
    for (const c of s.cornerScores) {
      assert.ok(c.errNorm <= T.maxCornerErrFrac, `corner ${c.gt} off by ${c.errPx.toFixed(2)} px`);
      if (c.angleErrDeg !== null) assert.ok(c.angleErrDeg <= T.maxAngleErrDeg, `corner ${c.gt} angle off by ${c.angleErrDeg.toFixed(1)}°`);
    }
    if (def.expect.inferred) {
      assert.equal(s.cornerScores.filter((c) => c.inferred).length, def.expect.inferred, 'inferred corners');
      for (const c of res.corners.filter((k) => k.inferred)) assert.equal(c.visible, false);
    }
    if (def.expect.pitch !== undefined) assert.ok(Math.abs(res.camera.pitchDeg - def.expect.pitch) <= 1.5, `pitch ${res.camera.pitchDeg}`);
    if (def.expect.camHeight !== undefined) assert.ok(Math.abs(res.camera.height - def.expect.camHeight) / def.expect.camHeight <= 0.05, `camera height ${res.camera.height}`);
  });
}

test('occlusion: the sofa is not wall, and the wall continues behind it', () => {
  const { scene, res } = run(SCENES.furniture);
  const back = res.walls.find((w) => Math.abs(w.plane.nz) > 0.9);
  const vis = res.masks.visible[back.index];
  const occ = res.masks.occluded[back.index];
  let sofaInWall = 0; let sofa = 0; let sofaOccluded = 0;
  for (let i = 0; i < scene.cls.length; i++) {
    if (scene.cls[i] !== 5) continue;
    sofa++;
    if (res.masks.visible.some((m) => m[i])) sofaInWall++;
    if (occ[i]) sofaOccluded++;
  }
  assert.equal(sofaInWall, 0, 'sofa pixels assigned to a wall');
  assert.ok(sofaOccluded / sofa > 0.5, `only ${(100 * sofaOccluded / sofa).toFixed(0)}% of the sofa is recorded as occluding the wall`);
  // The wall's quad reaches the floor behind the sofa: its extent is the
  // whole plane, not just the visible part above the sofa.
  assert.equal(back.polygon3D[2][1], 0);
  assert.ok(vis.some(Boolean));
});

test('pixel ownership is exclusive and deterministic', () => {
  const a = run(SCENES.three_walls).res;
  const b = run(SCENES.three_walls).res;
  const n = a.masks.width * a.masks.height;
  for (let i = 0; i < n; i++) {
    let owners = 0;
    for (const m of a.masks.visible) if (m[i]) owners++;
    assert.ok(owners <= 1, `pixel ${i} owned by ${owners} walls`);
    assert.equal(a.masks.owner[i], b.masks.owner[i]);
  }
  assert.deepEqual(a.walls.map((w) => w.id), b.walls.map((w) => w.id));
  assert.deepEqual(a.corners.map((c) => c.id), b.corners.map((c) => c.id));
});

test('a window frame, door frame and painting never become corners (rejected with a reason)', () => {
  for (const name of ['window', 'door_frame', 'painting', 'shadow']) {
    const { res } = run(SCENES[name]);
    assert.equal(res.corners.length, 2, `${name}: corners`);
    for (const r of res.rejectedCorners.filter((c) => c.source === 'rgb-column')) {
      assert.match(r.reason, /No supporting plane intersection|No wall plane|do not meet/);
    }
  }
});

test('non-90 corner angles are measured, not forced to 90', () => {
  const { res } = run(SCENES.non_90);
  const angles = res.corners.map((c) => c.angle).sort((p, q) => p - q);
  assert.ok(Math.abs(angles[0] - 90) < T.maxAngleErrDeg);
  assert.ok(Math.abs(angles[1] - 135) < T.maxAngleErrDeg, `angle ${angles[1]}`);
});

test('every corner carries its evidence', () => {
  const { res } = run(SCENES.three_walls);
  for (const c of res.corners) {
    for (const k of ['planeIntersection', 'depth', 'normals', 'lines', 'segmentation', 'rgb']) assert.ok(k in c.evidence, k);
    assert.ok(c.confidence > 0 && c.confidence <= 1);
    assert.ok(Array.isArray(c.position2D) && Array.isArray(c.position3D));
  }
});

test('quality levels all run', () => {
  for (const quality of ['fast', 'balanced', 'high']) {
    const scene = renderScene(SCENES.three_walls.spec);
    const res = scanGeometry(scanInput(scene, corruptDepth(scene)), { quality });
    assert.ok(res.ok, `${quality}: ${res.reason}`);
    assert.equal(res.walls.length, 3, quality);
  }
});

test('no wall pixels: a controlled failure, not an exception', () => {
  const scene = renderScene({ walls: [], ceiling: false });
  const res = scanGeometry(scanInput(scene, corruptDepth(scene)), {});
  assert.equal(res.ok, false);
  assert.ok(res.reason);
});
