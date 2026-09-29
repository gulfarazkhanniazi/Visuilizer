/**
 * The 3D room scanner.
 *
 *   segmentation (SegFormer, refined)  ─┐
 *   monocular metric depth             ─┼─> point cloud ─> floor plane ─> gravity
 *                                        │        │
 *                                        │        └─> normals ─> wall planes
 *                                        │                          │
 *   RGB edges, LSD lines  ──────────────┴──> corner evidence <─────┤ plane ∩ plane
 *                                                                   │
 *                              occlusion ─> extents ─> raycast every pixel
 *                                                         │
 *                                                  per-wall masks + quads
 *
 * `scanGeometry` is pure: it takes arrays and returns walls, corners and
 * masks, so the synthetic tests drive exactly the code the photographs do.
 */
import {
  buildGrid, backProject, computeNormals, depthEdges, horizontalPlanes, worldFrame,
  worldCloud, alignRelativeDepth, selfCalibrateFocal,
} from './pointcloud.js';
import { extractWalls } from './planes.js';
import { vanishingPoints } from './vanishing.js';
import { buildWalls } from './walls.js';
import { assignPixels } from './assign.js';
import {
  QUALITY, CAMERA, GEOMETRY_VERSION, CONFIDENCE_BANDS,
} from './config.js';
import { cross, unit, dot, rad, deg } from './math.js';

export { GEOMETRY_VERSION };

/**
 * @param input {
 *   w, h            photo size
 *   RW, RH          refinement-resolution size of masks/objects/luma
 *   masks           { wall, floor, ceiling } Uint8Array at RW x RH (0 / non-zero)
 *   objects         { wall, floor } Float32Array 0..1 at RW x RH (or null)
 *   luma            Float32Array 0..1 at RW x RH
 *   depth           { data, width, height, metric, confidence?, focalPx?, lines? }
 *   levelCam        legacy camera solve { f, cx, cy, height, source }, used to
 *                   align relative depth and as the fallback camera
 * }
 */
export function scanGeometry(input, opts = {}) {
  const ctx = prepareGeometry(input, opts);
  if (!ctx.ok) return ctx;
  const {
    grid, cam, frame, hp, ceilingY, timings, diag, ex, camera,
  } = ctx;
  const tick = (k, t) => { timings[k] = Math.round(performance.now() - t); };

  // --- corners, extents, polygons ----------------------------------------------
  let t = performance.now();
  const built = buildWalls(ctx, ex);
  tick('corners', t);

  // --- raycast every pixel -----------------------------------------------------
  t = performance.now();
  const assigned = assignPixels(ctx, built);
  tick('raycast', t);

  return {
    ok: true,
    camera,
    frame,
    cam,
    walls: built.walls,
    corners: built.corners,
    rejectedCorners: built.rejected,
    floor: built.floor,
    ceiling: built.ceiling,
    masks: assigned,
    grid,
    worldCloud: ctx.W,
    diag,
    timings,
    hp,
    ceilingY,
    confidenceBands: CONFIDENCE_BANDS,
    version: GEOMETRY_VERSION,
  };
}

/**
 * Everything up to and including wall-plane extraction: grid, intrinsics,
 * point cloud, normals, floor/ceiling, gravity, metric scale, planes.
 */
export function prepareGeometry(input, opts = {}) {
  const quality = QUALITY[opts.quality] ? opts.quality : 'balanced';
  const Q = QUALITY[quality];
  const roomHeight = opts.roomHeight ?? 2.7;
  const timings = {};
  const tick = (k, t) => { timings[k] = Math.round(performance.now() - t); };
  const { w, h } = input;
  const cx = w / 2; const cy = h / 2;
  const diag = {};
  const dbg = opts.debugWriter ?? null;

  // --- grid ------------------------------------------------------------------
  let t = performance.now();
  const grid = buildGrid({ ...input, gridW: Q.gridW });
  if (!input.depth.metric) {
    const f0 = input.levelCam?.f ?? (w / 2) / Math.tan(rad(CAMERA.defaultHfov) / 2);
    const al = alignRelativeDepth(grid, { f: f0, cy, camHeight: input.levelCam?.height ?? 1.5 });
    diag.depthAlignment = al;
    if (!al.ok) return fail(`relative depth could not be aligned: ${al.reason}`);
  }
  const edges = depthEdges(grid);
  tick('grid', t);

  // --- intrinsics --------------------------------------------------------------
  t = performance.now();
  const intr = estimateIntrinsics(input, grid, edges, Q, opts);
  const { hfov, focalSource } = intr;
  diag.intrinsics = intr.diag;
  const f = (w / 2) / Math.tan(rad(hfov) / 2);
  const cam = { f, cx, cy, w, h, hfov };
  tick('intrinsics', t);

  // --- point cloud + normals ---------------------------------------------------
  t = performance.now();
  const P = backProject(grid, cam);
  const N = computeNormals(grid, P, edges, { step: 2, smooth: 1 });
  tick('pointCloud', t);

  // --- floor / ceiling / gravity -----------------------------------------------
  t = performance.now();
  const hp = horizontalPlanes(grid, P);
  let up = null; let upSource = null; let camHeight = null;
  if (hp.floor) {
    up = hp.floor.n; upSource = 'floor';
    camHeight = hp.floor.d;
    if (hp.ceiling && Math.acos(Math.min(1, -dot(hp.ceiling.n, up))) < rad(5)) {
      up = unit([up[0] - hp.ceiling.n[0], up[1] - hp.ceiling.n[1], up[2] - hp.ceiling.n[2]]);
      upSource = 'floor+ceiling';
    }
  } else if (hp.ceiling) {
    up = hp.ceiling.n.map((v) => -v); upSource = 'ceiling';
  }
  if (!up) { up = [0, -1, 0]; upSource = 'level-assumption'; }
  // Snap a near-level camera to level: interior photos usually are, and it
  // keeps verticals exactly vertical in the quads.
  const tilt = deg(Math.acos(Math.min(1, -up[1])));
  if (tilt < CAMERA.levelSnapDeg) { up = [0, -1, 0]; upSource += ' (snapped level)'; }

  // --- metric scale ------------------------------------------------------------
  // Monocular "metric" depth gets the scale of a room wrong by tens of percent
  // (measured: ceilings at 4-7 m in ordinary living rooms). The ratio of the
  // camera's height to the ceiling's is scale-free and well measured, so when
  // both planes are visible the room-height prior fixes the scale -- the same
  // prior the junction scanner used, applied to far better geometry.
  const legacyH = input.levelCam?.height ?? 1.5;
  let ceilAbove = hp.ceiling ? Math.abs(hp.ceiling.d) : null;
  if (ceilAbove && Math.acos(Math.min(1, Math.abs(dot(hp.ceiling.n, up)))) > rad(8)) ceilAbove = null;
  let s = 1;
  let scaleSource;
  if (camHeight && ceilAbove) {
    s = roomHeight / (camHeight + ceilAbove);
    scaleSource = `room-height prior (${roomHeight} m) from floor and ceiling planes`;
  } else if (camHeight && camHeight > CAMERA.plausibleCamHeight[0] && camHeight < CAMERA.plausibleCamHeight[1]) {
    scaleSource = input.depth.metric ? 'metric depth' : 'aligned relative depth';
  } else if (camHeight) {
    s = legacyH / camHeight;
    scaleSource = `camera-height prior (${legacyH.toFixed(2)} m)`;
  } else if (ceilAbove) {
    s = (roomHeight - legacyH) / ceilAbove;
    scaleSource = 'ceiling plane + camera-height prior';
  } else {
    scaleSource = input.depth.metric ? 'metric depth (no floor or ceiling seen)' : 'aligned relative depth';
  }
  if (s !== 1) {
    for (let i = 0; i < grid.Z.length; i++) grid.Z[i] *= s;
    Object.assign(P, backProject(grid, cam));
  }
  camHeight = camHeight ? camHeight * s : legacyH;
  const ceilingY = ceilAbove ? camHeight + ceilAbove * s : null;
  diag.scale = { factor: +s.toFixed(4), source: scaleSource };

  const frame = worldFrame(up, camHeight);
  const W = worldCloud(frame, P, N);
  tick('horizontalPlanes', t);

  // --- walls -------------------------------------------------------------------
  t = performance.now();
  const ex = extractWalls(grid, W, { seed: 1234 });
  diag.planes = ex.diag;
  tick('planeFitting', t);

  if (dbg) {
    dbg.classes(grid);
    dbg.depth(grid);
    dbg.edges(grid, edges);
    dbg.normals(grid, W);
  }

  const camera = {
    focalPx: +f.toFixed(2),
    hfov: +hfov.toFixed(2),
    focalSource,
    principalPoint: [cx, cy],
    height: +camHeight.toFixed(3),
    heightSource: hp.floor ? 'floor-plane' : 'legacy',
    scaleSource,
    pitchDeg: +frame.pitch.toFixed(2),
    rollDeg: +frame.roll.toFixed(2),
    upSource,
    up: frame.up.map((v) => +v.toFixed(5)),
    ceilingHeight: ceilingY ? +ceilingY.toFixed(3) : null,
    frame: 'world: X right, Y up (floor at Y=0), Z forward-horizontal; origin on the floor below the camera',
  };

  const context = {
    ok: true, input, grid, edges, cam, frame, W, P, N, hp, ceilingY, roomHeight, quality, Q, timings, diag, dbg, ex, camera,
  };
  if (!ex.walls.length) return fail(ex.diag.reason ?? 'no wall plane passed validation', context);
  return context;

  function fail(reason, ctx = null) {
    return { ok: false, reason, diag, timings, grid, context: ctx };
  }
}

/**
 * Focal length, in order of trust:
 *   1. an explicit hfov from the caller (the Studio knows its camera)
 *   2. what the depth model measured (Depth Pro) or EXIF recorded
 *   3. vanishing points of the photo's line segments
 *   4. the configured default
 * The depth-orthogonality search is computed for diagnostics only: monocular
 * depth compression biases it towards wide lenses (see vanishing.js).
 */
export function estimateIntrinsics(input, grid, edges, Q, opts = {}) {
  const { w, h } = input;
  const diag = {};
  if (opts.hfov) return { hfov: opts.hfov, focalSource: opts.hfovSource ?? 'caller', diag };
  if (input.depth.focalPx) {
    return {
      hfov: deg(2 * Math.atan((w / 2) / input.depth.focalPx)),
      focalSource: input.depth.focalSource ?? 'depth-model',
      diag,
    };
  }
  if (input.depth.lines?.length) {
    const vp = vanishingPoints(input.depth.lines, w, h, { hfovRange: CAMERA.hfovRange });
    diag.vanishingPoints = {
      vps: vp.vps, candidates: vp.candidates?.map((c) => ({ hfov: +c.hfov.toFixed(1), pair: c.pair })),
    };
    if (vp.focalPx) return { hfov: vp.hfov, focalSource: `vanishing-points (${vp.method})`, diag, vp };
  }
  if (Q.focalSearch && opts.diagnoseFocal) {
    const sc = selfCalibrateFocal(grid, edges, { w, cx: w / 2, cy: h / 2 });
    diag.depthOrthogonality = sc && { hfov: +sc.hfov.toFixed(1), residual: +sc.verticalityResidual.toFixed(4) };
  }
  return { hfov: CAMERA.defaultHfov, focalSource: 'default', diag };
}
