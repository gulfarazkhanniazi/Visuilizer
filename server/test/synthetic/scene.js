/**
 * Synthetic rooms with exact ground truth.
 *
 * A tiny ray tracer: walls (vertical rectangles), floor, ceiling, boxes
 * (furniture) and decorations (windows, paintings, doors) seen by a pinhole
 * camera with any pitch and yaw. It renders what the scanner's inputs would
 * be -- a class map, a luminance image and a depth map -- plus the ground
 * truth the scanner is scored against: which wall every pixel belongs to,
 * each wall's plane, and where every corner is.
 *
 * The depth is then corrupted the way monocular networks corrupt it (smooth
 * low-frequency bias, pixel noise, blurred silhouettes, a wrong global
 * scale), so the tests exercise robustness, not just arithmetic.
 */
import { CLS } from '../../src/scanner/pointcloud.js';
import { rng } from '../../src/scanner/math.js';

const rad = (d) => (d * Math.PI) / 180;

/**
 * @param spec {
 *   w, h, hfov, camHeight, pitch, yaw,
 *   roomHeight, ceiling: bool,
 *   walls: [{ id, a:[x,z], b:[x,z], albedo }],   (base segment; wall runs 0..roomHeight)
 *   boxes: [{ min:[x,y,z], max:[x,y,z], albedo, cls }],
 *   decos: [{ wall, s:[s0,s1], y:[y0,y1], offset, albedo, cls, hole, behind }],
 *   shadows: [{ wall, s:[s0,s1], y:[y0,y1], factor, slant }],
 *   light: 'directional' | 'flat', brightness
 * }
 */
export function renderScene(spec) {
  const {
    w = 480, h = 360, hfov = 70, camHeight = 1.4, pitch = 0, yaw = 0,
    roomHeight = 2.7, ceiling = true, walls = [], boxes = [], decos = [], shadows = [],
    light = 'directional', brightness = 1,
  } = spec;
  const f = (w / 2) / Math.tan(rad(hfov) / 2);
  const cx = w / 2; const cy = h / 2;
  const th = rad(pitch); const ps = rad(yaw);
  const fwd = [Math.sin(ps) * Math.cos(th), Math.sin(th), Math.cos(ps) * Math.cos(th)];
  const right = [Math.cos(ps), 0, -Math.sin(ps)];
  const down = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  const C = [0, camHeight, 0];
  const L = normalize([0.45, 0.75, -0.5]);

  // Precompute wall geometry: unit tangent, normal facing the room centre.
  const W = walls.map((wl) => {
    const dx = wl.b[0] - wl.a[0]; const dz = wl.b[1] - wl.a[1];
    const len = Math.hypot(dx, dz);
    const t = [dx / len, dz / len];
    let n = [-t[1], t[0]];
    // Face the camera.
    const toCam = [C[0] - wl.a[0], C[2] - wl.a[1]];
    if (n[0] * toCam[0] + n[1] * toCam[1] < 0) n = [-n[0], -n[1]];
    return { ...wl, t, n, len, top: wl.top ?? roomHeight, o: n[0] * wl.a[0] + n[1] * wl.a[1] };
  });

  const n = w * h;
  const depth = new Float32Array(n);
  const cls = new Uint8Array(n);
  const wallId = new Int16Array(n).fill(-1);
  const luma = new Float32Array(n);
  // Which wall's plane lies behind each object pixel (for occlusion truth).
  const behindWall = new Int16Array(n).fill(-1);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = (x + 0.5 - cx) / f; const b = (y + 0.5 - cy) / f;
      const d = [right[0] * a + down[0] * b + fwd[0], right[1] * a + down[1] * b + fwd[1], right[2] * a + down[2] * b + fwd[2]];
      let best = { t: Infinity };
      const consider = (t, info) => { if (t > 1e-4 && t < best.t) best = { t, ...info }; };

      if (d[1] < 0) consider(-C[1] / d[1], { kind: 'floor', normal: [0, 1, 0], albedo: 0.55 });
      if (ceiling && d[1] > 0) consider((roomHeight - C[1]) / d[1], { kind: 'ceiling', normal: [0, -1, 0], albedo: 0.9 });
      let wallHit = { t: Infinity, k: -1 };
      W.forEach((wl, k) => {
        const den = wl.n[0] * d[0] + wl.n[1] * d[2];
        if (Math.abs(den) < 1e-9) return;
        const t = (wl.o - (wl.n[0] * C[0] + wl.n[1] * C[2])) / den;
        if (!(t > 1e-4)) return;
        const X = C[0] + d[0] * t; const Z = C[2] + d[2] * t; const Y = C[1] + d[1] * t;
        const s = (X - wl.a[0]) * wl.t[0] + (Z - wl.a[1]) * wl.t[1];
        if (s < 0 || s > wl.len || Y < 0 || Y > wl.top) return;
        // Holes (door / window openings) let the ray through.
        if (decos.some((dc) => dc.wall === wl.id && dc.hole && s >= dc.s[0] && s <= dc.s[1] && Y >= dc.y[0] && Y <= dc.y[1])) return;
        if (t < wallHit.t) wallHit = { t, k };
        consider(t, { kind: 'wall', k, s, Y, normal: [wl.n[0], 0, wl.n[1]], albedo: wl.albedo ?? 0.8 });
      });
      for (const dc of decos) {
        const wl = W.find((q) => q.id === dc.wall);
        const off = dc.offset ?? 0.02;
        // Plane parallel to the wall, `off` metres towards the room (negative:
        // recessed behind it, like a door leaf in its opening).
        const oo = wl.n[0] * (wl.a[0] + wl.n[0] * off) + wl.n[1] * (wl.a[1] + wl.n[1] * off);
        const den = wl.n[0] * d[0] + wl.n[1] * d[2];
        if (Math.abs(den) < 1e-9) continue;
        const t = (oo - (wl.n[0] * C[0] + wl.n[1] * C[2])) / den;
        if (!(t > 1e-4)) continue;
        const X = C[0] + d[0] * t; const Z = C[2] + d[2] * t; const Y = C[1] + d[1] * t;
        const s = (X - wl.a[0] - wl.n[0] * off) * wl.t[0] + (Z - wl.a[1] - wl.n[1] * off) * wl.t[1];
        if (s < dc.s[0] || s > dc.s[1] || Y < dc.y[0] || Y > dc.y[1]) continue;
        consider(t, { kind: 'deco', cls: dc.cls ?? CLS.WALL_OBJECT, normal: [wl.n[0], 0, wl.n[1]], albedo: dc.albedo ?? 0.3, behind: W.indexOf(wl), emissive: dc.emissive });
      }
      for (const bx of boxes) {
        const hit = rayBox(C, d, bx.min, bx.max);
        if (hit) consider(hit.t, { kind: 'box', cls: bx.cls ?? CLS.FLOOR_OBJECT, normal: hit.normal, albedo: bx.albedo ?? 0.35 });
      }

      const i = y * w + x;
      if (!Number.isFinite(best.t)) { depth[i] = 30; cls[i] = CLS.OTHER; luma[i] = 0.9; continue; }
      depth[i] = best.t;
      let shade = light === 'flat' ? 1 : 0.55 + 0.45 * Math.max(0, dot3(best.normal, L));
      let alb = best.albedo;
      if (best.emissive) { alb = best.emissive; shade = 1; }
      if (best.kind === 'wall') {
        cls[i] = CLS.WALL; wallId[i] = best.k;
        for (const sh of shadows) {
          if (sh.wall !== W[best.k].id) continue;
          const sOff = best.s - (sh.slant ?? 0) * best.Y;
          if (sOff >= sh.s[0] && sOff <= sh.s[1] && best.Y >= sh.y[0] && best.Y <= sh.y[1]) shade *= sh.factor ?? 0.45;
        }
      } else if (best.kind === 'floor') cls[i] = CLS.FLOOR;
      else if (best.kind === 'ceiling') cls[i] = CLS.CEILING;
      else {
        cls[i] = best.cls;
        if (wallHit.k >= 0 && wallHit.t >= best.t - 1e-6) behindWall[i] = wallHit.k;
      }
      luma[i] = Math.min(1, alb * shade * brightness);
    }
  }

  const project = (X, Y, Z) => {
    const p = [X - C[0], Y - C[1], Z - C[2]];
    const zc = dot3(p, fwd);
    if (zc <= 1e-3) return null;
    return [cx + (f * dot3(p, right)) / zc, cy + (f * dot3(p, down)) / zc];
  };
  // Ground-truth corners: every pair of walls sharing an endpoint, plus any
  // declared step corners (parallel walls joined by a return you cannot see).
  const corners = [];
  for (const sc of spec.stepCorners ?? []) {
    corners.push({
      walls: sc.walls, xz: sc.xz, parallel: true, angle: null,
      seg2D: [project(sc.xz[0], 0, sc.xz[1]), project(sc.xz[0], roomHeight, sc.xz[1])],
    });
  }
  for (let i = 0; i < W.length; i++) {
    for (let j = 0; j < W.length; j++) {
      if (i === j) continue;
      const A = W[i]; const B = W[j];
      const shared = [A.b, A.a].find((p) => [B.a, B.b].some((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6));
      if (!shared || i > j) continue;
      const cosN = A.n[0] * B.n[0] + A.n[1] * B.n[1];
      corners.push({
        walls: [A.id, B.id],
        xz: shared,
        parallel: Math.abs(cosN) > Math.cos(rad(12)),
        angle: 180 - (Math.acos(Math.max(-1, Math.min(1, cosN))) * 180) / Math.PI,
        seg2D: [project(shared[0], 0, shared[1]), project(shared[0], Math.min(A.top, B.top), shared[1])],
      });
    }
  }
  // What a line-segment detector would return: the visible runs of every
  // straight 3D edge -- wall/floor, wall/ceiling, wall/wall, decoration
  // frames -- projected, with half a pixel of endpoint noise.
  const rand = rng(17);
  const lines = [];
  const edge3D = (P, Q) => {
    const N = 80; let run = null;
    const flush = () => {
      if (run && Math.hypot(run[2] - run[0], run[3] - run[1]) > 0.03 * w) {
        lines.push(run.map((v) => v + (rand() - 0.5)));
      }
      run = null;
    };
    for (let k = 0; k <= N; k++) {
      const X = P[0] + ((Q[0] - P[0]) * k) / N; const Y = P[1] + ((Q[1] - P[1]) * k) / N; const Z = P[2] + ((Q[2] - P[2]) * k) / N;
      const p = project(X, Y, Z);
      let vis = false;
      if (p && p[0] >= 0 && p[1] >= 0 && p[0] < w && p[1] < h) {
        const zc = dot3([X - C[0], Y - C[1], Z - C[2]], fwd);
        // Visible if nothing is clearly in front of it.
        const i = Math.min(h - 1, Math.floor(p[1])) * w + Math.min(w - 1, Math.floor(p[0]));
        vis = depth[i] > zc * 0.97;
      }
      if (vis) { if (!run) run = [p[0], p[1], p[0], p[1]]; else { run[2] = p[0]; run[3] = p[1]; } } else flush();
    }
    flush();
  };
  for (const wl of W) {
    const a = [wl.a[0], 0, wl.a[1]]; const b = [wl.b[0], 0, wl.b[1]];
    edge3D(a, b);
    edge3D([a[0], wl.top, a[2]], [b[0], wl.top, b[2]]);
    edge3D(a, [a[0], wl.top, a[2]]);
    edge3D(b, [b[0], wl.top, b[2]]);
  }
  for (const dc of decos) {
    const wl = W.find((q) => q.id === dc.wall);
    const off = (dc.offset ?? 0.02) * (dc.hole ? 0 : 1);
    const at = (s, Y) => [wl.a[0] + wl.t[0] * s + wl.n[0] * off, Y, wl.a[1] + wl.t[1] * s + wl.n[1] * off];
    edge3D(at(dc.s[0], dc.y[0]), at(dc.s[1], dc.y[0]));
    edge3D(at(dc.s[0], dc.y[1]), at(dc.s[1], dc.y[1]));
    edge3D(at(dc.s[0], dc.y[0]), at(dc.s[0], dc.y[1]));
    edge3D(at(dc.s[1], dc.y[0]), at(dc.s[1], dc.y[1]));
  }
  // Floor tile joints every 0.6 m in both directions: the floorboards and
  // tile grids that give a real photograph most of its line segments.
  if (spec.floorGrid !== false) {
    const xs = W.flatMap((wl) => [wl.a[0], wl.b[0]]); const zs = W.flatMap((wl) => [wl.a[1], wl.b[1]]);
    const [xa, xb] = [Math.min(...xs), Math.max(...xs)]; const [za, zb] = [Math.max(0.3, Math.min(...zs)), Math.max(...zs)];
    for (let x = Math.ceil(xa / 0.6) * 0.6; x < xb; x += 0.6) edge3D([x, 0.001, za], [x, 0.001, zb]);
    for (let z = Math.ceil(za / 0.6) * 0.6; z < zb; z += 0.6) edge3D([xa, 0.001, z], [xb, 0.001, z]);
  }
  for (const bx of boxes) {
    const [x0, y0, z0] = bx.min; const [x1, y1, z1] = bx.max;
    edge3D([x0, y1, z0], [x1, y1, z0]); edge3D([x0, y0, z0], [x0, y1, z0]); edge3D([x1, y0, z0], [x1, y1, z0]);
  }

  return {
    w, h, f, cx, cy, depth, cls, wallId, luma, behindWall, lines,
    walls: W.map((wl) => ({ id: wl.id, normal: [wl.n[0], 0, wl.n[1]], o: wl.o, len: wl.len })),
    corners,
    camera: { hfov, camHeight, pitch, yaw, roomHeight },
  };
}

/**
 * Corrupt a depth map like a monocular network: smooth multiplicative bias,
 * pixel noise, blurred silhouettes, and a global scale error.
 */
export function corruptDepth(scene, {
  bias = 0.02, noise = 0.004, blur = 1, scale = 1, seed = 5,
} = {}) {
  const { w, h, depth } = scene;
  const rand = rng(seed);
  const p1 = rand() * 6.28; const p2 = rand() * 6.28;
  let out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const b = bias * Math.sin((2 * Math.PI * x) / w * 1.3 + p1) * Math.cos((2 * Math.PI * y) / h * 0.9 + p2);
      const g = (rand() + rand() + rand() - 1.5) * 2 * noise;
      out[i] = depth[i] * scale * (1 + b + g);
    }
  }
  for (let k = 0; k < blur; k++) {
    const next = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0; let c = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx; const yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            s += out[yy * w + xx]; c++;
          }
        }
        next[y * w + x] = s / c;
      }
    }
    out = next;
  }
  return out;
}

/** Build scanGeometry's input from a rendered scene. */
export function scanInput(scene, depthData, extra = {}) {
  const { w, h, cls } = scene;
  const n = w * h;
  const wall = new Uint8Array(n); const floor = new Uint8Array(n); const ceilingM = new Uint8Array(n);
  const objW = new Float32Array(n); const objF = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (cls[i] === CLS.WALL) wall[i] = 255;
    else if (cls[i] === CLS.FLOOR) floor[i] = 255;
    else if (cls[i] === CLS.CEILING) ceilingM[i] = 255;
    else if (cls[i] === CLS.WALL_OBJECT) objW[i] = 1;
    else if (cls[i] === CLS.FLOOR_OBJECT) objF[i] = 1;
  }
  return {
    w, h, RW: w, RH: h,
    masks: { wall, floor, ceiling: ceilingM },
    objects: { wall: objW, floor: objF },
    luma: scene.luma,
    depth: { data: depthData, width: w, height: h, metric: true, lines: extra.noLines ? [] : scene.lines, ...extra.depth },
    levelCam: { f: (w / 2) / Math.tan(rad(70) / 2), cx: w / 2, cy: h / 2, height: 1.5 },
  };
}

function rayBox(o, d, mn, mx) {
  let t0 = -Infinity; let t1 = Infinity; let axis = -1; let sign = 0;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-12) {
      if (o[k] < mn[k] || o[k] > mx[k]) return null;
      continue;
    }
    let a = (mn[k] - o[k]) / d[k]; let b = (mx[k] - o[k]) / d[k];
    let s = -1;
    if (a > b) { [a, b] = [b, a]; s = 1; }
    if (a > t0) { t0 = a; axis = k; sign = s; }
    if (b < t1) t1 = b;
    if (t0 > t1) return null;
  }
  if (t0 < 1e-4) return null;
  const normal = [0, 0, 0]; normal[axis] = sign;
  return { t: t0, normal };
}

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function normalize(v) { const l = Math.hypot(...v); return v.map((x) => x / l); }
