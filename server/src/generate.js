/**
 * Procedural demo assets.
 *
 * Demo rooms are *rendered from a real pinhole camera*, so the surface quads
 * that ship with them are the mathematically exact projections of a 3.2 m room
 * -- which makes them a genuine test of the homography path rather than
 * hand-nudged guesses. Swap them for photographs whenever you have some; the
 * studio authors those the same way.
 */

// ---------------------------------------------------------------- noise ----

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function valueNoise2D(seed) {
  const rand = mulberry32(seed);
  const size = 256;
  const grid = new Float32Array(size * size);
  for (let i = 0; i < grid.length; i++) grid[i] = rand();
  const at = (x, y) => grid[(((y % size) + size) % size) * size + (((x % size) + size) % size)];
  const smooth = (t) => t * t * (3 - 2 * t);

  return (x, y) => {
    const x0 = Math.floor(x); const y0 = Math.floor(y);
    const fx = smooth(x - x0); const fy = smooth(y - y0);
    const a = at(x0, y0); const b = at(x0 + 1, y0);
    const c = at(x0, y0 + 1); const d = at(x0 + 1, y0 + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
  };
}

function fbm(noise, x, y, octaves = 5, lacunarity = 2, gain = 0.5) {
  let sum = 0; let amp = 1; let norm = 0; let f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += noise(x * f, y * f) * amp;
    norm += amp;
    amp *= gain;
    f *= lacunarity;
  }
  return sum / norm;
}

const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v | 0);
const mix = (a, b, t) => a + (b - a) * t;

// ------------------------------------------------------------- textures ----

/**
 * Generate one seamless-ish tile face as a raw RGB buffer.
 * `kind` selects the material model; `seed` gives each random face of the same
 * product its own veining so a laid floor never visibly repeats.
 */
export function generateTileFace(kind, palette, seed, size = 512) {
  const n = valueNoise2D(seed);
  const n2 = valueNoise2D(seed + 977);
  const data = Buffer.alloc(size * size * 3);
  const [base, alt, accent] = palette;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size; const v = y / size;
      let r; let g; let b;

      if (kind === 'marble') {
        // Veins: warp the coordinate with fBm, then take a sharp sine ridge.
        const warp = fbm(n, u * 3, v * 3, 5) * 2.4;
        const vein = Math.abs(Math.sin((u * 5.5 + v * 2.1 + warp) * Math.PI));
        const t = Math.pow(1 - vein, 7);
        const grain = fbm(n2, u * 22, v * 22, 3) * 0.09 - 0.045;
        r = mix(base[0], accent[0], t) + grain * 255;
        g = mix(base[1], accent[1], t) + grain * 255;
        b = mix(base[2], accent[2], t) + grain * 255;
      } else if (kind === 'wood') {
        // Growth rings along the plank plus a little cathedral wander.
        const wander = fbm(n, u * 1.4, v * 5, 4) * 0.55;
        const rings = Math.abs(Math.sin((v * 13 + wander * 6) * Math.PI));
        const t = Math.pow(rings, 1.7);
        const pore = fbm(n2, u * 90, v * 8, 2) * 0.13 - 0.065;
        r = mix(base[0], alt[0], t) + pore * 255;
        g = mix(base[1], alt[1], t) + pore * 255;
        b = mix(base[2], alt[2], t) + pore * 255;
      } else if (kind === 'terrazzo') {
        // Chips: nearest-feature scatter over a cement ground.
        const cell = 13;
        let best = 1; let bestSeed = 0;
        const cx = Math.floor(u * cell); const cy = Math.floor(v * cell);
        for (let oy = -1; oy <= 1; oy++) {
          for (let ox = -1; ox <= 1; ox++) {
            const gx = cx + ox; const gy = cy + oy;
            const jr = mulberry32(seed + gx * 7919 + gy * 104729);
            const px = (gx + jr()) / cell; const py = (gy + jr()) / cell;
            const d = Math.hypot(u - px, v - py) * cell;
            if (d < best) { best = d; bestSeed = jr(); }
          }
        }
        const chip = best < 0.42 + bestSeed * 0.22 ? 1 : 0;
        const chipCol = bestSeed > 0.66 ? accent : bestSeed > 0.33 ? alt : [40, 40, 44];
        const speck = fbm(n2, u * 60, v * 60, 2) * 0.1 - 0.05;
        r = mix(base[0], chipCol[0], chip) + speck * 255;
        g = mix(base[1], chipCol[1], chip) + speck * 255;
        b = mix(base[2], chipCol[2], chip) + speck * 255;
      } else if (kind === 'concrete') {
        const blotch = fbm(n, u * 4, v * 4, 6);
        const grain = fbm(n2, u * 70, v * 70, 3) * 0.16 - 0.08;
        const t = blotch * 0.7 + 0.15;
        r = mix(base[0], alt[0], t) + grain * 255;
        g = mix(base[1], alt[1], t) + grain * 255;
        b = mix(base[2], alt[2], t) + grain * 255;
      } else {
        // Plain glazed body with a soft glaze pool toward the edges.
        const edge = Math.min(Math.min(u, 1 - u), Math.min(v, 1 - v));
        const pool = 1 - Math.min(1, edge * 6);
        const grain = fbm(n, u * 30, v * 30, 3) * 0.05 - 0.025;
        r = mix(base[0], alt[0], pool * 0.5) + grain * 255;
        g = mix(base[1], alt[1], pool * 0.5) + grain * 255;
        b = mix(base[2], alt[2], pool * 0.5) + grain * 255;
      }

      const i = (y * size + x) * 3;
      data[i] = clamp255(r);
      data[i + 1] = clamp255(g);
      data[i + 2] = clamp255(b);
    }
  }
  return { data, info: { width: size, height: size, channels: 3 } };
}

// ----------------------------------------------------------------- room ----

/**
 * A pinhole camera looking down -Z, image plane at +f.
 * Returns pixel coordinates, which is the space every mask and quad lives in.
 */
export function makeCamera({ width, height, f, camY }) {
  const cx = width / 2;
  const cy = height / 2;
  return {
    width, height, f, camY, cx, cy,
    project(X, Y, Z) {
      const d = -Z;
      return [cx + (f * X) / d, cy - (f * (Y - camY)) / d];
    },
    /** Depth at which the floor (Y = 0) crosses a given image row. */
    depthAtFloorRow(v) {
      return (f * camY) / (v - cy);
    },
  };
}

/**
 * Build a rectangular room and return both the SVG that draws it and the exact
 * surface quads, in the [topLeft, topRight, bottomRight, bottomLeft] order the
 * homography solver expects.
 */
export function buildRoom(spec) {
  const {
    width = 1600, height = 1100, f = 1000, camY = 1.4,
    halfW = 1.6, roomH = 2.6, backZ = -5,
    palette, props = [],
  } = spec;

  const cam = makeCamera({ width, height, f, camY });
  const P = (x, y, z) => cam.project(x, y, z).map((n) => Math.round(n * 100) / 100);

  // The nearest floor depth still inside the frame: the front edge of every
  // horizontal surface quad, and the bottom of both side walls.
  const nearZ = -cam.depthAtFloorRow(height);
  const depth = round2(Math.abs(nearZ - backZ)) || 1;

  const backTL = P(-halfW, roomH, backZ);
  const backTR = P(halfW, roomH, backZ);
  const backBR = P(halfW, 0, backZ);
  const backBL = P(-halfW, 0, backZ);

  const nearTL = P(-halfW, roomH, nearZ);
  const nearTR = P(halfW, roomH, nearZ);
  const nearBL = P(-halfW, 0, nearZ);
  const nearBR = P(halfW, 0, nearZ);

  const surfaces = [
    {
      name: 'floor',
      label: 'Floor',
      product_surface: 'floor',
      quad: [backBL, backBR, nearBR, nearBL],
      realSize: { w: halfW * 2, h: depth },
    },
    {
      name: 'back_wall',
      label: 'Back Wall',
      product_surface: 'wall',
      quad: [backTL, backTR, backBR, backBL],
      realSize: { w: halfW * 2, h: roomH },
    },
    {
      name: 'left_wall',
      label: 'Left Wall',
      product_surface: 'wall',
      quad: [backTL, nearTL, nearBL, backBL],
      realSize: { w: depth, h: roomH },
    },
    {
      name: 'right_wall',
      label: 'Right Wall',
      product_surface: 'wall',
      quad: [backTR, nearTR, nearBR, backBR],
      realSize: { w: depth, h: roomH },
    },
    {
      name: 'ceiling',
      label: 'Ceiling',
      product_surface: 'ceiling',
      quad: [backTL, backTR, nearTR, nearTL],
      realSize: { w: halfW * 2, h: depth },
    },
  ];

  return { cam, width, height, surfaces, nearZ, depth, palette, props, spec };
}

const poly = (pts) => pts.map((p) => `${p[0]},${p[1]}`).join(' ');

/** SVG for a room: flat planes, a window, its light spill, and some props. */
export function roomSvg(room) {
  const { cam, width, height, palette } = room;
  const s = Object.fromEntries(room.surfaces.map((x) => [x.name, x]));
  const P = (x, y, z) => cam.project(x, y, z);
  const { halfW = 1.6, roomH = 2.6, backZ = -5 } = room.spec;

  const win = room.spec.window ?? { z0: -4.4, z1: -3.2, y0: 0.95, y1: 2.15, side: 'right' };
  const wx = win.side === 'right' ? halfW - 0.001 : -halfW + 0.001;
  const winQuad = [
    P(wx, win.y1, win.z0), P(wx, win.y1, win.z1),
    P(wx, win.y0, win.z1), P(wx, win.y0, win.z0),
  ];

  // Light pool: where the window's beam lands on the floor, roughly.
  const spread = win.side === 'right' ? -1 : 1;
  const pool = [
    P(wx, 0, win.z0), P(wx, 0, win.z1),
    P(wx + spread * 2.4, 0, win.z1 + 0.9), P(wx + spread * 2.4, 0, win.z0 + 0.9),
  ];

  const propSvg = (room.props ?? []).map((p) => renderProp(P, p, palette)).join('\n');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <linearGradient id="wallGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${palette.wallTop}"/>
      <stop offset="100%" stop-color="${palette.wallBottom}"/>
    </linearGradient>
    <linearGradient id="floorGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${palette.floorFar}"/>
      <stop offset="100%" stop-color="${palette.floorNear}"/>
    </linearGradient>
    <radialGradient id="pool" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="winGlow" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#dbeafe" stop-opacity="0.75"/>
    </linearGradient>
    <radialGradient id="vignette" cx="50%" cy="48%" r="72%">
      <stop offset="55%" stop-color="#000000" stop-opacity="0"/>
      <stop offset="100%" stop-color="#000000" stop-opacity="0.30"/>
    </radialGradient>
    <filter id="soft"><feGaussianBlur stdDeviation="26"/></filter>
    <filter id="softer"><feGaussianBlur stdDeviation="52"/></filter>
    <filter id="contact"><feGaussianBlur stdDeviation="9"/></filter>
  </defs>

  <rect width="${width}" height="${height}" fill="${palette.wallBottom}"/>
  <polygon points="${poly(s.ceiling.quad)}" fill="${palette.ceiling}"/>
  <polygon points="${poly(s.back_wall.quad)}" fill="url(#wallGrad)"/>
  <polygon points="${poly(s.left_wall.quad)}" fill="${palette.wallSide}"/>
  <polygon points="${poly(s.right_wall.quad)}" fill="${palette.wallSideLit}"/>
  <polygon points="${poly(s.floor.quad)}" fill="url(#floorGrad)"/>

  <!-- window and the light it throws -->
  <polygon points="${poly(winQuad)}" fill="url(#winGlow)"/>
  <polygon points="${poly(winQuad)}" fill="none" stroke="${palette.trim}" stroke-width="6"/>
  <polygon points="${poly(pool)}" fill="url(#pool)" filter="url(#soft)"/>

  <!-- ambient occlusion where planes meet -->
  <polygon points="${poly(s.floor.quad)}" fill="none" stroke="#000" stroke-opacity="0.30" stroke-width="26" filter="url(#soft)"/>
  <polygon points="${poly(s.ceiling.quad)}" fill="none" stroke="#000" stroke-opacity="0.16" stroke-width="30" filter="url(#soft)"/>
  <polygon points="${poly(s.back_wall.quad)}" fill="none" stroke="#000" stroke-opacity="0.14" stroke-width="34" filter="url(#softer)"/>

  ${propSvg}

  <!-- overall vignette -->
  <rect width="${width}" height="${height}" fill="url(#vignette)"/>
</svg>`;
}

/**
 * Props are simple boxes placed in world space. They matter because they give
 * the masks something real to cut around and cast the contact shadows that the
 * relighting pass later reproduces under the new tile.
 */
function renderProp(P, prop, palette) {
  const { x, z, w, d, h, color = palette.prop, type = 'box' } = prop;
  const x0 = x - w / 2; const x1 = x + w / 2;
  const z0 = z - d / 2; const z1 = z + d / 2;

  const footprint = [P(x0, 0, z0), P(x1, 0, z0), P(x1, 0, z1), P(x0, 0, z1)];
  const shadow = [
    P(x0 - 0.16, 0, z0 - 0.1), P(x1 + 0.22, 0, z0 - 0.1),
    P(x1 + 0.3, 0, z1 + 0.18), P(x0 - 0.2, 0, z1 + 0.18),
  ];

  if (type === 'rug') {
    return `<polygon points="${poly(shadow)}" fill="#000" fill-opacity="0.22" filter="url(#contact)"/>
    <polygon points="${poly(footprint)}" fill="${color}"/>
    <polygon points="${poly(footprint)}" fill="none" stroke="#000" stroke-opacity="0.18" stroke-width="4"/>`;
  }

  const front = [P(x0, h, z1), P(x1, h, z1), P(x1, 0, z1), P(x0, 0, z1)];
  const top = [P(x0, h, z0), P(x1, h, z0), P(x1, h, z1), P(x0, h, z1)];
  const side = [P(x0, h, z0), P(x0, h, z1), P(x0, 0, z1), P(x0, 0, z0)];

  return `<polygon points="${poly(shadow)}" fill="#000" fill-opacity="0.34" filter="url(#contact)"/>
  <polygon points="${poly(side)}" fill="${shade(color, -18)}"/>
  <polygon points="${poly(top)}" fill="${shade(color, 16)}"/>
  <polygon points="${poly(front)}" fill="${color}"/>`;
}

function shade(hex, amount) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return hex;
  const c = [1, 2, 3].map((i) => clamp255(parseInt(m[i], 16) + amount));
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Cut every prop out of every surface it stands in front of, and clip each
 * surface to the frame.
 *
 * A prop occludes whatever is behind it, whichever plane that happens to be --
 * a sofa hides floor AND the wall behind it -- so the cut-out is the box's
 * silhouette: the convex hull of its eight projected corners. This is exactly
 * the work the studio does interactively; doing it here keeps the seeded rooms
 * honest rather than pretending the furniture is flat.
 */
export function surfaceMasks(room) {
  const { cam, width, height } = room;
  const P = (x, y, z) => cam.project(x, y, z);
  const clipToFrame = (pts) => pts.map(([x, y]) => [
    Math.max(-40, Math.min(width + 40, x)),
    Math.max(-40, Math.min(height + 40, y)),
  ]);

  const cutouts = (room.props ?? []).map((p) => {
    const x0 = p.x - p.w / 2; const x1 = p.x + p.w / 2;
    const z0 = p.z - p.d / 2; const z1 = p.z + p.d / 2;
    const corners = [];
    for (const x of [x0, x1]) {
      for (const z of [z0, z1]) {
        for (const y of [0, p.h]) corners.push(P(x, y, z));
      }
    }
    return {
      mode: 'subtract',
      points: convexHull(corners).map((q) => [round2(q[0]), round2(q[1])]),
    };
  });

  const masks = {};
  for (const s of room.surfaces) {
    const add = { mode: 'add', points: clipToFrame(s.quad).map((q) => [round2(q[0]), round2(q[1])]) };
    // Nothing in these rooms floats, so the ceiling is never occluded.
    masks[s.name] = {
      feather: 1.2,
      polygons: s.name === 'ceiling' ? [add] : [add, ...cutouts],
    };
  }
  return masks;
}

/** Andrew's monotone chain, returning the hull counter-clockwise. */
function convexHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const build = (list) => {
    const out = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...build(pts), ...build([...pts].reverse())];
}

const round2 = (n) => Math.round(n * 100) / 100;

// -------------------------------------------------------------- panorama ----

/**
 * Render an equirectangular panorama of a box room by ray-casting.
 *
 * Camera sits at the origin, so the panorama and the surface planes share one
 * coordinate frame -- which is what lets the 360 renderer find each surface by
 * intersection instead of by a hand-drawn mask.
 */
export function buildPanorama(spec) {
  const {
    width = 2048, height = 1024,
    halfW = 2.2, halfD = 2.8, roomH = 2.7, camY = 1.5,
    palette, window: win = { wall: 'right', u0: -0.9, u1: 0.9, v0: 0.9, v1: 2.2 },
    seed = 7,
  } = spec;

  const n = valueNoise2D(seed);
  const data = Buffer.alloc(width * height * 3);

  const floorY = -camY;
  const ceilY = roomH - camY;

  // Named faces in the same frame the shader uses.
  const faces = [
    { name: 'floor',      axis: 'y', at: floorY },
    { name: 'ceiling',    axis: 'y', at: ceilY },
    { name: 'wall_back',  axis: 'z', at: -halfD },
    { name: 'wall_front', axis: 'z', at: halfD },
    { name: 'wall_left',  axis: 'x', at: -halfW },
    { name: 'wall_right', axis: 'x', at: halfW },
  ];

  const inside = (name, p) => {
    if (name === 'floor' || name === 'ceiling') {
      return Math.abs(p[0]) <= halfW + 1e-6 && Math.abs(p[2]) <= halfD + 1e-6;
    }
    if (name.startsWith('wall_b') || name.startsWith('wall_f')) {
      return Math.abs(p[0]) <= halfW + 1e-6 && p[1] >= floorY - 1e-6 && p[1] <= ceilY + 1e-6;
    }
    return Math.abs(p[2]) <= halfD + 1e-6 && p[1] >= floorY - 1e-6 && p[1] <= ceilY + 1e-6;
  };

  // Window centre in world space, for the light it throws.
  const winCentre = win.wall === 'right'
    ? [halfW, floorY + (win.v0 + win.v1) / 2, (win.u0 + win.u1) / 2]
    : [-halfW, floorY + (win.v0 + win.v1) / 2, (win.u0 + win.u1) / 2];

  for (let y = 0; y < height; y++) {
    // Standard equirectangular: row 0 is the top of the sphere (straight up).
    // three flips the texture on load, so the shader's v = 0 lands on the
    // bottom row and the two conventions line up.
    const lat = (0.5 - (y + 0.5) / height) * Math.PI;
    const cl = Math.cos(lat);
    const dy = Math.sin(lat);

    for (let x = 0; x < width; x++) {
      const lon = ((x + 0.5) / width - 0.5) * 2 * Math.PI;
      const dx = cl * Math.sin(lon);
      const dz = -cl * Math.cos(lon);

      let bestT = Infinity;
      let bestFace = null;
      let bestP = null;
      for (const f of faces) {
        const d = f.axis === 'x' ? dx : f.axis === 'y' ? dy : dz;
        if (Math.abs(d) < 1e-9) continue;
        const t = f.at / d;
        if (t <= 1e-4 || t >= bestT) continue;
        const p = [dx * t, dy * t, dz * t];
        if (!inside(f.name, p)) continue;
        bestT = t; bestFace = f; bestP = p;
      }

      const i = (y * width + x) * 3;
      if (!bestFace) { data[i] = 20; data[i + 1] = 20; data[i + 2] = 24; continue; }

      let col;
      if (bestFace.name === 'floor') col = palette.floor;
      else if (bestFace.name === 'ceiling') col = palette.ceiling;
      else col = palette.wall;

      // Window: a bright rectangle on one wall, plus the light it spills.
      let light = 1;
      const onWinWall = (win.wall === 'right' && bestFace.name === 'wall_right')
        || (win.wall === 'left' && bestFace.name === 'wall_left');
      if (onWinWall) {
        const u = bestP[2];
        const v = bestP[1] - floorY;
        if (u >= win.u0 && u <= win.u1 && v >= win.v0 && v <= win.v1) {
          col = [246, 250, 255];
          light = 1.15;
        }
      }

      // Falloff from the window, so the room has a light direction. Kept well
      // under 1.15 so nothing but the window itself ever clips.
      const dist = Math.hypot(bestP[0] - winCentre[0], bestP[1] - winCentre[1], bestP[2] - winCentre[2]);
      light *= 0.62 + 0.42 / (1 + dist * dist * 0.22);

      // Corners darken; ceilings stay bright.
      if (bestFace.name === 'floor') light *= 0.86;
      if (bestFace.name === 'ceiling') light *= 1.12;

      const grain = fbm(n, bestP[0] * 3 + 10, bestP[2] * 3 + 10, 4) * 0.1 + 0.95;
      light *= grain;

      data[i] = clamp255(col[0] * light);
      data[i + 1] = clamp255(col[1] * light);
      data[i + 2] = clamp255(col[2] * light);
    }
  }

  const surfaces = [
    {
      name: 'floor', label: 'Floor', product_surface: 'floor', isMain: true,
      plane: {
        origin: [0, floorY, 0], normal: [0, 1, 0],
        axisU: [1, 0, 0], axisV: [0, 0, 1],
        extent: [-halfW, halfW, -halfD, halfD],
      },
      defaults: { tileSize: { w: 600, h: 600 }, layout: 'grid', grout: { size: 2, color: '#c9c9c4' } },
    },
    {
      name: 'ceiling', label: 'Ceiling', product_surface: 'ceiling',
      plane: {
        origin: [0, ceilY, 0], normal: [0, 1, 0],
        axisU: [1, 0, 0], axisV: [0, 0, 1],
        extent: [-halfW, halfW, -halfD, halfD],
      },
      defaults: { tileSize: { w: 600, h: 600 }, layout: 'grid', grout: { size: 2, color: '#f5f5f2' } },
    },
    wallSurface('wall_back', 'Back Wall', [0, floorY, -halfD], [0, 0, 1], [1, 0, 0], halfW, roomH),
    wallSurface('wall_front', 'Front Wall', [0, floorY, halfD], [0, 0, 1], [-1, 0, 0], halfW, roomH),
    wallSurface('wall_left', 'Left Wall', [-halfW, floorY, 0], [1, 0, 0], [0, 0, 1], halfD, roomH),
    wallSurface('wall_right', 'Right Wall', [halfW, floorY, 0], [1, 0, 0], [0, 0, -1], halfD, roomH),
  ];

  return { data, info: { width, height, channels: 3 }, surfaces };
}

function wallSurface(name, label, origin, normal, axisU, halfSpan, roomH) {
  return {
    name,
    label,
    product_surface: 'wall',
    plane: {
      origin, normal, axisU, axisV: [0, 1, 0],
      extent: [-halfSpan, halfSpan, 0, roomH],
    },
    defaults: { tileSize: { w: 300, h: 600 }, layout: 'brick', grout: { size: 2, color: '#f5f5f2' } },
  };
}
