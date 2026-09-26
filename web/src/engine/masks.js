/**
 * Surface masks.
 *
 * A surface is described by polygons in image space rather than by a shipped
 * PNG: it stays editable forever, costs a few hundred bytes, and rasterises to
 * a perfectly antialiased coverage map at whatever resolution we need.
 *
 * `add` polygons paint the surface in, `subtract` polygons cut occluders back
 * out (a rug on the floor, a vanity against the wall, a table leg).
 */
import { undistortPoint, undistortJacobian } from './homography.js';

/** Rasterise one surface's polygons into a single-channel coverage canvas. */
export function rasterizeMask(mask, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  const polys = mask?.polygons ?? [];
  // Additive first, then the cut-outs, so ordering inside the array does not
  // change the result and authors cannot paint themselves into a corner.
  for (const poly of polys) {
    if (poly.mode === 'subtract') continue;
    tracePolygon(ctx, poly.points);
    ctx.fillStyle = '#fff';
    ctx.fill('nonzero');
  }
  for (const poly of polys) {
    if (poly.mode !== 'subtract') continue;
    tracePolygon(ctx, poly.points);
    ctx.fillStyle = '#000';
    ctx.fill('nonzero');
  }

  if (mask?.feather > 0) {
    const blurred = document.createElement('canvas');
    blurred.width = width;
    blurred.height = height;
    const bctx = blurred.getContext('2d');
    bctx.filter = `blur(${mask.feather}px)`;
    bctx.drawImage(canvas, 0, 0);
    return blurred;
  }
  return canvas;
}

function tracePolygon(ctx, points) {
  if (!points || points.length < 3) {
    ctx.beginPath();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
}

/** Winding-rule point test, used for click-to-select in the viewer. */
export function pointInMask(mask, x, y) {
  let inside = false;
  for (const poly of mask?.polygons ?? []) {
    if (poly.mode === 'subtract') continue;
    if (pointInPolygon(poly.points, x, y)) { inside = true; break; }
  }
  if (!inside) return false;
  for (const poly of mask?.polygons ?? []) {
    if (poly.mode !== 'subtract') continue;
    if (pointInPolygon(poly.points, x, y)) return false;
  }
  return true;
}

export function pointInPolygon(points, x, y) {
  if (!points || points.length < 3) return false;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    const intersects = (yi > y) !== (yj > y)
      && x < ((xj - xi) * (y - yi)) / (yj - yi + Number.EPSILON) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/**
 * Reference luminance for a surface: the level the shader divides by so that
 * "as bright as the original surface" comes out as "unchanged brightness".
 *
 * A percentile rather than a mean, because floors are full of dark furniture
 * shadows that would otherwise drag the reference down and wash the tile out.
 */
export function referenceLuminance(imageData, mask, width, height, percentile = 0.6) {
  const hist = new Float64Array(256);
  let count = 0;
  const step = Math.max(1, Math.floor(Math.sqrt((width * height) / 40000)));

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      if (!pointInMask(mask, x, y)) continue;
      const i = (y * width + x) * 4;
      const l = (0.2126 * imageData[i] + 0.7152 * imageData[i + 1] + 0.0722 * imageData[i + 2]) / 255;
      hist[Math.min(255, Math.max(0, Math.round(l * 255)))] += 1;
      count++;
    }
  }
  if (!count) return 0.5;

  const target = count * percentile;
  let acc = 0;
  for (let i = 0; i < 256; i++) {
    acc += hist[i];
    if (acc >= target) return Math.max(0.02, i / 255);
  }
  return 0.5;
}

/** Bounding box of a surface's additive polygons, in image pixels. */
export function maskBounds(mask) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const poly of mask?.polygons ?? []) {
    if (poly.mode === 'subtract') continue;
    for (const [x, y] of poly.points) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return null;
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/** A rectangle covering the whole photo -- the starting point for a new surface. */
export function fullFrameMask(width, height) {
  return {
    feather: 1,
    polygons: [
      {
        mode: 'add',
        points: [
          [width * 0.15, height * 0.55],
          [width * 0.85, height * 0.55],
          [width * 0.98, height * 0.95],
          [width * 0.02, height * 0.95],
        ],
      },
    ],
  };
}

/**
 * True area of a masked surface, in square metres.
 *
 * The reference product makes you type the floor area in by hand. We already
 * know the mask and the homography, so we can just measure it: a projective map
 * has Jacobian determinant det(H) / (g*x + h*y + i)^3, so summing that over the
 * covered pixels converts image area straight into metric area -- correctly
 * weighting distant pixels, which cover far more floor than near ones.
 *
 * `hInv` is the pixels -> metres homography, column-major. When the room
 * carries a lens correction the homography lives in rectilinear space, so each
 * sample is lifted there first and the radial map's own Jacobian folded in --
 * otherwise a corrected wide-angle photo would measure short at the edges.
 */
export function surfaceAreaSqm(mask, hInv, width, height, step = 3, k1 = 0) {
  if (!hInv) return 0;
  // Row-major view, so the algebra below reads like the textbook form.
  const m = [hInv[0], hInv[3], hInv[6], hInv[1], hInv[4], hInv[7], hInv[2], hInv[5], hInv[8]];
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;

  const det =
    m00 * (m11 * m22 - m12 * m21) -
    m01 * (m10 * m22 - m12 * m20) +
    m02 * (m10 * m21 - m11 * m20);
  if (!Number.isFinite(det) || det === 0) return 0;

  // Rasterise at reduced resolution, which means scaling the polygons too --
  // they are stored in full image space.
  const small = {
    feather: 0,
    polygons: (mask?.polygons ?? []).map((p) => ({
      ...p,
      points: p.points.map(([x, y]) => [x / step, y / step]),
    })),
  };
  const canvas = rasterizeMask(small, Math.ceil(width / step), Math.ceil(height / step));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  let area = 0;
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      if (data[(y * canvas.width + x) * 4] < 128) continue;
      const sx = (x + 0.5) * step;
      const sy = (y + 0.5) * step;
      const [px, py] = k1 ? undistortPoint([sx, sy], k1, width, height) : [sx, sy];
      const w = m20 * px + m21 * py + m22;
      const jac = Math.abs(det) / Math.abs(w * w * w)
        * (k1 ? Math.abs(undistortJacobian([sx, sy], k1, width, height)) : 1);
      if (Number.isFinite(jac)) area += jac;
    }
  }
  // Each sampled cell stands for step x step image pixels.
  return area * step * step;
}

/**
 * Where to put a marker for a surface.
 *
 * The centroid is the obvious answer and the wrong one: an L-shaped floor's
 * centroid falls in the missing corner, and a wall's falls on the wardrobe cut
 * out of the middle of it. What is wanted is the pole of inaccessibility --
 * the point furthest from any edge -- which is also the point with the most
 * room around it for a label.
 *
 * A two-pass chamfer distance transform over a small raster is plenty: the
 * marker only has to be comfortably inside, not exact.
 */
export function maskAnchor(mask, width, height, maxSide = 220) {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));

  const small = {
    feather: 0,
    polygons: (mask?.polygons ?? []).map((p) => ({
      ...p,
      points: p.points.map(([x, y]) => [x * scale, y * scale]),
    })),
  };
  const canvas = rasterizeMask(small, w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const px = ctx.getImageData(0, 0, w, h).data;

  const INF = 1e9;
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) d[i] = px[i * 4] >= 128 ? INF : 0;

  /**
   * Chamfer 3-4: forward pass over the top-left neighbourhood, backward over
   * the bottom-right. Outside the mask stays at zero, so distance grows inward.
   *
   * The edge of the photograph counts as outside. Nearly every real surface is
   * cut off by the frame -- a floor runs to the bottom, a side wall to the left
   * -- and treating that cut as interior puts the deepest point *on* the frame
   * edge, which is the one place a marker cannot be seen: half of it is off
   * screen and the viewer's own bounds test hides it entirely.
   */
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : d[y * w + x]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (d[i] === 0) continue;
      d[i] = Math.min(
        d[i],
        at(x - 1, y) + 3, at(x, y - 1) + 3,
        at(x - 1, y - 1) + 4, at(x + 1, y - 1) + 4,
      );
    }
  }
  let best = -1;
  let bestD = 0;
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (d[i] === 0) continue;
      d[i] = Math.min(
        d[i],
        at(x + 1, y) + 3, at(x, y + 1) + 3,
        at(x + 1, y + 1) + 4, at(x - 1, y + 1) + 4,
      );
      if (d[i] > bestD) { bestD = d[i]; best = i; }
    }
  }
  if (best < 0) {
    const b = maskBounds(mask);
    return b ? { x: b.x + b.w / 2, y: b.y + b.h / 2, clearance: 0 } : null;
  }
  return {
    x: ((best % w) + 0.5) / scale,
    y: (Math.floor(best / w) + 0.5) / scale,
    // In photo pixels: how much space the marker has before it hits an edge.
    clearance: (bestD / 3) / scale,
  };
}
