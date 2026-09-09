/**
 * Homography utilities.
 *
 * A flat surface in a photo (floor, wall, backsplash) is a plane. The mapping
 * from that plane's own metric coordinates (metres) to pixel coordinates in the
 * photo is *exactly* a 3x3 projective homography -- no camera pose solve needed.
 *
 * The room author clicks the 4 corners of a rectangle they know the real size of
 * (e.g. "this floor patch is 4.0m x 3.0m"). From those 4 point pairs we solve H.
 * The shader then uses H^-1 to go pixel -> metres and look up the tile pattern,
 * which is what makes tiles converge correctly toward the vanishing point.
 *
 * Matrices are column-major length-9 arrays, matching THREE.Matrix3.elements:
 *   [ m00 m10 m20  m01 m11 m21  m02 m12 m22 ]
 */

/** Solve the 8x8 linear system for the homography mapping src[i] -> dst[i]. */
export function homographyFromPoints(src, dst) {
  if (src.length !== 4 || dst.length !== 4) {
    throw new Error('homographyFromPoints needs exactly 4 point pairs');
  }
  // Build A * h = b where h = [h00 h01 h02 h10 h11 h12 h20 h21] and h22 = 1.
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const h = solveLinear(A, b);
  if (!h) return null;
  // Row-major [h00 h01 h02 h10 h11 h12 h20 h21 1] -> column-major for THREE.
  return rowMajorToColumnMajor([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]);
}

/** Gauss-Jordan elimination with partial pivoting. Returns null if singular. */
function solveLinear(A, b) {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const d = m[col][col];
    for (let c = col; c <= n; c++) m[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row) => row[n]);
}

function rowMajorToColumnMajor(r) {
  return [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
}

/** Invert a 3x3 column-major matrix. Returns null if singular. */
export function invert3(e) {
  const [a, d, g, b, f, h, c, i, k] = e; // column-major -> rows are (a b c),(d f i)... careful below
  // Reconstruct row-major for clarity.
  const m = [e[0], e[3], e[6], e[1], e[4], e[7], e[2], e[5], e[8]];
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
  const c00 = m11 * m22 - m12 * m21;
  const c01 = m12 * m20 - m10 * m22;
  const c02 = m10 * m21 - m11 * m20;
  const det = m00 * c00 + m01 * c01 + m02 * c02;
  if (Math.abs(det) < 1e-14) return null;
  const id = 1 / det;
  const inv = [
    c00 * id, (m02 * m21 - m01 * m22) * id, (m01 * m12 - m02 * m11) * id,
    c01 * id, (m00 * m22 - m02 * m20) * id, (m02 * m10 - m00 * m12) * id,
    c02 * id, (m01 * m20 - m00 * m21) * id, (m00 * m11 - m01 * m10) * id,
  ];
  return rowMajorToColumnMajor(inv);
}

/** Apply a column-major 3x3 to a 2D point, dividing through by w. */
export function applyH(e, x, y) {
  const w = e[2] * x + e[5] * y + e[8];
  return [
    (e[0] * x + e[3] * y + e[6]) / w,
    (e[1] * x + e[4] * y + e[7]) / w,
  ];
}

/* ------------------------------------------------------- lens distortion -- */

/**
 * Barrel distortion, division model: r_u = r_d / (1 + k1 * r_d^2), with the
 * radius normalised by the photo's half-diagonal so k1 means the same thing
 * whatever the resolution.
 *
 * A homography models a pinhole camera exactly and a real one only
 * approximately: on a wide-angle phone photo the tile courses bow visibly near
 * the frame edge. Correcting for it needs the *inverse* map -- observed pixel
 * to ideal pixel -- and the division model is the one that inverts in closed
 * form, which is why it is used here rather than Brown's polynomial.
 *
 * k1 < 0 straightens barrel (the usual case); k1 > 0 straightens pincushion.
 */
export function lensScale(width, height) {
  return Math.hypot(width, height) / 2;
}

/** Observed pixel -> ideal (rectilinear) pixel. */
export function undistortPoint(p, k1, width, height) {
  if (!k1) return [p[0], p[1]];
  const cx = width / 2;
  const cy = height / 2;
  const s = lensScale(width, height);
  const dx = p[0] - cx;
  const dy = p[1] - cy;
  const r2 = (dx * dx + dy * dy) / (s * s);
  const f = 1 / (1 + k1 * r2);
  return [cx + dx * f, cy + dy * f];
}

/**
 * Ideal pixel -> observed pixel, the inverse of the above.
 *
 * r_u = r_d / (1 + k1 r_d^2) rearranges to k1 r_u r_d^2 - r_d + r_u = 0, whose
 * root closest to r_u is the one to take; the other branch folds the image
 * back through itself.
 */
export function distortPoint(p, k1, width, height) {
  if (!k1) return [p[0], p[1]];
  const cx = width / 2;
  const cy = height / 2;
  const s = lensScale(width, height);
  const dx = (p[0] - cx) / s;
  const dy = (p[1] - cy) / s;
  const ru = Math.hypot(dx, dy);
  if (ru < 1e-9) return [p[0], p[1]];
  const disc = 1 - 4 * k1 * ru * ru;
  if (disc < 0) return [p[0], p[1]];
  const rd = (1 - Math.sqrt(disc)) / (2 * k1 * ru);
  const f = rd / ru;
  return [cx + dx * s * f, cy + dy * s * f];
}

/** Jacobian determinant of the undistort map, for measuring true area. */
export function undistortJacobian(p, k1, width, height) {
  if (!k1) return 1;
  const s = lensScale(width, height);
  const dx = (p[0] - width / 2) / s;
  const dy = (p[1] - height / 2) / s;
  const r2 = dx * dx + dy * dy;
  const f = 1 / (1 + k1 * r2);
  return f * f * f * (1 - k1 * r2);
}

export const undistortQuad = (quad, k1, width, height) =>
  quad.map((p) => undistortPoint(p, k1, width, height));

/**
 * Build the plane-metres -> image-pixels homography for a surface.
 *
 * `quad` are the four clicked image-space corners in order
 * [topLeft, topRight, bottomRight, bottomLeft] as seen on the plane, and
 * `widthM`/`heightM` are that rectangle's true dimensions in metres.
 *
 * `lens` is optional `{ k1, width, height }`. When it is given the corners are
 * lifted into rectilinear space first, so the solved homography describes the
 * pinhole camera the photograph would have come from -- the shader undistorts
 * each pixel the same way before looking the pattern up.
 */
export function surfaceHomography(quad, widthM, heightM, lens = null) {
  const planeRect = [
    [0, 0],
    [widthM, 0],
    [widthM, heightM],
    [0, heightM],
  ];
  const corners = lens?.k1
    ? undistortQuad(quad, lens.k1, lens.width, lens.height)
    : quad;
  return homographyFromPoints(planeRect, corners);
}

/**
 * Rough check that a quad is a sane convex, non self-intersecting shape.
 * A bad quad produces a homography with a vanishing line crossing the surface,
 * which renders as a mirrored/exploded texture -- worth catching in the editor.
 */
export function isConvexQuad(quad) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % 4];
    const [cx, cy] = quad[(i + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (Math.abs(cross) < 1e-9) continue;
    const s = Math.sign(cross);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return sign !== 0;
}
