/**
 * Edge-aware refinement of the class masks.
 *
 * The segmenter runs at 512x512 and its output is a blob map: correct about
 * *what* is where, wrong by several pixels about exactly where it ends. Scaled
 * up to a 2400px photo those few pixels become a wobbling band tens of pixels
 * wide, so the wall material creeps over the ceiling line, the floor eats the
 * skirting, and the silhouette of a sofa comes out as a soft lump. No amount
 * of polygon smoothing fixes that: the information simply is not in the class
 * map.
 *
 * It is in the photograph. Every boundary that matters -- the ceiling line,
 * the skirting, the edge of the sofa -- is a strong intensity edge, and a
 * guided filter (He, Sun & Tang) is the standard way to pull one signal onto
 * another's edges: it fits a local linear model q = a*I + b of the mask
 * against the image over a small window, so the refined mask is forced to be
 * a linear function of image intensity nearby, and its transitions land where
 * the image's do.
 *
 * Everything here is O(number of pixels): box filters via a running sum, no
 * per-pixel window loops.
 */

/** Separable box blur by running sum, edges clamped. Radius in pixels. */
export function boxFilter(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const n = 2 * r + 1;

  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / n;
      sum += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }

  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / n;
      sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

/**
 * Guided filter: refine `p` so that its edges follow the guide image `I`.
 *
 * `eps` is the regularisation, and it is the dial that matters: too small and
 * texture inside a surface (a floorboard join, a curtain fold) is treated as a
 * boundary and eats holes in the mask; too large and the filter degenerates
 * into a plain blur that snaps to nothing.
 */
export function guidedFilter(I, p, w, h, r, eps) {
  const n = w * h;
  const Ip = new Float32Array(n);
  const II = new Float32Array(n);
  for (let i = 0; i < n; i++) { Ip[i] = I[i] * p[i]; II[i] = I[i] * I[i]; }

  const meanI = boxFilter(I, w, h, r);
  const meanP = boxFilter(p, w, h, r);
  const meanIp = boxFilter(Ip, w, h, r);
  const meanII = boxFilter(II, w, h, r);

  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = meanII[i] - meanI[i] * meanI[i];
    const covIp = meanIp[i] - meanI[i] * meanP[i];
    a[i] = covIp / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }

  const meanA = boxFilter(a, w, h, r);
  const meanB = boxFilter(b, w, h, r);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = meanA[i] * I[i] + meanB[i];
  return q;
}

/**
 * Bilinear resample of one of the model's masks into a soft 0..1 map.
 *
 * Bilinear rather than nearest because the guided filter wants a soft edge to
 * work with -- a hard staircase gives it nothing to move.  Merging by max
 * because one ADE20K label can come back as several disjoint regions.
 */
export function resampleMask(mask, dst, w, h) {
  const mw = mask.width;
  const mh = mask.height;
  const data = mask.data;
  for (let y = 0; y < h; y++) {
    const sy = ((y + 0.5) / h) * mh - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(mh - 1, y0 + 1);
    const fy = Math.min(1, Math.max(0, sy - y0));
    for (let x = 0; x < w; x++) {
      const sx = ((x + 0.5) / w) * mw - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(mw - 1, x0 + 1);
      const fx = Math.min(1, Math.max(0, sx - x0));
      const v = (data[y0 * mw + x0] * (1 - fx) + data[y0 * mw + x1] * fx) * (1 - fy)
              + (data[y1 * mw + x0] * (1 - fx) + data[y1 * mw + x1] * fx) * fy;
      const i = y * w + x;
      const s = v / 255;
      if (s > dst[i]) dst[i] = s;
    }
  }
  return dst;
}

/**
 * Snap a set of class maps to the photograph's edges and make them exclusive.
 *
 * Refining each class separately would let two of them claim the same pixel,
 * so the winner is decided afterwards by which class the refined maps score
 * highest for -- one pass, one owner per pixel, and no gap along a boundary
 * because the two sides are decided by the same comparison.
 *
 * A pixel none of them claims above `floor` belongs to something else in the
 * room -- a sofa, a curtain, a window -- and is left out, which is what makes
 * furniture silhouettes come out crisp: the mask stops where the image says
 * the sofa starts.
 */
export function refineClassMaps(guide, maps, w, h, { radius, eps = 1e-4, floor = 0.12, drop = [] } = {}) {
  const r = radius ?? Math.max(3, Math.round(Math.min(w, h) / 120));
  const keys = Object.keys(maps);
  const refined = {};
  for (const k of keys) refined[k] = guidedFilter(guide, maps[k], w, h, r, eps);

  const out = {};
  for (const k of keys) out[k] = new Uint8Array(w * h);

  // The comparison is between classes, not against a fixed level. An absolute
  // threshold throws away every pixel the segmenter was merely unsure about --
  // wall in shadow, wall seen through a net curtain, the strip above a door --
  // and those are exactly the areas that come out as holes in a wall. What
  // matters is not whether "wall" scored highly, but whether anything else
  // scored higher; `floor` is left only to reject pixels where nothing at all
  // was detected.
  for (let i = 0; i < w * h; i++) {
    let best = null;
    let bestV = floor;
    for (const k of keys) {
      const v = refined[k][i];
      if (v > bestV) { bestV = v; best = k; }
    }
    if (best) out[best][i] = 255;
  }
  for (const k of drop) delete out[k];
  return { masks: out, radius: r };
}

/** Greyscale luminance in 0..1, for use as the guide. */
export function toLuma(rgb, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (0.2126 * rgb[i * 3] + 0.7152 * rgb[i * 3 + 1] + 0.0722 * rgb[i * 3 + 2]) / 255;
  }
  return out;
}

/** Nearest-neighbour blow-up of a refined binary mask back to photo pixels. */
export function upscaleMask(src, sw, sh, dw, dh) {
  const out = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
      out[y * dw + x] = src[sy * sw + sx];
    }
  }
  return out;
}

/**
 * Binary dilation by `r` pixels, separable.
 *
 * Used to grow the occluder map slightly before it is punched out of the
 * surfaces. The asymmetry is deliberate: a tile drawn a few pixels onto the
 * edge of a rug is obvious and wrong, while a few pixels of untiled floor
 * around it is invisible. When the boundary is uncertain, err away from
 * tiling the object.
 */
export function dilate(mask, w, h, r) {
  if (r <= 0) return mask;
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < w && mask[row + xx]) v = 1;
      }
      tmp[row + x] = v;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < h && tmp[yy * w + x]) v = 1;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

/**
 * Morphological closing (dilate then erode).
 */
export function morphClose(mask, w, h, radius) {
  const dilated = bfsDilate(mask, w, h, radius);
  const inverted = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inverted[i] = dilated[i] ? 0 : 255;
  const erodedInvert = bfsDilate(inverted, w, h, radius);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = erodedInvert[i] ? 0 : 255;
  return out;
}

/**
 * Grows mask up to `maxReach` into unclassified pixels.
 * Halts at `occluder` pixels.
 */
export function growIntoUnclassified(mask, occluder, w, h, maxReach) {
  const out = new Uint8Array(mask);
  if (maxReach <= 0) return out;
  const queue = new Int32Array(w * h);
  let qHead = 0, qTail = 0;
  const dist = new Int32Array(w * h).fill(-1);
  for (let i = 0; i < w * h; i++) {
    if (out[i]) {
      dist[i] = 0;
      queue[qTail++] = i;
    }
  }
  while (qHead < qTail) {
    const p = queue[qHead++];
    const d = dist[p];
    if (d >= maxReach) continue;
    const x = p % w;
    const y = Math.floor(p / w);

    // Manual unrolling for maximum performance (avoids closure allocations)
    const nextD = d + 1;
    
    // Left
    if (x > 0) {
      const ni = y * w + (x - 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Right
    if (x < w - 1) {
      const ni = y * w + (x + 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top
    if (y > 0) {
      const ni = (y - 1) * w + x;
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom
    if (y < h - 1) {
      const ni = (y + 1) * w + x;
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top-left
    if (x > 0 && y > 0) {
      const ni = (y - 1) * w + (x - 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top-right
    if (x < w - 1 && y > 0) {
      const ni = (y - 1) * w + (x + 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom-left
    if (x > 0 && y < h - 1) {
      const ni = (y + 1) * w + (x - 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom-right
    if (x < w - 1 && y < h - 1) {
      const ni = (y + 1) * w + (x + 1);
      if (!out[ni] && (!occluder || !occluder[ni])) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
  }
  return out;
}

/**
 * Dilation using Breadth-First Search.
 * Ensures an exact circular expansion up to `radius`.
 */
export function bfsDilate(mask, w, h, radius) {
  const out = new Uint8Array(mask);
  if (radius <= 0) return out;
  const queue = new Int32Array(w * h);
  let qHead = 0, qTail = 0;
  const dist = new Int32Array(w * h).fill(-1);
  for (let i = 0; i < w * h; i++) {
    if (out[i]) {
      dist[i] = 0;
      queue[qTail++] = i;
    }
  }
  while (qHead < qTail) {
    const p = queue[qHead++];
    const d = dist[p];
    if (d >= radius) continue;
    const x = p % w;
    const y = Math.floor(p / w);
    
    const nextD = d + 1;

    // Left
    if (x > 0) {
      const ni = y * w + (x - 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Right
    if (x < w - 1) {
      const ni = y * w + (x + 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top
    if (y > 0) {
      const ni = (y - 1) * w + x;
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom
    if (y < h - 1) {
      const ni = (y + 1) * w + x;
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top-left
    if (x > 0 && y > 0) {
      const ni = (y - 1) * w + (x - 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Top-right
    if (x < w - 1 && y > 0) {
      const ni = (y - 1) * w + (x + 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom-left
    if (x > 0 && y < h - 1) {
      const ni = (y + 1) * w + (x - 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
    // Bottom-right
    if (x < w - 1 && y < h - 1) {
      const ni = (y + 1) * w + (x + 1);
      if (!out[ni]) { out[ni] = 255; dist[ni] = nextD; queue[qTail++] = ni; }
    }
  }
  return out;
}

export function floorBaselineUnblock(floorMask, wallMask, w, h) {
  const topY = new Int32Array(w).fill(-1);
  let hasFloor = false;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (floorMask[y * w + x]) {
        topY[x] = y;
        hasFloor = true;
        break;
      }
    }
  }
  const out = new Uint8Array(w * h);
  if (!hasFloor) return out;

  // Fill in missing columns by searching left/right
  for (let x = 0; x < w; x++) {
    if (topY[x] === -1) {
      let l = x - 1, r = x + 1;
      while (l >= 0 && topY[l] === -1) l--;
      while (r < w && topY[r] === -1) r++;
      if (l >= 0 && r < w) {
        topY[x] = topY[l] + (topY[r] - topY[l]) * ((x - l) / (r - l));
      } else if (l >= 0) {
        topY[x] = topY[l];
      } else if (r < w) {
        topY[x] = topY[r];
      }
    }
  }

  // Simple box blur
  const blurW = Math.max(5, Math.round(w * 0.025));
  const blurred = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let sum = 0, count = 0;
    for (let k = -blurW; k <= blurW; k++) {
      if (x + k >= 0 && x + k < w) { sum += topY[x + k]; count++; }
    }
    blurred[x] = sum / count;
  }

  const margin = Math.max(4, Math.round(h * 0.012));
  for (let x = 0; x < w; x++) {
    const y0 = Math.max(0, Math.round(blurred[x]) - margin);
    for (let y = y0; y < h; y++) {
      if (wallMask[y * w + x]) {
        out[y * w + x] = 1;
      }
    }
  }
  return out;
}

export function detectTexturedObjects(gray, surfaceMask, w, h) {
  const win = 9;
  const halfWin = Math.floor(win / 2);
  const out = new Uint8Array(w * h);

  // Box blur for mean
  const mean = new Float32Array(w * h);
  const sqMean = new Float32Array(w * h);
  
  // Separable box blur for performance
  const tempMean = new Float32Array(w * h);
  const tempSq = new Float32Array(w * h);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, sqSum = 0, count = 0;
      for (let k = -halfWin; k <= halfWin; k++) {
        const nx = x + k;
        if (nx >= 0 && nx < w) {
          const val = gray[y * w + nx];
          sum += val;
          sqSum += val * val;
          count++;
        }
      }
      tempMean[y * w + x] = sum / count;
      tempSq[y * w + x] = sqSum / count;
    }
  }

  // Vertical pass
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let sum = 0, sqSum = 0, count = 0;
      for (let k = -halfWin; k <= halfWin; k++) {
        const ny = y + k;
        if (ny >= 0 && ny < h) {
          sum += tempMean[ny * w + x];
          sqSum += tempSq[ny * w + x];
          count++;
        }
      }
      mean[y * w + x] = sum / count;
      sqMean[y * w + x] = sqSum / count;
    }
  }

  // Calculate local standard deviation
  for (let i = 0; i < w * h; i++) {
    if (!surfaceMask[i]) continue;
    const m = mean[i];
    const sq = sqMean[i];
    const variance = sq - m * m;
    if (variance > 0) {
      const stdDev = Math.sqrt(variance);
      if (stdDev > 12) {
        out[i] = 1;
      }
    }
  }

  return out;
}
