/**
 * Vertical corner detection.
 *
 * Where the wall/floor junction is hidden -- behind a sofa, a console table, a
 * run of units -- the piecewise fit has nothing to find a corner with, and two
 * walls come back as one badly-fitting plane. But the corner is still plainly
 * visible in the photograph: it is a vertical line.
 *
 * That is not a heuristic, it is the same camera assumption everything else
 * here rests on. Interior photographs are shot level to keep verticals
 * vertical, so a vertical line in the room projects to a vertical line in the
 * image -- which means a wall corner is a *column*, and looking for one is a
 * one-dimensional search.
 *
 * The hard part is not finding vertical edges; it is telling a corner from a
 * window frame, a door architrave or a picture. Three things separate them:
 *
 *   continuity  a corner runs the whole height of the wall; a frame stops
 *   contrast    two walls at different angles catch the light differently, so
 *               the mean brightness steps across a corner and does not across
 *               a frame, which has the same wall on both sides
 *   isolation   frames come in pairs a window's width apart; corners do not,
 *               so candidates are thinned to the strongest in a neighbourhood
 */

/** Sobel gradients. Returns |dx| and |dy| so verticality can be tested. */
function sobel(gray, w, h) {
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const at = (x, y) => gray[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const tl = at(x - 1, y - 1); const tc = at(x, y - 1); const tr = at(x + 1, y - 1);
      const ml = at(x - 1, y); const mr = at(x + 1, y);
      const bl = at(x - 1, y + 1); const bc = at(x, y + 1); const br = at(x + 1, y + 1);
      gx[y * w + x] = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
      gy[y * w + x] = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
    }
  }
  return { gx, gy };
}

/**
 * Candidate corner columns inside a wall region.
 *
 * `gray` and `mask` must be the same size. Returns columns in that coordinate
 * space, strongest first, each with the evidence that put it there so a caller
 * can decide how much to trust it.
 */
export function verticalCorners(gray, mask, w, h, {
  edge = 0.16,          // gradient magnitude that counts as an edge
  ratio = 1.8,          // how much more vertical than horizontal it must be
  continuity = 0.45,    // fraction of the wall's height the line must span
  contrast = 0.035,     // brightness step across the column, 0..1
  minSep = 0.05,        // candidates closer than this (fraction of w) merge
  limit = 6,
} = {}) {
  const { gx, gy } = sobel(gray, w, h);

  // Vertical extent of the wall in each column, and where it sits.
  const top = new Int32Array(w).fill(-1);
  const bottom = new Int32Array(w).fill(-1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) if (mask[y * w + x]) { top[x] = y; break; }
    for (let y = h - 1; y >= 0; y--) if (mask[y * w + x]) { bottom[x] = y; break; }
  }

  const scores = new Float32Array(w);
  const steps = new Float32Array(w);

  for (let x = 2; x < w - 2; x++) {
    if (top[x] < 0 || bottom[x] - top[x] < h * 0.12) continue;
    const height = bottom[x] - top[x] + 1;

    // Longest unbroken vertical edge in this column, within the wall. A corner
    // is one long line; texture is many short ones, and summing would rank
    // them the same.
    let best = 0;
    let run = 0;
    for (let y = top[x]; y <= bottom[x]; y++) {
      const i = y * w + x;
      const vx = Math.abs(gx[i]);
      const strong = vx > edge && vx > ratio * Math.abs(gy[i]);
      // A one-pixel gap is a JPEG artefact, not the end of the corner.
      if (strong || (run > 0 && y + 1 <= bottom[x]
        && Math.abs(gx[i + w]) > edge && Math.abs(gx[i + w]) > ratio * Math.abs(gy[i + w]))) {
        run += 1;
        if (run > best) best = run;
      } else {
        run = 0;
      }
    }
    scores[x] = best / height;

    // Brightness either side, over the wall's own rows only, so furniture in
    // front of it does not decide the question.
    let sl = 0; let nl = 0; let sr = 0; let nr = 0;
    for (let y = top[x]; y <= bottom[x]; y++) {
      for (let d = 2; d <= 6; d++) {
        const lx = x - d;
        const rx = x + d;
        if (lx >= 0 && mask[y * w + lx]) { sl += gray[y * w + lx]; nl++; }
        if (rx < w && mask[y * w + rx]) { sr += gray[y * w + rx]; nr++; }
      }
    }
    steps[x] = nl && nr ? Math.abs(sl / nl - sr / nr) : 0;
  }

  const found = [];
  for (let x = 2; x < w - 2; x++) {
    if (scores[x] < continuity || steps[x] < contrast) continue;
    found.push({ u: x, continuity: scores[x], contrast: steps[x], score: scores[x] * (1 + steps[x] * 4) });
  }
  found.sort((a, b) => b.score - a.score);

  // Thin to the strongest in each neighbourhood: a real corner produces a
  // cluster of adjacent columns, and a window produces two edges that must not
  // both survive as walls.
  const sep = Math.max(4, Math.round(w * minSep));
  const kept = [];
  for (const c of found) {
    if (kept.some((k) => Math.abs(k.u - c.u) < sep)) continue;
    kept.push(c);
    if (kept.length >= limit) break;
  }
  return kept.sort((a, b) => a.u - b.u);
}
