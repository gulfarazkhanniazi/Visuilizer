/**
 * Which wall owns each pixel: raycast against the final 3D wall planes.
 *
 * For every pixel of the refinement grid:
 *
 *   ray   = camera ray through the pixel centre
 *   hit_k = ray ∩ plane_k, kept only if it lands inside wall k's extent
 *           (along the wall, and between floor and ceiling)
 *
 * WALL pixels (segmentation says wall) go to the in-extent plane whose hit
 * best agrees with the observed depth (a small bonus for the plane the fit
 * assigned the pixel to). If none agrees the pixel is left unassigned -- it
 * is not forced onto a wall. An observed depth well in front of every hit
 * means something unsegmented stands in front; well behind, an opening.
 *
 * Ownership is deterministic and exclusive -- each pixel is written to at
 * most one wall:
 *   1. only planes whose hit lies inside their wall's extent compete; extents
 *      are snapped to validated corners, so past a corner the other wall's
 *      plane is out of the running
 *   2. lowest depth disagreement wins
 *   3. agreements within `tieRel` are a tie: the nearer hit wins -- at a
 *      corner line both planes are hit at the same depth, so this reduces to
 *      "which side of the projected intersection line is the pixel centre on"
 *   4. an exact tie goes to the lower wall index
 * The boundary is the sub-pixel projected corner line sampled at pixel
 * centres -- never a vertical column cut.
 *
 * OBJECT pixels (sofa, painting, curtain...) whose ray reaches a wall inside
 * its extent, with the object in front of or on that wall, are that wall's
 * OCCLUDED region: the plane continues behind them, but nothing there is
 * visible to tile.
 */
import { CLS, dirToWorld } from './pointcloud.js';
import { ASSIGN } from './config.js';
import { bilinear } from './math.js';

export function assignPixels(ctx, built) {
  const { input, grid, cam, frame } = ctx;
  const { RW, RH, w, h } = input;
  const walls = built.walls;
  const n = RW * RH;
  const visible = walls.map(() => new Uint8Array(n));
  const occluded = walls.map(() => new Uint8Array(n));
  // Non-wall pixels lying flush on a wall's plane: what the segmenter calls
  // an object but the depth says is the wall surface itself (wood panelling
  // labelled "cabinet"). Reported only; nothing above uses it.
  const flush = walls.map(() => new Uint8Array(n));
  const counts = walls.map(() => ({ visible: 0, occluded: 0 }));
  const owner = new Int16Array(n).fill(-1);
  const stats = {
    wallPixels: 0, assigned: 0, looseDepth: 0, unassigned: 0, frontOccluder: 0, opening: 0, outsideExtents: 0,
  };
  const geo = walls.map((wl) => ({
    nx: wl.plane.nx, nz: wl.plane.nz, o: wl.plane.o, t0: wl.t[0], t1: wl.t[1],
    sLo: wl.sMin - ASSIGN.extentPad, sHi: wl.sMax + ASSIGN.extentPad,
    sLoLoose: wl.sMin - 4 * ASSIGN.extentPad, sHiLoose: wl.sMax + 4 * ASSIGN.extentPad,
    yLo: -ASSIGN.heightPad, yHi: wl.top + ASSIGN.heightPad,
  }));
  const H = frame.camHeight;
  const gxScale = grid.GW / RW; const gyScale = grid.GH / RH;
  const wallM = input.masks.wall; const floorM = input.masks.floor; const ceilM = input.masks.ceiling;
  const objW = input.objects?.wall; const objF = input.objects?.floor;
  const tHit = new Float64Array(walls.length);
  const inExt = new Uint8Array(walls.length);
  const inLoose = new Uint8Array(walls.length);

  for (let y = 0; y < RH; y++) {
    const v = ((y + 0.5) * h) / RH;
    for (let x = 0; x < RW; x++) {
      const i = y * RW + x;
      const isWall = !!wallM?.[i];
      const isFloor = !!floorM?.[i];
      const isCeil = !!ceilM?.[i];
      if (!isWall && (isFloor || isCeil)) continue;
      const u = ((x + 0.5) * w) / RW;
      const d = dirToWorld(frame, [(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1]);
      let anyExt = false;
      for (let k = 0; k < geo.length; k++) {
        const g = geo[k];
        const den = g.nx * d[0] + g.nz * d[2];
        inExt[k] = 0; inLoose[k] = 0;
        if (Math.abs(den) < 1e-9) { tHit[k] = Infinity; continue; }
        const tt = g.o / den;
        tHit[k] = tt;
        if (!(tt > 0.05)) continue;
        const X = d[0] * tt; const Z = d[2] * tt; const Y = H + d[1] * tt;
        if (Y < g.yLo || Y > g.yHi) continue;
        const s = g.t0 * X + g.t1 * Z;
        if (s >= g.sLoLoose && s <= g.sHiLoose) inLoose[k] = 1;
        if (s >= g.sLo && s <= g.sHi) { inExt[k] = 1; anyExt = true; }
      }
      const zObs = bilinear(grid.Z, grid.GW, grid.GH, (x + 0.5) * gxScale - 0.5, (y + 0.5) * gyScale - 0.5);

      if (isWall) {
        stats.wallPixels++;
        let near = -1;
        for (let k = 0; k < geo.length; k++) if (inExt[k] && (near < 0 || tHit[k] < tHit[near])) near = k;
        const err = (k) => Math.abs(tHit[k] - zObs) / Math.max(zObs, 1e-3);
        // Among in-extent planes whose hit agrees with the observed depth,
        // the best-agreeing one; the plane the fit already gave this pixel
        // gets a small bonus, and near-equal agreement (at a corner line both
        // planes are hit at the same depth) goes to the nearer hit, then to
        // the lower wall index. Nearest-hit alone fails for two almost
        // parallel planes a few centimetres apart (a panel on a wall).
        const gl = grid.label ? grid.label[Math.min(grid.GH - 1, Math.floor((y + 0.5) * gyScale)) * grid.GW + Math.min(grid.GW - 1, Math.floor((x + 0.5) * gxScale))] : -1;
        let pick = -1; let pickCost = Infinity;
        for (let k = 0; k < geo.length; k++) {
          if (!inExt[k]) continue;
          const e = err(k);
          if (e > ASSIGN.depthTolRel) continue;
          const cost = e - (k === gl ? ASSIGN.labelBonus : 0);
          if (cost < pickCost - ASSIGN.tieRel
              || (Math.abs(cost - pickCost) <= ASSIGN.tieRel && tHit[k] < tHit[pick])) {
            pick = k; pickCost = Math.min(cost, pickCost);
          }
        }
        if (pick < 0) {
          // The nearest in-extent plane disagrees with the depth. Accept the
          // best-agreeing plane that is at least roughly in extent.
          let best = -1;
          for (let k = 0; k < geo.length; k++) {
            if (!inLoose[k] || !(tHit[k] > 0.05)) continue;
            if (best < 0 || err(k) < err(best)) best = k;
          }
          if (best >= 0 && err(best) <= ASSIGN.depthTolRelLoose) { pick = best; stats.looseDepth++; }
        }
        if (pick >= 0) {
          visible[pick][i] = 255; owner[i] = pick; counts[pick].visible++; stats.assigned++;
        } else {
          stats.unassigned++;
          if (!anyExt) stats.outsideExtents++;
          else if (near >= 0 && zObs < tHit[near] * (1 - ASSIGN.depthTolRelLoose)) stats.frontOccluder++;
          else stats.opening++;
        }
        continue;
      }

      // Not wall, floor or ceiling: an object, or something unlabelled.
      if (!anyExt) continue;
      let near = -1;
      for (let k = 0; k < geo.length; k++) if (inExt[k] && (near < 0 || tHit[k] < tHit[near])) near = k;
      if (near < 0) continue;
      if (Math.abs(zObs - tHit[near]) <= ASSIGN.flushTolRel * tHit[near]) flush[near][i] = 255;
      const isObject = (objW && objW[i] >= 0.3) || (objF && objF[i] >= 0.3);
      // In front of or on the wall (not seen through it, like a window view).
      if (zObs <= tHit[near] * (1 + ASSIGN.depthTolRel) && (isObject || zObs < tHit[near] * (1 - ASSIGN.depthTolRel))) {
        occluded[near][i] = 255; counts[near].occluded++;
      }
    }
  }
  return {
    width: RW, height: RH, visible, occluded, flush, owner, counts, stats,
  };
}
