import { Quad } from "./types";

export function quadWidthPx(quad: Quad): number {
  const top = Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
  const bottom = Math.hypot(quad[2].x - quad[3].x, quad[2].y - quad[3].y);
  return (top + bottom) / 2;
}

export function quadHeightPx(quad: Quad): number {
  const left = Math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y);
  const right = Math.hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y);
  return (left + right) / 2;
}

/**
 * Chooses tile-texture repeat counts so each individual tile reads as the *same real-world size*
 * everywhere it's used, on any quad, regardless of that particular surface's own size — the same
 * way a real tile installation works: a small return wall shows fewer tiles than the big wall
 * next to it, not the same handful of tiles blown up to fill it. `referenceWidthPx` anchors that
 * apparent tile size once per photo (`density` tiles across a surface that wide), so a half-width
 * wall gets roughly half as many repeats — smaller in count, identical in apparent size — instead
 * of the same repeat count stretched to a different physical size.
 *
 * Also returns a UV `offsetX`/`offsetY` anchored to the quad's own top-left corner in *absolute*
 * image-pixel space (scaled by that same physical tile size), instead of every quad restarting
 * its tile pattern at UV (0,0). A floor is frequently rendered as several quads — stacked bands
 * fit to an irregular mask shape, or several separate detected floor regions split apart by
 * furniture — and without this, each one shows a tile grid with its own arbitrary phase, so the
 * whole floor reads as disconnected patches instead of one continuous, identically-tiled surface.
 * Two quads that are genuinely adjacent in the photo share (near enough) the same pixel position
 * along their common edge, so anchoring offset to absolute position makes their tile grids line
 * up there automatically — the same way real tile installers snap every cut to one shared grid
 * instead of starting each panel's pattern over from its own corner.
 */
export function computeTileUv(
  quad: Quad,
  texWidth: number,
  texHeight: number,
  density: number,
  referenceWidthPx: number,
): { repeatX: number; repeatY: number; offsetX: number; offsetY: number } {
  const w = quadWidthPx(quad);
  const h = quadHeightPx(quad);
  const aspect = texWidth / texHeight || 1;
  const tileWidthPx = Math.max(1, referenceWidthPx) / Math.max(1, density);
  const tileHeightPx = tileWidthPx / aspect;
  // Deliberately *not* floored at 1: a real physical strip narrower than one tile shows a
  // cropped slice of that tile, not one whole tile stretched to fill the strip — flooring this
  // at 1 was exactly what broke scale consistency on any small/narrow selected part, since it
  // silently blew a fractional tile back up to a full one while its wider neighbor, needing no
  // such rescue, stayed at the correct real scale. The tile texture wraps (`gl.REPEAT`), so a
  // fractional repeat still samples correctly — it's just less than one full cycle of it.
  const repeatX = Math.max(0.02, w / tileWidthPx);
  const repeatY = Math.max(0.02, (repeatX * aspect * h) / Math.max(1, w));
  const offsetX = quad[0].x / tileWidthPx;
  const offsetY = quad[0].y / tileHeightPx;
  return { repeatX, repeatY, offsetX, offsetY };
}
