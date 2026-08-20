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

/** Chooses tile-texture repeat counts so tiling density stays consistent regardless of the quad's shape. */
export function computeRepeat(
  quad: Quad,
  texWidth: number,
  texHeight: number,
  density: number,
): { repeatX: number; repeatY: number } {
  const w = quadWidthPx(quad);
  const h = quadHeightPx(quad);
  const aspect = texWidth / texHeight || 1;
  const repeatX = Math.max(1, density);
  const repeatY = Math.max(1, (repeatX * aspect * h) / Math.max(1, w));
  return { repeatX, repeatY };
}
