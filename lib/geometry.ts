import { Quad, RegionKind, TileSizeMm } from "./types";

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

export interface TileUv {
  repeatX: number;
  repeatY: number;
  offsetX: number;
  offsetY: number;
}

// Typical interior dimensions, used when this photo's perspective could not be estimated.
// `widthM` is across the surface; `otherM` is into the room for a floor, or floor-to-ceiling for a
// wall. Both are ordinary domestic figures, and they are assumptions rather than measurements —
// which is exactly why they are named and in one place.
const ASSUMED_ROOM = {
  floor: { widthM: 4.5, otherM: 3.5 },
  wall: { widthM: 4.0, otherM: 2.4 },
} as const;

/** Real size in metres of the surface patch one quad covers, as measured by the service. */
export interface QuadWorldSize {
  widthM: number;
  heightM: number;
}

/**
 * Repeat counts that make one tile cover its real manufactured size on the real surface.
 *
 * This used to take a `density` — "about 10 tiles across the photo" — which made every product
 * cover a room in the same number of pieces regardless of its actual size, so 600x600mm porcelain
 * and a 150x900mm plank looked identical. Physical size is the whole point of a visualizer: it is
 * how someone judges whether a tile suits a room.
 *
 * With the patch's real extent known, "tiles across" is simply that extent divided by the tile's
 * width, and the perspective foreshortening is entirely the shader homography's job — the quad is
 * a real plane's projection, so mapping the unit square onto it projectively reproduces exactly
 * what a camera does to a tiled floor.
 */
export function computeTileUv(
  quad: Quad,
  kind: RegionKind,
  tile: TileSizeMm,
  world: QuadWorldSize | null,
): TileUv {
  // Defensive: a catalog entry from a cached payload predating physical sizes has no `size` at
  // all, and reading through it threw rather than degrading. The renderer should never be the
  // thing that breaks over a missing product attribute.
  const safe: TileSizeMm =
    tile && tile.widthMm > 0 && tile.heightMm > 0 ? tile : { widthMm: 600, heightMm: 600, known: false };
  const tileW = safe.widthMm / 1000;
  const tileH = safe.heightMm / 1000;

  let widthM: number;
  let heightM: number;
  if (world && world.widthM > 0.05 && world.heightM > 0.05) {
    widthM = world.widthM;
    heightM = world.heightM;
  } else {
    // The surface didn't fit one plane well enough for the service to measure it. Assume an
    // ordinary room rather than dropping back to a scale-free repeat count: real product sizes
    // then still differ from each other correctly, and only the absolute size is assumed.
    const assumed = ASSUMED_ROOM[kind];
    if (kind === "floor") {
      widthM = assumed.widthM;
      heightM = assumed.otherM;
    } else {
      // Anchor on floor-to-ceiling height, the one wall dimension that is reliably standard, and
      // derive this face's width from its own pixel aspect so every face shares one scale.
      const perPixel = assumed.otherM / Math.max(1, quadHeightPx(quad));
      heightM = assumed.otherM;
      widthM = Math.max(0.2, quadWidthPx(quad) * perPixel);
    }
  }

  // Anchor the pattern's phase to absolute image position, so two quads meeting at a real corner
  // continue one grid rather than each restarting its own. Expressed in tiles: how many tile
  // widths from the image origin this quad's corner sits, using this surface's own metres-per-pixel.
  const metresPerPx = widthM / Math.max(1, quadWidthPx(quad));
  const tileWidthPx = tileW / Math.max(1e-6, metresPerPx);

  return {
    repeatX: Math.max(0.02, widthM / tileW),
    repeatY: Math.max(0.02, heightM / tileH),
    offsetX: quad[0].x / Math.max(1, tileWidthPx),
    offsetY: 0,
  };
}
