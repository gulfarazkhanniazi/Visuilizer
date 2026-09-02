export type Point = { x: number; y: number };

/** Four corners in TL, TR, BR, BL order, in source-image pixel space — used only for perspective warping. */
export type Quad = [Point, Point, Point, Point];

export type RegionKind = "floor" | "wall";

export interface DetectedRegion {
  id: string;
  kind: RegionKind;
  /** One perspective quad per real plane inside this surface — a wall carries one per
   * architectural face, so a design stays correctly warped around every corner in the photo, and
   * a floor carries one per fitted band. All of them share this region's single mask: a corner
   * changes how the texture is warped across it, never which surface the design belongs to. */
  quads: Quad[];
  /** Most-interior point of the surface, in source-image pixel space. */
  centroid: Point;
  /** The real size, in metres, of the surface patch each quad covers — aligned with `quads`,
   * measured by the service from metric depth. This is what lets a 600x600mm tile cover 0.36m of
   * real surface. null when the surface didn't fit one plane well enough to measure honestly, in
   * which case the renderer falls back to an assumed room size. */
  worldQuads: { widthM: number; heightM: number }[] | null;
  /** Per-pixel presence mask (at the segmentation model's resolution) — 255 = this region, 0 = not. */
  maskData: Uint8Array;
  maskWidth: number;
  maskHeight: number;
  meanIntensity: number;
}

/** A design's real manufactured size. Tile scale is only meaningful in physical units: 600x600mm
 * porcelain and a 150x900mm plank must not cover a room in the same number of pieces. */
export interface TileSizeMm {
  widthMm: number;
  heightMm: number;
  /** False when the size is a category default rather than read from the product itself. */
  known: boolean;
}

export interface CatalogItem {
  id: string;
  /** Which surface(s) this design can be applied to — tiles are typically wall+floor, flooring is floor-only. */
  applicableTo: RegionKind[];
  label: string;
  src: string;
  size: TileSizeMm;
}

/** Maps a detected region's id to the chosen catalog item id. */
export type DesignAssignment = Record<string, string>;
