export type Point = { x: number; y: number };

/** Four corners in TL, TR, BR, BL order, in source-image pixel space — used only for perspective warping. */
export type Quad = [Point, Point, Point, Point];

export type RegionKind = "floor" | "wall";

export interface DetectedRegion {
  id: string;
  kind: RegionKind;
  /** Perspective quad(s) for this region's own plane. A wall spanning a real architectural
   * corner is detected as separate regions, one per plane, each with its own mask, selection,
   * and design, and a single quad here. */
  quads: Quad[];
  /** Where to place the selection hotspot, in source-image pixel space. */
  centroid: Point;
  /** Per-pixel presence mask (at the segmentation model's resolution) — 255 = this region, 0 = not. */
  maskData: Uint8Array;
  maskWidth: number;
  maskHeight: number;
  meanIntensity: number;
}

export interface CatalogItem {
  id: string;
  /** Which surface(s) this design can be applied to — tiles are typically wall+floor, flooring is floor-only. */
  applicableTo: RegionKind[];
  label: string;
  src: string;
}

/** Maps a detected region's id to the chosen catalog item id. */
export type DesignAssignment = Record<string, string>;
