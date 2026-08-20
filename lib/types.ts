export type Point = { x: number; y: number };

/** Four corners in TL, TR, BR, BL order, in source-image pixel space — used only for perspective warping. */
export type Quad = [Point, Point, Point, Point];

export type RegionKind = "floor" | "wall";

export interface DetectedRegion {
  id: string;
  kind: RegionKind;
  /** One quad per detected plane (e.g. two for a wall spanning a real corner), each giving that
   * plane its own correct perspective warp — all sharing the same mask/selection/design, so a
   * corner never splits the region itself, only how its texture is warped on each side of it. */
  quads: Quad[];
  /** Where to place the selection hotspot, in source-image pixel space. */
  centroid: Point;
  /** Per-pixel presence mask (at the segmentation model's resolution) — 255 = this region, 0 = not. */
  maskData: Uint8Array;
  maskWidth: number;
  maskHeight: number;
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
