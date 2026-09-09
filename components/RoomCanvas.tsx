"use client";

import { useEffect, useRef, useState } from "react";
import { RoomCompositor, RenderRegion } from "@/lib/perspective";
import { CatalogItem, DesignAssignment, DetectedRegion } from "@/lib/types";
import { computeTileUv } from "@/lib/geometry";
import { findCatalogItem } from "@/lib/catalog";
import { loadImage } from "@/lib/imageCache";
import { ErrorBanner } from "./ErrorBanner";

const DENSITY = { floor: 10, wall: 6 } as const;

// The hotspot's own fixed h-9/w-9 size in px (see `Hotspot` below). Used only to keep the
// displayed circle fully inside the container's visible, `overflow-hidden` bounds; it never touches
// the region's real centroid or any coordinate the backend/WebGL side actually uses.
const HOTSPOT_SIZE_PX = 36;

/** Clamps a 0-1 position fraction so a `sizePx`-wide/tall element centered there (via the same
 * -translate-1/2 the hotspot itself uses) never extends past the edge of a `containerPx`-sized
 * box — a no-op for any position already comfortably inside, which is every hotspot that isn't
 * within half its own size of an edge. Falls back to the raw fraction when the container hasn't
 * been measured yet or is smaller than the hotspot itself (degenerate layout), rather than
 * clamping against a bogus margin that could push the hotspot the wrong way. */
function clampHotspotFraction(fraction: number, containerPx: number, sizePx: number): number {
  if (!containerPx || containerPx <= sizePx) return fraction;
  const marginFrac = sizePx / 2 / containerPx;
  return Math.min(Math.max(fraction, marginFrac), 1 - marginFrac);
}

interface RoomCanvasProps {
  image: HTMLImageElement;
  imgWidth: number;
  imgHeight: number;
  floorRegions: DetectedRegion[];
  wallRegions: DetectedRegion[];
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onDeselectAll: () => void;
  onSelectAllSurfaces: () => void;
  onSelectAllFloor: () => void;
  onSelectAllWall: () => void;
  assignments: DesignAssignment;
  catalog: CatalogItem[];
}

// The hotspot itself doubles as the surface checkbox — a black-bordered square with a visible
// check icon when selected — so there's no separate Floor/Wall toggle to keep in sync with it.
function Hotspot({
  x,
  y,
  selected,
  onClick,
}: {
  x: number;
  y: number;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={selected}
      onClick={onClick}
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
      className={`absolute flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-black shadow-lg transition-colors ${
        selected ? "bg-black" : "bg-white/40 backdrop-blur-sm hover:bg-white/60"
      }`}
    >
      {selected && (
        <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
          <path
            d="M4.5 10.5l3.5 3.5 7.5-8.5"
            stroke="currentColor"
            strokeWidth="2.3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-white"
          />
        </svg>
      )}
    </button>
  );
}

export function RoomCanvas({
  image,
  imgWidth,
  imgHeight,
  floorRegions,
  wallRegions,
  selectedIds,
  onToggleSelect,
  onDeselectAll,
  onSelectAllSurfaces,
  onSelectAllFloor,
  onSelectAllWall,
  assignments,
  catalog,
}: RoomCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const compositorRef = useRef<RoomCompositor | null>(null);
  const textureCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const maskCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const [renderError, setRenderError] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);
  // The hotspot layer positions itself as a percentage of this container's own rendered CSS box
  // (same box the canvas fills via `inset-0 h-full w-full`) — never the WebGL drawing buffer,
  // which is fixed at the source photo's resolution and can be a completely different size (see
  // `RoomCompositor.setRoomImage`). Tracked live via ResizeObserver since that CSS size changes
  // with the viewport (desktop vs. mobile, window resizing) while the underlying image/region
  // coordinates never do.
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setContainerSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // No separate surface toggle — every detected floor and wall part gets its own hotspot on the
  // image at once, and tapping one is both "make it visible/selected" and the design target.
  const activeRegions = [...floorRegions, ...wallRegions];

  useEffect(() => {
    if (!canvasRef.current) return;
    try {
      const compositor = new RoomCompositor(canvasRef.current);
      compositor.setRoomImage(image, imgWidth, imgHeight);
      compositorRef.current = compositor;
      // `region.id` is only unique *within* one photo's analysis (vision.py names them
      // deterministically, e.g. "floor-0") — it repeats across different photos. The mask-texture
      // cache below is keyed by that id and never expires an entry on its own, so if this
      // component instance ever renders a second photo (a new `image`) without unmounting, its
      // "floor-0" would silently reuse the *previous* photo's cached texture instead of building
      // a new one from the current `floorRegions`/`wallRegions` — a real photo showing a stale
      // mask from an old analysis. Tying the cache's lifetime to `image` identity, not to how long
      // the component happens to stay mounted, is what makes a new photo always get a fresh cache.
      for (const tex of maskCacheRef.current.values()) compositor.deleteTexture(tex);
      maskCacheRef.current.clear();
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "This browser doesn't support WebGL2, which the live preview needs.";
      queueMicrotask(() => setRenderError(message));
    }
  }, [image, imgWidth, imgHeight]);

  // Build mask textures once per detected region (masks never change after detection).
  useEffect(() => {
    const compositor = compositorRef.current;
    if (!compositor) return;
    for (const region of [...floorRegions, ...wallRegions]) {
      if (maskCacheRef.current.has(region.id)) continue;
      const tex = compositor.createMaskTexture(region.maskData, region.maskWidth, region.maskHeight);
      maskCacheRef.current.set(region.id, tex);
    }
  }, [floorRegions, wallRegions]);

  useEffect(() => {
    const compositor = compositorRef.current;
    if (!compositor) return;
    let cancelled = false;

    async function getTileTexture(catalogId: string) {
      const item = findCatalogItem(catalog, catalogId);
      if (!item) return null;
      const img = await loadImage(item.src);
      if (cancelled) return null;
      let texture = textureCacheRef.current.get(catalogId);
      if (!texture) {
        texture = compositor!.createTileTexture(img);
        textureCacheRef.current.set(catalogId, texture);
      }
      return { texture, w: img.naturalWidth, h: img.naturalHeight };
    }

    async function draw() {
      // Sized to the container's real CSS display size (times devicePixelRatio), not the source
      // photo's own resolution — a small stock photo displayed at a much wider layout width would
      // otherwise leave the browser to upscale a low-res raster via plain CSS image scaling,
      // blurring every edge in the process, including a mask boundary's subtle anti-aliased
      // feather (see `resizeDrawingBuffer`'s own doc comment). A no-op once already at the right
      // size, so this costs nothing on every other render.
      if (containerSize.width > 0 && containerSize.height > 0) {
        compositor!.resizeDrawingBuffer(containerSize.width, containerSize.height, window.devicePixelRatio || 1);
      }

      if (showOriginal) {
        if (!cancelled) compositor!.render([]);
        return;
      }

      const regions: RenderRegion[] = [];
      for (const kind of ["floor", "wall"] as const) {
        const list = kind === "floor" ? floorRegions : wallRegions;
        for (const region of list) {
          const catalogId = assignments[region.id];
          if (!catalogId) continue;
          const maskTexture = maskCacheRef.current.get(region.id);
          if (!maskTexture) continue;
          const tile = await getTileTexture(catalogId);
          if (!tile) continue;
          // One draw per plane (a wall spanning a real corner has two) so each gets correct
          // perspective orientation; all share this same mask, so a corner still never splits
          // the underlying selection/design, only how the texture is warped on each side of it.
          for (const quad of region.quads) {
            // `imgWidth` is the shared reference every quad's repeat count is anchored to, so a
            // tile reads as the same real-world size on a small wall segment and a large one —
            // see computeTileUv's own doc comment for why a per-quad-only density doesn't do that.
            const { repeatX, repeatY, offsetX, offsetY } = computeTileUv(
              quad,
              tile.w,
              tile.h,
              DENSITY[kind],
              imgWidth,
            );
            regions.push({
              quad,
              texture: tile.texture,
              maskTexture,
              repeatX,
              repeatY,
              offsetX,
              offsetY,
              meanIntensity: region.meanIntensity,
            });
          }
        }
      }

      if (!cancelled) compositor!.render(regions);
    }

    draw().catch((err) => {
      if (!cancelled) {
        setRenderError(err instanceof Error ? err.message : "Couldn't render the preview.");
      }
    });

    return () => {
      cancelled = true;
    };
  }, [floorRegions, wallRegions, assignments, showOriginal, catalog, containerSize]);

  const handleDownload = () => {
    const compositor = compositorRef.current;
    if (!compositor) return;
    const url = compositor.toDataURL();
    const a = document.createElement("a");
    a.href = url;
    a.download = "room-visualization.png";
    a.click();
  };

  return (
    <div className="relative">
      <div className="mb-3 flex flex-wrap items-center justify-end gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={floorRegions.length === 0}
            onClick={onSelectAllFloor}
            className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Select whole floor
          </button>
          <button
            type="button"
            disabled={wallRegions.length === 0}
            onClick={onSelectAllWall}
            className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Select whole wall
          </button>
          <button
            type="button"
            disabled={floorRegions.length === 0 || wallRegions.length === 0}
            onClick={onSelectAllSurfaces}
            className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Select all surfaces
          </button>
          <button
            type="button"
            onClick={() => setShowOriginal((v) => !v)}
            className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            {showOriginal ? "Show design" : "Show original"}
          </button>
          <button
            type="button"
            onClick={handleDownload}
            className="rounded-full bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
          >
            Download image
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative w-full overflow-hidden rounded-2xl bg-zinc-900"
        style={{ aspectRatio: `${imgWidth} / ${imgHeight}` }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        {!showOriginal &&
          activeRegions.map((region) => (
            <Hotspot
              key={region.id}
              x={clampHotspotFraction(region.centroid.x / imgWidth, containerSize.width, HOTSPOT_SIZE_PX)}
              y={clampHotspotFraction(region.centroid.y / imgHeight, containerSize.height, HOTSPOT_SIZE_PX)}
              selected={selectedIds.has(region.id)}
              onClick={() => onToggleSelect(region.id)}
            />
          ))}

        {selectedIds.size > 0 && (
          <button
            type="button"
            onClick={onDeselectAll}
            className="absolute left-1/2 top-4 flex -translate-x-1/2 items-center gap-2 rounded-full bg-zinc-900/85 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur-sm"
          >
            Deselect all
          </button>
        )}
      </div>

      {renderError && (
        <div className="mt-3">
          <ErrorBanner title="Preview unavailable" message={renderError} />
        </div>
      )}
    </div>
  );
}
