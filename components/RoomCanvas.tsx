"use client";

import { useEffect, useRef, useState } from "react";
import { RoomCompositor, RenderRegion } from "@/lib/perspective";
import { CatalogItem, DesignAssignment, DetectedRegion, RegionKind } from "@/lib/types";
import { computeRepeat } from "@/lib/geometry";
import { findCatalogItem } from "@/lib/catalog";
import { loadImage } from "@/lib/imageCache";
import { ErrorBanner } from "./ErrorBanner";

const DENSITY = { floor: 10, wall: 6 } as const;

interface RoomCanvasProps {
  image: HTMLImageElement;
  imgWidth: number;
  imgHeight: number;
  floorRegions: DetectedRegion[];
  wallRegions: DetectedRegion[];
  activeKind: RegionKind;
  onActiveKindChange: (kind: RegionKind) => void;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onDeselectAll: () => void;
  assignments: DesignAssignment;
  catalog: CatalogItem[];
}

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
      onClick={onClick}
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
      className={`absolute flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white shadow-lg transition-colors ${
        selected ? "bg-zinc-900" : "bg-white/40 backdrop-blur-sm hover:bg-white/60"
      }`}
    >
      {selected && (
        <svg viewBox="0 0 20 20" fill="white" className="h-4 w-4">
          <path
            fillRule="evenodd"
            d="M16.704 5.29a1 1 0 0 1 0 1.415l-7.5 7.5a1 1 0 0 1-1.414 0l-3.5-3.5a1 1 0 1 1 1.414-1.414L8.5 12.086l6.79-6.796a1 1 0 0 1 1.414 0Z"
            clipRule="evenodd"
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
  activeKind,
  onActiveKindChange,
  selectedIds,
  onToggleSelect,
  onDeselectAll,
  assignments,
  catalog,
}: RoomCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const compositorRef = useRef<RoomCompositor | null>(null);
  const textureCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const maskCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const [renderError, setRenderError] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);

  const activeRegions = activeKind === "floor" ? floorRegions : wallRegions;

  useEffect(() => {
    if (!canvasRef.current) return;
    try {
      const compositor = new RoomCompositor(canvasRef.current);
      compositor.setRoomImage(image, imgWidth, imgHeight);
      compositorRef.current = compositor;
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
            const { repeatX, repeatY } = computeRepeat(quad, tile.w, tile.h, DENSITY[kind]);
            regions.push({ quad, texture: tile.texture, maskTexture, repeatX, repeatY });
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
  }, [floorRegions, wallRegions, assignments, showOriginal, catalog]);

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
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex gap-1 rounded-full bg-zinc-100 p-1 dark:bg-zinc-800">
          <button
            type="button"
            disabled={floorRegions.length === 0}
            onClick={() => onActiveKindChange("floor")}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              activeKind === "floor"
                ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-950 dark:text-zinc-100"
                : floorRegions.length === 0
                  ? "cursor-not-allowed text-zinc-400 dark:text-zinc-600"
                  : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-100"
            }`}
          >
            Floors
          </button>
          <button
            type="button"
            disabled={wallRegions.length === 0}
            onClick={() => onActiveKindChange("wall")}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              activeKind === "wall"
                ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-950 dark:text-zinc-100"
                : wallRegions.length === 0
                  ? "cursor-not-allowed text-zinc-400 dark:text-zinc-600"
                  : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-100"
            }`}
          >
            Walls
          </button>
        </div>

        <div className="flex items-center gap-2">
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
        className="relative w-full overflow-hidden rounded-2xl bg-zinc-900"
        style={{ aspectRatio: `${imgWidth} / ${imgHeight}` }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        {!showOriginal &&
          activeRegions.map((region) => (
            <Hotspot
              key={region.id}
              x={region.centroid.x / imgWidth}
              y={region.centroid.y / imgHeight}
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
