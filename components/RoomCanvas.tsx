"use client";

import { useEffect, useRef, useState } from "react";
import { RoomCompositor, RenderRegion } from "@/lib/perspective";
import { CatalogItem, DesignAssignment, DetectedRegion } from "@/lib/types";
import { computeTileUv } from "@/lib/geometry";
import { findCatalogItem } from "@/lib/catalog";
import { loadImage } from "@/lib/imageCache";
import { ErrorBanner } from "./ErrorBanner";

// How much of the photo's own light and shadow each surface carries into its design.
//
// Both surfaces were previously on 0.3 strength with a [0.75, 1.3] bound, which in practice meant
// no shading at all: a real contact shadow under a sofa comes in around a 0.4 ratio, gets clamped
// up to 0.75, then blended 30% of the way there, arriving at 0.925 - a 7% darkening where the
// photo had deep shadow. That is why a design read as a flat sticker laid over the room instead of
// a material sitting in it, and it is the single largest difference from a reference visualizer,
// whose floors show clear shadows under every piece of furniture.
//
// The floor gets the wider range and stronger blend: it is where contact shadows live, it is lit
// indirectly so its own colour cast is weak, and furniture sitting on it is what sells the effect.
// The wall stays tighter - it is more likely to carry a strong colour cast from the old paint, and
// its job is mainly keeping adjacent corner planes distinguishable.
const SHADING = {
  floor: { strength: 0.7, range: [0.4, 1.5] as [number, number] },
  wall: { strength: 0.45, range: [0.62, 1.38] as [number, number] },
} as const;

// Ceiling on the drawing buffer, matching the analysis service's own mask encode cap - past this
// there is no mask detail left to reveal, so a bigger buffer would cost memory for nothing. A
// photo larger than this is drawn at 2600 and downscaled, which loses nothing visible: the frame
// is never displayed wider than the photo, so it is being minified regardless.
const MAX_RENDER_WIDTH = 2600;

interface RoomCanvasProps {
  image: HTMLImageElement;
  imgWidth: number;
  imgHeight: number;
  floorRegions: DetectedRegion[];
  wallRegions: DetectedRegion[];
  assignments: DesignAssignment;
  catalog: CatalogItem[];
}

export function RoomCanvas({
  image,
  imgWidth,
  imgHeight,
  floorRegions,
  wallRegions,
  assignments,
  catalog,
}: RoomCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const compositorRef = useRef<RoomCompositor | null>(null);
  const textureCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const maskCacheRef = useRef<Map<string, WebGLTexture>>(new Map());
  const [renderError, setRenderError] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);

  // Never display the photo larger than it actually is.
  //
  // The frame used to fill its column whatever the photo's size, so a 547x365 stock photo was
  // stretched across ~890 CSS px - a 1.6x magnification - and no amount of filtering can invent
  // the detail that implies. Capping the frame at the photo's own pixel count means a small photo
  // is shown at 1:1 and stays genuinely sharp, while a large one still shrinks to fit as before.
  //
  // Set on the DOM node in an effect rather than through React state on purpose: it depends on
  // `devicePixelRatio`, which does not exist during server rendering, so putting it in the
  // rendered output would risk a hydration mismatch for a value that is purely presentational.
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    frame.style.maxWidth = `${Math.round(imgWidth / dpr)}px`;

    // Snap the frame onto whole device pixels.
    //
    // Displaying the photo at 1:1 only pays off if its pixels land on the screen's pixels. Centred
    // in a column of odd width, and stacked under rows of fractional height, the canvas was sitting
    // at x=280.5, y=142.39 — so the browser resampled the entire image by a sub-pixel shift on the
    // way to the screen. Measured, that alone destroyed nearly two thirds of the photo's fine
    // detail while the canvas itself was pixel-perfect: a defect entirely invisible in the app's
    // own output, and only findable by comparing what is actually on screen against the file.
    //
    // A sub-pixel translation is imperceptible as movement but makes the raster align exactly.
    const snap = () => {
      frame.style.transform = "";
      const rect = frame.getBoundingClientRect();
      const dx = (Math.round(rect.x * dpr) - rect.x * dpr) / dpr;
      const dy = (Math.round(rect.y * dpr) - rect.y * dpr) / dpr;
      frame.style.transform = dx || dy ? `translate(${dx}px, ${dy}px)` : "";
    };
    snap();

    // Layout moves for reasons this component never sees — a sidebar reflowing, the window
    // resizing, fonts settling — and each one can reintroduce the fractional offset.
    const observer = new ResizeObserver(snap);
    observer.observe(document.body);
    window.addEventListener("resize", snap);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", snap);
    };
  }, [imgWidth]);

  useEffect(() => {
    if (!canvasRef.current) return;
    try {
      const compositor = new RoomCompositor(canvasRef.current);
      // Size the drawing buffer to how big the canvas actually is on screen, not to the source
      // photo. A small stock photo would otherwise render into a small buffer that CSS stretches
      // back up, softening the tile detail and mask edges that the pipeline worked to preserve.
      // Never smaller than the photo itself, so a large photo keeps its full detail.
      const displayWidth = canvasRef.current.clientWidth || imgWidth;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      // Exactly the device pixels on screen, or the photo's own resolution if that is larger —
      // and deliberately no more.
      //
      // This used to draw at 2x the display size and let the browser scale down, on the reasoning
      // that supersampling anti-aliases the tile pattern. Measured, it cost far more than it
      // bought: the photo was upscaled into the oversized buffer and then downscaled back out, and
      // that round trip is a low-pass filter, not an identity. It retained only 44% of the source
      // photo's detail. Nothing needed it either — the mask arrives already feathered into a soft
      // alpha matte by the analysis service, so surface boundaries are anti-aliased in the data
      // rather than relying on extra samples here, and the tile textures carry mipmaps.
      const renderWidth = Math.min(
        MAX_RENDER_WIDTH,
        Math.max(imgWidth, Math.round(displayWidth * dpr)),
      );
      const renderHeight = Math.round((renderWidth * imgHeight) / imgWidth);
      compositor.setRoomImage(image, imgWidth, imgHeight, renderWidth, renderHeight);
      compositorRef.current = compositor;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "This browser doesn't support WebGL2, which the live preview needs.";
      queueMicrotask(() => setRenderError(message));
    }
  }, [image, imgWidth, imgHeight]);

  // Build mask textures once per detected surface (masks never change after detection).
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
      return { texture, w: img.naturalWidth, h: img.naturalHeight, size: item.size };
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
          // One draw per plane — a wall spanning real architectural corners has one quad per
          // face, and a floor fit as stacked bands has one per band — so each gets correct
          // perspective orientation. All of a surface's quads share that surface's single mask,
          // so a corner only changes how the texture is warped across it, never which surface
          // the design belongs to.
          for (const [quadIndex, quad] of region.quads.entries()) {
            // Repeat counts come from the tile's real manufactured size and the measured real
            // size of the patch this quad covers, so one tile covers its true area.
            const { repeatX, repeatY, offsetX, offsetY } = computeTileUv(
              quad,
              kind,
              tile.size,
              region.worldQuads?.[quadIndex] ?? null,
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
              shadingStrength: SHADING[kind].strength,
              shadingRange: SHADING[kind].range,
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
      <div className="mb-3 flex flex-wrap items-center justify-end gap-2">
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

      <div
        ref={frameRef}
        className="relative mx-auto w-full overflow-hidden rounded-2xl bg-zinc-900"
        style={{ aspectRatio: `${imgWidth} / ${imgHeight}` }}
      >
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      </div>

      {renderError && (
        <div className="mt-3">
          <ErrorBanner title="Preview unavailable" message={renderError} />
        </div>
      )}
    </div>
  );
}
