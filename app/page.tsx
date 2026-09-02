"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { UploadDropzone } from "@/components/UploadDropzone";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { ErrorBanner } from "@/components/ErrorBanner";
import { DesignPicker } from "@/components/DesignPicker";
import { RoomCanvas } from "@/components/RoomCanvas";
import { catalogByKind, fetchCatalog } from "@/lib/catalog";
import { loadImage } from "@/lib/imageCache";
import { CatalogItem, DesignAssignment, DetectedRegion, RegionKind } from "@/lib/types";

type Phase = "upload" | "processing" | "workspace";

interface Photo {
  url: string;
  image: HTMLImageElement;
  width: number;
  height: number;
}

interface Regions {
  floor: DetectedRegion[];
  wall: DetectedRegion[];
}

// A surface deliberately put back to the original photo, as opposed to one simply not chosen
// yet. The two need to be distinguishable: the seeded surface falls back to a design when
// unchosen, so "no key" cannot mean "show the photo".
const NO_DESIGN = "__none__";

const SURFACE_ORDER: RegionKind[] = ["floor", "wall"];
const SURFACE_LABEL: Record<RegionKind, string> = { floor: "Floor", wall: "Walls" };
// Singular for the picker heading — "Walls design" reads as a typo where "Wall design" doesn't.
const SURFACE_DESIGN_LABEL: Record<RegionKind, string> = { floor: "Floor", wall: "Wall" };

export default function Home() {
  const [phase, setPhase] = useState<Phase>("upload");
  const [processingMessage, setProcessingMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [regions, setRegions] = useState<Regions>({ floor: [], wall: [] });
  const [assignments, setAssignments] = useState<DesignAssignment>({});
  const [activeSurface, setActiveSurface] = useState<RegionKind>("floor");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  useEffect(() => {
    fetchCatalog()
      .then((items) => {
        setCatalog(items);
        // Warm the image cache so applying a design feels instant instead of waiting on a fetch.
        items.forEach((item) => {
          loadImage(item.src).catch(() => {});
        });
      })
      .catch((err) => setCatalogError(err instanceof Error ? err.message : "Failed to load designs."));
  }, []);

  const runDetection = useCallback(async (file: File) => {
    setPendingFile(file);
    setError(null);
    setPhase("processing");
    setProcessingMessage("Loading your photo…");

    const url = URL.createObjectURL(file);
    const reassureTimer = setTimeout(() => {
      setProcessingMessage("Still working — larger photos can take a little longer to analyze…");
    }, 6000);

    try {
      const fullImage = await loadImage(url);
      const width = fullImage.naturalWidth;
      const height = fullImage.naturalHeight;

      setProcessingMessage("Finding the floor and walls…");

      const { detectFloorAndWall } = await import("@/lib/segmentation");
      const result = await detectFloorAndWall(file, width, height);

      if (result.floor.length === 0 && result.wall.length === 0) {
        setError(
          "We couldn't confidently find a floor or wall in this photo. Try a brighter, wider shot that shows more of the room.",
        );
        setPhase("upload");
        return;
      }

      setPhoto({ url, image: fullImage, width, height });
      setRegions(result);
      setAssignments({});
      setActiveSurface(result.floor.length > 0 ? "floor" : "wall");
      setPhase("workspace");
    } catch (err) {
      console.error(err);
      URL.revokeObjectURL(url);
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong analyzing that photo. Please try again.",
      );
      setPhase("upload");
    } finally {
      clearTimeout(reassureTimer);
    }
  }, []);

  const reset = () => {
    if (photo) URL.revokeObjectURL(photo.url);
    setPhase("upload");
    setError(null);
    setPhoto(null);
    setPendingFile(null);
    setRegions({ floor: [], wall: [] });
    setAssignments({});
  };

  const regionsOf = useCallback(
    (kind: RegionKind) => (kind === "floor" ? regions.floor : regions.wall),
    [regions],
  );

  /** Assigns one design to a whole surface at once — a surface is the unit a design applies to. */
  const applyToSurface = useCallback(
    (kind: RegionKind, catalogId: string) =>
      setAssignments((prev) => {
        const next = { ...prev };
        for (const region of regionsOf(kind)) next[region.id] = catalogId;
        return next;
      }),
    [regionsOf],
  );

  /** Puts one surface back to how it looks in the photo.
   *
   * Needed now that a surface is only painted when it is explicitly chosen: without this, trying
   * a wall design is a one-way door, and the only way back is re-analysing the whole photo from
   * "Start over". The sentinel matters — deleting the key would fall back to the seeded default
   * for the seeded surface, which is a design, not the original photo.
   */
  const clearSurface = useCallback(
    (kind: RegionKind) =>
      setAssignments((prev) => {
        const next = { ...prev };
        for (const region of regionsOf(kind)) next[region.id] = NO_DESIGN;
        return next;
      }),
    [regionsOf],
  );

  // Exactly one surface gets a design on arrival, and only that one — every other surface stays
  // as it looks in the photo until a design is picked for it explicitly.
  //
  // Seeding *every* detected surface was wrong, and visibly so. Someone who came to look at
  // flooring got their walls retiled too, in a design they never chose; and since a wall boundary
  // is much harder than a floor one (pendant lights, curtain edges, the ceiling line), that
  // unrequested wall paint is also where the roughest edges in the whole render are. So it made
  // the app look broken while answering a question nobody asked. One surface at a time is also
  // how the surface toggle already reads: switching to Walls means "now I want to change the
  // walls", not "the walls have been changed for a while and here are the controls".
  //
  // Still derived rather than written into `assignments` on load, so `assignments` holds only
  // real choices — no "already seeded this photo?" flag, and no frame where the room shows
  // nothing while seeding catches up.
  const seededSurface: RegionKind | null = useMemo(() => {
    // Floor first when the photo has one: it is the larger, better-behaved surface and the usual
    // reason someone opens a room visualizer at all. A wall-only photo seeds the wall instead,
    // so the detection still visibly did something.
    if (regionsOf("floor").length > 0) return "floor";
    if (regionsOf("wall").length > 0) return "wall";
    return null;
  }, [regionsOf]);

  const effectiveAssignments = useMemo(() => {
    const next: DesignAssignment = {};
    for (const kind of SURFACE_ORDER) {
      const fallback = kind === seededSurface ? catalogByKind(catalog, kind)[0] : undefined;
      for (const region of regionsOf(kind)) {
        const design = assignments[region.id] ?? fallback?.id;
        if (design && design !== NO_DESIGN) next[region.id] = design;
      }
    }
    return next;
  }, [assignments, catalog, regionsOf, seededSurface]);

  const availableSurfaces = useMemo(
    () => SURFACE_ORDER.filter((kind) => regionsOf(kind).length > 0),
    [regionsOf],
  );

  const activeCatalog = useMemo(
    () => catalogByKind(catalog, activeSurface),
    [catalog, activeSurface],
  );

  const activeDesignId = useMemo(() => {
    const list = regionsOf(activeSurface);
    return list.length > 0 ? effectiveAssignments[list[0].id] : undefined;
  }, [regionsOf, activeSurface, effectiveAssignments]);

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 dark:bg-black">
      <header className="border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <span className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            Room Visualizer
          </span>
          {phase === "workspace" && (
            <button
              onClick={reset}
              className="text-sm font-medium text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
            >
              Start over
            </button>
          )}
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-10">
        {phase !== "workspace" && (
          <div className="mx-auto flex w-full max-w-xl flex-1 flex-col justify-center gap-6">
            <div className="text-center">
              <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
                See new flooring and tiles in your own room
              </h1>
              <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
                Upload a photo, and we&apos;ll find the floor and walls automatically and apply a
                design straight away.
              </p>
            </div>

            {error && (
              <ErrorBanner
                title="Couldn't process that photo"
                message={error}
                onRetry={pendingFile ? () => runDetection(pendingFile) : undefined}
                onDismiss={() => setError(null)}
              />
            )}

            <UploadDropzone onFile={runDetection} />
          </div>
        )}

        {phase === "processing" && (
          <div className="relative min-h-[50vh] flex-1 rounded-2xl border border-zinc-200 dark:border-zinc-800">
            <LoadingOverlay message={processingMessage} />
          </div>
        )}

        {phase === "workspace" && photo && (
          <div className="grid flex-1 grid-cols-1 gap-8 lg:grid-cols-[1fr_360px]">
            <div className="flex flex-col gap-3">
              <RoomCanvas
                image={photo.image}
                imgWidth={photo.width}
                imgHeight={photo.height}
                floorRegions={regions.floor}
                wallRegions={regions.wall}
                assignments={effectiveAssignments}
                catalog={catalog}
              />
              <p className="text-xs text-zinc-400">
                Pick a design and it covers the whole surface at once — the floor is one surface and
                the walls are another, each kept in correct perspective around every corner the
                photo actually has.
              </p>
            </div>

            <div className="flex flex-col gap-4">
              {catalogError && (
                <ErrorBanner title="Couldn't load designs" message={catalogError} />
              )}

              {availableSurfaces.length > 1 && (
                <div className="flex rounded-full border border-zinc-200 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900">
                  {availableSurfaces.map((kind) => (
                    <button
                      key={kind}
                      type="button"
                      onClick={() => setActiveSurface(kind)}
                      aria-pressed={activeSurface === kind}
                      className={`flex-1 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                        activeSurface === kind
                          ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                          : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                      }`}
                    >
                      {SURFACE_LABEL[kind]}
                    </button>
                  ))}
                </div>
              )}

              {activeCatalog.length === 0 ? (
                <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-zinc-300 p-6 text-center text-sm text-zinc-400 dark:border-zinc-700">
                  No designs available for this surface yet.
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <DesignPicker
                    title={`${SURFACE_DESIGN_LABEL[activeSurface]} design`}
                    items={activeCatalog}
                    selectedId={activeDesignId}
                    onSelect={(id) => applyToSurface(activeSurface, id)}
                  />
                  {activeDesignId ? (
                    <button
                      type="button"
                      onClick={() => clearSurface(activeSurface)}
                      className="self-start text-xs font-medium text-zinc-500 underline decoration-dotted hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
                    >
                      Remove {SURFACE_DESIGN_LABEL[activeSurface].toLowerCase()} design
                    </button>
                  ) : (
                    <p className="text-xs text-zinc-400">
                      Showing the original {SURFACE_DESIGN_LABEL[activeSurface].toLowerCase()} —
                      pick a design to change it.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
