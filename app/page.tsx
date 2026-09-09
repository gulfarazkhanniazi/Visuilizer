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

export default function Home() {
  const [phase, setPhase] = useState<Phase>("upload");
  const [processingMessage, setProcessingMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [regions, setRegions] = useState<Regions>({ floor: [], wall: [] });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [assignments, setAssignments] = useState<DesignAssignment>({});
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const loadCatalog = useCallback((opts?: { force?: boolean }) => {
    fetchCatalog(opts)
      .then((items) => {
        setCatalog(items);
        // Warm the image cache so applying a design feels instant instead of waiting on a fetch.
        items.forEach((item) => {
          loadImage(item.src).catch(() => {});
        });
      })
      .catch((err) => setCatalogError(err instanceof Error ? err.message : "Failed to load designs."));
  }, []);

  useEffect(() => {
    loadCatalog();
  }, [loadCatalog]);

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
      setSelectedIds(new Set());
      setAssignments({});
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
    setSelectedIds(new Set());
    setAssignments({});
    // Starting a fresh session is the natural point a user expects a design they just added (or
    // moved between folders) to show up — bypass the in-memory catalog cache here rather than
    // requiring a full page reload for it to appear (see `fetchCatalog`'s own doc comment).
    loadCatalog({ force: true });
  };

  const toggleSelect = (id: string) =>
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const deselectAll = () => setSelectedIds(new Set());

  const selectAllSurfaces = () =>
    setSelectedIds(new Set([...regions.floor, ...regions.wall].map((r) => r.id)));

  const selectAllFloor = () => setSelectedIds(new Set(regions.floor.map((r) => r.id)));
  const selectAllWall = () => setSelectedIds(new Set(regions.wall.map((r) => r.id)));

  const applyDesignToSelection = (catalogId: string) =>
    setAssignments((prev) => {
      const next = { ...prev };
      for (const id of selectedIds) next[id] = catalogId;
      return next;
    });

  const applyDesignToIds = (ids: string[], catalogId: string) =>
    setAssignments((prev) => {
      const next = { ...prev };
      for (const id of ids) next[id] = catalogId;
      return next;
    });

  const floorIdSet = useMemo(() => new Set(regions.floor.map((r) => r.id)), [regions]);
  const wallIdSet = useMemo(() => new Set(regions.wall.map((r) => r.id)), [regions]);

  const selectedFloorIds = useMemo(() => [...selectedIds].filter((id) => floorIdSet.has(id)), [selectedIds, floorIdSet]);
  const selectedWallIds = useMemo(() => [...selectedIds].filter((id) => wallIdSet.has(id)), [selectedIds, wallIdSet]);

  // When the selection spans both floors and walls (e.g. via "Select all surfaces"), only
  // designs valid for both kinds make sense to offer in the shared picker — anything floor-only
  // or wall-only would silently do nothing on the other kind's regions if applied from there.
  const selectedKinds = useMemo(() => {
    const kinds = new Set<RegionKind>();
    if (selectedFloorIds.length > 0) kinds.add("floor");
    if (selectedWallIds.length > 0) kinds.add("wall");
    return kinds;
  }, [selectedFloorIds, selectedWallIds]);

  const isMixedSelection = selectedKinds.size > 1;

  const activeCatalog = useMemo(() => {
    const kind = [...selectedKinds][0];
    return kind ? catalogByKind(catalog, kind) : catalog;
  }, [catalog, selectedKinds]);

  // Split out for the mixed-selection case: a shared picker (designs rated for both, e.g. tile)
  // applies to the whole selection, but each surface also gets its own picker underneath for
  // designs that only make sense on it (flooring is floor-only) — picking one of those only
  // reassigns that surface's regions, leaving whatever the shared picker set for the other alone.
  const sharedCatalog = useMemo(
    () => catalog.filter((item) => item.applicableTo.includes("floor") && item.applicableTo.includes("wall")),
    [catalog],
  );
  const floorOnlyCatalog = useMemo(
    () => catalog.filter((item) => item.applicableTo.includes("floor") && !item.applicableTo.includes("wall")),
    [catalog],
  );
  const wallOnlyCatalog = useMemo(
    () => catalog.filter((item) => item.applicableTo.includes("wall") && !item.applicableTo.includes("floor")),
    [catalog],
  );

  const commonDesignFor = (ids: string[]) => {
    const values = ids.map((id) => assignments[id]);
    if (values.length === 0) return undefined;
    return values.every((v) => v === values[0]) ? values[0] : undefined;
  };

  const selectedCommonDesign = useMemo(() => commonDesignFor([...selectedIds]), [selectedIds, assignments]);
  const floorCommonDesign = useMemo(() => commonDesignFor(selectedFloorIds), [selectedFloorIds, assignments]);
  const wallCommonDesign = useMemo(() => commonDesignFor(selectedWallIds), [selectedWallIds, assignments]);

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
                Upload a photo, and we&apos;ll find the floor and walls automatically so you can try
                on real designs.
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
                selectedIds={selectedIds}
                onToggleSelect={toggleSelect}
                onDeselectAll={deselectAll}
                onSelectAllSurfaces={selectAllSurfaces}
                onSelectAllFloor={selectAllFloor}
                onSelectAllWall={selectAllWall}
                assignments={assignments}
                catalog={catalog}
              />
              <p className="text-xs text-zinc-400">
                Tap a highlighted checkbox on the photo to select that part — walls are split at
                each detected corner, so you can give each side its own design. Use &ldquo;Select
                whole floor/wall&rdquo; to apply one design across every part of a surface instead.
              </p>
            </div>

            <div className="flex flex-col gap-4">
              {catalogError && (
                <ErrorBanner title="Couldn't load designs" message={catalogError} />
              )}

              {selectedIds.size === 0 ? (
                <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-zinc-300 p-6 text-center text-sm text-zinc-400 dark:border-zinc-700">
                  Select one or more highlighted areas on the photo to choose a design for them.
                </div>
              ) : isMixedSelection ? (
                <div className="flex flex-col gap-6">
                  <DesignPicker
                    title="Wall + floor design"
                    items={sharedCatalog}
                    selectedId={selectedCommonDesign}
                    onSelect={applyDesignToSelection}
                  />
                  {floorOnlyCatalog.length > 0 && (
                    <DesignPicker
                      title="Floor only"
                      items={floorOnlyCatalog}
                      selectedId={floorCommonDesign}
                      onSelect={(id) => applyDesignToIds(selectedFloorIds, id)}
                    />
                  )}
                  {wallOnlyCatalog.length > 0 && (
                    <DesignPicker
                      title="Wall only"
                      items={wallOnlyCatalog}
                      selectedId={wallCommonDesign}
                      onSelect={(id) => applyDesignToIds(selectedWallIds, id)}
                    />
                  )}
                </div>
              ) : (
                <DesignPicker
                  title={`${[...selectedKinds][0] === "wall" ? "Wall" : "Floor"} design`}
                  items={activeCatalog}
                  selectedId={selectedCommonDesign}
                  onSelect={applyDesignToSelection}
                />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
