import { CatalogItem, RegionKind, TileSizeMm } from "./types";

// Last-resort size for a catalog entry that arrives without one.
//
// The endpoint always sends a size now, but that is not enough on its own to rely on: the route is
// cached (`revalidate`), the client caches the parsed result in module state, and the browser
// caches the response — so right after the size field was introduced, a still-cached payload from
// before it was served and the renderer threw "Cannot read properties of undefined (reading
// 'widthMm')". Normalising on the way in means a stale or hand-edited payload degrades to a
// sensible tile scale instead of crashing the preview.
const FALLBACK_SIZE: TileSizeMm = { widthMm: 600, heightMm: 600, known: false };

function withSize(item: CatalogItem): CatalogItem {
  const s = item.size;
  if (s && Number.isFinite(s.widthMm) && Number.isFinite(s.heightMm) && s.widthMm > 0 && s.heightMm > 0) {
    return item;
  }
  return { ...item, size: FALLBACK_SIZE };
}

let cache: CatalogItem[] | null = null;
let inflight: Promise<CatalogItem[]> | null = null;

/** Fetches the design catalog (auto-discovered from public/flooring and public/tiles), cached in memory. */
export function fetchCatalog(): Promise<CatalogItem[]> {
  if (cache) return Promise.resolve(cache);
  if (!inflight) {
    inflight = fetch("/api/catalog")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load the design catalog.");
        return res.json() as Promise<CatalogItem[]>;
      })
      .then((raw) => {
        const items = (raw ?? []).map(withSize);
        cache = items;
        inflight = null;
        return items;
      })
      .catch((err) => {
        inflight = null;
        throw err;
      });
  }
  return inflight;
}

export function catalogByKind(catalog: CatalogItem[], kind: RegionKind): CatalogItem[] {
  return catalog.filter((item) => item.applicableTo.includes(kind));
}

export function findCatalogItem(catalog: CatalogItem[], id: string): CatalogItem | undefined {
  return catalog.find((item) => item.id === id);
}
