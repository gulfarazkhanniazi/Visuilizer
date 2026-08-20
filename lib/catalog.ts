import { CatalogItem, RegionKind } from "./types";

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
      .then((items) => {
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
