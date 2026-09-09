import { CatalogItem, RegionKind } from "./types";

let cache: CatalogItem[] | null = null;
let inflight: Promise<CatalogItem[]> | null = null;

/** Fetches the design catalog (auto-discovered from public/flooring, public/tiles, and
 * public/wall-panels), cached in memory for the lifetime of this page load — the catalog rarely
 * changes mid-session, and re-fetching on every render would be wasteful. Pass `force: true` to
 * bypass that cache: a design dropped into (or moved between) those folders while this tab has
 * been open otherwise stays invisible until a full page reload, since the cache has no other way
 * to know it's gone stale. `page.tsx` calls this with `force: true` at the natural point a user
 * would expect a newly added design to show up — starting a fresh session. */
export function fetchCatalog(opts?: { force?: boolean }): Promise<CatalogItem[]> {
  if (cache && !opts?.force) return Promise.resolve(cache);
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
