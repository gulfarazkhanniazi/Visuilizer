import { readdir } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { CatalogItem, RegionKind } from "@/lib/types";

// Re-scan the public design folders at most this often; keeps the catalog endpoint
// fast under load while still picking up newly added designs within a few minutes.
export const revalidate = 300;

const IMAGE_EXTENSIONS = new Set([".webp", ".jpg", ".jpeg", ".png"]);
// Flooring is floor-only; tiles are commonly rated for both wall and floor installation;
// wall panels mount to a vertical surface only, never a floor.
//
// `defaultSize` is the category's typical manufactured size in millimetres, used only when the
// product's own filename doesn't state one. It has to be *some* real size rather than an abstract
// "tiles across the photo": how much floor one piece covers is a physical fact, and a 600x600mm
// porcelain tile and a 150x900mm plank cover very different amounts of it.
const FOLDERS: { dir: string; applicableTo: RegionKind[]; defaultSize: [number, number] }[] = [
  // A typical LVT / laminate plank.
  { dir: "flooring", applicableTo: ["floor"], defaultSize: [1220, 190] },
  // The most common large-format porcelain square.
  { dir: "tiles", applicableTo: ["wall", "floor"], defaultSize: [600, 600] },
  // Dumawall-style interlocking wall panel.
  { dir: "wall-panels", applicableTo: ["wall"], defaultSize: [375, 650] },
];

// Product names carry the real size when there is one: "600mm-x-600mm", "150-x-900mm",
// "915-x-471-8mm" (the trailing 8mm there is board thickness, not a dimension). Two to four
// digits each, so a pack count or a product code can't be mistaken for a dimension.
const SIZE_PATTERN = /(\d{2,4})\s*(?:mm)?[-_\s]*x[-_\s]*(\d{2,4})\s*(?:mm)?/i;

function parseSize(filename: string, fallback: [number, number]) {
  const m = filename.match(SIZE_PATTERN);
  if (!m) return { widthMm: fallback[0], heightMm: fallback[1], known: false };
  const a = Number(m[1]);
  const b = Number(m[2]);
  // Sanity bounds: a real wall or floor covering piece is between 5cm and 3m on a side.
  if (!(a >= 50 && a <= 3000 && b >= 50 && b <= 3000)) {
    return { widthMm: fallback[0], heightMm: fallback[1], known: false };
  }
  return { widthMm: a, heightMm: b, known: true };
}

function toLabel(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "");
  return base
    .split(/[-_]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

async function readFolder(
  dir: string,
  applicableTo: RegionKind[],
  defaultSize: [number, number],
): Promise<CatalogItem[]> {
  const publicDir = path.join(process.cwd(), "public", dir);
  let entries: string[];
  try {
    entries = await readdir(publicDir);
  } catch {
    return [];
  }
  return entries
    .filter((name) => IMAGE_EXTENSIONS.has(path.extname(name).toLowerCase()))
    .sort()
    .map((name) => ({
      id: `${dir}:${name}`,
      applicableTo,
      label: toLabel(name),
      src: `/${dir}/${name}`,
      size: parseSize(name, defaultSize),
    }));
}

export async function GET() {
  const results = await Promise.all(
    FOLDERS.map((f) => readFolder(f.dir, f.applicableTo, f.defaultSize)),
  );
  const catalog: CatalogItem[] = results.flat();
  return NextResponse.json(catalog);
}
