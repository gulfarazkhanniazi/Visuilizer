import { readdir } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { CatalogItem, RegionKind } from "@/lib/types";

// Re-scan the public design folders at most this often; keeps the catalog endpoint
// fast under load while still picking up newly added designs within a few minutes.
export const revalidate = 300;

const IMAGE_EXTENSIONS = new Set([".webp", ".jpg", ".jpeg", ".png"]);
// Flooring is floor-only; tiles are commonly rated for both wall and floor installation.
const FOLDERS: { dir: string; applicableTo: RegionKind[] }[] = [
  { dir: "flooring", applicableTo: ["floor"] },
  { dir: "tiles", applicableTo: ["wall", "floor"] },
];

function toLabel(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "");
  return base
    .split(/[-_]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

async function readFolder(dir: string, applicableTo: RegionKind[]): Promise<CatalogItem[]> {
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
    }));
}

export async function GET() {
  const results = await Promise.all(FOLDERS.map((f) => readFolder(f.dir, f.applicableTo)));
  const catalog: CatalogItem[] = results.flat();
  return NextResponse.json(catalog);
}
