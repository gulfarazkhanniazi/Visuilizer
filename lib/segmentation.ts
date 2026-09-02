import { DetectedRegion, Point, Quad } from "./types";

const ANALYSIS_SERVICE_URL =
  process.env.NEXT_PUBLIC_ANALYSIS_SERVICE_URL ?? "http://localhost:8000";

interface AnalyzeRegion {
  id: string;
  kind: "floor" | "wall";
  quads: [number, number][][];
  centroid: [number, number];
  maskPng: string | null;
  maskWidth: number;
  maskHeight: number;
  meanIntensity: number;
  worldQuads: { widthM: number; heightM: number }[] | null;
}

interface AnalyzeResponse {
  floor: AnalyzeRegion[];
  wall: AnalyzeRegion[];
}

/** Decodes a base64 PNG mask into a flat single-channel presence array (grayscale, so R=G=B). */
async function decodeMaskPng(base64: string, width: number, height: number): Promise<Uint8Array> {
  const blob = await (await fetch(`data:image/png;base64,${base64}`)).blob();
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const out = new Uint8Array(width * height);
  if (!ctx) return out;
  ctx.drawImage(bitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = data[p];
  return out;
}

async function toDetectedRegion(kind: "floor" | "wall", r: AnalyzeRegion): Promise<DetectedRegion | null> {
  if (!r.maskPng) return null;
  const maskData = await decodeMaskPng(r.maskPng, r.maskWidth, r.maskHeight);
  const quads = r.quads.map((quad) => quad.map(([x, y]) => ({ x, y }))) as Quad[];
  const centroid: Point = { x: r.centroid[0], y: r.centroid[1] };
  return {
    id: r.id,
    kind,
    quads,
    centroid,
    worldQuads: r.worldQuads ?? null,
    maskData,
    maskWidth: r.maskWidth,
    maskHeight: r.maskHeight,
    meanIntensity: r.meanIntensity ?? 0.5,
  };
}

export interface DetectionResult {
  floor: DetectedRegion[];
  wall: DetectedRegion[];
}

export class AnalysisServiceError extends Error {}

export async function detectFloorAndWall(
  file: Blob,
  imgWidth: number,
  imgHeight: number,
): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("target_width", String(imgWidth));
  form.append("target_height", String(imgHeight));

  let res: Response;
  try {
    res = await fetch(`${ANALYSIS_SERVICE_URL}/analyze`, { method: "POST", body: form });
  } catch {
    throw new AnalysisServiceError(
      "Couldn't reach the analysis service. Make sure the Python service (python-service/) is running.",
    );
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new AnalysisServiceError(detail?.detail ?? `Analysis failed (${res.status}).`);
  }

  const data: AnalyzeResponse = await res.json();
  const floor = (await Promise.all(data.floor.map((r) => toDetectedRegion("floor", r)))).filter(
    (r): r is DetectedRegion => r !== null,
  );
  const wall = (await Promise.all(data.wall.map((r) => toDetectedRegion("wall", r)))).filter(
    (r): r is DetectedRegion => r !== null,
  );

  return { floor, wall };
}
