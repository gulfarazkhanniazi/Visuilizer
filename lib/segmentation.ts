import { DetectedRegion, Point, Quad } from "./types";

// The analysis service URL must be provided via NEXT_PUBLIC_ANALYSIS_SERVICE_URL.
// It is injected at build time for client-side code.
const ANALYSIS_SERVICE_URL = process.env.NEXT_PUBLIC_ANALYSIS_SERVICE_URL;
if (!ANALYSIS_SERVICE_URL) {
  throw new Error(
    "NEXT_PUBLIC_ANALYSIS_SERVICE_URL is not defined. Please set it in .env.local",
  );
}

interface AnalyzeRegion {
  id: string;
  kind: "floor" | "wall";
  quads: [number, number][][];
  centroid: [number, number];
  maskPng: string | null;
  maskWidth: number;
  maskHeight: number;
  meanIntensity: number;
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
  console.log('Calling analysis service at', `${ANALYSIS_SERVICE_URL}/analyze`);
  try {
    res = await fetch(`${ANALYSIS_SERVICE_URL}/analyze`, { method: "POST", body: form, mode: "cors" });
  } catch (err) {
    console.error('Analysis service unreachable:', err);
    // A network-level failure (the service is down, unreachable, or CORS-rejected) is not the
    // same situation as a real analysis that legitimately found no floor or wall — returning an
    // empty result here made the two indistinguishable to the caller, so `runDetection` always
    // showed "we couldn't confidently find a floor or wall in this photo" even when the actual
    // problem was that the backend never got a chance to look at it. Throwing the same
    // `AnalysisServiceError` the HTTP-error branch below already uses (rather than inventing a
    // second error path) lets `page.tsx`'s existing catch block show its own, more accurate
    // message instead — no new error-handling machinery needed.
    throw new AnalysisServiceError(
      "Couldn't reach the analysis service. Check your connection and try again.",
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
