/**
 * Monocular depth for the scanner, from the best source available:
 *
 *   1. the CV service (cv-service/server.py): metric depth from
 *      Depth-Anything-V2 Metric-Indoor (or Depth Pro, when opted in), plus LSD
 *      line segments and a focal estimate where the model gives one
 *   2. in-process ONNX Depth-Anything-V2-Small via transformers.js, OPT-IN
 *      (ONNX_DEPTH=on): relative (affine-invariant) disparity, CPU only,
 *      aligned to metric depth against the floor under a level-camera
 *      assumption. Measured on the test photos it is clearly worse than the
 *      service (one room came out as six walls), so by default a missing
 *      service falls back to the junction scanner instead. The service
 *      itself runs on CPU when there is no GPU.
 *   3. nothing -- the caller falls back to the junction-based scanner.
 *
 * No stage throws for an unavailable model: each failure is recorded and the
 * next source is tried, so a missing GPU or a stopped Python process costs
 * accuracy, never the scan.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { DATA_DIR } from '../db.js';

const CV_URL = process.env.CV_SERVICE_URL || 'http://127.0.0.1:5179';
const CV_TIMEOUT_MS = Number(process.env.CV_TIMEOUT_MS || 90000);
const ONNX_DEPTH_MODEL = process.env.ONNX_DEPTH_MODEL || 'onnx-community/depth-anything-v2-small';

let onnxPromise = null;
let serviceDownUntil = 0;

async function postJson(url, body, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || json.success === false) {
      const err = new Error(json.error || json.detail || `HTTP ${res.status}`);
      err.stage = json.stage;
      throw err;
    }
    return json;
  } finally {
    clearTimeout(timer);
  }
}

function decodeF16(b64, n) {
  const buf = Buffer.from(b64, 'base64');
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = u16[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
        : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

async function fromService(imagePath, { quality, lines, depthModel }) {
  if (Date.now() < serviceDownUntil) throw new Error('CV service unavailable (recently unreachable)');
  const image = (await fs.readFile(imagePath)).toString('base64');
  let res;
  try {
    res = await postJson(`${CV_URL}/scan`, { image, options: { quality, lines, depthModel } }, CV_TIMEOUT_MS);
  } catch (e) {
    // Connection refused: do not pay the connect attempt on every scan for
    // the next minute.
    if (e.cause?.code === 'ECONNREFUSED' || /fetch failed/i.test(e.message)) {
      serviceDownUntil = Date.now() + 60_000;
      throw new Error('CV service unavailable');
    }
    throw e;
  }
  const { width, height } = res.depth;
  const conf = res.depth.confidence ? Uint8Array.from(Buffer.from(res.depth.confidence, 'base64')) : null;
  return {
    source: 'cv-service',
    metric: !!res.depth.metric,
    data: decodeF16(res.depth.data, width * height),
    confidence: conf,
    width,
    height,
    focalPx: res.camera?.focalPx ?? null,
    focalSource: res.camera?.focalSource ?? null,
    lines: res.lines ?? [],
    modelVersions: res.metadata?.modelVersions ?? {},
    serviceTimings: res.metadata?.timingsMs ?? {},
    device: res.metadata?.device,
    cached: !!res.metadata?.cached,
  };
}

async function getOnnx() {
  if (!onnxPromise) {
    onnxPromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      env.cacheDir = path.join(DATA_DIR, 'models');
      return pipeline('depth-estimation', ONNX_DEPTH_MODEL, { dtype: 'q8' });
    })().catch((e) => { onnxPromise = null; throw e; });
  }
  return onnxPromise;
}

async function fromOnnx(imagePath) {
  const estimator = await getOnnx();
  const out = await estimator(imagePath);
  // transformers.js returns the raw prediction as a tensor of disparity-like
  // values (larger = nearer) at the model's resolution.
  const t = out.predicted_depth;
  const [height, width] = t.dims.slice(-2);
  return {
    source: 'onnx',
    metric: false,
    data: Float32Array.from(t.data),
    confidence: null,
    width,
    height,
    focalPx: null,
    focalSource: null,
    lines: [],
    modelVersions: { depthModel: ONNX_DEPTH_MODEL, depthLicense: 'Apache-2.0' },
    serviceTimings: {},
  };
}

/**
 * Depth for one photograph. Returns null (with `why` filled in) only when no
 * source could produce any.
 */
export async function estimateDepth(imagePath, {
  quality = 'balanced', lines = true, allowOnnx = true, depthModel = undefined,
} = {}, why = {}) {
  const errors = [];
  if (process.env.CV_SERVICE !== 'off') {
    try {
      return await fromService(imagePath, { quality, lines, depthModel });
    } catch (e) {
      errors.push(`cv-service: ${e.message}`);
    }
  }
  if (allowOnnx && process.env.ONNX_DEPTH === 'on') {
    try {
      return await fromOnnx(imagePath);
    } catch (e) {
      errors.push(`onnx: ${e.message}`);
    }
  }
  why.depthErrors = errors;
  return null;
}

/** SAM 2 masks for box prompts, or null when the service cannot provide them. */
export async function refineWithSam(imagePath, boxes, maxWidth = 512) {
  if (!boxes.length || Date.now() < serviceDownUntil) return null;
  try {
    const image = (await fs.readFile(imagePath)).toString('base64');
    const res = await postJson(`${CV_URL}/refine`, { image, boxes: boxes.map((b) => ({ box: b })), maxWidth }, CV_TIMEOUT_MS);
    return res.masks.map((m) => ({
      width: m.width,
      height: m.height,
      score: m.score,
      data: Uint8Array.from(Buffer.from(m.data, 'base64')),
    }));
  } catch {
    return null;
  }
}

export async function serviceHealth() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(`${CV_URL}/health`, { signal: ctrl.signal });
    clearTimeout(timer);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
