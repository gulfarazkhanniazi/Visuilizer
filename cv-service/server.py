"""
Computer-vision service for the room scanner.

Express stays the application API; this process only does the work that needs
PyTorch -- monocular metric depth, line segments and (optionally) SAM 2 mask
refinement -- and hands back plain arrays. All wall/corner geometry is done by
the Node scanner (server/src/scanner), so it runs with or without this service.

    POST /scan     image (base64) + options -> depth map, focal, line segments
    POST /refine   image + box prompts        -> SAM 2 masks          (quality=high)
    GET  /health   which models are loaded, which device

Run:  python3 cv-service/server.py            (listens on 127.0.0.1:5179)

Model choice is about licence as much as accuracy. The default depth model,
Depth-Anything-V2 Metric-Indoor *Small*, is Apache-2.0. The Base/Large variants
(CC-BY-NC-4.0) and Apple's Depth Pro (apple-amlr, research only) are supported
for evaluation but must be opted into with DEPTH_MODEL_<QUALITY>, because a
commercial deployment may not be allowed to ship them.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

GEOMETRY_API_VERSION = "cv-1.0.0"

MODELS = {
    "da2-metric-indoor-small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "Apache-2.0"),
    "da2-metric-indoor-base": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", "CC-BY-NC-4.0"),
    "da2-metric-indoor-large": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "CC-BY-NC-4.0"),
    "depth-pro": ("apple/DepthPro-hf", "apple-amlr (research only)"),
}
DEFAULT_BY_QUALITY = {
    "fast": os.environ.get("DEPTH_MODEL_FAST", "da2-metric-indoor-small"),
    "balanced": os.environ.get("DEPTH_MODEL_BALANCED", "da2-metric-indoor-small"),
    "high": os.environ.get("DEPTH_MODEL_HIGH", "da2-metric-indoor-small"),
}
SAM_MODEL = os.environ.get("SAM_MODEL", "facebook/sam2.1-hiera-small")

# Depth goes back at most this wide. The scanner's geometry grid is 256-512
# columns; shipping more is bandwidth for nothing.
MAX_DEPTH_W = int(os.environ.get("MAX_DEPTH_W", "640"))


def pick_device() -> str:
    forced = os.environ.get("CV_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
_load_lock = threading.Lock()
_run_lock = threading.Lock()          # one inference at a time: GPU memory is the limit
_depth_models: dict[str, tuple] = {}
_sam = None
_cache: "OrderedDict[str, dict]" = OrderedDict()
CACHE_SIZE = 16


def _load_depth(key: str):
    if key not in MODELS:
        raise HTTPException(400, f"unknown depth model '{key}'")
    with _load_lock:
        if key in _depth_models:
            return _depth_models[key]
        repo, _ = MODELS[key]
        t0 = time.time()
        if key == "depth-pro":
            from transformers import DepthProForDepthEstimation, DepthProImageProcessor
            proc = DepthProImageProcessor.from_pretrained(repo)
            model = DepthProForDepthEstimation.from_pretrained(repo, torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32)
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForDepthEstimation.from_pretrained(repo)
        device = DEVICE
        try:
            model = model.to(device).eval()
        except Exception:                                     # noqa: BLE001 - any device failure means CPU
            device = "cpu"
            model = model.float().to(device).eval()
        _depth_models[key] = (proc, model, device)
        print(f"  cv  loaded {repo} on {device} in {time.time() - t0:.1f}s", flush=True)
        return _depth_models[key]


def _decode_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception as e:                                    # noqa: BLE001
        raise HTTPException(400, f"invalid image: {e}") from e


def _exif_focal_px(b64: str, width: int) -> float | None:
    """35mm-equivalent focal length from EXIF, in pixels of this image."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[-1])))
        exif = img.getexif()
        ifd = exif.get_ifd(0x8769)
        f35 = ifd.get(0xA405)                                 # FocalLengthIn35mmFilm
        if f35 and float(f35) > 5:
            return float(f35) / 36.0 * width                  # 36 mm = full-frame width
    except Exception:                                         # noqa: BLE001
        pass
    return None


def run_depth(img: Image.Image, key: str) -> tuple[np.ndarray, float | None, str]:
    proc, model, device = _load_depth(key)
    W, H = img.size
    with _run_lock, torch.inference_mode():
        inputs = proc(images=img, return_tensors="pt")
        dtype = next(model.parameters()).dtype
        inputs = {k: (v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}
        out = model(**inputs)
        post = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
    depth = post["predicted_depth"].float().cpu().numpy()
    focal = None
    if "focal_length" in post and post["focal_length"] is not None:
        focal = float(post["focal_length"])
    return depth, focal, device


def depth_confidence(depth: np.ndarray) -> np.ndarray:
    """
    Per-pixel confidence proxy: none of these models ship an uncertainty head,
    so use local relative depth variation. Monocular depth is least reliable
    exactly where it changes fastest -- object silhouettes -- which is also
    where a plane fit should trust it least.
    """
    d = depth.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    rel = np.sqrt(gx * gx + gy * gy) / np.maximum(d, 1e-3)
    return np.clip(1.0 - rel * 4.0, 0.0, 1.0)


def line_segments(img: Image.Image, max_w: int = 1024) -> tuple[list, float]:
    """LSD line segments in original image pixels, longest first."""
    W, H = img.size
    s = min(1.0, max_w / W)
    g = cv2.cvtColor(np.asarray(img.resize((round(W * s), round(H * s)))), cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(g)[0]
    if lines is None:
        return [], s
    lines = lines.reshape(-1, 4) / s
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    keep = lengths > 0.02 * np.hypot(W, H)
    lines, lengths = lines[keep], lengths[keep]
    order = np.argsort(-lengths)[:600]
    return [[round(float(v), 1) for v in lines[i]] for i in order], s


def encode_f16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode()


def encode_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


class ScanOptions(BaseModel):
    quality: str = "balanced"
    debug: bool = False
    depthModel: str | None = None
    lines: bool = True


class ScanRequest(BaseModel):
    image: str
    options: ScanOptions = Field(default_factory=ScanOptions)


class Box(BaseModel):
    box: list[float]                 # x0, y0, x1, y1 in image pixels
    points: list[list[float]] = []   # optional positive clicks


class RefineRequest(BaseModel):
    image: str
    boxes: list[Box]
    maxWidth: int = 512


app = FastAPI(title="scanner-cv", version=GEOMETRY_API_VERSION)


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "loaded": sorted(_depth_models),
        "defaults": DEFAULT_BY_QUALITY,
        "models": {k: {"repo": v[0], "license": v[1]} for k, v in MODELS.items()},
        "version": GEOMETRY_API_VERSION,
    }


@app.post("/scan")
def scan(req: ScanRequest):
    t_all = time.time()
    opts = req.options
    if opts.quality not in DEFAULT_BY_QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(DEFAULT_BY_QUALITY)}")
    key = opts.depthModel or DEFAULT_BY_QUALITY[opts.quality]

    digest = hashlib.sha1(req.image.encode()).hexdigest()
    cache_key = f"{digest}:{key}:{opts.lines}"
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        hit = dict(_cache[cache_key])
        hit["metadata"] = {**hit["metadata"], "cached": True}
        return hit

    timings = {}
    t = time.time()
    img = _decode_image(req.image)
    W, H = img.size
    timings["decode"] = round((time.time() - t) * 1000)

    t = time.time()
    try:
        depth, focal, device = run_depth(img, key)
    except HTTPException:
        raise
    except Exception as e:                                    # noqa: BLE001
        return JSONResponse(status_code=503, content={"success": False, "error": f"depth failure: {e}", "stage": "depth"})
    timings["depth"] = round((time.time() - t) * 1000)
    if not np.isfinite(depth).all() or depth.max() <= 0:
        return JSONResponse(status_code=422, content={"success": False, "error": "depth model returned invalid values", "stage": "depth"})

    # Down-sample for transfer. INTER_AREA averages, so a thin object does not
    # alias into a speckle of wrong depth.
    dw = min(W, MAX_DEPTH_W)
    dh = max(1, round(H * dw / W))
    small = cv2.resize(depth, (dw, dh), interpolation=cv2.INTER_AREA) if dw != W else depth
    conf = depth_confidence(small)

    lines = []
    if opts.lines:
        t = time.time()
        lines, _ = line_segments(img)
        timings["lines"] = round((time.time() - t) * 1000)

    exif_f = _exif_focal_px(req.image, W)
    camera = {"width": W, "height": H}
    if focal:
        camera.update(focalPx=focal, focalSource=key)
    elif exif_f:
        camera.update(focalPx=exif_f, focalSource="exif")

    timings["total"] = round((time.time() - t_all) * 1000)
    result = {
        "success": True,
        "depth": {
            "width": dw, "height": dh, "dtype": "float16", "metric": True,
            "data": encode_f16(small),
            "confidence": encode_u8(np.round(conf * 255)),
            "min": float(small.min()), "max": float(small.max()),
        },
        "camera": camera,
        "lines": lines,
        "metadata": {
            "processingTimeMs": timings["total"],
            "timingsMs": timings,
            "device": device,
            "modelVersions": {"depthModel": MODELS[key][0], "depthLicense": MODELS[key][1], "cvService": GEOMETRY_API_VERSION},
            "cached": False,
        },
    }
    _cache[cache_key] = result
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def _load_sam():
    global _sam
    with _load_lock:
        if _sam is None:
            from transformers import Sam2Model, Sam2Processor
            proc = Sam2Processor.from_pretrained(SAM_MODEL)
            model = Sam2Model.from_pretrained(SAM_MODEL).to(DEVICE).eval()
            _sam = (proc, model)
            print(f"  cv  loaded {SAM_MODEL} on {DEVICE}", flush=True)
    return _sam


@app.post("/refine")
def refine(req: RefineRequest):
    """
    SAM 2 masks for a handful of box prompts -- the objects whose silhouettes
    the semantic mask is least sure of. Masks come back at `maxWidth`, which is
    all the scanner's refinement grid uses.
    """
    t0 = time.time()
    if not req.boxes:
        return {"success": True, "masks": [], "metadata": {"processingTimeMs": 0}}
    img = _decode_image(req.image)
    try:
        proc, model = _load_sam()
    except Exception as e:                                    # noqa: BLE001
        return JSONResponse(status_code=503, content={"success": False, "error": f"SAM unavailable: {e}", "stage": "sam"})
    W, H = img.size
    boxes = [[b.box for b in req.boxes]]
    with _run_lock, torch.inference_mode():
        inputs = proc(images=img, input_boxes=boxes, return_tensors="pt").to(DEVICE)
        out = model(**inputs, multimask_output=False)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
    scores = out.iou_scores.cpu().numpy().reshape(-1)
    mw = min(W, req.maxWidth)
    mh = max(1, round(H * mw / W))
    res = []
    for i in range(masks.shape[0]):
        m = masks[i].reshape(H, W).numpy().astype(np.uint8) * 255
        m = cv2.resize(m, (mw, mh), interpolation=cv2.INTER_AREA)
        res.append({"width": mw, "height": mh, "data": encode_u8(m), "score": float(scores[i]) if i < len(scores) else None})
    return {"success": True, "masks": res, "metadata": {"processingTimeMs": round((time.time() - t0) * 1000), "model": SAM_MODEL}}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CV_PORT", "5179"))
    uvicorn.run(app, host=os.environ.get("CV_HOST", "127.0.0.1"), port=port, log_level="warning")
