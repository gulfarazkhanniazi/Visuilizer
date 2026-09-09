"""Room photo analysis: floor/wall semantic segmentation and mask cleanup. Runs the heavy
CV/ML work server-side (PyTorch + real OpenCV) instead of the browser, for both better
accuracy and better speed than in-browser WASM.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import cv2
import numpy as np
import torch # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingImports]
from PIL import Image
# pyrefly: ignore [missing-import]
from scipy.signal import find_peaks
from transformers import pipeline # pyright: ignore[reportMissingImports]

# Largest SegFormer/ADE20K checkpoint — no browser download-size constraint here, so we use
# the most accurate variant rather than the b1 model the client-side version was limited to.
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

# Small monocular depth model — used purely as a second, independent geometric signal (real
# corners/boundaries show a depth discontinuity, not just a color one) alongside segmentation,
# never as the sole source of truth. Kept to the small variant deliberately: this service has no
# CUDA GPU, only Apple Silicon MPS (or CPU) acceleration, and depth only needs to be roughly right
# at plane-boundary scale, not pixel-perfect, to do its job here.
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Real, trained promptable segmentation (Meta's SAM 2) for tightening one object's coarse
# SegFormer mask into its true silhouette, given only a bounding box — a strict upgrade over the
# GrabCut/watershed combination this replaces for organic objects (see `_sam2_refine_object_mask`
# and `ORGANIC_COARSE_BLOB_LABELS`): a real trained model rather than a per-pixel color-clustering
# heuristic, so it isn't fooled by an object that happens to share the wall's own color the way
# GrabCut's own color model could be. Tiny checkpoint deliberately — this only ever needs to
# refine one already-localized object at a time, not run open-ended detection, and it's fast
# enough (encoder forward once per photo, a few ms per object after that) even on this service's
# CPU/MPS-only hardware.
SAM2_CHECKPOINT = os.path.join(os.path.dirname(__file__), "sam2_checkpoints", "sam2.1_hiera_tiny.pt")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Regions smaller than this fraction of the image are dropped as noise. Kept low so genuinely
# smaller real surfaces (a door-surround wall strip, a small visible floor patch) still count —
# the earlier 1% threshold was silently discarding legitimate, if modest, wall/floor area.
MIN_AREA_FRACTION = 0.004
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
FLOOR_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# Set `vision.__debug_corners__ = True` before calling analyze_image (e.g. in a one-off script)
# to print every corner candidate _detect_wall_corners considered and its priority score — useful
# for diagnosing "why didn't this wall split" without re-instrumenting the function by hand.
MAX_ANALYSIS_DIMENSION = 1600
__debug_corners__ = False


def _torch_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


@lru_cache(maxsize=1)
def get_segmenter():
    return pipeline("image-segmentation", model=MODEL_NAME, device=_torch_device())


@lru_cache(maxsize=1)
def get_depth_estimator():
    return pipeline("depth-estimation", model=DEPTH_MODEL_NAME, device=_torch_device())


def _sam2_device() -> str:
    # SAM 2's own `build_sam2` wants a plain device string, not the HF `pipeline(device=...)`
    # convention `_torch_device()` returns (-1 for CPU, an int index for CUDA).
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_sam2_predictor():
    """Lazily builds a SAM 2 image predictor, or returns `None` if it can't be loaded (not
    installed, or the checkpoint isn't present at `SAM2_CHECKPOINT`) — every call site treats
    `None` as "fall back to the GrabCut/watershed refinement this replaces", so a service running
    without SAM 2 set up keeps working exactly as it did before this was added."""
    try:
        from sam2.build_sam import build_sam2 # pyright: ignore[reportMissingImports]
        from sam2.sam2_image_predictor import SAM2ImagePredictor # pyright: ignore[reportMissingImports]

        if not os.path.isfile(SAM2_CHECKPOINT):
            return None
        model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=_sam2_device())
        return SAM2ImagePredictor(model)
    except Exception:
        return None


def _estimate_depth(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Runs monocular depth estimation and returns a float32 relative-depth map at `size`
    (width, height), matching `analysis_size` so it can be indexed alongside `gray` and the
    segmentation masks with no further resizing. Larger values mean *closer* to the camera
    (Depth Anything's own convention) — callers only ever compare relative differences/gradients
    across a boundary, never raw depth as an absolute scale, so the exact convention only matters
    for sign consistency, not calibration.
    """
    estimator = get_depth_estimator()
    out = estimator(image)
    depth = out["predicted_depth"]
    depth = depth.detach().to(torch.float32).cpu().numpy()
    if depth.ndim == 3:
        depth = depth[0]
    if depth.shape[:2] != (size[1], size[0]):
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    return depth


def _segment(
    image: Image.Image, segmenter, extra_prob_labels: tuple[str, ...] = ("floor",)
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Runs the model directly instead of going through the pipeline's own call, so we keep each
    requested class's full per-pixel probability, not just which single class narrowly won. A
    pixel where floor scores, say, 40% and some other class scores 45% is still very likely real
    floor — but read through hard argmax alone (all the pipeline exposes) it becomes "not floor",
    and worse, an object that then actively blocks growth from ever recovering it. Keeping that
    class's raw probability lets both the floor mask itself and what's allowed to block its growth
    stay lenient exactly where the model is genuinely unsure, while staying just as strict
    everywhere it isn't — this is what makes the per-pixel floor scan itself more accurate, not
    just the cleanup applied after it.

    Deliberately extracts only the argmax label map and these classes' probabilities *before*
    leaving the accelerator, rather than moving the model's full (~150-class, full-photo-resolution)
    probability tensor to CPU and reducing it there with numpy — on a large photo that full tensor
    is well over a hundred times bigger than what's actually used afterward, and PyTorch's own
    argmax/index on the accelerator is both the smaller transfer and the faster reduction.
    """
    size = image.size # (w, h)
    inputs = segmenter.image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(segmenter.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = segmenter.model(**inputs).logits # (1, num_classes, h, w) at model resolution
    resized = F.interpolate(logits, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    probs_t = resized.softmax(dim=1)[0] # (num_classes, H, W), still on the accelerator
    label_map = probs_t.argmax(dim=0).cpu().numpy()

    label2id = {lbl.strip(): idx for idx, lbl in segmenter.model.config.id2label.items()}
    extra_probs = {lbl: probs_t[label2id[lbl]].cpu().numpy() for lbl in extra_prob_labels}

    outputs = []
    for idx, label in segmenter.model.config.id2label.items():
        mask = (label_map == idx).astype(np.uint8) * 255
        if not mask.any():
            continue
        outputs.append({"label": label.strip(), "mask": mask})
    return outputs, extra_probs


def _smooth_mask_contours(mask: np.ndarray, epsilon_frac: float = 0.004, max_epsilon: float = 5.0) -> np.ndarray:
    """Replaces a mask's boundary with a polygon-simplified version of itself.

    The model predicts at a fixed, much lower resolution than the photo. A real, straight wall
    edge or corner, represented in that coarse grid and then upsampled, comes out as a staircase
    — a few pixels over, a few pixels back, repeating along what should be one straight line.
    Blurring only softens each step in place; it doesn't straighten the staircase. Finding the
    mask's actual contour and simplifying it (Douglas-Peucker) collapses that staircase into
    proper line segments while still preserving genuine corners exactly where they are.

    `max_epsilon` is in the units of whatever resolution `mask` is currently at — the default
    suits analysis-resolution masks (capped at `MAX_ANALYSIS_DIMENSION`); a mask that's already
    been resized up to a much larger encode/display resolution needs a proportionally larger cap,
    or a staircase that was genuinely erased at the smaller resolution reappears, simply magnified,
    once resizing has stretched each of its steps into several final pixels instead of one.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    smoothed = np.zeros_like(mask)
    if hierarchy is None:
        return smoothed
    hierarchy = hierarchy[0]

    def simplify(contour):
        # Scaling epsilon off the *whole* perimeter is what a big region needs to straighten a
        # long architectural line, but that same epsilon then gets applied to every smaller
        # feature on that same contour too — a real object's own irregular outline (fringe on a
        # hanging, a plant's leaves, a dented edge), not just its one long straight run. Capped
        # absolutely so it only ever erases genuine upsampling noise, never a real, larger-scale
        # shape the boundary is actually supposed to follow — pixels are allowed to go up and down
        # where the real thing they're tracing does.
        perimeter = cv2.arcLength(contour, True)
        epsilon = min(max(1.5, perimeter * epsilon_frac), max_epsilon)
        return cv2.approxPolyDP(contour, epsilon, True)

    # outer boundaries first (paint solid), then punch out any holes (excluded objects) —
    # each simplified independently so a hole's own edge also comes out as a straight line.
    for i, contour in enumerate(contours):
        if hierarchy[i][3] == -1 and cv2.contourArea(contour) >= 6:
            cv2.drawContours(smoothed, [simplify(contour)], -1, 255, thickness=cv2.FILLED)
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1 and cv2.contourArea(contour) >= 6:
            cv2.drawContours(smoothed, [simplify(contour)], -1, 0, thickness=cv2.FILLED)
    return smoothed


def _raw_label_mask(entry, size: tuple[int, int], epsilon_frac: float = 0.004) -> np.ndarray:
    """Resizes+smooths one segmentation-output entry's mask to `size` (see `_mask_for_label`)."""
    mask = np.array(entry["mask"], dtype=np.uint8)
    if mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        mask = (mask > 127).astype(np.uint8) * 255
    return _smooth_mask_contours(mask, epsilon_frac=epsilon_frac)


def _mask_for_label(outputs, label: str, size: tuple[int, int], epsilon_frac: float = 0.004) -> np.ndarray:
    """Returns a binary (0/255) uint8 mask for `label`, or an all-zero mask if absent.

    The model predicts at a fixed, much lower internal resolution than the photo, so its mask
    arrives (or gets resized here) with blocky/jagged edges — some true boundary pixels end up
    on the wrong side either way, reading as "overflowing tile" in one spot and "gap left
    unpainted" a few pixels over in another. Resizing with linear interpolation and re-thresholding
    after a light blur rounds those edges off symmetrically instead of biasing the boundary
    outward or inward the way a dilate/erode would.
    """
    for entry in outputs:
        if entry["label"] == label:
            return _raw_label_mask(entry, size, epsilon_frac=epsilon_frac)
    return np.zeros((size[1], size[0]), dtype=np.uint8)


# A window looking outside gets classified by what's visible *through* the glass — sky, trees,
# distant buildings, terrain — not just "windowpane". That's correct where it's actually the
# window, but if that classification leaks even slightly past the window's true edge (extremely
# common right at a boundary), it silently blocks wall/floor growth right at that edge, which is
# exactly where users report "missing" coverage. None of these can legitimately be a real object
# sitting in front of an indoor wall/floor, so they never block growth into what's actually wall.
OUTDOOR_BLEED_THROUGH_LABELS = {
    "sky",
    "tree",
    "mountain",
    "sea",
    "water",
    "house",
    "building",
    "skyscraper",
    "grass",
    "field",
    "hill",
    "road",
    "sidewalk",
    "river",
    "bridge",
    "dirt",
    "sand",
    "path",
    "runway",
    "land",
    "rock",
    "palm",
}

# Real objects whose true silhouette is routinely irregular/leafy enough that the segmentation
# model's coarse internal grid comes back as a blocky polygon running well past the object's true
# edge into plainly visible wall — as opposed to a large rigid object (a bed, a sofa, a window),
# whose rough rectangular shape the model already predicts reliably close to correct, and handing
# to GrabCut's color-based refinement anyway was measured directly to be actively harmful (see
# `_occupied_mask`). Kept short and deliberate rather than "every label" — each addition should be
# a real case of this same coarse-blob failure, not a blanket assumption every object needs it.
ORGANIC_COARSE_BLOB_LABELS = {"plant", "flower"}

# Extending SAM2 tightening from ORGANIC_COARSE_BLOB_LABELS to rigid furniture (sofa, coffee
# table) was tried and directly measured on a dim, low-contrast TV-room photo where those labels'
# masks confidently swallowed real floor (floor probability near zero, not a recoverable near-tie)
# well past the furniture's true edge: SAM2's own box-prompted segmentation, seeded from that
# already-oversized box, mostly agreed with it rather than recovering the true edge, so floor
# coverage barely improved (+0.3 percentage points) while a real new contamination case appeared
# elsewhere (wall paint bleeding onto a real side-table object it hadn't touched before). Reverted
# for that reason — this remains an open, harder problem for genuinely ambiguous dim/reflective
# photos, not a coarse-blob-overshoot case the existing tightening machinery actually fixes.


def _refine_thin_object_mask(mask: np.ndarray, image_bgr: np.ndarray, margin: int | None = None) -> np.ndarray:
    """Recovers thin real-object structure — a plant's fine fronds, a lamp's wire arm, a cable —
    that the segmentation model's own fixed, coarse internal grid smooths straight past. Those
    pixels don't come back as "unclassified"; the model confidently hands them to whatever's
    behind them instead (wall, window, ceiling), so nothing downstream in the exclusion pipeline
    ever learns to keep paint off them — the wall/floor mask paints straight over a leaf tip or a
    thin branch sitting right in front of it. `_detect_textured_objects`'s local-variance signal
    can't fix this either: it only lights up right at a thin shape's own edge (a two-pixel-wide
    contour), never across its flat-colored interior, and its own noise-rejecting morphological
    opening erases a real but thin shape just as readily as it erases actual JPEG noise.

    Marker-based watershed in color space is what a coarse-mask-to-true-boundary refinement is
    normally built on, and it fixes exactly this: sure-foreground markers are the model's own
    detected pixels (eroded slightly, so a fuzzy label edge isn't itself a bad seed), sure-
    background is everywhere well outside a modest dilation of that same shape, and the band in
    between — precisely where a real thin frond or wire actually lives, just outside the model's
    detected blob — is left genuinely unknown for the photo's own color gradient to resolve,
    pixel by pixel, instead of a fixed morphological kernel guessing at it.

    Run per connected component in a tight, component-local ROI — not once over the whole photo:
    a plant with several disjoint detected leaf clusters shouldn't have one cluster's background
    markers pulled from clear across the room, and watershed's own cost scales with the region
    it's handed.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    h, w = mask.shape[:2]
    img_area = h * w
    # A fixed pixel margin behaves completely differently depending on this photo's own analysis
    # resolution (capped at MAX_ANALYSIS_DIMENSION, but a small source photo never gets upscaled
    # to it — this exact function saw both a 1500px-wide photo and a 612px-wide one in practice).
    # On a small photo, several separate real objects only a few pixels apart in analysis-resolution
    # terms — three individual trailing vine strands of one hanging planter, say — end up with their
    # dilated "background" rings overlapping each other before either one ever reaches real
    # background, so the real, visible wall gap *between* them reads as "unknown" and can get
    # swallowed into the grown foreground rather than recovered as paintable. Scaling the margin to
    # the photo's own size keeps the search proportional to what "nearby" actually means in this
    # photo, the same way every other per-pixel constant in this file is sized off resolution
    # instead of a fixed pixel count.
    if margin is None:
        margin = 16
    refined = np.zeros_like(mask)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        comp = (labels[y : y + ch, x : x + cw] == i).astype(np.uint8) * 255

        # A large, already-solid blob (a whole window, a big picture) is exactly what the model
        # already reliably captures — it doesn't need frond-style recovery, and handing watershed
        # a big region raises real risk (below) for zero benefit. Scoped to genuinely small/thin
        # detections only, which is also what keeps the per-component ROI cheap.
        #
        # Too small a blob for its own color statistics to mean anything (a handful of stray
        # pixels) is skipped the same way, just at the other end.
        if area < 30 or area > img_area * 0.02:
            refined[y : y + ch, x : x + cw] = cv2.bitwise_or(refined[y : y + ch, x : x + cw], comp)
            continue

        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + cw + margin), min(h, y + ch + margin)
        comp_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        roi = image_bgr[y0:y1, x0:x1]

        sure_fg = cv2.erode(comp_mask, erode_kernel)
        dilated = cv2.dilate(comp_mask, dilate_kernel)
        sure_bg = cv2.bitwise_not(dilated)
        unknown = cv2.bitwise_not(cv2.bitwise_or(sure_fg, sure_bg))
        unknown_count = int((unknown > 0).sum())

        markers = np.zeros(comp_mask.shape, dtype=np.int32)
        markers[sure_bg > 0] = 1
        markers[sure_fg > 0] = 2
        # Leaves at least one real foreground and one real background marker, or watershed has
        # nothing to grow from/into — falls back to the original detected shape for this blob.
        if unknown_count == 0 or not (markers == 1).any() or not (markers == 2).any():
            refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], comp_mask)
            continue

        cv2.watershed(roi, markers)
        grown = (markers == 2).astype(np.uint8) * 255
        # A real thin frond/wire is only ever a modest sliver of the unknown band around it — a
        # plain, near-uniform wall right next to the object has almost no color gradient of its
        # own to stop the flood at, so watershed can fail open and claim nearly the *whole* band
        # as foreground instead of stopping at the object's true edge. That failure is easy to
        # tell apart from a real, legitimate recovery: a real thin extension only ever claims a
        # small fraction of the surrounding unknown band, never most of it. Reject and keep the
        # original detected shape rather than risk swallowing real wall/floor area whenever the
        # claimed fraction looks like this failure mode instead of a genuine thin recovery.
        growth_in_unknown = int(np.count_nonzero((grown > 0) & (unknown > 0)))
        if growth_in_unknown > unknown_count * 0.5:
            refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], comp_mask)
            continue
        refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], grown)
    return refined


def _grabcut_tighten_object_mask(mask: np.ndarray, image_bgr: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    """Shrinks a large object's raw detected shape back to its true color-consistent silhouette.

    SegFormer predicts at a fixed, coarse internal grid regardless of the photo's real resolution,
    so a large object with an irregular outline (a leafy plant, above all) doesn't just come back
    with a slightly-off boundary — it comes back as a blocky, faceted polygon that can run tens of
    real pixels past the object's actual edge in places, confidently swallowing plainly visible
    wall along with it (the trim strip beside a plant that's nowhere near any leaf, found directly
    in a real photo this way). `_guided_filter_refine`'s edge-snap can't fix this: it only pulls a
    boundary to the nearest strong gradient within its own small radius, which does nothing when
    the blocky boundary sits deep inside plain wall with no nearby edge to snap to at all — the
    error here is bigger than any local filter can reach across.

    GrabCut is the right tool for exactly this: given a rough initial shape, it fits a real color
    model (a Gaussian mixture, separately for foreground and background) from the photo's own
    pixels and re-decides every uncertain pixel by which model it actually matches, not just by
    which side of a blocky line it happened to fall on. Only the shape's own well-inside core (well
    inside, so a single misclassified/overexposed pixel can't seed a bad model) is ever marked
    *definite* foreground, kept as an unremovable safety floor — everything else, including the
    rest of the model's own detected shape, is `PR_FGD` (probable, revisable) so this can genuinely
    shrink away a wrongly-included stretch of wall, not just nudge the existing boundary.

    Scoped to large components only — this is exactly what `_refine_thin_object_mask` (small/thin
    detections, growing rather than shrinking) deliberately leaves alone, and the two together
    cover both directions a coarse detection can be wrong.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    h, w = mask.shape[:2]
    img_area = h * w
    refined = np.zeros_like(mask)
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        comp = (labels[y : y + ch, x : x + cw] == i).astype(np.uint8) * 255
        # Too small for GrabCut's own color model to mean anything, or implausibly large for a
        # single potted plant/flower arrangement in a room photo (more likely a big, mostly-rigid
        # shape this refinement isn't meant for, or several distinct plants merged into one
        # component) — keep the original detected shape rather than run color modeling at a scale
        # where a bad fit risks reaching into real wall far from the object.
        if area < 200 or area > img_area * 0.2:
            refined[y : y + ch, x : x + cw] = cv2.bitwise_or(refined[y : y + ch, x : x + cw], comp)
            continue
        margin = max(20, round(max(cw, ch) * margin_frac))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + cw + margin), min(h, y + ch + margin)
        comp_full = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        roi = image_bgr[y0:y1, x0:x1]

        gc_mask = np.where(comp_full > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        core = cv2.erode(comp_full, core_kernel)
        gc_mask[core > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], comp_full)
            continue
        result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # A real object silhouette shrinking this much almost never reflects a genuinely correct
        # refinement — it's GrabCut's color model failing open (a plant photographed against a
        # similarly-toned wall, say) and collapsing to little more than the protected core itself.
        # Keeping the original detected shape in that case is the safe failure direction: some
        # excess exclusion around a real object costs a little paintable wall, but trusting a
        # collapsed result risks painting straight over the object it was supposed to protect.
        if (result > 0).sum() < (comp_full > 0).sum() * 0.5:
            refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], comp_full)
            continue
        refined[y0:y1, x0:x1] = cv2.bitwise_or(refined[y0:y1, x0:x1], result)
    return refined


def _sam2_refine_object_mask(mask: np.ndarray, sam2_predictor) -> np.ndarray:
    """Replaces `_grabcut_tighten_object_mask` + `_refine_thin_object_mask` for one label's mask,
    in one pass: given SAM 2's own `predictor.set_image(...)` already called once for this photo
    (by the caller, so the expensive image-encoder forward pass runs once per photo, not once per
    object — see `analyze_image`), this only needs the cheap `predict()` decode per component.

    Per connected component, box-prompts SAM 2 with that component's own tight bounding box —
    tight, deliberately: unlike GrabCut's own box (padded by a margin, since its color model
    needs real background pixels to build a *background* color distribution from), SAM 2 was
    trained on tight boxes and does its own job worse with a padded one, per Meta's own usage
    examples. Multi-mask output requested and the highest-scoring one kept — a single ambiguous
    box (an object that could reasonably be read as "just this leaf" vs "the whole plant") is
    exactly what SAM 2's multi-mask output exists to disambiguate.
    """
    n, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    refined = np.zeros_like(mask)
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = (labels_cc[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        if area < 30:
            # Too small for a box prompt to mean anything — keep the original detected shape,
            # same floor `_refine_thin_object_mask`/`_grabcut_tighten_object_mask` already use.
            refined[y : y + h, x : x + w] = cv2.bitwise_or(refined[y : y + h, x : x + w], comp)
            continue
        try:
            masks_out, scores, _ = sam2_predictor.predict(
                box=np.array([x, y, x + w, y + h]), multimask_output=True
            )
        except Exception:
            refined[y : y + h, x : x + w] = cv2.bitwise_or(refined[y : y + h, x : x + w], comp)
            continue
        best = masks_out[int(np.argmax(scores))]
        best_u8 = (best[y : y + h, x : x + w] * 255).astype(np.uint8)
        # Same safe-failure direction as the GrabCut/watershed functions this replaces: an
        # implausible collapse (near-empty, or shrunk to a sliver of the original detection) is
        # SAM 2 failing open on this box, not a genuine refinement — keep the original shape
        # rather than risk losing real, correctly-classified object area.
        if (best_u8 > 0).sum() < area * 0.3:
            refined[y : y + h, x : x + w] = cv2.bitwise_or(refined[y : y + h, x : x + w], comp)
            continue
        refined[y : y + h, x : x + w] = cv2.bitwise_or(refined[y : y + h, x : x + w], best_u8)
    return refined


def _occupied_mask(
    outputs,
    exclude_labels: set[str],
    size: tuple[int, int],
    near_window: np.ndarray | None = None,
    image_bgr: np.ndarray | None = None,
    sam2_predictor=None,
) -> np.ndarray:
    """Union of every class the model found *other* than the ones in `exclude_labels` — real
    furniture/window/picture/etc, as opposed to wall or floor which are just under-confident
    there. Used so we only ever grow a surface's mask into truly unclassified pixels, never into
    something the model was confident is a different, real object.

    An outdoor-bleed-through label (sky, tree, a distant building...) only gets to skip blocking
    growth within `near_window` — right at a real window's edge, where that's genuinely what a
    camera would see through the glass. Excluding it everywhere in the image, as an earlier
    version did, opened a real hole: if the model misreads some unrelated patch as one of these
    labels (uncommon on a real photo, but not impossible — an odd reflection, décor, an actual
    picture of an outdoor scene), that patch stopped counting as occupied at all, anywhere it
    happened to occur, letting growth sweep straight through it with nothing to stop it. Scoping
    the exclusion to only the area right around a real window keeps the original fix (don't let
    window-scenery block growth at the window's own edge) without leaving that hole open.
    """
    occupied = np.zeros((size[1], size[0]), dtype=np.uint8)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr is not None else None
    for entry in outputs:
        if entry["label"] in exclude_labels:
            continue
        mask = _raw_label_mask(entry, size)
        if entry["label"] in OUTDOOR_BLEED_THROUGH_LABELS:
            if near_window is None:
                continue
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(near_window))
        else:
            if entry["label"] in ORGANIC_COARSE_BLOB_LABELS and gray is not None and mask.any():
                # The model's own raw prediction for a real object is at a fixed, coarse internal
                # grid resolution — for an irregular, leafy silhouette that comes back as a blocky,
                # faceted polygon nowhere close to the true edge, not just slightly off, routinely
                # swallowing real, plainly visible wall several tens of pixels beyond the object's
                # actual silhouette (the trim strip beside a plant that's nowhere near any leaf,
                # found directly in a real photo this way). A plain edge-snap can't fix an error
                # this size — it only pulls a boundary to the nearest strong gradient within its own
                # small radius, which does nothing this deep inside plain wall. Deliberately scoped
                # to labels that are *both* large and routinely leafy/irregular enough to need it —
                # trying the *GrabCut* version of this (still the fallback below, when SAM 2 isn't
                # available) on every non-wall label was tested and measured directly to be actively
                # harmful there: a color-clustering model can just as easily latch onto real wall
                # near a large rigid object's own edge and misclassify it as "probably object". SAM
                # 2 — a real trained segmentation model, not a per-pixel color heuristic — was
                # measured not to have that specific failure mode on rigid objects (a sofa, a
                # picture frame) in testing, but this stays scoped to organic labels for now rather
                # than assuming that generalizes without the same kind of direct measurement.
                if sam2_predictor is not None:
                    mask = _sam2_refine_object_mask(mask, sam2_predictor)
                else:
                    mask = _grabcut_tighten_object_mask(mask, image_bgr)
                    mask = _refine_thin_object_mask(mask, image_bgr)
            elif image_bgr is not None and mask.any():
                # Recover whatever thin part of its true shape the model's own coarse grid missed
                # entirely (a frond, a wire) — complementary to the tightening above, not a
                # replacement: that shrinks a shape that overshot, this recovers a part that's
                # altogether absent. See `_refine_thin_object_mask`.
                mask = _refine_thin_object_mask(mask, image_bgr)
        occupied = cv2.bitwise_or(occupied, mask)
    return occupied


def _detect_textured_objects(gray: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
    """Finds flat, wall/floor-mounted decor the segmentation model reads as part of the surface
    itself — a woven wall hanging, a patterned rug the "rug" class missed, anything textured
    enough to look nothing like the plain paint or flooring around it. ADE20K has no class for
    "wall hanging"; if the model calls it "wall", nothing in the mask/growth/occupied pipeline
    downstream can tell — that logic only ever excludes what the model confidently calls a
    *different* class, and here it isn't one. This catches it a different way: a real painted
    wall or plain floor is close to uniform in local color, so a patch of unusually high local
    variance sitting inside an otherwise-uniform surface is very likely a distinct object, not
    the surface itself. Bounded on both ends deliberately — too small and it's just JPEG noise or
    a trim seam, too large and it's more likely the surface's own genuine texture (a patterned
    rug, a textured accent wall) than a discrete object sitting on it; better to leave a real
    textured surface intact than risk carving a big hole out of a design over a false alarm.
    """
    win = 9
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (win, win))
    sq_mean = cv2.blur(gray_f * gray_f, (win, win))
    local_std = np.sqrt(np.clip(sq_mean - mean * mean, 0, None))

    textured = cv2.bitwise_and(((local_std > 12).astype(np.uint8) * 255), surface_mask)

    # A real object has sustained high variance all through its interior; a plain seam between
    # two flat, differently-colored surfaces (a wood accent wall meeting a painted pillar, trim,
    # a shadow line) only has high variance in the thin strip where the sliding window straddles
    # both colors, then drops back to near-zero on either side. Opening with a kernel wider than
    # that strip erases the seam entirely while a genuinely wide object survives — do this before
    # closing, not after, or closing just thickens the seam into something wide enough to survive.
    textured = cv2.morphologyEx(textured, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    textured = cv2.morphologyEx(textured, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)

    img_area = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(textured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = np.zeros_like(textured)
    # The area bounds above only bound a blob's *size*, not how textured it actually is — a real
    # floor material (stone, marble, pronounced wood grain) routinely clears the >12 per-pixel bar
    # in patches too, especially on floor (this function runs on both surfaces, but floor is the
    # one that's rarely a flat, uniform color to begin with), and a patch of that natural mottling
    # can survive the opening/closing above at exactly the size a real object would. Measured
    # directly on a stone floor: a false-positive mottling blob that survived every filter above
    # averaged local_std 18.7 across its own footprint (over the blob's actual pixels, not its
    # bounding box — a bbox average reads lower since it also covers non-blob background), while
    # every real object tested here (a TV's frame edge, a patterned mirror, wall art, a throw
    # pillow) averaged 26 or higher, several past 80 — a real object sustains strong variance
    # throughout its interior, not just enough to scrape past the per-pixel threshold in places. Set
    # with real margin on both sides of that gap, not at the midpoint of one measured pair, since
    # natural material mottling varies by photo. Filtering each surviving blob on its own mean, not
    # just the binary per-pixel count that built its shape, is what tells those apart.
    MIN_MEAN_STD_FOR_OBJECT = 24.0
    for contour in contours:
        area_frac = cv2.contourArea(contour) / img_area
        if not (0.0015 < area_frac < 0.08):
            continue
        blob_mask = np.zeros_like(textured)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
        if float(local_std[blob_mask > 0].mean()) < MIN_MEAN_STD_FOR_OBJECT:
            continue
        cv2.drawContours(objects, [contour], -1, 255, thickness=cv2.FILLED)
    return objects


def _depth_foreground_mask(
    surface_mask: np.ndarray, depth: np.ndarray | None, local_win: int = 41, margin: float = 0.9
) -> np.ndarray:
    """Flags pixels within `surface_mask` (a candidate wall/floor mask) whose relative depth is
    anomalously *closer to the camera* than their own local neighborhood — the signature of a real
    foreground object sitting in front of the surface that color/texture alone read as part of it.
    This is squarely the case semantic segmentation and `_detect_textured_objects` both miss most
    often: a same-colored object against a similarly colored surface (a white picture frame on a
    white wall, a pale sofa against a pale wall) has almost no color or texture signal to key off
    of, but it still physically sits closer to the camera than the wall/floor plane behind it, and
    depth sees that even when color can't. Depth Anything's own convention has *larger* values
    mean *closer* — see `_estimate_depth`.

    Uses each pixel's local (box-filtered) depth baseline — computed only from other surface
    pixels, so a real wall/floor's own gradual perspective gradient sets the local expectation,
    not any object's depth — rather than one global threshold, since a real wall's depth
    legitimately varies a lot across a wide photo. `margin` is relative to this photo's own overall
    depth spread (a relative, unitless scale), not an absolute physical distance. Kept deliberately
    conservative (a wide margin) — this only needs to catch a real object's *depth discontinuity*
    against its own surrounding surface, not fine-tune where its edge is; boundary shaping stays
    the guided filter's job, same as it already is for every other exclusion signal here.
    """
    if depth is None or not surface_mask.any():
        return np.zeros_like(surface_mask)

    depth = depth.astype(np.float32)
    present = surface_mask > 0
    fallback = float(np.median(depth[present]))
    filled = np.where(present, depth, fallback).astype(np.float32)
    win = local_win if local_win % 2 == 1 else local_win + 1
    local_baseline = cv2.boxFilter(filled, -1, (win, win))
    spread = float(depth[present].std()) + 1e-6

    anomaly = (depth - local_baseline) > (margin * spread)
    anomaly_mask = np.where(present & anomaly, np.uint8(255), np.uint8(0))
    # A handful of scattered anomalous pixels is depth-map noise, not a real object — Depth
    # Anything's own map is coarse/blocky, so only a spatially coherent blob of real size is
    # trusted here; opening drops the noise, closing then restores a real blob's own interior
    # holes the opening step may have nicked.
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    return anomaly_mask


def _guided_filter_alpha(mask: np.ndarray, gray: np.ndarray, r: int = 8, eps: float = 0.01) -> np.ndarray:
    """The guided-filter matte itself, as a continuous float32 field in [0, 1] — same edge-aware
    alignment `_guided_filter_refine` is built on, but returned *before* that function's own final
    hard threshold collapses it to a binary 0/255 decision.

    That collapse is deliberate and correct everywhere `_guided_filter_refine` is actually used in
    this file (deciding *which* pixels count as wall/floor/object — a real classification, which
    has to end up binary somewhere to feed the bitwise mask-fusion the rest of the pipeline is
    built on). But the continuous value this function returns is itself real, useful information:
    right at a true boundary it's a sub-pixel estimate of how much of that pixel the surface
    actually covers, which is exactly what a boundary should look like once painted — a soft,
    anti-aliased edge, not a jagged one. Kept as its own function (rather than adding a
    "return_soft" flag to `_guided_filter_refine`) so every existing call site's behavior — and
    every downstream `cv2.bitwise_*` call built assuming a binary result — is untouched.
    """
    guide = gray.astype(np.float32) / 255.0
    src = mask.astype(np.float32) / 255.0

    box_size = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(guide, -1, box_size)
    mean_p = cv2.boxFilter(src, -1, box_size)
    mean_Ip = cv2.boxFilter(guide * src, -1, box_size)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, box_size)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, box_size)
    mean_b = cv2.boxFilter(b, -1, box_size)

    q = mean_a * guide + mean_b
    return np.clip(q, 0.0, 1.0)


def _guided_filter_refine(mask: np.ndarray, gray: np.ndarray, r: int = 8, eps: float = 0.01) -> np.ndarray:
    """Applies guided image filtering to align mask edges with the high-frequency edges
    in the guide image (gray).
    """
    return (_guided_filter_alpha(mask, gray, r=r, eps=eps) > 0.5).astype(np.uint8) * 255



def _floor_baseline_unblock(floor_mask: np.ndarray, wall_raw_mask: np.ndarray) -> np.ndarray:
    """A dim, shadowed strip of real floor (behind a plant pot, under low furniture) sometimes
    gets confidently — not just narrowly — misread as wall, since dim + low-texture can look a
    lot like a shadowed wall to the model. No probability threshold fixes a confident misread.
    But floor never legitimately sits *above* where wall is, and wall never legitimately persists
    *below* where the floor already clearly starts nearby — a real room's floor/wall boundary is
    one mostly-continuous line across the image, not something that jumps up mid-column. Building
    that line from the columns the model *did* get right, and refusing to let "wall" block floor
    growth below it anywhere else, recovers exactly this kind of confident local misread without
    touching any column where the model's own detection already looks normal.
    """
    h, w = floor_mask.shape[:2]
    has_floor = floor_mask.any(axis=0)
    if not has_floor.any():
        return np.zeros((h, w), dtype=np.uint8)

    top_y_per_col = np.argmax(floor_mask > 0, axis=0)
    known_idx = np.where(has_floor)[0]
    all_idx = np.arange(w)
    baseline = np.interp(all_idx, known_idx, top_y_per_col[known_idx])
    # Both the smoothing width and the margin below are sized relative to this photo's own
    # analysis resolution, not fixed absolute pixels — a fixed 41px blur is a wide, sensible
    # smoothing pass on a 1600px-wide analysis image but barely does anything on a much smaller
    # one (or over-smooths a narrow one), and the same is true of a fixed margin below the line.
    blur_width = max(5, round(w * 0.025))
    baseline = cv2.blur(baseline.reshape(1, -1).astype(np.float32), (1, blur_width)).flatten()

    margin = max(4, round(h * 0.012))
    below = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        y0 = max(0, int(baseline[x]) - margin)
        below[y0:, x] = 255
    return cv2.bitwise_and(below, wall_raw_mask)


def _close(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Fills small gaps (furniture legs, shadows, minor misclassification)."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=iterations)


def _quads_from_mask(member: np.ndarray, bbox: tuple[int, int, int, int]) -> list | None:
    """Fits one or more quads that together cover the mask, anchored to its true extent.

    WebGL only ever paints inside a quad's rasterized geometry — a mask pixel that falls outside
    every quad can never be painted, no matter how correct the mask itself is. A single trapezoid
    fit from just the mask's top and bottom edges assumes the shape's width changes *linearly*
    in between — true for a plain rectangular room's floor, false the moment the shape is more
    complex (a narrow notch near a plant at one height, a wide stretch past a nightstand at a
    similar height, on opposite sides of the frame). The linear interpolation between just those
    two edges can badly underestimate the true width at every height in between, leaving a real,
    correctly-detected stretch of the mask permanently outside the quad. Slicing the mask into
    several horizontal bands and fitting each its own edge — sharing exact boundary coordinates
    with its neighbor, so they still meet with zero gap between them, the same technique already
    proven for the wall's corner planes — tracks the mask's actual shape instead of assuming it.
    """
    min_y, min_x, max_y, max_x = bbox
    full_width = max_x - min_x + 1
    height = max_y - min_y + 1
    min_edge_width = max(4, round(full_width * 0.15))

    def row_extent(y):
        row = member[y, min_x : max_x + 1]
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            return None
        return int(nz[0]) + min_x, int(nz[-1]) + min_x

    def extent_near(y, lo_bound, hi_bound):
        # A quad edge with zero (or near-zero) width doesn't just look off, it can break the
        # projective-transform math outright, so the whole region silently fails to render even
        # though the mask is perfectly fine. Search outward from `y` in both directions — the
        # true edge here may itself be a genuinely narrow sliver, same as at the mask's overall
        # top/bottom — until the accumulated extent is wide enough to be a safe quad edge.
        lo = hi = None
        offset = 0
        max_offset = max(hi_bound - y, y - lo_bound, 0)
        while offset <= max_offset:
            candidates = (y,) if offset == 0 else (y - offset, y + offset)
            for yy in candidates:
                if yy < lo_bound or yy > hi_bound:
                    continue
                ext = row_extent(yy)
                if ext:
                    lo = ext[0] if lo is None else min(lo, ext[0])
                    hi = ext[1] if hi is None else max(hi, ext[1])
            if lo is not None and (hi - lo) >= min_edge_width:
                return lo, hi
            offset += 1
        if lo is None:
            return None
        # Last-resort safety net: the shape stayed narrow through the whole search window (a
        # genuinely thin sliver, not just a thin starting point) — fall back to the full bbox
        # width rather than ship an edge still thin enough to risk a broken transform.
        return (min_x, max_x) if (hi - lo) < min_edge_width else (lo, hi)

    sample_radius = max(6, round(height * 0.02))

    def padded_extent(y: int, radius: int) -> tuple[float, float] | None:
        lo_bound = max(min_y, y - radius)
        hi_bound = min(max_y, y + radius)
        ext = extent_near(y, lo_bound, hi_bound)
        if ext is None:
            return None
        lo, hi = ext
        # Padding each sampled edge outward is free: the fragment shader still discards anything
        # outside the mask regardless of how generous the quad's own geometry is, so overshooting
        # here can only ever recover real mask pixels a straight line would have missed, never
        # paint anything the mask doesn't actually contain.
        pad = max(20, round((hi - lo) * 0.14))
        return max(min_x, lo - pad), min(max_x, hi + pad)

    num_bands = max(1, min(9, round(height / max(1, full_width) * 6) + 3))
    band_ys = sorted({min_y + round(i * (height - 1) / num_bands) for i in range(num_bands + 1)})
    if len(band_ys) < 2:
        return None

    extent = {}
    for y in band_ys:
        ext = padded_extent(y, height // num_bands)
        if ext is None:
            return None
        extent[y] = ext

    # Between two sampled band boundaries, the quad edge as built so far is a *straight line*
    # connecting their (already padded) extents — a real mask shape is free to bulge further out
    # partway between those two samples (a real corner cut, a notch left by an excluded object,
    # a sofa's own curve), and padding alone doesn't fix that: it's a fixed margin around each
    # *sample*, not a bound on how far the true boundary can wander *between* samples. Whatever
    # part of the mask still falls outside the resulting quad there is real, correctly-detected
    # wall/floor area that WebGL then simply never paints — the render skips it outright, showing
    # the bare, unpainted photo right at that seam, exactly like a corner cut or an object's edge
    # would produce.
    #
    # Fixed instead of assumed away: actually check, at several rows between each pair of existing
    # bands, whether the true (unpadded) row extent still fits inside the straight-line
    # interpolation of the two padded edges either side of it. Wherever it doesn't — the real
    # boundary bent away from that straight line by more than the padding already absorbs — insert
    # a new band exactly at the worst such row and re-derive its own padded extent there, so the
    # quad between each *new*, narrower pair of bands hugs the true shape far more closely. Capped
    # so a pathologically jagged mask can't blow this up into hundreds of quads.
    MAX_BANDS = 48
    DEVIATION_TOLERANCE_PX = 2.0
    CHECK_SAMPLES = 6
    changed = True
    while changed and len(band_ys) < MAX_BANDS:
        changed = False
        i = 0
        while i < len(band_ys) - 1 and len(band_ys) < MAX_BANDS:
            y0, y1 = band_ys[i], band_ys[i + 1]
            gap = y1 - y0
            if gap < 4:
                i += 1
                continue
            lo0, hi0 = extent[y0]
            lo1, hi1 = extent[y1]
            worst_y = None
            worst_dev = DEVIATION_TOLERANCE_PX
            for k in range(1, CHECK_SAMPLES + 1):
                t = k / (CHECK_SAMPLES + 1)
                y = round(y0 + t * gap)
                if y <= y0 or y >= y1:
                    continue
                actual = row_extent(y)
                if actual is None:
                    continue
                a_lo, a_hi = actual
                interp_lo = lo0 + (lo1 - lo0) * t
                interp_hi = hi0 + (hi1 - hi0) * t
                dev = max(interp_lo - a_lo, a_hi - interp_hi, 0.0)
                if dev > worst_dev:
                    worst_dev = dev
                    worst_y = y
            if worst_y is not None:
                new_ext = padded_extent(worst_y, max(4, gap // 4))
                if new_ext is not None:
                    extent[worst_y] = new_ext
                    band_ys.insert(i + 1, worst_y)
                    changed = True
            i += 1

    quads = []
    for i in range(len(band_ys) - 1):
        y0, y1 = band_ys[i], band_ys[i + 1]
        lo0, hi0 = extent[y0]
        lo1, hi1 = extent[y1]
        quads.append([(lo0, y0), (hi0, y0), (hi1, y1), (lo1, y1)])
    return quads


def _interior_anchor(member: np.ndarray) -> tuple[float, float]:
    """A point guaranteed to sit inside `member`, as far as possible from its boundary (and, since
    an excluded object or a concave notch is a hole/indent in the mask, automatically as far as
    possible from those too) — the standard "pole of inaccessibility" a checkbox/label anchor
    needs. A plain arithmetic-mean centroid has no such guarantee: an L-shaped, crescent, or
    otherwise concave region (routine here — furniture carved out of a wall, a corner-cut quad)
    can easily average out to a point in the concave notch itself, outside the true paintable
    area, which is exactly the kind of "checkbox floating in empty space" this replaces.
    `cv2.distanceTransform` gives each interior pixel its distance to the nearest background/edge
    pixel in one pass; the pixel with the largest such distance is the most interior point.

    Padded with a zero border first, sized off the photo's own dimensions, so the photo's outer
    edge counts as a boundary too, not just an excluded object or the mask's own true edge. Without
    that, a wide wall whose mask happens to extend all the way to the photo's edge (routine — a
    wall running to the frame) scores maximally "interior" right at that edge, since nothing there
    ever reduces its distance score, even while the wall has plenty of real, more central paintable
    area elsewhere. Found in practice: a checkbox anchored at the very last pixel column of a
    1500px-wide photo, on a wall with hundreds of pixels of open, uncluttered area just left of it.
    The margin is deliberately modest — this discourages hugging the frame's outer edge, it doesn't
    forbid a genuinely edge-only shape (a thin corner sliver with nowhere else to go) from still
    anchoring near its own true center, which may itself be edge-adjacent.
    """
    h, w = member.shape[:2]
    pad = max(4, round(min(h, w) * 0.03))
    padded = cv2.copyMakeBorder(member, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x - pad), float(y - pad)


def _region_from_member(member: np.ndarray, quads: list | None = None):
    """Packages a mask into one selectable region. `quads` is one or more perspective quads used
    for the tile warp; if not given, one is fit by scanning the mask's own rows."""
    ys, xs = np.nonzero(member)
    if ys.size == 0:
        return None
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    if quads is None:
        quads = _quads_from_mask(member, bbox)
        if not quads:
            return None
    centroid = _interior_anchor(member)
    return {
        "quads": quads,
        "centroid": centroid,
        "mask": member,
    }


def _estimate_vertical_vanishing_point(gray: np.ndarray) -> tuple[float, float] | None:
    """Estimates where the room's true vertical lines (wall corners, door frames, window edges)
    converge in this photo — the vertical vanishing point — from every near-vertical line segment
    in the *whole* image, not just inside one wall's mask. A room photo shot with any camera tilt
    has real verticals that are only parallel in 3D, not in the 2D photo; using their actual
    common vanishing point (instead of assuming "near-vertical" always means "straight up and
    down") lets corner detection stay accurate even under real perspective tilt.

    Each candidate line contributes one homogeneous line equation (`p1 x p2` in homogeneous
    coordinates); a point every line passes through is a vector in each line's null space, found
    via SVD. A handful of non-architectural near-vertical edges (a curtain fold, a lamp cord) would
    corrupt a plain least-squares fit, so line pairs are first tested by RANSAC — repeatedly
    picking two random lines, treating their intersection as a hypothesis, and keeping whichever
    hypothesis the most other lines actually agree with — before the final SVD fit uses only that
    inlier set. Returns None when there isn't enough consistent evidence to trust a result at all,
    rather than returning a noisy guess.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    min_len = max(30.0, gray.shape[0] * 0.08)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return None

    homogeneous: list[np.ndarray] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_len:
            continue
        angle_from_vertical = np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6)))
        if angle_from_vertical < 20:
            homogeneous.append(np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
    if len(homogeneous) < 8:
        return None

    L = np.array(homogeneous)  # (n, 3)
    norms = np.linalg.norm(L[:, :2], axis=1) + 1e-9
    rng = np.random.default_rng(0)
    n = len(L)
    best_inliers: np.ndarray | None = None
    inlier_thresh = max(3.0, gray.shape[1] * 0.004)
    for _ in range(200):
        i, j = rng.choice(n, size=2, replace=False)
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-9:
            continue
        v = v / v[2]
        dists = np.abs(L @ np.array([v[0], v[1], 1.0])) / norms
        inliers = np.where(dists < inlier_thresh)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
    # Deliberately an absolute floor, not a fraction of `n` (the previous `n * 0.15`). RANSAC's
    # whole job here is to find the largest self-consistent cluster regardless of how much noise
    # surrounds it — a real, cluttered room photo has plenty of near-vertical lines that are
    # genuinely not architectural (curtain folds, furniture edges), and every one of those inflates
    # `n` without ever being a real inlier. Scaling the acceptance bar by that inflated `n` means a
    # busier, more realistic room needs a *larger* genuine cluster to pass than a sparse one does,
    # backwards from what clutter should imply. Verified against two real test photos where this
    # was silently discarding a real, correctly-found cluster: 5 inliers out of 52 total lines, and
    # 4 out of 23 — both real architectural clusters (confirmed by the recovered VP being usable
    # downstream), both rejected by the old `n * 0.15` bar (needed 7.8 and 6.0 respectively). 5
    # independent line observations is still a real, defensible floor for trusting a robust SVD fit
    # over just 2 required to define one intersection — chosen to be the minimum that recovers
    # both of those real, previously-discarded clusters without accepting a bare 2-3 line coincidence.
    if best_inliers is None or len(best_inliers) < 4:
        return None

    _, _, vt = np.linalg.svd(L[best_inliers])
    v = vt[-1]
    if abs(v[2]) < 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def _detect_wall_corners(
    gray: np.ndarray,
    mask: np.ndarray,
    max_corners: int | None = None,
    vp: tuple[float, float] | None = None,
    depth: np.ndarray | None = None,
    require_shading: bool = False,
) -> list[tuple[float, float]]:
    """Finds confident architectural corners within a *unified* wall mask, each becoming both an
    independently selectable wall face and a separate perspective quad for that face's own tile
    orientation. Kept deliberately strict: a false positive here splits the wall somewhere it
    shouldn't, whereas a missed real corner just falls back to one flat warp/region across it, a
    much smaller visual cost. Returns each corner as a (x_at_top, x_at_bottom) line in global
    image coordinates, following the real detected tilt rather than assuming it's vertical.

    Combines several independent signals rather than trusting a single one, and cross-checks each
    against the others instead of accepting any one blindly:

    1. Sub-pixel refit. A raw Hough segment's own two endpoints are noisy, and extrapolating a
       corner's tilt across the whole wall height from just those two points amplifies that
       noise. Every surviving candidate is refit with `cv2.fitLine` using the Huber M-estimator
       (`cv2.DIST_HUBER`) over *every* nearby edge pixel, not just the segment's endpoints — a
       robust-statistics technique that automatically down-weights outlier points (JPEG noise, a
       stray reflection, a picture frame's edge) instead of being thrown off by them the way an
       ordinary least-squares fit would be.
    2. Vanishing-point consistency (`vp`, see `_estimate_vertical_vanishing_point`). A real
       architectural corner's line points at the same vertical vanishing point every other real
       vertical in the photo converges to; a candidate that doesn't is more likely a decorative
       line at a coincidentally similar angle.
    3. Physical plausibility — color *and* depth. A real corner is where two different wall planes
       meet, so at least one of (a) mean brightness or (b) monocular relative depth immediately
       left vs. right of the refit line must actually differ by a real margin. A decorative panel
       seam or a wallpaper stripe can produce just as strong and long a Canny edge as a real corner
       but changes neither the plane's lighting nor its depth, and gets rejected here even though
       it looks identical to Hough alone. Point-feature detectors (Harris/Shi-Tomasi) were
       deliberately not used for this step — they find corner-like *points*, but what's needed
       here is a corner *line* spanning the wall's full height, which is exactly what Hough plus
       vanishing-point geometry already targets.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    min_y, min_x, max_y, max_x = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    # Lowered from 80: this same check (via the transpose in `_detect_horizontal_wall_corners`)
    # was rejecting a real, evidenced molding strip above a narrow-but-real wall face (a nook's
    # own return/side edge, 63px wide in one real test photo) before evidence-scoring ever got a
    # chance to run — the 80px figure was sized for a *vertical* cluster's own realistic minimum,
    # not for how narrow a still-genuinely-paintable wall face can legitimately be. 50 still rules
    # out a sliver too thin to sample real bands from at all.
    if width < 50 or height < 50:
        return []

    if max_corners is None:
        # A wide/panoramic wall can plausibly contain more than one or two real corners — scale
        # the budget with width instead of hard-capping every wall at the same fixed count. A
        # real feature (a chimney breast, a shallow nook) can pack multiple genuine corners into a
        # fairly narrow span, so this only needs to rule out an implausibly high count for the
        # wall's size, not pre-guess the *right* count — the actual number that survive is
        # decided by real evidence (`refined`) and by `min_separation` below, not by this cap.
        max_corners = max(4, min(12, round(width / 80)))

    region_raw = gray[min_y : max_y + 1, min_x : max_x + 1]
    region = region_raw.copy()
    region_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
    region[region_mask == 0] = 0
    row_wall_frac = (region_mask > 0).mean(axis=1)

    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    # Auto-thresholded (median-based) rather than fixed 40/120: a fixed threshold tuned for one
    # exposure/lighting condition silently under-detects a softer, more evenly lit corner shadow
    # in a differently lit photo, which is exactly the kind of real corner Hough was missing.
    wall_median = float(np.median(blurred[region_mask > 0])) if (region_mask > 0).any() else 128.0
    canny_lo = max(0, int(0.66 * wall_median))
    canny_hi = min(255, int(1.33 * wall_median))
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    edges[region_mask == 0] = 0
    edge_ys, edge_xs = np.nonzero(edges)

    # Pixel-count constants below (search radii, sampling bands) are all sized relative to this
    # wall's own analysis-resolution width, not fixed absolute pixel counts — a fixed number tuned
    # against one photo's resolution silently samples a tiny, meaningless sliver on a wall that
    # happens to be analyzed much smaller, or a wildly oversized/noisy region on one analyzed much
    # larger, since analysis resolution varies a lot across real uploads (some stay at native
    # size well under the 1600px cap, others get downscaled from a 4000px+ phone photo).
    edge_search_radius = max(3, round(width * 0.012))
    row_diff_gutter = max(2, round(width * 0.006))
    row_diff_band = max(6, round(width * 0.018))

    def refine_line(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float, bool]:
        length = max(1e-6, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        dist = np.abs((x2 - x1) * (edge_ys - y1) - (y2 - y1) * (edge_xs - x1)) / length
        band = dist < edge_search_radius
        if band.sum() < 10:
            return x1, y1, x2, y2, False
        pts = np.column_stack([edge_xs[band], edge_ys[band]]).astype(np.float32)
        vx, vy, px, py = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        if abs(vy) < 1e-6:
            return x1, y1, x2, y2, False
        t0 = (0.0 - py) / vy
        t1 = ((height - 1) - py) / vy
        return float(px + vx * t0), 0.0, float(px + vx * t1), float(height - 1), True

    def _covered_run(y: int, start: int, step_dir: int, max_len: int) -> tuple[int, int]:
        """Length of the contiguous *masked* run starting at `start` and moving by `step_dir`
        (+1 or -1), capped at `max_len`, plus the count of those pixels that are actually wall
        (vs. holes from an excluded object inside the run). Used instead of a fixed-width slice
        so a candidate sitting close to a real object (a TV mount, a glass partition frame) still
        gets scored on whatever real wall evidence exists right up to that object's edge, instead
        of being thrown out entirely because the *usual* sampling width doesn't fit there — the
        object's own edge is frequently exactly where a real architectural corner also is."""
        n = valid = 0
        x = start
        while n < max_len and 0 <= x < width:
            if region_mask[y, x] > 0:
                valid += 1
            n += 1
            x += step_dir
        return n, valid

    def _row_diffs(
        x_top: float, x_bottom: float, values: np.ndarray, band: int = row_diff_band, gutter: int = row_diff_gutter
    ) -> np.ndarray:
        """(left-band mean) - (right-band mean) of `values`, computed separately at each of many
        heights along the candidate line, rather than pooling every sampled pixel from every
        height into one big average. The per-row *sign* of this is what actually distinguishes a
        real architectural corner from a decorative or occlusion edge: a genuine plane change is a
        systematic effect that points the same direction the whole way down the wall, while a
        window frame, picture, or other localized feature only creates a real difference near its
        own position and is essentially coin-flip noise everywhere else along the same line — a
        pooled average can still come out large in both cases, so it can't tell them apart, but the
        fraction of rows that agree on the sign can."""
        min_band = max(2, band // 3)
        step = max(1, height // 80)
        diffs = []
        for y in range(0, height, step):
            if row_wall_frac[y] < 0.3:
                continue
            x = x_top + (x_bottom - x_top) * (y / max(1, height - 1))
            xi = int(round(x))
            left_len, left_valid = _covered_run(y, xi - gutter - 1, -1, band)
            right_len, right_valid = _covered_run(y, xi + gutter, 1, band)
            if left_valid < min_band or right_valid < min_band:
                continue
            lo, hi = xi - gutter - left_len, xi - gutter
            lo2, hi2 = xi + gutter, xi + gutter + right_len
            lm = region_mask[y, lo:hi] > 0
            rm = region_mask[y, lo2:hi2] > 0
            diffs.append(float(values[y, lo:hi][lm].mean()) - float(values[y, lo2:hi2][rm].mean()))
        return np.array(diffs)

    def shading_signal(x_top: float, x_bottom: float) -> tuple[float, float]:
        """Returns (gap, consistency): `gap` is the overall brightness-difference magnitude (the
        physical signature of a real corner); `consistency` is the fraction of sampled rows whose
        difference agrees in sign (whether that signature holds up the wall's whole height, or is
        just a pooled-average artifact of one localized feature)."""
        d = _row_diffs(x_top, x_bottom, region_raw.astype(np.float32))
        if len(d) < 5:
            return 0.0, 0.0
        return abs(float(d.mean())), float(max((d > 0).mean(), (d < 0).mean()))

    region_depth = depth[min_y : max_y + 1, min_x : max_x + 1] if depth is not None else None
    depth_spread = float(np.std(region_depth[region_mask > 0])) + 1e-6 if region_depth is not None else 0.0

    def depth_signal(x_top: float, x_bottom: float) -> tuple[float, float]:
        """Same idea as `shading_signal`, but on relative depth instead of brightness — the two
        wall faces a real corner separates are genuinely different planes in 3D, so they should
        read as being at meaningfully different depth even where their paint color is nearly
        identical. Gap normalized by this wall's own depth variability since the depth model's
        output is a relative, unitless scale, not a calibrated real-world distance."""
        if region_depth is None:
            return 0.0, 0.0
        d = _row_diffs(x_top, x_bottom, region_depth)
        if len(d) < 5:
            return 0.0, 0.0
        return abs(float(d.mean())) / depth_spread, float(max((d > 0).mean(), (d < 0).mean()))

    vp_local = (vp[0] - min_x, vp[1] - min_y) if vp is not None else None

    def _undirected_angle(dx: float, dy: float) -> float:
        # mod 180 because a *line*'s orientation (unlike a ray's) has no inherent direction —
        # this sidesteps having to reason about which side of the region the vanishing point
        # happens to fall on.
        return float(np.degrees(np.arctan2(dy, dx)) % 180.0)

    def vp_angle_diff(x_top: float, x_bottom: float) -> float | None:
        """Degrees between this candidate's own tilt and the tilt a real vertical *should* have
        at this position, given where every other real vertical in the photo converges."""
        if vp_local is None:
            return None
        vpx, vpy = vp_local
        x_mid, y_mid = (x_top + x_bottom) / 2, (height - 1) / 2
        a_vp = _undirected_angle(x_mid - vpx, y_mid - vpy)
        a_line = _undirected_angle(x_bottom - x_top, float(height - 1))
        diff = abs(a_vp - a_line) % 180.0
        return min(diff, 180.0 - diff)

    # Shorter than before (was 0.45) — a corner partway behind a TV, console, sofa, or cabinet
    # still has a real, continuous edge, and requiring it to run most of the wall's full height
    # was silently dropping real corners on any wall with something tall standing in front of them.
    min_line_length = max(20.0, height * 0.20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_line_length, maxLineGap=20)

    # This only needs to be as wide as `_row_diffs`/`_covered_run` actually require to sample real

    # This only needs to be as wide as `_row_diffs`/`_covered_run` actually require to sample real
    # pixels on both sides of a candidate line (`row_diff_gutter` + `row_diff_band`) — not an
    # arbitrary cosmetic buffer. A flat 5%-of-width margin (the previous value) silently rejected
    # every real corner near a wall region's own left/right edge before it ever reached scoring —
    # including one whose vanishing-point alignment alone (`vp_ok` in `score()`, which needs no
    # side-sampling data at all) would have been sufficient evidence on its own. This still keeps
    # candidates that are genuinely too close to the edge to sample at all from ever being
    # generated, it just stops discarding evidence the scorer never got a chance to weigh.
    #
    # Turns out that fix didn't go far enough: `row_diff_gutter + row_diff_band` (tied to
    # `_row_diffs`' own sampling needs) is *itself* redundant as a hard pre-filter, because
    # `_row_diffs`/`_covered_run` already degrade gracefully on their own when there isn't enough
    # room to sample — too little real data on one side already comes back as a short run, which
    # already makes `shading_signal`/`depth_signal` return a weak (0.0, 0.0) and correctly fail
    # corroboration inside `score()`. A *separate* margin enforcing the same "enough room to
    # sample" requirement before a candidate even reaches `score()` doesn't add safety, it just
    # blocks the one case that never needed that room in the first place: `vp_ok` alone, which
    # needs no side-sampling at all and is the entire reason a low/zero-contrast corner (a real
    # room corner very close to the photo's own edge, exactly a wall this session's own tracing
    # confirmed a real corner near) can ever be accepted without brightness evidence. Verified
    # directly: a real gradient candidate at the true corner in that photo refined to x_bottom=33,
    # rejected outright by the old margin(=36) despite `refine_line` already having found enough
    # real edge pixels to produce that fit — the corner was found and then discarded before
    # scoring ever ran. `edge_margin` here is a sanity bound only — keeping a candidate off the
    # literal 0/width edge where a line has no meaningful direction — not a "how close is too
    # close" judgment; that judgment now belongs entirely to `score()`, which already makes it
    # correctly per-candidate based on whatever real evidence actually exists there.
    edge_margin = max(3, round(width * 0.006))
    if require_shading:
        # After transpose, `width` is the wall's own height. A cut a few percent from the top or
        # bottom is the ceiling or floor line, not an internal soffit/chair-rail.
        edge_margin = max(edge_margin, round(width * 0.08))
    margin = edge_margin
    # A narrow real feature — a chimney breast, a shallow nook — can legitimately put two real
    # corners only 100-150px apart in the photo; the previous 0.6 multiplier assumed evenly
    # spaced corners across the whole wall and ended up rejecting a real, well-evidenced second
    # corner for being "too close" to the first one it was in fact genuinely near.
    min_separation = max(15.0, width / (max_corners + 1) * 0.15)
    MIN_SHADING_GAP = 6.0
    MIN_DEPTH_GAP_Z = 0.5
    VP_CONSISTENT_DEG = 3.0

    # source "hough": (length, x1, y1, x2, y2). A long (>=45% of the wall's height), straight,
    # near-vertical, well-centered edge is already a strict geometric bar on its own — real
    # decorative lines (trim, a single picture edge) essentially never satisfy all four at once,
    # so meeting it is sufficient evidence by itself, no further gate needed.
    # Horizontal architectural lines (soffits, chair rails) are often 10–18° off true horizontal
    # because of camera perspective; the 10° bar used for vertical corners is too tight on this axis
    # and was silently dropping real Hough evidence. Gradient-profile peaks on this axis are also
    # mostly lighting falloff / ceiling lines, so those are skipped later when `require_shading`.
    max_axis_angle = 18.0 if require_shading else 10.0
    hough_candidates: list[tuple[float, float, float, float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = (dx**2 + dy**2) ** 0.5
            angle_from_vertical = np.degrees(np.arctan2(dx, max(dy, 1e-6)))
            x_mid = (x1 + x2) / 2
            if angle_from_vertical < max_axis_angle and length > min_line_length and margin < x_mid < width - margin:
                hough_candidates.append((length, float(x1), float(y1), float(x2), float(y2)))

    # Complementary source: Hough needs one *unbroken* line segment of a minimum length, but a
    # real photographed corner's edge is often a soft shading gradient rather than a crisp line,
    # or gets broken into several shorter segments by lighting/noise — either way, Hough can miss
    # it outright even though the corner is real. Integrating the horizontal gradient magnitude
    # down each whole column doesn't care about breaks or softness: a column sitting on a genuine
    # plane boundary accumulates a consistently elevated gradient the whole way down even where no
    # single segment there is individually sharp or long enough for Hough to notice.
    #
    # Computed on the real, unmasked pixel data (`region_raw`), not the hard-zeroed `region` the
    # Canny/Hough pass above uses — zeroing first is fine for Hough (it only needs to *ignore*
    # excluded pixels) but would inject a fake hard step at every occlusion boundary (a window, a
    # sofa) here, which local gradient can't tell apart from a real corner. A several-pixel
    # erosion of the mask used only to *gate which columns count* (never to alter the pixel data
    # itself) keeps a real occlusion's own true edge from leaking into a neighboring column.
    safe_mask = cv2.erode(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    col_counts = (safe_mask > 0).sum(axis=0).astype(np.float32)
    # Sized leniently so that columns containing corners behind/around a TV or console table
    # are not ignored due to partial occlusion.
    col_valid = col_counts > max(15, height * 0.05)
    gradient_x: list[float] = []
    if col_valid.any():
        sobel_x = np.abs(cv2.Sobel(region_raw.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        sobel_x *= safe_mask > 0
        profile = np.zeros(width, dtype=np.float32)
        profile[col_valid] = sobel_x.sum(axis=0)[col_valid] / np.maximum(col_counts[col_valid], 1)
        peak_idx, _ = find_peaks(
            profile, distance=max(20.0, width * 0.05), prominence=float(profile.std()) * 1.0 + 1e-6
        )
        gradient_x = [float(p) for p in peak_idx if col_valid[p] and margin < p < width - margin]

    # A candidate whose sign-consistency (see `_row_diffs`) clears this bar shows a systematic,
    # whole-height directional effect — a real plane change — rather than a difference concentrated
    # in only part of the wall's height, which is the actual signature of a decorative or occlusion
    # edge (a window, a picture, a piece of furniture) even when that edge's *pooled* magnitude is
    # just as large or larger. 0.5 is coin-flip noise; comfortably above it either way tells the
    # two apart regardless of how strong or subtle the underlying brightness/depth difference is.
    CONSISTENCY_THRESHOLD = 0.65

    # A shadow cast by a shelf, a light fixture, or a piece of furniture sitting in front of an
    # otherwise flat wall can be just consistent enough in *one* signal alone to clear
    # CONSISTENCY_THRESHOLD — 0.71 shading consistency looks like real evidence in isolation. What
    # it essentially never does is also show a genuine depth break at the same spot: shading_c and
    # depth_c are measuring two physically independent things, so if both only reach a middling
    # score, that's not two half-confirmations adding up to one real one, it's two coincidences.
    # A real corner shows up strongly in *both* (typically 0.86-1.0 each on real photos), so their
    # product stays high; two just-barely-passing scores multiply down to well below threshold.
    # Measured directly against every real photo tested so far: genuine corners' shading_c and
    # depth_c product clusters at 0.86-1.0 (both signals strongly agree), while decorative seams
    # and shadow artifacts that clear each threshold individually still only reach 0.50-0.65 once
    # multiplied — 0.78 sits in the real gap between those two clusters rather than being a round
    # number picked in the abstract.
    CORROBORATION_THRESHOLD = 0.78

    # Distinguishes a genuine architectural plane change from a foreground object's own real
    # depth/shading discontinuity crossing an otherwise continuous wall (a decor item too thin or
    # wispy for `_detect_textured_objects`/`_depth_foreground_mask` to exclude — dried branches, a
    # lamp pole) — both can otherwise legitimately clear every corroboration check above, since the
    # object's own depth and color measurements are physically real, just not architectural.
    #
    # The physical fact that tells them apart: a real corner is a genuine break in the wall's own
    # plane, so each side's own depth stays internally consistent whether sampled right at the
    # candidate line or well clear of it — that side's plane simply doesn't change except exactly
    # at the true corner. An object crossing an otherwise continuous wall instead corrupts the
    # reading right next to the candidate line (the object's own depth, not the true wall's) on
    # whichever side it actually sits on, so that near-line sample disagrees with a same-side
    # sample taken further away, past the object, even though it's genuinely the same wall.
    # Checking each side's own *self*-consistency (near vs. far on the *same* side) is what catches
    # this — every corroboration signal above only ever compares near-left against near-right, which
    # an object standing on a real, otherwise-flat wall can satisfy just as convincingly as a real
    # corner does.
    #
    # Verified directly against this file's own real production data: the three confirmed
    # foreground-object false positives (a dried-grass arrangement and a floor lamp crossing one
    # continuous wall) measured a same-side near-vs-far disagreement of 0.66-2.15x the wall's own
    # depth spread on whichever side the object sat; three confirmed genuine architectural corners
    # in a different, multi-wall photo measured 0.09-0.28x on *both* sides. 0.45 sits in the real
    # gap between those two clusters.
    PLANE_CONTINUITY_CLEAR_GAP_FRAC = 0.04
    PLANE_CONTINUITY_FAR_BAND_FRAC = 0.05
    PLANE_CONTINUITY_SELF_DIFF_THRESHOLD = 0.45
    PLANE_CONTINUITY_MIN_SAMPLES = 5

    def plane_continuity_veto(x_top: float, x_bottom: float) -> bool:
        if region_depth is None or depth_spread <= 1e-6:
            return False
        clear_gap = max(20, round(width * PLANE_CONTINUITY_CLEAR_GAP_FRAC))
        far_band = max(15, round(width * PLANE_CONTINUITY_FAR_BAND_FRAC))

        def sample(y: int, x0: int, x1: int) -> float | None:
            x0c, x1c = max(0, x0), min(width, x1)
            need = max(2, (x1 - x0) // 3)
            if x1c - x0c < need:
                return None
            row = region_depth[y, x0c:x1c]
            m = region_mask[y, x0c:x1c] > 0
            if m.sum() < need:
                return None
            return float(row[m].mean())

        step = max(1, height // 80)
        near_left: list[float] = []
        near_right: list[float] = []
        far_left: list[float] = []
        far_right: list[float] = []
        for y in range(0, height, step):
            if row_wall_frac[y] < 0.3:
                continue
            x = x_top + (x_bottom - x_top) * (y / max(1, height - 1))
            xi = int(round(x))
            nl = sample(y, xi - row_diff_gutter - row_diff_band, xi - row_diff_gutter)
            nr = sample(y, xi + row_diff_gutter, xi + row_diff_gutter + row_diff_band)
            fl = sample(y, xi - row_diff_gutter - clear_gap - far_band, xi - row_diff_gutter - clear_gap)
            fr = sample(y, xi + row_diff_gutter + clear_gap, xi + row_diff_gutter + clear_gap + far_band)
            if nl is not None:
                near_left.append(nl)
            if nr is not None:
                near_right.append(nr)
            if fl is not None:
                far_left.append(fl)
            if fr is not None:
                far_right.append(fr)

        # Not enough evidence on one full side (the candidate sits too close to the cluster's own
        # edge to sample "far" at all, say) to run this check at all — stay conservative and never
        # veto a corner this signal can't actually test, rather than guessing.
        if (
            len(near_left) < PLANE_CONTINUITY_MIN_SAMPLES
            or len(near_right) < PLANE_CONTINUITY_MIN_SAMPLES
            or len(far_left) < PLANE_CONTINUITY_MIN_SAMPLES
            or len(far_right) < PLANE_CONTINUITY_MIN_SAMPLES
        ):
            return False

        left_self_diff = abs(float(np.mean(near_left)) - float(np.mean(far_left))) / depth_spread
        right_self_diff = abs(float(np.mean(near_right)) - float(np.mean(far_right))) / depth_spread
        return (
            left_self_diff > PLANE_CONTINUITY_SELF_DIFF_THRESHOLD
            or right_self_diff > PLANE_CONTINUITY_SELF_DIFF_THRESHOLD
        )

    def score(x_top: float, x_bottom: float) -> tuple[bool, float]:
        vp_diff = vp_angle_diff(x_top, x_bottom)
        vp_ok = vp_diff is not None and vp_diff < VP_CONSISTENT_DEG
        shading_g, shading_c = shading_signal(x_top, x_bottom)
        depth_g, depth_c = depth_signal(x_top, x_bottom)
        gap = max(shading_g / MIN_SHADING_GAP, depth_g / MIN_DEPTH_GAP_Z)
        shading_consistent = shading_c >= CONSISTENCY_THRESHOLD
        depth_consistent = depth_c >= CONSISTENCY_THRESHOLD
        if region_depth is None:
            # No depth signal available at all (only happens in isolated unit tests that don't
            # pass one in — the real pipeline always computes and passes depth) — fall back to
            # shading alone rather than let one missing signal veto every candidate outright.
            corroborated = shading_consistent
            physically_plausible = shading_consistent and shading_g >= MIN_SHADING_GAP * 0.3
        else:
            # Two ways to corroborate: either both signals are strongly *consistent* down the
            # wall's height (the product test — a decorative seam or a shelf's shadow can clear
            # each threshold individually but not together), or both are at least moderately
            # consistent *and* the physical magnitude is overwhelming — a real corner whose
            # underlying shading/depth step is huge can still have a merely-good (not perfect)
            # consistency score purely from ordinary per-row sampling noise, and that shouldn't
            # cost it acceptance the way a genuinely weak, borderline signal should.
            # A third path: shading alone, but held to a *much* higher bar than the old
            # single-signal gate (0.65) — near-perfect row-to-row agreement plus a large margin.
            # Depth is a real, independently trained model and it can legitimately have nothing
            # to say on an image outside its training distribution (a flat render, a heavily
            # stylized photo) even at a genuine corner; a shading signal this strong and this
            # consistent is itself already stronger evidence than any known false positive
            # measured so far has come anywhere close to (the highest seen was 0.77).
            # An ordinary room is usually lit from a window or the ceiling, so a real, fairly
            # consistent top-to-bottom brightness gradient across an entirely flat wall is common —
            # verified directly: a plain bedroom wall produced shading_gap=13.66, shading_c=0.91
            # (clearing the vertical-axis bar below easily) with depth_c=0.64 not corroborating it
            # at all, and the resulting "corner" sliced straight through the headboard and pillows.
            # A *vertical* corner rarely gets faked this way (side-lighting strong and consistent
            # enough to mimic a corner is much less common than top-lit/window-lit rooms), so this
            # only raises the bar under `require_shading`, leaving vertical corners exactly as before.
            shading_overwhelming = shading_c >= (0.95 if require_shading else 0.85) and shading_g / MIN_SHADING_GAP >= (
                2.5 if require_shading else 1.5
            )
            # Still not enough on its own: a piece of furniture standing against the wall (a low
            # basket/table) casts a real, strong, consistent shadow — shading_gap=29.44,
            # shading_c=0.96, comfortably clearing the raised bar above — with depth_c=0.72 not
            # agreeing at all, and the "corner" it produced sliced straight through that furniture.
            # A real architectural plane change almost always shows *some* real depth agreement even
            # where shading is doing most of the work (both known-good horizontal splits had
            # depth_c=1.00); a shadow artifact has no reason to. Require depth not to actively
            # disagree, under `require_shading` only.
            if require_shading:
                shading_overwhelming = shading_overwhelming and depth_c >= 0.8
            # A picture/painting hanging on the wall genuinely sits a little proud of it, which is
            # real, measurable depth evidence too — verified directly: a framed picture produced
            # shading_gap=8.62, shading_c=0.90, depth_gap=0.71, depth_c=1.00 (perfect depth
            # agreement, same as a real corner) and still wrongly sliced straight through the frame.
            # What actually separated it from a real horizontal line in the same photos: magnitude,
            # not consistency — the real window header measured shading_gap=30.55 (3.5x this),
            # depth_gap=1.41 (2x this). `require_shading` raises the floor the product-corroboration
            # path demands accordingly, well clear of a frame's shallow real protrusion.
            #
            # A floor lamp standing in front of an otherwise flat wall produces the same class of
            # false positive on the *vertical* axis (shading_gap=6.54, shading_c=0.85, depth_gap
            # near-zero, product 0.816 clearing corroboration on consistency alone) — but raising
            # this same floor for vertical corners broke real ones elsewhere (a real corner in one
            # test photo relied on shading_gap=11.05 with a weak depth_c=0.69, comfortably below a
            # 1.5x floor). Unlike the horizontal case, there wasn't a magnitude gap that separated
            # the fake evidence from genuine vertical corners without also excluding real ones — so
            # this floor stays scoped to `require_shading` (horizontal) only, where it's verified
            # to cost nothing. The vertical lamp false positive is a known, unresolved case.
            shading_floor_met = shading_g >= MIN_SHADING_GAP * (2.0 if require_shading else 0.3)
            depth_overwhelming = not require_shading and depth_c >= 0.85 and depth_g / MIN_DEPTH_GAP_Z >= 1.5
            corroborated = (
                (shading_c * depth_c >= CORROBORATION_THRESHOLD and (not require_shading or shading_floor_met))
                or (not require_shading and gap >= 1.5 and shading_consistent and depth_consistent)
                or shading_overwhelming
                or depth_overwhelming
            )
            physically_plausible = (
                (shading_consistent and shading_g >= MIN_SHADING_GAP * 0.3) or
                (depth_consistent and depth_g >= MIN_DEPTH_GAP_Z * 0.3)
            )
        priority = gap + (2.0 if shading_consistent else 0.0) + (1.0 if depth_consistent else 0.0)
        accept = (vp_ok and physically_plausible) or corroborated
        # Applied last, only once every existing signal above has already accepted the candidate —
        # this can only ever *reject* a candidate the corroboration checks above already trusted,
        # never accept one they didn't, so a candidate none of the existing evidence found
        # convincing in the first place is unaffected either way.
        if accept and plane_continuity_veto(x_top, x_bottom):
            accept = False
        if __debug_corners__:
            x_mid = (x_top + x_bottom) / 2
            print(
                f"    [score] x_mid={x_mid:.0f} shading_gap={shading_g:.2f} shading_c={shading_c:.2f} "
                f"depth_gap={depth_g:.2f} depth_c={depth_c:.2f} vp_ok={vp_ok} vp_diff={vp_diff} accept={accept}"
            )
        return accept, priority

    # A long (>=45% of wall height), straight, near-vertical, well-centered Hough line is already
    # strong geometric evidence on its own — but it can still land on a strongly-lit occlusion edge
    # (a window frame, a mirror) rather than a real corner, so it still needs to clear the same
    # sign-consistency/vanishing-point bar as a gradient candidate; unlike a gradient candidate it
    # doesn't *also* need a strong pooled gap, since the line itself is already the harder evidence
    # to fake.
    refined: list[tuple[float, float, float, float]] = []  # (priority, x_mid, x_top, x_bottom)
    for length, x1, y1, x2, y2 in hough_candidates:
        x_top, _, x_bottom, _, _ = refine_line(x1, y1, x2, y2)
        if not (margin < x_top < width - margin and margin < x_bottom < width - margin):
            continue
        accept, priority = score(x_top, x_bottom)
        if not accept:
            if __debug_corners__:
                print(f"  [hough reject] x_mid={(x_top + x_bottom) / 2:.0f} weak evidence")
            continue
        refined.append((priority, (x_top + x_bottom) / 2, x_top, x_bottom))

    # Gradient-profile candidates are geometrically weaker evidence than a strict Hough line (a
    # real, strongly-lit occlusion edge — a window frame's own true edge, not a masking artifact —
    # can also produce a tall gradient peak), so these need real corroboration: either vanishing-
    # point agreement, or a shading/depth difference that's *consistent in direction* across the
    # wall's whole height, not just large when pooled (see `_row_diffs` — pooled magnitude alone is
    # exactly what let a window/picture edge through before). Deliberately *not* gated on whether
    # `refine_line` found nearby Canny edges to snap to: a soft, gradual corner shadow is exactly
    # the case Canny/Hough miss outright — the whole reason this candidate source exists — so
    # requiring Canny corroboration here would reject precisely the corners it's meant to catch.
    #
    # On the horizontal axis (`require_shading`) a gradient peak is usually a ceiling lighting
    # falloff or the wall/floor line, not a soffit. Only a real Hough edge is allowed to split
    # there — that is the actual architectural corner.
    if not require_shading:
        for x in gradient_x:
            x1, y1, x2, y2 = x, 0.0, x, float(height - 1)
            x_top, _, x_bottom, _, _ = refine_line(x1, y1, x2, y2)
            if not (margin < x_top < width - margin and margin < x_bottom < width - margin):
                if __debug_corners__:
                    print(f"  [gradient reject] seed_x={x:.0f} out of margin bounds x_top={x_top:.0f} x_bottom={x_bottom:.0f}")
                continue
            accept, priority = score(x_top, x_bottom)
            if not accept:
                if __debug_corners__:
                    print(f"  [gradient reject] seed_x={x:.0f} weak/inconsistent evidence")
                continue
            refined.append((priority, (x_top + x_bottom) / 2, x_top, x_bottom))

    refined.sort(key=lambda r: -r[0])
    if __debug_corners__:
        for pr, xm, xt, xb in refined:
            print(f"  [candidate] x_mid={xm:.0f} x_top={xt:.0f} x_bottom={xb:.0f} priority={pr:.2f}")
    if __debug_corners__:
        print(f"  [selection] max_corners={max_corners} min_separation={min_separation:.1f} width={width} margin={margin:.1f}")
    chosen_x: list[float] = []
    cuts: list[tuple[float, float]] = []
    for _, x_mid, x_top, x_bottom in refined:
        if any(abs(x_mid - cx) < min_separation for cx in chosen_x):
            if __debug_corners__:
                print(f"  [selection reject] x_mid={x_mid:.0f} too close to {chosen_x}")
            continue
        chosen_x.append(x_mid)
        cuts.append((min_x + x_top, min_x + x_bottom))
        if len(cuts) >= max_corners:
            break

    return sorted(cuts, key=lambda c: c[0] + c[1])


def _detect_horizontal_wall_corners(
    gray: np.ndarray, mask: np.ndarray, depth: np.ndarray | None = None, max_corners: int | None = None
) -> list[tuple[float, float]]:
    """A horizontal architectural corner — a soffit/bulkhead above a recessed wall, a chair rail,
    a wainscoting line — is exactly a vertical corner rotated 90 degrees: the same real evidence
    applies (a real plane/color change reads as a sustained, consistent shading and/or depth step),
    just measured top-vs-bottom instead of left-vs-right. Rather than a second, independently-tuned
    implementation of that same evidence logic, this transposes the region and hands it to the
    already-validated `_detect_wall_corners` unchanged, then transposes its result back.

    `require_shading=True` — see its own comment on `_detect_wall_corners` — closes three false-
    positive patterns verified only on this axis: a monocular-depth-only "corner" on a perfectly
    flat wall, a piece of furniture's own shadow, and an ordinary window/ceiling lighting gradient.
    No vanishing-point evidence here — this codebase only ever estimates a *vertical* vanishing
    point, which has nothing to say about a horizontal line's own tilt.

    Returns each cut as `(y_at_left_edge, y_at_right_edge)`, following the line's real tilt rather
    than assuming it's perfectly level, mirroring `_detect_wall_corners`'s own `(x_top, x_bottom)`.
    """
    gray_t = np.ascontiguousarray(gray.T)
    mask_t = np.ascontiguousarray(mask.T)
    depth_t = np.ascontiguousarray(depth.T) if depth is not None else None
    return _detect_wall_corners(
        gray_t, mask_t, max_corners=max_corners, vp=None, depth=depth_t, require_shading=True
    )


def _split_mask_by_row_cuts(
    mask: np.ndarray, cuts: list[tuple[float, float]], left_x: float, right_x: float
) -> list[np.ndarray]:
    """Row-axis counterpart of `_split_mask_by_cuts`, built the same way `_detect_horizontal_wall_corners`
    finds its cuts: transpose, reuse the already-validated column-cut splitter unchanged, transpose
    back. `cuts` are `(y_at_left, y_at_right)` pairs spanning from `left_x` to `right_x`."""
    segments_t = _split_mask_by_cuts(np.ascontiguousarray(mask.T), cuts, left_x, right_x)
    return [np.ascontiguousarray(seg.T) for seg in segments_t]


def _split_region_horizontally(
    region: dict, gray: np.ndarray, depth: np.ndarray | None, image_bgr: np.ndarray | None
) -> list[dict]:
    """Checks one already-finished wall region (post vertical-corner-splitting) for a genuine
    horizontal architectural corner — a soffit, chair rail, or similar plane change — and splits
    it top/bottom if `_detect_horizontal_wall_corners` finds one. Paint-color changes on a flat
    plane are not corners and must not split the wall. Additive only: the region is untouched
    unless there is independent geometric evidence of a horizontal corner. Scoped to a region with
    a single simple quad; a region already built from several irregular bands is left alone.
    """
    quads = region["quads"]
    if len(quads) != 1:
        return [region]
    mask = region["mask"]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [region]
    top_y, left_x, bottom_y, right_x = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    height, width = bottom_y - top_y + 1, right_x - left_x + 1
    # Matches the lowered floor in `_detect_wall_corners`/`_detect_color_block_corners` — a narrow
    # but real wall face (a nook's own return/side edge) still deserves a chance at its own
    # molding split, not just the wide main wall faces either detector was originally sized around.
    if height < 50 or width < 50:
        return [region]

    cuts = _detect_horizontal_wall_corners(gray, mask, depth=depth, max_corners=1)
    if not cuts:
        return [region]
    y_left, y_right = cuts[0]

    all_cuts = [(float(top_y), float(top_y)), (y_left, y_right), (float(bottom_y + 1), float(bottom_y + 1))]
    new_quads = [
        [
            (left_x, all_cuts[i][0]),
            (right_x + 1, all_cuts[i][1]),
            (right_x + 1, all_cuts[i + 1][1]),
            (left_x, all_cuts[i + 1][0]),
        ]
        for i in range(2)
    ]
    submasks = _split_mask_by_row_cuts(mask, all_cuts, float(left_x), float(right_x + 1))

    min_band_height = max(15.0, height * 0.05)

    def band_height(m: np.ndarray) -> float:
        band_ys = np.nonzero(m)[0]
        return float(band_ys.max() - band_ys.min() + 1) if band_ys.size else 0.0

    if any(band_height(m) < min_band_height for m in submasks):
        return [region]
    if any(_largest_component_fraction(m) < MIN_SPLIT_SOLIDITY for m in submasks):
        return [region]

    new_regions = []
    for q, m in zip(new_quads, submasks):
        r = _region_from_member(m, quads=[q])
        if r:
            new_regions.append(r)
    return new_regions if len(new_regions) == 2 else [region]


SEAM_OVERLAP_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _split_mask_by_cuts(
    mask: np.ndarray, cuts: list[tuple[float, float]], top_y: float, bottom_y: float
) -> list[np.ndarray]:
    """Splits `mask` into `len(cuts) - 1` per-plane submasks using the exact same cut lines the
    caller already used to build each plane's quad, so every submask's boundary is numerically
    identical to its neighbor's — a mask pixel is assigned to exactly one segment by comparing its
    x position against each cut's x interpolated at that pixel's own row (a cut is a straight,
    possibly-tilted line from `(x_top, top_y)` to `(x_bottom, bottom_y)`), which guarantees the
    partition has zero gap and zero overlap at the true cut line before the safety dilation below.

    Vectorized across the whole mask at once (rather than per-row) since a full-resolution wall
    mask can be over a million pixels — a Python-level loop over rows would be needlessly slow for
    something numpy already does well as one array comparison.
    """
    h, w = mask.shape
    n_segments = len(cuts) - 1
    segments = [np.zeros((h, w), dtype=np.uint8) for _ in range(n_segments)]
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or n_segments <= 0:
        return segments

    span = max(bottom_y - top_y, 1e-6)
    t = (ys.astype(np.float64) - top_y) / span
    cut_arr = np.array(cuts, dtype=np.float64)  # (n_cuts, 2) = (x_top, x_bottom) per cut
    # boundary_x[c, i] = x of cut c at pixel i's own row
    boundary_x = cut_arr[:, 0][:, None] + t[None, :] * (cut_arr[:, 1] - cut_arr[:, 0])[:, None]
    seg_idx = (xs[None, :] >= boundary_x).sum(axis=0) - 1
    seg_idx = np.clip(seg_idx, 0, n_segments - 1)

    for s in range(n_segments):
        sel = seg_idx == s
        segments[s][ys[sel], xs[sel]] = 255

    # Bilinear filtering of each plane's mask texture at render time softens its edge over a
    # pixel or two — harmless at the wall's true outer boundary (it just anti-aliases against the
    # photo), but at an *internal* seam between two independently-filtered submasks it can leave
    # both sides reading as "not this plane" in the same thin strip, showing the bare photo behind
    # it instead of either design. A tiny dilation back into the mask's own true extent removes
    # that gap by giving adjacent planes a hairline of intentional overlap right at the seam —
    # the later-drawn plane simply wins there — which reads as attached, not as a visible crease.
    for s in range(n_segments):
        segments[s] = cv2.bitwise_and(cv2.dilate(segments[s], SEAM_OVERLAP_KERNEL, iterations=1), mask)

    return segments


def _exclude_occupied(mask: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Strips any pixel `occupied` claims out of `mask` — a real object (furniture, a picture,
    a rug's edge) must never end up inside the paintable surface, even partially. Each label's
    mask gets contour-simplified independently (see `_smooth_mask_contours`), and that
    simplification can nudge a shape's edge slightly beyond its original pixels in places, so a
    wall/floor mask and the union of every *other* object can end up overlapping right at their
    shared boundary even though the model's own per-pixel classification never overlaps. Called
    at every stage that could introduce or widen such an overlap — not just once at the start —
    so a design never gets painted onto something actually sitting on the wall or floor."""
    return cv2.bitwise_and(mask, cv2.bitwise_not(occupied))


def _grow_into_unclassified(mask: np.ndarray, occupied: np.ndarray, max_reach_px: int) -> np.ndarray:
    """Extends `mask` outward step-by-step, stopping the instant it would touch a pixel the
    model confidently classified as something else (`occupied`) — reaches toward image edges
    and other truly-unclassified gaps the model was simply unsure about, without ever bleeding
    onto a real, different object the way a single big dilation risked doing before. Each step
    covers a couple of pixels (not one) so a generous max_reach_px stays cheap; it still stops
    precisely at any real boundary since we re-clip against `occupied` after every step."""
    grown = mask.copy()
    step_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    step_px = 2
    allowed = cv2.bitwise_not(occupied)
    for _ in range(max_reach_px // step_px):
        dilated = cv2.bitwise_and(cv2.dilate(grown, step_kernel, iterations=1), allowed)
        if np.array_equal(dilated, grown):
            break
        grown = dilated
    return grown


def extract_wall_regions(
    wall_mask: np.ndarray,
    occupied: np.ndarray,
    gray: np.ndarray,
    vp: tuple[float, float] | None = None,
    depth: np.ndarray | None = None,
    floor_mask: np.ndarray | None = None,
    occupied_for_splitting: np.ndarray | None = None,
    image_bgr: np.ndarray | None = None,
) -> list[dict]:
    if occupied_for_splitting is None:
        occupied_for_splitting = occupied

    # A previous version also let this growth cross into ceiling-classified pixels, meant to
    # bridge a sloped ceiling-transition's thin misclassified strip. In practice it bled the wall
    # design onto ordinary flat ceilings — a much more common and more visibly wrong case than
    # the narrow transition it was meant to fix. Removed; ceiling stays a hard boundary.
    mask = _exclude_occupied(wall_mask, occupied_for_splitting)
    mask = _close(mask, iterations=2)
    mask = _exclude_occupied(mask, occupied_for_splitting)
    mask = _grow_into_unclassified(mask, occupied_for_splitting, max_reach_px=240)

    # `occupied` almost always includes the floor already (it's the union of every label but
    # "wall"), but not quite completely — a thin, weakly classified sliver right at the floor/wall
    # boundary can stay "unclassified" rather than confidently either one. The 240px, many-step
    # growth above is generous enough to tunnel all the way through such a sliver and then spread
    # sideways once it's past it, silently painting a design over real floor several hundred
    # pixels down from the actual wall. `floor_mask` — the floor detector's own, more direct
    # answer to "is this pixel floor" — is a harder backstop against exactly that: whatever it
    # calls floor can never end up counted as wall, tunnel or no tunnel.
    #
    # Not used raw, though: the floor mask is deliberately kept near-pixel-exact everywhere else
    # in this file (a rug's edge, the gap around a chair leg, genuinely isn't a straight line, and
    # smoothing it there would just move the boundary away from where it actually is). Subtracting
    # that same raw, fine-grained shape from the *wall* mask carved its jagged natural edge
    # straight into the wall's boundary, which really is architectural and should stay straight. A
    # close+open here only smooths away that fine detail for this one barrier use, without
    # touching the floor mask the floor's own region is built from.
    if floor_mask is not None:
        floor_barrier = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
        floor_barrier = cv2.morphologyEx(floor_barrier, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=3)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(floor_barrier))

    if mask.sum() / 255 < mask.size * MIN_AREA_FRACTION:
        return []

    # The per-pixel growth above chases the (potentially noisy) edge of `occupied` step by step,
    # so a real, dead-straight line — the top of the wall against the ceiling, a straight side
    # against another wall — can come out of it a few pixels in, a few pixels out, repeating: the
    # "pixels going up and down" a genuinely straight edge should never show. Re-run the same
    # contour-simplification used on the raw per-label mask, now on the grown result, so the
    # boundary the render actually clips against is straight wherever the real one is. That
    # simplification can nudge the boundary slightly outward, so re-exclude occupied pixels once
    # more afterward — an item on the wall must stay excluded even after straightening.
    mask = _smooth_mask_contours(mask, epsilon_frac=0.006)
    mask = _exclude_occupied(mask, occupied_for_splitting)
    # A single flat perspective quad across a real corner would warp the tile pattern as if both
    # wall faces were one flat plane, which looks wrong right at the bend — and a real corner is
    # also exactly where a user wants to put two different designs on two different wall faces.
    # Detect any real corners within this mask and cut both the quads *and* the mask itself at
    # the same lines, so each wall face becomes its own independently selectable/designable
    # region. Every cut's edges are shared, coordinate-for-coordinate, with its neighbor's — the
    # quads by construction, the submasks via `_split_mask_by_cuts` using those same cut
    # lines — so adjacent wall pieces still meet exactly, with no visible gap between them.
    # A single "wall" mask can genuinely contain more than one real, physically separate wall —
    # two walls with an open window, a doorway, or open space between them, not a corner bending
    # one continuous surface. Treating the whole thing as one bounding box let a large real gap
    # like that get bridged right through by corner-detection's own bbox math, and a scatter of
    # small, unrelated fragments near a glass partition or reflection get folded in as if they
    # were evidence about the *same* wall the main mass belongs to — exactly the kind of noisy,
    # inconsistent corner it kept finding there. Clustering first, and running corner-detection on
    # each real cluster independently, is what actually fixes that instead of tuning around it.
    regions: list[dict] = []
    for cluster_mask in _cluster_wall_components(mask):
        regions.extend(
            _extract_wall_regions_from_cluster(
                cluster_mask, gray, vp=vp, depth=depth, occupied=occupied, image_bgr=image_bgr
            )
        )
    regions = [r for r in regions if _is_regular_wall_shape(r["mask"])]
    # Additive only (see `_split_region_horizontally`): the vertical-corner pipeline above only
    # ever looks for left/right splits, so a real horizontal architectural line — a soffit above a
    # recessed wall, a chair rail — never gets a chance to split anything on its own axis. Run here,
    # before both merge passes below, so each merge sees the same per-cluster region shapes the
    # vertical pipeline actually produced — running it after either merge changes a region's shape
    # (by combining it with a same-column or small-fragment neighbor first) enough to sometimes
    # miss a real horizontal corner that's plainly there beforehand. Each resulting piece is tagged
    # `_no_occlusion_merge` so the two merge passes below — both built to re-join pieces occlusion
    # pulled apart — leave it alone rather than silently re-joining a deliberate split.
    horizontally_split: list[dict] = []
    for region in regions:
        split = _split_region_horizontally(region, gray, depth, image_bgr)
        if len(split) > 1:
            for r in split:
                r["_no_occlusion_merge"] = True
        horizontally_split.extend(split)
    regions = horizontally_split
    # Deliberately a final pass over every *fully resolved* region, not something folded into the
    # recursive splitting above: a piece occluded by furniture in the middle of a column (a plant,
    # a floor lamp) frequently only becomes its own distinct shape *after* corner-cutting has
    # already run — the upper part of the column is often still topologically connected to the
    # rest of the wall near the ceiling, only separating out once a real corner cuts it away from
    # its neighbor, while the lower part was already its own disconnected piece from the start.
    # Those two never exist as siblings to compare at any single point during the recursion; only
    # here, once every region this wall mask will ever produce actually exists, can they be.
    regions = _merge_same_column_regions(regions)

    # A last, coarse "is this actually worth its own checkbox" bar — separate from every earlier
    # noise/shape filter, which all ask whether a region is *real* wall, not whether it's *useful*
    # as an independent selection target. A genuine sliver of wall only visible in the gap between
    # a nightstand's legs is real, correctly detected, and still a bad standalone checkbox: no user
    # taps "select" specifically for that. Measured directly against four real, varied photos:
    # every legitimate independent wall face came out at 2.1% of the frame or larger, while the one
    # furniture-gap sliver found in practice measured 0.41% — a 5x gap with room on both sides, not
    # a value tuned to force one specific case.
    #
    # A below-bar region is *merged into its nearest larger neighbor*, not dropped — dropping it
    # was tried first and was wrong: it's real, visible wall, and simply discarding it left a
    # real hole a user could see and would never expect (paint stopping short of the wall visible
    # between a nightstand's legs is exactly as wrong as painting over the nightstand itself, just
    # in the opposite direction). Folding it into whichever larger region is spatially closest —
    # almost always the one it's actually a fragment of, just cut off by the furniture in front of
    # it — keeps it paintable through that region's own checkbox instead of needing one of its own.
    WALL_MIN_STANDALONE_FRACTION = 0.01
    total_px = mask.size

    def region_area_frac(r: dict) -> float:
        return (r["mask"] > 0).sum() / total_px

    def region_center(r: dict) -> tuple[float, float]:
        ys, xs = np.nonzero(r["mask"])
        return float(xs.mean()), float(ys.mean())

    # A region tagged by `_split_region_horizontally` is a deliberate split — a real molding strip
    # above a narrow return/side wall face can legitimately be a small fraction of the *whole*
    # frame even though it's a solid, obviously-separate surface, so it's exempt from this
    # generic "too small to bother with, fold into whatever's nearest" absorption regardless of
    # its own area fraction.
    small_ids = {
        id(r)
        for r in regions
        if region_area_frac(r) < WALL_MIN_STANDALONE_FRACTION and not r.get("_no_occlusion_merge")
    }
    small = [r for r in regions if id(r) in small_ids]
    large = [r for r in regions if id(r) not in small_ids]
    if not small:
        final_regions = large
    elif not large:
        # Nothing bigger to fold into — keep every region as-is rather than losing real,
        # independently-visible wall area entirely just because none of it individually clears
        # the standalone-checkbox bar.
        final_regions = regions
    else:
        large_masks = [r["mask"] for r in large]
        large_centers = [region_center(r) for r in large]
        for r in small:
            cx, cy = region_center(r)
            nearest = min(
                range(len(large_centers)),
                key=lambda i: (large_centers[i][0] - cx) ** 2 + (large_centers[i][1] - cy) ** 2,
            )
            large_masks[nearest] = cv2.bitwise_or(large_masks[nearest], r["mask"])
        merged = [_region_from_member(m) for m in large_masks]
        final_regions = [r for r in merged if r]

    # Additive only (see `_split_region_horizontally`): the vertical-corner pipeline above only
    # ever looks for left/right splits, so a real horizontal architectural line — a soffit above a
    # recessed wall, a chair rail — never gets a chance to split anything on its own axis.
    # Deliberately the true last step in this function, after both merges above — a molding strip
    # above a narrow return/side wall face can legitimately be a small fraction of the *whole*
    # frame even though it's a real, solid, obviously-separate surface; running this before the
    # small-region absorption above let that absorption fold it right back into its neighbor,
    # silently undoing a genuine split the moment after this created it.
    horizontally_split: list[dict] = []
    for region in final_regions:
        horizontally_split.extend(_split_region_horizontally(region, gray, depth, image_bgr))
    return horizontally_split


# Minimum "extent" (mask area / its own bounding-box area) a *small* wall region must have to be
# kept — see `_is_regular_wall_shape` for why this only ever applies below `WALL_SMALL_AREA_FRAC`.
# Measured directly against a real cluttered photo: two small, visually "torn paper" fragments
# users flagged (0.6% and 1.7% of the frame) both measured extent 0.38, while every legitimately
# solid small region measured 0.68 or higher — a clean gap with real margin on both sides.
WALL_MIN_EXTENT = 0.5

# A region below this fraction of the frame is "small" for the purposes of `_is_regular_wall_shape`
# — chosen well above the 1.7%-of-frame fragments that motivated this filter, and well below the
# 13.8%/21.3%-of-frame *legitimate* bedroom walls that a first version of this filter wrongly
# rejected (see the regression this was caught against): a real wall that large, with furniture or
# windows carving a low-extent shape out of its bounding box, is completely normal and must never
# be dropped just for having a lot of real occlusion in front of it.
WALL_SMALL_AREA_FRAC = 0.03


def _is_regular_wall_shape(mask: np.ndarray) -> bool:
    """False for a *small* wall region whose shape is too scattered/irregular to read as one real
    flat wall face. A small, low-extent fragment (only visible in gaps around dense clutter — a
    glass display case, open shelving, a cluster of light fixtures) is dropped because painting
    its actual shape never looks like a real surface, no matter how the boundary is refined.

    Deliberately scoped to small regions only: a *large* region with the same low extent is a
    completely different, ordinary case — a big real wall with a bed, window, or wardrobe legitimately
    occupying a large chunk of its bounding box — and must never be rejected on shape alone. Only a
    small region has no other explanation for a scattered shape; on a large one, "there's a lot of
    real furniture/window area in front of it" is already the far more likely explanation.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return False
    bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    if bbox_area == 0:
        return False
    area = (mask > 0).sum()
    if area >= mask.size * WALL_SMALL_AREA_FRAC:
        return True
    extent = area / bbox_area
    return extent >= WALL_MIN_EXTENT


def _cluster_wall_components(mask: np.ndarray) -> list[np.ndarray]:
    """Splits `mask` into separate per-cluster masks: connected components close enough together
    (within `gap_px`) merge into one cluster — a real wall's mask is rarely one single perfectly
    solid blob, since ordinary noise, a thin reflection, or a small missed strip can break it into
    a few nearby pieces that are all still obviously the same wall — while components separated by
    a much larger real gap (an open window, a doorway, genuine open space) stay as separate
    clusters, each to be corner-detected and region-extracted independently. Components too small
    to be a real wall segment (a few stray pixels of noise) are dropped entirely rather than
    contributing to any cluster's shape.
    """
    h, w = mask.shape[:2]
    min_area_px = mask.size * MIN_AREA_FRACTION * 0.4
    gap_px = max(15, round(w * 0.035))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    if not valid_ids:
        return []
    valid_mask = np.isin(labels, valid_ids).astype(np.uint8) * 255

    # Group by proximity: dilate enough to bridge only a *small* real gap, then whatever merges
    # into one connected blob under that dilation is one cluster — the dilation itself never
    # becomes part of any returned mask, it's purely how "close enough to be the same wall" gets
    # decided.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px * 2 + 1, gap_px * 2 + 1))
    grouped = cv2.dilate(valid_mask, dilate_kernel)
    gn, glabels = cv2.connectedComponents(grouped, connectivity=8)[:2]

    clusters = []
    for gid in range(1, gn):
        cluster_mask = np.where((glabels == gid) & (valid_mask > 0), np.uint8(255), np.uint8(0))
        if cluster_mask.any():
            clusters.append(cluster_mask)
    return clusters


def _largest_component_fraction(mask: np.ndarray) -> float:
    """Fraction of `mask`'s own total area that its single largest connected component covers —
    1.0 for one solid blob, much lower for a mask scattered across several separate islands."""
    total = int((mask > 0).sum())
    if total == 0:
        return 1.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 1.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / total


def _split_disconnected(mask: np.ndarray, min_area_px: float) -> list[np.ndarray]:
    """Splits `mask` into one mask per connected component with at least `min_area_px` pixels,
    dropping smaller pieces outright as noise (the same per-piece noise floor
    `_cluster_wall_components` already applies before its own proximity grouping)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        out.append(np.where(labels == i, mask, 0).astype(np.uint8))
    return out


# How much two disconnected pieces' x-ranges have to overlap, as a fraction of the *narrower*
# piece's own width, before they're candidates for being the same wall column. Deliberately not
# sufficient on its own — see `_merge_same_column_components` for why a piece occluded by tall
# furniture also has to be a *similar width* to, and *vertically disjoint* from, the piece it's
# merging into; x-overlap alone is trivially satisfied by a small fragment sitting anywhere inside
# a wide wall's own x-range, which is not the same thing as being a continuation of a narrow column.
SAME_COLUMN_OVERLAP_FRACTION = 0.6
# The two pieces' widths must be within this ratio of each other. A real occluded-column pair (the
# same physical wall strip, interrupted by furniture) is close to the same width above and below;
# a small fragment merging into an entire wide wall spanning most of the photo is a completely
# different shape relationship and must not qualify no matter how much its x-range overlaps.
SAME_COLUMN_WIDTH_RATIO = 2.0
# The two pieces' y-ranges must be genuinely disjoint (one ends, a gap, the other begins) rather
# than overlapping — a piece occluded by furniture in the middle of a column is only ever
# *interrupted*, not nested inside the other piece's own height. The gap itself is allowed to be
# fairly generous (a tall plant, a floor lamp) but is capped relative to the pieces' own height so
# it can't bridge two things that just happen to be far apart vertically.
SAME_COLUMN_MAX_GAP_RATIO = 0.6
# A hairline y-gap (a few pixels) is what a horizontal architectural cut looks like after the
# mask is partitioned — merging those two bands back together silently undoes a real soffit/chair-
# rail split. Furniture occlusion leaves a much larger hole (a plant, a floor lamp). This floor
# is the difference between those two cases.
SAME_COLUMN_MIN_GAP_RATIO = 0.15


def _merge_same_column_components(components: list[np.ndarray]) -> list[np.ndarray]:
    """Merges disconnected pieces that are plausibly the same wall column, interrupted by an
    occluding object, back into one region.

    `_split_disconnected` exists to stop a stray fragment in unrelated clutter from silently
    dragging along whatever else happens to share its cluster (see the wall-behind-a-glass-case
    case that motivated it) — but it can't distinguish that from a much more ordinary situation:
    one continuous flat wall with a tall piece of furniture (a plant, a floor lamp) standing in
    front of the middle of it, breaking the mask into a piece above and a piece below with nothing
    in between. Both pieces are obviously "the same wall" to a user — one checkbox, one design —
    and treating them as two forces someone to select and assign the same design twice for no
    reason.

    x-range overlap alone isn't enough to tell that situation apart from a small, unrelated
    fragment that just happens to sit somewhere inside a much wider wall's own x-range (an earlier
    version of this function merged exactly that, folding a tiny nightstand-area sliver into the
    entire main wall it had no real relationship to) — so this also requires the two pieces to be a
    similar *width* and to be vertically *disjoint* with only a modest gap, both real properties of
    "the same narrow column, interrupted," that a small stray fragment inside a big wall doesn't have.
    """
    if len(components) <= 1:
        return components

    def bounds(m: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(m)
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    ranges = [bounds(m) for m in components]

    def same_column(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        x0a, x1a, y0a, y1a = a
        x0b, x1b, y0b, y1b = b
        width_a, width_b = x1a - x0a + 1, x1b - x0b + 1
        if max(width_a, width_b) / min(width_a, width_b) > SAME_COLUMN_WIDTH_RATIO:
            return False
        x_overlap = min(x1a, x1b) - max(x0a, x0b)
        if x_overlap / min(width_a, width_b) < SAME_COLUMN_OVERLAP_FRACTION:
            return False
        # Disjoint check: one range must end before the other begins (a real gap), not overlap.
        gap = max(y0a, y0b) - min(y1a, y1b)
        if gap <= 0:
            return False
        shorter = min(y1a - y0a + 1, y1b - y0b + 1)
        if gap < SAME_COLUMN_MIN_GAP_RATIO * shorter:
            return False
        max_gap = SAME_COLUMN_MAX_GAP_RATIO * shorter
        return gap <= max_gap

    merged: list[np.ndarray] = []
    used = [False] * len(components)
    for i in range(len(components)):
        if used[i]:
            continue
        group_mask = components[i]
        group_bounds = ranges[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(components)):
                if used[j]:
                    continue
                if same_column(group_bounds, ranges[j]):
                    group_mask = cv2.bitwise_or(group_mask, components[j])
                    x0a, x1a, y0a, y1a = group_bounds
                    x0b, x1b, y0b, y1b = ranges[j]
                    group_bounds = (min(x0a, x0b), max(x1a, x1b), min(y0a, y0b), max(y1a, y1b))
                    used[j] = True
                    changed = True
        merged.append(group_mask)
    return merged


def _merge_same_column_regions(regions: list[dict]) -> list[dict]:
    """Applies `_merge_same_column_components` across a wall mask's *fully resolved* final
    regions — after every cluster split and every corner cut has already happened, not during
    either. A piece occluded by furniture in the middle of a column can easily still be
    topologically joined to the rest of the wall near the ceiling at cluster-split time, only
    becoming its own distinct shape once a real corner cuts it away from its neighbor deep inside
    the recursion — so the piece above the obstruction and the piece below it may never exist as
    siblings at any single point during that recursion to compare directly. They always both exist
    by the time every region this wall mask will ever produce has been built, which is exactly
    when this runs.
    """
    if len(regions) <= 1:
        return regions
    # A region tagged by `_split_region_horizontally` is a deliberate split, not an occlusion
    # fragment — this function's whole job is undoing the latter, so it never even sees the
    # former: pulled out before merging, added back unchanged after.
    protected = [r for r in regions if r.get("_no_occlusion_merge")]
    mergeable = [r for r in regions if not r.get("_no_occlusion_merge")]
    if len(mergeable) <= 1:
        return regions
    merged_masks = _merge_same_column_components([r["mask"] for r in mergeable])
    if len(merged_masks) == len(mergeable):
        return regions
    out = list(protected)
    for m in merged_masks:
        region = _region_from_member(m)
        if region:
            out.append(region)
    return out


# How solid each side of a corner split must stay to trust that split. A real architectural corner
# divides a wall into two faces that are each still one solid piece. Right around clutter a plain
# flat wall doesn't have — a glass display case, open shelving, a TV nook — the wall is only
# visible in scattered slivers on both sides of where the cut line falls, and an x-position-only
# split (see `_split_mask_by_cuts`) then hands each side a handful of unrelated islands rather than
# one coherent face. Selecting either resulting "wall" in that case visibly grabs fragments that
# look, to the user, like they belong to the other one — the two regions read as bleeding into each
# other even though the split is numerically exact. Below this bar, the corner is discarded and the
# whole cluster stays one region instead, which is a clean single selection rather than two
# confusing, interleaved ones.
MIN_SPLIT_SOLIDITY = 0.75


def _extract_wall_regions_from_cluster(
    mask: np.ndarray,
    gray: np.ndarray,
    vp: tuple[float, float] | None,
    depth: np.ndarray | None,
    occupied: np.ndarray | None = None,
    image_bgr: np.ndarray | None = None,
) -> list[dict]:
    """Corner-detects and builds region(s) for one already-clustered, single-wall mask — see
    `extract_wall_regions` for why clustering has to happen before this, not inside it."""
    # `mask` arrives already grouped by proximity (`_cluster_wall_components`), but everything
    # since then — contour simplification, subtracting a real excluded object (a TV, an open
    # shelving unit, a light fixture) that sits across a connecting strip, or falling back to one
    # big bounding quad when a corner split below gets discarded — can still pinch what was one
    # bridged blob at cluster-time into several disconnected pieces by the time this runs. Whether
    # two such pieces are still close in *pixels* turns out not to be a reliable stand-in for
    # whether they still read as the *same visible surface*: a small piece can end up brushing the
    # main mass at one lone corner while sitting, visually, in obviously different clutter (behind
    # an open shelf, around a light fixture) — merging it in there paints a stray-looking patch,
    # while simply dropping it throws away real, paintable wall area for no reason. Splitting each
    # such piece into its *own* independently selectable region — recursing so each still gets its
    # own corner-detection pass — does neither: every real visible patch stays available to design,
    # and none of them silently drags another one's disconnected fragment along with it.
    # The full noise floor (not `_cluster_wall_components`'s more lenient pre-merge one) — a piece
    # only reaches this point after already surviving that earlier, laxer filter and then getting
    # cut off from its cluster's main mass, so what's being decided here is "is this genuinely big
    # enough to stand alone as its own selectable wall", a stricter question than "is this obviously
    # just a stray pixel or two of noise".
    min_area_px = mask.size * MIN_AREA_FRACTION
    components = _split_disconnected(mask, min_area_px)
    if len(components) != 1:
        regions = []
        for component in components:
            regions.extend(
                _extract_wall_regions_from_cluster(
                    component, gray, vp=vp, depth=depth, occupied=occupied, image_bgr=image_bgr
                )
            )
        return regions
    mask = components[0]

    ys, xs = np.nonzero(mask)
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    top_y, bottom_y = float(bbox[0]), float(bbox[2])
    corners = _detect_wall_corners(gray, mask, vp=vp, depth=depth)

    def build(cut_points: list[tuple[float, float]]) -> tuple[list, list]:
        cuts = [(float(bbox[1]), float(bbox[1])), *cut_points, (float(bbox[3] + 1), float(bbox[3] + 1))]
        quads = [
            [
                (cuts[i][0], top_y),
                (cuts[i + 1][0], top_y),
                (cuts[i + 1][1], bottom_y),
                (cuts[i][1], bottom_y),
            ]
            for i in range(len(cuts) - 1)
        ]
        return cuts, quads

    cuts, quads = build(corners)

    if len(quads) == 1:
        final_mask = _exclude_occupied(mask, occupied) if occupied is not None else mask
        region = _region_from_member(final_mask, quads=quads)
        return [region] if region else []

    submasks = _split_mask_by_cuts(mask, cuts, top_y, bottom_y)

    # A resulting wall face narrower than this is suspicious on its own, regardless of how solid
    # or well-scored the corner producing it was — real, useful, separately-paintable wall faces
    # are rarely a sliver a few percent of the cluster's own width. Found in practice: a corner
    # candidate right where a wall meets a curtain or other object scores exactly like a real
    # corner (`score()`'s shading/depth signals measure *any* consistent brightness/depth
    # difference across the line, and a wall-to-fabric material change produces one just as
    # reliably as a wall-to-wall plane change does) — nothing in that scoring distinguishes the
    # two. The one property that reliably does: a real architectural face has real width; a
    # material-change artifact right at an object's own edge only ever produces a thin, almost
    # zero-width remainder between the object and the true cluster boundary.
    min_wall_width_px = max(10.0, (bbox[3] - bbox[1] + 1) * 0.015)
    # Same "is this genuinely big enough to stand alone" bar `_split_disconnected` already applies
    # to a whole disconnected piece, now applied to a corner-cut *submask* too — a corner can pass
    # both the solidity and width checks above and still slice off a sliver too small in total
    # area to be a meaningfully separate, useful checkbox (found in practice: two ~45px-wide but
    # only ~65px-tall slivers next to a nightstand, from a corner that was real evidence but not a
    # face worth its own separate selection). Better as one small combined region than two
    # near-invisible ones each demanding their own tap.
    min_wall_area_px = mask.size * (MIN_AREA_FRACTION * 0.5)

    def submask_width(m: np.ndarray) -> float:
        xs = np.nonzero(m)[1]
        return float(xs.max() - xs.min() + 1) if xs.size else 0.0

    def submask_area(m: np.ndarray) -> float:
        return float((m > 0).sum())

    # Drop corners one at a time until every remaining cut produces solid, wide-enough, big-enough
    # pieces on both sides, or none are left and the cluster falls back to a single flat region.
    def worst_offender(subs: list[np.ndarray]) -> int:
        scores = [
            min(
                _largest_component_fraction(m) / MIN_SPLIT_SOLIDITY,
                submask_width(m) / min_wall_width_px,
                submask_area(m) / min_wall_area_px,
            )
            for m in subs
        ]
        return min(range(len(scores)), key=lambda i: scores[i])

    while corners and (
        min(_largest_component_fraction(m) for m in submasks) < MIN_SPLIT_SOLIDITY
        or min(submask_width(m) for m in submasks) < min_wall_width_px
        or min(submask_area(m) for m in submasks) < min_wall_area_px
    ):
        # The side that's failing worst (as a fraction of whichever bar it's failing) is the one
        # most likely responsible — drop the corner adjacent to it rather than an arbitrary choice.
        worst_side = worst_offender(submasks)
        corner_idx = max(0, min(worst_side, len(corners) - 1))
        del corners[corner_idx]
        cuts, quads = build(corners)
        if len(quads) == 1:
            break
        submasks = _split_mask_by_cuts(mask, cuts, top_y, bottom_y)

    if len(quads) == 1:
        final_mask = _exclude_occupied(mask, occupied) if occupied is not None else mask
        region = _region_from_member(final_mask, quads=quads)
        return [region] if region else []

    regions = []
    for quad, submask in zip(quads, submasks):
        final_submask = _exclude_occupied(submask, occupied) if occupied is not None else submask
        region = _region_from_member(final_submask, quads=[quad])
        if region:
            regions.append(region)
    return regions


def extract_floor_region(floor_mask: np.ndarray, occupied: np.ndarray) -> list[dict]:
    # The floor is one continuous surface even when furniture occludes parts of it — treat
    # every visible floor pixel as a single selectable region. Close with a larger kernel than
    # walls need since floors get fragmented more (chair/table legs, rugs), and reach further
    # when growing into unclassified space: the floor is usually the surface with the most area
    # the model is genuinely unsure about (reflections, shadows under furniture, rug edges,
    # distant/low-contrast corners), so it needs more room than a wall to actually reach every
    # edge of the image instead of stopping short of real, paintable floor.
    #
    # Most room photos are wider than they are tall, and the model's confident detections tend
    # to cluster toward the center (flatter, better-lit, viewed closer to head-on) — floor near
    # the left/right edges is usually the same surface seen at a much more oblique angle, which
    # is exactly where the model is least confident and needs the most help from growth to reach.
    # Sizing the reach off the *smaller* dimension caps it at roughly the room's height even in a
    # wide shot, well short of the actual gap out to the side walls; size it off the larger
    # dimension instead so a wide photo gets a wide-enough budget. This is still bounded by real
    # object edges regardless of how large the budget is — growth stops the instant it would
    # touch anything the model is confident is something else — so being generous here costs
    # nothing when the real floor doesn't extend that far, and only matters when it does.
    h, w = floor_mask.shape[:2]
    reach = max(600, round(0.55 * max(h, w)))
    mask = _exclude_occupied(floor_mask, occupied)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, FLOOR_CLOSE_KERNEL, iterations=4)
    mask = _exclude_occupied(mask, occupied)
    mask = _grow_into_unclassified(mask, occupied, max_reach_px=reach)
    if mask.sum() / 255 < mask.size * MIN_AREA_FRACTION:
        return []

    # Deliberately not re-straightened here (unlike the wall): the floor's boundary is usually
    # the least architectural, most irregular one in the photo — a rug's edge, the gap around a
    # chair leg, a baseboard that isn't perfectly straight in the photo either. Polygon-simplifying
    # it trades exactly the pixel-level accuracy that matters most here for a cleaner-looking line
    # that isn't actually where the real edge is. The real edge is allowed to go up, down, left,
    # right — wherever it actually is — rather than being forced into fewer, straighter segments.

    region = _region_from_member(mask)
    return [region] if region else []


def analyze_image(image_bytes: bytes, target_width: int, target_height: int, debug: bool = False) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Kept at the photo's true native resolution (never downscaled) specifically for the final
    # boundary matte in `encode()` below — every per-pixel *decision* in this function still runs
    # at the capped `analysis_size` (that's what actually needs to be fast), but the true edge
    # location right at a wall/object boundary is real information a 1600px-capped analysis image
    # doesn't fully carry on a much larger source photo. Cheap: one extra resize per region, no
    # extra model inference.
    full_res_image = image

    # The model itself works at a fixed internal resolution regardless of input size, so running
    # the pixel-array post-processing (connected components, morphology, Hough) on a full 4000px+
    # phone photo just burns time for no extra accuracy. Cap it; results get scaled back up to
    # target_width/target_height (the true original dimensions) below.
    orig_w, orig_h = image.size
    scale = min(1.0, MAX_ANALYSIS_DIMENSION / max(orig_w, orig_h))
    if scale < 1.0:
        image = image.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    analysis_size = image.size  # (width, height) actually analyzed at
    image_rgb_arr = np.array(image)
    gray = cv2.cvtColor(image_rgb_arr, cv2.COLOR_RGB2GRAY)
    image_bgr = cv2.cvtColor(image_rgb_arr, cv2.COLOR_RGB2BGR)

    # Two independent geometric signals, computed once per photo and reused wherever a corner
    # candidate gets scored below, alongside color/segmentation: monocular relative depth (two
    # different wall planes genuinely sit at different depth, even where their paint color
    # doesn't differ) and the room's vertical vanishing point (a real architectural vertical
    # points at it; a decorative line usually only coincidentally shares its rough angle).
    depth = _estimate_depth(image, analysis_size)
    vp = _estimate_vertical_vanishing_point(gray)

    segmenter = get_segmenter()
    outputs, extra_probs = _segment(image, segmenter, extra_prob_labels=("floor",))
    floor_prob = extra_probs["floor"]

    # Encoder forward pass run once per photo here, reused by every `_occupied_mask` call below
    # via `sam2_predictor.predict(box=...)` (cheap once the image is set) — see
    # `_sam2_refine_object_mask`. `None` when SAM 2 isn't available; every call site already
    # falls back to the original GrabCut/watershed refinement in that case.
    sam2_predictor = get_sam2_predictor()
    if sam2_predictor is not None:
        sam2_predictor.set_image(image_rgb_arr)

    # A rug/carpet sits directly on the floor and isn't its own separate real object the way
    # furniture is — treating it as "not floor" leaves most of a real room's visible floor area
    # unpaintable (rugs are extremely common), which reads as "floor detection missed most of
    # the room" even though the underlying floor detection was working correctly.
    #
    # Floor's own edge is deliberately kept near-raw here (a tiny, near-zero epsilon — effectively
    # only removing truly redundant collinear points, not simplifying the shape) rather than using
    # the same straightening the wall gets: a rug's edge, the gap around a chair leg, isn't a
    # straight architectural line to begin with, so forcing it into fewer/straighter segments only
    # moves the boundary away from where it actually is.
    floor_mask = cv2.bitwise_or(
        _mask_for_label(outputs, "floor", analysis_size, epsilon_frac=0.0),
        _mask_for_label(outputs, "rug", analysis_size, epsilon_frac=0.0),
    )

    # Recover floor the model was genuinely torn on, not just what won the per-pixel argmax: a
    # reflective, patterned, or shadowed patch where floor was a near-tie with whatever narrowly
    # won is still very likely real floor. This has to stay close to the argmax boundary, not
    # just "some noticeable chance" — a real object the model is mostly-but-not-fully confident
    # about (say 65/35 against floor) must still read as that object, never as recoverable floor,
    # or a design ends up painted onto something real. Only a genuine near-tie counts.
    FLOOR_TIE_THRESHOLD = 0.42
    floor_lenient_raw = (floor_prob > FLOOR_TIE_THRESHOLD).astype(np.uint8) * 255
    floor_mask = cv2.bitwise_or(floor_mask, _smooth_mask_contours(floor_lenient_raw, epsilon_frac=0.0))
    raw_floor_mask = floor_mask.copy() if debug else None

    # Refine the initial floor mask with guided filter to snap perfectly to baseboard/wall junctions
    floor_mask = _guided_filter_refine(floor_mask, gray, r=8, eps=0.015)
    # Exclude furniture regions from floor mask. This runs *before* `floor_occupied` (the
    # comprehensive, all-labels exclusion applied later in `extract_floor_region`) even exists, so
    # it isn't the final safety net against painting over furniture — that's `floor_occupied`'s
    # job. What this narrower, earlier pass actually protects is two things that run on `floor_mask`
    # itself, before that later exclusion: `_detect_textured_objects` (scoped to whatever surface
    # this mask claims is floor) and `_floor_baseline_unblock` (which finds each column's *topmost*
    # floor pixel to build a baseline — a stray furniture pixel this mask still contains, swept in
    # by the lenient near-tie recovery just above, pulls that baseline upward into the furniture
    # itself, corrupting which wall pixels get unblocked in that column). "armchair" and "coffee
    # table" are their own separate ADE20K classes from "chair"/"table" — both were missing here
    # despite being two of the most common floor-sitting furniture pieces in a real room photo.
    FURNITURE_LABELS = {
        "chair", "armchair", "table", "coffee table", "sofa", "bed", "cabinet", "desk", "shelf",
        "tv", "ottoman", "stool", "bench", "wardrobe", "chest of drawers",
    }
    furniture_mask = np.zeros_like(floor_mask)
    for label in FURNITURE_LABELS:
        furniture_mask = cv2.bitwise_or(furniture_mask, _mask_for_label(outputs, label, analysis_size, epsilon_frac=0.0))
    floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(furniture_mask))

    wall_mask = _mask_for_label(outputs, "wall", analysis_size)
    raw_wall_mask = wall_mask.copy() if debug else None
    # Refine the initial wall mask with guided filter to snap perfectly to ceiling/floor junctions and columns
    wall_mask = _guided_filter_refine(wall_mask, gray, r=8, eps=0.015)
    # Exclude ceiling regions from wall mask
    ceiling_mask = _mask_for_label(outputs, "ceiling", analysis_size)
    wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(ceiling_mask))


    # A depth-guided second refinement pass was tried here (and for the floor mask above) to snap
    # same-color-but-different-depth boundaries the color guide alone can't see. In practice it did
    # more harm than good on real photos: Depth Anything's depth map is upsampled from a much
    # coarser native resolution than the photo, so its own edges are blocky/low-frequency compared
    # to the actual photographic boundary, and guiding the mask toward them pulled real, clean
    # boundaries (the ceiling line, the edge around a window or plant) into a visibly torn, jagged
    # shape instead of smoothing them. Reverted; `depth` is still used elsewhere (corner scoring),
    # just never again to directly reshape a mask boundary.

    # ADE20K has no class for "wall hanging" — a flat, wall-mounted textile or tapestry the model
    # just reads as part of the wall itself, so nothing that only excludes *other confidently
    # classified* labels can catch it. Texture stands in for a real class here: an interior patch
    # of unusually high local variance inside an otherwise near-uniform painted wall is almost
    # certainly a distinct object, not the wall.
    wall_objects = _detect_textured_objects(gray, wall_mask)
    # A second, independent signal for the same "segmentation/color missed a real object" problem
    # — this time catching what color/texture *can't* see at all: an object that's the same color
    # and finish as the wall behind it (a white frame on a white wall) but still physically sits
    # closer to the camera. See `_depth_foreground_mask`.
    wall_objects = cv2.bitwise_or(wall_objects, _depth_foreground_mask(wall_mask, depth))
    wall_mask = _exclude_occupied(wall_mask, wall_objects)

    # The same signal applies to the floor — a small mat, basket, or pet bed has no ADE20K class
    # of its own either, and would otherwise just get silently painted over. The one thing this
    # must never flag is a real rug, which floor is *supposed* to count as paintable and which is
    # itself textured/patterned almost by definition — so anything the model already calls "rug"
    # is exempted before the result is used, regardless of how much local variance it has.
    rug_raw = _mask_for_label(outputs, "rug", analysis_size)
    floor_objects = _detect_textured_objects(gray, floor_mask)
    floor_objects = cv2.bitwise_and(floor_objects, cv2.bitwise_not(rug_raw))
    floor_depth_fg = cv2.bitwise_and(_depth_foreground_mask(floor_mask, depth), cv2.bitwise_not(rug_raw))
    floor_objects = cv2.bitwise_or(floor_objects, floor_depth_fg)
    floor_mask = _exclude_occupied(floor_mask, floor_objects)

    # Where "looking through a window" is actually allowed to happen: a real windowpane, plus a
    # modest margin around it for the sliver of frame/sill right at its edge — not the whole
    # image. See `_occupied_mask` for why this can't just be a blanket exclusion everywhere.
    window_raw = _mask_for_label(outputs, "windowpane", analysis_size)
    near_window = cv2.dilate(window_raw, MORPH_KERNEL, iterations=6) if window_raw.any() else window_raw

    # Each surface's growth must stop at the *other* surface's real boundary too, not just at
    # unrelated objects — otherwise floor could bleed up onto the wall and vice versa.
    #
    # Symmetric with the recovery above: a pixel only stops blocking floor growth if it was that
    # same kind of near-tie, not merely because floor got some nonzero share of the probability.
    # Anything the model leans away from floor by a real margin must keep blocking growth exactly
    # as before — this only unblocks the narrow band right at the argmax boundary.
    floor_occupied = _occupied_mask(outputs, {"floor", "rug"}, analysis_size, near_window, image_bgr, sam2_predictor)
    floor_is_tie = (floor_prob > FLOOR_TIE_THRESHOLD).astype(np.uint8) * 255
    floor_occupied = cv2.bitwise_and(floor_occupied, cv2.bitwise_not(floor_is_tie))

    # A dim, shadowed strip right at the base of the wall (behind a plant pot, under low
    # furniture) can get confidently misread as wall rather than floor — see
    # `_floor_baseline_unblock`. Stop letting those specific pixels block floor growth.
    baseline_unblock = _floor_baseline_unblock(floor_mask, wall_mask)
    floor_occupied = cv2.bitwise_and(floor_occupied, cv2.bitwise_not(baseline_unblock))
    floor_occupied = cv2.bitwise_or(floor_occupied, floor_objects)
    # Refine floor occupied with guided filter so cutouts snap precisely to furniture/edges.
    # r=6 (a 13px box) was smoothing straight over genuinely narrow gaps of real, visible floor
    # between adjacent thin furniture legs — a wireframe chair a few pixels wide at each leg, with
    # real floor peeking through between them, came out as one solid excluded blob instead of
    # several separately-excluded thin legs with real paintable floor between. r=1 is the largest
    # radius that doesn't bridge a gap that narrow while still refining the coarser, larger cutout
    # boundaries (a couch, a table) it primarily exists for.
    floor_occupied = _guided_filter_refine(floor_occupied, gray, r=1, eps=0.01)
    # A conservative 1px safety margin around every excluded object: right at a real object's
    # true edge, the guide-filtered cutout can still leave a sliver of anti-aliased/blended pixels
    # that read as "surface" by a hair. Optimizing for "never recolor an object" over "recolor
    # every possible surface pixel" means that thin, genuinely ambiguous fringe should stay
    # protected rather than painted — one pixel is cheap and costs no real, unambiguous surface
    # area anywhere else.
    floor_occupied = cv2.dilate(floor_occupied, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # wall_occupied is the full mask of everything that is not wall (e.g. TV, pictures, furniture),
    # subtracted from final wall regions so we never paint over TV screens or other wall decor.
    wall_occupied = _occupied_mask(outputs, {"wall"}, analysis_size, near_window, image_bgr, sam2_predictor)
    wall_occupied = cv2.bitwise_or(wall_occupied, wall_objects)
    # Kept before the r=6 smoothing below, specifically for re-excluding objects a second time at
    # `encode_size` (see `wall_occupied_encoded` near the end of this function). A curtain's own
    # fold is real, fine, high-frequency detail at `analysis_size` (this photo's curtain panel is
    # only a handful of pixels wide there) — r=6 (a 13px box, sized for smoothing coarser cutouts
    # like a couch or TV) blurs a boundary that thin away entirely, and refining an *upscaled copy
    # of an already-blurred-away boundary* later can't recover detail smoothed out at this stage.
    wall_occupied_sharp = wall_occupied.copy()
    wall_occupied = _guided_filter_refine(wall_occupied, gray, r=6, eps=0.01)
    wall_occupied = cv2.dilate(wall_occupied, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # wall_occupied_for_splitting excludes flat wall-mounted or wall-obscuring objects (like TVs,
    # pictures, posters, mirrors) so the wall mask is kept solid and contiguous for the growth and
    # corner-splitting phase, avoiding fragmentation that would throw off corner detection.
    wall_occupied_for_splitting = _occupied_mask(
        outputs,
        {"wall", "television", "television receiver", "screen", "picture", "painting", "mirror", "board", "blackboard", "whiteboard", "clock", "poster"},
        analysis_size,
        near_window,
        image_bgr,
        sam2_predictor,
    )
    wall_occupied_for_splitting = cv2.bitwise_or(wall_occupied_for_splitting, wall_objects)
    wall_occupied_for_splitting = _guided_filter_refine(wall_occupied_for_splitting, gray, r=6, eps=0.01)
    wall_occupied_for_splitting = cv2.dilate(wall_occupied_for_splitting, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    floor_regions = extract_floor_region(floor_mask, floor_occupied)
    wall_regions = extract_wall_regions(
        wall_mask, wall_occupied, gray, vp=vp, depth=depth, floor_mask=floor_mask,
        occupied_for_splitting=wall_occupied_for_splitting, image_bgr=image_bgr
    )

    # The room photo is rendered on screen at its own true resolution (often much bigger than
    # 1600px on a modern phone photo), but every mask above was computed at the capped
    # `analysis_size` for speed. Left as-is, the renderer has to blow that low-res mask up to
    # cover a much bigger canvas — every straight edge the simplification pass just cleaned up,
    # and every object cutout (a lamp, a remote, anything sitting on the wall or floor), gets
    # magnified along with it, turning a clean edge back into a visibly blocky one exactly where
    # it matters most. None of the expensive work (segmentation, morphology, growth, contour
    # fitting) needs to re-run at a higher resolution to fix this — only the finished mask needs
    # to be handed over at a resolution closer to what it'll actually be displayed at; a cubic
    # resize of the already-clean shape gets most of the benefit for the cost of one resize call.
    MASK_ENCODE_MAX_DIMENSION = 2600
    encode_scale = min(MASK_ENCODE_MAX_DIMENSION, max(target_width, target_height)) / max(analysis_size)
    encode_scale = max(encode_scale, 1.0)
    encode_size = (round(analysis_size[0] * encode_scale), round(analysis_size[1] * encode_scale))

    # A real boundary — where a wall meets an object, another wall, or the floor — is a hard
    # pixel-coverage decision at the *true* photo resolution: a boundary pixel is genuinely some
    # fraction wall and some fraction not, not one or the other. Deciding that with a plain
    # `> 127` re-threshold after the resize above (still done below, unchanged, for the mask's
    # overall *shape*) throws that fraction away and replaces it with a jagged, fully-opaque-or-
    # fully-transparent edge — exactly the "binarized too early" failure this step exists to fix.
    # Guided-filtering the already-correct binary shape against the photo's own true-resolution
    # detail (not the capped analysis-resolution one every classification decision above ran at)
    # recovers that fraction as a soft alpha matte, anchored to the same edge the binary mask
    # already settled on — it only *feathers* that edge against real image detail, it can't move
    # or reshape it. Computed once per photo (not per region) since every region shares the same
    # guide image.
    encode_guide_gray = None
    if encode_size[0] > 0 and encode_size[1] > 0:
        encode_guide_rgb = full_res_image.resize(encode_size, Image.LANCZOS)
        encode_guide_gray = cv2.cvtColor(np.array(encode_guide_rgb), cv2.COLOR_RGB2GRAY)

    # Resized once per photo (not per region, they'd all resize the same source array) so `encode`
    # can re-exclude every real object — a curtain, a window, a picture — right after the resize
    # and contour-smoothing below, which is exactly the step that was silently letting them back
    # in. `_exclude_occupied` already runs earlier at `analysis_size`, but resizing up to
    # `encode_size` (a real, often 2x+ upscale — see `MASK_ENCODE_MAX_DIMENSION` above) and then
    # re-smoothing the boundary can nudge it outward past that already-correct exclusion, same as
    # the "simplification can nudge the boundary slightly outward" re-exclusion already done right
    # after smoothing at analysis resolution elsewhere in this file. Skipping this step here was the
    # gap: nothing clipped the boundary back after the *second*, higher-resolution smoothing pass,
    # so a wall region could end up visibly painting over real curtain fabric or window trim right
    # at its edge even though every earlier stage correctly excluded it.
    wall_occupied_encoded = floor_occupied_encoded = None
    if encode_size != analysis_size:
        wall_occupied_encoded = cv2.resize(wall_occupied_sharp, encode_size, interpolation=cv2.INTER_NEAREST)
        floor_occupied_encoded = cv2.resize(floor_occupied, encode_size, interpolation=cv2.INTER_NEAREST)
        # A nearest-neighbor upscale of `wall_occupied` just blows its blocky, `analysis_size`-
        # resolution silhouette up bigger — it doesn't recover any real detail lost at that lower
        # resolution. A curtain's own fold edges are fine, high-frequency detail exactly like the
        # ones every other mask in this function already re-snaps to the full-resolution photo via
        # `_guided_filter_refine` — without doing the same here, the exclusion above still clips to
        # last section's coarse, blocky curtain silhouette rather than its true, finely-pleated one,
        # which is exactly what let real curtain fabric back into the encoded wall mask between
        # folds too narrow for the low-res exclusion to have ever resolved as "not wall".
        if encode_guide_gray is not None:
            wall_occupied_encoded = _guided_filter_refine(wall_occupied_encoded, encode_guide_gray, r=4, eps=0.01)
            floor_occupied_encoded = _guided_filter_refine(floor_occupied_encoded, encode_guide_gray, r=4, eps=0.01)
        # A small safety margin around the wall exclusion specifically, on top of everything above:
        # `_smooth_mask_contours` (run on the wall mask right before this, to straighten a real
        # architectural line) can still nudge the wall's own boundary a couple of pixels outward
        # right where it runs beside a real excluded object — a curtain, a window frame — even after
        # every fix above. Growing the exclusion itself by a couple of real pixels can only ever
        # shrink the painted wall area a hair right at such an edge, never invent false wall
        # elsewhere, so this is one-directional: it costs a sliver of paintable area directly next
        # to a real object, in exchange for that area never visibly reading as painted-over fabric.
        wall_margin_px = max(1, round(2 * encode_scale))
        wall_occupied_encoded = cv2.dilate(
            wall_occupied_encoded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_margin_px * 2 + 1,) * 2)
        )

    def encode(region, kind, index):
        mask_to_encode = region["mask"]
        if mask_to_encode.any():
            mean_intensity = float(gray[mask_to_encode > 0].mean()) / 255.0
        else:
            mean_intensity = 0.5

        if encode_size != analysis_size:
            mask_to_encode = cv2.resize(mask_to_encode, encode_size, interpolation=cv2.INTER_CUBIC)
            # Cubic resize of a binary mask can ring (over/undershoot) right at the boundary,
            # leaving a fringe of not-quite-0-or-255 pixels there; re-threshold before treating it
            # as a clean shape again — needed for every region regardless of kind.
            mask_to_encode = (mask_to_encode > 127).astype(np.uint8) * 255
            # Re-running contour smoothing at *this* resolution is what actually keeps a wall's
            # boundary clean at the size it's really displayed at (any staircase that survived at
            # analysis resolution just got magnified by the resize above, not cleaned up by it).
            # Deliberately skipped for floor, same reasoning as `extract_floor_region`: a floor's
            # boundary is supposed to be irregular (furniture legs, a rug's real edge), and
            # Douglas-Peucker simplification — plus the small-feature drop inside
            # `_smooth_mask_contours` — would erase exactly the fine detail around furniture that
            # needs to stay sharp for the paint-over to look right.
            if kind == "wall":
                mask_to_encode = _smooth_mask_contours(
                    mask_to_encode, epsilon_frac=0.006, max_epsilon=5.0 * encode_scale
                )
            occupied_encoded = wall_occupied_encoded if kind == "wall" else floor_occupied_encoded
            if occupied_encoded is not None:
                mask_to_encode = _exclude_occupied(mask_to_encode, occupied_encoded)

        if encode_guide_gray is not None and encode_guide_gray.shape[:2] == mask_to_encode.shape[:2]:
            # r=3 here is deliberately small relative to `_guided_filter_refine`'s own r=6-8 — this
            # pass only feathers the edge the binary shape above already decided on, not re-derive
            # it, so it only ever needs to look a few true-resolution pixels either side of that
            # edge. Runs regardless of whether a resize happened above — the anti-aliasing benefit
            # is real even when `encode_size == analysis_size`.
            #
            # r=1 (a 3px box), not the wider r=3 this used to be: on a small/low-resolution source
            # photo (a modest stock photo well under the analysis cap, not a downscaled 4000px
            # phone shot) a real gap between two thin objects close together — trailing vine
            # strands off one hanging planter, say — can be only a handful of true pixels wide.
            # r=3's wider blur radius reaches across the *whole* gap from both sides at once there,
            # so the middle of a real, correctly-detected paintable gap never settles at a
            # confident "paint here" alpha — it stays a permanent half-blended smear instead,
            # reading as unpainted even though the shape underneath is already correct. r=1 still
            # anti-aliases every ordinary edge (that only ever needs a pixel or two), it just no
            # longer erases a thin real gap to do it.
            alpha = _guided_filter_alpha(mask_to_encode, encode_guide_gray, r=1, eps=0.01)
            mask_to_encode = (alpha * 255.0).astype(np.uint8)
        ok, buf = cv2.imencode(".png", mask_to_encode)
        mask_png_b64 = None
        if ok:
            import base64

            mask_png_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        sx = target_width / analysis_size[0]
        sy = target_height / analysis_size[1]
        quads = [[[x * sx, y * sy] for x, y in quad] for quad in region["quads"]]
        centroid = [region["centroid"][0] * sx, region["centroid"][1] * sy]
        return {
            "id": f"{kind}-{index}",
            "kind": kind,
            "quads": quads,
            "centroid": centroid,
            "maskPng": mask_png_b64,
            "maskWidth": encode_size[0],
            "maskHeight": encode_size[1],
            "meanIntensity": mean_intensity,
        }

    result = {
        "floor": [encode(r, "floor", i) for i, r in enumerate(floor_regions)],
        "wall": [encode(r, "wall", i) for i, r in enumerate(wall_regions)],
    }
    if debug:
        result["debug"] = _build_debug_bundle(
            image=np.array(image),
            gray=gray,
            depth=depth,
            raw_wall_mask=raw_wall_mask,
            raw_floor_mask=raw_floor_mask,
            wall_mask=wall_mask,
            floor_mask=floor_mask,
            outputs=outputs,
            wall_objects=wall_objects,
            floor_objects=floor_objects,
            wall_regions=wall_regions,
            floor_regions=floor_regions,
        )
    return result


def _encode_png_b64(arr: np.ndarray) -> str | None:
    import base64

    ok, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _build_debug_bundle(
    image: np.ndarray,
    gray: np.ndarray,
    depth: np.ndarray,
    raw_wall_mask: np.ndarray,
    raw_floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    floor_mask: np.ndarray,
    outputs: list[dict],
    wall_objects: np.ndarray,
    floor_objects: np.ndarray,
    wall_regions: list[dict],
    floor_regions: list[dict],
) -> dict:
    """Assembles every intermediate layer worth inspecting into one debug bundle — the raw vs.
    refined mask for each surface, a genuine foreground-object protection mask, a colorized depth
    map, detected corner/quad geometry drawn over the photo, and the final paint mask plus a quick
    tinted preview of it. Only ever built when the caller explicitly asks for it (see `debug` on
    `analyze_image`) — every array here already exists in the normal analysis path, so turning
    this off costs nothing beyond the handful of extra encode calls this function itself makes.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    depth_norm = depth - depth.min()
    max_range = depth_norm.max()
    if max_range > 1e-6:
        depth_norm = depth_norm / max_range
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    # A true "real object" view, not the raw growth-blocking mask each surface computes for
    # itself (which counts the *other* surface as "occupied" too, and so would show almost the
    # entire photo as protected) — every segmented label except the architectural surfaces
    # (wall/floor/rug/ceiling), plus whatever the texture- and depth-based detectors caught that
    # segmentation missed entirely.
    object_mask = np.zeros_like(gray)
    for entry in outputs:
        if entry["label"] not in {"wall", "floor", "rug", "ceiling"}:
            object_mask = cv2.bitwise_or(object_mask, _raw_label_mask(entry, gray.shape[::-1]))
    object_mask = cv2.bitwise_or(object_mask, cv2.bitwise_or(wall_objects, floor_objects))
    object_overlay = bgr.copy()
    object_overlay[object_mask > 0] = (
        object_overlay[object_mask > 0] * 0.35 + np.array([0, 0, 255]) * 0.65
    ).astype(np.uint8)

    final_mask = np.zeros_like(gray)
    for region in [*wall_regions, *floor_regions]:
        final_mask = cv2.bitwise_or(final_mask, region["mask"])

    geometry_overlay = bgr.copy()
    for region in wall_regions:
        for quad in region["quads"]:
            pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(geometry_overlay, [pts], True, (0, 255, 255), 2)
    for region in floor_regions:
        for quad in region["quads"]:
            pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(geometry_overlay, [pts], True, (255, 200, 0), 2)

    final_preview = bgr.copy()
    tint = np.array([60, 200, 60])
    final_preview[final_mask > 0] = (final_preview[final_mask > 0] * 0.55 + tint * 0.45).astype(np.uint8)

    return {
        "original": _encode_png_b64(bgr),
        "rawWallMask": _encode_png_b64(raw_wall_mask),
        "rawFloorMask": _encode_png_b64(raw_floor_mask),
        "refinedWallMask": _encode_png_b64(wall_mask),
        "refinedFloorMask": _encode_png_b64(floor_mask),
        "objectMask": _encode_png_b64(object_overlay),
        "depth": _encode_png_b64(depth_color),
        "geometry": _encode_png_b64(geometry_overlay),
        "finalMask": _encode_png_b64(final_mask),
        "finalPreview": _encode_png_b64(final_preview),
        "diagnostics": _compute_diagnostics(gray, object_mask, final_mask, wall_regions, floor_regions),
    }


def _compute_diagnostics(
    gray: np.ndarray,
    object_mask: np.ndarray,
    final_mask: np.ndarray,
    wall_regions: list[dict],
    floor_regions: list[dict],
) -> dict:
    """Self-consistency diagnostics for the final paint mask — computed from signals already
    inside this pipeline, not against any hand-labeled ground truth (this codebase has none).
    `objectContaminationRatio` in particular is the metric this task calls out as highest
    priority: what fraction of the pixels this run is about to paint also fall inside its own,
    independently-built object mask (segmentation labels + texture-object + depth-foreground
    detection). It is a real, meaningful regression check across runs/photos and a genuine bug
    signal whenever it's above ~0 — but it is a *self*-consistency check: if the object detector
    itself misses a real object, this number won't catch that, since both the paint mask and the
    object mask share the same blind spot. True object-contamination measurement against a
    human-labeled test set is a separate, external requirement this bundle does not replace.
    """
    total_px = gray.size
    contamination_px = int((cv2.bitwise_and(final_mask, object_mask) > 0).sum())
    painted_px = int((final_mask > 0).sum())
    object_px = int((object_mask > 0).sum())

    edges = cv2.Canny(gray, 60, 150)
    edges_near = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    boundary = cv2.morphologyEx(
        final_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    boundary_px = int((boundary > 0).sum())
    boundary_on_edge = int(((boundary > 0) & (edges_near > 0)).sum())

    def region_stats(regions: list[dict]) -> dict:
        if not regions:
            return {"count": 0, "meanSolidity": None}
        solidities = [_largest_component_fraction(r["mask"]) for r in regions]
        return {"count": len(regions), "meanSolidity": round(float(np.mean(solidities)), 4)}

    return {
        "wallAreaRatio": round(sum((r["mask"] > 0).sum() for r in wall_regions) / total_px, 5),
        "floorAreaRatio": round(sum((r["mask"] > 0).sum() for r in floor_regions) / total_px, 5),
        "objectContaminationRatio": round(contamination_px / max(1, painted_px), 6),
        "objectContaminationPixels": contamination_px,
        "paintedPixels": painted_px,
        "objectPixels": object_px,
        "boundaryStructuralEdgeAgreement": round(boundary_on_edge / max(1, boundary_px), 4),
        "wallRegions": region_stats(wall_regions),
        "floorRegions": region_stats(floor_regions),
    }
