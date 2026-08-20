"""Room photo analysis: floor/wall semantic segmentation and mask cleanup. Runs the heavy
CV/ML work server-side (PyTorch + real OpenCV) instead of the browser, for both better
accuracy and better speed than in-browser WASM.
"""

from __future__ import annotations

import io
from functools import lru_cache

import cv2
import numpy as np
import torch # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingImports]
from PIL import Image
from transformers import pipeline # pyright: ignore[reportMissingImports]

# Largest SegFormer/ADE20K checkpoint — no browser download-size constraint here, so we use
# the most accurate variant rather than the b1 model the client-side version was limited to.
MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

# Regions smaller than this fraction of the image are dropped as noise. Kept low so genuinely
# smaller real surfaces (a door-surround wall strip, a small visible floor patch) still count —
# the earlier 1% threshold was silently discarding legitimate, if modest, wall/floor area.
MIN_AREA_FRACTION = 0.004
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
FLOOR_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
MAX_ANALYSIS_DIMENSION = 1600


@lru_cache(maxsize=1)
def get_segmenter():
    return pipeline("image-segmentation", model=MODEL_NAME)


def _segment(image: Image.Image, segmenter) -> tuple[list[dict], np.ndarray]:
    """Runs the model directly instead of going through the pipeline's own call, so we keep each
    pixel's full per-class probability, not just which single class narrowly won. A pixel where
    floor scores, say, 40% and some other class scores 45% is still very likely real floor — but
    read through hard argmax alone (all the pipeline exposes) it becomes "not floor", and worse,
    an object that then actively blocks growth from ever recovering it. Keeping the raw
    probabilities lets both the floor mask itself and what's allowed to block its growth stay
    lenient exactly where the model is genuinely unsure, while staying just as strict everywhere
    it isn't — this is what makes the per-pixel floor scan itself more accurate, not just the
    cleanup applied after it.
    """
    size = image.size # (w, h)
    inputs = segmenter.image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(segmenter.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = segmenter.model(**inputs).logits # (1, num_classes, h, w) at model resolution
    resized = F.interpolate(logits, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    probs = resized.softmax(dim=1)[0].cpu().numpy() # (num_classes, H, W)
    label_map = probs.argmax(axis=0)

    outputs = []
    for idx, label in segmenter.model.config.id2label.items():
        mask = (label_map == idx).astype(np.uint8) * 255
        if not mask.any():
            continue
        outputs.append({"label": label.strip(), "mask": mask})
    return outputs, probs


def _smooth_mask_contours(mask: np.ndarray, epsilon_frac: float = 0.004) -> np.ndarray:
    """Replaces a mask's boundary with a polygon-simplified version of itself.

    The model predicts at a fixed, much lower resolution than the photo. A real, straight wall
    edge or corner, represented in that coarse grid and then upsampled, comes out as a staircase
    — a few pixels over, a few pixels back, repeating along what should be one straight line.
    Blurring only softens each step in place; it doesn't straighten the staircase. Finding the
    mask's actual contour and simplifying it (Douglas-Peucker) collapses that staircase into
    proper line segments while still preserving genuine corners exactly where they are.
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
        # absolutely so it only ever erases genuine few-pixel upsampling noise, never a real,
        # larger-scale shape the boundary is actually supposed to follow — pixels are allowed to
        # go up and down where the real thing they're tracing does.
        perimeter = cv2.arcLength(contour, True)
        epsilon = min(max(1.5, perimeter * epsilon_frac), 5.0)
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


def _occupied_mask(
    outputs, exclude_labels: set[str], size: tuple[int, int], near_window: np.ndarray | None = None
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
    for entry in outputs:
        if entry["label"] in exclude_labels:
            continue
        mask = _raw_label_mask(entry, size)
        if entry["label"] in OUTDOOR_BLEED_THROUGH_LABELS:
            if near_window is None:
                continue
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(near_window))
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
    for contour in contours:
        area_frac = cv2.contourArea(contour) / img_area
        if 0.0015 < area_frac < 0.08:
            cv2.drawContours(objects, [contour], -1, 255, thickness=cv2.FILLED)
    return objects


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
    baseline = cv2.blur(baseline.reshape(1, -1).astype(np.float32), (1, 41)).flatten()

    margin = 10
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

    num_bands = max(1, min(6, round(height / max(1, full_width) * 4) + 2))
    band_ys = sorted({min_y + round(i * (height - 1) / num_bands) for i in range(num_bands + 1)})
    if len(band_ys) < 2:
        return None

    extents = []
    for y in band_ys:
        lo_bound = max(min_y, y - height // num_bands)
        hi_bound = min(max_y, y + height // num_bands)
        ext = extent_near(y, lo_bound, hi_bound)
        if ext is None:
            return None
        extents.append(ext)

    quads = []
    for i in range(len(band_ys) - 1):
        y0, y1 = band_ys[i], band_ys[i + 1]
        lo0, hi0 = extents[i]
        lo1, hi1 = extents[i + 1]
        quads.append([(lo0, y0), (hi0, y0), (hi1, y1), (lo1, y1)])
    return quads


def _region_from_member(member: np.ndarray, quads: list | None = None):
    """Packages a mask into a region. `quads` is one or more perspective quads used for the tile
    warp; if not given, one is fit by scanning the mask's own rows. Multiple quads let a single
    unified, selectable surface (e.g. a whole wall spanning a real corner) render each plane
    with its own correct perspective orientation while staying one mask, one selection, one
    design assignment — actually splitting the mask into separate selectable regions per plane
    is what caused the persistent gap/misalignment bugs; this only changes how the same mask is
    warped, never what's selected or clipped.
    """
    ys, xs = np.nonzero(member)
    if ys.size == 0:
        return None
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    if quads is None:
        quads = _quads_from_mask(member, bbox)
        if not quads:
            return None
    centroid = (float(xs.mean()), float(ys.mean()))
    return {
        "quads": quads,
        "centroid": centroid,
        "mask": member,
    }


def _detect_wall_corners(gray: np.ndarray, mask: np.ndarray, max_corners: int = 2) -> list[tuple[float, float]]:
    """Finds up to `max_corners` confident architectural corners within a *unified* wall mask —
    used only to choose additional perspective quads for correct-looking tile orientation on
    each plane, never to split the mask/selection itself. Kept deliberately strict (tall,
    near-vertical, well-centered lines only): a false positive here draws a visible fake crease
    in the tile pattern, whereas a missed real corner just falls back to one flat warp across it,
    a much smaller visual cost. Returns each corner as a (x_at_top, x_at_bottom) line in global
    image coordinates, following the real detected tilt rather than assuming it's vertical.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    min_y, min_x, max_y, max_x = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width < 80 or height < 80:
        return []

    region = gray[min_y : max_y + 1, min_x : max_x + 1].copy()
    region_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
    region[region_mask == 0] = 0

    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges[region_mask == 0] = 0

    min_line_length = height * 0.6
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_line_length, maxLineGap=10)
    if lines is None:
        return []

    margin = width * 0.08
    min_separation = width * 0.2
    candidates: list[tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = (dx**2 + dy**2) ** 0.5
        angle_from_vertical = np.degrees(np.arctan2(dx, max(dy, 1e-6)))
        x_mid = (x1 + x2) / 2
        if angle_from_vertical < 10 and length > min_line_length and margin < x_mid < width - margin:
            candidates.append((length, float(x1), float(y1), float(x2), float(y2)))

    candidates.sort(key=lambda c: -c[0])
    chosen_x: list[float] = []
    cuts: list[tuple[float, float]] = []
    for length, x1, y1, x2, y2 in candidates:
        x_mid = (x1 + x2) / 2
        if any(abs(x_mid - cx) < min_separation for cx in chosen_x):
            continue
        chosen_x.append(x_mid)
        if y2 == y1:
            continue
        slope = (x2 - x1) / (y2 - y1)
        x_top = x1 + (0 - y1) * slope
        x_bottom = x1 + ((height - 1) - y1) * slope
        cuts.append((min_x + x_top, min_x + x_bottom))
        if len(cuts) >= max_corners:
            break

    return sorted(cuts, key=lambda c: c[0] + c[1])


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


def extract_wall_regions(wall_mask: np.ndarray, occupied: np.ndarray, gray: np.ndarray) -> list[dict]:
    # Treated as one continuous, single *selectable* surface, the same way the floor is —
    # splitting the mask itself into separate selectable wall-plane sections was the source of
    # every gap/misalignment bug this pipeline had (adjacent sections' independently-fit quads
    # not lining up with each other's true tilted edge, false splits from decorative panelling,
    # etc). One unified, precisely-masked wall region sidesteps that whole class of problem.
    #
    # A previous version also let this growth cross into ceiling-classified pixels, meant to
    # bridge a sloped ceiling-transition's thin misclassified strip. In practice it bled the wall
    # design onto ordinary flat ceilings — a much more common and more visibly wrong case than
    # the narrow transition it was meant to fix. Removed; ceiling stays a hard boundary.
    mask = _exclude_occupied(wall_mask, occupied)
    mask = _close(mask, iterations=2)
    mask = _exclude_occupied(mask, occupied)
    mask = _grow_into_unclassified(mask, occupied, max_reach_px=240)
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
    mask = _exclude_occupied(mask, occupied)

    # The mask/selection stays unified above, but a single flat perspective quad across a real
    # corner would warp the tile pattern as if both wall faces were one flat plane, which looks
    # wrong right at the bend. Detect any real corners within this one mask and use one
    # perspective quad per plane instead — each quad's edges come from the exact same cut line
    # as its neighbor's, so they still meet with zero gap between them (same technique already
    # proven for the old split-region case, just applied to quads instead of separate masks).
    ys, xs = np.nonzero(mask)
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    top_y, bottom_y = float(bbox[0]), float(bbox[2])
    corners = _detect_wall_corners(gray, mask)

    left_edge = (float(bbox[1]), float(bbox[1]))
    right_edge = (float(bbox[3] + 1), float(bbox[3] + 1))
    cuts = [left_edge, *corners, right_edge]
    quads = [
        [
            (cuts[i][0], top_y),
            (cuts[i + 1][0], top_y),
            (cuts[i + 1][1], bottom_y),
            (cuts[i][1], bottom_y),
        ]
        for i in range(len(cuts) - 1)
    ]

    region = _region_from_member(mask, quads=quads)
    return [region] if region else []


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


def analyze_image(image_bytes: bytes, target_width: int, target_height: int) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # The model itself works at a fixed internal resolution regardless of input size, so running
    # the pixel-array post-processing (connected components, morphology, Hough) on a full 4000px+
    # phone photo just burns time for no extra accuracy. Cap it; results get scaled back up to
    # target_width/target_height (the true original dimensions) below.
    orig_w, orig_h = image.size
    scale = min(1.0, MAX_ANALYSIS_DIMENSION / max(orig_w, orig_h))
    if scale < 1.0:
        image = image.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    analysis_size = image.size  # (width, height) actually analyzed at

    segmenter = get_segmenter()
    outputs, probs = _segment(image, segmenter)
    floor_idx = next(idx for idx, lbl in segmenter.model.config.id2label.items() if lbl.strip() == "floor")
    floor_prob = probs[floor_idx]

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

    wall_mask = _mask_for_label(outputs, "wall", analysis_size)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # ADE20K has no class for "wall hanging" — a flat, wall-mounted textile or tapestry the model
    # just reads as part of the wall itself, so nothing that only excludes *other confidently
    # classified* labels can catch it. Texture stands in for a real class here: an interior patch
    # of unusually high local variance inside an otherwise near-uniform painted wall is almost
    # certainly a distinct object, not the wall. Wall-only, deliberately — the same signal on the
    # floor would just as happily flag a real rug, which floor is *supposed* to count as paintable.
    wall_objects = _detect_textured_objects(gray, wall_mask)
    wall_mask = _exclude_occupied(wall_mask, wall_objects)

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
    floor_occupied = _occupied_mask(outputs, {"floor", "rug"}, analysis_size, near_window)
    floor_is_tie = (floor_prob > FLOOR_TIE_THRESHOLD).astype(np.uint8) * 255
    floor_occupied = cv2.bitwise_and(floor_occupied, cv2.bitwise_not(floor_is_tie))

    # A dim, shadowed strip right at the base of the wall (behind a plant pot, under low
    # furniture) can get confidently misread as wall rather than floor — see
    # `_floor_baseline_unblock`. Stop letting those specific pixels block floor growth.
    baseline_unblock = _floor_baseline_unblock(floor_mask, wall_mask)
    floor_occupied = cv2.bitwise_and(floor_occupied, cv2.bitwise_not(baseline_unblock))

    wall_occupied = _occupied_mask(outputs, {"wall"}, analysis_size, near_window)
    wall_occupied = cv2.bitwise_or(wall_occupied, wall_objects)

    floor_regions = extract_floor_region(floor_mask, floor_occupied)
    wall_regions = extract_wall_regions(wall_mask, wall_occupied, gray)

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

    def encode(region, kind, index):
        mask_to_encode = region["mask"]
        if encode_size != analysis_size:
            mask_to_encode = cv2.resize(mask_to_encode, encode_size, interpolation=cv2.INTER_CUBIC)
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
        }

    return {
        "floor": [encode(r, "floor", i) for i, r in enumerate(floor_regions)],
        "wall": [encode(r, "wall", i) for i, r in enumerate(wall_regions)],
    }
