"""Shared code for capturing and checking a wall/floor detection regression baseline.

Why this exists: every fix made to `vision.py` this project has been verified by hand — run the
pipeline on the test photos, look at the numbers, look at the overlay images. That works, but
nothing stops the *next* change from silently undoing one of those fixes; the extent filter added
at one point did exactly that (broke two legitimate bedroom walls) and was only caught because it
happened to get checked by hand immediately afterward. This turns that manual check into something
that runs the same way every time and says PASS/FAIL instead of relying on someone remembering to
look.

This is a real regression check against the pipeline's own prior output, not a ground-truth
accuracy benchmark — it has no hand-labeled "correct" answer to compare against (this project has
none), so it can't tell you the masks are *right*. What it can tell you is whether a code change
made them meaningfully *different* on the exact photos already known to matter, including the two
difficult cases (fragmented/fragile regions, a known model failure) found during this project.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vision  # noqa: E402

TEST_IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "test-images"
BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"

# Real photos only — skips nothing currently, but keeps this list explicit so a new test image
# dropped into test-images/ doesn't silently join the baseline without a deliberate re-capture.
IMAGE_NAMES = [
    "images.jpeg",
    "istockphoto-2077892760-612x612.jpg",
    "put-together-a-perfect-guest-room-1976987-hero-223e3e8f697e4b13b62ad4fe898d492d.jpg",
]


def _region_summary(regions: list[dict], total_px: int) -> dict:
    areas = [int((r["mask"] > 0).sum()) for r in regions]
    return {
        "count": len(regions),
        "totalAreaFraction": round(sum(areas) / total_px, 5) if total_px else 0.0,
        "meanSolidity": round(float(np.mean([vision._largest_component_fraction(r["mask"]) for r in regions])), 4)
        if regions
        else None,
    }


def analyze_test_image(name: str) -> dict:
    """Runs the full production pipeline (unmodified) on one test image and returns a compact,
    JSON-serializable summary — not the full mask bitmaps, which would make the baseline file huge
    and would fail on any harmless sub-pixel resize difference; the summary statistics are exactly
    what `_compute_diagnostics` already treats as the meaningful signal.
    """
    path = TEST_IMAGES_DIR / name
    image_bytes = path.read_bytes()
    im = Image.open(io.BytesIO(image_bytes))
    w, h = im.size

    wall_regions: list[dict] = []
    floor_regions: list[dict] = []
    orig_extract_wall = vision.extract_wall_regions
    orig_extract_floor = vision.extract_floor_region

    def traced_wall(*a, **kw):
        regions = orig_extract_wall(*a, **kw)
        wall_regions.extend(regions)
        return regions

    def traced_floor(*a, **kw):
        regions = orig_extract_floor(*a, **kw)
        floor_regions.extend(regions)
        return regions

    vision.extract_wall_regions = traced_wall
    vision.extract_floor_region = traced_floor
    try:
        result = vision.analyze_image(image_bytes, w, h, debug=True)
    finally:
        vision.extract_wall_regions = orig_extract_wall
        vision.extract_floor_region = orig_extract_floor

    total_px = wall_regions[0]["mask"].size if wall_regions else (floor_regions[0]["mask"].size if floor_regions else 1)
    diagnostics = result["debug"]["diagnostics"]

    return {
        "wall": _region_summary(wall_regions, total_px),
        "floor": _region_summary(floor_regions, total_px),
        "objectContaminationRatio": diagnostics["objectContaminationRatio"],
        "boundaryStructuralEdgeAgreement": diagnostics["boundaryStructuralEdgeAgreement"],
    }


def capture_baseline() -> dict:
    baseline = {name: analyze_test_image(name) for name in IMAGE_NAMES}
    return baseline


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"No baseline at {BASELINE_PATH}. Run `python tests/capture_baseline.py` once against "
            "known-good code to create one before relying on `check_regression.py`."
        )
    return json.loads(BASELINE_PATH.read_text())
