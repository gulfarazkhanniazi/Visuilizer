"""Dump per-wall horizontal-corner attempts on every file in test-images/."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import vision  # noqa: E402

IMAGES = sorted(p.name for p in (ROOT.parent / "test-images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def main() -> None:
    orig_split = vision._split_region_horizontally
    orig_detect_h = vision._detect_horizontal_wall_corners
    orig_extract = vision.extract_wall_regions

    log: list[str] = []

    def detect_h(gray, mask, depth=None, max_corners=None):
        cuts = orig_detect_h(gray, mask, depth=depth, max_corners=max_corners)
        ys, xs = np.nonzero(mask)
        h = int(ys.max() - ys.min() + 1) if ys.size else 0
        w = int(xs.max() - xs.min() + 1) if xs.size else 0
        log.append(f"    detect_h bbox={w}x{h} max_corners={max_corners} cuts={[(round(a), round(b)) for a,b in cuts]}")
        return cuts

    def split_h(region, gray, depth, image_bgr):
        nq = len(region.get("quads") or [])
        ys, xs = np.nonzero(region["mask"])
        h = int(ys.max() - ys.min() + 1) if ys.size else 0
        w = int(xs.max() - xs.min() + 1) if xs.size else 0
        log.append(f"  split_h quads={nq} bbox={w}x{h}")
        out = orig_split(region, gray, depth, image_bgr)
        log.append(f"    -> {len(out)} region(s)")
        return out

    def extract(*a, **kw):
        regions = orig_extract(*a, **kw)
        log.append(f"  extract_wall_regions -> {len(regions)} walls")
        for i, r in enumerate(regions):
            ys, xs = np.nonzero(r["mask"])
            log.append(
                f"    wall[{i}] bbox=({int(xs.min())},{int(ys.min())})-({int(xs.max())},{int(ys.max())}) "
                f"quads={len(r['quads'])} area={(r['mask']>0).sum()}"
            )
        return regions

    vision._detect_horizontal_wall_corners = detect_h
    vision._split_region_horizontally = split_h
    vision.extract_wall_regions = extract

    img_dir = ROOT.parent / "test-images"
    try:
        for name in IMAGES:
            log.append(f"\n=== {name} ===")
            print(f"analyzing {name}...", flush=True)
            path = img_dir / name
            im = Image.open(path)
            w, h = im.size
            vision.analyze_image(path.read_bytes(), w, h, debug=False)
    finally:
        vision._detect_horizontal_wall_corners = orig_detect_h
        vision._split_region_horizontally = orig_split
        vision.extract_wall_regions = orig_extract

    print("\n".join(log))


if __name__ == "__main__":
    main()
