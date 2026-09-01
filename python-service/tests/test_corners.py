"""Geometric corner-only checks. Run:

    ../venv/bin/python tests/test_corners.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vision  # noqa: E402

H, W = 240, 320
MASK = np.full((H, W), 255, np.uint8)
TOL = 12.0


def _mids(cuts: list[tuple[float, float]]) -> list[float]:
    return [0.5 * (a + b) for a, b in cuts]


def _assert_cuts(name: str, cuts: list[tuple[float, float]], expected: list[float]) -> None:
    mids = _mids(cuts)
    if len(cuts) != len(expected) or any(not any(abs(m - e) <= TOL for m in mids) for e in expected):
        raise AssertionError(f"{name}: got mids={mids} expected={expected}")
    print(f"PASS {name}: {['%.1f' % m for m in mids]}")


def main() -> int:
    gray_v = np.zeros((H, W), np.uint8)
    gray_v[:, :160] = 70
    gray_v[:, 160:] = 190
    depth_v = np.zeros((H, W), np.float32)
    depth_v[:, :160] = 1.0
    depth_v[:, 160:] = 3.0
    _assert_cuts("vertical corner", vision._detect_wall_corners(gray_v, MASK, depth=depth_v), [160])

    gray_h = np.zeros((H, W), np.uint8)
    gray_h[:120, :] = 70
    gray_h[120:, :] = 190
    depth_h = np.zeros((H, W), np.float32)
    depth_h[:120, :] = 1.0
    depth_h[120:, :] = 3.0
    _assert_cuts(
        "horizontal corner",
        vision._detect_horizontal_wall_corners(gray_h, MASK, depth=depth_h, max_corners=1),
        [120],
    )

    gray_f = np.full((H, W), 128, np.uint8)
    for y in range(H):
        gray_f[y, :] = np.clip(110 + y * 0.15, 0, 255)
    depth_f = np.full((H, W), 2.0, np.float32)
    _assert_cuts("vertical flat", vision._detect_wall_corners(gray_f, MASK, depth=depth_f), [])
    _assert_cuts(
        "horizontal flat",
        vision._detect_horizontal_wall_corners(gray_f, MASK, depth=depth_f, max_corners=1),
        [],
    )
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
