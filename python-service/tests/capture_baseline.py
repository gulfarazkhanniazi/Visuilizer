"""Captures the current pipeline's output on every test image as the regression baseline.

Run this deliberately, on code you've already verified by hand (visually checked the overlays,
confirmed no known-bad pattern like the ceiling-bleed or the interleaved-region bug) — it does not
know what "correct" looks like, only what "current" looks like. Re-run it any time a change is
meant to *intentionally* move these numbers (e.g. a real accuracy fix); otherwise `check_regression.py`
will keep flagging the old baseline as violated forever.

Usage:
    ./venv/bin/python tests/capture_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_lib import BASELINE_PATH, IMAGE_NAMES, analyze_test_image  # noqa: E402


def main() -> None:
    baseline = {}
    for name in IMAGE_NAMES:
        print(f"analyzing {name} ...")
        baseline[name] = analyze_test_image(name)
        print(f"  {json.dumps(baseline[name])}")

    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"\nWrote baseline for {len(baseline)} images to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
