"""Regression check for the wall/floor detection pipeline — compares current output against the
committed baseline (`tests/baseline.json`) on every test image and reports PASS/FAIL per metric.

This is NOT a ground-truth accuracy check. It has no hand-labeled "correct" answer to compare
against — this project doesn't have one. What it verifies is narrower but still real: that a code
change didn't silently shift the pipeline's own output on photos already known to matter,
especially in the direction this project cares about most — object contamination going *up*.

Exit code is 0 only if every image passes every hard check. Run before considering any change to
`vision.py` (or its geometry/perspective consumers) done:

    ./venv/bin/python tests/check_regression.py

If a change is *meant* to move these numbers (a real, verified accuracy improvement), re-run
`capture_baseline.py` afterward to accept the new numbers as the baseline going forward.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_lib import IMAGE_NAMES, analyze_test_image, load_baseline  # noqa: E402

# Object contamination is the one metric this project has repeatedly treated as non-negotiable
# ("false-positive painting is worse than missing a few wall pixels") — any real increase is a
# hard failure, not a warning. A tiny epsilon absorbs float noise, nothing more.
CONTAMINATION_INCREASE_TOLERANCE = 0.0005

# Area-fraction changes below this relative threshold are treated as noise (minor mask-boundary
# jitter from guided filtering, not a meaningful behavior change).
AREA_WARN_RELATIVE = 0.10
AREA_FAIL_RELATIVE = 0.30

REGION_COUNT_WARN_DELTA = 1


def _relative_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return abs(new - old) / abs(old)


def check_image(name: str, baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    for surface in ("wall", "floor"):
        old_area = baseline[surface]["totalAreaFraction"]
        new_area = current[surface]["totalAreaFraction"]
        rel = _relative_change(old_area, new_area)
        if rel > AREA_FAIL_RELATIVE:
            failures.append(
                f"{surface} area changed {old_area:.4f} -> {new_area:.4f} ({rel*100:.0f}% relative, "
                f">{AREA_FAIL_RELATIVE*100:.0f}% threshold)"
            )
        elif rel > AREA_WARN_RELATIVE:
            warnings.append(f"{surface} area changed {old_area:.4f} -> {new_area:.4f} ({rel*100:.0f}% relative)")

        old_count = baseline[surface]["count"]
        new_count = current[surface]["count"]
        if abs(new_count - old_count) > REGION_COUNT_WARN_DELTA:
            warnings.append(f"{surface} region count changed {old_count} -> {new_count}")

    old_contam = baseline["objectContaminationRatio"]
    new_contam = current["objectContaminationRatio"]
    if new_contam - old_contam > CONTAMINATION_INCREASE_TOLERANCE:
        failures.append(
            f"object contamination INCREASED {old_contam:.5f} -> {new_contam:.5f} "
            f"(this is the highest-priority regression this project tracks)"
        )

    old_edge = baseline["boundaryStructuralEdgeAgreement"]
    new_edge = current["boundaryStructuralEdgeAgreement"]
    if old_edge and _relative_change(old_edge, new_edge) > AREA_WARN_RELATIVE:
        warnings.append(f"boundary/structural-edge agreement changed {old_edge:.3f} -> {new_edge:.3f}")

    return failures, warnings


def main() -> int:
    baseline = load_baseline()
    any_failures = False

    for name in IMAGE_NAMES:
        if name not in baseline:
            print(f"[SKIP] {name}: not in baseline (run capture_baseline.py to add it)")
            continue
        print(f"analyzing {name} ...")
        current = analyze_test_image(name)
        failures, warnings = check_image(name, baseline[name], current)

        if failures:
            any_failures = True
            print(f"[FAIL] {name}")
            for f in failures:
                print(f"    FAIL: {f}")
        else:
            print(f"[PASS] {name}")
        for w in warnings:
            print(f"    warn: {w}")

    print()
    if any_failures:
        print("REGRESSION DETECTED — see FAIL lines above.")
        return 1
    print("No regressions detected against baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
