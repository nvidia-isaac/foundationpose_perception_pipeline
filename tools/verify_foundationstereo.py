#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test that the depth engine is built, loadable, and predicts sane disparity.

The third of the install checks, beside `verify_sam3.py` and `verify_foundationpose.py`, and it
covers the one component the other two do not: the TensorRT engine and the native CUDA Runtime bindings
underneath it. **Needs no dataset** -- which is the point, because an engine is built long before
any capture is on the machine, and "did the build work" should be answerable on its own.

It runs one inference on a **synthetic rectified stereo pair with a known disparity**: a
band-limited noise texture, and the same texture shifted. A shifted-texture pair is exactly what a
stereo matcher is trained to solve, so a working install recovers the shift to well under a pixel.
That makes this a numerical check rather than a "tensors came back" check.

What it exercises, all of which fail separately in practice:

- TensorRT and CUDA Runtime bindings are installed and can load the engine;
- TensorRT executes it and returns disparity of the expected shape and magnitude.

What it does NOT catch, stated rather than implied: an input-normalisation mistake. A clean,
high-contrast synthetic pair is recoverable under either convention, so this cannot separate them
-- normalisation only shows up on real imagery, where the wrong one roughly doubles depth error
without failing. `check_engine_depth_smoke.py --engine` is what covers it here: it reports the
convention the depth stage actually used (`normalization=imagenet`) on a real scene.

Usage:
    ./.venv/bin/python tools/verify_foundationstereo.py --config ${your_dataset}
    ./.venv/bin/python tools/verify_foundationstereo.py --config ${your_dataset} --engine /path/to.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "src"))

from foundationpose_perception_pipeline.config import add_config_argument, settings_from_argv  # noqa: E402

# Tolerance on the recovered shift. A working engine lands well inside a pixel; 2 px is
# deliberately loose, because a fixed-shape engine resamples the pair to its own input size and
# the shift is measured back at that scale. Anything approaching this bound is badly broken.
DISPARITY_TOLERANCE_PX = 2.0
# Fraction of measurable pixels that must produce finite, positive disparity.
MIN_VALID_FRACTION = 0.9
# Used only when the engine's profile is dynamic and there is no shape to copy.
DEFAULT_HEIGHT, DEFAULT_WIDTH = 480, 800


def make_synthetic_pair(height: int, width: int, shift_px: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectified `(left, right)` RGB pair whose true disparity is `shift_px`.

    Band-limited noise rather than white noise: a matcher needs structure at a scale its cost
    volume can see, and per-pixel noise is both unrealistic and ambiguous. Blurred random blocks
    give unambiguous, well-conditioned correspondences.

    Disparity convention: a point at column `x` on the left appears at `x - shift_px` on the
    right, so `right` is `left` shifted left. The rightmost `shift_px` columns have no counterpart
    and are excluded from the measurement.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.random((height // 8 + 1, width // 8 + 1, 3), dtype=np.float32)
    texture = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[:height, :width]

    kernel = 3
    padded = np.pad(texture, ((kernel, kernel), (kernel, kernel), (0, 0)), mode="edge")
    blurred = np.zeros_like(texture)
    for dy in range(-kernel, kernel + 1):
        for dx in range(-kernel, kernel + 1):
            blurred += padded[kernel + dy : kernel + dy + height, kernel + dx : kernel + dx + width]
    blurred /= (2 * kernel + 1) ** 2

    left = (blurred * 255.0).astype(np.uint8)
    right = np.empty_like(left)
    right[:, : width - shift_px] = left[:, shift_px:]
    right[:, width - shift_px :] = left[:, width - 1 : width]
    return left, right


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the engine verification."""
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument(
        "--engine",
        type=Path,
        default=settings.depth.engine,
        help="TensorRT engine to verify. Defaults to depth.engine from the config profile.",
    )
    parser.add_argument("--shift-px", type=int, default=24, help="Ground-truth disparity of the synthetic pair.")
    return parser.parse_args()


def main() -> None:
    """Load the engine, run one synthetic pair through it, and assert the disparity is right."""
    args = parse_args()
    if args.engine is None:
        raise SystemExit(
            "No engine to verify. Build one with tools/build_stereo_engine.py, then pass --engine "
            "or set depth.engine in the config profile."
        )
    engine_path = Path(args.engine).expanduser()
    if not engine_path.exists():
        raise SystemExit(
            f"Engine not found: {engine_path}. Engines are machine-specific and are not "
            "committed; build one with tools/build_stereo_engine.py."
        )

    # Load only after validating the requested path.
    from foundationpose_perception_pipeline.inference.stereo import load_engine, release_engines

    engine = load_engine(str(engine_path.resolve()))
    height, width = engine.fixed_hw or (DEFAULT_HEIGHT, DEFAULT_WIDTH)
    print(f"engine     : {engine_path.name}")
    print(f"input shape: {width}x{height} ({'fixed' if engine.fixed_hw else 'dynamic profile'})")

    left, right = make_synthetic_pair(height, width, args.shift_px)
    try:
        disparity = engine.infer_disparity(left, right)
    finally:
        # Free the TensorRT scratch before exiting, the same way the depth stage does; a verify
        # script that leaves 10 GB resident is a nuisance when it is run in a loop.
        release_engines()

    if disparity.shape[:2] != (height, width):
        raise SystemExit(f"FAILED: disparity shape {disparity.shape[:2]}, expected {(height, width)}")

    # Measure away from the border: the rightmost `shift_px` columns have no correspondence, and
    # the outermost rows/columns carry the usual edge effects.
    margin = max(args.shift_px, 16)
    interior = disparity[margin:-margin, margin : width - margin]
    finite = np.isfinite(interior) & (interior > 0)
    valid_fraction = float(finite.mean())
    median_disparity = float(np.median(interior[finite])) if finite.any() else float("nan")
    error_px = abs(median_disparity - args.shift_px)

    print(
        f"true disparity {args.shift_px} px -> predicted median {median_disparity:.2f} px "
        f"(error {error_px:.2f} px), valid {valid_fraction:.3f}"
    )

    failures: list[str] = []
    if valid_fraction < MIN_VALID_FRACTION:
        failures.append(f"valid fraction {valid_fraction:.3f} < {MIN_VALID_FRACTION}")
    if not (error_px <= DISPARITY_TOLERANCE_PX):
        failures.append(f"disparity error {error_px:.2f} px > {DISPARITY_TOLERANCE_PX} px")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

    print(
        "\nFoundationStereo OK: engine loaded, forward pass ran, recovered the synthetic "
        f"disparity to within {error_px:.2f} px."
    )


if __name__ == "__main__":
    main()
