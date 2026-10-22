#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-check the depth engine on one real scene.

**Why this exists as a separate check.** Nothing else runs the engine on real imagery:
`tools/verify_foundationstereo.py` uses a synthetic pair, and the checks that need no GPU stop
short of executing the model. A refactor of how depth is invoked can therefore break the only
path that ships while everything else stays green.

It is a smoke test, not an accuracy test: it asserts the engine ran, produced a depth map with a
plausible amount of valid data at a plausible distance, and used the input convention the model
actually wants. Depth *quality* needs ground truth and is a separate evaluation; this needs only a
dataset and an engine, so it stays cheap enough to run on every change.

The `normalization` assertion is the load-bearing one. A TAO deployable export fed raw 0-255
instead of ImageNet-normalised input produces a depth map that looks entirely reasonable and is
roughly twice as wrong, which no shape or range check would catch.

Usage:
    ./.venv/bin/python test/check_engine_depth_smoke.py --config ${your_dataset}
    ./.venv/bin/python test/check_engine_depth_smoke.py --config ${your_dataset} --engine /path/to.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR / "src"))

from foundationpose_perception_pipeline.config import (  # noqa: E402
    add_config_argument,
    settings_from_argv,
)

MIN_VALID_FRACTION = 0.30
MIN_MEDIAN_DEPTH_M = 0.2
MAX_MEDIAN_DEPTH_M = 3.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the engine depth smoke check."""
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    # Defaults to the profile's `regression.clean_dataset` when it sets one, and is otherwise
    # required: a hardcoded fallback would name a capture that only exists on one rig.
    parser.add_argument("--dataset", default=settings.dataset.regression.get("clean_dataset"))
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=settings.depth.foundation_stereo_max_width)
    parser.add_argument(
        "--engine",
        type=Path,
        default=settings.depth.engine,
        help="TensorRT engine to check. Selects the native TensorRT backend, which runs in THIS "
             "interpreter and this process -- no subprocess and no second environment. "
             "Defaults to depth.engine from the config profile.",
    )
    return parser.parse_args()


def check_engine(args) -> tuple[dict, np.ndarray]:
    """Run one scene through the native TensorRT backend, in this process.

    No subprocess and no second interpreter: that is the whole point of this backend, so the
    check exercises it the way the pipeline does rather than through a CLI.
    """
    from foundationpose_perception_pipeline.inference.stereo import load_engine, scene_depth

    scene_dir = args.dataset_root / args.dataset / "test" / f"{args.scene:06d}"
    if not scene_dir.exists():
        raise SystemExit(
            f"Missing scene directory: {scene_dir}. The dataset root came from the config "
            f"profile ({args.dataset_root}); pass --dataset-root to point somewhere else."
        )
    if not Path(args.engine).exists():
        raise SystemExit(
            f"Engine not found: {args.engine}. Build one with tools/build_stereo_engine.py -- "
            "engines are machine-specific and are not committed."
        )

    print(f"dataset    : {args.dataset}/{args.scene:06d}")
    print(f"model      : {Path(args.engine).name}")
    print("interpreter: this one (native TensorRT runs in-process)")

    result = scene_depth(
        scene_dir,
        engine=load_engine(str(Path(args.engine).expanduser().resolve())),
        max_width=args.max_width,
    )
    return result.metadata, result.depth_m


def main() -> None:
    """Run one scene through a commercial-model backend and assert the output is sane."""
    args = parse_args()
    if not args.dataset:
        raise SystemExit(
            "No dataset: pass --dataset, or set `dataset.regression.clean_dataset` in the config "
            "profile so this check has one to default to."
        )
    if args.engine is not None:
        metadata, depth = check_engine(args)
        report_depth_result(metadata, depth, "tensorrt", "imagenet")
        return
    raise SystemExit(
        "No engine. Pass --engine <path>.engine, or set depth.engine in the config profile."
    )


def report_depth_result(metadata: dict, depth: np.ndarray, expected_backend: str, expected_normalization: str) -> None:
    """Assert one depth result is plausible and was produced the way it was meant to be."""
    failures: list[str] = []
    if metadata.get("backend") != expected_backend:
        failures.append(
            f"backend is {metadata.get('backend')!r}, expected {expected_backend!r} -- "
            "the backend under test did not run"
        )
    if metadata.get("normalization") != expected_normalization:
        failures.append(
            f"normalization is {metadata.get('normalization')!r}, expected {expected_normalization!r} "
            "for a deployable export; the wrong convention roughly doubles depth error without failing"
        )
    valid = np.isfinite(depth) & (depth > 0)
    valid_fraction = float(valid.mean())
    median_depth = float(np.median(depth[valid])) if valid.any() else float("nan")
    if valid_fraction < MIN_VALID_FRACTION:
        failures.append(f"valid fraction {valid_fraction:.4f} < {MIN_VALID_FRACTION}")
    if not (MIN_MEDIAN_DEPTH_M <= median_depth <= MAX_MEDIAN_DEPTH_M):
        failures.append(f"median depth {median_depth:.4f} m outside [{MIN_MEDIAN_DEPTH_M}, {MAX_MEDIAN_DEPTH_M}]")

    print(
        f"backend={metadata.get('backend')}  normalization={metadata.get('normalization')}  "
        f"fixed_hw={metadata.get('model_fixed_hw')}  rectified={metadata.get('rectified_size_wh')}"
    )
    print(f"valid_fraction={valid_fraction:.4f}  median_depth={median_depth:.4f} m  shape={list(depth.shape)}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"\nPASSED: {expected_backend} depth backend ran and produced a plausible depth map.")


if __name__ == "__main__":
    main()
