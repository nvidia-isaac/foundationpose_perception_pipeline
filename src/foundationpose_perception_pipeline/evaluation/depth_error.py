#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Depth-error scoring: predicted depth against the collected ground-truth map.

Separate from `inference/depth.py`, and the distinction is the whole point: *producing* depth
is inference and happens on every run, while *scoring* it needs a collected ground-truth map
that only an annotated capture has. A deployment does the first and never the second.

Read the object-pixel row, not the whole-image one. Whole-image error is dominated by the mat,
bin and floor -- large flat textured surfaces any stereo model matches easily -- and understates
error on the parts by several fold. FoundationPose only ever consumes depth inside the SAM3
mask, so the object row is what predicts pose quality.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from foundationpose_perception_pipeline.dataset import targets_for_scene_frame0
from foundationpose_perception_pipeline.evaluation.gt import (
    GroundTruthRenderer,
    render_gt_entries,
    render_gt_entries_cached,
)

# Reading a collected depth PNG is an IO concern, not a scoring one, so it lives in `io.bop`
# where any caller can reach it without importing anything from this package.
from foundationpose_perception_pipeline.io.bop import load_collected_depth_m  # noqa: F401


def resize_depth_nearest(depth_m: np.ndarray, target_size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a depth map with nearest-neighbour sampling.

    Nearest, not linear: interpolating across a depth discontinuity invents surfaces that lie
    between the foreground and the background, at depths where nothing exists.
    """
    target_w, target_h = target_size_wh
    if depth_m.shape == (target_h, target_w):
        return depth_m
    return cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def scene_object_mask(
    *,
    dataset_dir: Path,
    scene_id: int,
    image_size_wh: tuple[int, int],
    gt_renderer: GroundTruthRenderer,
    gt_cache_root: Path,
    use_gt_cache: bool,
    min_visible_fraction: float,
    split: str = "test",
) -> np.ndarray | None:
    """Union of the occlusion-aware GT masks of every object class in a scene.

    These are the pixels a part actually occupies, so restricting depth error to them removes
    the mat, the bin and the floor -- which cover most of the frame and are far easier to match
    than machined metal, and therefore flatter the whole-image number.

    Served from the GT z-buffer cache when present. Without it this rasterizes
    (~12-14 s per target) and the main loop rasterizes the same scene again shortly after, so
    expect roughly double the GT cost on an uncached run.
    """
    masks: list[np.ndarray] = []
    for target in targets_for_scene_frame0(dataset_dir, scene_id, split):
        if use_gt_cache:
            bundle = render_gt_entries_cached(
                gt_renderer, target, image_size_wh, gt_cache_root, min_visible_fraction
            )
        else:
            bundle = render_gt_entries(gt_renderer, target, image_size_wh, min_visible_fraction)
        masks.extend(bundle[1])
    if not masks:
        return None
    union = np.zeros((image_size_wh[1], image_size_wh[0]), dtype=bool)
    for mask in masks:
        union |= mask
    return union


def compare_depths(
    estimated_m: np.ndarray,
    collected_m: np.ndarray,
    object_mask: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    """Compare predicted depth against the collected GT depth, on pixels valid in both.

    Reported twice when `object_mask` is given. The whole-image figures come first and are kept
    because they catch gross regressions cheaply, but they are **dominated by the mat, bin and
    floor**: large, flat, well-textured surfaces that any stereo model matches easily. The gap is
    not subtle -- whole-image error has been observed several times lower than the error on the
    parts in the same scene, which is exactly the wrong way round for predicting pose quality.

    The `object_*` figures are the ones that predict pose quality, because FoundationPose only
    ever consumes depth inside the SAM3 mask. A depth-only sweep reports the same
    restriction with more detail.
    """
    if estimated_m.shape != collected_m.shape:
        estimated_m = resize_depth_nearest(estimated_m, (collected_m.shape[1], collected_m.shape[0]))
    estimated_valid = np.isfinite(estimated_m) & (estimated_m > 0.0)
    collected_valid = np.isfinite(collected_m) & (collected_m > 0.0)
    both = estimated_valid & collected_valid
    count = int(both.sum())
    if count == 0:
        return {
            "comparison_mode": "foundationstereo_vs_collected_gt",
            "valid_count": 0,
            "valid_fraction": 0.0,
            "estimated_valid_fraction": float(estimated_valid.mean()),
            "collected_valid_fraction": float(collected_valid.mean()),
            "mae_mm": None,
            "rmse_mm": None,
            "median_abs_mm": None,
            "sum_abs_mm": 0.0,
            "sum_sq_mm": 0.0,
            "object_valid_count": 0,
            "object_coverage": None,
            "object_mae_mm": None,
            "object_rmse_mm": None,
            "object_median_abs_mm": None,
        }
    error_mm = (estimated_m[both] - collected_m[both]) * 1000.0
    absolute = np.abs(error_mm)
    return {
        "comparison_mode": "foundationstereo_vs_collected_gt",
        "valid_count": count,
        "valid_fraction": float(both.mean()),
        "estimated_valid_fraction": float(estimated_valid.mean()),
        "collected_valid_fraction": float(collected_valid.mean()),
        "mae_mm": float(absolute.mean()),
        "rmse_mm": float(np.sqrt(np.mean(error_mm**2))),
        "median_abs_mm": float(np.median(absolute)),
        # Kept as sums so the dataset-level aggregate is a true pooled statistic rather than a
        # mean of per-scene means, which would weight sparse scenes equally with dense ones.
        "sum_abs_mm": float(absolute.sum()),
        "sum_sq_mm": float(np.sum(error_mm**2)),
        **object_depth_metrics(estimated_m, collected_m, both, object_mask),
    }


def object_depth_metrics(
    estimated_m: np.ndarray,
    collected_m: np.ndarray,
    comparable: np.ndarray,
    object_mask: np.ndarray | None,
) -> dict[str, float | int | None]:
    """Depth error restricted to pixels a GT object occupies."""
    if object_mask is None or object_mask.shape != comparable.shape:
        return {
            "object_valid_count": None,
            "object_coverage": None,
            "object_mae_mm": None,
            "object_rmse_mm": None,
            "object_median_abs_mm": None,
        }
    selected = comparable & object_mask
    count = int(selected.sum())
    object_pixels = int(object_mask.sum())
    if count == 0:
        return {
            "object_valid_count": 0,
            # Coverage is the share of object pixels that got a usable depth at all -- the
            # fraction FoundationPose actually receives.
            "object_coverage": 0.0 if object_pixels else None,
            "object_mae_mm": None,
            "object_rmse_mm": None,
            "object_median_abs_mm": None,
        }
    error_mm = (estimated_m[selected] - collected_m[selected]) * 1000.0
    absolute = np.abs(error_mm)
    return {
        "object_valid_count": count,
        "object_coverage": float(count / object_pixels) if object_pixels else None,
        "object_mae_mm": float(absolute.mean()),
        "object_rmse_mm": float(np.sqrt(np.mean(error_mm**2))),
        "object_median_abs_mm": float(np.median(absolute)),
    }
