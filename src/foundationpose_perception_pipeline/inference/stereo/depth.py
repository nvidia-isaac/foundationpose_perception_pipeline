#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric depth for one BOP scene, from a stereo pair, in the base camera's frame.

    scene_camera.json
        |
        +-- select_partner_camera        pick the base camera's partner (lowest rectification
        |                                rotation error, subject to parallax bounds)
        +-- rectify_pair                 -> rectified RGB pair + K_rect, R1, baseline
        +-- StereoEngine.infer_disparity -> disparity in the rectified frame
        +-- disparity_to_depth_m         Z = f * B / d
        +-- depth_rectified_to_base      warp back to the base camera

The result is aligned with `rgb/<base>.png`, so it lines up with what SAM3 segments and can be
handed straight to FoundationPose.

This is the commercial path: a TensorRT engine, in the pipeline's own environment, called as a
function. See `stereo/rectify.py` for the invariants the geometry has to hold to.

Runnable as a module for one-scene debugging, and as an escape hatch if the in-process CUDA
context ever needs isolating again -- see `__main__.py`:

    ./.venv/bin/python -m foundationpose_perception_pipeline.inference.stereo \\
        --scene-dir <dataset_root>/<dataset>/test/000000 --out-dir /tmp/fs --engine <path>.engine
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from foundationpose_perception_pipeline.inference.stereo.rectify import rectify_pair, select_partner_camera

if TYPE_CHECKING:
    # Type-only. Importing this for real pulls in `tensorrt` and `cuda.bindings`, and the
    # geometry in this module (fit_to_model, disparity_to_depth_m, the rectification plumbing)
    # has to stay importable on a machine with no CUDA. `main()` imports the engine loader
    # where it is actually needed.
    from foundationpose_perception_pipeline.inference.stereo.trt_processor import (
        FoundationStereoTrtProcessor as StereoEngine,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedPair:
    """A rectified pair prepared for a static engine, and how to undo that preparation."""

    left: np.ndarray
    right: np.ndarray
    width_scale: float
    """Factor the resulting disparity must be *divided* by to return to rectified resolution."""

    content_height: int
    """Rows of the engine's input that hold image rather than padding; crop disparity to this."""


def fit_to_model(engine: StereoEngine, left_rgb: np.ndarray, right_rgb: np.ndarray) -> FittedPair:
    """Prepare a rectified pair for a static engine: scale by width, then pad the height.

    **Scale and pad, not stretch.** The obvious implementation resizes straight to the engine's
    HxW, and that is wrong in a way that does not announce itself: the rectified pair's aspect
    ratio is whatever `stereoRectify` produced, the engine's is whatever it was built for, and
    forcing one into the other stretches the image vertically. Disparity survives a *uniform*
    scale and survives a vertical stretch in principle -- it is a horizontal quantity -- but the
    model does not: it sees objects at the wrong aspect and the cost volume it matches on is not
    the one it was trained for. Measured against the same weights, a 6% stretch moved the p99
    depth difference into metres while the median stayed under a millimetre, i.e. it degrades
    exactly the edges that matter and nowhere a summary statistic would notice.

    Scaling by width and replicate-padding the remaining rows keeps the aspect exact and matches
    what the geometry expects when it pads a rectified pair to a multiple of 32 -- which is what
    keeps a comparison between two runtimes from becoming a comparison between two preprocessing
    schemes.

    Note the direction of `width_scale`: TAO's own evaluator rescales the ground truth *up* to
    engine resolution instead (`stereo_evaluator.py`). Ours is the direction that keeps the depth
    map on the base camera's grid, which is what FoundationPose consumes.
    """
    if engine.fixed_hw is None:
        return FittedPair(left_rgb, right_rgb, 1.0, left_rgb.shape[0])

    model_h, model_w = engine.fixed_hw
    src_h, src_w = left_rgb.shape[:2]
    if (src_h, src_w) == (model_h, model_w):
        return FittedPair(left_rgb, right_rgb, 1.0, model_h)

    width_scale = model_w / src_w
    scaled_h = round(src_h * width_scale)
    interpolation = cv2.INTER_AREA if width_scale < 1.0 else cv2.INTER_LINEAR
    left = cv2.resize(left_rgb, (model_w, scaled_h), interpolation=interpolation)
    right = cv2.resize(right_rgb, (model_w, scaled_h), interpolation=interpolation)

    if scaled_h == model_h:
        return FittedPair(left, right, width_scale, model_h)
    if scaled_h < model_h:
        # Pad the bottom only: the origin stays put, so the rectified intrinsics still describe
        # the image. Replicate rather than a constant, matching the padding in `infer_disparity`.
        pad = model_h - scaled_h
        left = cv2.copyMakeBorder(left, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        right = cv2.copyMakeBorder(right, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        return FittedPair(left, right, width_scale, scaled_h)

    # Taller than the engine even after scaling by width: the engine's aspect ratio is wrong for
    # this rig and no amount of padding helps. Crop rather than stretch -- a crop loses the
    # bottom of the frame, which is visible and local, where a stretch corrupts the whole image
    # in a way nothing downstream can detect.
    logger.warning(
        "rectified pair scales to %dx%d, taller than the engine's %dx%d input: cropping %d rows. "
        "Build an engine whose aspect ratio matches this dataset (tools/build_stereo_engine.py "
        "--shape-from-scene) to use the full frame.",
        model_w,
        scaled_h,
        model_w,
        model_h,
        scaled_h - model_h,
    )
    return FittedPair(left[:model_h], right[:model_h], width_scale, model_h)


def disparity_to_depth_m(
    disparity_px: np.ndarray, focal_px: float, baseline_m: float, min_disparity: float = 1e-3
) -> np.ndarray:
    """Convert rectified disparity to metric depth: Z = f * B / d."""
    depth = np.full(disparity_px.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity_px) & (disparity_px > min_disparity)
    depth[valid] = (focal_px * baseline_m / disparity_px[valid]).astype(np.float32)
    return depth

MM_PER_M = 1000.0

# Cap on the rectified width fed to a dynamic-shape engine, to bound GPU memory. This does not
# change the output resolution, which is always the full base-camera size. Feeding the full
# 3860x2178 pair to a dynamic model is an out-of-memory crash on a 32 GB card.
DEFAULT_MAX_WIDTH = 800

# FoundationStereo's disparity search range, in the model's own input pixels: the architecture
# builds its cost volumes at `max_disp//4` with `max_disp` 416, and the TAO `deployable_*`
# exports inherit it. A surface whose disparity exceeds this cannot be matched, and the returned
# value CLAMPS rather than failing.
MAX_MODEL_DISPARITY = 416.0
# Fraction of the range above which a result counts as saturated. Below 416 but close to it the
# matcher is already choosing from a truncated hypothesis set, so this fires early.
DISPARITY_SATURATION_FRACTION = 0.985


class StereoDepthError(RuntimeError):
    """Raised when a scene cannot produce depth -- no usable pair, missing files, bad arguments.

    A real exception rather than `SystemExit`: this path is a library call, and its caller is the
    pipeline rather than a shell.
    """


@dataclass(frozen=True)
class SceneDepth:
    """Depth for one scene, in the base camera's own image grid."""

    depth_m: np.ndarray
    """(H, W) float32 metric depth, NaN where invalid."""

    disparity_px: np.ndarray
    """Disparity in the rectified frame, at rectified resolution."""

    depth_rectified_m: np.ndarray
    """Metric depth in the rectified frame, kept for inspection."""

    camera_matrix: np.ndarray
    """The base camera's true K -- NOT the rectified one. See `rectify.py`."""

    metadata: dict[str, Any] = field(default_factory=dict)


def load_cameras(scene_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load per-camera intrinsics and world-to-camera extrinsics, translations in metres."""
    camera_file = scene_dir / "scene_camera.json"
    if not camera_file.exists():
        raise StereoDepthError(f"No scene_camera.json in {scene_dir}")
    data = json.loads(camera_file.read_text(encoding="utf-8"))
    keys = sorted(data, key=int)
    intrinsics = np.stack([np.asarray(data[k]["cam_K"], dtype=np.float64).reshape(3, 3) for k in keys])
    extrinsics = np.stack(
        [
            np.hstack(
                [
                    np.asarray(data[k]["cam_R_w2c"], dtype=np.float64).reshape(3, 3),
                    np.asarray(data[k]["cam_t_w2c"], dtype=np.float64).reshape(3, 1) / MM_PER_M,
                ]
            )
            for k in keys
        ]
    )
    return intrinsics, extrinsics, keys


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise StereoDepthError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def depth_rectified_to_base(
    depth_rect_m: np.ndarray,
    k_rect: np.ndarray,
    r1: np.ndarray,
    k_base: np.ndarray,
    base_shape: tuple[int, int],
) -> np.ndarray:
    """Resample rectified depth into the BASE camera's image grid.

    Valid only for the left camera of the rectified pair, because rectification is a pure
    rotation about that camera's optical centre -- so a ray-rotation warp is exact. Warping to any
    *other* camera would need a full SE(3) transform (the rig cameras sit 0.3-0.4 m apart), which
    a rotation-only warp cannot express.

    For each base-camera pixel: build its ray, rotate into the rectified frame, project, sample
    the rectified depth, recover the 3D point, rotate back, and take its z.
    """
    height, width = base_shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    rays_base = np.stack(
        [(xx - k_base[0, 2]) / k_base[0, 0], (yy - k_base[1, 2]) / k_base[1, 1], np.ones_like(xx)], axis=-1
    )
    rays_rect = rays_base @ np.asarray(r1, dtype=np.float64).T

    with np.errstate(divide="ignore", invalid="ignore"):
        u = k_rect[0, 0] * rays_rect[..., 0] / rays_rect[..., 2] + k_rect[0, 2]
        v = k_rect[1, 1] * rays_rect[..., 1] / rays_rect[..., 2] + k_rect[1, 2]
    behind = rays_rect[..., 2] <= 0

    sampled_z = cv2.remap(
        depth_rect_m,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).astype(np.float64)

    # Rectified depth is z along the rectified axis. Recover the full 3D point there, rotate it
    # back into the base camera frame, and read off its z.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = sampled_z / rays_rect[..., 2]
    points_rect = rays_rect * scale[..., None]
    points_base = points_rect @ np.asarray(r1, dtype=np.float64)  # R1^T applied on the right
    depth_base = points_base[..., 2]
    depth_base[behind] = np.nan
    return depth_base.astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int = 8, detail_boost: float = 0.0) -> np.ndarray:
    """Enhance contrast on an RGB image's luminance, optionally boosting the detail layer.

    These captures are grayscale replicated across three channels (measured per-pixel chroma
    exactly 0.00) with mean intensity around 60/255. Machined metal on a dark table gives them
    very high dynamic range: cut faces approach 255 while cylinder bodies sit at luma 30-60,
    overlapping the table's own distribution, so the matcher has almost no contrast to work with
    on exactly the surfaces that matter. The failure is underexposure, not blown highlights.

    Runs on the L channel of LAB, leaving chroma alone: shifting hue would alter the matching
    signal for a reason unrelated to contrast.

    `detail_boost > 0` adds an edge-preserving base/detail split (bilateral) and amplifies the
    detail layer, which sharpens scratch-scale texture without touching the base layer that
    carries left/right photometric agreement. Measured, it beat plain CLAHE on both datasets.

    **Both images of a pair must get identical treatment**, which is why this takes one image and
    the caller applies it twice with the same parameters. Stereo matching is a photometric-
    consistency problem and CLAHE is only approximately consistent -- its tile grid is fixed in
    image coordinates, so a point at x on the left and x-d on the right can land in tiles with
    different mappings. Measured, the contrast gain outweighs that; transforms that rewrite local
    mean and variance outright do not -- local contrast normalisation and global shadow lift were
    both tried and both made depth error worse.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if detail_boost:
        base = cv2.bilateralFilter(out, 9, 40, 9).astype(np.float32)
        out = np.clip(base + detail_boost * (out.astype(np.float32) - base), 0, 255).astype(np.uint8)
    return out


def disparity_pre_shift(
    focal_px_model: float,
    baseline_m: float,
    min_working_distance_m: float | None,
    max_working_distance_m: float | None,
    max_disparity_px: float = MAX_MODEL_DISPARITY,
    headroom: float = 0.95,
) -> int:
    """Columns to slide the right image by so the scene's disparity fits the model's search range.

    The model searches `[0, max_disparity_px)`. A scene occupying `[Z_min, Z_max]` needs
    `[f*B/Z_max, f*B/Z_min]`, whose *width* is usually far smaller than its upper end. Translating
    the right image right by `d0` subtracts `d0` from every disparity, sliding that window down
    into range without losing any of it; adding `d0` back afterwards recovers the true value.
    This is the coarse-to-fine trick FoundationStereo's own `run_hierachical` uses via `init_disp`.

    **Returns 0 when the scene already fits.** That matters: an unconditional shift measurably
    hurt the datasets that were already good (2.55 -> 2.74 mm object MAE there), because it moves
    the window for scenes that never needed it.

    Also returns 0 -- with a warning -- when the *span* itself exceeds the range, since no
    translation can cover the whole scene then; the fix there is a shorter baseline or tighter
    bounds.

    Needs no ground truth: `focal_px_model` and `baseline_m` come from `scene_camera.json`, and
    the working volume is a property of the rig. Set the bounds to the actual bin, tightly --
    their *difference* has to fit the range, so padding them costs the correction.
    """
    if not min_working_distance_m or not max_working_distance_m:
        return 0
    if min_working_distance_m <= 0 or max_working_distance_m <= 0:
        return 0

    near_px = focal_px_model * baseline_m / max(min_working_distance_m, 1e-6)
    far_px = focal_px_model * baseline_m / max(max_working_distance_m, 1e-6)
    usable = max_disparity_px * headroom
    if near_px <= usable:
        return 0
    if (near_px - far_px) > usable:
        logger.warning(
            "this pair needs %.0f px of disparity span for [%.2f, %.2f] m but the model's range "
            "is %.0f px, so no translation can cover the whole scene; narrow the working-distance "
            "bounds or use a shorter-baseline partner",
            near_px - far_px, min_working_distance_m, max_working_distance_m, max_disparity_px,
        )
        return 0
    # Push the near end just inside the range, but never so far that the far end goes negative.
    return int(max(min(near_px - usable, far_px * 0.9), 0.0))


def saturated_fraction(disparity_model_px: np.ndarray, max_disparity_px: float = MAX_MODEL_DISPARITY) -> float:
    """Fraction of valid disparity sitting at or above the model's search range.

    A disparity that reaches the range was chosen from a truncated set of hypotheses, so it is a
    lower bound rather than a match, and `Z = f*B/d` turns that into depth biased too far. It
    looks entirely plausible in the depth map, which is why it has to be reported explicitly
    rather than left to be noticed downstream.

    The threshold is slightly below the range: close to it the matcher is already choosing from a
    truncated hypothesis set, so the signal fires before the hard limit.
    """
    limit = max_disparity_px * DISPARITY_SATURATION_FRACTION
    finite = np.isfinite(disparity_model_px)
    if not finite.any():
        return 0.0
    return float((disparity_model_px[finite] >= limit).mean())


def scene_depth(
    scene_dir: Path | str,
    *,
    engine: StereoEngine,
    base_camera: int = 0,
    partner_camera: int | None = None,
    max_width: int | None = DEFAULT_MAX_WIDTH,
    alpha: float = -1.0,
    min_working_distance_m: float | None = None,
    max_working_distance_m: float | None = None,
    clahe_clip_limit: float | None = None,
    clahe_detail_boost: float = 0.0,
) -> SceneDepth:
    """Produce metric depth for one scene, aligned with the base camera's RGB image.

    `alpha` is `cv2.stereoRectify`'s free-scaling parameter. -1 keeps the full field of view; do
    NOT pass 0 -- with the ~15 degree rectification rotations in this rig it crops each view to a
    different valid region and the pair stops being stereo (measured median epipolar residual
    159 px at alpha=0 against 0.63 px at alpha=-1).
    """
    scene_dir = Path(scene_dir).expanduser().resolve()
    intrinsics, extrinsics, _ = load_cameras(scene_dir)

    sample = read_rgb(scene_dir / "rgb" / f"{base_camera:06d}.png")
    height, width = sample.shape[:2]

    # The base camera is always the reference image of the pair; whether it is physically the
    # left or the right camera is handled by mirroring below, so only the partner is chosen.
    if partner_camera is not None:
        partner_id, pair_source, rotation_error = int(partner_camera), "manual", None
    else:
        selection = select_partner_camera(
            base_camera=int(base_camera),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_width=width,
            image_height=height,
            # Feasibility-aware ranking: prefer a partner whose baseline keeps the nearest
            # expected surface inside the model's search range. `engine.fixed_hw` is the width the
            # model actually sees; for a dynamic engine it is the max_width cap.
            min_working_distance_m=min_working_distance_m,
            model_width=(engine.fixed_hw[1] if engine.fixed_hw else max_width),
            max_disparity_px=MAX_MODEL_DISPARITY,
        )
        if selection is None:
            raise StereoDepthError(
                f"No camera forms a horizontal stereo pair with camera {base_camera} in "
                f"{scene_dir}. Pass partner_camera to force one, or relax the parallax thresholds "
                "in stereo/rectify.py."
            )
        partner_id, _, rotation_error = selection
        pair_source = "select_partner_camera"

    left_id, right_id = int(base_camera), partner_id
    logger.info(
        "stereo pair: base=%d partner=%d (%s%s)",
        left_id,
        right_id,
        pair_source,
        f", rectification rotation {rotation_error:.1f} deg" if rotation_error is not None else "",
    )

    left_rgb = read_rgb(scene_dir / "rgb" / f"{left_id:06d}.png")
    right_rgb = read_rgb(scene_dir / "rgb" / f"{right_id:06d}.png")

    # The base camera's TRUE intrinsics, kept aside from anything rectification produces.
    # `depth_rectified_to_base` builds its rays from these, and `metadata.json` publishes them as
    # the camera matrix that goes with `depth_m.npy`. Substituting the rectified principal point
    # here translates the whole depth map by their difference -- worth 0.80 mm -> 11.64 mm of
    # object-pixel median error on its own.
    k_base = np.asarray(intrinsics[left_id], dtype=np.float64).copy()

    rect = rectify_pair(
        left_rgb,
        right_rgb,
        intrinsics[left_id],
        intrinsics[right_id],
        extrinsics[left_id],
        extrinsics[right_id],
        alpha=alpha,
    )
    rect_left, rect_right, k_rect = rect.left, rect.right, rect.k_rect

    # `max_width` exists to cap GPU memory, which only a dynamic-shape engine can exceed: it
    # consumes whatever resolution it is handed. A static-profile engine cannot -- `fit_to_model`
    # resizes to its declared input size regardless -- so downscaling first would only destroy
    # detail that the mandatory resize then interpolates back up. Measured with a fixed-shape
    # export, feeding full resolution beat feeding a downscaled pair from the same weights, and
    # peak memory was identical either way -- so there is nothing to trade.
    if engine.fixed_hw is not None:
        if max_width is not None and rect_left.shape[1] > max_width:
            logger.info(
                "ignoring max_width=%d: %s has a fixed %dx%d input, so the pair is resized to it "
                "directly from full resolution",
                max_width,
                engine.engine_path.name,
                engine.fixed_hw[1],
                engine.fixed_hw[0],
            )
    elif max_width is not None and rect_left.shape[1] > max_width:
        scale = max_width / rect_left.shape[1]
        new_size = (max_width, round(rect_left.shape[0] * scale))
        rect_left = cv2.resize(rect_left, new_size, interpolation=cv2.INTER_AREA)
        rect_right = cv2.resize(rect_right, new_size, interpolation=cv2.INTER_AREA)
        k_rect = k_rect.copy()
        k_rect[0, :] *= scale
        k_rect[1, :] *= scale
        logger.info("downscaled rectified pair to %dx%d", new_size[0], new_size[1])

    # FoundationStereo returns non-negative disparity measured on its LEFT input, so the base
    # camera has to *be* the left camera of the rectified pair. It is not always: the capture
    # rig's camera ids are re-assigned per scene, so the rectified baseline flips sign from scene
    # to scene.
    #
    # Feeding a right-camera image as "left" asks for negative disparity; the model returns
    # near-zero instead, which Z = f*B/d turns into metres of error rather than a visible failure
    # (measured 6-8 m, with the depth map still looking plausible). Mirroring both rectified
    # images swaps the handedness so the base camera becomes the left one, and mirroring the
    # disparity back lands it on the base camera's own rectified grid with the sign the depth
    # conversion expects. The intrinsics need no change because the flip is undone.
    baseline_m_signed = abs(float(rect.baseline_x_signed_m))
    mirrored = bool(rect.baseline_x_signed_m > 0)
    if mirrored:
        rect_left = np.ascontiguousarray(rect_left[:, ::-1])
        rect_right = np.ascontiguousarray(rect_right[:, ::-1])
        logger.info("camera %d is the right camera of this pair; mirroring for inference", left_id)

    fitted = fit_to_model(engine, rect_left, rect_right)

    # CLAHE after rectification, not before: the remap would otherwise smear the equalised result,
    # and both images must be treated identically for the correspondence to survive.
    if clahe_clip_limit:
        fitted = replace(
            fitted,
            left=apply_clahe(fitted.left, clahe_clip_limit, detail_boost=clahe_detail_boost),
            right=apply_clahe(fitted.right, clahe_clip_limit, detail_boost=clahe_detail_boost),
        )

    # The shift is in MODEL space, so the focal length has to be too: `fit_to_model` has already
    # scaled the pair by `width_scale`, and disparity scales with width.
    focal_px_model = float(k_rect[0, 0]) * fitted.width_scale
    pre_shift = disparity_pre_shift(
        focal_px_model, baseline_m_signed, min_working_distance_m, max_working_distance_m
    )
    if pre_shift:
        logger.info(
            "disparity pre-shift %d px (working volume [%.2f, %.2f] m, range %.0f px)",
            pre_shift, min_working_distance_m, max_working_distance_m, MAX_MODEL_DISPARITY,
        )
    disparity = engine.infer_disparity(fitted.left, fitted.right, pre_shift_px=pre_shift)
    # Key off the SHAPE, not `width_scale != 1.0`. The two are not the same test: an engine whose
    # width equals the rectified width gets width_scale == 1.0 while `fit_to_model` still pads the
    # height (451 -> 480 for the 480x800 engine at --max-width 800, which is this rig's default
    # configuration). Keying off the scale alone left those replicated rows attached to the
    # disparity, so every array derived from `rect_left` was one shape and the disparity another.
    if fitted.width_scale != 1.0 or disparity.shape[:2] != rect_left.shape[:2]:
        # Crop the padding off before rescaling: those rows are replicated pixels, and
        # interpolating them back up would smear invented content into the real image.
        disparity = disparity[: fitted.content_height]
        disparity = (
            cv2.resize(disparity, (rect_left.shape[1], rect_left.shape[0]), interpolation=cv2.INTER_LINEAR)
            / fitted.width_scale
        )
    if mirrored:
        disparity = np.ascontiguousarray(disparity[:, ::-1])
        rect_left = np.ascontiguousarray(rect_left[:, ::-1])
        rect_right = np.ascontiguousarray(rect_right[:, ::-1])

    # Reported before the depth conversion hides it: a clamped disparity produces depth that is
    # biased too far and looks entirely plausible. Measured at model scale, with the pre-shift
    # removed, because that is the space the search range lives in.
    saturation = saturated_fraction(
        disparity * fitted.width_scale - pre_shift
    )
    if saturation > 0.001:
        logger.warning(
            "%.2f%% of valid disparity is at or above %.0f px, the model's search range. Those "
            "pixels are clamped, not matched, and their depth is biased too far. Widen the "
            "working-volume bounds or force a shorter-baseline partner.",
            100 * saturation, MAX_MODEL_DISPARITY * DISPARITY_SATURATION_FRACTION,
        )

    baseline_m = float(abs(rect.baseline_x_signed_m))
    depth_rect_m = disparity_to_depth_m(disparity, focal_px=float(k_rect[0, 0]), baseline_m=baseline_m)
    depth_base_m = depth_rectified_to_base(depth_rect_m, k_rect, rect.r1, k_base, (height, width))

    finite = np.isfinite(depth_base_m)
    metadata: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "base_camera": int(base_camera),
        "left_camera": left_id,
        "right_camera": right_id,
        "pair_source": pair_source,
        "rectification_rotation_deg": rotation_error,
        "model": str(engine.engine_path),
        "providers": ["TensorRT"],
        "model_fixed_hw": list(engine.fixed_hw) if engine.fixed_hw else None,
        "backend": "tensorrt",
        "normalization": "imagenet",
        "rectified_size_wh": [int(rect_left.shape[1]), int(rect_left.shape[0])],
        "baseline_m": baseline_m,
        "mirrored_for_inference": mirrored,
        "disparity_pre_shift_px": pre_shift,
        "disparity_saturated_fraction": saturation,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_detail_boost": clahe_detail_boost,
        "focal_px_rect": float(k_rect[0, 0]),
        "camera_matrix": k_base.tolist(),
        "camera_matrix_rectified": np.asarray(k_rect).tolist(),
        "depth_m_valid_fraction": float(finite.mean()),
        "depth_m_valid_min": float(np.nanmin(depth_base_m)) if finite.any() else None,
        "depth_m_valid_max": float(np.nanmax(depth_base_m)) if finite.any() else None,
        "depth_m_valid_median": float(np.nanmedian(depth_base_m)) if finite.any() else None,
    }
    logger.info(
        "valid_fraction=%.4f  median_depth=%s",
        metadata["depth_m_valid_fraction"],
        metadata["depth_m_valid_median"],
    )
    return SceneDepth(
        depth_m=depth_base_m,
        disparity_px=disparity,
        depth_rectified_m=depth_rect_m,
        camera_matrix=k_base,
        metadata=metadata,
    )


def colorize(values: np.ndarray) -> np.ndarray:
    """Colourise a float map for quick visual inspection, NaNs rendered black."""
    valid = np.isfinite(values)
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(values[valid], [2, 98])
    norm = np.zeros_like(values, dtype=np.float32)
    norm[valid] = np.clip((values[valid] - lo) / max(hi - lo, 1e-9), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def write_scene_depth(result: SceneDepth, out_dir: Path, save_vis: bool = False) -> Path:
    """Write the on-disk artifacts, in the layout the rest of the pipeline already expects.

    The file names and the `metadata.json` schema are a contract with
    `inference/depth.py` and with every consumer of a cached depth directory -- so a cached path
    writes exactly the same set. Returns the path to `depth_m.npy`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_path = out_dir / "depth_m.npy"
    np.save(depth_path, result.depth_m)
    np.save(out_dir / "depth_rectified_m.npy", result.depth_rectified_m)
    np.save(out_dir / "disparity_px.npy", result.disparity_px)
    if save_vis:
        cv2.imwrite(str(out_dir / "depth_m_vis.png"), colorize(result.depth_m))
        cv2.imwrite(str(out_dir / "disparity_vis.png"), colorize(result.disparity_px))
    # Written last: `generate_scene_depth` treats the presence of both depth_m.npy and
    # metadata.json as "this scene is done", so a crash mid-write must not look complete.
    (out_dir / "metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    return depth_path


def main(argv: list[str] | None = None) -> None:
    """One-scene CLI, for debugging and for running this path in its own process."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=None,
                        help="ONNX or TensorRT plan; defaults to the stereo model under MODELS_DIR.")
    parser.add_argument("--base-camera", type=int, default=0)
    parser.add_argument("--partner-camera", type=int, default=None, help="Override pair selection.")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    parser.add_argument("--fixed-height", type=int, default=None)
    parser.add_argument("--min-working-distance", type=float, default=None)
    parser.add_argument("--max-working-distance", type=float, default=None)
    parser.add_argument("--clahe-clip-limit", type=float, default=None)
    parser.add_argument("--clahe-detail-boost", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Imported here, not at module scope: this is the only entry point in this module that
    # needs a GPU, and the geometry above must stay importable without one.
    from foundationpose_perception_pipeline.inference.stereo.trt_processor import load_engine

    result = scene_depth(
        args.scene_dir,
        engine=load_engine(args.engine, max_width=args.max_width, fixed_height=args.fixed_height),
        base_camera=args.base_camera,
        partner_camera=args.partner_camera,
        max_width=args.max_width,
        min_working_distance_m=args.min_working_distance,
        max_working_distance_m=args.max_working_distance,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_detail_boost=args.clahe_detail_boost,
    )
    path = write_scene_depth(result, args.out_dir, save_vis=args.save_vis)
    logger.info("-> %s", path)
