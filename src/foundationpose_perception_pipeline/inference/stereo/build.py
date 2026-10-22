#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FoundationStereo shape profiles and prebuilding through the shared TensorRT cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from foundationpose_perception_pipeline.inference.trt_build import build_cached_engine, engine_path_for_model

PRECISIONS = ("fp32", "fp16", "bf16")
DIVISIBILITY = 32


@dataclass(frozen=True)
class ShapeProfile:
    """The optimisation profile's spatial bounds, as `(height, width)` triples."""

    min_hw: tuple[int, int]
    opt_hw: tuple[int, int]
    max_hw: tuple[int, int]

    @classmethod
    def static(cls, height: int, width: int) -> ShapeProfile:
        """A profile pinned to one shape -- min = opt = max."""
        return cls((height, width), (height, width), (height, width))

    @property
    def is_static(self) -> bool:
        return self.min_hw == self.opt_hw == self.max_hw

    def validate(self) -> None:
        """Reject shapes the network cannot consume, and bounds that are not ordered."""
        for name, (height, width) in (("min", self.min_hw), ("opt", self.opt_hw), ("max", self.max_hw)):
            if height <= 0 or width <= 0 or height % DIVISIBILITY or width % DIVISIBILITY:
                raise ValueError(
                    f"{name} shape {height}x{width} is not a multiple of {DIVISIBILITY}. "
                    "FoundationStereo downsamples by 32; a non-multiple makes its skip "
                    "connections disagree at runtime."
                )
        for axis, index in (("height", 0), ("width", 1)):
            low, opt, high = self.min_hw[index], self.opt_hw[index], self.max_hw[index]
            if not low <= opt <= high:
                raise ValueError(f"{axis} bounds must satisfy min <= opt <= max, got {low} <= {opt} <= {high}")

    def label(self) -> str:
        """Short form for the cache key."""
        if self.is_static:
            return f"{self.opt_hw[0]}x{self.opt_hw[1]}"
        return f"{self.min_hw[0]}x{self.min_hw[1]}-{self.max_hw[0]}x{self.max_hw[1]}"


def _input_profiles(profile: ShapeProfile):
    profile.validate()
    shapes = tuple((1, 3, *hw) for hw in (profile.min_hw, profile.opt_hw, profile.max_hw))
    return {name: shapes for name in ("left_image", "right_image")}


def engine_path_for(
    onnx_path: Path, profile: ShapeProfile, precision: str, models_dir: Path | None = None, *,
    workspace_mb: int | None = None, tf32: bool = True,
) -> Path:
    """Resolve the same cache key used by prebuilding and automatic compilation."""
    return engine_path_for_model(
        onnx_path, profiles=_input_profiles(profile), precision=precision,
        models_dir=models_dir, workspace_mb=workspace_mb, tf32=tf32,
    )


def build_engine(
    onnx_path: Path | str, *, profile: ShapeProfile, precision: str = "fp32",
    models_dir: Path | None = None, workspace_mb: int | None = None,
    tf32: bool = True, force: bool = False,
) -> Path:
    """Build or reuse a stereo plan; the runtime uses the same builder and cache."""
    return build_cached_engine(
        onnx_path, profiles=_input_profiles(profile), precision=precision,
        models_dir=models_dir, workspace_mb=workspace_mb, tf32=tf32, force=force,
    )


def parse_shape(text: str) -> tuple[int, int]:
    """Parse an `HxW` argument."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected a shape as HxW, got {text!r}")
    return int(parts[0]), int(parts[1])


def padded_shape(height: int, width: int) -> tuple[int, int]:
    """Round each dimension up to a multiple of 32, which is what the network can consume.

    A primitive, and on its own NOT the shape an engine should be built for: padding the width
    changes the width the model sees, and the runtime preserves aspect ratio when fitting to it.
    Use :func:`model_shape_for_rectified`, which accounts for that.
    """
    return (
        height + (-height) % DIVISIBILITY,
        width + (-width) % DIVISIBILITY,
    )


def model_shape_for_rectified(
    rect_height: int, rect_width: int, max_width: int | None = None
) -> tuple[int, int]:
    """The engine input shape a rectified pair of this size needs, so nothing is cropped.

    Padding the two dimensions independently is the obvious thing to do and it is wrong, because
    the runtime does not pad the width -- it RESIZES to it. `stereo/depth.py:fit_to_model` scales
    the pair by width to the engine's width, keeps the aspect ratio, and then pads or crops the
    height. So the height the engine must accommodate follows from the PADDED width:

        model_w  = pad32(rect_width)
        scaled_h = round(rect_height * model_w / rect_width)      <- what the runtime produces
        model_h  = pad32(scaled_h)

    Worked example, a 720x540 rectified pair: the width pads to 736, which scales the height to
    552, so the engine needs 576 rows. Padding independently would give 544 and the runtime would
    crop 8 rows off the bottom of every frame -- quietly, since a crop is a warning and not an
    error. The two agree whenever the rectified width is already a multiple of 32, which is why
    this only bites some rigs.

    `max_width` caps the width first, mirroring the depth stage. Note the cap applies to DYNAMIC
    engines only -- a fixed-shape engine resizes to its own input regardless (`stereo/depth.py`),
    which is exactly the resize this function is sizing for.
    """
    width = min(rect_width, max_width) if max_width else rect_width
    model_w = width + (-width) % DIVISIBILITY
    scaled_h = round(rect_height * model_w / rect_width)
    return scaled_h + (-scaled_h) % DIVISIBILITY, model_w
