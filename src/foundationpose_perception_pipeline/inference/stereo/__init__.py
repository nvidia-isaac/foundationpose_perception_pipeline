#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stereo geometry and native TensorRT inference for TAO FoundationStereo exports.

Importing this package must not require CUDA. The geometry half -- rectification, disparity
conversion, the scene layout -- is pure NumPy/OpenCV and is used by tooling that never touches
a GPU (`tools/bop_adapt/partners.py`, for one). The inference half reaches
`inference/trt.py`, which imports `tensorrt` and `cuda.bindings` at module scope.

So the TensorRT-backed names are resolved on first attribute access rather than at import.
`from ...stereo import FoundationStereoTrtProcessor` still works and still fails loudly if
TensorRT is missing; `from ...stereo import fit_to_model` no longer fails at all.
"""

from typing import TYPE_CHECKING, Any

from foundationpose_perception_pipeline.inference.stereo.build import (
    ShapeProfile,
    build_engine,
    engine_path_for,
)
from foundationpose_perception_pipeline.inference.stereo.depth import (
    SceneDepth,
    StereoDepthError,
    disparity_to_depth_m,
    fit_to_model,
    scene_depth,
    write_scene_depth,
)

if TYPE_CHECKING:
    from foundationpose_perception_pipeline.inference.stereo.trt_processor import (
        FoundationStereoTrtProcessor,
        load_engine,
        normalize_for_model,
        release_engines,
    )

_TRT_EXPORTS = frozenset(
    {"FoundationStereoTrtProcessor", "load_engine", "normalize_for_model", "release_engines"}
)


def __getattr__(name: str) -> Any:
    """Resolve the TensorRT-backed exports lazily -- see the module docstring."""
    if name not in _TRT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from foundationpose_perception_pipeline.inference.stereo import trt_processor

    return getattr(trt_processor, name)


__all__ = [
    "FoundationStereoTrtProcessor",
    "SceneDepth",
    "ShapeProfile",
    "StereoDepthError",
    "build_engine",
    "disparity_to_depth_m",
    "engine_path_for",
    "fit_to_model",
    "load_engine",
    "normalize_for_model",
    "release_engines",
    "scene_depth",
    "write_scene_depth",
]
