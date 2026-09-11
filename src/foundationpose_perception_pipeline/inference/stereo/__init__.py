#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stereo depth from the commercially-licensable FoundationStereo model, via TAO Deploy.

This is the depth path that ships. It runs a TAO `deployable_*` export as a TensorRT engine,
in the pipeline's own environment and in the pipeline's own process -- no FoundationStereo source
checkout, no second venv, no subprocess per scene.

    from foundationpose_perception_pipeline.inference.stereo import load_engine, scene_depth, write_scene_depth

    depth = scene_depth(scene_dir, engine=load_engine(engine_path), max_width=800)
    write_scene_depth(depth, out_dir)


Requires `nvidia-tao-deploy` and `pycuda`, installed out-of-band -- see README.md's
FoundationStereo section, and pyproject.toml for why they are not a `uv` extra. Nothing in this
package is imported unless a caller asks for it, so a checkout without them still works for every
other stage.
"""

from foundationpose_perception_pipeline.inference.stereo.build import (
    ShapeProfile,
    build_engine,
    engine_path_for,
)
from foundationpose_perception_pipeline.inference.stereo.depth import (
    SceneDepth,
    StereoDepthError,
    scene_depth,
    write_scene_depth,
)
from foundationpose_perception_pipeline.inference.stereo.tao import (
    StereoEngine,
    load_engine,
    normalize_for_model,
)

__all__ = [
    "SceneDepth",
    "ShapeProfile",
    "StereoDepthError",
    "StereoEngine",
    "build_engine",
    "engine_path_for",
    "load_engine",
    "normalize_for_model",
    "scene_depth",
    "write_scene_depth",
]
