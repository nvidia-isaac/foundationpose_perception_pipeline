#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic model locations: ONNX sources beside an engine_cache directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from foundationpose_perception_pipeline.config import models_dir_default

STEREO_MODEL = "deployable_foundation_stereo_s_dynamic_v2.0"
SAM3_MODELS = {
    "vision": "sam3_vision_encoder",
    "text": "sam3_text_encoder",
    "decoder": "sam3_mask_decoder",
    "box_decoder": "sam3_box_decoder",
}


@dataclass(frozen=True)
class ModelPaths:
    """Use an explicit directory, MODELS_DIR, or the YAML models_dir setting.

    Named plans are user-supplied overrides. Automatically built plans live in the
    same cache directory with a fingerprint suffix; there is no directory search.
    """

    root: Path

    @classmethod
    def configured(cls, root: Path | str | None = None) -> ModelPaths:
        return cls(Path(root).expanduser().resolve() if root is not None else models_dir_default())

    @property
    def engine_cache(self) -> Path:
        return self.root / "engine_cache"

    def onnx(self, name: str) -> Path:
        return self.root / f"{name}.onnx"

    def engine(self, name: str) -> Path:
        return self.engine_cache / f"{name}.plan"

    def preferred(self, name: str) -> Path:
        """Return the best artifact available for `name`, in precedence order.

        1. `{name}.plan`, a user-supplied override.
        2. A single `{name}__{fingerprint}.plan` from an earlier build. Returning it directly
           skips re-hashing the ONNX, which `build_cached_engine` would otherwise do purely to
           recompute a filename that is already sitting in the cache. For multi-gigabyte
           encoders that hash dominates startup.
        3. The ONNX source, which `TRTEngine` compiles through `build_cached_engine`.

        Several fingerprinted plans means several build specs -- different GPU, TensorRT
        version, precision or profile. Choosing between those is precisely the job the
        fingerprint exists to do, so that case deliberately falls through to the ONNX and lets
        `build_cached_engine` recompute the real fingerprint and select exactly.

        Case 2 trusts a plan on its name alone: a plan left over from a superseded ONNX export
        is reused rather than rebuilt. Remove stale plans from the cache after re-exporting.
        """
        engine = self.engine(name)
        if engine.is_file():
            return engine
        cached = sorted(self.engine_cache.glob(f"{name}__*.plan"))
        if len(cached) == 1:
            return cached[0]
        return self.onnx(name)
