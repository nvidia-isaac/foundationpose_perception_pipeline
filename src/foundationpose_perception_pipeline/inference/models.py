#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic model locations: ONNX sources beside an engine_cache directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from foundationpose_perception_pipeline.config import models_dir_default

STEREO_MODEL = "deployable_foundation_stereo_s_dynamic_v2.0"


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
        engine = self.engine(name)
        return engine if engine.is_file() else self.onnx(name)
