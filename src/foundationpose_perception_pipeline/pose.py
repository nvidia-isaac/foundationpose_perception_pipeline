#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FoundationPose runtime plumbing: estimator lifecycle, CAD rendering, path resolution.

Pose *metrics* live in `detection.PoseMetricRegistry`; this module is the runtime side
-- constructing estimators, rendering CAD silhouettes at estimated poses for the render-IoU
features the reranker scores on, and validating the FoundationPose checkout layout.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foundationpose_perception_pipeline.config import foundationpose_root_missing_message, help_requested


@dataclass(frozen=True)
class PoseFilterResult:
    """FoundationPose verification diagnostics for one SAM3 proposal."""

    pred_index: int
    kept: bool
    fp_score: float | None
    fp_elapsed_sec: float | None
    proposal_render_mask_iou: float
    proposal_render_box_iou: float
    pose_row_major: list[float] | None
    render_box_xyxy: list[float] | None
    error: str | None


class PoseRenderer:
    """Render CAD silhouettes from FoundationPose output poses."""

    def __init__(self, dataset_root: Path, models_subdir: str = "models_cad") -> None:
        """Initialize the mesh cache used for pose-conditioned rendering.

        Reads the SAME `models_subdir` the pose stage estimates against.
        """
        self.dataset_root = dataset_root
        self.models_subdir = models_subdir
        self.mesh_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    def mesh(self, dataset: str, obj_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load and cache the render mesh for one dataset/object pair."""
        from foundationpose_perception_pipeline.geometry import read_binary_little_endian_ply

        key = (dataset, obj_id)
        if key not in self.mesh_cache:
            path = self.dataset_root / dataset / self.models_subdir / f"obj_{obj_id:06d}.ply"
            self.mesh_cache[key] = read_binary_little_endian_ply(path)
        return self.mesh_cache[key]

    def render_mask_from_pose(
        self,
        dataset: str,
        obj_id: int,
        pose_row_major: np.ndarray,
        camera_matrix: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Render the object's silhouette from one predicted FoundationPose pose."""
        from foundationpose_perception_pipeline.geometry import render_mask

        vertices, faces = self.mesh(dataset, obj_id)
        rotation = pose_row_major[:3, :3].astype(np.float64)
        translation_mm = pose_row_major[:3, 3].astype(np.float64) * 1000.0
        return render_mask(vertices, faces, rotation, translation_mm, camera_matrix, image_size)


class FoundationPoseRegistry:
    """Lazy cache of single-object FoundationPose estimators."""

    def __init__(
        self,
        engine_cache_dir: Path,
        refine_model_path: Path,
        score_model_path: Path,
        device_id: int,
        prepare_batch: int,
        models_subdir: str = "models_cad",
    ) -> None:
        """Store estimator configuration and prepare an empty estimator cache.

        `models_subdir` is where the CAD meshes live under `<dataset_root>/<dataset>/`. It is a
        parameter rather than a constant because standard BOP datasets ship meshes in `models/`,
        while this project's captures use `models_cad/` -- and a hardcoded name here would make
        `--models-subdir` a flag that the evaluator honoured and the pose stage ignored.
        """
        self.models_subdir = models_subdir
        self.engine_cache_dir = engine_cache_dir
        self.refine_model_path = refine_model_path
        self.score_model_path = score_model_path
        self.device_id = device_id
        self.prepare_batch = prepare_batch
        self.estimators: dict[tuple[str, int, int, int], Any] = {}

    def close(self) -> None:
        """Close all cached estimators and clear the registry."""
        for estimator in self.estimators.values():
            # Teardown is best-effort by design: this releases native FoundationPose handles at
            # the end of a run, and one estimator failing to close must not prevent the rest from
            # being closed or leave the registry populated.
            with contextlib.suppress(Exception):
                estimator.close()
        self.estimators.clear()

    def get(self, dataset_root: Path, dataset: str, obj_id: int, image_size: tuple[int, int]) -> Any:
        """Create or reuse a single-object FoundationPose estimator for one image size."""
        from foundation_pose_nvidia import Estimator, EstimatorOptions, RuntimeConfig

        width, height = image_size
        key = (dataset, obj_id, width, height)
        if key not in self.estimators:
            cad_path = dataset_root / dataset / self.models_subdir / f"obj_{obj_id:06d}.ply"
            options = EstimatorOptions.from_env(
                cad_path,
                refine_model_path=self.refine_model_path,
                score_model_path=self.score_model_path,
                # ONE DIRECTORY FOR EVERYTHING, deliberately: no per-object and no per-dataset
                # component. The refine and score networks depend on neither. They are built from
                # two fixed ONNX graphs over fixed-size 160x160 crops; the mesh is consumed by the
                # renderer and never reaches either net, and nothing else about a dataset enters
                # their construction. So any subdivision here isolates nothing -- it only hides the
                # engine the previous object or dataset built, and each one then pays a
                # multi-minute TensorRT build for a byte-identical result.
                #
                # The FoundationPose Inference Library agrees: `multi_object.py` hands every object
                # in a group the same `engine_cache_dir`, varying only `cad_path` and
                # `mesh_unit_scale`. So does
                # `run_batch_eval.py`, which deliberately passes ONE root for a whole sweep --
                # a per-dataset component here quietly undid that.
                #
                # The tell, if either component is reintroduced: every directory holds a plan
                # whose filename carries the FoundationPose Inference Library's own cache key,
                # and the keys are all equal.
                # Measured that way before removing each: 30 directories and 2 distinct engines on
                # one dataset, then 82 directories and still 2 across a 32-dataset sweep.
                engine_cache_dir=self.engine_cache_dir,
                device_id=self.device_id,
                mesh_unit_scale=0.001,
            )
            config = RuntimeConfig(max_image_width=width, max_image_height=height)
            self.estimators[key] = Estimator(options, config, prepare_batch=self.prepare_batch)
        return self.estimators[key]


def inject_external_paths(repo_root: Path, foundationpose_root: Path | None) -> None:
    """Put the sam3 and FoundationPose checkouts on `sys.path`, and set FP_LIBRARY.

    Both are external checkouts rather than installed packages, so they must be located before
    anything imports them -- which is before argument parsing, so `--foundationpose-root`
    cannot help. Every entry point needs this identically; when it lived in one script's
    preamble, a second entry point silently lost it and failed with
    `ModuleNotFoundError: No module named 'foundation_pose_nvidia'` only once it reached the
    first pose estimate.

    A MISSING CHECKOUT IS FATAL FOR A RUN AND IRRELEVANT TO `--help`. This runs at module scope,
    before argparse exists, so raising here means `python run_pipeline.py --help` answers "set
    FOUNDATIONPOSE_ROOT" to someone who asked what the flags are -- and `--foundationpose-root`
    is one of the flags that answer would have shown them. Nothing on the help path touches the
    checkout: both `foundation_pose_nvidia` and `sam3` are imported inside the functions that
    use them, never at module scope, and argparse exits before any of those are reached. So a
    help request injects nothing and returns; every other invocation still stops here.
    """
    import os
    import sys

    if foundationpose_root is None:
        if help_requested():
            return
        raise SystemExit(foundationpose_root_missing_message())
    os.environ.setdefault(
        "FP_LIBRARY", str((foundationpose_root / "build" / "libfoundation_pose_nvidia.so").resolve())
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for path in (repo_root / "sam3", foundationpose_root / "python" / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_foundationpose_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve and validate the FoundationPose library and model paths."""
    if args.foundationpose_root is None:
        raise FileNotFoundError(foundationpose_root_missing_message())
    root = args.foundationpose_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"FoundationPose root does not exist: {root}")

    library = (
        args.fp_library.expanduser().resolve()
        if args.fp_library is not None
        else (root / "build" / "libfoundation_pose_nvidia.so").resolve()
    )
    refine = (
        args.fp_refine_model_path.expanduser().resolve()
        if args.fp_refine_model_path is not None
        else (root / "weights" / "refiner_net.onnx").resolve()
    )
    score = (
        args.fp_score_model_path.expanduser().resolve()
        if args.fp_score_model_path is not None
        else (root / "weights" / "score_net.onnx").resolve()
    )
    for path, label in ((library, "library"), (refine, "refine model"), (score, "score model")):
        if not path.exists():
            raise FileNotFoundError(f"Missing FoundationPose {label}: {path}")
    os.environ["FP_LIBRARY"] = str(library)
    return library, refine, score
