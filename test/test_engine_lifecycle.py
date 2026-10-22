#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When PoseEstimator releases its FoundationPose context, with and without `retain_pose_context`.

Runs with the standard library only: PYTHONPATH=src python -m unittest discover -s test
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from foundationpose_perception_pipeline.inference.engine import PoseEstimator

SIZE = (64, 48)
OTHER_SIZE = (128, 96)


class CountingRegistry:
    """Stands in for FoundationPoseRegistry and counts what the estimator asks of it."""

    def __init__(self, failures: int = 0) -> None:
        self.closes = 0
        self.gets = 0
        self._failures = failures

    def close(self) -> None:
        self.closes += 1

    def get(self, dataset_root: Path, dataset: str, obj_id: int, image_size: tuple[int, int]) -> object:
        self.gets += 1
        if self._failures:
            self._failures -= 1
            raise RuntimeError("context acquisition failed")
        return object()


def make_estimator(registry: CountingRegistry, **kwargs: bool) -> PoseEstimator:
    return PoseEstimator(
        processor=SimpleNamespace(set_image=lambda image: {}),
        pose_registry=registry,
        pose_renderer=None,
        dataset_root=Path("."),
        run_foundationpose_for_proposals=lambda **_: [],
        apply_sam3_refinement=lambda **kw: (kw["raw_boxes"], kw["raw_masks"], kw["raw_scores"], [], {}, []),
        select_proposals=lambda **_: ([], [], {}),
        mark_selected_filter_results=lambda results, kept: [],
        base_text_state_from_prompt_state=lambda state: state,
        no_refinement_policy="none",
        **kwargs,
    )


def run_scene(estimator: PoseEstimator, obj_id: int, size: tuple[int, int] = SIZE) -> None:
    """One scene with one target: begin_scene -> run_target -> end_scene."""
    width, height = size
    estimator.begin_scene(image=None, depth_image_size=size)
    estimator.run_target(
        config=SimpleNamespace(
            pose=SimpleNamespace(n_refine=1, n_hypotheses=1),
            refinement=SimpleNamespace(policy="none"),
            selection=None,
        ),
        target=SimpleNamespace(dataset="d", obj_id=obj_id),
        image=np.zeros((height, width, 3), dtype=np.uint8),
        camera_matrix=np.eye(3),
        depth_result={},
        pose_image_np=None,
        depth_image_size=size,
        raw_boxes=np.zeros((0, 4)),
        raw_scores=np.zeros((0,)),
        raw_masks=np.zeros((0, height, width), dtype=bool),
        base_text_state={},
    )
    estimator.end_scene()


class DefaultLifecycleTest(unittest.TestCase):
    def test_context_is_released_after_every_scene(self) -> None:
        registry = CountingRegistry()
        estimator = make_estimator(registry)

        run_scene(estimator, obj_id=1)
        closes = registry.closes
        run_scene(estimator, obj_id=1)

        self.assertGreater(closes, 0)
        self.assertGreater(registry.closes, closes)


class RetainedLifecycleTest(unittest.TestCase):
    def test_same_object_keeps_its_context_across_scenes(self) -> None:
        registry = CountingRegistry()
        estimator = make_estimator(registry, retain_pose_context=True)

        run_scene(estimator, obj_id=1)
        closes = registry.closes
        run_scene(estimator, obj_id=1)
        run_scene(estimator, obj_id=1)

        self.assertEqual(registry.closes, closes)
        self.assertEqual(registry.gets, 3)

    def test_switching_object_releases_the_previous_context(self) -> None:
        registry = CountingRegistry()
        estimator = make_estimator(registry, retain_pose_context=True)

        run_scene(estimator, obj_id=1)
        closes = registry.closes
        run_scene(estimator, obj_id=2)

        self.assertGreater(registry.closes, closes)

    def test_changing_image_size_releases_the_context(self) -> None:
        registry = CountingRegistry()
        estimator = make_estimator(registry, retain_pose_context=True)

        run_scene(estimator, obj_id=1)
        closes = registry.closes
        run_scene(estimator, obj_id=1, size=OTHER_SIZE)

        self.assertGreater(registry.closes, closes)

    def test_failed_acquisition_still_releases_and_retries(self) -> None:
        registry = CountingRegistry()
        estimator = make_estimator(registry, retain_pose_context=True)
        run_scene(estimator, obj_id=1)
        closes, gets = registry.closes, registry.gets

        registry._failures = 1
        run_scene(estimator, obj_id=1)

        self.assertGreater(registry.closes, closes)
        self.assertEqual(registry.gets, gets + 2)


if __name__ == "__main__":
    unittest.main()
