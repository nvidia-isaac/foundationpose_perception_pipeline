#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pose estimator: proposals, poses, refinement and selection for one target.

Owns the two pieces of mutable state the pipeline used to keep in `main` as loose variables,
`active_pose_image_size` and `active_target_obj_id`. Both exist to decide *when to tear down a
FoundationPose context*, and getting that wrong is expensive in opposite directions: too eager
and every target rebuilds its TensorRT engines, too lazy and a stale context is handed an image
of the wrong size. Keeping them beside the registry they guard is the point of this class.

Nothing here reads ground truth. The camera matrix arrives as an argument rather than being
taken from the GT bundle -- it is intrinsics from `scene_camera.json`, and the GT renderer was
only ever a convenient place to find it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foundationpose_perception_pipeline.inference.config import InferenceConfig


@dataclass
class TargetInference:
    """Raw output of one target's inference pass, before anything is scored.

    Deliberately not a `TargetPrediction`: this carries the working arrays and the
    FoundationPose result objects that evaluation and serialization both need. The caller
    assembles the serializable form.
    """

    raw_boxes: np.ndarray
    raw_scores: np.ndarray
    raw_masks: np.ndarray
    raw_filter_results: list[Any]
    pose_input_boxes: np.ndarray
    pose_input_masks: np.ndarray
    pose_input_scores: np.ndarray
    pose_input_source_indices: Any
    sam3_refinement_summary: Any
    sam3_refinement_candidates: Any
    filter_results: list[Any]
    kept_indices: list[int]
    rerank_rows: list[dict[str, Any]]
    selection_summary: Any
    selection_results: list[Any]
    filtered_boxes: np.ndarray
    filtered_masks: np.ndarray
    filtered_scores: np.ndarray


class PoseEstimator:
    """SAM3 proposals -> FoundationPose -> CAD-prompt refinement -> rerank, for one target.

    Collaborators are injected rather than constructed here so that the caller keeps control of
    model loading, which is slow and belongs to the process, not to this object.
    """

    def __init__(
        self,
        *,
        processor: Any,
        pose_registry: Any,
        pose_renderer: Any,
        dataset_root: Path,
        run_foundationpose_for_proposals: Any,
        apply_sam3_refinement: Any,
        select_proposals: Any,
        mark_selected_filter_results: Any,
        base_text_state_from_prompt_state: Any,
        no_refinement_policy: str,
    ) -> None:
        self.processor = processor
        self.pose_registry = pose_registry
        self.pose_renderer = pose_renderer
        self.dataset_root = dataset_root
        self._run_foundationpose = run_foundationpose_for_proposals
        self._apply_refinement = apply_sam3_refinement
        self._select_proposals = select_proposals
        self._mark_selected = mark_selected_filter_results
        self._base_text_state = base_text_state_from_prompt_state
        self._no_refinement_policy = no_refinement_policy
        self._active_pose_image_size: tuple[int, int] | None = None
        self._active_target_obj_id: int | None = None

    def _release(self) -> None:
        """Tear down the current FoundationPose context and reclaim its GPU memory.

        `gc.collect()` is the whole reclamation story: every model in this pipeline is a
        TensorRT engine, so device memory is owned by the engine wrappers and released by
        their finalizers, not by a caching allocator that would need to be drained.
        """
        self.pose_registry.close()
        gc.collect()

    def begin_scene(self, image: Any, depth_image_size: tuple[int, int]) -> dict[str, Any]:
        """Prepare for a new scene: size-check the pose context, embed the image once.

        The SAM3 image embedding is computed once per scene and reused by every target in it,
        which is why this is a separate call rather than part of `run_target`.
        """
        if self._active_pose_image_size != depth_image_size:
            self.pose_registry.close()
            self._active_pose_image_size = depth_image_size
        self._active_target_obj_id = None
        return dict(self.processor.set_image(image))

    def end_scene(self) -> None:
        """Release everything held for the scene just finished.

        FoundationPose/TensorRT memory grows across a long run, so contexts are torn down
        between scenes rather than kept warm. Both lifecycle fields are cleared as well: the
        next scene must not believe a context still exists for its image size or object.
        """
        self.pose_registry.close()
        self._active_pose_image_size = None
        self._active_target_obj_id = None
        gc.collect()

    def _acquire(self, target: Any, depth_image_size: tuple[int, int]) -> Any:
        """Return a FoundationPose estimator for this object, rebuilding the context if needed.

        Only one object is processed at a time, so the previous object's context is released
        first. Acquisition can still fail on a fragmented GPU, and a forced teardown plus a second
        attempt recovers it -- so the failure is caught once, by any type. FoundationPose is a
        native library behind bindings and surfaces memory and handle failures as whatever
        exception the C++ stack happened to reach, so the type carries no signal worth branching
        on. The retry itself is unguarded: if a clean GPU does not fix it, the run should stop.
        """
        if self._active_target_obj_id != target.obj_id:
            self._release()
            self._active_target_obj_id = target.obj_id
        try:
            return self.pose_registry.get(self.dataset_root, target.dataset, target.obj_id, depth_image_size)
        except Exception:  # noqa: BLE001 -- see docstring: the exception type carries no signal
            self._release()
            return self.pose_registry.get(self.dataset_root, target.dataset, target.obj_id, depth_image_size)

    def propose(self, prompt: str, image_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Run SAM3 for one text prompt against the scene's cached image embedding."""
        prompt_state = dict(image_state)
        prompt_state = self.processor.set_text_prompt(prompt=prompt, state=prompt_state)
        return (
            prompt_state["boxes"],
            prompt_state["scores"],
            prompt_state["masks"][:, 0].astype(bool),
            self._base_text_state(prompt_state),
        )

    def run_target(
        self,
        *,
        config: InferenceConfig,
        target: Any,
        image: Any,
        camera_matrix: np.ndarray,
        depth_result: dict[str, Any],
        pose_image_np: np.ndarray,
        depth_image_size: tuple[int, int],
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        raw_masks: np.ndarray,
        base_text_state: Any,
    ) -> TargetInference:
        """Pose, refine, pose again, then select -- the whole GT-free half of a target.

        Takes an `InferenceConfig` rather than an `argparse.Namespace`: each stage receives only
        its own section, so a caller that is not a CLI can drive this without fabricating flags.

        The second FoundationPose pass is skipped when refinement is off, because the refined
        proposal set is then identical to the raw one and re-posing it would be pure cost.
        """
        estimator = self._acquire(target, depth_image_size)
        raw_filter_results = self._run_foundationpose(
            estimator=estimator,
            target=target,
            masks=raw_masks,
            pose_renderer=self.pose_renderer,
            depth_result=depth_result,
            pose_image_np=pose_image_np,
            depth_image_size=depth_image_size,
            rgb_image_size=image.size,
            n_refine=config.pose.n_refine,
            n_hypotheses=config.pose.n_hypotheses,
        )
        (
            pose_input_boxes,
            pose_input_masks,
            pose_input_scores,
            pose_input_source_indices,
            sam3_refinement_summary,
            sam3_refinement_candidates,
        ) = self._apply_refinement(
            config=config.refinement,
            processor=self.processor,
            image=image,
            target=target,
            camera_matrix_rgb=camera_matrix,
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_masks=raw_masks,
            raw_filter_results=raw_filter_results,
            base_text_state=base_text_state,
            pose_renderer=self.pose_renderer,
        )
        if config.refinement.policy == self._no_refinement_policy:
            filter_results = raw_filter_results
        else:
            filter_results = self._run_foundationpose(
                estimator=estimator,
                target=target,
                masks=pose_input_masks,
                pose_renderer=self.pose_renderer,
                depth_result=depth_result,
                pose_image_np=pose_image_np,
                depth_image_size=depth_image_size,
                rgb_image_size=image.size,
                n_refine=config.pose.n_refine,
                n_hypotheses=config.pose.n_hypotheses,
            )

        kept_indices, rerank_rows, selection_summary = self._select_proposals(
            scores=pose_input_scores, filter_results=filter_results, config=config.selection
        )
        for row in rerank_rows:
            row["source_raw_pred_index"] = int(pose_input_source_indices[int(row["pred_index"])])
        selection_results = self._mark_selected(filter_results, kept_indices)

        filtered_boxes = pose_input_boxes[kept_indices] if kept_indices else np.zeros((0, 4), dtype=np.float64)
        filtered_masks = (
            pose_input_masks[kept_indices] if kept_indices else np.zeros((0, *raw_masks.shape[1:]), dtype=bool)
        )
        filtered_scores = pose_input_scores[kept_indices] if kept_indices else np.zeros((0,), dtype=np.float64)

        return TargetInference(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_masks=raw_masks,
            raw_filter_results=raw_filter_results,
            pose_input_boxes=pose_input_boxes,
            pose_input_masks=pose_input_masks,
            pose_input_scores=pose_input_scores,
            pose_input_source_indices=pose_input_source_indices,
            sam3_refinement_summary=sam3_refinement_summary,
            sam3_refinement_candidates=sam3_refinement_candidates,
            filter_results=filter_results,
            kept_indices=kept_indices,
            rerank_rows=rerank_rows,
            selection_summary=selection_summary,
            selection_results=selection_results,
            filtered_boxes=filtered_boxes,
            filtered_masks=filtered_masks,
            filtered_scores=filtered_scores,
        )
