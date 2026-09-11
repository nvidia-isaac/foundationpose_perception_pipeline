#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refinement: replace mid-quality proposals using a CAD box prompt.

Runs SAM3 a second time, prompted by the box of the CAD model rendered at the pose just
estimated. Refinement *replaces* masks and can therefore raise true positives, unlike the
selection stage which only drops proposals -- keeping the two separable is what lets a report
attribute a change to one or the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from foundationpose_perception_pipeline.config import (
    NO_REFINEMENT_POLICY,
    REPLACE_MID_NMS06_REFINEMENT_POLICY,
)
from foundationpose_perception_pipeline.dataset import Target
from foundationpose_perception_pipeline.geometry import binary_mask_iou
from foundationpose_perception_pipeline.inference.config import RefinementConfig
from foundationpose_perception_pipeline.inference.detect import (
    base_text_state_from_prompt_state,  # noqa: F401
)
from foundationpose_perception_pipeline.pose import PoseFilterResult, PoseRenderer
from foundationpose_perception_pipeline.runtime import inference_context, tensor_to_numpy
from foundationpose_perception_pipeline.visualize import xyxy_to_norm_cxcywh


def apply_mask_nms_with_indices(
    *,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Apply score-ordered mask NMS and return kept indices with filtered arrays."""
    if len(boxes) == 0:
        return [], boxes, masks, scores
    order = np.argsort(-scores)
    keep: list[int] = []
    for index in order.tolist():
        if any(binary_mask_iou(masks[index], masks[kept]) > threshold for kept in keep):
            continue
        keep.append(index)
    return keep, boxes[keep], masks[keep], scores[keep]

def best_refined_candidate(
    *,
    processor: Any,  # Sam3Processor; untyped to keep sam3 out of this module's imports
    image: Image.Image,
    base_text_state: dict[str, Any],
    render_box_xyxy: list[float],
    rendered_mask: np.ndarray,
    device: str,
) -> dict[str, Any] | None:
    """Run one box-prompted SAM3 refinement and keep the best candidate mask."""
    state = {
        "original_height": base_text_state["original_height"],
        "original_width": base_text_state["original_width"],
        "backbone_out": base_text_state["backbone_out"],
    }
    with inference_context(device):
        state = processor.add_geometric_prompt(
            box=xyxy_to_norm_cxcywh(render_box_xyxy, image.size),
            label=True,
            state=state,
        )
    boxes = tensor_to_numpy(state["boxes"])
    scores = tensor_to_numpy(state["scores"])
    masks = tensor_to_numpy(state["masks"][:, 0]).astype(bool)
    if len(boxes) == 0:
        return None

    best_index = max(
        range(len(boxes)),
        key=lambda idx: (binary_mask_iou(masks[idx], rendered_mask), float(scores[idx])),
    )
    return {
        "box": boxes[best_index].astype(np.float64),
        "mask": masks[best_index],
        "score": float(scores[best_index]),
        "candidate_count": len(boxes),
        "render_mask_iou": float(binary_mask_iou(masks[best_index], rendered_mask)),
    }

def apply_sam3_refinement(
    *,
    config: RefinementConfig,
    processor: Any,  # Sam3Processor; untyped to keep sam3 out of this module's imports
    image: Image.Image,
    target: Target,
    camera_matrix_rgb: np.ndarray,
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    raw_masks: np.ndarray,
    raw_filter_results: list[PoseFilterResult],
    base_text_state: dict[str, Any],
    pose_renderer: PoseRenderer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, Any], list[dict[str, Any]]]:
    """Optionally replace mid-overlap proposals with box-prompted SAM3 refinements."""
    if config.policy == NO_REFINEMENT_POLICY:
        return (
            raw_boxes,
            raw_masks,
            raw_scores,
            list(range(len(raw_boxes))),
            {
                "policy": config.policy,
                "candidate_count": 0,
                "replaced_count": 0,
                "pre_nms_count": len(raw_boxes),
                "post_nms_count": len(raw_boxes),
                "replace_miou_low": None,
                "replace_miou_high": None,
                "nms_threshold": None,
                "replaced_indices": [],
            },
            [],
        )

    if config.policy != REPLACE_MID_NMS06_REFINEMENT_POLICY:
        raise ValueError(f"Unsupported SAM3 refinement policy: {config.policy}")

    refined_variants: list[dict[str, Any] | None] = [None] * len(raw_masks)
    candidate_rows: list[dict[str, Any]] = []
    for pose_row in raw_filter_results:
        pred_index = int(pose_row.pred_index)
        candidate_row = {
            "pred_index": pred_index,
            "eligible": False,
            "candidate_count": 0,
            "render_prompt_mask_iou": None,
            "refined_score": None,
            "replaced": False,
        }
        if pose_row.pose_row_major is None or pose_row.error is not None or pose_row.render_box_xyxy is None:
            candidate_rows.append(candidate_row)
            continue
        pose_row_major_np = np.asarray(pose_row.pose_row_major, dtype=np.float32).reshape(4, 4)
        rendered_mask = pose_renderer.render_mask_from_pose(
            target.dataset,
            target.obj_id,
            pose_row_major_np,
            camera_matrix_rgb.astype(np.float32),
            image.size,
        )
        candidate_row["eligible"] = True
        refined = best_refined_candidate(
            processor=processor,
            image=image,
            base_text_state=base_text_state,
            render_box_xyxy=pose_row.render_box_xyxy,
            rendered_mask=rendered_mask,
            device=config.device,
        )
        if refined is not None:
            refined_variants[pred_index] = refined
            candidate_row["candidate_count"] = int(refined["candidate_count"])
            candidate_row["render_prompt_mask_iou"] = float(refined["render_mask_iou"])
            candidate_row["refined_score"] = float(refined["score"])
        candidate_rows.append(candidate_row)

    replace_boxes = raw_boxes.copy()
    replace_masks = raw_masks.copy()
    replace_scores = raw_scores.copy()
    replaced_indices: list[int] = []
    for pred_index, refined in enumerate(refined_variants):
        if refined is None:
            continue
        pose_row = raw_filter_results[pred_index]
        if (
            config.low_miou
            <= float(pose_row.proposal_render_mask_iou)
            <= config.high_miou
        ):
            replace_boxes[pred_index] = refined["box"]
            replace_masks[pred_index] = refined["mask"]
            replace_scores[pred_index] = float(max(raw_scores[pred_index], refined["score"]))
            replaced_indices.append(pred_index)

    replaced_lookup = set(replaced_indices)
    for row in candidate_rows:
        row["replaced"] = int(row["pred_index"]) in replaced_lookup

    keep_indices, refined_boxes, refined_masks, refined_scores = apply_mask_nms_with_indices(
        boxes=replace_boxes,
        masks=replace_masks,
        scores=replace_scores,
        threshold=config.nms_threshold,
    )
    summary = {
        "policy": config.policy,
        "candidate_count": int(sum(row["eligible"] for row in candidate_rows)),
        "replaced_count": len(replaced_indices),
        "pre_nms_count": len(raw_boxes),
        "post_nms_count": len(keep_indices),
        "replace_miou_low": config.low_miou,
        "replace_miou_high": config.high_miou,
        "nms_threshold": config.nms_threshold,
        "replaced_indices": replaced_indices,
        "pose_input_source_indices": keep_indices,
    }
    return refined_boxes, refined_masks, refined_scores, keep_indices, summary, candidate_rows
