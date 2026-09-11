#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pose estimation: FoundationPose over a set of proposal masks.

Ground-truth free. The `proposal_render_*_iou` values this produces compare a proposal's mask
against a CAD render of **its own predicted pose**, never against annotations -- which is what
lets the selection stage downstream be deployed rather than only evaluated.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from foundationpose_perception_pipeline.dataset import Target
from foundationpose_perception_pipeline.geometry import bbox_from_mask, binary_mask_iou, box_iou
from foundationpose_perception_pipeline.pose import PoseFilterResult, PoseRenderer


def resize_mask_nearest(mask: np.ndarray, target_size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbor sampling."""
    target_w, target_h = target_size_wh
    if mask.shape == (target_h, target_w):
        return mask.astype(bool, copy=False)
    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)

def scale_box_xyxy(
    box: np.ndarray | list[float],
    *,
    src_size_wh: tuple[int, int],
    dst_size_wh: tuple[int, int],
) -> list[float]:
    """Scale an XYXY box from one image size to another."""
    src_w, src_h = src_size_wh
    dst_w, dst_h = dst_size_wh
    sx = dst_w / src_w
    sy = dst_h / src_h
    x0, y0, x1, y1 = [float(value) for value in box]
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

def run_foundationpose_for_proposals(
    *,
    estimator: Any,
    target: Target,
    masks: np.ndarray,
    pose_renderer: PoseRenderer,
    depth_result: dict[str, Any],
    pose_image_np: np.ndarray,
    depth_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    n_refine: int,
    n_hypotheses: int,
) -> list[PoseFilterResult]:
    """Run FoundationPose on each proposal mask and collect overlap diagnostics."""
    # Deferred: the FoundationPose Inference Library is an external checkout whose path
    # `ensure_foundationpose_paths` injects at start-up, so importing it at module scope
    # would make this package unimportable on a machine that has not built it.
    from foundation_pose_nvidia import RgbdFrame

    filter_results: list[PoseFilterResult] = []
    for pred_index, pred_mask in enumerate(masks):
        fp_score = None
        fp_elapsed_sec = None
        proposal_render_mask_iou = 0.0
        proposal_render_box_iou = 0.0
        pose_row_major = None
        render_box_xyxy = None
        error = None
        kept = True
        try:
            pred_mask_pose = resize_mask_nearest(pred_mask, depth_image_size)
            pred_box_pose = bbox_from_mask(pred_mask_pose) if pred_mask_pose.any() else None
            mask_u8 = np.where(pred_mask_pose, 255, 0).astype(np.uint8)
            result = estimator.register(
                RgbdFrame(
                    rgb=pose_image_np,
                    depth_m=depth_result["depth_m"],
                    intrinsics=depth_result["camera_matrix"].astype(np.float32),
                    mask=mask_u8,
                ),
                n_refine=n_refine,
                n_hypotheses=n_hypotheses,
            )
            fp_score = float(result.score)
            fp_elapsed_sec = float(result.elapsed_s)
            pose_row_major_np = result.pose.astype(np.float32)
            rendered_mask = pose_renderer.render_mask_from_pose(
                target.dataset,
                target.obj_id,
                pose_row_major_np,
                depth_result["camera_matrix"],
                depth_image_size,
            )
            if rendered_mask.any():
                render_box = bbox_from_mask(rendered_mask)
                render_box_xyxy = scale_box_xyxy(
                    render_box,
                    src_size_wh=depth_image_size,
                    dst_size_wh=rgb_image_size,
                )
                proposal_render_mask_iou = float(binary_mask_iou(rendered_mask, pred_mask_pose))
                if pred_box_pose is not None:
                    proposal_render_box_iou = float(box_iou(render_box, pred_box_pose))
            pose_row_major = pose_row_major_np.reshape(-1).astype(float).tolist()
        except Exception as exc:  # noqa: BLE001 -- recorded per proposal, not swallowed: the
            # message goes into the prediction record so one failed proposal cannot end a scene,
            # and the failure stays visible in predictions.jsonl.
            error = f"{type(exc).__name__}: {exc}"

        filter_results.append(
            PoseFilterResult(
                pred_index=pred_index,
                kept=kept,
                fp_score=fp_score,
                fp_elapsed_sec=fp_elapsed_sec,
                proposal_render_mask_iou=proposal_render_mask_iou,
                proposal_render_box_iou=proposal_render_box_iou,
                pose_row_major=pose_row_major,
                render_box_xyxy=render_box_xyxy,
                error=error,
            )
        )
    return filter_results
