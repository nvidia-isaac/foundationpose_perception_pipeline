#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT SAM 3 Zero-Shot Segmentation & Refinement Processor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from foundationpose_perception_pipeline.inference.models import SAM3_MODELS, ModelPaths
from foundationpose_perception_pipeline.inference.sam3.tokenizer import (
    BPE_VOCAB_FILENAME,
    DEFAULT_CONTEXT_LENGTH,
    Sam3TextTokenizer,
)
from foundationpose_perception_pipeline.inference.trt import TRTEngine

DEFAULT_SAM3_IMAGE_SIZE: int = 1008
DEFAULT_DEVICE_ID: int = 0
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0
SAM3_NORM_MEAN: float = 0.5
SAM3_NORM_STD: float = 0.5
UINT8_MAX: float = 255.0
SIGMOID_CLAMP_BOUND: float = 80.0

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid clipped to [-SIGMOID_CLAMP_BOUND, SIGMOID_CLAMP_BOUND]."""
    clipped = np.clip(x, -SIGMOID_CLAMP_BOUND, SIGMOID_CLAMP_BOUND)
    return 1.0 / (1.0 + np.exp(-clipped))


# -----------------------------------------------------------------------------
# SAM 3 TensorRT Processor
# -----------------------------------------------------------------------------
class Sam3TrtProcessor:
    """Drop-in TensorRT replacement for PyTorch Sam3Processor with Pass 1 & Pass 2 support."""

    def __init__(
        self,
        models_dir: Path | str | None = None,
        resolution: int = DEFAULT_SAM3_IMAGE_SIZE,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        if resolution != DEFAULT_SAM3_IMAGE_SIZE:
            raise ValueError(f"SAM3 exports use a fixed {DEFAULT_SAM3_IMAGE_SIZE}x{DEFAULT_SAM3_IMAGE_SIZE} input, got {resolution}")
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("TensorRT SAM3 requires cuda or cuda:<device_id>")
        self.device_id = int(device.split(":")[1]) if ":" in device else DEFAULT_DEVICE_ID
        self.resolution = int(resolution)
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.paths = ModelPaths.configured(models_dir)
        vocab = self.paths.root / BPE_VOCAB_FILENAME
        self.tokenizer = Sam3TextTokenizer(vocab) if vocab.exists() else None
        self.v_engine = self._load_engine("vision")
        self.t_engine = self._load_engine("text")
        self.d_engine = self._load_engine("decoder")
        self.b_engine = None

    def _load_engine(self, component: str) -> TRTEngine:
        return TRTEngine(
            self.paths.preferred(SAM3_MODELS[component]), device_id=self.device_id,
            models_dir=self.paths.root,
        )

    def set_image(self, image: Image.Image | np.ndarray, state: dict[str, Any] | None = None) -> dict[str, Any]:
        if state is None:
            state = {}
        if isinstance(image, Image.Image):
            width, height = image.size
            rgb = np.array(image.convert("RGB"))
        else:
            height, width = image.shape[:2]
            rgb = image

        resized = np.asarray(Image.fromarray(rgb).resize(
            (self.resolution, self.resolution), Image.Resampling.BILINEAR
        ), dtype=np.float32) / UINT8_MAX
        norm_img = (resized - SAM3_NORM_MEAN) / SAM3_NORM_STD
        img_tensor = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[None, ...])

        v_out = self.v_engine.infer({"image": img_tensor}, device_outputs=True)
        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = v_out
        return state

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[SAM3] Running SAM3 with prompt: '%s' (confidence_threshold=%.4f)", prompt, self.confidence_threshold)
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before set_text_prompt")
        if self.tokenizer is None:
            raise RuntimeError("SAM3 vocabulary is missing; copy it from the export directory")

        tokens_np = self.tokenizer.tokenize([prompt], context_length=DEFAULT_CONTEXT_LENGTH)
        t_out = self.t_engine.infer({"input_ids": tokens_np}, device_outputs=True)

        v_out = state["backbone_out"]
        v_out.update(lang_feat=t_out["lang_feat"], lang_mask=t_out["lang_mask"])
        return self._decode(self.d_engine, state)

    def add_geometric_prompt(self, box: list[float], label: bool, state: dict[str, Any]) -> dict[str, Any]:
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before add_geometric_prompt")

        v_out = state["backbone_out"]
        lang_feat = v_out.get("lang_feat")
        lang_mask = v_out.get("lang_mask")
        if lang_feat is None or lang_mask is None:
            if self.tokenizer is None:
                raise RuntimeError("SAM3 vocabulary is missing; copy it from the export directory")
            dummy = self.tokenizer.tokenize(["visual"], context_length=DEFAULT_CONTEXT_LENGTH)
            t_out = self.t_engine.infer({"input_ids": dummy}, device_outputs=True)
            lang_feat, lang_mask = t_out["lang_feat"], t_out["lang_mask"]

        v_out.update(lang_feat=lang_feat, lang_mask=lang_mask)
        if self.b_engine is None:
            self.b_engine = self._load_engine("box_decoder")
        return self._decode(self.b_engine, state, {
            "prompt_boxes": np.asarray(box, dtype=np.float32).reshape(1, 1, 4),
            "prompt_labels": np.asarray([[label]], dtype=np.int64),
        })

    def _decode(
        self, engine: TRTEngine, state: dict[str, Any], prompts: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        backbone = state["backbone_out"]
        feed = {name: backbone[name] for name in engine.input_names if name in backbone}
        feed.update(prompts or {})
        boxes, scores, masks = self._postprocess(
            engine.infer(feed), state["original_height"], state["original_width"]
        )
        state.update(boxes=boxes, scores=scores, masks=masks[:, None, :, :])
        return state

    def _postprocess(self, d_out: dict[str, np.ndarray], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_boxes = d_out["pred_boxes"][0]
        pred_logits = d_out["pred_logits"][0]
        pred_masks = d_out["pred_masks"][0]

        scores = _sigmoid(pred_logits.reshape(-1))
        presence = _sigmoid(d_out["presence_logit"].reshape(-1)[0])
        scores *= presence
        top_5_scores = sorted([f"{s:.4f}" for s in scores], reverse=True)[:5]
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        logger.info("[SAM3 Postprocess] BEFORE threshold filtering: count=%d, presence=%.4f, max_score=%.4f, top_5=%s", len(scores), presence, max_score, top_5_scores)

        valid = scores > self.confidence_threshold
        if not np.any(valid):
            logger.info("[SAM3 Postprocess] AFTER threshold filtering (threshold=%.4f): 0 / %d proposals survived (max raw score was %.4f)", self.confidence_threshold, len(scores), max_score)
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, height, width), dtype=bool),
            )

        valid_boxes = pred_boxes[valid]
        valid_scores = scores[valid]
        valid_masks = pred_masks[valid]
        surviving_scores = [f"{s:.4f}" for s in valid_scores]
        logger.info("[SAM3 Postprocess] AFTER threshold filtering (threshold=%.4f): %d / %d proposals survived. Surviving scores: %s", self.confidence_threshold, len(valid_scores), len(scores), surviving_scores)

        xyxy = np.zeros_like(valid_boxes)
        xyxy[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0) * width
        xyxy[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0) * height
        xyxy[:, 2] = (valid_boxes[:, 0] + valid_boxes[:, 2] / 2.0) * width
        xyxy[:, 3] = (valid_boxes[:, 1] + valid_boxes[:, 3] / 2.0) * height

        out_masks = np.zeros((len(valid_masks), height, width), dtype=bool)
        for i, m in enumerate(valid_masks):
            # Interpolate logits before sigmoid, as upstream does. sigmoid(x) > .5 iff x > 0.
            out_masks[i] = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR) > 0

        return xyxy, valid_scores, out_masks

    def release(self) -> None:
        self.v_engine.release()
        self.t_engine.release()
        self.d_engine.release()
        if self.b_engine is not None:
            self.b_engine.release()


__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_SAM3_IMAGE_SIZE",
    "Sam3TrtProcessor",
]
