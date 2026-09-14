#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load exported SAM3 TensorRT engines and check one synthetic image without a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_SYNTHETIC_SIZE: int = 1024
SYNTHETIC_BACKGROUND_COLOR: tuple[int, int, int] = (60, 60, 60)
SYNTHETIC_FOREGROUND_COLOR: tuple[int, int, int] = (230, 230, 230)
DEFAULT_VERIFY_PROMPT: str = "square"


def make_synthetic_image(size: int = DEFAULT_SYNTHETIC_SIZE) -> Image.Image:
    """Draw a simple shape on a plain background."""
    image = Image.new("RGB", (size, size), color=SYNTHETIC_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    margin = size // 4
    draw.rectangle([margin, margin, size - margin, size - margin], fill=SYNTHETIC_FOREGROUND_COLOR)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-models-dir", type=Path, default=None)
    parser.add_argument(
        "--prompt", default=DEFAULT_VERIFY_PROMPT, help="Text prompt to run against the synthetic image."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = make_synthetic_image()

    print("Using Pure TensorRT SAM 3 backend...")
    from foundationpose_perception_pipeline.inference.sam3.trt_processor import Sam3TrtProcessor

    processor = Sam3TrtProcessor(
        models_dir=args.sam3_models_dir,
        confidence_threshold=0.0,
        device=args.device,
    )
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt=args.prompt, state=state)
    boxes = state["boxes"]
    scores = state["scores"]
    masks = state["masks"][:, 0] if "masks" in state and state["masks"] is not None else np.empty((0,))

    max_score = scores.max() if len(scores) else 0.0
    print(f"prompt={args.prompt!r} -> {len(boxes)} proposal(s), max score={max_score:.3f}")

    if len(boxes) == 0 or masks.shape[0] != len(boxes) or masks.shape[-2:] != image.size[::-1]:
        raise SystemExit(
            "SAM3 ran but the output tensors look malformed (missing proposals, or "
            "boxes/masks count or mask resolution mismatch) -- install looks broken, not just "
            "a low-confidence prompt. Check the traceback-free run above for a real error."
        )

    print("SAM3 OK (TensorRT): forward pass ran, output tensors well-formed.")
    print(
        "(Low max score is expected here -- this is a synthetic doodle, not a natural image. "
        "Run the pipeline on a couple of real scenes to sanity-check detection quality: "
        "`script/run_pipeline.py --dataset <name> --max-scenes 2`.)"
    )


if __name__ == "__main__":
    main()
