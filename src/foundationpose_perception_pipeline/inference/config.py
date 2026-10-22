#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage configuration for the inference path.

**Distinct from `foundationpose_perception_pipeline.config`**, which loads YAML profiles and tuned defaults
from disk. These are the small, immutable objects a stage actually needs at call time. The
former answers "what are the defaults for this deployment"; the latter answers "what is this
one stage configured to do right now".

The problem they solve: every stage used to read attributes straight off an
`argparse.Namespace`. That coupled the algorithms to *the command line*, with three costs that
only get worse as options multiply --

- adding an option meant editing every entry point that constructs the namespace, and forgetting
  one produced an `AttributeError` at the first call rather than at start-up;
- a library caller -- a service, a notebook, a test -- had to fabricate a Namespace carrying
  fields it did not care about, just to call one function;
- nothing declared which of the ~45 CLI flags a given stage actually read, so the blast radius
  of changing one was unknowable without grepping.

Each config is frozen, so a stage cannot mutate what it was handed, and each carries a
`from_args` builder so entry points keep parsing a CLI exactly as before. Adding a knob is now
one field plus one line in one builder, and the type says which stage owns it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    """SAM3 proposal generation."""

    confidence_threshold: float
    resolution: int = 1008

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DetectionConfig:
        """Build from a parsed CLI namespace."""
        return cls(
            confidence_threshold=float(args.confidence_threshold),
            resolution=int(getattr(args, "resolution", 1008)),
        )


@dataclass(frozen=True)
class PoseConfig:
    """FoundationPose hypothesis generation and refinement."""

    n_refine: int
    n_hypotheses: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PoseConfig:
        """Build from a parsed CLI namespace."""
        return cls(n_refine=int(args.fp_n_refine), n_hypotheses=int(args.fp_n_hypotheses))


@dataclass(frozen=True)
class RefinementConfig:
    """CAD box-prompt refinement: which proposals get a second SAM3 pass, and how they merge.

    `low_miou`/`high_miou` bound the band that gets replaced. Proposals below it are too wrong
    to rescue and above it are already good, so re-prompting either is cost without benefit.
    """

    policy: str
    low_miou: float
    high_miou: float
    nms_threshold: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RefinementConfig:
        """Build from a parsed CLI namespace."""
        return cls(
            policy=str(args.sam3_refinement_policy),
            low_miou=float(args.refinement_low_miou),
            high_miou=float(args.refinement_high_miou),
            nms_threshold=float(args.refinement_nms_threshold),
        )


@dataclass(frozen=True)
class SelectionConfig:
    """Proposal reranking and the keep/drop cutoff.

    Every weight here multiplies a quantity inference produced about itself -- SAM3's own score
    and the IoUs between a proposal's mask and a CAD render of *its own predicted pose*. None of
    it touches ground truth, which is why this policy is deployable rather than an evaluation
    artefact.
    """

    policy: str
    cutoff: float
    weight_sam_score: float = 2.0
    weight_render_mask_iou: float = 4.0
    weight_render_box_iou: float = 1.0
    fp_score_divisor: float = 100.0

    def __post_init__(self) -> None:
        """Reject a divisor of zero at construction rather than at the first proposal."""
        if self.fp_score_divisor == 0:
            raise ValueError("fp_score_divisor must be non-zero")

    def formula_text(self) -> str:
        """Human-readable form of the rerank score, for reports and provenance."""
        return (
            f"{self.weight_sam_score:g} * sam_score"
            f" + {self.weight_render_mask_iou:g} * render_mask_iou"
            f" + {self.weight_render_box_iou:g} * render_box_iou"
            f" - fp_score / {self.fp_score_divisor:g}"
        )

    def score(self, *, sam_score: float, render_mask_iou: float, render_box_iou: float, fp_score: float) -> float:
        """Evaluate the rerank score for one proposal."""
        return (
            self.weight_sam_score * sam_score
            + self.weight_render_mask_iou * render_mask_iou
            + self.weight_render_box_iou * render_box_iou
            - fp_score / self.fp_score_divisor
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SelectionConfig:
        """Build from a parsed CLI namespace."""
        return cls(
            policy=str(args.proposal_selection_policy),
            cutoff=float(args.rerank_cutoff),
            weight_sam_score=float(args.rerank_weight_sam_score),
            weight_render_mask_iou=float(args.rerank_weight_render_mask_iou),
            weight_render_box_iou=float(args.rerank_weight_render_box_iou),
            fp_score_divisor=float(args.rerank_fp_score_divisor),
        )


@dataclass(frozen=True)
class InferenceConfig:
    """The four stage configs, built once per run.

    Bundled so an entry point hands the estimator one object instead of four, while each stage
    still receives only its own.
    """

    detection: DetectionConfig
    pose: PoseConfig
    refinement: RefinementConfig
    selection: SelectionConfig

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> InferenceConfig:
        """Build every stage config from one parsed CLI namespace."""
        return cls(
            detection=DetectionConfig.from_args(args),
            pose=PoseConfig.from_args(args),
            refinement=RefinementConfig.from_args(args),
            selection=SelectionConfig.from_args(args),
        )
