#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one dataset end to end: depth -> SAM3 -> FoundationPose -> metrics.

This entry point is an **orchestrator**. It owns the combined CLI and the output layout, and
delegates the work to the two halves of the pipeline, in-process and in order:

    infer.run_inference       images + intrinsics -> depth, poses, predictions.jsonl
    evaluate.run_evaluation   predictions.jsonl  -> metrics, summaries, report.md

Use it when you want both halves with one command against an annotated capture. Use
`infer.py` alone for a capture with no ground truth, and `evaluate.py` alone to re-score a
finished run at a different threshold, band or rerank cutoff without touching a model.

Depth is predicted from the stereo pair. Prompts, thresholds and paths come from the config
profile -- see README.md.

Stage flow (per target = one (scene, obj_id) pair)::

    raw SAM3 proposals
       |
       +-- FoundationPose pass 1   run_foundationpose_for_proposals(masks=raw_masks)
       |     -> raw_filter_results: pose + score + rendered CAD box per proposal
       |
       +-- SAM3 refinement          apply_sam3_refinement(...)
       |     consumes the pass-1 poses: renders the CAD box at each estimated pose and
       |     re-prompts SAM3 with it, replacing proposals whose render-mask IoU falls in
       |     [0.2, 0.8], then mask-NMS at 0.6.  -> pose_input_masks
       |
       +-- FoundationPose pass 2   run_foundationpose_for_proposals(masks=pose_input_masks)
       |     -> filter_results: the poses that are actually reported
       |
       +-- rerank (soft_global_v1) -> kept_indices
             pure selection.  NO pose computation: it thresholds a score built from values
             already produced by pass 2 (sam_score, render_mask_iou, render_box_iou,
             fp_score).  Dropped proposals just have their existing poses discarded.

Pass 2 exists because refinement *replaces* masks, and a FoundationPose pose is only valid
for the mask it was registered from -- reusing a pass-1 pose with a pass-2 mask would be
incoherent.  With ``--sam3-refinement-policy none`` pass 2 is skipped entirely
(``filter_results = raw_filter_results``), so FoundationPose runs exactly once and the
rerank filters the pass-1 poses.  Refinement therefore roughly doubles runtime; the rerank
is nearly free.

Both passes are kept in ``predictions.jsonl`` as ``raw_proposal_pose_filter`` (pass 1) and
``proposal_pose_filter`` (pass 2).  All reported pose metrics come from pass 2, indexed via
``pred_index_pose_input``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foundationpose_perception_pipeline.config import (
    DEFAULT_RERANK_POLICY,
    DEFAULT_SAM3_REFINEMENT_POLICY,
    DEFAULT_SAM3_RESOLUTION,
    FOUNDATIONPOSE_ROOT_DEFAULT,
    KEEP_ALL_RERANK_POLICY,
    NO_REFINEMENT_POLICY,
    REPLACE_MID_NMS06_REFINEMENT_POLICY,
    SOFT_GLOBAL_RERANK_POLICY,
    add_config_argument,
    rerank_weights_from_formula,
    settings_from_argv,
)
from foundationpose_perception_pipeline.inference.depth import (
    add_backend_arguments,
    depth_backend_choices,
)
from foundationpose_perception_pipeline.inference.source import (
    add_source_arguments,
    depth_source_choices,
    registered_sources,
)
from foundationpose_perception_pipeline.pose import inject_external_paths

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
# The pipeline's own modules come from the installed `foundationpose_perception_pipeline` package, so only the
# external checkouts -- sam3 and FoundationPose -- still need injecting onto sys.path. This is
# the SHARED bootstrap rather than a copy of it: the same four steps were written out here and
# in `infer.py`, which is the arrangement `inject_external_paths` exists to prevent -- its
# docstring records a second entry point silently losing one of them. It also refuses to run
# without the checkout, and answers `--help` instead of refusing.
inject_external_paths(REPO_ROOT, FOUNDATIONPOSE_ROOT_DEFAULT)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SAM3, depth, and FoundationPose pipeline.

    The dataset profile is read first, from `--config`, because it supplies this parser's own
    path and threshold defaults -- so `--help` shows the values a bare run would actually use,
    and a profile's `overrides:` block reaches the CLI.
    """
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--split", default=settings.dataset.split)
    parser.add_argument(
        "--models-subdir",
        default=settings.dataset.models_subdir,
        help="Meshes the pose stage estimates and renders against. Standard BOP ships these in "
             "`models`; this project's captures use `models_cad`.",
    )
    parser.add_argument(
        "--models-eval-subdir",
        default=settings.dataset.models_eval_subdir,
        help="Meshes the pose METRICS are computed over. Standard BOP ships a decimated copy in "
             "`models_eval` for exactly this purpose; defaults to --models-subdir otherwise.",
    )
    parser.add_argument(
        "--collected-root",
        type=Path,
        default=settings.dataset.collected_depth_root,
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset folder under --dataset-root to evaluate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to `pipeline/output/<dataset>`.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=DEFAULT_SAM3_RESOLUTION)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=settings.detection.sam3_confidence_threshold,
    )
    parser.add_argument(
        "--depth-source",
        choices=depth_source_choices(),
        default=depth_source_choices()[0],
        help="Where FoundationPose's depth comes from. "
             + "; ".join(f"{s.name}: {s.describe}" for s in registered_sources().values()),
    )
    parser.add_argument(
        "--no-depth-metrics",
        action="store_true",
        help=(
            "Skip scoring predicted depth against the collected ground-truth depth. The "
            "collected tree is then not needed at all, so this is what lets a "
            "'--depth-source foundationstereo' run work on a machine that does not have it. "
            "Detection and pose metrics are unaffected -- they come from scene_gt.json and the "
            "GT cache. The depth table in the report reads 'n/a' rather than disappearing, so "
            "a run without the comparison cannot be mistaken for one that made it."
        ),
    )
    parser.add_argument(
        "--foundation-stereo-model",
        type=Path,
        default=settings.depth.engine,
        help="Depth model. Its path decides which registered backend runs it -- see "
             "--depth-backend, which asserts that choice rather than making it. Defaults to "
             "depth.engine from the config profile.",
    )
    parser.add_argument(
        "--min-working-distance-m",
        type=float,
        default=settings.depth.min_working_distance_m,
        help="Nearest surface the scene contains. With --max-working-distance-m this enables the "
             "disparity pre-shift and feasibility-aware partner ranking; see the depth block "
             "in config/defaults.yaml for what each costs.",
    )
    parser.add_argument(
        "--max-working-distance-m",
        type=float,
        default=settings.depth.max_working_distance_m,
        help="Farthest surface the scene contains. Both or neither -- a half-specified volume "
             "silently disables the pre-shift.",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=settings.depth.clahe_clip_limit,
        help="CLAHE clip limit for the rectified pair on the commercial backend; 0 disables it.",
    )
    parser.add_argument(
        "--clahe-detail-boost",
        type=float,
        default=settings.depth.clahe_detail_boost,
        help="Detail-layer gain for the bilateral split after CLAHE; 0 disables the split.",
    )
    parser.add_argument(
        "--depth-backend",
        choices=depth_backend_choices(),
        default="auto",
        help="Assert which backend this run uses. The model path decides; this only refuses to "
             "run if the two disagree, so 'which licence am I under' is answerable without "
             "reading a file extension.",
    )
    parser.add_argument(
        "--foundation-stereo-max-width",
        type=int,
        default=settings.depth.foundation_stereo_max_width,
        help="Rectified width fed to the model. Only affects the internal stereo resolution; "
             "the depth map is always emitted at full base-camera resolution.",
    )
    parser.add_argument(
        "--foundation-stereo-fixed-height",
        type=int,
        default=settings.depth.foundation_stereo_fixed_height,
        help="Fixed height for FoundationStereo input, or None to determine dynamically.",
    )
    # Whatever the registered backends and sources need on top of the flags above.
    add_backend_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument(
        "--pose-min-visible-fraction",
        type=float,
        nargs="+",
        default=list(settings.pose.min_visible_fractions),
        help=(
            "Visibility bands to report pose metrics at, e.g. `0.0 0.9`. Each band scores only "
            "matched instances at least that visible. Detection metrics and the GT population "
            "are unaffected -- use --min-visible-fraction to change those, but note that it "
            "turns detections of excluded instances into false positives."
        ),
    )
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=settings.ground_truth.min_visible_fraction,
        help=(
            "GT instances visible below this fraction of their full silhouette are excluded from "
            "detection and pose scoring. GT masks are occlusion-aware (shared z-buffer)."
        ),
    )
    parser.add_argument(
        "--gt-cache-root",
        type=Path,
        default=settings.dataset.gt_cache_root,
        help=(
            "Directory of precomputed per-scene GT z-buffers from script/build_gt_cache.py. Scenes without a "
            "cache entry fall back to rendering GT on the fly (slow: ~12-14s per target). Run "
            "`python script/build_gt_cache.py --dataset <name>` ahead of time to populate this."
        ),
    )
    parser.add_argument(
        "--no-gt-cache",
        action="store_true",
        help="Always render GT on the fly, ignoring --gt-cache-root even if populated.",
    )
    parser.add_argument(
        "--pose-match-threshold",
        type=float,
        default=0.5,
        help="Mask IoU threshold used to define matched pose predictions.",
    )
    parser.add_argument(
        "--proposal-render-mask-iou-threshold",
        type=float,
        default=0.5,
        help="Legacy diagnostic threshold recorded with each prediction; proposal filtering is not applied.",
    )
    parser.add_argument(
        "--proposal-render-box-iou-threshold",
        type=float,
        default=None,
        help="Legacy diagnostic threshold recorded with each prediction; proposal filtering is not applied.",
    )
    parser.add_argument(
        "--proposal-selection-policy",
        choices=[KEEP_ALL_RERANK_POLICY, SOFT_GLOBAL_RERANK_POLICY],
        default=DEFAULT_RERANK_POLICY,
        help=(
            "Optional post-pose proposal selection stage. "
            "`all` keeps every SAM3 proposal; "
            "`soft_global_v1` applies a single object-agnostic rerank rule."
        ),
    )
    parser.add_argument(
        "--sam3-refinement-policy",
        choices=[NO_REFINEMENT_POLICY, REPLACE_MID_NMS06_REFINEMENT_POLICY],
        default=DEFAULT_SAM3_REFINEMENT_POLICY,
        help=(
            "Optional CAD box-prompt refinement stage before the final FoundationPose/rerank pass. "
            "`none` uses raw SAM3 proposals; "
            "`replace_mid_nms06` replaces mid-overlap proposals with the best CAD box-prompted SAM3 mask "
            "and then applies mask NMS at 0.6."
        ),
    )
    parser.add_argument(
        "--refinement-low-miou",
        type=float,
        default=settings.refinement.low_miou,
        help="Lower render-mask-IoU bound of the band whose proposals get replaced.",
    )
    parser.add_argument(
        "--refinement-high-miou",
        type=float,
        default=settings.refinement.high_miou,
        help="Upper render-mask-IoU bound of the band whose proposals get replaced.",
    )
    parser.add_argument(
        "--refinement-nms-threshold",
        type=float,
        default=settings.refinement.nms_threshold,
        help="Mask-NMS IoU threshold applied after refinement replacement.",
    )
    parser.add_argument(
        "--rerank-cutoff",
        type=float,
        default=settings.rerank.cutoff,
        help="Keep proposals whose soft-global rerank score is at least this cutoff.",
    )
    # Parsed from the profile's `rerank.formula` rather than hardcoded here, so the string in the
    # config and the arithmetic that runs cannot drift apart. Override a single weight on the CLI
    # if you need to; the report records the weights actually applied, not the string.
    rerank_weights = rerank_weights_from_formula(settings.rerank.formula)
    parser.add_argument(
        "--rerank-weight-sam-score", type=float, default=rerank_weights["weight_sam_score"]
    )
    parser.add_argument(
        "--rerank-weight-render-mask-iou", type=float, default=rerank_weights["weight_render_mask_iou"]
    )
    parser.add_argument(
        "--rerank-weight-render-box-iou", type=float, default=rerank_weights["weight_render_box_iou"]
    )
    parser.add_argument(
        "--rerank-fp-score-divisor",
        type=float,
        default=rerank_weights["fp_score_divisor"],
        help=(
            "FoundationPose score term divisor for soft-global reranking. "
            "The score contribution is `-fp_score / rerank_fp_score_divisor`."
        ),
    )
    parser.add_argument("--foundationpose-root", type=Path, default=FOUNDATIONPOSE_ROOT_DEFAULT)
    parser.add_argument("--fp-library", type=Path, default=None)
    parser.add_argument("--fp-refine-model-path", type=Path, default=None)
    parser.add_argument("--fp-score-model-path", type=Path, default=None)
    parser.add_argument("--fp-engine-cache-dir", type=Path, default=None)
    parser.add_argument("--fp-device-id", type=int, default=0)
    parser.add_argument("--fp-prepare-batch", type=int, default=64)
    parser.add_argument("--fp-n-hypotheses", type=int, default=64)
    parser.add_argument("--fp-n-refine", type=int, default=3)
    parser.add_argument("--sam3-models-dir", type=Path, default=None, help="Directory containing exported SAM3 engines and vocabulary.")
    parser.add_argument("--overwrite-depth", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the full SAM3, GT-depth, and FoundationPose evaluation pipeline."""
    args = parse_args()
    if args.rerank_fp_score_divisor <= 0:
        raise ValueError("--rerank-fp-score-divisor must be > 0")

    # Resolve the dataset-specific input/output locations and fail early if the
    # required BOP RGB tree or collected depth tree is missing.
    if not args.dataset:
        raise SystemExit(
            "No dataset given. Pass --dataset <name> (a subfolder of "
            f"{args.dataset_root}), e.g. --dataset "
            f"{next((p.name for p in sorted(args.dataset_root.glob('*')) if p.is_dir()), '<name>')}."
        )
    dataset_dir = args.dataset_root / args.dataset
    # None when the profile sets no `collected_depth_root` and no --collected-root was passed.
    # That is a supported configuration -- a capture with no ground-truth depth runs with
    # --no-depth-metrics -- so this stays None here and is only rejected below if something is
    # actually going to read it.
    collected_dir = args.collected_root / args.dataset if args.collected_root else None
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")
    # Collected depth can have two consumers, and a run may want neither: it is what predicted
    # depth is scored against, and a source may also feed it to FoundationPose as the pose input.
    # Ask the source rather than testing for its name, and say which consumer is asking -- the
    # remedy differs.
    depth_source = registered_sources()[args.depth_source]
    depth_metrics_enabled = depth_source.depth_metrics_meaningful and not args.no_depth_metrics
    if depth_source.needs_collected_depth:
        reason = f"--depth-source {depth_source.name} feeds it to FoundationPose as the pose input"
    else:
        reason = ("predicted depth is scored against it; pass --no-depth-metrics to skip that "
                  "and run without it")
    if depth_metrics_enabled or depth_source.needs_collected_depth:
        if collected_dir is None:
            raise SystemExit(
                "No collected-depth root is configured, and this run needs one: "
                f"{reason}. Set `dataset.collected_depth_root` in the config profile, or pass "
                "--collected-root."
            )
        if not collected_dir.exists():
            raise FileNotFoundError(f"Missing collected-depth directory: {collected_dir} ({reason})")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (settings_from_argv().dataset.output_root / args.dataset).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_results:
        # Every artifact below is rewritten from scratch by the two halves anyway; deleting
        # them first is what keeps a *failed* rerun from leaving last run's numbers in place,
        # looking like they describe this one.
        for name in (
            "predictions.jsonl",
            "evaluations.jsonl",
            "inference_config.json",
            "runtime_summary.json",
            "runtime_summary.csv",
            "raw_detection_summary.json",
            "raw_detection_summary.csv",
            "pose_input_detection_summary.json",
            "pose_input_detection_summary.csv",
            "filtered_detection_summary.json",
            "filtered_detection_summary.csv",
            "selected_detection_summary.json",
            "selected_detection_summary.csv",
            "pose_summary.json",
            "pose_summary.csv",
            "depth_summary.json",
            "depth_summary.csv",
            "report.md",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()

    # ---------------------------------------------------------------------------------
    # Delegation. This entry point is now an ORCHESTRATOR: it owns the combined CLI and the
    # artifact layout, and hands the actual work to the two halves of the pipeline.
    #
    #   infer.run_inference    images + intrinsics -> depth, poses, predictions.jsonl
    #   evaluate.run_evaluation  predictions.jsonl -> metrics, summaries, report.md
    #
    # Called in-process rather than as subprocesses so there is one implementation and
    # exceptions propagate normally. The two passes are sequential: all inference completes
    # before any scoring starts, so SAM3 and FoundationPose are released before ground-truth
    # rasterization begins.
    # ---------------------------------------------------------------------------------
    sys.path.insert(0, str(PIPELINE_ROOT / "script"))
    from evaluate import run_evaluation
    from infer import run_inference

    inference_args = argparse.Namespace(**vars(args))
    inference_args.output_dir = output_dir
    # Objects come from each scene's ground truth here, matching this entry point's historical
    # behaviour; `infer.py --objects` is the deployment-shaped alternative.
    inference_args.objects = None
    inference_args.no_overlays = getattr(args, "no_overlays", False)
    run_inference(inference_args)

    evaluation_args = argparse.Namespace(**vars(args))
    evaluation_args.run = output_dir
    evaluation_args.output_dir = output_dir
    # Declared on this parser, so inference and evaluation cannot be pointed at different
    # meshes -- scoring vertices the pose stage never saw is silent and looks like model error.
    evaluation_args.models_subdir = args.models_subdir
    evaluation_args.models_eval_subdir = args.models_eval_subdir
    # None means "score the selection inference already made", rather than re-applying a
    # cutoff. Passing args.rerank_cutoff here would be a no-op in the normal case but would
    # silently re-select if the two ever diverged.
    evaluation_args.rerank_cutoff = None
    evaluation_args.no_depth_metrics = args.no_depth_metrics or not depth_source.depth_metrics_meaningful
    # Settings `evaluate.py` exposes as flags on its own parser and this one does not. They are
    # supplied from the resolved profile rather than left absent: this namespace is built from
    # THIS parser's arguments, so anything only the other parser declares is missing entirely, and
    # the failure is an AttributeError partway through scoring rather than at launch.
    settings = settings_from_argv()
    evaluation_args.max_vertex_error_threshold_mm = settings.pose.max_vertex_error_threshold_mm
    evaluation_args.max_vertex_error_required_rate = settings.pose.max_vertex_error_required_rate
    evaluation_args.collected_depth_filename = settings.dataset.depth_filename
    run_evaluation(evaluation_args)


if __name__ == "__main__":
    main()
