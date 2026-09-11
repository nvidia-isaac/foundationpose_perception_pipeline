#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score a finished inference run against ground truth.

Consumes what `infer.py` wrote -- `predictions.jsonl`, the mask sidecars and `depth_m.npy` --
and produces the detection, pose and depth summaries plus `report.md`. **No model is loaded and
no GPU work is done beyond ground-truth rasterization**, which the `gt_cache` usually serves
from disk.

That is the payoff of the split. Re-scoring at a different IoU threshold, visibility band or
rerank cutoff used to mean re-running SAM3 and FoundationPose -- hours for a sweep -- for
arithmetic that takes seconds. Because `predictions.jsonl` keeps *every* proposal with its
scores rather than only the survivors, `--rerank-cutoff` can be re-applied here too.

The cutoff actually applied is recorded in the report, because once it can differ from the one
inference used, "what the pipeline would deploy" and "what was scored" can diverge, and a
number without its operating point is not interpretable.

Usage:
    ./.venv/bin/python script/evaluate.py --dataset ${your_dataset} --run output/infer_dataset
    ./.venv/bin/python script/evaluate.py --dataset ${your_dataset} --run output/infer_dataset \
        --rerank-cutoff 5.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from foundationpose_perception_pipeline.config import (  # noqa: E402
    NO_REFINEMENT_POLICY,
    REPLACE_MID_NMS06_REFINEMENT_POLICY,
    SOFT_GLOBAL_RERANK_POLICY,
    add_config_argument,
    settings_from_argv,
)
from foundationpose_perception_pipeline.dataset import Target  # noqa: E402
from foundationpose_perception_pipeline.evaluation import (  # noqa: E402
    render_target_gt,
    score_detection,
    score_matched_poses,
)
from foundationpose_perception_pipeline.evaluation.depth_error import (  # noqa: E402
    compare_depths,
    load_collected_depth_m,
    scene_object_mask,
)
from foundationpose_perception_pipeline.evaluation.detection import (  # noqa: E402
    PoseMetricRegistry,
    accumulate_detection,
    compact_metrics,
    detection_template,
)
from foundationpose_perception_pipeline.evaluation.gt import GroundTruthRenderer  # noqa: E402
from foundationpose_perception_pipeline.evaluation.report import (  # noqa: E402
    PipelineReportConfig,
    write_pipeline_outputs,
)


class _PoseOnlyResult:
    """The single field `score_matched_poses` reads from a FoundationPose result.

    Reconstructed from `predictions.jsonl` rather than re-running the model. Deliberately
    carries nothing else: if the pose scorer ever starts reading another field, this fails
    loudly instead of silently scoring against a default.
    """

    __slots__ = ("pose_row_major",)

    def __init__(self, pose_row_major: list[float] | None) -> None:
        self.pose_row_major = pose_row_major


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation."""
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--run", type=Path, required=True, help="Directory holding predictions.jsonl.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --run.")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--pose-match-threshold", type=float, default=0.5)
    parser.add_argument("--min-visible-fraction", type=float, default=settings.ground_truth.min_visible_fraction)
    parser.add_argument(
        "--pose-min-visible-fraction", type=float, nargs="+", default=list(settings.pose.min_visible_fractions)
    )
    parser.add_argument("--gt-cache-root", type=Path, default=settings.dataset.gt_cache_root)
    parser.add_argument("--no-gt-cache", action="store_true")
    parser.add_argument(
        "--models-subdir",
        default=settings.dataset.models_subdir,
        help="Meshes the pose stage used. Only needed here so GT rendering matches inference.",
    )
    parser.add_argument(
        "--models-eval-subdir",
        default=settings.dataset.models_eval_subdir,
        help="Meshes the POSE METRICS are computed over, plus the models_info.json supplying "
             "diameter and symmetries. BOP ships a decimated copy in `models_eval` so that "
             "averaging a distance over its vertices is meaningful.",
    )
    parser.add_argument("--split", default=settings.dataset.split)
    # From the resolved profile, not the module-level defaults. These decide the report's PASS/FAIL
    # line, so reading them off `defaults.yaml` while the report states the profile's values is a
    # verdict computed against a number the reader was told was not in use.
    parser.add_argument("--max-vertex-error-threshold-mm", type=float,
                        default=settings.pose.max_vertex_error_threshold_mm)
    parser.add_argument("--max-vertex-error-required-rate", type=float,
                        default=settings.pose.max_vertex_error_required_rate)
    parser.add_argument(
        "--collected-depth-filename",
        default=settings.dataset.depth_filename,
        help="Collected-depth file to score against, inside each scene directory. Defaults to "
             "the profile's base_camera_id, so predicted depth and the map it is compared with "
             "describe the same camera.",
    )
    parser.add_argument(
        "--collected-root",
        type=Path,
        default=settings.dataset.collected_depth_root,
        help="Collected cam0 depth, used only to score predicted depth. Skipped when absent or "
             "with --no-depth-metrics -- detection and pose scoring do not need it.",
    )
    parser.add_argument("--no-depth-metrics", action="store_true", help="Skip depth scoring entirely.")
    parser.add_argument(
        "--rerank-cutoff",
        type=float,
        default=None,
        help="Re-apply the selection cutoff at scoring time. Omit to score the selection "
             "inference already made. Every proposal and its score is in predictions.jsonl, so "
             "this needs no GPU.",
    )
    return parser.parse_args()


def load_masks(path: str | None) -> np.ndarray | None:
    """Load a mask sidecar written by inference."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return np.load(file_path)["masks"].astype(bool)


def main() -> None:
    """CLI entry point."""
    run_evaluation(parse_args())


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Score a run directory and write every summary plus the Markdown report.

    Split from `main` for the same reason as `infer.run_inference`. Returns the summaries so a
    caller can report on them without re-reading the files just written.
    """
    settings = settings_from_argv()
    run_dir = args.run.expanduser().resolve()
    output_dir = (args.output_dir or run_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise SystemExit(f"No predictions.jsonl in {run_dir}. Run script/infer.py first.")

    dataset_dir = args.dataset_root / args.dataset
    gt_renderer = GroundTruthRenderer(args.dataset_root, args.split, args.models_subdir)
    pose_metric_registry = PoseMetricRegistry(args.dataset_root, args.models_eval_subdir)

    raw_counts = detection_template(args.iou_thresholds)
    pose_input_counts = detection_template(args.iou_thresholds)
    filtered_counts = detection_template(args.iou_thresholds)
    raw_by_object: dict[str, Any] = defaultdict(lambda: detection_template(args.iou_thresholds))
    pose_input_by_object: dict[str, Any] = defaultdict(lambda: detection_template(args.iou_thresholds))
    filtered_by_object: dict[str, Any] = defaultdict(lambda: detection_template(args.iou_thresholds))
    pose_matches_all: list[dict[str, Any]] = []
    pose_matches_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prompts_used: dict[str, str] = {}
    scene_ids: list[int] = []
    # The evaluation-side mirror of `predictions.jsonl`: metrics only, joined on `target_key`.
    # Two artifacts rather than one combined file, because each is then produced by exactly one
    # half of the pipeline -- which is the point of the split.
    evaluation_records: list[dict[str, Any]] = []
    # Scoring cost per scene, summed over that scene's targets and its depth comparison. Reported
    # rather than folded into `runtime_sec`: GT rasterization is real work but it is not work a
    # deployment does, and the runtime figure is meant to describe the production path. `infer.py`
    # cannot measure this -- it never scores -- so the field it leaves at 0.0 is filled here.
    scoring_sec_by_scene: dict[int, float] = defaultdict(float)

    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scene_id = int(record["scene_id"])
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)
        target_scoring_started = time.perf_counter()
        object_name = record["object_name"]
        prompts_used[object_name] = record["prompt"]
        target = Target(
            dataset=record["dataset"],
            scene_id=scene_id,
            im_id=int(record["im_id"]),
            obj_id=int(record["obj_id"]),
            # Ground-truth instance count. It exists on `Target` for reporting only; nothing in
            # scoring reads it, and inference never had it.
            inst_count=0,
        )
        raw_masks = load_masks(record.get("raw_mask_path"))
        pose_input_masks = load_masks(record.get("pose_input_mask_path"))
        if raw_masks is None or pose_input_masks is None:
            raise SystemExit(f"Missing mask sidecars for {record['target_key']}; cannot score.")
        # A target that produced ZERO proposals must still be scored: its ground-truth
        # instances are false negatives, and skipping it removes them from the denominator and
        # inflates recall. That needs a frame size the masks cannot supply, hence
        # `image_size_wh` in the artifact; the fallback keeps older runs readable.
        if record.get("image_size_wh"):
            image_size_wh = (int(record["image_size_wh"][0]), int(record["image_size_wh"][1]))
        elif raw_masks.size:
            image_size_wh = (int(raw_masks.shape[2]), int(raw_masks.shape[1]))
        else:
            raise SystemExit(
                f"{record['target_key']} has no proposals and no image_size_wh; cannot score it. "
                "Re-run inference to record the frame size."
            )
        empty = np.zeros((0, image_size_wh[1], image_size_wh[0]), dtype=bool)
        if not raw_masks.size:
            raw_masks = empty
        if not pose_input_masks.size:
            pose_input_masks = empty

        gt_entries, gt_masks, gt_boxes, _camera_matrix, gt_visible_fractions = render_target_gt(
            gt_renderer=gt_renderer,
            target=target,
            image_size_wh=image_size_wh,
            gt_cache_root=args.gt_cache_root,
            use_gt_cache=not args.no_gt_cache,
            min_visible_fraction=args.min_visible_fraction,
        )

        # Re-apply the cutoff if asked, otherwise honour the selection inference made.
        proposals = record["proposals"]
        if args.rerank_cutoff is None:
            kept_indices = list(record["kept_indices"])
        else:
            kept_indices = [
                int(p["pose_input_index"])
                for p in proposals
                if p.get("rerank_score") is not None and float(p["rerank_score"]) >= args.rerank_cutoff
            ]
        kept_masks = pose_input_masks[kept_indices] if kept_indices else np.zeros((0, *pose_input_masks.shape[1:]), bool)
        pose_input_boxes = np.asarray(record["pose_input_boxes_xyxy"], dtype=np.float64)
        kept_boxes = pose_input_boxes[kept_indices] if kept_indices else np.zeros((0, 4), dtype=np.float64)

        raw_box_metrics, raw_mask_metrics = score_detection(
            boxes=np.asarray(record["boxes_xyxy"], dtype=np.float64), masks=raw_masks,
            gt_boxes=gt_boxes, gt_masks=gt_masks, iou_thresholds=args.iou_thresholds,
        )
        pi_box_metrics, pi_mask_metrics = score_detection(
            boxes=pose_input_boxes, masks=pose_input_masks,
            gt_boxes=gt_boxes, gt_masks=gt_masks, iou_thresholds=args.iou_thresholds,
        )
        f_box_metrics, f_mask_metrics = score_detection(
            boxes=kept_boxes, masks=kept_masks,
            gt_boxes=gt_boxes, gt_masks=gt_masks, iou_thresholds=args.iou_thresholds,
        )

        threshold_key = str(args.pose_match_threshold)
        if threshold_key not in f_mask_metrics:
            raise SystemExit(
                f"--pose-match-threshold {args.pose_match_threshold} is not in --iou-thresholds {args.iou_thresholds}"
            )
        filter_results = [_PoseOnlyResult(p.get("pose_row_major")) for p in proposals]
        matches = score_matched_poses(
            matches=f_mask_metrics[threshold_key]["matches"],
            kept_indices=kept_indices,
            pose_input_source_indices=np.asarray(record["pose_input_source_indices"], dtype=np.int64),
            filter_results=filter_results,
            gt_entries=gt_entries,
            gt_visible_fractions=gt_visible_fractions,
            pose_metric_registry=pose_metric_registry,
            dataset=record["dataset"],
            obj_id=int(record["obj_id"]),
        )

        for counts, by_object, box_metrics, mask_metrics in (
            (raw_counts, raw_by_object, raw_box_metrics, raw_mask_metrics),
            (pose_input_counts, pose_input_by_object, pi_box_metrics, pi_mask_metrics),
            (filtered_counts, filtered_by_object, f_box_metrics, f_mask_metrics),
        ):
            payload = {"box_metrics": compact_metrics(box_metrics), "mask_metrics": compact_metrics(mask_metrics)}
            accumulate_detection(counts, payload)
            accumulate_detection(by_object[object_name], payload)

        match_dicts = [asdict(match) | {"object_name": object_name} for match in matches]
        pose_matches_all.extend(match_dicts)
        pose_matches_by_object[object_name].extend(match_dicts)

        evaluation_records.append(
            {
                "target_key": record["target_key"],
                "dataset": record["dataset"],
                "scene_id": scene_id,
                "im_id": int(record["im_id"]),
                "obj_id": int(record["obj_id"]),
                "object_name": object_name,
                # Echoed from the prediction record: the aggregate report builds its prompt map
                # from these rows, and joining two artifacts for one string is not worth it.
                "prompt": record["prompt"],
                "gt_inst_count": len(gt_entries),
                "visible_gt_inst_count": len(gt_masks),
                "gt_visible_fractions": [round(float(value), 4) for value in gt_visible_fractions],
                "min_visible_fraction": args.min_visible_fraction,
                "pose_match_threshold": args.pose_match_threshold,
                "scored_kept_indices": list(kept_indices),
                "raw_box_metrics": compact_metrics(raw_box_metrics),
                "raw_mask_metrics": compact_metrics(raw_mask_metrics),
                "pose_input_box_metrics": compact_metrics(pi_box_metrics),
                "pose_input_mask_metrics": compact_metrics(pi_mask_metrics),
                "filtered_box_metrics": compact_metrics(f_box_metrics),
                "filtered_mask_metrics": compact_metrics(f_mask_metrics),
                "matched_pose_metrics": [asdict(match) for match in matches],
            }
        )
        scoring_sec_by_scene[scene_id] += time.perf_counter() - target_scoring_started

    # Depth scoring is per SCENE, not per target, so it runs after the record loop rather than
    # inside it -- one comparison per scene instead of one per object class in it.
    depth_rows: list[dict[str, Any]] = []
    # None when no collected-depth root is configured. Treated exactly like a configured root
    # whose directory is absent -- depth scoring is skipped and the summaries are written empty,
    # which is what `--no-depth-metrics` does too.
    collected_dataset_dir = args.collected_root / args.dataset if args.collected_root else None
    depth_metrics_enabled = (
        not args.no_depth_metrics
        and collected_dataset_dir is not None
        and collected_dataset_dir.exists()
    )
    # SAY WHY, once, when depth is not scored. Skipping is a normal outcome -- three of the four
    # reasons below are configuration rather than error -- but the artifacts it produces are an
    # empty `depth_summary` and a report row of `n/a`, which look identical whichever reason
    # applied and identical to a run that measured nothing. Without this line the only way to
    # tell "I asked it not to" from "it could not find the tree" is to re-read the command.
    if not depth_metrics_enabled:
        if args.no_depth_metrics:
            why = "--no-depth-metrics was passed"
        elif collected_dataset_dir is None:
            why = ("no collected-depth root is configured: set `dataset.collected_depth_root` in "
                   "the config profile, or pass --collected-root")
        else:
            why = f"no collected depth at {collected_dataset_dir}"
        print(f"depth metrics skipped: {why}. depth_summary will be empty and the report's depth "
              f"table will read n/a.")
    if depth_metrics_enabled:
        for scene_id in sorted(scene_ids):
            depth_scoring_started = time.perf_counter()
            depth_path = run_dir / "depth" / f"{scene_id:06d}" / "depth_m.npy"
            collected_scene_dir = collected_dataset_dir / args.split / f"{scene_id:06d}"
            if not depth_path.exists() or not collected_scene_dir.exists():
                missing = "predicted depth" if not depth_path.exists() else "collected depth"
                print(f"  scene {scene_id:06d}: no {missing}, not scored for depth")
                continue
            estimated = np.load(depth_path)
            collected = load_collected_depth_m(collected_scene_dir, args.collected_depth_filename)
            object_mask = scene_object_mask(
                dataset_dir=dataset_dir,
                scene_id=scene_id,
                image_size_wh=(collected.shape[1], collected.shape[0]),
                gt_renderer=gt_renderer,
                gt_cache_root=args.gt_cache_root,
                use_gt_cache=not args.no_gt_cache,
                min_visible_fraction=args.min_visible_fraction,
                split=args.split,
            )
            row = compare_depths(estimated, collected, object_mask=object_mask)
            metadata_path = run_dir / "depth" / f"{scene_id:06d}" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            right_camera = metadata.get("right_camera")
            row.update(
                {
                    "scene_id": scene_id,
                    "depth_source": "foundationstereo",
                    "comparison_mode": row["comparison_mode"],
                    "left_camera": int(metadata.get("left_camera", 0)),
                    "right_camera": right_camera,
                    "selected_pair": (
                        sorted([int(metadata.get("left_camera", 0)), int(right_camera)])
                        if right_camera is not None
                        else None
                    ),
                    "baseline_m": metadata.get("baseline_m"),
                    "stereo_axis": "horizontal" if right_camera is not None else None,
                    "estimated_depth_shape_hw": list(estimated.shape),
                    "collected_depth_shape_hw": list(collected.shape),
                }
            )
            depth_rows.append(row)
            scoring_sec_by_scene[scene_id] += time.perf_counter() - depth_scoring_started

    # What the run actually did, as recorded by inference. Preferred over the profile wherever
    # the two could disagree: `--rerank-weight-*` can override the parsed formula for a single
    # run, and a report that quoted the config string would then describe scoring that did not
    # happen. infer.py writes `rerank_formula` from the SelectionConfig it really used.
    inference_config: dict[str, Any] = {}
    inference_config_path = run_dir / "inference_config.json"
    if inference_config_path.exists():
        try:
            inference_config = json.loads(inference_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            inference_config = {}
    rerank_formula = inference_config.get("rerank_formula")
    rerank_formula_from_run = rerank_formula is not None
    if not rerank_formula_from_run:
        # Falling back to the profile is the very misdescription the preference above exists to
        # avoid, so it is announced rather than absorbed: a run dir with no recorded formula may
        # have been scored with `--rerank-weight-*` values this string does not describe.
        rerank_formula = settings.rerank.formula
        print(
            f"warning: no rerank_formula recorded in {inference_config_path} -- the report will "
            "quote the config profile, which does not reflect any --rerank-weight-* override "
            "used at inference time."
        )

    config = PipelineReportConfig(
        dataset=args.dataset,
        confidence_threshold=settings.detection.sam3_confidence_threshold,
        depth_source="foundationstereo",
        min_visible_fraction=args.min_visible_fraction,
        pose_match_threshold=args.pose_match_threshold,
        sam3_refinement_policy=settings.refinement.policy,
        proposal_selection_policy=settings.rerank.policy,
        rerank_cutoff=args.rerank_cutoff if args.rerank_cutoff is not None else settings.rerank.cutoff,
        rerank_formula=rerank_formula,
        rerank_formula_from_run=rerank_formula_from_run,
        prompt_source="predictions.jsonl (recorded at inference time)",
        soft_global_rerank_policy=SOFT_GLOBAL_RERANK_POLICY,
        replace_mid_nms06_refinement_policy=REPLACE_MID_NMS06_REFINEMENT_POLICY,
        default_sam3_refinement_policy=NO_REFINEMENT_POLICY,
        refinement_low_miou=settings.refinement.low_miou,
        refinement_high_miou=settings.refinement.high_miou,
        refinement_nms_threshold=settings.refinement.nms_threshold,
        max_vertex_error_threshold_mm=args.max_vertex_error_threshold_mm,
        max_vertex_error_required_rate=args.max_vertex_error_required_rate,
        base_camera_id=settings.dataset.base_camera_id,
        collected_depth_filename=args.collected_depth_filename,
        depth_metrics_enabled=depth_metrics_enabled,
    )
    # Per-scene timings are produced by infer.py in a separate process and handed over through
    # inference_config.json. It is not recoverable here -- these are wall-clock measurements taken 
    # as each stage ran, so re-scoring cannot reconstruct them. Re-run inference if the per-scene
    # table is wanted.
    runtime_rows: list[dict[str, Any]] = []
    if inference_config:
        runtime_rows = inference_config.get("runtime_by_scene") or []
    # Merge the scoring cost measured above into the rows inference handed over. `runtime_sec` is
    # left alone: it is the production path (depth + the per-target loop) and must not absorb work
    # a deployment never does.
    for row in runtime_rows:
        row["scoring_sec"] = round(scoring_sec_by_scene.get(int(row["scene_id"]), 0.0), 4)
    if not runtime_rows and scene_ids:
        print(
            f"warning: no per-scene runtime in {inference_config_path} -- "
            "runtime_summary.json will be empty and runtime_summary.csv will not be written."
        )

    summaries = write_pipeline_outputs(
        output_dir=output_dir,
        runtime_rows=runtime_rows,
        config=config,
        scene_ids=sorted(scene_ids),
        # Local date, deliberately: this stamps a human-facing report, and a reader comparing it
        # against when they ran the job expects their own calendar day, not UTC's.
        generated_on=date.today().isoformat(),  # noqa: DTZ011
        prompts_used=prompts_used,
        raw_detection_counts=raw_counts,
        pose_input_detection_counts=pose_input_counts,
        filtered_detection_counts=filtered_counts,
        raw_by_object=raw_by_object,
        pose_input_by_object=pose_input_by_object,
        filtered_by_object=filtered_by_object,
        pose_matches_all=pose_matches_all,
        pose_matches_by_object=pose_matches_by_object,
        depth_rows=depth_rows,
        pose_visibility_bands=args.pose_min_visible_fraction,
    )

    # Depth metrics are per scene; the per-target evaluation records carry the scene's row so a
    # consumer reading one target does not have to join against the depth summary.
    depth_by_scene = {int(row["scene_id"]): row for row in depth_rows}
    evaluations_path = output_dir / "evaluations.jsonl"
    with evaluations_path.open("w", encoding="utf-8") as evaluations_file:
        for evaluation in evaluation_records:
            evaluation["depth_metrics"] = depth_by_scene.get(int(evaluation["scene_id"]))
            evaluations_file.write(json.dumps(evaluation) + "\n")

    overall = summaries["pose_summary"]["overall"]
    applied = args.rerank_cutoff if args.rerank_cutoff is not None else "as inferred"
    print(f"scored {len(scene_ids)} scene(s) from {predictions_path}")
    print(f"rerank cutoff applied: {applied}")
    print(
        f"matched={overall['matched_predictions']}  "
        f"<=5mm={overall['max_vertex_error_within_threshold_rate']:.4f}  "
        f"median_max_vertex={overall['max_vertex_error_mm_median']:.4f}"
    )
    print(f"wrote {evaluations_path}")
    print(f"wrote {output_dir / 'report.md'}")
    return summaries


if __name__ == "__main__":
    main()
