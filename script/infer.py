#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure inference: images and intrinsics in, poses out.

**No ground truth is read and no metric is computed.** This is the deployable path — it runs on
a capture that has `rgb/`, `scene_camera.json` and CAD meshes, and nothing else. Scoring is
`evaluate.py`'s job, over the `predictions.jsonl` written here.

That separation buys three things:

- **A capture with no annotations can be processed at all.** Any run that also scores needs
  `scene_gt.json` and the CAD meshes to rasterize it; this half needs neither, so a capture that
  has only images and intrinsics still produces poses.
- **Re-scoring costs seconds, not hours.** Changing an IoU threshold, a visibility band or a
  rerank cutoff re-reads this artifact instead of re-running SAM3 and FoundationPose.
- **The reported runtime is honest.** There is no scoring work to subtract.

Which objects to look for is a *task specification*, supplied with `--objects`. With no
`--objects` it falls back to the classes named in `scene_gt.json`, which is convenient on an
annotated dataset and is the only line here that touches ground truth -- it reads class names,
never poses.

Usage:
    ./.venv/bin/python script/infer.py --dataset ${your_dataset} --max-scenes 2 \
        --output-dir output/infer_dataset
    ./.venv/bin/python script/infer.py --dataset ${your_dataset} --output-dir output/infer_dataset \
        --objects ${object1_name} ${object2_name}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from foundationpose_perception_pipeline.config import (  # noqa: E402
    FOUNDATIONPOSE_ROOT_DEFAULT,
    KEEP_ALL_RERANK_POLICY,
    NO_REFINEMENT_POLICY,
    add_config_argument,
    rerank_weights_from_formula,
    settings_from_argv,
)
from foundationpose_perception_pipeline.dataset import image_path  # noqa: E402
from foundationpose_perception_pipeline.inference import (  # noqa: E402
    StagePredictions,
    TargetPrediction,
    proposals_from_stages,
)
from foundationpose_perception_pipeline.inference.depth import (  # noqa: E402
    add_backend_arguments,
    depth_backend_choices,
    resolve_backend,
)
from foundationpose_perception_pipeline.inference.engine import PoseEstimator  # noqa: E402
from foundationpose_perception_pipeline.inference.source import (  # noqa: E402
    add_source_arguments,
    depth_source_choices,
    registered_sources,
)
from foundationpose_perception_pipeline.io.bop import (  # noqa: E402
    dataset_scene_ids,
    object_specs_for_scene,
    object_specs_from_names,
    scene_camera_matrix,
)
from foundationpose_perception_pipeline.pose import (  # noqa: E402
    FoundationPoseRegistry,
    PoseRenderer,
    ensure_foundationpose_paths,
    inject_external_paths,
)

# External checkouts must be locatable before any model import; see `inject_external_paths`.
inject_external_paths(REPO_ROOT, FOUNDATIONPOSE_ROOT_DEFAULT)
from foundationpose_perception_pipeline.inference.config import InferenceConfig  # noqa: E402
from foundationpose_perception_pipeline.inference.detect import (  # noqa: E402
    base_text_state_from_prompt_state,
)
from foundationpose_perception_pipeline.inference.pose import run_foundationpose_for_proposals  # noqa: E402
from foundationpose_perception_pipeline.inference.refine import apply_sam3_refinement  # noqa: E402
from foundationpose_perception_pipeline.inference.select import (  # noqa: E402
    mark_selected_filter_results,
    select_proposals,
)
from foundationpose_perception_pipeline.runtime import inference_context, tensor_to_numpy  # noqa: E402
from foundationpose_perception_pipeline.visualize import (  # noqa: E402
    draw_overlay,
    draw_pose_filter_overlay,
    draw_scene_overlay,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference-only execution."""
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument(
        "--split",
        default=settings.dataset.split,
        help="BOP split directory under <dataset_root>/<dataset>/. From the profile's "
             "dataset.split; standard BOP trees also carry `train` and `val`.",
    )
    parser.add_argument(
        "--models-subdir",
        default=settings.dataset.models_subdir,
        help="Directory under <dataset_root>/<dataset>/ holding the CAD meshes. Standard BOP "
             "datasets ship these in `models`; this project's captures use `models_cad`.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Object names to look for. Omit to take the classes named in scene_gt.json, which "
             "is the only ground-truth read in this script.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=settings.detection.sam3_confidence_threshold)
    parser.add_argument("--sam3-refinement-policy", default=settings.refinement.policy)
    parser.add_argument("--refinement-low-miou", type=float, default=settings.refinement.low_miou)
    parser.add_argument("--refinement-high-miou", type=float, default=settings.refinement.high_miou)
    parser.add_argument("--refinement-nms-threshold", type=float, default=settings.refinement.nms_threshold)
    parser.add_argument("--proposal-selection-policy", default=settings.rerank.policy)
    parser.add_argument("--rerank-cutoff", type=float, default=settings.rerank.cutoff)
    # Weights come from the profile's `rerank.formula`; see run_pipeline.py for the reasoning.
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
        "--rerank-fp-score-divisor", type=float, default=rerank_weights["fp_score_divisor"]
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
    parser.add_argument("--foundation-stereo-model", type=Path, default=settings.depth.engine)
    parser.add_argument("--foundation-stereo-max-width", type=int, default=settings.depth.foundation_stereo_max_width)
    parser.add_argument("--depth-backend", choices=depth_backend_choices(), default="auto")
    parser.add_argument("--min-working-distance-m", type=float, default=settings.depth.min_working_distance_m)
    parser.add_argument("--max-working-distance-m", type=float, default=settings.depth.max_working_distance_m)
    parser.add_argument("--clahe-clip-limit", type=float, default=settings.depth.clahe_clip_limit)
    parser.add_argument("--clahe-detail-boost", type=float, default=settings.depth.clahe_detail_boost)
    parser.add_argument("--overwrite-depth", action="store_true")
    # Choices come from the registry, so a source registered by an extension appears here without
    # this file knowing it exists. With nothing registered there is one choice, and the flag is
    # then just provenance recorded in the artifacts.
    parser.add_argument(
        "--depth-source",
        choices=depth_source_choices(),
        default=depth_source_choices()[0],
        help="Where FoundationPose's depth comes from. "
             + "; ".join(f"{s.name}: {s.describe}" for s in registered_sources().values()),
    )
    # Whatever the registered backends and sources need on top of the flags above.
    add_backend_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--collected-root", type=Path, default=settings.dataset.collected_depth_root)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip diagnostic overlay images. They need no ground truth -- they draw proposals "
             "and keep/drop decisions -- but they are pure diagnostics and cost wall-clock.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run_inference(parse_args())


def run_inference(args: argparse.Namespace) -> Path:
    """Run detection, depth, pose and selection over a dataset, writing predictions only.

    Split from `main` so `run_pipeline.py` can drive the same code in-process rather than
    shelling out -- one implementation, no argv round-trip. Returns the run directory.
    """
    settings = settings_from_argv()
    dataset_dir = args.dataset_root / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = dataset_scene_ids(dataset_dir, args.split)
    if args.max_scenes is not None:
        scene_ids = scene_ids[: args.max_scenes]

    # `run_pipeline` uses an empty override map too; prompts come from the profile, with the
    # object name as the fallback. Kept as a variable rather than inlined so a future
    # `--prompt-overrides` flag has an obvious home.
    prompt_overrides: dict[str, str] = {}
    explicit_specs = (
        object_specs_from_names(
            dataset_dir=dataset_dir,
            object_names=args.objects,
            prompt_overrides=prompt_overrides,
            prompts=settings.dataset.prompts,
        )
        if args.objects
        else None
    )

    ensure_foundationpose_paths(args)
    pose_renderer = PoseRenderer(args.dataset_root, args.models_subdir)
    pose_registry = FoundationPoseRegistry(
        engine_cache_dir=(args.fp_engine_cache_dir or (output_dir / "foundationpose_engine_cache")).resolve(),
        refine_model_path=(
            args.fp_refine_model_path.expanduser().resolve()
            if args.fp_refine_model_path is not None
            else (args.foundationpose_root / "weights" / "refiner_net.onnx").resolve()
        ),
        score_model_path=(
            args.fp_score_model_path.expanduser().resolve()
            if args.fp_score_model_path is not None
            else (args.foundationpose_root / "weights" / "score_net.onnx").resolve()
        ),
        models_subdir=args.models_subdir,
        device_id=args.fp_device_id,
        prepare_batch=args.fp_prepare_batch,
    )

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
        load_from_HF=not args.no_hf,
    )
    processor = Sam3Processor(
        model, resolution=args.resolution, device=args.device, confidence_threshold=args.confidence_threshold
    )
    # Built once from the CLI; each stage then receives only its own section.
    inference_config = InferenceConfig.from_args(args)
    pose_estimator = PoseEstimator(
        processor=processor,
        pose_registry=pose_registry,
        pose_renderer=pose_renderer,
        dataset_root=args.dataset_root,
        device=args.device,
        inference_context=inference_context,
        run_foundationpose_for_proposals=run_foundationpose_for_proposals,
        apply_sam3_refinement=apply_sam3_refinement,
        select_proposals=select_proposals,
        mark_selected_filter_results=mark_selected_filter_results,
        base_text_state_from_prompt_state=base_text_state_from_prompt_state,
        tensor_to_numpy=tensor_to_numpy,
        torch=torch,
        no_refinement_policy=NO_REFINEMENT_POLICY,
    )

    # Resolved once, before any scene: which depth backend this run uses is a property of the
    # run, and `--depth-backend` exists so a contradiction fails here rather than being
    # discovered in the metrics.
    depth_source = registered_sources()[args.depth_source]
    if depth_source.uses_depth_backend:
        depth_backend = resolve_backend(args.foundation_stereo_model, getattr(args, "depth_backend", "auto"))
        print(
            f"depth backend: {depth_backend}"
            + (f" ({args.foundation_stereo_model})" if args.foundation_stereo_model else ""),
            flush=True,
        )

    predictions_path = output_dir / "predictions.jsonl"
    runtime_rows: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for scene_id in tqdm(scene_ids, desc=f"{args.dataset} inference"):
            scene_started = time.perf_counter()
            scene_dir = dataset_dir / args.split / f"{scene_id:06d}"
            depth_dir = output_dir / "depth" / f"{scene_id:06d}"
            depth_started = time.perf_counter()
            depth_m = depth_source.provide(
                scene_dir=scene_dir,
                depth_dir=depth_dir,
                args=args,
                dataset=args.dataset,
                scene_id=scene_id,
            )
            depth_sec = time.perf_counter() - depth_started
            camera_matrix = scene_camera_matrix(dataset_dir, scene_id, split=args.split)
            depth_result = {"depth_m": depth_m.astype(np.float32), "camera_matrix": camera_matrix}

            image = Image.open(image_path(dataset_dir, scene_id, 0, args.split)).convert("RGB")
            depth_h, depth_w = depth_m.shape
            depth_image_size = (depth_w, depth_h)
            pose_image_np = np.asarray(image.resize(depth_image_size, Image.Resampling.BILINEAR), dtype=np.uint8)
            image_state = pose_estimator.begin_scene(image, depth_image_size)

            specs = explicit_specs or object_specs_for_scene(
                dataset_dir=dataset_dir,
                scene_id=scene_id,
                prompt_overrides=prompt_overrides,
                prompts=settings.dataset.prompts,
                split=args.split,
            )
            target_dir = output_dir / "predictions" / f"{scene_id:06d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            scene_predictions: list[dict[str, Any]] = []

            scene_loop_started = time.perf_counter()
            for spec in specs:
                target = _Target(dataset=args.dataset, scene_id=scene_id, im_id=0, obj_id=spec.obj_id)
                raw_boxes, raw_scores, raw_masks, base_text_state = pose_estimator.propose(spec.prompt, image_state)
                inference = pose_estimator.run_target(
                    config=inference_config,
                    target=target,
                    image=image,
                    camera_matrix=camera_matrix,
                    depth_result=depth_result,
                    pose_image_np=pose_image_np,
                    depth_image_size=depth_image_size,
                    raw_boxes=raw_boxes,
                    raw_scores=raw_scores,
                    raw_masks=raw_masks,
                    base_text_state=base_text_state,
                )
                mask_paths = {
                    "raw_mask_path": target_dir / f"obj{spec.obj_id:06d}_raw_masks.npz",
                    "pose_input_mask_path": target_dir / f"obj{spec.obj_id:06d}_pose_input_masks.npz",
                    "kept_mask_path": target_dir / f"obj{spec.obj_id:06d}_kept_masks.npz",
                }
                np.savez_compressed(mask_paths["raw_mask_path"], masks=inference.raw_masks.astype(np.uint8))
                np.savez_compressed(
                    mask_paths["pose_input_mask_path"], masks=inference.pose_input_masks.astype(np.uint8)
                )
                np.savez_compressed(mask_paths["kept_mask_path"], masks=inference.filtered_masks.astype(np.uint8))

                pose_input_stage = StagePredictions(
                    boxes_xyxy=inference.pose_input_boxes,
                    scores=inference.pose_input_scores,
                    masks=inference.pose_input_masks,
                    source_indices=[int(index) for index in inference.pose_input_source_indices],
                )
                raw_stage = StagePredictions(
                    boxes_xyxy=inference.raw_boxes, scores=inference.raw_scores, masks=inference.raw_masks
                )
                prediction = TargetPrediction(
                    target_key=f"{args.dataset}:{scene_id:06d}:000000:{spec.obj_id:06d}",
                    dataset=args.dataset,
                    scene_id=scene_id,
                    im_id=0,
                    obj_id=spec.obj_id,
                    object_name=spec.object_key,
                    prompt=spec.prompt,
                    raw=raw_stage,
                    pose_input=pose_input_stage,
                    kept=StagePredictions(
                        boxes_xyxy=inference.filtered_boxes,
                        scores=inference.filtered_scores,
                        masks=inference.filtered_masks,
                    ),
                    proposals=proposals_from_stages(
                        pose_input=pose_input_stage,
                        kept_indices=inference.kept_indices,
                        selection_results=inference.selection_results,
                        rerank_rows=inference.rerank_rows,
                    ),
                    kept_indices=inference.kept_indices,
                    extra={
                        "confidence_threshold": args.confidence_threshold,
                        # Needed so evaluation can rasterize ground truth for a target that
                        # produced ZERO proposals -- otherwise its GT instances vanish from the
                        # denominator and recall is silently inflated.
                        "image_size_wh": [int(image.size[0]), int(image.size[1])],
                        "image_path": str(image_path(dataset_dir, scene_id, 0, args.split)),
                        "depth_scene_dir": str(depth_dir),
                        "sam3_refinement_applied": args.sam3_refinement_policy != NO_REFINEMENT_POLICY,
                        "sam3_refinement_policy": args.sam3_refinement_policy,
                        "sam3_refinement_summary": inference.sam3_refinement_summary,
                        "sam3_refinement_candidates": inference.sam3_refinement_candidates,
                        "proposal_selection_applied": args.proposal_selection_policy != KEEP_ALL_RERANK_POLICY,
                        "proposal_selection_policy": args.proposal_selection_policy,
                        "proposal_selection_summary": inference.selection_summary,
                        "raw_proposal_pose_filter": [asdict(r) for r in inference.raw_filter_results],
                        "proposal_pose_filter": [asdict(r) for r in inference.selection_results],
                        "kept_source_indices": [
                            int(inference.pose_input_source_indices[i]) for i in inference.kept_indices
                        ],
                        "raw_overlay_path": str(
                            output_dir / "overlays" / f"{scene_id:06d}" / f"obj{spec.obj_id:06d}_raw.png"
                        ),
                        "pose_overlay_path": str(
                            output_dir / "overlays" / f"{scene_id:06d}" / f"obj{spec.obj_id:06d}_pose.png"
                        ),
                    },
                )
                if not args.no_overlays:
                    overlay_dir = output_dir / "overlays" / f"{scene_id:06d}"
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    draw_overlay(
                        image, inference.raw_boxes, inference.raw_scores,
                        inference.raw_masks.astype(np.uint8), spec.prompt, None,
                        overlay_dir / f"obj{spec.obj_id:06d}_raw.png",
                    )
                    draw_pose_filter_overlay(
                        image, inference.pose_input_boxes, inference.pose_input_scores,
                        inference.pose_input_masks, spec.prompt, inference.selection_results,
                        overlay_dir / f"obj{spec.obj_id:06d}_pose.png",
                    )
                    scene_predictions.append(
                        {
                            "masks": inference.raw_masks,
                            "boxes": inference.raw_boxes,
                            "scores": inference.raw_scores,
                            "pose_input_masks": inference.pose_input_masks,
                            "pose_input_boxes": inference.pose_input_boxes,
                            "pose_input_scores": inference.pose_input_scores,
                            "kept_indices": inference.kept_indices,
                            # draw_scene_overlay labels each mask and builds a legend from these.
                            "object_name": spec.object_key,
                            "prompt": spec.prompt,
                        }
                    )

                payload = prediction.to_dict(mask_paths=mask_paths)
                # Written per target as well as to the JSONL so a long run stays inspectable
                # even if a later scene fails.
                (target_dir / f"obj{spec.obj_id:06d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                predictions_file.write(json.dumps(payload) + "\n")
                predictions_file.flush()

            scene_loop_sec = time.perf_counter() - scene_loop_started

            overlay_started = time.perf_counter()
            if not args.no_overlays and scene_predictions:
                overlay_dir = output_dir / "overlays" / f"{scene_id:06d}"
                draw_scene_overlay(
                    image=image, predictions=scene_predictions,
                    output_path=overlay_dir / "scene_raw_all.png", kept_only=False,
                )
                draw_scene_overlay(
                    image=image, predictions=scene_predictions,
                    output_path=overlay_dir / "scene_kept_all.png", kept_only=True,
                )
            overlay_sec = time.perf_counter() - overlay_started

            pose_estimator.end_scene()
            # `runtime_sec` is the PRODUCTION path only, which is what ARCHITECTURE.md and the
            # generated report both claim it is: depth generation plus the per-target loop (SAM3,
            # FoundationPose, refinement, rerank). Overlay drawing is excluded and reported
            # separately so what was removed stays visible; `scoring_sec` belongs to evaluate.py
            # and is filled in there, since inference never scores against ground truth.
            runtime_rows.append(
                {
                    "scene_id": scene_id,
                    "runtime_sec": depth_sec + scene_loop_sec,
                    "depth_sec": depth_sec,
                    "scene_loop_sec": scene_loop_sec,
                    "overlay_sec": overlay_sec,
                    "scoring_sec": 0.0,
                    "wall_sec": time.perf_counter() - scene_started,
                }
            )

    pose_registry.close()
    (output_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "scenes": scene_ids,
                "objects": [spec.object_key for spec in explicit_specs] if explicit_specs else "from scene_gt.json",
                "sam3_refinement_policy": args.sam3_refinement_policy,
                "proposal_selection_policy": args.proposal_selection_policy,
                "rerank_cutoff": args.rerank_cutoff,
                "rerank_formula": inference_config.selection.formula_text(),
                "confidence_threshold": args.confidence_threshold,
                "foundation_stereo_model": str(args.foundation_stereo_model) if args.foundation_stereo_model else None,
                "foundation_stereo_max_width": args.foundation_stereo_max_width,
                "runtime_sec_total": sum(row["runtime_sec"] for row in runtime_rows),
                # Per-scene rows, not just the total: `evaluate.py` runs in a separate process, so
                # this file is the only way the timings reach the summariser. Without them
                # `runtime_summary.json` is an empty scaffold and the CSV is never written.
                "runtime_by_scene": runtime_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {predictions_path}")
    print(f"wrote {output_dir / 'inference_config.json'}")
    return output_dir


class _Target:
    """Minimal stand-in for `dataset.Target`, carrying only what inference reads.

    The real `Target` also carries `inst_count`, which comes from ground truth. Inference never
    reads it -- verified by grep across the codebase -- so this deliberately cannot supply it.
    """

    def __init__(self, *, dataset: str, scene_id: int, im_id: int, obj_id: int) -> None:
        self.dataset = dataset
        self.scene_id = scene_id
        self.im_id = im_id
        self.obj_id = obj_id

    @property
    def key(self) -> str:
        """BOP-style target identifier."""
        return f"{self.dataset}:{self.scene_id:06d}:{self.im_id:06d}:{self.obj_id:06d}"


if __name__ == "__main__":
    main()
