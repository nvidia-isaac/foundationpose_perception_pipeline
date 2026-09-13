#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the camera-0 pipeline across all the datasets and aggregate results.

Every dataset runs the same configuration as ``script/run_pipeline.py``, so a bare invocation of
either measures the same path. ``--depth-source`` offers whatever is registered; see
``foundationpose_perception_pipeline.inference.source``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from foundationpose_perception_pipeline.config import (
    DEFAULT_DATE,
    DEFAULT_RERANK_CUTOFF,
    DEFAULT_RERANK_FORMULA,
    add_config_argument,
    settings_from_argv,
)
from foundationpose_perception_pipeline.config import (
    DEFAULT_SAM3_REFINEMENT_HIGH_MIOU as DEFAULT_REFINEMENT_HIGH_MIOU,
)
from foundationpose_perception_pipeline.config import (
    DEFAULT_SAM3_REFINEMENT_LOW_MIOU as DEFAULT_REFINEMENT_LOW_MIOU,
)
from foundationpose_perception_pipeline.config import (
    DEFAULT_SAM3_REFINEMENT_NMS_THRESHOLD as DEFAULT_REFINEMENT_NMS_THRESHOLD,
)
from foundationpose_perception_pipeline.evaluation.report import (
    MultiDatasetReportConfig,
    MultiDatasetReportGenerator,
)
from foundationpose_perception_pipeline.inference.depth import (
    add_backend_arguments,
    backend_forwarded_flags,
    depth_backend_choices,
)
from foundationpose_perception_pipeline.inference.source import (
    add_source_arguments,
    depth_source_choices,
    registered_sources,
)

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent

PIPELINE_SCRIPT = PIPELINE_ROOT / "script" / "run_pipeline.py"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multi-dataset pipeline execution.

    The dataset profile is read first, from `--config`, because it supplies this parser's own
    path and threshold defaults -- so `--help` shows the values a bare run would actually use.
    """
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--dataset-root", type=Path, default=settings.dataset.root)
    parser.add_argument(
        "--collected-root",
        type=Path,
        default=settings.dataset.collected_depth_root,
    )
    parser.add_argument("--pipeline-script", type=Path, default=PIPELINE_SCRIPT)
    parser.add_argument("--output-root", type=Path, default=settings.dataset.batch_output_root)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--confidence-threshold", type=float, default=settings.detection.sam3_confidence_threshold
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--depth-source",
        choices=depth_source_choices(),
        default=depth_source_choices()[0],
        help="Where FoundationPose's depth comes from, for every dataset in the batch. "
             + "; ".join(f"{s.name}: {s.describe}" for s in registered_sources().values()),
    )
    # Read from the same config as run_pipeline.py rather than hardcoded here. These two used to
    # be literals, which silently gave the batch runner different defaults from the single-dataset
    # entry point for the same flag names.
    parser.add_argument("--sam3-refinement-policy", default=settings.refinement.policy)
    parser.add_argument("--proposal-selection-policy", default=settings.rerank.policy)
    parser.add_argument("--fp-prepare-batch", type=int, default=64)
    parser.add_argument("--fp-n-hypotheses", type=int, default=64)
    parser.add_argument("--fp-n-refine", type=int, default=3)
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument(
        "--pose-min-visible-fraction",
        type=float,
        nargs="+",
        default=list(settings.pose.min_visible_fractions),
        help="Visibility bands for pose metrics, forwarded to every dataset.",
    )
    parser.add_argument(
        "--gt-cache-root",
        type=Path,
        default=None,
        help=(
            "Per-scene GT z-buffer cache from script/build_gt_cache.py. Only forwarded to the per-dataset "
            "runs when given; otherwise each run falls back to the cache root in its own "
            "profile."
        ),
    )
    parser.add_argument("--no-gt-cache", action="store_true", help="Force uncached GT rendering.")
    parser.add_argument(
        "--no-depth-metrics",
        action="store_true",
        help=(
            "Forwarded to every dataset: skip scoring predicted depth against collected "
            "ground-truth depth, so the sweep runs without a collected tree. Also drops it from "
            "dataset discovery, which otherwise FAILS when the collected tree is absent and "
            "selects nothing when it is present but shares no dataset names."
        ),
    )
    parser.add_argument("--foundation-stereo-model", type=Path, default=settings.depth.engine)
    parser.add_argument(
        "--depth-backend",
        choices=depth_backend_choices(),
        default="auto",
        help=(
            "Assert which depth backend every dataset uses. The model path decides; this only "
            "refuses to run when the two disagree. Worth setting on a long sweep: a batch that "
            "silently ran something else looks exactly like one that did not."
        ),
    )
    parser.add_argument(
        "--foundation-stereo-max-width",
        type=int,
        default=settings.depth.foundation_stereo_max_width,
    )
    parser.add_argument(
        "--foundation-stereo-fixed-height",
        type=int,
        default=settings.depth.foundation_stereo_fixed_height,
    )
    # Whatever the registered backends and sources need; forwarded to every dataset below.
    add_backend_arguments(parser)
    add_source_arguments(parser)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def dataset_names(dataset_root: Path, collected_root: Path | None, glob: str = "*") -> list[str]:
    """Return datasets present in the BOP tree, and in the collected-depth tree if used.

    `glob` is the profile's dataset-discovery pattern, so which subfolders count as datasets is
    configuration rather than a naming convention baked into the code.

    `collected_root` is None when no per-dataset run will read collected depth -- predicted
    depth with `--no-depth-metrics`. Requiring it then would make dataset discovery fail on a
    machine that has no collected tree, for a sweep that never opens one.
    """
    dataset_names = {path.name for path in dataset_root.glob(glob) if path.is_dir()}
    if collected_root is None:
        return sorted(dataset_names)
    # Checked rather than left to raise: `iterdir()` on a missing directory gives a bare
    # FileNotFoundError, where run_pipeline.py names the same tree and the way out.
    if not collected_root.is_dir():
        raise SystemExit(
            f"Missing collected-depth directory: {collected_root} (needed to intersect dataset "
            "discovery, and to score predicted depth against it). Pass --no-depth-metrics to "
            "drop it from both and sweep the BOP tree alone."
        )
    collected_names = {path.name for path in collected_root.iterdir() if path.is_dir()}
    return sorted(dataset_names & collected_names)


def recorded_runtime_seconds(dataset_output_dir: Path) -> float | None:
    """Production-path runtime a completed dataset recorded for itself, if available."""
    summary_path = dataset_output_dir / "runtime_summary.json"
    if not summary_path.exists():
        return None
    try:
        overall = json.loads(summary_path.read_text(encoding="utf-8"))["overall"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    total = overall.get("runtime_sec_total")
    return float(total) if total is not None else None


def run_dataset(args: argparse.Namespace, dataset: str, shared_engine_cache_dir: Path) -> dict[str, Any]:
    """Run the single-dataset pipeline script and capture status plus log paths."""
    dataset_output_dir = args.output_root / dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    # Completion marker: the per-target evaluation records plus the report mean this dataset
    # already ran.
    results_path = dataset_output_dir / "evaluations.jsonl"
    report_path = dataset_output_dir / "report.md"
    if results_path.exists() and report_path.exists() and not args.overwrite_datasets:
        print(f"[skip] {dataset} existing outputs at {dataset_output_dir}", flush=True)
        return {
            "dataset": dataset,
            "status": "skipped_existing",
            # Recovered from the dataset's own runtime_summary.json rather than left null.
            # Regenerating the aggregate report must not blank the runtime column just because
            # no subprocess ran this time -- and the pipeline's own figure is the better one
            # anyway, since it excludes GT scoring and overlay rendering.
            "runtime_seconds": recorded_runtime_seconds(dataset_output_dir),
            "output_dir": str(dataset_output_dir),
        }

    stdout_path = dataset_output_dir / "run_stdout.log"
    stderr_path = dataset_output_dir / "run_stderr.log"
    cmd = [
        sys.executable,
        str(args.pipeline_script),
        "--dataset-root",
        str(args.dataset_root),
        "--collected-root",
        str(args.collected_root),
        "--dataset",
        dataset,
        "--output-dir",
        str(dataset_output_dir),
        "--depth-source",
        args.depth_source,
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--resolution",
        str(args.resolution),
        "--sam3-refinement-policy",
        args.sam3_refinement_policy,
        "--proposal-selection-policy",
        args.proposal_selection_policy,
        "--fp-engine-cache-dir",
        str(shared_engine_cache_dir),
        "--fp-prepare-batch",
        str(args.fp_prepare_batch),
        "--fp-n-hypotheses",
        str(args.fp_n_hypotheses),
        "--fp-n-refine",
        str(args.fp_n_refine),
        "--overwrite-results",
    ]
    if args.max_scenes is not None:
        cmd.extend(["--max-scenes", str(args.max_scenes)])
    # Forward the GT cache and the FoundationStereo settings. Without the cache every dataset
    # re-rasterizes ground truth at ~10 s per target, which is ~4 h across the full sweep and
    # produces byte-identical results either way.
    if args.gt_cache_root is not None:
        cmd.extend(["--gt-cache-root", str(args.gt_cache_root)])
    if args.no_gt_cache:
        cmd.append("--no-gt-cache")
    if args.no_depth_metrics:
        cmd.append("--no-depth-metrics")
    cmd.extend(["--pose-min-visible-fraction", *[str(v) for v in args.pose_min_visible_fraction]])
    # Only the sources that run a model care about these, and asking the source is how this
    # stays true when a new one is registered.
    if registered_sources()[args.depth_source].uses_depth_backend:
        cmd.extend(["--foundation-stereo-max-width", str(args.foundation_stereo_max_width)])
        if args.foundation_stereo_fixed_height is not None:
            cmd.extend(["--foundation-stereo-fixed-height", str(args.foundation_stereo_fixed_height)])
        for flag, value in backend_forwarded_flags(args).items():
            cmd.extend([flag, str(value)])
        if args.foundation_stereo_model is not None:
            # Resolved, not passed through: the per-dataset run is a subprocess with a different
            # working directory, so a relative model path silently resolves somewhere else. It
            # fails fast with FileNotFoundError, but 32 times in a row at the start of an
            # overnight sweep.
            cmd.extend(["--foundation-stereo-model", str(Path(args.foundation_stereo_model).expanduser().resolve())])
        cmd.extend(["--depth-backend", args.depth_backend])

    env = dict(os.environ)
    env["PIPELINE_REPORT_DATE"] = DEFAULT_DATE
    start = time.time()
    print(f"[start] {dataset}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )
    runtime_seconds = time.time() - start
    status = "ok" if result.returncode == 0 else "failed"
    row = {
        "dataset": dataset,
        "status": status,
        "returncode": int(result.returncode),
        "runtime_seconds": runtime_seconds,
        "output_dir": str(dataset_output_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if result.returncode != 0:
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        row["error_preview"] = stderr_text.strip()[:4000]
        print(f"[fail] {dataset} exit={result.returncode} runtime_s={runtime_seconds:.1f}", flush=True)
    else:
        print(f"[done] {dataset} runtime_s={runtime_seconds:.1f}", flush=True)
    return row


def main() -> None:
    """Run per-dataset pipeline jobs and continuously refresh aggregate outputs."""
    args = parse_args()
    args.pipeline_script = args.pipeline_script.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    depth_source = registered_sources()[args.depth_source]
    collected_depth_needed = (
        depth_source.needs_collected_depth
        or (depth_source.depth_metrics_meaningful and not args.no_depth_metrics)
    )
    available_datasets = dataset_names(
        args.dataset_root,
        args.collected_root if collected_depth_needed else None,
        settings_from_argv().dataset.glob,
    )
    selected_datasets = args.datasets or available_datasets
    missing = [dataset for dataset in selected_datasets if dataset not in available_datasets]
    if missing:
        raise SystemExit(f"Unknown or unavailable datasets: {missing}")
    if args.max_datasets is not None:
        selected_datasets = selected_datasets[: args.max_datasets]
    if not selected_datasets:
        raise SystemExit(
            f"No datasets selected. Discovery matched {len(available_datasets)} dataset(s) under "
            f"{args.dataset_root} with glob {settings_from_argv().dataset.glob!r}"
            + (
                ", intersected against the collected-depth tree."
                if collected_depth_needed
                else "."
            )
            + " A sweep that selects nothing would otherwise exit 0 having done no work."
        )
    print(f"[discovery] {len(selected_datasets)} dataset(s) selected: {', '.join(selected_datasets)}", flush=True)
    args.datasets = selected_datasets

    shared_engine_cache_dir = args.output_root / "shared_foundationpose_engine_cache"
    shared_engine_cache_dir.mkdir(parents=True, exist_ok=True)
    report_generator = MultiDatasetReportGenerator(
        MultiDatasetReportConfig(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            datasets=args.datasets,
            generated_on=DEFAULT_DATE,
            depth_source=args.depth_source,
            confidence_threshold=args.confidence_threshold,
            resolution=args.resolution,
            sam3_refinement_policy=args.sam3_refinement_policy,
            proposal_selection_policy=args.proposal_selection_policy,
            refinement_replace_miou_low=DEFAULT_REFINEMENT_LOW_MIOU,
            refinement_replace_miou_high=DEFAULT_REFINEMENT_HIGH_MIOU,
            refinement_nms_threshold=DEFAULT_REFINEMENT_NMS_THRESHOLD,
            rerank_formula=DEFAULT_RERANK_FORMULA,
            rerank_cutoff=DEFAULT_RERANK_CUTOFF,
            fp_prepare_batch=args.fp_prepare_batch,
            fp_n_hypotheses=args.fp_n_hypotheses,
            fp_n_refine=args.fp_n_refine,
            pose_error_threshold_mm=settings_from_argv().pose.max_vertex_error_threshold_mm,
            pose_error_required_rate=settings_from_argv().pose.max_vertex_error_required_rate,
            prompt_source=f"{args.config or 'config/<profile>.yaml'} prompts, with object-name fallback",
            pose_min_visible_fractions=tuple(args.pose_min_visible_fraction),
        )
    )

    run_rows: list[dict[str, Any]] = []
    if not args.aggregate_only:
        for dataset in args.datasets:
            row = run_dataset(args, dataset, shared_engine_cache_dir)
            run_rows.append(row)
            aggregate = report_generator.aggregate_outputs(run_rows)
            report_generator.write_outputs(aggregate, run_rows)
            if row["status"] == "failed" and not args.continue_on_error:
                raise SystemExit(f"Dataset {dataset} failed; see {row['stderr_path']}")
    else:
        existing_run_status_path = args.output_root / "run_status.jsonl"
        existing_rows_by_dataset: dict[str, dict[str, Any]] = {}
        if existing_run_status_path.exists():
            for line in existing_run_status_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                existing_rows_by_dataset[str(row["dataset"])] = row
        for dataset in args.datasets:
            run_rows.append(
                existing_rows_by_dataset.get(
                    dataset,
                    {
                        "dataset": dataset,
                        "status": "aggregate_only",
                        "output_dir": str(args.output_root / dataset),
                        "runtime_seconds": None,
                    },
                )
            )

    aggregate = report_generator.aggregate_outputs(run_rows)
    report_generator.write_outputs(aggregate, run_rows)
    print(f"wrote {args.output_root / 'summary.json'}")
    print(f"wrote {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
