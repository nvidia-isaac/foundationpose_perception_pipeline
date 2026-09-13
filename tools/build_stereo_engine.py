#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the TensorRT engine for the commercial FoundationStereo model.

One-time per machine, per shape, per precision. The engine is not portable across GPU
architecture or TensorRT version and must not be committed; the filename encodes both so a stale
one is never picked up silently.

    ./.venv/bin/python tools/build_stereo_engine.py \\
        --onnx ../models/deployable_foundation_stereo_s_dynamic_v2.0.onnx \\
        --shape-from-scene <dataset_root>/<dataset>/<split>/000000

`--shape-from-scene` is the form to use whenever a dataset is present, and the example above is
deliberately not a literal `--shape`: a shape copied out of documentation is a shape nobody
measured on the rig it is about to run on, and a static engine fed a differently-sized pair
RESCALES rather than refusing, so a wrong one costs accuracy without ever raising. It takes a
path rather than resolving one, so `<split>` is yours to fill in from the profile's
`dataset.split`: `--config` supplies the width, not the scene.

The export this pipeline is developed and measured against is NGC model version
`nvidia/tao/foundationstereo:deployable_foundation_stereo_s_dynamic_v2.0` -- a *dynamic* export,
which is the kind to prefer. `../models/` is the sibling directory in README.md's layout; there is
no FoundationStereo checkout in it and none is needed, since only the built `.engine` is ever
read.

`--shape` is the *padded* rectified size the pipeline will feed -- both dimensions a multiple of
32. It exists for the case where no dataset has arrived yet and the install still has to be
finished; the engine it produces is a PLACEHOLDER and has to be rebuilt with `--shape-from-scene`
before any accuracy or regression figure is taken. When you have to pick one blind, derive it
rather than copying a number:

    height = round-up-to-32(rectified height at --max-width), width = --max-width

which for the default 800 px width lands at 480x800 on a 16:9-ish rig. That is a starting value
for one class of rig and not a default for yours -- the rectified height depends on how much of
the frame survives rectification, which is a property of the stereo pair's geometry.

A static profile (min=opt=max) keeps the execution shape and memory requirements predictable.
See README.md for the native TensorRT runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foundationpose_perception_pipeline.config import add_config_argument, settings_from_argv
from foundationpose_perception_pipeline.inference.models import STEREO_MODEL, ModelPaths
from foundationpose_perception_pipeline.inference.stereo.build import (
    PRECISIONS,
    ShapeProfile,
    build_engine,
    model_shape_for_rectified,
    parse_shape,
)


def rectified_shape_for_scene(scene_dir: Path, base_camera: int, max_width: int) -> tuple[int, int]:
    """Return the engine (height, width) this scene needs.

    Runs the real pair selection and rectification, so the answer matches what
    `stereo/depth.py` will actually produce rather than an estimate from the raw image size --
    rectification at `alpha=-1` does not preserve the input dimensions' aspect handling, and the
    `max_width` downscale is applied on top.

    `model_shape_for_rectified` then turns that rectified size into the shape the engine must
    accept. It is not a plain round-up of both dimensions: the runtime resizes to the engine's
    width rather than padding to it, so the height follows from the padded width. See that
    function for the failure this avoids.
    """
    from foundationpose_perception_pipeline.inference.stereo.depth import load_cameras, read_rgb
    from foundationpose_perception_pipeline.inference.stereo.rectify import (
        rectify_pair,
        select_partner_camera,
    )

    intrinsics, extrinsics, _ = load_cameras(scene_dir)
    sample = read_rgb(scene_dir / "rgb" / f"{base_camera:06d}.png")
    height, width = sample.shape[:2]

    selection = select_partner_camera(
        base_camera=base_camera,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_width=width,
        image_height=height,
    )
    if selection is None:
        raise SystemExit(f"No horizontal stereo partner for camera {base_camera} in {scene_dir}")
    partner = selection[0]

    rect = rectify_pair(
        read_rgb(scene_dir / "rgb" / f"{base_camera:06d}.png"),
        read_rgb(scene_dir / "rgb" / f"{partner:06d}.png"),
        intrinsics[base_camera],
        intrinsics[partner],
        extrinsics[base_camera],
        extrinsics[partner],
    )
    rect_h, rect_w = rect.left.shape[:2]
    return model_shape_for_rectified(rect_h, rect_w, max_width)


def main() -> None:
    """Parse arguments and build one engine."""
    # Before the parser, because `--max-width` defaults from it. `settings_from_argv` answers a
    # `--help` request from `defaults.yaml` rather than raising, so building the parser after this
    # line does not cost the reader their documentation on a tree with two profiles.
    settings = settings_from_argv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--onnx", type=Path, default=None,
                        help="Stereo ONNX export; defaults to the conventional name under MODELS_DIR.")
    parser.add_argument("--shape", type=str, default=None, help="Static input shape as HxW (multiple of 32)")
    parser.add_argument(
        "--shape-from-scene",
        type=Path,
        default=None,
        help="Derive the static shape from a scene directory, using the real rectification.",
    )
    parser.add_argument("--base-camera", type=int, default=0, help="With --shape-from-scene.")
    parser.add_argument(
        "--max-width", type=int, default=settings.depth.foundation_stereo_max_width,
        help="With --shape-from-scene. Defaults to the profile's "
             "depth.foundation_stereo_max_width, because the engine has to be built for the width "
             "the pipeline will actually feed it -- a static engine fed a different width rescales "
             "silently rather than failing, so a hardcoded value here builds the wrong engine for "
             "any profile that overrides the setting.",
    )
    parser.add_argument("--min", type=str, default=None, help="Dynamic profile lower bound, HxW")
    parser.add_argument("--opt", type=str, default=None, help="Dynamic profile optimum, HxW")
    parser.add_argument("--max", type=str, default=None, help="Dynamic profile upper bound, HxW")
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Clear BuilderFlag.TF32. Only for bit-level comparisons -- TF32 is on by default in "
        "TensorRT and in onnxruntime's CUDA EP alike, so leaving it on matches the baseline.",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=None,
        help="Scratch-memory cap for tactic selection. Unset means TensorRT's default (the whole "
        "device), which is what this model needs -- 4096 makes it fail to build entirely.",
    )
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir,
                        help="Model directory; compiled plans go in its engine_cache/ subdirectory.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a matching engine exists.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    given = [bool(args.shape), bool(args.shape_from_scene), bool(args.min or args.opt or args.max)]
    if sum(given) != 1:
        raise SystemExit("Pass exactly one of --shape, --shape-from-scene, or --min/--opt/--max.")

    if args.shape:
        profile = ShapeProfile.static(*parse_shape(args.shape))
    elif args.shape_from_scene:
        height, width = rectified_shape_for_scene(
            args.shape_from_scene.expanduser().resolve(), args.base_camera, args.max_width
        )
        print(f"scene rectifies to {width}x{height} (padded to a multiple of 32)")
        profile = ShapeProfile.static(height, width)
    else:
        if not (args.min and args.opt and args.max):
            raise SystemExit("A dynamic profile needs all three of --min, --opt and --max.")
        profile = ShapeProfile(parse_shape(args.min), parse_shape(args.opt), parse_shape(args.max))

    path = build_engine(
        args.onnx or ModelPaths.configured(args.models_dir).onnx(STEREO_MODEL),
        profile=profile,
        precision=args.precision,
        models_dir=args.models_dir,
        workspace_mb=args.workspace_mb,
        tf32=not args.no_tf32,
        force=args.force,
    )
    print(path)


if __name__ == "__main__":
    main()
