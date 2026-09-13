#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the pose input comes from: the depth-source registry, and the source that ships.

A *backend* (`inference/depth.py`) is a way of computing depth from a stereo pair. A *source* is
the answer to a different question -- where FoundationPose's depth comes from at all. The shipped
answer is "predict it", and it is the only one that makes sense for a deployment, but a capture
that ships measured depth can feed that instead, and evaluation setups sometimes want to.

Keeping them separate is what makes both extensible without either knowing about the other: a
source that reads depth from disk needs no backend, and a new backend needs no new source.

Registering one is the same shape as registering a backend, and for the same reason -- the CLI
choices are the registry keys, so an entry point never enumerates what exists:

    from foundationpose_perception_pipeline.inference.source import DepthSource, register_source

    register_source(DepthSource(name="mine", describe="…", provide=…))

See `foundationpose_perception_pipeline.extensions` for how a package outside this one gets loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PREDICTED_SOURCE = "foundationstereo"


@dataclass(frozen=True)
class DepthSource:
    """One way of obtaining the depth map FoundationPose registers against.

    `provide` writes `depth_m.npy` into `depth_dir` and returns the array. Returning it as well
    as writing it saves every caller a `np.load` of the file it just produced.

    `uses_depth_backend` says whether this source runs a model at all. It is what lets an entry
    point resolve (and assert) the depth backend once per run without asking what the source is.
    """

    name: str
    provide: Callable[..., np.ndarray]
    describe: str = ""
    uses_depth_backend: bool = True
    # Whether this source reads the collected-depth tree, and whether scoring predicted depth
    # against that tree means anything for it. Both are asked by entry points that have to decide
    # whether the tree is required for a run -- the alternative is every entry point knowing the
    # name of every source, which is exactly what the registry exists to avoid.
    needs_collected_depth: bool = False
    depth_metrics_meaningful: bool = True
    add_arguments: Callable[[Any], None] | None = None
    options: Callable[[Any], dict[str, Any]] = field(default=lambda _args: {})


_SOURCES: dict[str, DepthSource] = {}


def register_source(source: DepthSource) -> None:
    """Add a depth source to the registry, refusing a duplicate name."""
    if source.name in _SOURCES:
        raise ValueError(f"depth source {source.name!r} is already registered")
    _SOURCES[source.name] = source


def registered_sources() -> dict[str, DepthSource]:
    """Return the registry, extensions included."""
    from foundationpose_perception_pipeline.extensions import load_extensions

    load_extensions()
    return dict(_SOURCES)


def depth_source_choices() -> tuple[str, ...]:
    """Values `--depth-source` accepts, shipped source first so it reads as the default."""
    others = sorted(name for name in registered_sources() if name != PREDICTED_SOURCE)
    return (PREDICTED_SOURCE, *others)


def add_source_arguments(parser: Any) -> None:
    """Let every registered source add its own flags."""
    for source in registered_sources().values():
        if source.add_arguments is not None:
            source.add_arguments(parser)


def provide_predicted_depth(
    *,
    scene_dir: Path,
    depth_dir: Path,
    args: Any,
    **_ignored: Any,
) -> np.ndarray:
    """Predict depth from the scene's stereo pair, through the resolved depth backend."""
    from foundationpose_perception_pipeline.inference.depth import backend_options, generate_scene_depth

    generate_scene_depth(
        scene_dir=scene_dir,
        depth_dir=depth_dir,
        max_width=args.foundation_stereo_max_width,
        fixed_height=args.foundation_stereo_fixed_height,
        model=args.foundation_stereo_model,
        overwrite=args.overwrite_depth,
        backend=getattr(args, "depth_backend", "auto"),
        min_working_distance_m=args.min_working_distance_m,
        max_working_distance_m=args.max_working_distance_m,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_detail_boost=args.clahe_detail_boost,
        **backend_options(args),
    )
    return np.load(depth_dir / "depth_m.npy")


register_source(
    DepthSource(
        name=PREDICTED_SOURCE,
        provide=provide_predicted_depth,
        describe="predicted from the scene's stereo pair",
        uses_depth_backend=True,
    )
)
