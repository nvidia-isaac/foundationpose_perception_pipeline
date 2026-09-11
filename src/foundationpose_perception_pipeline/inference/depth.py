#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Depth for one scene: the backend registry, and the backend that ships.

Depth is inference, not evaluation: it is computed from the stereo pair alone and is what
FoundationPose consumes. Scoring it against a collected ground-truth map is a separate, optional
activity that lives in `evaluation/`.

**One backend ships, and it is a registry entry like any other.** `commercial` runs a TAO
`deployable_*` export as a TensorRT engine through TAO Deploy, in this environment and this
process -- see `foundationpose_perception_pipeline.inference.stereo`. A function call, not a process: no
interpreter start-up, no torch import, no CUDA context creation per scene, and the depth comes
back as arrays rather than through a `.npy` round-trip. What it does still re-pay per scene is
deserializing the engine (~1.4 s), for the reason in `generate_with_engine`.

A site with another model registers a second backend from outside this package; see
`foundationpose_perception_pipeline.extensions`. Registration decides three things at once -- which model paths
the backend claims, how it produces depth, and what it adds to the command line -- so adding one
is a single object rather than edits scattered across three entry points.

Every backend writes the same files, `depth_m.npy` plus `metadata.json`, so nothing downstream of
depth can tell them apart and a cached depth directory is reusable across backends.

Taking arguments explicitly rather than an `argparse.Namespace` is deliberate: it is what lets a
caller that is not a CLI -- a service, a test -- ask for depth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ENGINE_SUFFIXES = (".engine", ".trt")

COMMERCIAL_BACKEND = "commercial"
AUTO_BACKEND = "auto"


class GenerateDepth(Protocol):
    """What a backend does: write `depth_m.npy` + `metadata.json` into `depth_dir`."""

    def __call__(self, *, scene_dir: Path, depth_dir: Path, model: Path | None, **options: Any) -> Path:
        """Produce depth for one scene and return the path to the written `depth_m.npy`."""


@dataclass(frozen=True)
class DepthBackend:
    """One way of turning a stereo pair into metric depth.

    `claims` is what keeps `--depth-backend` an assertion rather than a choice: the model path
    decides which backend runs, and the flag only refuses a run whose belief contradicts it.

    `add_arguments` and `options` are the CLI seam. A backend that needs its own flags adds them
    to whichever parser generates depth, and turns the parsed namespace back into keyword
    arguments for `generate`. The shipped backend needs neither, which is the point: the base
    case stays free of registry plumbing.
    """

    name: str
    claims: Callable[[Path | None], bool]
    generate: GenerateDepth
    describe: str = ""
    add_arguments: Callable[[Any], None] | None = None
    options: Callable[[Any], dict[str, Any]] = field(default=lambda _args: {})
    # `{"--flag": value}` for a runner that shells out to another entry point. Separate from
    # `options` because the CLI spelling of a setting is not derivable from its keyword name: a
    # backend's `interpreter=` keyword may be spelled anything on the command line, so the
    # mapping has to be given rather than inferred.
    forwarded_flags: Callable[[Any], dict[str, Any]] = field(default=lambda _args: {})


_BACKENDS: dict[str, DepthBackend] = {}


def register_backend(backend: DepthBackend) -> None:
    """Add a backend to the registry, refusing a duplicate name.

    Duplicate registration is an error rather than an overwrite: two backends answering to one
    name means one of them silently never runs, and the symptom would be metrics produced by a
    model nobody believes is in use.
    """
    if backend.name in _BACKENDS:
        raise ValueError(f"depth backend {backend.name!r} is already registered")
    _BACKENDS[backend.name] = backend


def registered_backends() -> dict[str, DepthBackend]:
    """Return the registry, extensions included."""
    from foundationpose_perception_pipeline.extensions import load_extensions

    load_extensions()
    return dict(_BACKENDS)


def depth_backend_choices() -> tuple[str, ...]:
    """Values `--depth-backend` accepts: `auto` plus every registered backend."""
    return (AUTO_BACKEND, *sorted(registered_backends()))


def add_backend_arguments(parser: Any) -> None:
    """Let every registered backend add its own flags to a parser that generates depth."""
    for backend in registered_backends().values():
        if backend.add_arguments is not None:
            backend.add_arguments(parser)


def backend_options(args: Any) -> dict[str, Any]:
    """Collect every registered backend's keyword arguments from a parsed namespace."""
    options: dict[str, Any] = {}
    for backend in registered_backends().values():
        options.update(backend.options(args))
    return options


def backend_forwarded_flags(args: Any) -> dict[str, Any]:
    """Collect the `--flag value` pairs a per-dataset subprocess needs, across all backends."""
    flags: dict[str, Any] = {}
    for backend in registered_backends().values():
        flags.update({flag: value for flag, value in backend.forwarded_flags(args).items() if value is not None})
    return flags


def is_engine(model: Path | str | None) -> bool:
    """Whether a model path names a TensorRT engine, i.e. selects the shipped backend."""
    return model is not None and Path(model).suffix.lower() in ENGINE_SUFFIXES


def _active_profile_hint() -> str:
    """Name the profile file this run resolved, for the "no engine" message.

    Worth the lookup: "the config profile" leaves the reader to work out which file that is, and
    the answer depends on `--config`, the environment variable and `--dataset`. Resolved the same
    way the run did. Falls back to the generic phrase rather than raising -- this is already an
    error path, and an error raised while building an error message helps nobody.
    """
    try:
        from foundationpose_perception_pipeline.config import (
            preparse_config,
            preparse_dataset,
            resolve_config_path,
        )

        return str(resolve_config_path(preparse_config(), preparse_dataset()))
    except Exception:  # noqa: BLE001 -- failing to name the file is not worth failing over
        return "the config profile"


def resolve_backend(model: Path | str | None, requested: str = AUTO_BACKEND) -> str:
    """Name the backend a model selects, and refuse a request that contradicts it.

    The model path stays the single source of truth -- a `.engine` is the shipped model and
    nothing else is -- so `--depth-backend` does not choose anything. It *asserts*, which is the
    useful half: "which licence is this run under" should be answerable without reading a file
    extension, and a run that believes it is one thing and silently is another is exactly the
    mistake worth failing on.
    """
    backends = registered_backends()
    choices = depth_backend_choices()
    if requested not in choices:
        raise ValueError(f"depth backend must be one of {choices}, got {requested!r}")

    path = Path(model) if model is not None else None
    claimed = [backend.name for backend in backends.values() if backend.claims(path)]
    if not claimed:
        known = ", ".join(f"{b.name} ({b.describe})" for b in backends.values() if b.describe)
        # Anything that is not a `.engine` gets the specific message, not the generic one. The
        # mistake people actually make is naming the ONNX -- they fetch the export, build the
        # engine beside it, then paste the path already in their shell history -- but a `.pth`,
        # a directory or a typo all land here too, and "no backend handles this" tells none of
        # them what the depth stage actually wants. Reached only when no backend claimed the
        # path, so a registered backend that legitimately takes another suffix is unaffected.
        if path is not None and path.suffix != ".engine":
            engines = sorted(path.parent.glob(f"{path.stem}__*.engine")) or sorted(
                path.parent.glob("*.engine")
            )
            found = f" Found in the same directory: {engines[0].name}" if engines else ""
            raise SystemExit(
                f"{model} is not a TensorRT engine, and no registered backend claims it. "
                f"Registered: {known or 'none'}. Build an engine with `tools/build_tao_engine.py "
                f"--onnx <deployable>.onnx --shape-from-scene <scene_dir>`; it writes a `.engine` "
                f"beside the ONNX. Point --foundation-stereo-model or depth.engine at that "
                f"file.{found}"
            )
        raise SystemExit(
            f"No depth backend handles {model or 'an unset model'}. Registered: {known or 'none'}. "
            f"Build an engine with `tools/build_tao_engine.py --shape-from-scene <scene_dir>`, "
            f"then either pass it with --foundation-stereo-model or set `depth.engine` under "
            f"`overrides: depth:` in {_active_profile_hint()}."
        )
    if len(claimed) > 1:
        raise SystemExit(f"{model} is claimed by more than one depth backend: {', '.join(claimed)}")

    actual = claimed[0]
    if requested != AUTO_BACKEND and requested != actual:
        raise SystemExit(
            f"--depth-backend {requested} contradicts the model in use: {model} is handled by "
            f"the {actual!r} backend."
        )
    return actual


def generate_scene_depth(
    *,
    scene_dir: Path,
    depth_dir: Path,
    max_width: int,
    model: Path | None = None,
    base_camera: int = 0,
    overwrite: bool = False,
    backend: str = AUTO_BACKEND,
    **options: Any,
) -> Path:
    """Produce `depth_m.npy` for one scene, reusing a cached result unless `overwrite`.

    Returns the path to the depth map. Caching on the presence of both `depth_m.npy` and
    `metadata.json` rather than the array alone means a half-written scene is regenerated instead
    of being silently trusted.

    `**options` reach the backend untouched, which is what lets a registered backend take
    settings this module has never heard of.
    """
    depth_path = depth_dir / "depth_m.npy"
    if depth_path.exists() and (depth_dir / "metadata.json").exists() and not overwrite:
        return depth_path

    chosen = registered_backends()[resolve_backend(model, backend)]
    return chosen.generate(
        scene_dir=scene_dir,
        depth_dir=depth_dir,
        model=model,
        max_width=max_width,
        base_camera=base_camera,
        **options,
    )


def generate_with_engine(
    *,
    scene_dir: Path,
    depth_dir: Path,
    model: Path | None,
    max_width: int,
    base_camera: int = 0,
    min_working_distance_m: float | None = None,
    max_working_distance_m: float | None = None,
    clahe_clip_limit: float | None = None,
    clahe_detail_boost: float = 0.0,
    **_ignored: Any,
) -> Path:
    """Depth for one scene, through TAO Deploy, in this process.

    The stereo package is imported here rather than at module scope so that `pycuda.autoinit`,
    which takes a CUDA context merely by being imported, is never triggered until depth is
    actually generated.
    """
    from foundationpose_perception_pipeline.inference.stereo import (
        load_engine,
        scene_depth,
        write_scene_depth,
    )
    from foundationpose_perception_pipeline.inference.stereo.tao import release_engines

    try:
        result = scene_depth(
            scene_dir,
            engine=load_engine(str(Path(model).expanduser().resolve())),  # type: ignore[arg-type]
            base_camera=base_camera,
            max_width=max_width,
            min_working_distance_m=min_working_distance_m,
            max_working_distance_m=max_working_distance_m,
            clahe_clip_limit=clahe_clip_limit,
            clahe_detail_boost=clahe_detail_boost,
        )
        return write_scene_depth(result, depth_dir)
    finally:
        # Release before returning, because the caller's next move is the pose stage and the two
        # cannot both be resident. FoundationStereo's TensorRT scratch is 10-13 GB depending on
        # the profile; holding it across a run makes peak memory the SUM of the stages instead of
        # the max, and SAM3 dies with `torch.OutOfMemoryError` a few scenes in. Measured on a
        # 32 GB card: 30.1 GB in use with only 3.8 GB of it torch's.
        #
        # The cost is re-deserializing the engine next scene, about 1.4 s against ~32 s of
        # per-scene work. A caller that owns the whole GPU -- a depth-only sweep, a test --
        # should use `scene_depth` directly and keep the engine loaded instead.
        release_engines()


register_backend(
    DepthBackend(
        name=COMMERCIAL_BACKEND,
        claims=is_engine,
        generate=generate_with_engine,
        describe="a TensorRT engine built from a TAO deployable export",
    )
)
