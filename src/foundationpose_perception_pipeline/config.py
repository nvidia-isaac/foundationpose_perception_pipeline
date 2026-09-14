#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for the pipeline.

This module holds no configuration *values*. Every tuned constant lives in
``config/defaults.yaml``; every dataset-specific path, prompt and name lives in a dataset
profile under ``config/``. This module reads them and hands back typed objects.

Precedence, lowest to highest::

    config/defaults.yaml  ->  profile `overrides:` block  ->  CLI flag

Two ways in:

- :func:`load_settings` returns the merged :class:`Settings` for one profile. Entry points call
  this before building their argument parser, so ``--help`` shows the effective values and a
  profile's ``overrides:`` block reaches the CLI defaults.
- The module-level ``DEFAULT_*`` names are the values from ``config/defaults.yaml`` only, with
  no profile applied. They exist so library functions can carry a sensible keyword default
  (``def f(..., min_visible_fraction=DEFAULT_MIN_VISIBLE_FRACTION)``) without threading a
  ``Settings`` object through every call. They are read from YAML at import time -- they are not
  a second source of truth.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import yaml

# src/foundationpose_perception_pipeline/config.py -> src/foundationpose_perception_pipeline -> src -> pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PIPELINE_ROOT / "config"
DEFAULTS_PATH = CONFIG_DIR / "defaults.yaml"
# Not a profile -- it is the annotated template users copy, and it has placeholder paths.
TEMPLATE_PROFILE_NAME = "example_bop"
CONFIG_ENV_VAR = "PERCEPTION_PIPELINE_CONFIG"


class ConfigError(RuntimeError):
    """Raised when a config file is missing, unreadable, or has an unexpected shape."""


# --------------------------------------------------------------------------------------
# FoundationPose checkout discovery (a procedure, not a value)
# --------------------------------------------------------------------------------------


def foundationpose_root_default() -> Path | None:
    """Return the FoundationPose checkout root from ``FOUNDATIONPOSE_ROOT``, or None.

    There is deliberately no built-in fallback path: FoundationPose is an external checkout
    whose location is site-specific. Callers that need it should raise
    :func:`foundationpose_root_missing_message` when this returns None. Every entry-point
    script also exposes ``--foundationpose-root`` to override it at the CLI.
    """

    env_value = os.environ.get("FOUNDATIONPOSE_ROOT")
    return Path(env_value) if env_value else None


def foundationpose_root_missing_message() -> str:
    """Return the error text shown when FoundationPose cannot be located."""
    return (
        "FoundationPose checkout not found. Set FOUNDATIONPOSE_ROOT to your checkout, or pass "
        "--foundationpose-root. See the FoundationPose section of README.md."
    )


def models_dir_default() -> Path:
    """Resolve the model directory without requiring a dataset for standalone tools."""
    config, dataset = preparse_config(), preparse_dataset()
    if config or dataset or os.environ.get(CONFIG_ENV_VAR) or len(available_profiles()) == 1:
        return settings_from_argv().models_dir
    return _models_dir(_read_yaml(DEFAULTS_PATH), {}, DEFAULTS_PATH)


def _models_dir(defaults: dict, overrides: dict, profile_path: Path) -> Path:
    value = os.environ.get("MODELS_DIR")
    if value:
        return Path(value).expanduser().resolve()
    value = overrides.get("models_dir", defaults["models_dir"])
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{profile_path}: models_dir must be a non-empty path")
    base = profile_path.parent if "models_dir" in overrides else DEFAULTS_PATH.parent
    return _resolve_path(value, base)


def help_requested(argv: list[str] | None = None) -> bool:
    """Whether this invocation is asking for ``--help`` rather than asking to run.

    ONE PREDICATE, because the entry points have two separate start-up preconditions -- a
    resolvable config profile, and a locatable FoundationPose checkout -- and both are checked
    BEFORE argparse can print anything. Each one, unguarded, answers a request for documentation
    with an error about something the reader was not asking about. They should agree on what a
    help request looks like, so they share this.

    Deliberately a plain token scan and nothing cleverer: argparse itself treats the first bare
    `-h`/`--help` as help wherever it appears, and anything subtler here could disagree with the
    parser that runs moments later.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return "-h" in args or "--help" in args


# --------------------------------------------------------------------------------------
# Typed settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetProfile:
    """Where one dataset lives and what its objects are called."""

    name: str
    root: Path
    glob: str
    split: str
    base_camera_id: int
    # Where the CAD meshes live, under <root>/<dataset>/. TWO keys, because BOP means two
    # different things by them and conflating them is a real error, not a naming preference:
    #
    #   models_subdir      -- the mesh POSE ESTIMATION and RENDERING use. Full-resolution CAD.
    #   models_eval_subdir -- the mesh METRICS use. BOP ships a decimated, roughly uniformly
    #                         resampled copy in `models_eval/` precisely so that averaging a
    #                         distance over its vertices is meaningful; ADD/ADD-S and max-vertex
    #                         error over a non-uniform mesh weight wherever the tessellation
    #                         happens to be dense.
    #
    # This project's own captures ship `models/`, `models_cad/` and `models_eval/` as three
    # byte-identical copies, so both default to `models_cad` and nothing changes. A standard BOP
    # dataset should set `models_subdir: models` and `models_eval_subdir: models_eval`.
    models_subdir: str
    models_eval_subdir: str
    collected_depth_root: Path | None
    gt_cache_root: Path
    output_root: Path
    batch_subdir: str
    prompts: dict[str, str] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)

    @property
    def depth_filename(self) -> str:
        """Return the collected-depth filename for this profile's base camera."""
        return f"scene_cam{self.base_camera_id}_depth.png"

    @property
    def batch_output_root(self) -> Path:
        """Return the aggregate-output folder used by the batch eval runner."""
        return self.output_root / self.batch_subdir


@dataclass(frozen=True)
class DetectionSettings:
    """SAM3 detection thresholds."""

    sam3_confidence_threshold: float


@dataclass(frozen=True)
class RerankSettings:
    """Proposal reranking policy and its operating point."""

    policy: str
    cutoff: float
    formula: str


@dataclass(frozen=True)
class RefinementSettings:
    """CAD box-prompt refinement policy and its thresholds."""

    policy: str
    low_miou: float
    high_miou: float
    nms_threshold: float


@dataclass(frozen=True)
class PoseSettings:
    """Pose acceptance criterion and the visibility bands it is reported at."""

    max_vertex_error_threshold_mm: float
    max_vertex_error_required_rate: float
    min_visible_fractions: tuple[float, ...]


@dataclass(frozen=True)
class GroundTruthSettings:
    """Which ground-truth instances count, and how they are rasterized."""

    min_visible_fraction: float
    rasterizer_near_mm: float


@dataclass(frozen=True)
class DepthSettings:
    """FoundationStereo depth generation: which engine, and how the pair is preprocessed."""

    foundation_stereo_max_width: int
    foundation_stereo_fixed_height: int | None
    """Fixed rectified height for a static-height stereo plan, or None to follow the pair.

    Pairs with `foundation_stereo_max_width`: a static plan is built for one HxW, so the two are
    set together or not at all.
    """

    min_working_distance_m: float | None
    """Nearest surface the scene is expected to contain, in metres.

    With `max_working_distance_m`, enables the disparity pre-shift and the feasibility-aware
    partner ranking. Both are no-ops when either is unset.
    """

    max_working_distance_m: float | None
    """Farthest surface the scene is expected to contain, in metres.

    The pair bounds a working *volume*. FoundationStereo searches [0, 416) px; the near end of
    the volume sets the largest disparity asked for and the far end the smallest, and the shift
    slides that window into range only when it does not already fit. Set them to the actual bin,
    tightly: their difference has to fit the range.
    """

    clahe_detail_boost: float
    """Detail-layer gain for the CLAHE base/detail split; 0 disables the split."""

    clahe_clip_limit: float | None
    """CLAHE clip limit for the rectified pair, or None to disable.

    Worth having on a capture that is grayscale and underexposed, where equalising locally
    recovers texture the matcher can lock onto; measured there it cut mean object depth error
    materially and composed with the disparity pre-shift rather than overlapping it. On a
    well-exposed capture it may buy nothing, so treat it as a knob to sweep, not a default to
    assume.
    """

    engine: Path | None
    """TensorRT engine the depth stage runs, through TensorRT, in this process.

    Accepts an ONNX source (compiled on demand) or a precompiled engine path.
    None uses the conventional stereo model under MODELS_DIR, with plans in engine_cache/.
    """


@dataclass(frozen=True)
class ValidationSettings:
    """Tolerances used by the GT-depth registration check and the offline cutoff sweep."""

    tolerance_mm: float
    min_within_5mm: float
    cutoffs: list[float]


DEFAULT_SAM3_RESOLUTION: int = 1008
"""Square input the SAM3 graphs are exported at. Fixed by the export, not a tunable."""


@dataclass(frozen=True)
class Settings:
    """``defaults.yaml`` merged with one profile's ``overrides:`` block."""

    dataset: DatasetProfile
    detection: DetectionSettings
    rerank: RerankSettings
    refinement: RefinementSettings
    pose: PoseSettings
    ground_truth: GroundTruthSettings
    depth: DepthSettings
    validation: ValidationSettings
    models_dir: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file, raising ConfigError with the path on any failure."""
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at the top level of {path}, got {type(data).__name__}")
    return data


def available_profiles() -> list[str]:
    """Return the profile names in ``config/``, excluding defaults and the template."""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(
        path.stem
        for path in CONFIG_DIR.glob("*.yaml")
        if path.stem not in {"defaults", TEMPLATE_PROFILE_NAME}
    )


def resolve_config_path(config: str | Path | None = None, dataset: str | None = None) -> Path:
    """Resolve a profile name or path to a YAML file.

    Order, most specific first: the ``config`` argument, ``$PERCEPTION_PIPELINE_CONFIG``, a
    profile named after ``dataset``, then the sole profile in ``config/`` if there is exactly
    one. A bare name such as ``my_dataset`` resolves to ``config/my_dataset.yaml``.

    The ``dataset`` rule is not a guess: the caller passed ``--dataset X`` and ``config/X.yaml``
    exists, so the profile is the one they named. It is ranked BELOW the explicit two so neither
    can be overridden by it, and above the sole-profile rule because it is more specific -- and
    it only ever fires where the alternative is the "refuses to guess" error, so it cannot change
    what an already-working command resolves to. A dataset with no same-named profile still
    errors, which is right: several datasets can share one profile, and nothing here can know
    which.
    """
    candidate = config if config is not None else os.environ.get(CONFIG_ENV_VAR)

    if candidate:
        path = Path(candidate)
        if path.suffix in {".yaml", ".yml"} or path.exists():
            return path if path.is_absolute() else path.resolve()
        named = CONFIG_DIR / f"{candidate}.yaml"
        if named.exists():
            return named
        raise ConfigError(
            f"No such config profile: {candidate!r}. Available: {available_profiles() or 'none'} "
            f"(looked in {CONFIG_DIR})"
        )

    if dataset:
        named = CONFIG_DIR / f"{dataset}.yaml"
        if named.exists():
            return named

    profiles = available_profiles()
    if len(profiles) == 1:
        return CONFIG_DIR / f"{profiles[0]}.yaml"
    if not profiles:
        raise ConfigError(
            f"No dataset profile found in {CONFIG_DIR}. Copy {TEMPLATE_PROFILE_NAME}.yaml to "
            f"{CONFIG_DIR}/<your-dataset>.yaml and edit it."
        )
    raise ConfigError(
        f"Multiple config profiles available ({', '.join(profiles)}); pass --config <name> or "
        f"set {CONFIG_ENV_VAR}."
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return `base` with `override` merged in, recursing into nested mappings."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    """Fetch a required key, raising ConfigError naming the file section when absent."""
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return section[key]


def _resolve_path(value: Any, base_dir: Path) -> Path:
    """Resolve a config path value against the config file's own directory."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _reject_unknown_overrides(overrides: dict[str, Any], defaults: dict[str, Any], profile_path: Path) -> None:
    """Refuse an `overrides:` key that `defaults.yaml` does not define.

    `_deep_merge` accepts anything, so an override under the wrong section, or with a typo, used
    to be merged into a key nothing reads and then silently ignored. That failure is close to
    undiagnosable from the outside: the profile plainly contains the setting, the run behaves as
    though it does not, and no output mentions either fact. The case that prompted this was an
    `engine:` line indented under `rerank:` because the `depth:` above it was still commented out
    -- the run then reported no engine configured while the file appeared to configure one.

    `defaults.yaml` is the single source of truth for tunables, so a key absent from it is always
    a mistake rather than an extension point. Where the same key exists under a different section
    the message says so, because a misplaced key is far more common than an invented one.
    """
    for section, values in overrides.items():
        if section not in defaults:
            raise ConfigError(
                f"{profile_path}: `overrides:` has no section {section!r}. "
                f"Sections come from {DEFAULTS_PATH.name}: {', '.join(sorted(defaults))}."
            )
        if not isinstance(values, dict) or not isinstance(defaults[section], dict):
            continue
        for key, val in values.items():
            if key in defaults[section]:
                if isinstance(val, dict) and isinstance(defaults[section][key], dict):
                    for subkey in val:
                        if subkey not in defaults[section][key]:
                            raise ConfigError(
                                f"{profile_path}: `overrides: {section}: {key}:` has no key {subkey!r}. "
                                f"Keys of {section}.{key}: {', '.join(sorted(defaults[section][key]))}."
                            )
                continue
            elsewhere = sorted(
                other for other, block in defaults.items()
                if isinstance(block, dict) and key in block
            )
            hint = (f" It is a key of {' and '.join(elsewhere)}, so check the indentation -- a "
                    f"commented-out parent leaves the line attached to the section above it."
                    if elsewhere else
                    f" Keys of {section}: {', '.join(sorted(defaults[section]))}.")
            raise ConfigError(
                f"{profile_path}: `overrides: {section}:` has no key {key!r}.{hint}"
            )


def _algorithm_settings(
    merged: dict[str, Any], profile_dir: Path, profile_path: Path
) -> tuple[
    DetectionSettings, RerankSettings, RefinementSettings, PoseSettings,
    GroundTruthSettings, DepthSettings, ValidationSettings,
]:
    """Build every non-dataset settings section from an already-merged mapping.

    Split out of `load_settings` so the defaults-only path below can reuse it verbatim rather
    than growing a second copy of the same field mapping, which is the way the two would drift.
    `profile_path` is only ever used to name the file in an error message.
    """
    try:
        detection = DetectionSettings(
            sam3_confidence_threshold=float(merged["detection"]["sam3_confidence_threshold"]),
        )
        rerank = RerankSettings(
            policy=str(merged["rerank"]["policy"]),
            cutoff=float(merged["rerank"]["cutoff"]),
            formula=str(merged["rerank"]["formula"]),
        )
        refinement = RefinementSettings(
            policy=str(merged["refinement"]["policy"]),
            low_miou=float(merged["refinement"]["low_miou"]),
            high_miou=float(merged["refinement"]["high_miou"]),
            nms_threshold=float(merged["refinement"]["nms_threshold"]),
        )
        pose = PoseSettings(
            max_vertex_error_threshold_mm=float(merged["pose"]["max_vertex_error_threshold_mm"]),
            max_vertex_error_required_rate=float(merged["pose"]["max_vertex_error_required_rate"]),
            min_visible_fractions=tuple(float(v) for v in merged["pose"]["min_visible_fractions"]),
        )
        ground_truth = GroundTruthSettings(
            min_visible_fraction=float(merged["ground_truth"]["min_visible_fraction"]),
            rasterizer_near_mm=float(merged["ground_truth"]["rasterizer_near_mm"]),
        )
        depth = DepthSettings(
            foundation_stereo_max_width=int(merged["depth"]["foundation_stereo_max_width"]),
            foundation_stereo_fixed_height=(
                int(merged["depth"]["foundation_stereo_fixed_height"])
                if merged["depth"].get("foundation_stereo_fixed_height") is not None else None
            ),
            # Resolved against the config file's directory like every other path in a profile,
            # so a profile can point at an engine built for its own rectified size.
            engine=(
                _resolve_path(merged["depth"]["engine"], profile_dir)
                if merged["depth"].get("engine")
                else None
            ),
            min_working_distance_m=(
                float(merged["depth"]["min_working_distance_m"])
                if merged["depth"].get("min_working_distance_m") else None
            ),
            max_working_distance_m=(
                float(merged["depth"]["max_working_distance_m"])
                if merged["depth"].get("max_working_distance_m") else None
            ),
            clahe_clip_limit=(
                float(merged["depth"]["clahe_clip_limit"])
                if merged["depth"].get("clahe_clip_limit") else None
            ),
            clahe_detail_boost=float(merged["depth"].get("clahe_detail_boost") or 0.0),
        )
        validation = ValidationSettings(
            tolerance_mm=float(merged["validation"]["tolerance_mm"]),
            min_within_5mm=float(merged["validation"]["min_within_5mm"]),
            cutoffs=[float(v) for v in merged["validation"]["cutoffs"]],
        )
    except KeyError as exc:
        raise ConfigError(
            f"Missing key {exc} after merging {DEFAULTS_PATH} with the overrides in {profile_path}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Bad value in {DEFAULTS_PATH} or {profile_path}: {exc}") from exc

    return detection, rerank, refinement, pose, ground_truth, depth, validation


@cache
def load_settings(config: str | Path | None = None, dataset: str | None = None) -> Settings:
    """Load ``defaults.yaml`` merged with one dataset profile.

    Cached: a sweep resolves thousands of targets and must not re-read YAML per target. Both
    arguments are strings or None, so adding the second keeps the key hashable.
    """
    profile_path = resolve_config_path(config, dataset)
    profile_dir = profile_path.parent

    defaults = _read_yaml(DEFAULTS_PATH)
    profile = _read_yaml(profile_path)
    overrides = profile.get("overrides") or {}
    _reject_unknown_overrides(overrides, defaults, profile_path)
    merged = _deep_merge(defaults, overrides)

    dataset_section = _require(profile, "dataset", str(profile_path))
    output_section = profile.get("output") or {}
    collected = dataset_section.get("collected_depth_root")

    dataset_profile = DatasetProfile(
        name=str(_require(dataset_section, "name", f"{profile_path}:dataset")),
        root=_resolve_path(_require(dataset_section, "root", f"{profile_path}:dataset"), profile_dir),
        glob=str(dataset_section.get("glob", "*")),
        split=str(dataset_section.get("split", "test")),
        base_camera_id=int(dataset_section.get("base_camera_id", 0)),
        models_subdir=str(dataset_section.get("models_subdir", "models_cad")),
        # Defaults to models_subdir rather than to "models_eval": a dataset that ships only one
        # mesh directory is then correct by default, and a BOP dataset states both explicitly.
        models_eval_subdir=str(
            dataset_section.get("models_eval_subdir", dataset_section.get("models_subdir", "models_cad"))
        ),
        collected_depth_root=_resolve_path(collected, profile_dir) if collected else None,
        gt_cache_root=_resolve_path(dataset_section.get("gt_cache_root", "../gt_cache"), profile_dir),
        output_root=_resolve_path(output_section.get("root", "../output"), profile_dir),
        batch_subdir=str(output_section.get("batch_subdir", "batch_run")),
        prompts={str(k): str(v) for k, v in (profile.get("prompts") or {}).items()},
        regression=dict(profile.get("regression") or {}),
    )

    detection, rerank, refinement, pose, ground_truth, depth, validation = _algorithm_settings(
        merged, profile_dir, profile_path
    )
    return Settings(
        dataset=dataset_profile,
        models_dir=_models_dir(defaults, overrides, profile_path),
        detection=detection,
        rerank=rerank,
        refinement=refinement,
        pose=pose,
        ground_truth=ground_truth,
        depth=depth,
        validation=validation,
    )


@cache
def defaults_only_settings() -> Settings:
    """Settings from ``defaults.yaml`` alone, with a placeholder dataset and no profile merged.

    THIS IS NOT A FALLBACK FOR A RUN. It exists so that `--help` can be answered on a tree with
    more than one profile and no `--config`: the values it carries are the shipped defaults, which
    is the honest answer to "what does this flag default to" when the caller named no profile.

    The dataset half is a placeholder and is deliberately unusable rather than plausible -- an
    empty name and `config/` as the root -- because nothing may run against it. Anything that
    needs real dataset paths resolves a profile properly and fails loudly when it cannot.
    """
    defaults = _read_yaml(DEFAULTS_PATH)
    detection, rerank, refinement, pose, ground_truth, depth, validation = _algorithm_settings(
        defaults, CONFIG_DIR, DEFAULTS_PATH
    )
    return Settings(
        dataset=DatasetProfile(
            name="",
            root=CONFIG_DIR,
            glob="*",
            split="test",
            base_camera_id=0,
            models_subdir="models_cad",
            models_eval_subdir="models_cad",
            collected_depth_root=None,
            gt_cache_root=CONFIG_DIR / "../gt_cache",
            output_root=CONFIG_DIR / "../output",
            batch_subdir="batch_run",
        ),
        models_dir=_models_dir(defaults, {}, DEFAULTS_PATH),
        detection=detection,
        rerank=rerank,
        refinement=refinement,
        pose=pose,
        ground_truth=ground_truth,
        depth=depth,
        validation=validation,
    )


# --------------------------------------------------------------------------------------
# argparse integration
# --------------------------------------------------------------------------------------


def add_config_argument(parser: Any) -> None:
    """Register the standard ``--config`` flag on an argument parser."""
    parser.add_argument(
        "--config",
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Dataset profile to load: a bare name resolved against config/, "
            f"or a path to a YAML file. Defaults to ${CONFIG_ENV_VAR}, or the single profile "
            "in config/ when there is exactly one."
        ),
    )


def preparse_config(argv: list[str] | None = None) -> str | None:
    """Extract ``--config`` from argv before the real parser is built.

    The profile supplies the parser's own defaults, so it has to be known first. This scans for
    ``--config VALUE`` and ``--config=VALUE`` and ignores everything else -- the real parser
    still validates the full command line afterwards.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(args):
        if token == "--config" and index + 1 < len(args):
            return args[index + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def preparse_dataset(argv: list[str] | None = None) -> str | None:
    """Extract ``--dataset`` from argv, for use only as a profile fallback.

    Same scan as :func:`preparse_config`, and used only when neither ``--config`` nor the
    environment variable resolved. The real parser still validates the value afterwards; a
    dataset that names no profile simply leaves the fallback unused.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(args):
        if token == "--dataset" and index + 1 < len(args):
            return args[index + 1]
        if token.startswith("--dataset="):
            return token.split("=", 1)[1]
    return None


def settings_from_argv(argv: list[str] | None = None) -> Settings:
    """Load the settings named by ``--config``, or by ``--dataset`` when that names a profile.

    A REQUEST FOR ``--help`` IS ANSWERED, NEVER REFUSED. Entry points call this BEFORE building
    their parser, because several flags take their default from the profile. That ordering means
    an unresolvable profile kills `--help` before argparse prints a line -- and with two profiles
    in `config/`, "unresolvable" is the DEFAULT state of a bare command, so asking a script what
    its flags are answers with a configuration error instead of documentation. Seven entry points
    behaved that way. So when argv asks for help and no profile resolves, fall back to
    `defaults.yaml` alone instead of raising.

    Narrow on purpose, in three ways. It fires only on a `ConfigError`, so a broken profile still
    reports itself. It fires only when `-h`/`--help` is present, and `--help` exits inside
    `parse_args()` -- the settings are used to fill in defaults that are about to be PRINTED and
    never to run anything, so no real run can reach this path. And a help request that DOES name a
    `--config` resolves normally and prints that profile's values, so the fallback never shadows
    an answer the caller actually asked for.
    """
    try:
        return load_settings(preparse_config(argv), preparse_dataset(argv))
    except ConfigError:
        if help_requested(argv):
            return defaults_only_settings()
        raise


def active_settings() -> Settings:
    """The settings for the command currently running, for library code with no `Settings` to hand.

    Library fallbacks must resolve the profile the SAME way the entry point did, which means
    reading the command line rather than calling `load_settings()` with no arguments. Bare
    `load_settings()` re-resolves from scratch: it cannot see `--config` or `--dataset`, so it
    falls through to the sole-profile rule and either fails on a tree with two profiles or --
    worse, and silently -- answers from a different profile than the run is using.

    Passing a `Settings` down is still better where a caller has one; this is the fallback for
    where none is threaded through.
    """
    return settings_from_argv()


# --------------------------------------------------------------------------------------
# Module-level defaults, read from config/defaults.yaml at import time.
#
# These carry no profile overrides. They exist only so library functions can declare a keyword
# default without taking a Settings argument. Entry points should use load_settings() instead.
# --------------------------------------------------------------------------------------

_DEFAULTS = _read_yaml(DEFAULTS_PATH)

DEFAULT_SAM3_CONFIDENCE_THRESHOLD = float(_DEFAULTS["detection"]["sam3_confidence_threshold"])

# The policy *identifiers* are enum-like constants, deliberately NOT read from YAML: they name
# code paths, not tunable values. Keeping them separate from the DEFAULT_* below is what lets the
# default change without silently redefining what "disabled" means: if a policy identifier and a
# default shared one constant, flipping the default would invert every `policy == off` check.
KEEP_ALL_RERANK_POLICY = "all"          # off: keep every proposal
SOFT_GLOBAL_RERANK_POLICY = "soft_global_v1"
NO_REFINEMENT_POLICY = "none"           # off: no CAD box-prompt refinement pass
REPLACE_MID_NMS06_REFINEMENT_POLICY = "replace_mid_nms06"

DEFAULT_RERANK_POLICY = str(_DEFAULTS["rerank"]["policy"])
DEFAULT_RERANK_CUTOFF = float(_DEFAULTS["rerank"]["cutoff"])
DEFAULT_RERANK_FORMULA = str(_DEFAULTS["rerank"]["formula"])

# The canonical shape of `rerank.formula`. Parsed with a strict pattern rather than `eval`: this
# string comes from a config file, and a scoring rule is not a place to execute arbitrary text.
_RERANK_FORMULA_PATTERN = re.compile(
    r"^\s*(?P<sam>[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*\*\s*sam_score"
    r"\s*\+\s*(?P<mask>[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*\*\s*render_mask_iou"
    r"\s*\+\s*(?P<box>[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*\*\s*render_box_iou"
    r"\s*-\s*fp_score\s*/\s*(?P<divisor>[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*$"
)


def rerank_weights_from_formula(formula: str) -> dict[str, float]:
    """Parse `rerank.formula` into the weights the scorer actually applies.

    The formula is real configuration, not a caption. Before this existed the weights were
    hardcoded in three places while the string was read only for report provenance, so editing it
    changed what the report *claimed* the scoring was while leaving the scoring untouched -- the
    one failure mode where an artifact and the run it describes can disagree without anything
    raising.

    Raises rather than falling back to defaults on an unparseable string. A silent fallback would
    reintroduce exactly the divergence this function exists to prevent.
    """
    match = _RERANK_FORMULA_PATTERN.match(formula)
    if match is None:
        raise ValueError(
            f"Unparseable rerank.formula: {formula!r}\n"
            "Expected the form: "
            "'<a> * sam_score + <b> * render_mask_iou + <c> * render_box_iou - fp_score / <d>'\n"
            "To change the weights, edit the numbers in place; to change the terms themselves, "
            "inference/config.py's SelectionConfig.score() has to change too."
        )
    divisor = float(match.group("divisor"))
    if divisor == 0:
        raise ValueError(f"rerank.formula divides fp_score by zero: {formula!r}")
    return {
        "weight_sam_score": float(match.group("sam")),
        "weight_render_mask_iou": float(match.group("mask")),
        "weight_render_box_iou": float(match.group("box")),
        "fp_score_divisor": divisor,
    }


DEFAULT_RERANK_WEIGHTS = rerank_weights_from_formula(DEFAULT_RERANK_FORMULA)

DEFAULT_SAM3_REFINEMENT_POLICY = str(_DEFAULTS["refinement"]["policy"])
DEFAULT_SAM3_REFINEMENT_LOW_MIOU = float(_DEFAULTS["refinement"]["low_miou"])
DEFAULT_SAM3_REFINEMENT_HIGH_MIOU = float(_DEFAULTS["refinement"]["high_miou"])
DEFAULT_SAM3_REFINEMENT_NMS_THRESHOLD = float(_DEFAULTS["refinement"]["nms_threshold"])

DEFAULT_MAX_VERTEX_ERROR_THRESHOLD_MM = float(_DEFAULTS["pose"]["max_vertex_error_threshold_mm"])
DEFAULT_MAX_VERTEX_ERROR_REQUIRED_RATE = float(_DEFAULTS["pose"]["max_vertex_error_required_rate"])
DEFAULT_POSE_MIN_VISIBLE_FRACTIONS = tuple(
    float(v) for v in _DEFAULTS["pose"]["min_visible_fractions"]
)

DEFAULT_MIN_VISIBLE_FRACTION = float(_DEFAULTS["ground_truth"]["min_visible_fraction"])
GT_RASTERIZER_NEAR_MM = float(_DEFAULTS["ground_truth"]["rasterizer_near_mm"])

DEFAULT_FS_MAX_WIDTH = int(_DEFAULTS["depth"]["foundation_stereo_max_width"])

DEFAULT_TOLERANCE_MM = float(_DEFAULTS["validation"]["tolerance_mm"])
DEFAULT_MIN_WITHIN_5MM = float(_DEFAULTS["validation"]["min_within_5mm"])
DEFAULT_CUTOFFS = [float(v) for v in _DEFAULTS["validation"]["cutoffs"]]

FOUNDATIONPOSE_ROOT_DEFAULT = foundationpose_root_default()
# Local date, deliberately -- see the note at evaluate.py's `generated_on`. This is report
# provenance for a person, not a timestamp anything computes with.
DEFAULT_DATE = date.today().isoformat()  # noqa: DTZ011
