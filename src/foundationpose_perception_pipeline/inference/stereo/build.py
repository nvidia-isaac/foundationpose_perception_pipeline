#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a TensorRT engine from a TAO deployable FoundationStereo ONNX export.

TAO Deploy has no ONNX execution path -- `DepthNetInferencer` deserializes an engine -- so this
build is a required one-time step per machine, not an optimisation. An engine is specific to the
source ONNX, the TensorRT version, the GPU architecture, the precision and the shape profile;
none of that is recorded inside the file in a form we can check cheaply, so it is recorded in the
file *name* (the cache key) and in a sidecar JSON. A stale engine silently producing wrong-shaped
or wrong-precision output is the failure this exists to prevent.

**Precision.** FP32 by default, deliberately. TAO's own schema default is also FP32
(`DepthNetTrtConfig.data_type`); only the spec *template* under `cv/depth_net/specs/` says fp16,
and that template is not shipped in the wheel. FP16 is available and is roughly twice as fast,
but a 5 mm pose acceptance bar leaves little room to absorb a silent precision change -- measure
it against an FP32 engine before adopting it.

Note that "FP32" in TensorRT still means TF32 for convolutions and matmuls, because
`BuilderFlag.TF32` is on by default and TAO Deploy never clears it. That matches what
onnxruntime's CUDA EP does today, so it is not a regression against the current baseline; pass
`--no-tf32` if a bit-level comparison is needed.

**Shape profile.** `--shape HxW` builds a static profile (min=opt=max) and is the default
recommendation. A dynamic profile is supported via `--min/--opt/--max`, but two things argue
against a generous one: `DepthNetInferencer` allocates host and device buffers at the profile's
MAX shape, so every scene pays for the largest shape allowed; and TAO's own export config warns
that dynamic H/W is unsafe for FoundationStereo, whose DINOv2 backbone constant-folds the
trace-time patch count into its positional-embedding shape arithmetic. That warning is aimed at
re-exporting rather than at the already-dynamic published ONNX, but it points at where wrong answers
would come from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16", "bf16")
DIVISIBILITY = 32


@dataclass(frozen=True)
class ShapeProfile:
    """The optimisation profile's spatial bounds, as `(height, width)` triples."""

    min_hw: tuple[int, int]
    opt_hw: tuple[int, int]
    max_hw: tuple[int, int]

    @classmethod
    def static(cls, height: int, width: int) -> ShapeProfile:
        """A profile pinned to one shape -- min = opt = max."""
        return cls((height, width), (height, width), (height, width))

    @property
    def is_static(self) -> bool:
        return self.min_hw == self.opt_hw == self.max_hw

    def validate(self) -> None:
        """Reject shapes the network cannot consume, and bounds that are not ordered."""
        for name, (height, width) in (("min", self.min_hw), ("opt", self.opt_hw), ("max", self.max_hw)):
            if height % DIVISIBILITY or width % DIVISIBILITY:
                raise ValueError(
                    f"{name} shape {height}x{width} is not a multiple of {DIVISIBILITY}. "
                    "FoundationStereo downsamples by 32; a non-multiple makes its skip "
                    "connections disagree at runtime."
                )
        for axis, index in (("height", 0), ("width", 1)):
            low, opt, high = self.min_hw[index], self.opt_hw[index], self.max_hw[index]
            if not low <= opt <= high:
                raise ValueError(f"{axis} bounds must satisfy min <= opt <= max, got {low} <= {opt} <= {high}")

    def label(self) -> str:
        """Short form for the cache key."""
        if self.is_static:
            return f"{self.opt_hw[0]}x{self.opt_hw[1]}"
        return f"{self.min_hw[0]}x{self.min_hw[1]}-{self.max_hw[0]}x{self.max_hw[1]}"


def gpu_name() -> str:
    """The GPU this engine is being built for, as reported by the driver.

    Part of the cache key because an engine built for one architecture will not deserialize on
    another -- and because the failure, when it happens, is far from the cause.
    """
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            # Explicitly not check=True: a missing or failing nvidia-smi is handled two lines
            # below by falling back to "unknown-gpu", which keeps engine naming working on a
            # box without the driver utility rather than aborting the build.
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown-gpu"
    if probe.returncode != 0 or not probe.stdout.strip():
        return "unknown-gpu"
    return probe.stdout.strip().splitlines()[0].strip().replace(" ", "-")


def tensorrt_version() -> str:
    """The TensorRT version in this environment."""
    import tensorrt as trt

    return str(trt.__version__)


def file_sha256(path: Path) -> str:
    """Hash a file in chunks -- the ONNX exports run to hundreds of MB."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_path_for(onnx_path: Path, profile: ShapeProfile, precision: str, out_dir: Path | None = None) -> Path:
    """Where the engine for this (model, shape, precision, TRT, GPU) combination belongs.

    Encoding the key in the filename rather than only in the sidecar means two configurations
    coexist instead of overwriting each other -- per-dataset shapes, or an fp16 A/B against the
    fp32 baseline.
    """
    directory = Path(out_dir) if out_dir is not None else onnx_path.parent
    stem = onnx_path.stem
    key = f"{profile.label()}_{precision}_trt{tensorrt_version()}_{gpu_name()}"
    return directory / f"{stem}__{key}.engine"


def sidecar_path_for(engine_path: Path) -> Path:
    """The provenance file that sits beside an engine."""
    return engine_path.with_suffix(engine_path.suffix + ".json")


def read_sidecar(engine_path: Path) -> dict | None:
    """Return the recorded provenance for an engine, or None if it has none."""
    path = sidecar_path_for(engine_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_engine(
    onnx_path: Path | str,
    *,
    profile: ShapeProfile,
    precision: str = "fp32",
    out_dir: Path | None = None,
    workspace_mb: int | None = None,
    tf32: bool = True,
    force: bool = False,
) -> Path:
    """Build (or reuse) the engine for one ONNX + profile + precision, and return its path.

    Built with TensorRT's Python API directly rather than through TAO's `gen_trt_engine`
    entrypoint. The two produce the same engine -- `EngineBuilder` is a wrapper over these same
    calls -- but `gen_trt_engine` imports `nvidia_tao_deploy.utils.decoding` at module scope,
    which imports `eff`, which is not on public PyPI. Requiring a separate package-index login to
    build an engine from an unencrypted ONNX would be a real cost for no benefit. Inference still runs
    through TAO Deploy's `DepthNetInferencer`, which is where the supported path actually is.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_path).expanduser().resolve()
    if not onnx_path.exists():
        # Worth more than the path: on a fresh machine this is almost always "the export was
        # never fetched", and the argument is the only thing that locates it -- there is no
        # search path and no conventional directory to have got wrong.
        raise FileNotFoundError(
            f"ONNX export not found: {onnx_path}. Fetch the pipeline's FP32 baseline export from "
            "https://huggingface.co/nvidia/c-foundationstereo-s/blob/main/onnx/"
            "deployable_foundation_stereo_s_dynamic.onnx "
            "and point --onnx at it; it may live anywhere. Use --precision fp32; "
            "the model card reports FP16 TensorRT conversion failure for the dynamic export."
        )
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    profile.validate()

    engine_path = engine_path_for(onnx_path, profile, precision, out_dir)
    onnx_sha = file_sha256(onnx_path)

    if engine_path.exists() and not force:
        recorded = read_sidecar(engine_path)
        if recorded is None:
            raise RuntimeError(
                f"{engine_path} exists but has no sidecar, so its provenance is unknown. "
                "Delete it or pass force=True."
            )
        if recorded.get("onnx_sha256") != onnx_sha:
            raise RuntimeError(
                f"{engine_path} was built from a different ONNX "
                f"({recorded.get('onnx_sha256', '?')[:12]} vs {onnx_sha[:12]}). "
                "Delete it or pass force=True."
            )
        logger.info("reusing engine %s", engine_path.name)
        return engine_path

    logger.info(
        "building %s engine %s from %s", precision, profile.label(), onnx_path.name
    )
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    # Left at TensorRT's own default (the whole device) unless asked otherwise. Capping this is
    # not the harmless safety measure it looks like: at 4096 MB this model fails to build at all
    # with "Error Code 10: Could not find any implementation for node ... In computeCosts",
    # because every tactic for one of its Concat subgraphs needs more scratch than that. The
    # limit is scratch space during inference, not a cap on the engine's own memory, so a
    # too-small value trades an unbuildable engine for nothing.
    if workspace_mb:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    # fp32: no flag. TensorRT's default is already fp32 kernels -- but with TF32 permitted for
    # convolutions and matmuls unless it is explicitly cleared.
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    optimization_profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        optimization_profile.set_shape(
            tensor.name,
            (1, 3, *profile.min_hw),
            (1, 3, *profile.opt_hw),
            (1, 3, *profile.max_hw),
        )
    config.add_optimization_profile(optimization_profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)

    provenance = {
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_sha,
        "precision": precision,
        "tf32": bool(tf32),
        "tensorrt": tensorrt_version(),
        "gpu": gpu_name(),
        "inputs": [network.get_input(i).name for i in range(network.num_inputs)],
        "outputs": [network.get_output(i).name for i in range(network.num_outputs)],
        **{f"shape_{k}": list(v) for k, v in asdict(profile).items()},
    }
    sidecar_path_for(engine_path).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("wrote %s (%.1f MB)", engine_path, engine_path.stat().st_size / (1 << 20))
    return engine_path


def parse_shape(text: str) -> tuple[int, int]:
    """Parse an `HxW` argument."""
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected a shape as HxW, got {text!r}")
    return int(parts[0]), int(parts[1])


def padded_shape(height: int, width: int) -> tuple[int, int]:
    """Round each dimension up to a multiple of 32, which is what the network can consume.

    A primitive, and on its own NOT the shape an engine should be built for: padding the width
    changes the width the model sees, and the runtime preserves aspect ratio when fitting to it.
    Use :func:`model_shape_for_rectified`, which accounts for that.
    """
    return (
        height + (-height) % DIVISIBILITY,
        width + (-width) % DIVISIBILITY,
    )


def model_shape_for_rectified(
    rect_height: int, rect_width: int, max_width: int | None = None
) -> tuple[int, int]:
    """The engine input shape a rectified pair of this size needs, so nothing is cropped.

    Padding the two dimensions independently is the obvious thing to do and it is wrong, because
    the runtime does not pad the width -- it RESIZES to it. `stereo/tao.py:fit_to_model` scales
    the pair by width to the engine's width, keeps the aspect ratio, and then pads or crops the
    height. So the height the engine must accommodate follows from the PADDED width:

        model_w  = pad32(rect_width)
        scaled_h = round(rect_height * model_w / rect_width)      <- what the runtime produces
        model_h  = pad32(scaled_h)

    Worked example, a 720x540 rectified pair: the width pads to 736, which scales the height to
    552, so the engine needs 576 rows. Padding independently would give 544 and the runtime would
    crop 8 rows off the bottom of every frame -- quietly, since a crop is a warning and not an
    error. The two agree whenever the rectified width is already a multiple of 32, which is why
    this only bites some rigs.

    `max_width` caps the width first, mirroring the depth stage. Note the cap applies to DYNAMIC
    engines only -- a fixed-shape engine resizes to its own input regardless (`stereo/depth.py`),
    which is exactly the resize this function is sizing for.
    """
    width = min(rect_width, max_width) if max_width else rect_width
    model_w = width + (-width) % DIVISIBILITY
    scaled_h = round(rect_height * model_w / rect_width)
    return scaled_h + (-scaled_h) % DIVISIBILITY, model_w
