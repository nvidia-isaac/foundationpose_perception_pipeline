#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared TensorRT compilation for runtime cache misses and stereo prebuilding.

GPU dependencies are imported only when a build or cache lookup is requested.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from foundationpose_perception_pipeline.inference.models import ModelPaths

BYTES_PER_MIB: int = 1 << 20
CHUNK_SIZE_BYTES: int = BYTES_PER_MIB
FINGERPRINT_LENGTH: int = 20
DEFAULT_PRECISION: str = "fp32"
SUPPORTED_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")
DEFAULT_DEVICE_ID: int = 0

Shape = tuple[int, ...]
Profiles = dict[str, tuple[Shape, Shape, Shape]]


def _build_spec(source: Path, profiles: Profiles, precision: str, tf32: bool,
                workspace_mb: int | None, device_id: int) -> dict:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    status, = cudart.cudaSetDevice(device_id)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot select GPU {device_id}: {status}")
    status, properties = cudart.cudaGetDeviceProperties(device_id)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot inspect GPU {device_id}: {status}")
    gpu = bytes(properties.name).split(b"\0", 1)[0].decode()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    ext_data = source.with_name(source.name + "_data")
    if ext_data.is_file():
        with ext_data.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
                digest.update(chunk)
    return {
        "model_sha256": digest.hexdigest(), "profiles": profiles, "precision": precision,
        "tf32": tf32, "workspace_mb": workspace_mb, "tensorrt": trt.__version__, "gpu": gpu,
    }


def _destination(source: Path, spec: dict, models_dir: Path | str | None) -> Path:
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:FINGERPRINT_LENGTH]
    directory = ModelPaths.configured(models_dir).engine_cache
    return directory / f"{source.stem}__{fingerprint}.plan"


def engine_path_for_model(
    onnx_path: Path | str, *, profiles: Profiles | None = None, precision: str = DEFAULT_PRECISION,
    tf32: bool = True, workspace_mb: int | None = None, device_id: int = DEFAULT_DEVICE_ID, models_dir: Path | str | None = None,
) -> Path:
    source = Path(onnx_path).expanduser().resolve()
    spec = _build_spec(source, profiles or {}, precision, tf32, workspace_mb, device_id)
    return _destination(source, spec, models_dir)


def build_cached_engine(
    onnx_path: Path | str, *, input_shapes: dict[str, Shape] | None = None,
    profiles: Profiles | None = None, precision: str = DEFAULT_PRECISION, tf32: bool = True,
    workspace_mb: int | None = None, device_id: int = DEFAULT_DEVICE_ID, models_dir: Path | str | None = None, force: bool = False,
) -> Path:
    """Build a fingerprinted plan, or reuse it. Explicit input_shapes build static profiles."""
    import fcntl
    import logging
    import os
    import tempfile

    import tensorrt as trt

    if profiles is not None and input_shapes is not None:
        raise ValueError("Specify profiles or input_shapes, not both")
    profiles = profiles or {name: (shape, shape, shape) for name, shape in (input_shapes or {}).items()}
    source = Path(onnx_path).expanduser().resolve()
    spec = _build_spec(source, profiles, precision, tf32, workspace_mb, device_id)
    destination = _destination(source, spec, models_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and destination.is_file() and destination.stat().st_size:
            return destination
        logging.getLogger(__name__).info("Building TensorRT engine %s from %s", destination, source)
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(source)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Cannot parse {source}:\n{errors}")
        config = builder.create_builder_config()
        if precision != "fp32":
            config.set_flag(trt.BuilderFlag.FP16 if precision == "fp16" else trt.BuilderFlag.BF16)
        if not tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        if workspace_mb is not None:
            if workspace_mb <= 0:
                raise ValueError("workspace_mb must be positive")
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * BYTES_PER_MIB)
        profile = builder.create_optimization_profile()
        has_dynamic = False
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shapes = profiles.get(tensor.name)
            if any(dim < 0 for dim in tensor.shape):
                if shapes is None:
                    raise ValueError(f"A concrete profile is required for {tensor.name} in {source}")
                profile.set_shape(tensor.name, *shapes)
                has_dynamic = True
            elif shapes is not None and any(tuple(shape) != tuple(tensor.shape) for shape in shapes):
                raise ValueError(f"Profile for {tensor.name} disagrees with fixed ONNX shape {tensor.shape}")
        if has_dynamic:
            config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT failed to build {source}")
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".plan.tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(serialized)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        destination.with_suffix(".plan.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return destination
