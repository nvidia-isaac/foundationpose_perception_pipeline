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


def _environment(device_id: int) -> tuple[str, str]:
    """The TensorRT version and GPU this process would build for."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    status, = cudart.cudaSetDevice(device_id)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot select GPU {device_id}: {status}")
    status, properties = cudart.cudaGetDeviceProperties(device_id)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Cannot inspect GPU {device_id}: {status}")
    return str(trt.__version__), bytes(properties.name).split(b"\0", 1)[0].decode()


def _key_spec(profiles: Profiles, precision: str, device_id: int) -> dict:
    """The cache key: what makes two plans mutually unusable, and is free to compute.

    Deliberately excludes the ONNX digest. Hashing the source to derive a *filename* means
    reading gigabytes before the cache can even be consulted, which for the SAM3 encoders
    dominates startup. Provenance still gets recorded -- see `_sidecar`, which is written at
    build time and checked on reuse -- it just is not on the lookup path.

    `profiles` and `tensorrt` are here because a mismatch in either yields a plan that cannot
    be deserialised or cannot accept the caller's shapes, and both are free to obtain.
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    tensorrt_version, gpu = _environment(device_id)
    return {"profiles": profiles, "tensorrt": tensorrt_version, "gpu": gpu}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    external = path.with_name(path.name + "_data")
    if external.is_file():
        with external.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sidecar(source: Path, key: dict, precision: str, tf32: bool, workspace_mb: int | None) -> dict:
    """Everything worth knowing about a built plan, including the settings not in its key.

    Written once per build, where a sha256 is negligible beside compiling an engine. `size`
    and `mtime_ns` exist so a reader can detect a re-exported ONNX without re-hashing it.
    """
    stat = source.stat()
    return dict(
        key,
        model_sha256=_file_digest(source),
        model_size=stat.st_size,
        model_mtime_ns=stat.st_mtime_ns,
        precision=precision,
        tf32=tf32,
        workspace_mb=workspace_mb,
    )


def _check_reusable(
    destination: Path, source: Path, precision: str, tf32: bool, workspace_mb: int | None
) -> None:
    """Reject a cached plan whose build settings differ from the ones asked for.

    `precision`, `tf32` and `workspace_mb` are not in the cache key, so a plan built with
    different values lives at this same path. Without this check an fp16 plan would be served
    to a caller that asked for fp32 -- no error, just quietly different numerics. The sidecar
    makes that detectable for free.

    A re-exported ONNX is caught the same way, comparing size and mtime rather than re-reading
    the file; the digest is only consulted if those already disagree.
    """
    import logging

    sidecar = destination.with_suffix(".plan.json")
    if not sidecar.is_file():
        return  # Plan supplied by hand, e.g. staged into a container image. Trust the caller.
    try:
        recorded = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return

    for field, wanted in (("precision", precision), ("tf32", tf32), ("workspace_mb", workspace_mb)):
        if field in recorded and recorded[field] != wanted:
            raise RuntimeError(
                f"{destination.name} was built with {field}={recorded[field]!r} but "
                f"{wanted!r} was requested. These settings are not part of the cache key, so "
                f"the two cannot coexist: delete the plan and rebuild, or pass force=True."
            )

    stat = source.stat()
    if "model_size" in recorded and (
        recorded["model_size"] != stat.st_size or recorded.get("model_mtime_ns") != stat.st_mtime_ns
    ):
        if recorded.get("model_sha256") != _file_digest(source):
            raise RuntimeError(
                f"{destination.name} was built from a different {source.name}. Delete the "
                f"stale plan and rebuild, or pass force=True."
            )
        logging.getLogger(__name__).debug("%s touched but unchanged; reusing %s", source.name, destination.name)


def _destination(source: Path, key: dict, models_dir: Path | str | None) -> Path:
    fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:FINGERPRINT_LENGTH]
    directory = ModelPaths.configured(models_dir).engine_cache
    return directory / f"{source.stem}__{fingerprint}.plan"


def engine_path_for_model(
    onnx_path: Path | str, *, profiles: Profiles | None = None, precision: str = DEFAULT_PRECISION,
    tf32: bool = True, workspace_mb: int | None = None, device_id: int = DEFAULT_DEVICE_ID, models_dir: Path | str | None = None,
) -> Path:
    """Where the plan for this model belongs. Reads nothing; safe before the ONNX exists.

    `tf32` and `workspace_mb` are accepted so this mirrors `build_cached_engine`, but they no
    longer affect the path: they are recorded in the sidecar and verified on reuse instead of
    being encoded in the name. `precision` is likewise validated here but not part of the key.
    """
    del tf32, workspace_mb  # Recorded in the sidecar, checked by _check_reusable.
    source = Path(onnx_path).expanduser().resolve()
    return _destination(source, _key_spec(profiles or {}, precision, device_id), models_dir)


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
    key = _key_spec(profiles, precision, device_id)
    destination = _destination(source, key, models_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and destination.is_file() and destination.stat().st_size:
            _check_reusable(destination, source, precision, tf32, workspace_mb)
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
        destination.with_suffix(".plan.json").write_text(
            json.dumps(_sidecar(source, key, precision, tf32, workspace_mb), indent=2), encoding="utf-8"
        )
        return destination
