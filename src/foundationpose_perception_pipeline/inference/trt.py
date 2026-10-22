#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequential TensorRT execution with host or borrowed device tensors."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from foundationpose_perception_pipeline.inference.trt_build import build_cached_engine

DEFAULT_DEVICE_ID: int = 0
MIN_ALLOCATION_BYTES: int = 1


def _cuda_check(result: tuple[Any, ...]) -> Any:
    """CUDA Python returns status codes rather than raising on runtime failures."""
    status, *values = result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime call failed: {status}")
    return values[0] if values else None


@dataclass(frozen=True)
class DeviceTensor:
    """Borrowed contiguous engine output, ready on return from infer().

    Keeps its producer alive. Valid only until that producer's next inference or
    release; consumers bind the allocation directly and never own or free it.
    """

    ptr: int
    shape: tuple[int, ...]
    dtype: np.dtype
    owner: TRTEngine
    generation: int

    def validate(self) -> None:
        if self.generation != self.owner._generation:
            raise ValueError("DeviceTensor expired after its producer ran again or was released")


class TRTEngine:
    """Sequential TensorRT executor with cached buffers and optional device outputs.

    Calls synchronize before returning, including device outputs, so another engine
    can consume them on its own stream. Instances are not safe for concurrent calls.
    """

    def __init__(
        self, engine_path: Path | str, device_id: int = DEFAULT_DEVICE_ID, *,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        models_dir: Path | str | None = None,
    ):
        self.engine_path = Path(engine_path).expanduser().resolve()
        self._buffers: dict[str, tuple[int, int]] = {}
        self._generation = 0
        self.stream = 0
        self.device_id = device_id
        _cuda_check(cudart.cudaSetDevice(device_id))
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        self._runtime = trt.Runtime(self._logger)
        self.engine: trt.ICudaEngine | None = None

        if self.engine_path.suffix.lower() == ".onnx":
            self.engine_path = build_cached_engine(
                self.engine_path, input_shapes=input_shapes, device_id=device_id, models_dir=models_dir
            )

        if self.engine_path.exists() and self.engine_path.stat().st_size > 0:
            with open(self.engine_path, "rb") as f:
                self.engine = self._runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT plan: {self.engine_path}. "
                "Supply a compatible plan or an ONNX source for automatic compilation."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {self.engine_path}")
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.stream = _cuda_check(cudart.cudaStreamCreate())

    def _get_buffer(self, name: str, nbytes: int) -> int:
        if name in self._buffers:
            ptr, cap = self._buffers[name]
            if cap >= nbytes:
                return ptr
            _cuda_check(cudart.cudaFree(ptr))
            del self._buffers[name]
        ptr = _cuda_check(cudart.cudaMalloc(max(nbytes, MIN_ALLOCATION_BYTES)))
        self._buffers[name] = (ptr, nbytes)
        return ptr

    def infer(
        self, feed_dict: dict[str, np.ndarray | DeviceTensor], *, device_outputs: bool = False
    ) -> dict[str, np.ndarray | DeviceTensor]:
        if self.engine is None or self.context is None:
            raise RuntimeError("TensorRT engine has been released")
        _cuda_check(cudart.cudaSetDevice(self.device_id))
        missing = set(self.input_names) - feed_dict.keys()
        if missing:
            raise ValueError(f"Missing engine inputs: {sorted(missing)}")
        # Invalidate earlier views before any buffer can be overwritten or resized.
        self._generation += 1
        for name in self.input_names:
            value = feed_dict[name]
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if isinstance(value, DeviceTensor):
                value.validate()
                if value.owner.device_id != self.device_id:
                    raise ValueError("DeviceTensor belongs to a different GPU")
                if value.dtype != dtype:
                    raise ValueError(f"Input {name} expects {dtype}, got device tensor {value.dtype}")
                shape, d_buf = value.shape, value.ptr
            else:
                arr = np.ascontiguousarray(value, dtype=dtype)
                shape = arr.shape
                d_buf = self._get_buffer(name, arr.nbytes)
                _cuda_check(cudart.cudaMemcpy(
                    d_buf, arr.ctypes.data, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                ))
            if not self.context.set_input_shape(name, shape):
                raise ValueError(f"Unsupported shape for {name}: {shape}")
            if not self.context.set_tensor_address(name, d_buf):
                raise RuntimeError(f"Failed to bind input {name}")

        outputs: dict[str, np.ndarray | DeviceTensor] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(f"Unresolved output shape for {name}: {shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            d_buf = self._get_buffer(name, int(np.prod(shape)) * dtype.itemsize)
            if not self.context.set_tensor_address(name, d_buf):
                raise RuntimeError(f"Failed to bind output {name}")
            outputs[name] = (
                DeviceTensor(d_buf, shape, dtype, self, self._generation)
                if device_outputs else np.empty(shape, dtype=dtype)
            )

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        # Complete the producer stream before host reads or another engine's bindings.
        _cuda_check(cudart.cudaStreamSynchronize(self.stream))

        if not device_outputs:
            for name, out_arr in outputs.items():
                d_buf = self._buffers[name][0]
                _cuda_check(cudart.cudaMemcpy(
                    out_arr.ctypes.data, d_buf, out_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
                ))

        return outputs

    def release(self) -> None:
        self._generation += 1
        _cuda_check(cudart.cudaSetDevice(self.device_id))
        for ptr, _ in self._buffers.values():
            _cuda_check(cudart.cudaFree(ptr))
        self._buffers.clear()
        if hasattr(self, "stream") and self.stream:
            _cuda_check(cudart.cudaStreamDestroy(self.stream))
            self.stream = 0

        self.context = None
        self.engine = None
        self._runtime = None

    def __del__(self) -> None:
        # Explicit release reports errors; destructors may run during CUDA teardown.
        with suppress(Exception):
            self.release()
