#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FoundationStereo through TAO Deploy: the commercially-licensable depth path.

The `deployable_*` exports from
https://huggingface.co/nvidia/c-foundationstereo-s carry the terms on that
model page. This module exists so that model runs the way NVIDIA supports running it: a
TensorRT engine, driven by TAO Deploy's own `DepthNetInferencer`.

Two consequences, and they are the point rather than side effects:

- **No FoundationStereo source.** An engine is executed by TensorRT alone, so nothing here
  imports FoundationStereo's model classes -- no checkout is needed, only the model files.
- **No second environment.** TensorRT and numpy, no torch. It runs in the pipeline's own venv.

Engines are not portable across GPU architecture or TensorRT version. Build one with
`tools/build_tao_engine.py`; see `build.py` for the cache key that keeps a stale one from being
picked up silently.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# TAO Deploy's CUDA context, parked off-thread. See `tao_context`.
_TAO_CONTEXT: Any = None

# Every engine `load_engine` has handed out, so `release_engines` can free their device memory.
# A plain list rather than anything clever: `lru_cache` does not expose its values, and reaching
# into its internals to get them would break on a CPython change.
_LOADED_ENGINES: list[StereoEngine] = []

# TAO Deploy's own dataloader normalises with these
# (`nvidia_tao_deploy/cv/depth_net/dataloader.py` -> `preprocess_input(mode='torch')`), and they
# are the values the deployable export is validated against. See `normalize_for_model`.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# FoundationStereo downsamples by 32; an input that is not a multiple of this in both dimensions
# makes the skip connections disagree ("Concat ... mismatched dimensions of 135 and 136").
DIVISIBILITY = 32

# The names in the deployable ONNX, and therefore in any engine built from it. `infer()` looks
# both up literally, so these are a contract with TAO Deploy rather than a convenience.
LEFT_INPUT = "left_image"
RIGHT_INPUT = "right_image"


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to [0,1] and apply ImageNet statistics.

    Deliberately not configurable. Normalisation is a per-model convention rather than a knob:
    it serves three model families with two opposite conventions -- the repo exports and the
    `.pth` checkpoint normalise *inside* the graph and must be fed raw 0-255. A TAO deployable
    export does not, and TAO's own dataloader feeds it exactly this. Getting it wrong roughly
    doubles depth error without ever failing, so the one model this module serves gets the one
    convention its vendor validates.

    This is a hand-rolled copy of TAO's own `preprocess_input(mode='torch')` -- theirs takes a
    file path and resizes, ours takes the rectified pair already in memory. Change one and you
    have silently forked from the convention the export was trained under.
    """
    return (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def _import_tao_inferencer() -> Any:
    """Import TAO Deploy, then get its CUDA context out of the pipeline's way.

    **This is the load-bearing function of the whole in-process design.** TAO Deploy's
    `inferencer/utils.py` does `import pycuda.autoinit` at module scope, and `autoinit` calls
    `make_default_context()`, which creates a *new* CUDA context with `cuCtxCreate` and makes it
    current on this thread. Torch does not use that context -- it uses the device's **primary**
    context -- and CUDA streams and events are not valid across contexts. So the moment TAO
    Deploy is imported, every torch stream and event created earlier belongs to a context that is
    no longer current, and the next torch call that touches one dies with:

        terminate called after throwing an instance of 'c10::AcceleratorError'
          what():  CUDA error: invalid resource handle

    Measured, not theorised: depth completed, then FoundationPose aborted the process on its
    first CUDA event. The failure is a native abort with no Python traceback pointing anywhere
    near here, which is what makes it worth this much comment.

    The fix is to stop leaving pycuda's context current. Pop it immediately after import and
    push it only around the calls that need it (`tao_context`), so the primary context is current
    everywhere else and torch never notices this backend exists.

    The atexit handler pushes the context back before `pycuda.autoinit`'s own teardown runs.
    autoinit registered `context.pop()` at import time; ours registers later and atexit runs
    LIFO, so ours runs first and leaves the stack in the state autoinit expects. Without it,
    every process using this backend ends with an exception during interpreter shutdown.
    """
    global _TAO_CONTEXT  # noqa: PLW0603 -- one CUDA/TensorRT context per process, by design

    from nvidia_tao_deploy.cv.depth_net.inferencer import DepthNetInferencer

    if _TAO_CONTEXT is None:
        import pycuda.autoinit

        _TAO_CONTEXT = pycuda.autoinit.context
        _TAO_CONTEXT.pop()
        atexit.register(_restore_context_for_teardown)
        logger.debug("TAO Deploy CUDA context parked off-thread; torch keeps the primary context")
    return DepthNetInferencer


def _restore_context_for_teardown() -> None:
    """Make the context current again so `pycuda.autoinit`'s atexit pop is balanced."""
    if _TAO_CONTEXT is not None:
        with contextlib.suppress(Exception):
            _TAO_CONTEXT.push()


@contextlib.contextmanager
def tao_context():
    """Make TAO Deploy's CUDA context current for the duration of the block.

    Every allocation and launch that TAO Deploy performs -- engine deserialization, buffer
    allocation, and each `infer()` -- has to happen inside this. Anything else in the process
    (torch, FoundationPose) runs outside it, on the primary context. See `_import_tao_inferencer`.
    """
    if _TAO_CONTEXT is None:
        yield
        return
    _TAO_CONTEXT.push()
    try:
        yield
    finally:
        _TAO_CONTEXT.pop()


@dataclass
class StereoEngine:
    """A loaded TAO Deploy TensorRT engine for FoundationStereo.

    `fixed_hw` is None for an engine whose optimisation profile has dynamic spatial dimensions;
    it consumes the rectified pair at whatever size it is handed. A static-profile engine pins
    the input size, exactly as a fixed-shape ONNX export does, and `fit_to_model` resizes to it.
    """

    inferencer: Any
    fixed_hw: tuple[int, int] | None
    engine_path: Path

    @classmethod
    def load(cls, engine_path: Path | str) -> StereoEngine:
        """Deserialize an engine and record whether its input shape is fixed.

        The TAO Deploy import is deferred to here, not done at module scope, for two reasons:
        `nvidia-tao-deploy` is an optional extra (`pip install '.[tao]'`), so a checkout without
        it must still be able to import `foundationpose_perception_pipeline`; and TAO Deploy's inferencer
        module does `import pycuda.autoinit`, which creates a CUDA context as a side effect of
        being imported. A run that never asks for this backend should pay neither cost.

        A third, cosmetic consequence: something on that import chain calls
        `logging.basicConfig`, so a process that has not configured logging by this point
        inherits TAO's `[TAO Toolkit]` line format for everything afterwards. Harmless, and it
        does not override a configuration already in place -- entry points that call
        `basicConfig` first keep their own format.
        """
        engine_path = Path(engine_path).expanduser()
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}. Build one with "
                "tools/build_tao_engine.py -- engines are machine-specific and are not committed."
            )

        depth_net_inferencer = _import_tao_inferencer()
        try:
            with tao_context():
                inferencer = depth_net_inferencer(str(engine_path), batch_size=1)
        except Exception as exc:
            # Almost always memory, and the message TensorRT gives ("Error Code 2: OutOfMemory
            # (Requested size was 14124467200 bytes)") does not say what to do about it.
            #
            # This model's execution context needs 10-14 GB of scratch, and that is a property of
            # the model rather than something a build flag can shrink -- capping the build
            # workspace does not reduce it, it only makes the engine unbuildable. The scratch
            # scales with the profile's MAX shape, so a dynamic profile costs more than a static
            # one sized to the dataset: measured 13.5 GB for 320x736-544x896 against 10.6 GB for
            # a static 480x800. On a 32 GB card shared with SAM3 and FoundationPose, that
            # difference decides whether the run fits.
            raise RuntimeError(
                f"Could not create an execution context for {engine_path.name}: {exc}\n"
                "If this is an out-of-memory error, the engine's scratch does not fit alongside "
                "the rest of the pipeline. Build a STATIC engine sized to this dataset rather "
                "than a dynamic-profile one -- the scratch scales with the profile's max shape:\n"
                "  tools/build_tao_engine.py --onnx <export>.onnx --shape-from-scene <scene_dir>"
            ) from exc
        tensor = inferencer.input_tensors[0]
        dynamic = getattr(tensor, "optimization_profile", None) is not None
        _, height, width = tensor.shape  # C, H, W
        fixed_hw = None if dynamic else (int(height), int(width))
        engine = cls(inferencer=inferencer, fixed_hw=fixed_hw, engine_path=engine_path)
        _LOADED_ENGINES.append(engine)
        logger.info(
            "loaded engine %s (%s input %s)",
            engine_path.name,
            "dynamic" if dynamic else "fixed",
            "any" if fixed_hw is None else f"{fixed_hw[1]}x{fixed_hw[0]}",
        )
        return engine

    def infer_disparity(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray, pre_shift_px: int = 0
    ) -> np.ndarray:
        """Run a rectified RGB pair through the engine; return disparity at the input resolution.

        Padding is on the right and bottom edges only. That is not arbitrary: it leaves the image
        origin -- and therefore the intrinsics -- unchanged, so the disparity can be cropped back
        with no geometric correction.

        `pre_shift_px` slides the right image right by that many columns before inference, so the
        network only has to resolve `d - pre_shift_px`, and adds it back to the result. Disparity
        is `x_left - x_right`, so moving the right image right subtracts from every correspondence
        -- it is a change of origin for the search, not a change of geometry.

        This exists because a cost-volume network has a bounded disparity range (Fast-Foundation-
        Stereo documents 192 px) and a wide-baseline rig routinely exceeds it -- at a 0.43 m
        baseline and 800 px input the majority of scenes can sit past the limit. Those scenes come
        back with a confident, badly wrong small disparity rather than a failure. Shifting centres
        the network's window on the range the rig actually produces; measured where it was needed
        it cut object depth error by roughly two thirds.

        The caller is responsible for choosing a value no larger than the farthest visible
        surface's true disparity; see `stereo/depth.py`. Over-shooting drives distant pixels to
        non-positive disparity, which `disparity_to_depth_m` turns into NaN -- missing data rather
        than a plausible wrong number.
        """
        if left_rgb.shape != right_rgb.shape:
            raise ValueError(f"Rectified pair shape mismatch: {left_rgb.shape} vs {right_rgb.shape}")
        if pre_shift_px < 0:
            raise ValueError(f"pre_shift_px must be non-negative, got {pre_shift_px}")

        if pre_shift_px:
            shifted = np.empty_like(right_rgb)
            shifted[:, pre_shift_px:] = right_rgb[:, : right_rgb.shape[1] - pre_shift_px]
            # Replicate the leftmost column into the vacated strip rather than filling black: a
            # hard edge there is a strong false feature for a matcher.
            shifted[:, :pre_shift_px] = right_rgb[:, :1]
            right_rgb = shifted

        target_h, target_w = left_rgb.shape[:2]
        pad_h = (-target_h) % DIVISIBILITY
        pad_w = (-target_w) % DIVISIBILITY
        if pad_h or pad_w:
            left_rgb = cv2.copyMakeBorder(left_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            right_rgb = cv2.copyMakeBorder(right_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

        feed = {
            LEFT_INPUT: normalize_for_model(left_rgb)[None].transpose(0, 3, 1, 2),
            RIGHT_INPUT: normalize_for_model(right_rgb)[None].transpose(0, 3, 1, 2),
        }
        # Inside TAO Deploy's own CUDA context: its buffers and stream live there, and torch's
        # do not. See `_import_tao_inferencer` for what happens when this is skipped.
        with tao_context():
            disparity = np.asarray(self.inferencer.infer(feed)).squeeze().astype(np.float32)

        if pad_h or pad_w:
            disparity = disparity[:target_h, :target_w]
        if pre_shift_px:
            disparity = disparity + pre_shift_px
        if disparity.shape != (target_h, target_w):
            # Disparity is measured in pixels, so it scales with width when the model returns a
            # different size than it was given.
            scale = target_w / disparity.shape[1]
            disparity = cv2.resize(disparity, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            disparity *= scale
        return disparity


@lru_cache(maxsize=2)
def load_engine(engine_path: str) -> StereoEngine:
    """Load an engine, reusing it for subsequent calls with the same path.

    Cached on the path string rather than the `Path` so that two equal paths hit the same entry.
    `maxsize=2` because a run uses one engine; the second slot only exists so an A/B does not
    thrash.

    **Caching is not free here, and the caller decides.** A loaded engine holds its execution
    context's scratch memory for as long as it lives, and for this model that is a lot: 13.5 GB
    for the dynamic 320x736-544x896 profile, 10.6 GB for the static 480x800 one, on a 32 GB card
    that also has to hold SAM3 and FoundationPose. Keeping it loaded across a whole run makes the
    peak the *sum* of the stages rather than the max, which is a `torch.OutOfMemoryError` in the
    pose stage a few scenes in. Reloading costs about 1.4 s. See `release_engines`.

    That scratch requirement is a property of the model, not a tuning mistake -- capping the
    build-time workspace does not shrink it, it just makes the engine unbuildable -- 4 GB and
    6 GB workspace caps both fail outright.
    """
    return StereoEngine.load(engine_path)


def release_engines() -> None:
    """Free every cached engine's device memory now.

    TensorRT does not return an execution context's scratch when the Python object merely goes out
    of scope -- measured: `del` on the `StereoEngine` left all 15 GB allocated. The context and
    the engine have to be dropped explicitly, and pycuda's device allocations only run their
    deallocators once nothing references them, so the collection is forced here rather than left
    to chance.

    Called by `inference/depth.py` after each scene, so the depth stage's memory is gone before
    the pose stage asks for its own.
    """
    load_engine.cache_clear()
    inferencers = [e.inferencer for e in _LOADED_ENGINES if e.inferencer is not None]
    for engine in _LOADED_ENGINES:
        engine.inferencer = None
    _LOADED_ENGINES.clear()
    if not inferencers:
        return

    # Free TAO Deploy's buffers *explicitly*, and only those.
    #
    # The obvious implementation -- drop the last reference and `gc.collect()` inside
    # `tao_context()`, letting `TRTInferencer.__del__` do the work -- reintroduces the exact
    # crash that `_import_tao_inferencer` exists to prevent. `gc.collect()` is global: it runs
    # *every* pending destructor in the process, so torch objects awaiting collection release
    # their memory while a foreign CUDA context is current, and the next torch call dies with
    # "invalid resource handle". Measured -- depth for scene 0 completed, then the process
    # aborted on FoundationPose's first CUDA event.
    #
    # So: free the device allocations by hand, inside the context they belong to, touching
    # nothing else. Emptying `inputs`/`outputs` afterwards keeps `__del__` from double-freeing
    # when it eventually runs -- it iterates those lists, so empty is the safe state, whereas
    # None would make it raise during interpreter teardown.
    with tao_context():
        for inferencer in inferencers:
            for buffer in (*getattr(inferencer, "inputs", []), *getattr(inferencer, "outputs", [])):
                with contextlib.suppress(Exception):
                    buffer.device.free()
            inferencer.inputs = []
            inferencer.outputs = []
            for attribute in ("context", "engine", "trt_runtime", "stream"):
                with contextlib.suppress(Exception):
                    setattr(inferencer, attribute, None)
    del inferencers
    # Outside the context: this is the pipeline's own garbage, not TAO's.
    gc.collect()


@dataclass(frozen=True)
class FittedPair:
    """A rectified pair prepared for a static engine, and how to undo that preparation."""

    left: np.ndarray
    right: np.ndarray
    width_scale: float
    """Factor the resulting disparity must be *divided* by to return to rectified resolution."""

    content_height: int
    """Rows of the engine's input that hold image rather than padding; crop disparity to this."""


def fit_to_model(engine: StereoEngine, left_rgb: np.ndarray, right_rgb: np.ndarray) -> FittedPair:
    """Prepare a rectified pair for a static engine: scale by width, then pad the height.

    **Scale and pad, not stretch.** The obvious implementation resizes straight to the engine's
    HxW, and that is wrong in a way that does not announce itself: the rectified pair's aspect
    ratio is whatever `stereoRectify` produced, the engine's is whatever it was built for, and
    forcing one into the other stretches the image vertically. Disparity survives a *uniform*
    scale and survives a vertical stretch in principle -- it is a horizontal quantity -- but the
    model does not: it sees objects at the wrong aspect and the cost volume it matches on is not
    the one it was trained for. Measured against the same weights under onnxruntime, a 6% stretch
    moved the p99 depth difference into metres while the median stayed under a millimetre, i.e.
    it degrades exactly the edges that matter and nowhere a summary statistic would notice.

    Scaling by width and replicate-padding the remaining rows keeps the aspect exact and matches
    what the geometry expects when it pads a rectified pair to a multiple of 32 -- which is what
    keeps a comparison between two runtimes from becoming a comparison between two preprocessing
    schemes.

    Note the direction of `width_scale`: TAO's own evaluator rescales the ground truth *up* to
    engine resolution instead (`stereo_evaluator.py`). Ours is the direction that keeps the depth
    map on the base camera's grid, which is what FoundationPose consumes.
    """
    if engine.fixed_hw is None:
        return FittedPair(left_rgb, right_rgb, 1.0, left_rgb.shape[0])

    model_h, model_w = engine.fixed_hw
    src_h, src_w = left_rgb.shape[:2]
    if (src_h, src_w) == (model_h, model_w):
        return FittedPair(left_rgb, right_rgb, 1.0, model_h)

    width_scale = model_w / src_w
    scaled_h = round(src_h * width_scale)
    interpolation = cv2.INTER_AREA if width_scale < 1.0 else cv2.INTER_LINEAR
    left = cv2.resize(left_rgb, (model_w, scaled_h), interpolation=interpolation)
    right = cv2.resize(right_rgb, (model_w, scaled_h), interpolation=interpolation)

    if scaled_h == model_h:
        return FittedPair(left, right, width_scale, model_h)
    if scaled_h < model_h:
        # Pad the bottom only: the origin stays put, so the rectified intrinsics still describe
        # the image. Replicate rather than a constant, matching the padding in `infer_disparity`.
        pad = model_h - scaled_h
        left = cv2.copyMakeBorder(left, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        right = cv2.copyMakeBorder(right, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        return FittedPair(left, right, width_scale, scaled_h)

    # Taller than the engine even after scaling by width: the engine's aspect ratio is wrong for
    # this rig and no amount of padding helps. Crop rather than stretch -- a crop loses the
    # bottom of the frame, which is visible and local, where a stretch corrupts the whole image
    # in a way nothing downstream can detect.
    logger.warning(
        "rectified pair scales to %dx%d, taller than the engine's %dx%d input: cropping %d rows. "
        "Build an engine whose aspect ratio matches this dataset (tools/build_tao_engine.py "
        "--shape-from-scene) to use the full frame.",
        model_w,
        scaled_h,
        model_w,
        model_h,
        scaled_h - model_h,
    )
    return FittedPair(left[:model_h], right[:model_h], width_scale, model_h)


def disparity_to_depth_m(
    disparity_px: np.ndarray, focal_px: float, baseline_m: float, min_disparity: float = 1e-3
) -> np.ndarray:
    """Convert rectified disparity to metric depth: Z = f * B / d.

    Non-positive or negligible disparity becomes NaN rather than infinity, so downstream validity
    masks are simply `np.isfinite`.
    """
    depth = np.full(disparity_px.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity_px) & (disparity_px > min_disparity)
    depth[valid] = (focal_px * baseline_m / disparity_px[valid]).astype(np.float32)
    return depth
