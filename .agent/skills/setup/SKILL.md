---
name: setup
description: Installs and verifies the perception pipeline environment, native TensorRT stereo/SAM3 runtime, SAM3 ONNX export tools, and the FoundationPose Inference Library. Use for setup, provisioning, missing dependencies or CUDA/TensorRT library failures.
---

# Pipeline setup

Read [README.md](../../../README.md) for requirements, commands and model configuration;
[ARCHITECTURE.md](../../../ARCHITECTURE.md) explains ownership and stage boundaries.
Respect the user's execution constraints. The commands below are a procedure, not evidence that
the migration has been built or tested.

## 0. Choose runtime or export setup

Runtime needs the pipeline environment, compatible NVIDIA GPU/runtime, FoundationPose Inference Library,
stereo ONNX or plan, and SAM3 exported ONNX/plans plus vocabulary. SAM3 source and its gated
checkpoint are needed only to export.

The documented SDK environment targets Python 3.12, Ubuntu 24.04 with glibc ≥ 2.38 /
GLIBCXX ≥ 3.4.31, and driver ≥ 580 for its CUDA 13 TensorRT runtime. Check the installed
SDK's requirements if it differs. The SDK Docker build also needs working GPU-enabled Docker,
git, wget and unzip.

Default layout:

```text
<parent>/
  pipeline/
  sam3/                       export only
  foundation-pose-inference-library/
  models/                     configurable
    *.onnx
    bpe_simple_vocab_16e6.txt.gz
    engine_cache/*.plan
```

The examples assume the repository directory is named `pipeline`; adjust paths if it is not.

## 1. Pipeline environment

From the repository root:

```bash
uv venv --python=3.12
uv sync
source .venv/bin/activate
```

TensorRT, CUDA runtime and `cuda-bindings` are core dependencies, so `uv sync` alone gives a
working install; there is no GPU extra. `regex` and `ftfy` are project
dependencies, both for upstream-faithful SAM3 prompt tokenizing. ONNX is **not** a dependency:
graphs are written by `torch.onnx.export` and read by TensorRT, so nothing imports `onnx`. Torch
remains installed although stereo/SAM3 inference uses TRT.
Keep the NumPy/OpenCV/SciPy/setuptools bounds in `pyproject.toml`.

The migration lockfile still needs regeneration on the supported Linux environment.
Do not claim a verified frozen install from the current lockfile. Do not run dependency
resolution if the user has prohibited execution. If SAM3 is installed editable for export,
subsequent syncs need `--inexact` to retain that out-of-band package.

## 2. Configure model storage

Set the common root in `config/defaults.yaml`, or in a profile:

```yaml
overrides:
  models_dir: /srv/perception/models
  depth:
    engine: null
```

Relative paths resolve against the YAML file defining them. `MODELS_DIR` overrides YAML.
Keep each ONNX's external weights at the relative locations recorded inside the graph.
Generated stereo and SAM3 plans live inside `<models_dir>/engine_cache/`.

A conventional `engine_cache/<model-stem>.plan` wins over `<model-stem>.onnx`.
Otherwise the runtime hashes the ONNX/weights and build specification, reuses its fingerprinted
plan, or compiles on a cache miss. Named plans require manual replacement after source or runtime
changes. An explicit stereo `depth.engine` / `--foundation-stereo-model` may name ONNX or a
`.plan`, `.engine` or `.trt` file. Its path is authoritative; no directory scan is performed.

## 3. SAM3 export (skip when exported artifacts are supplied)

```bash
cd ..
git clone https://github.com/facebookresearch/sam3
cd sam3
git checkout 96914d2425f90a64f45ca977c2b5165418099543
uv pip install --python ../pipeline/.venv/bin/python -e .
cd ../pipeline
hf auth login
```

Obtain `sam3.pt` through the gated upstream repository under its terms. Authentication is
needed for obtaining the checkpoint, not for TensorRT inference. Use existing user-authorized
access; never expose credentials. The exporter targets the revision above, so review and
validate wrappers before changing it.

```bash
python tools/export_sam3_to_onnx.py --config <profile> --checkpoint /path/to/sam3.pt
```

Export requires a CUDA device; select one with `CUDA_VISIBLE_DEVICES`.
`--output-dir` overrides the destination; configure runtime to use the same directory.
The exporter writes `sam3_vision_encoder.onnx`, `sam3_text_encoder.onnx`,
`sam3_mask_decoder.onnx`, `sam3_box_decoder.onnx` and the BPE vocabulary.
The four graphs use batch one, vision 1008×1008 and text length 32. The single-box decoder
is separate to retain the upstream geometric-prompt branch.

Automatic builds are the default; optional `trtexec` commands and exact conventional plan
names are in [README §2.5](../../../README.md#25-export-and-build-sam3). The old root-level
`sam3_vision_fp32.plan` / `sam3_text_fp32.plan` names are not discovered.

Building the box decoder needs one library no TensorRT wheel ships. Do this once, before the
first build, or that build alone fails with `INTERNAL_ERROR: Unable to open library:
libnvinfer_vc_plugin.so.10`:

```bash
D=$(mktemp -d)
curl -sSL -o "$D/vc.deb" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer-vc-plugin10_10.16.1.11-1+cuda13.2_amd64.deb
dpkg-deb -x "$D/vc.deb" "$D/x"
cp "$D/x/usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.16.1" \
   .venv/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_vc_plugin.so.10
```

Not a broken install; `LD_LIBRARY_PATH` and `uv sync` will not fix it. Keep the version matched
to the `tensorrt-cu13` pin in `pyproject.toml`.

## 4. FoundationPose Inference Library

```bash
cd ..
git clone https://github.com/nvidia-isaac/foundation-pose-inference-library.git
cd foundation-pose-inference-library
cp .env.example .env
sed -i "s/^FP_UID.*/FP_UID=$(id -u)/" .env
sed -i "s/^FP_GID.*/FP_GID=$(id -g)/" .env
mkdir -p data weights engine_cache
./run_dev.sh build
./run_dev.sh run --rm build
scripts/download_weights.sh
cd ../pipeline
```

The pipeline loads the SDK in-process. It defaults to the library under
`FOUNDATIONPOSE_ROOT/build/` and refiner/scorer ONNX under `FOUNDATIONPOSE_ROOT/weights/`.
The SDK's Docker `.env` does not configure those pipeline paths. If relocating artifacts, use
`--fp-library`, `--fp-refine-model-path` and `--fp-score-model-path` as appropriate.
This migration leaves the FoundationPose weight layout and SDK cache behavior unchanged.

```bash
export FOUNDATIONPOSE_ROOT=$(realpath ../foundation-pose-inference-library)
PIPELINE_SITE=$(realpath .venv/lib/python3.12/site-packages)
export LD_LIBRARY_PATH="${PIPELINE_SITE}/tensorrt_libs:${PIPELINE_SITE}/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
ldd "$FOUNDATIONPOSE_ROOT/build/libfoundation_pose_nvidia.so" | grep "not found"
```

Use absolute library directories and do not mix another environment's CUDA libraries in.
A missing library and a library requiring a newer glibc are different failures; see README.

## 5. FoundationStereo

Obtain the TAO `deployable_foundation_stereo_s_dynamic_v2.0` ONNX from the NGC model page,
and place it at:

```text
<models_dir>/deployable_foundation_stereo_s_dynamic_v2.0.onnx
```

Retain external weights if present. No FoundationStereo source checkout is needed.
With `depth.engine: null`, the runtime finds that name and builds a static TRT profile from the
actual padded input on first use. Each new shape/configuration may incur a long compilation.

Optional prebuilding uses an adapted scene and the profile's width setting:

```bash
python tools/build_stereo_engine.py --config <profile> --shape-from-scene <dataset-root>/<dataset>/<split>/000000
```

The tool prints a fingerprinted plan under `models_dir/engine_cache/`. Use that printed path
in `overrides.depth.engine` to select its exact fixed fitting profile. Prebuilt and automatic
profiles share a cache entry only when their shapes and build options match.

For an external source or another model directory:

```bash
python tools/build_stereo_engine.py --config <profile> --onnx /path/to/stereo.onnx --models-dir /srv/perception/models --shape-from-scene <scene-dir>
```

`--models-dir` specifies where plans are stored: plans always go beneath the model root.
FP32/TF32 is the default. `--shape HxW`, dynamic `--min/--opt/--max`, precision,
workspace and rebuild controls are documented in README. Fixed plans resize and pad; excess
height fails instead of silently cropping.

## 6. Optional verification and handoff

Run these only within the user's authorized execution scope. First-use compilation may be slow.

```bash
python tools/verify_sam3.py --sam3-models-dir /srv/perception/models
python tools/verify_foundationpose.py
python tools/verify_foundationstereo.py --config <profile> --engine /path/to/stereo.plan
python test/check_engine_depth_smoke.py --config <profile> --dataset <dataset> --engine /path/to/stereo.plan
```

The stereo verifier requires an explicit path or `depth.engine` even though the pipeline itself
supports automatic conventional lookup. The real-scene smoke check should report
`backend=tensorrt` and `normalization=imagenet`. Neither establishes full accuracy parity.
Do not claim success for any command that has not run.

Report the selected model directory, source/plan paths, build mode, SDK location, any unresolved
dependencies, and which checks actually ran. Separate cold-build timings from cached runs.
On a failed named plan, check GPU/TRT compatibility and the selected artifact.
