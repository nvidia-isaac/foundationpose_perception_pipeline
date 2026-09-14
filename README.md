# FoundationPose Perception Pipeline

depth → SAM3 → FoundationPose evaluation on BOP-format datasets.

Per scene the pipeline predicts depth from the stereo pair, runs text-prompted SAM3 instance
segmentation at a size derived from that depth map, registers each proposal with FoundationPose,
optionally refines and reranks the proposal set, and scores everything against occlusion-aware
ground truth. Depth is first because SAM3's scene state is built at the depth map's resolution —
see [ARCHITECTURE.md](ARCHITECTURE.md) for the call sequence.

Commands below assume the repository root and an activated environment. Replace angle-bracket
placeholders with your own paths/names before running them.

This file covers **installing, configuring and running** it. For how it is put together — module
layout, the inference/evaluation split, the pose metric, the optional refinement and rerank
stages, and the output artifacts — see
**[ARCHITECTURE.md](ARCHITECTURE.md)**. The local agent guides under `.agent/skills/` provide setup and run procedures.

**Contents:** [Requirements](#1-requirements) · [Install](#2-install) · [Verify](#3-verify-the-install) ·
[Quickstart](#4-quickstart) · [Configuration](#5-configuration) ·
[Dataset adaptation](#6-dataset-adaptation) · [Running](#7-running) ·
[Licenses](#8-licenses) · [Contributing](#9-Contributing)

Contributions are welcome. All commits must be signed off under the Developer Certificate of Origin
 - see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 1. Requirements

- **glibc ≥ 2.38 and GLIBCXX ≥ 3.4.31** — in practice **Ubuntu 24.04 or newer**. This is the
  floor that is easiest to miss and most expensive to discover: the FoundationPose Inference Library ships a
  prebuilt `libfoundation_pose_nvidia.so` linked against those versions, and **no wheel, venv or
  `LD_LIBRARY_PATH` can supply them** — the fix is a different OS. Ubuntu 22.04 (glibc 2.35,
  GLIBCXX 3.4.30) meets every other requirement on this page and still cannot load the pose
  stage. Check before provisioning:
  ```bash
  ldd --version | head -1                                     # want >= 2.38
  strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -c GLIBCXX_3.4.31   # want 1
  ```
- **GPU memory depends on engine shapes and scene complexity.** Stereo is released before
  pose; SAM3 retains its scene features on the GPU. The box-prompt decoder is loaded on first
  refinement use and shares the vision/text features, but has its own TensorRT weights and
  execution context. Measure peak memory on the target GPU.
- **Disk: budget by scene count, not by dataset count.** Depth is cached as three float32 arrays
  per scene at the base camera's resolution (`depth_m.npy`, `depth_rectified_m.npy`,
  `disparity_px.npy`), and they persist by design — `--overwrite-depth` exists precisely so a
  re-run can reuse them, so the cost accumulates across runs rather than replacing itself.
  Measured: **4.7 MB per scene at 720×540** (a 56-scene run is 250 MB) and **101 MB per scene at
  3860×2178** (a 34-scene dataset is 3.4 GB). Scale by your own resolution — it is
  `width × height × 4 × 3` — and note that a multi-dataset sweep is the case that gets expensive:
  ~1000 scenes at the larger size is ~100 GB of depth arrays alone, before predictions, overlays
  or engine caches.
- **CUDA ≥ 12.8** for SAM3/torch.
- **NVIDIA driver ≥ 580** for FoundationPose — its TensorRT build links against the CUDA 13
  runtime. This supersedes the 12.8 floor. Check `nvidia-smi` (top-right `Driver Version`).
- Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

Expected checkout layout — the default config paths assume it:

```text
<parent>/
  pipeline/            this repo
    .venv/                   Python 3.12
  sam3/                export-only source checkout
  foundation-pose-inference-library/
  models/              configurable ONNX/vocabulary directory, not a checkout
    engine_cache/      stereo and SAM3 TensorRT plans
  <your datasets>/     wherever you like; `dataset.root` in the profile points at it (§5)
```

---

## 2. Install

### 2.1 uv and the pipeline environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

cd pipeline
uv venv --python=3.12
uv sync

source .venv/bin/activate   # every `python ...` below assumes this
```

**Activate the venv** — the rest of this file writes `python script/...`, which resolves to the
venv's interpreter only once activated. On Ubuntu there is no bare `python` on `PATH` at all
(only `python3`), so skipping this gives `Command 'python' not found`. If you would rather not
activate, prefix every command with `./.venv/bin/python` instead.

`uv sync` is the reproducible path: `uv.lock` is committed and pins every transitive
dependency. Use it **once, here, while the venv is empty** — afterwards always add `--inexact`,
for the reason explained in §2.2. If you would rather not use uv:

```bash
uv pip install -e .
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

**Keep the upper bounds in `pyproject.toml`.** `sam3` pins `numpy<2,>=1.26`, but current
opencv-python (5.x) and scipy (≥1.16) require `numpy>=2` and will happily upgrade numpy out from
under it. Each `pip install` resolves on its own, so this does not surface as a conflict — it
just leaves a broken venv. Confirm:

```bash
.venv/bin/python -c "import numpy, cv2, scipy; print(numpy.__version__, cv2.__version__, scipy.__version__)"
```

Expect numpy `1.26.x`. If it prints 2.x, repair with
`uv pip install "numpy<2" "opencv-python<5" "scipy<1.16" "setuptools<81"`.

`setuptools<81` is in that list for a different reason and is easy to miss: `sam3` imports
`pkg_resources`, which setuptools 81 removed. A pre-existing venv keeps working because the
package is already installed, so this only bites a **fresh** one — which is why it reaches new
machines and not the author's. The symptom is `ModuleNotFoundError: No module named
'pkg_resources'` from `sam3/model_builder.py`.

### 2.2 SAM3 source and checkpoint (export only)

```bash
cd ..
git clone https://github.com/facebookresearch/sam3
cd sam3
git checkout 96914d2425f90a64f45ca977c2b5165418099543
uv pip install --python ../pipeline/.venv/bin/python -e .
cd ../pipeline
```

The SAM3 checkpoint is **gated**. Request access at
<https://huggingface.co/facebook/sam3>, then authenticate:

```bash
hf auth login
```

Download the checkpoint (`sam3.pt`, 3.45 GB) from the gated repository before running the exporter.
`sam3` is deliberately not a declared dependency — it is a sibling checkout, and a bare `sam3` on
PyPI is a different package.

**Export source revision.** The wrappers target the following source/checkpoint combination:

| | |
|---|---|
| `facebookresearch/sam3` commit | `96914d2425f90a64f45ca977c2b5165418099543` |
| `facebook/sam3` checkpoint revision | `3c879f39826c281e95690f02c7821c4de09afae7` |
| `sam3.pt` sha256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

The exporter calls upstream internal APIs, so changing revisions requires reviewing the wrappers
and validating outputs. `tools/verify_sam3.py` checks the exported TensorRT runtime; it does not
inspect the SAM3 source revision. Deployment with exported artifacts does not import the SAM3
checkout or require checkpoint authentication. Torch remains a declared project dependency.

> **After this point, always use `uv sync --inexact`.** A plain `uv sync` makes the environment
> match `uv.lock` *exactly*, and because sam3 is installed out-of-band it counts as extraneous —
> the sync uninstalls it along with ~17 of its dependencies, and the next run fails with
> `ModuleNotFoundError: No module named 'sam3'`. `--inexact` keeps packages that are not in the
> lock, so sam3 survives. Deleting `.venv` still requires re-running the editable install above,
> since that removes sam3 from disk.

### 2.3 FoundationPose

The FoundationPose Inference Library is open source (Apache-2.0):

```bash
cd ..
git clone https://github.com/nvidia-isaac/foundation-pose-inference-library.git
cd foundation-pose-inference-library

cp .env.example .env    # set FP_DATA_DIR, FP_WEIGHTS_DIR for your machine
sed -i "s/^FP_UID.*/FP_UID=$(id -u)/" .env
sed -i "s/^FP_GID.*/FP_GID=$(id -g)/" .env

./run_dev.sh build
./run_dev.sh run --rm build
scripts/download_weights.sh        # ONNX weights into $FP_WEIGHTS_DIR
```

The pipeline calls FoundationPose **in-process** through its Python bindings
(`foundation_pose_nvidia`, ctypes-loading `build/libfoundation_pose_nvidia.so`) rather than
through `run_dev.sh`/Docker, so the pipeline venv needs its own copies of what that `.so` links
against:

`uv sync` in §2.1 already installed those (`tensorrt-cu13`, `nvidia-cuda-runtime`) — they are core
dependencies, so nothing extra is needed here. If the venv has drifted, re-sync:

```bash
cd ../pipeline
uv sync --inexact
```

**`--inexact` matters.** Without it this very command uninstalls the sam3 you installed in §2.2.
Install these through `uv sync` rather than a bare `uv pip install`: they are declared in
`pyproject.toml` and lock-managed, so an ad-hoc install is pruned by the next sync and the pose
stage breaks.


Check nothing is still missing:

```bash
ldd ../foundation-pose-inference-library/build/libfoundation_pose_nvidia.so | grep "not found"
```

If that prints anything, find which wheel under `.venv/lib/python3.12/site-packages/` owns the
missing `.so` and add its directory to `LD_LIBRARY_PATH`.

Two environment variables every entry point needs:

```bash
export FOUNDATIONPOSE_ROOT=$(realpath ../foundation-pose-inference-library)
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/tensorrt_libs:.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
```

`FOUNDATIONPOSE_ROOT` is read at import time, before argparse runs, so `--foundationpose-root`
alone is not enough — set both consistently if you use the flag. There is no built-in fallback
path: an unset variable produces a clear error rather than silently pointing somewhere wrong.

### 2.4 FoundationStereo

Use the TAO `deployable_foundation_stereo_s_dynamic_v2.0` ONNX from the
[NGC model page](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo).
The pipeline runs the compiled engine directly through TensorRT and `cuda.bindings.runtime`.
Model terms remain those of the NGC artifact.

Place the ONNX and any external weights in the configured model directory (see
[model configuration](#model-directory-and-engine-cache)). With `depth.engine: null`, the default
filename is `deployable_foundation_stereo_s_dynamic_v2.0.onnx`.

Nothing else is required here: the runtime compiles a static profile from the actual padded
stereo input shape on the first cache miss, so a normal run in §3 builds it. That first
compilation can take a long time, and another input shape can require another plan, so
prebuilding is worth it during development:

```bash
python tools/build_stereo_engine.py --config <profile> \
  --shape-from-scene <dataset-root>/<dataset>/<split>/000000
```

The command prints a fingerprinted `.plan` under `models_dir/engine_cache/`. It shares the
runtime's builder and cache key: identical source, profile and build options reuse the same plan.
Prebuilding from a scene derives a shape suitable for fixed-width fitting; automatic mode pads
the actual input. Those shapes can differ, so pass the printed plan explicitly when you want
that exact prebuilt profile:

```bash
python script/run_pipeline.py --config <profile> --dataset <dataset> \
  --foundation-stereo-model /srv/perception/models/engine_cache/<printed-plan>.plan
```

Alternatively set `overrides.depth.engine` to that path. A nonstandard ONNX path is accepted
through the same setting/flag. `build_stereo_engine.py --onnx <source>.onnx --models-dir <directory>`
overrides the source and model root; generated plans still go in that root's `engine_cache/`.

For a known input size, use `--shape HxW`; both dimensions must be positive multiples of 32.
Dynamic profiles use `--min HxW --opt HxW --max HxW`. Choose exactly one shape mode.
FP32 with TF32 permitted is the default. `--precision fp16` or `bf16`, `--no-tf32`,
`--workspace-mb` and `--force` are optional build controls. Different build options produce
different cache keys. Fixed plans resize by width and pad height; a pair that is still too tall
after that is cropped, with a warning naming the rows lost. Rebuild a suitable profile when the
rig or fitting requirements change.

### 2.5 Export and build SAM3

The checkout/checkpoint in §2.2 are needed for export, not TensorRT inference. Export writes
ONNX through `torch.onnx`, so the default `uv sync` already has everything the exporter needs.
Export into the configured model directory:

```bash
python tools/export_sam3_to_onnx.py --config <profile> --checkpoint /path/to/sam3.pt
```

`--output-dir /path/to/models` overrides the export destination. If using that flag, configure
the runtime to use the same directory. Export requires a CUDA device — upstream SAM3 builds two
of its caches with a hardcoded `device="cuda"`, so a CPU model produces `Expected all tensors to
be on the same device` on the decoder graphs. Select a GPU with `CUDA_VISIBLE_DEVICES`.

The exporter calls upstream backbone and grounding forwards, uses actual feature shapes and
sequence-first text features, and copies the matching BPE vocabulary. It produces four static,
batch-one graphs: vision encoder, text encoder, text mask decoder and single-box decoder.
Vision input is 1008×1008; text input is 32 tokens. Separate grounding graphs preserve the
empty/non-empty geometric-prompt branches. Box prompts use normalized `cxcywh` and labels.

No separate compilation command is required. SAM3 builds or reuses vision, text and mask-decoder
plans when the processor is created. The box decoder builds/loads lazily on first refinement.

A compiled plan is **not portable** — it is specific to the GPU architecture, the TensorRT
version, the precision and the input shape. `engine_cache/` keys on exactly those plus the source
model's hash, so a stale plan is never silently reused. Do not commit plans; rebuild after any
TensorRT change, including one driven by the FoundationPose Inference Library, which pins the same
`tensorrt-cu13` version.

For precompiled deployment, use the exact conventional names below with your deployment
TensorRT installation's `trtexec`:

```bash
MODEL_DIR=/srv/perception/models   # same directory as models_dir in your YAML
mkdir -p "$MODEL_DIR/engine_cache"
trtexec --onnx="$MODEL_DIR/sam3_vision_encoder.onnx" --saveEngine="$MODEL_DIR/engine_cache/sam3_vision_encoder.plan" --skipInference
trtexec --onnx="$MODEL_DIR/sam3_text_encoder.onnx" --saveEngine="$MODEL_DIR/engine_cache/sam3_text_encoder.plan" --skipInference
trtexec --onnx="$MODEL_DIR/sam3_mask_decoder.onnx" --saveEngine="$MODEL_DIR/engine_cache/sam3_mask_decoder.plan" --skipInference
trtexec --onnx="$MODEL_DIR/sam3_box_decoder.onnx" --saveEngine="$MODEL_DIR/engine_cache/sam3_box_decoder.plan" --skipInference
```

These named plans take precedence over ONNX and are not automatically invalidated. Replace or
remove them when re-exporting. `--sam3-models-dir` selects another SAM3 directory.

**Before the first box-decoder build, add `libnvinfer_vc_plugin.so.10`.** No TensorRT wheel ships
it, so that one build — and only that one — fails with `(parseFromFile): INTERNAL_ERROR: Unable
to open library: libnvinfer_vc_plugin.so.10`. Fetch it once:

```bash
D=$(mktemp -d)
curl -sSL -o "$D/vc.deb" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer-vc-plugin10_10.16.1.11-1+cuda13.2_amd64.deb
dpkg-deb -x "$D/vc.deb" "$D/x"
cp "$D/x/usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.16.1" \
   .venv/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_vc_plugin.so.10
```

Match the version to the `tensorrt-cu13` pin in `pyproject.toml` whenever that pin moves. A
`.venv` rebuild loses the file; `uv sync` does not. `apt install libnvinfer-vc-plugin10` is
equivalent if you have root and the CUDA repo configured.

SAM3 follows `--device` (`cuda` or `cuda:N`); stereo currently uses device 0.
Use `--device cuda:0 --fp-device-id 0` for the single-GPU pipeline.

---

## 3. Verify the install

In order, cheapest first — each is independent, and only the last one needs a dataset. A check
that runs an engine may take a few minutes on first use if it must compile it.

```bash
python tools/verify_sam3.py --sam3-models-dir /srv/perception/models
```
→ loads the exported TensorRT engines and reports `SAM3 OK (TensorRT)` when outputs are well formed.
The printed proposal confidence is low; that is expected, the test image is a synthetic doodle.
Checkpoint access is required during export, not during TensorRT inference.

```bash
python tools/verify_foundationpose.py
```
→ ends with `FoundationPose OK: library loaded, build info printed above, CUDA device
synchronized.`

| Failure | Cause |
|---|---|
| `... cannot open shared object file: No such file or directory` | A transitive `.so` is **missing from the search path**. Run the `ldd \| grep "not found"` check and extend `LD_LIBRARY_PATH`. |
| `libc.so.6: version GLIBC_2.38 not found` / `libstdc++.so.6: version GLIBCXX_3.4.31 not found` | **A different failure entirely, despite surfacing through the same `ldd` check.** The library is found; the host's C/C++ runtime is too old. `LD_LIBRARY_PATH` cannot fix this — no path on the machine contains the needed glibc, and no wheel ships one. The host OS is below the floor in §1; use Ubuntu 24.04 or newer. Distinguish the two by the word **`version`** in the message. |
| `(parseFromFile): INTERNAL_ERROR: Unable to open library: libnvinfer_vc_plugin.so.10` | Not the row above, despite the wording — no path will help. No TensorRT wheel ships this library and only the box-decoder build needs it. Fetch it once, per [§2.5](#25-export-and-build-sam3). |
| `CUDA driver version is insufficient` | Host driver older than 580. Check `nvidia-smi`, upgrade, reboot. |
| `FoundationPose checkout not found` | `FOUNDATIONPOSE_ROOT` unset. |
| `FoundationPose root does not exist` / missing library or weights | `run_dev.sh build` or `download_weights.sh` did not finish. |

```bash
python tools/verify_foundationstereo.py --config <name> --engine <path>.engine
```
→ ends with `FoundationStereo OK: engine loaded, forward pass ran, recovered the synthetic
disparity to within N px.` Needs the engine but **no dataset**: it runs one synthetic stereo pair
with a known disparity through the engine and checks the answer. The script requires error ≤ 2 px
and a valid fraction ≥ 0.9. This is what tells you the TensorRT engine and CUDA runtime bindings are all
working, as opposed to merely installed.

`--engine` can be omitted once `depth.engine` is set in the config profile ([§5](#5-configuration));
until then it is required by this verifier, and leaving it off reports `No engine to verify`.
The pipeline itself supports conventional model lookup with an unset `depth.engine`.

One more for the depth stage, and the only check that needs a dataset:

```bash
python test/check_engine_depth_smoke.py --config <name> --engine <path>
```

- `check_engine_depth_smoke.py --engine` is the only check that loads the engine, in this process,
  the way the pipeline does. Expect `backend=tensorrt`, `normalization=imagenet`, and a plausible
  valid fraction.
- **`--dataset` is optional and defaults to the profile's `regression.clean_dataset`** — which is
  why the command above names no dataset. Pass it when the one you are about to run is not that
  one: an engine is built for a rectified size, and the whole point of this check is to confirm
  the engine matches the rig you are about to spend an hour on. A profile that sets no
  `clean_dataset` says so rather than guessing.
- **Re-run it after any change to the depth stage.** It needs one scene and the engine, and it is
  the only check that exercises the depth path end to end, in this process, the way a run does.
  The three `tools/verify_*.py` commands are install checks; account for possible compilation
  when estimating their first-run time.

**None of the checks on this page is an accuracy gate.** They establish that a stage runs, not
that it is right. Accuracy is established by running a dataset and comparing `pose_summary.json`
against a saved baseline. Compare raw proposals as well as poses; do not assume bitwise
agreement across TensorRT builds, precisions or changed exports.

```bash
python test/check_engine_depth_smoke.py --config <name> --dataset <name> --engine <path>
```

It finds the dataset through `dataset.root` in the config profile, like every other entry point.
If yours is somewhere else, either set that key ([§5](#5-configuration)) or pass the path per run:

```bash
python test/check_engine_depth_smoke.py --config <name> --engine <path> --dataset-root /path/to/datasets
```

---

## 4. Quickstart

With the install verified and `config/<your-dataset>.yaml` in place (see
[§5](#5-configuration)):

```bash
# 1. precompute the GT z-buffer cache for that dataset (once; saves ~12-14 s per target)
python script/build_gt_cache.py --dataset <name>

# 2. two scenes, end to end
python script/run_pipeline.py --dataset <name> --max-scenes 2 --overwrite-results
```

These pass `--dataset` and no `--config`, which works because the profile is
`config/<name>.yaml` — the same `<name>`. If your profile is named something else, add
`--config <profile>`; see [`--config`, `--dataset`, or both?](#--config---dataset-or-both).

Step 2 writes `output/<name>/report.md` plus the CSV/JSON summaries described in
[ARCHITECTURE.md → Outputs](ARCHITECTURE.md#outputs).

---

## 5. Configuration

Every YAML the pipeline reads lives in `config/`. No script carries a dataset path or a tuned
threshold as a Python default, so pointing the pipeline at your own data should never mean
editing a `.py` file.

| File | One per | Holds |
|---|---|---|
| `defaults.yaml` | repo | Model directory and algorithm behaviour: thresholds, rerank, refinement, depth. |
| `<dataset>.yaml` | dataset | Where the data is, what the objects are called, which dataset the checks default to (`regression.clean_dataset`), and any `overrides:`. |
| `example_bop.yaml` | — | Annotated template. Not a profile; copy it to make one. |

### Precedence

```
config/defaults.yaml   ->   config/<dataset>.yaml `overrides:`   ->   CLI flag
```

Last one wins. A profile's `overrides:` block mirrors the section structure of `defaults.yaml`
and is merged key by key, so it lists only what differs.

**Every key in `overrides:` must exist in `defaults.yaml`.** An unknown section or key is a
launch-time error, not a no-op — it used to be merged somewhere nothing reads and silently
ignored, which made a misindented line indistinguishable from a setting that had no effect. The
error names the offending key, and when the same key exists under a different section it says so,
because the usual cause is a commented-out parent leaving the line attached to the section above.

### Model directory and engine cache

Set `models_dir` in `config/defaults.yaml`, or override it in a dataset profile:

```yaml
overrides:
  models_dir: /srv/perception/models
  depth:
    engine: null
```

The shipped default is `../../models`, relative to `config/defaults.yaml`. An overridden
relative path resolves against the profile that defines it, even when that profile is outside
this repository. `MODELS_DIR` overrides the YAML value. `--sam3-models-dir` overrides the
directory for SAM3; the stereo prebuild tool accepts `--models-dir`. There is no general
`--models-dir` flag on `run_pipeline.py`: use YAML or `MODELS_DIR` for the shared root.

```text
<models_dir>/
  deployable_foundation_stereo_s_dynamic_v2.0.onnx
  sam3_vision_encoder.onnx
  sam3_text_encoder.onnx
  sam3_mask_decoder.onnx
  sam3_box_decoder.onnx
  bpe_simple_vocab_16e6.txt.gz
  ... ONNX external weight files, at the relative locations recorded in each graph
  engine_cache/
    <model-stem>.plan                 optional user-supplied plan
    <model-stem>__<fingerprint>.plan   automatically compiled plan
    <model-stem>__<fingerprint>.plan.json
    <model-stem>__<fingerprint>.lock
```

Resolution is deterministic; the runtime does not scan for the newest plan:

1. An explicit stereo `--foundation-stereo-model` / `depth.engine` path is used directly.
   It may name ONNX or a precompiled `.plan`, `.engine`, or `.trt` file.
2. With no stereo path, or for each SAM3 component, look for
   `engine_cache/<model-stem>.plan` under its model directory.
3. If no named plan exists, use `<model-stem>.onnx`. Compute its cache key and reuse the
   corresponding fingerprinted plan, or compile it on the deployment GPU.

The key includes source ONNX and referenced external initializer weight contents, GPU name,
TensorRT version, input profiles, precision, TF32 and workspace settings. Builds use a file lock
and publish plans by atomic rename. Cache lookup still reads and fingerprints the ONNX files;
keep them and their external weights available when using automatic mode.

A named or explicitly selected plan bypasses ONNX fingerprinting. You are responsible for its
compatibility and freshness; a bad supplied plan fails to load rather than silently rebuilding.
Generated plans always go under the configured `models_dir/engine_cache/`, even when an explicit
stereo ONNX lives elsewhere. SAM3's directory override also moves its cache beneath that directory.
The model directory must be writable for automatic compilation.

`depth.engine: null` selects the conventional stereo name above; it does not disable depth.
Changing models or preprocessing does not invalidate already written scene depth automatically:
use `--overwrite-depth`, and `--overwrite-results` for new segmentation/pose results.
FoundationPose keeps its existing SDK cache separately, under the run's
`foundationpose_engine_cache/` unless `--fp-engine-cache-dir` overrides it.

### Selecting a profile

Every entry point that reads a profile takes `--config`, which accepts a bare name or a path.
That now includes `tools/build_stereo_engine.py`, which takes its `--max-width` default from
`depth.foundation_stereo_max_width` so the engine is built for the width the pipeline will feed
it. The rule runs the other way too, so the flag is never missing where it would do
something: an entry point that loads no profile takes no `--config`. `tools/verify_sam3.py`
and `tools/verify_foundationpose.py` are the two you will meet — both check an install and
touch no dataset.

```bash
python script/run_pipeline.py --config tless --dataset tless     # -> config/tless.yaml
python script/run_pipeline.py --config /abs/path/to/mine.yaml    # explicit path
```

Resolution order, most specific first: `--config`, then `$PERCEPTION_PIPELINE_CONFIG`, then a
profile named after `--dataset` if one exists, then the sole profile in `config/` if there is
exactly one. If none resolve, the pipeline errors and lists the profiles it can see — a fresh
clone with one profile needs no flag; a clone with three refuses to guess.

### `--config`, `--dataset`, or both?

> **If the dataset directory and the profile file have the same name, `--dataset` alone is
> enough. If the names differ, pass both.**

| your layout | what to pass |
|---|---|
| `<root>/tless/` and `config/tless.yaml` — **names match** | `--dataset tless` |
| `<root>/run_07/` and `config/my_rig.yaml` — **names differ** | `--config my_rig --dataset run_07` |

`--dataset X` looks for `config/X.yaml` and uses it when that file exists. It is a filename
match and nothing more: no part of the profile's contents is consulted, and a dataset whose name
matches no profile leaves the resolver with nothing to go on.

Differing names are common rather than exotic — one profile usually serves many dataset
directories, since `dataset.glob` is what selects them (`run_01`, `run_02`, … under one
`my_rig.yaml`). Every one of those needs `--config`.

Two cases need `--config` whatever the names are, because no dataset is named at all:

- commands that take no `--dataset`, such as `script/run_batch_eval.py`
- `script/build_gt_cache.py --all`

Setting `PERCEPTION_PIPELINE_CONFIG=<name>` in your shell replaces `--config` for every command
in that shell.

### Adding a dataset

```bash
cp config/example_bop.yaml config/my_dataset.yaml
$EDITOR config/my_dataset.yaml
python script/run_pipeline.py --config my_dataset --dataset <subfolder>
```

`example_bop.yaml` documents every key, including the BOP directory layout expected under
`dataset.root`.

**Where the data lives is `dataset.root`** — the one key nothing can guess for you. There are two
ways to set it, and no third:

- **In the profile**, for a machine whose data does not move. This is the normal case:
  `root: ../../my_datasets` under `dataset:` in `config/my_dataset.yaml`.
- **Per run, with `--dataset-root`**, which the entry points that read a dataset tree accept —
  `run_pipeline.py`, `infer.py`, `evaluate.py`, `run_batch_eval.py`, `build_gt_cache.py`,
  `test/check_engine_depth_smoke.py` and `tools/sweep_rerank_cutoff.py`. The adapter is the
  exception and takes neither: it WRITES a dataset tree rather than reading one, so it has
  `--src` for the source and `--out-root` / `--depth-root` for what it produces. Use it for a
  one-off location, a second copy of the data, or a checkout whose profile points elsewhere:

  ```bash
  python script/run_pipeline.py --config my_dataset --dataset <subfolder> \
      --dataset-root /path/to/datasets
  ```

If neither is right, the failure is a `Missing scene directory` naming both the path it wanted and
the root that produced it — the profile's value, unless a flag overrode it.

Two things that surprise people:

- **Relative paths resolve against `config/`**, not your shell's working directory. That is what
  lets `../../bop_adapted` mean the same thing from the repo root and from `pipeline/`.
- **The comments in `defaults.yaml` are load-bearing.** Several record the measurement that
  justifies a value — why the visibility bands are `[0.0, 0.9]` rather than one higher threshold,
  why `min_visible_fraction` is `0.1`, what the CLAHE and working-distance settings are worth in
  millimetres. Do not change a number without reading the comment above it, and do not drop the
  comment when changing the number.

---

## 6. Dataset adaptation

The pipeline reads one layout. BOP datasets do not ship in it, and no two of them are wrong in
the same way, so `tools/bop_adapt/` converts them into it:

```text
<dataset.root>/<name>/test/<scene>/rgb/<im_id>.png   im_id 0 is the BASE camera
                                  /scene_camera.json  im_ids are the CAMERAS of a rig
                                  /scene_gt.json
                                  /scene_gt_info.json
<dataset.root>/<name>/models/              binary little-endian PLY + models_info.json
<dataset.root>/<name>/models_eval/         BOP's decimated copy, for the pose metrics
<dataset.root>/<name>/dataset_map.json     {object_name: obj_id}
<dataset.root>/<name>/scene_index.json     provenance back to the source dataset
<dataset.collected_depth_root>/<name>/test/<scene>/scene_cam0_depth.png   uint16 millimetres
```

Four properties of that tree are load-bearing, each with a consumer that fails quietly without
it: **im_id 0 is the base camera** (pose is estimated in `rgb/000000.png` and ground truth scored
for im_id 0); **im_ids are contiguous from 0** (the partner selector indexes the camera array);
**meshes are binary PLY** (neither the pipeline's PLY reader nor FoundationPose parses ASCII);
and **collected depth is uint16 millimetres with 0 meaning invalid**.

A capture already in this layout needs no adapter — point `dataset.root` at it and run.

### Adapting T-LESS

T-LESS is the worked example, and its profile ships as `config/tless.yaml`.

**1. Download.** Three files from the BOP mirror on Hugging Face, `bop-benchmark/tless` — the
models and the split with public annotations:

```bash
DEST=../bop_datasets/tless && mkdir -p "$DEST" && cd "$DEST"
for f in tless_base.zip tless_models.zip tless_test_primesense_bop19.zip; do
  curl -fLO "https://huggingface.co/datasets/bop-benchmark/tless/resolve/main/$f"
  unzip -q -o "$f"
done
mv tless/* . 2>/dev/null; rmdir tless 2>/dev/null   # base/ unpacks one level deep
```

About 0.9 GB. `--src` must end up holding `models_cad/`, `models_eval/` and `test_primesense/`.
T-LESS has no `val`: it is a BOP-2019-era dataset whose test ground truth is public, so
`test_primesense_bop19` is the split that can be scored.

**2. Adapt.**

```bash
python tools/bop_adapt/adapt.py --config tless --src ../bop_datasets/tless
```

The dataset module is chosen by the profile's `dataset.name`, and the output paths come from the
same profile — so the adapter and the pipeline that reads its output cannot disagree about where
the data lives. With the shipped defaults this reports 30 meshes converted and **56 adapted
scenes** from 20 source scenes.

More scenes out than in, because one adapted scene is one *(source scene, base frame)* pair:
the pipeline scores one frame per scene, so 20 source scenes would otherwise give 20 scored
frames. `--frame-stride` is a stride, not a count: the adapter takes every Nth base frame, so a
LARGER stride yields FEWER adapted scenes.

**3. Make the stereo ONNX available in `models_dir`.** Leave `depth.engine: null`
to compile on first use, or prebuild for the adapted scene:

```bash
python tools/build_stereo_engine.py --config tless \
  --shape-from-scene ../bop_adapted/tless/test/000000
```

To use that exact prebuilt profile, set `overrides.depth.engine` to the printed plan.
Use the scene-derived shape instead of independently rounding raw image dimensions: fixed-plan
fitting resizes by width and then pads height. An insufficient engine height is rejected.

**4. Cache ground truth and run**, exactly as for any other dataset:

```bash
python script/build_gt_cache.py --dataset tless
python script/run_pipeline.py --dataset tless --max-scenes 2 --overwrite-results
```

### What the adapter decides, and why it is not a default you can ignore

T-LESS is a single-camera dataset, and this is a stereo pipeline. What makes the conversion
possible is that its scenes are static and it ships `cam_R_w2c`/`cam_t_w2c`, so any two frames of
one scene form a valid stereo pair — the adapter turns *frames* into *cameras*.

Which frames it pairs is the main tunable, and it is a cliff rather than a slope. `--baseline-min`
and `--baseline-max` bound the distance between camera centres that may be offered as a partner.
A longer baseline is more precise per pixel of disparity error and harder to match; past roughly
45 degrees of rectification rotation the overlap shrinks, matching degrades, and depth becomes
the pipeline's dominant error source. The shipped band is measured — see the header comment in
`config/tless.yaml` for the grouping that set it.

Distance alone is not a sufficient test. Roughly a quarter of T-LESS pairs fail the depth stage's
parallax bounds, so the adapter runs the pipeline's *own* rectification check on every candidate
and exposes only those that survive. A base frame left with no usable partner is skipped, which
is why the run reports how many were dropped — with a narrow band that count is normally large
and is not an error.

### Adding another dataset

One module per dataset under `tools/bop_adapt/datasets/`, exposing `add_arguments(parser)` and
`adapt(...)`, registered in `adapt.py` and selected by the profile's `dataset.name`. Shared work
belongs in `emit.py` (the layout above), `partners.py` (stereo selection for a static scene) and
`ply.py` (mesh conversion) — an adapter should not reimplement any of it.

What goes in the module is what no flag can express: ground truth annotated in only one sensor's
frame and needing transformation into the base camera's, images in a format or bit depth the
pipeline does not read, or depth delivered at a different resolution than the base camera and
needing reprojection.

The object-name table is `config/<name>/object_names.json`, beside the profile rather than beside
the adapter, because it is dataset metadata and the profile's `prompts:` block is keyed by the
names in it. BOP ships numeric `obj_id`s and no names, so this table is written by hand.

---

## 7. Running

```bash
source .venv/bin/activate
export FOUNDATIONPOSE_ROOT=$(realpath ../foundation-pose-inference-library)
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/tensorrt_libs:.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
```

No `PYTHONPATH` is needed — `foundationpose_perception_pipeline` is an installed package. Do **not** add any
other environment's CUDA libraries to `LD_LIBRARY_PATH`: the pipeline's own torch resolves its
cuDNN from the wheels above, and a foreign copy on the path is what makes it die with
`Could not load symbol cudnnGetLibConfig`.

| Goal | Command |
|---|---|
| Precompute GT cache | `python script/build_gt_cache.py --dataset <name>` (or `--config <name> --all`) |
| Single dataset, end to end | `python script/run_pipeline.py --dataset <name> --overwrite-results` |
| Override the configured stereo model | `... --foundation-stereo-model <path>.engine --depth-backend commercial` |
| Without reranking | `... --proposal-selection-policy all` |
| Without CAD refinement | `... --sam3-refinement-policy none` |
| No collected depth on this machine | `... --no-depth-metrics` |
| All datasets + aggregate | `python script/run_batch_eval.py --config <name>` |
| Inference only, no ground truth | `python script/infer.py --dataset <name> --output-dir <run>` |
| Re-score a finished run | `python script/evaluate.py --dataset <name> --run <run>` |

`run_pipeline.py` is an orchestrator over the last two: it runs `infer.py`'s inference pass and
then `evaluate.py`'s scoring pass, in one process, against one output directory. The two are
worth knowing about separately — `infer.py` is the only half that can run on a capture with **no
`scene_gt.json` at all**, and `evaluate.py` re-scores a finished run at a different IoU
threshold, visibility band or rerank cutoff in seconds, without loading a model.
[ARCHITECTURE.md → Stage flow](ARCHITECTURE.md#stage-flow) has the split in full.

Run them separately by pointing the second at what the first wrote:

```bash
# 1. inference — loads SAM3, the depth engine and FoundationPose; writes predictions.jsonl,
#    the mask sidecars and depth_m.npy under --output-dir
python script/infer.py --dataset <name> --output-dir output/<run> --max-scenes 2

# 2. scoring — no model, no GPU beyond GT rasterization; --run is step 1's --output-dir
python script/evaluate.py --dataset <name> --run output/<run>
```

Both need `--dataset`: `evaluate.py` reads `scene_gt.json` from the dataset, not from the run.
It writes the summaries and `report.md` back into `--run` unless `--output-dir` says otherwise, so
re-scoring the same run twice overwrites the report — give the second one its own `--output-dir`
to keep both. Add `--no-depth-metrics` if this machine has no collected depth tree to compare
against.

`--depth-backend` selects nothing; the model path decides. It *asserts*, so a run that believes
it is under the commercial licence and is not fails at launch rather than in the metrics.

The first run of a scene generates depth — a few seconds per scene, in this process. Precompute
the GT cache before a multi-dataset run, or every target re-rasterizes it at ~12–14 s.

**Retuning, as distinct from re-scoring.** `tools/sweep_rerank_cutoff.py` sweeps the rerank
cutoff offline from a finished run, so a cutoff can be fitted to your data without re-running
inference:

```bash
python tools/sweep_rerank_cutoff.py --config <name> --results-root output --datasets <name>
python tools/sweep_rerank_cutoff.py --config <name> --results-root output/<batch run>
```

`--results-root` is the directory holding one subdirectory per dataset, which is what
`run_batch_eval.py` writes — so a single-dataset run under `output/<name>/` is reached by
pointing at `output` and naming it with `--datasets`. `--datasets` is plural and does not stand
in for `--dataset`, so this needs `--config` either way.

---

## 8. Licenses

This project is released under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
Third-party attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). To report a
security vulnerability, follow [SECURITY.md](SECURITY.md) rather than opening an issue.

Nothing below is redistributed with this code — each is obtained separately under its own terms.
**Code and weights are licensed separately** for every one of them; an Apache-2.0 repository does
not make its published checkpoints Apache-2.0.

| Component | Code | Weights / checkpoint |
|---|---|---|
| FoundationPose Inference Library | Apache-2.0, public on [GitHub](https://github.com/nvidia-isaac/foundation-pose-inference-library) | separate NGC artifact — `nvidia/tao/foundationpose:deployable_v1.0` terms, **not** Apache-2.0 |
| SAM3 | `LicenseRef-Meta-SAM` (Meta's custom SAM License, not OSI-approved) | same license, and the checkpoint is **gated** — request access at <https://huggingface.co/facebook/sam3> |
| FoundationStereo (TAO `deployable_*`) | executed as a TensorRT engine; no source is imported | separate NGC artifact — the [model page](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo)'s terms |

Three things to know before shipping anything built on this:

- **The depth model is an NGC artifact under the model page's terms.** Nothing here imports
  FoundationStereo's source: an engine is executed by TensorRT alone, so the obligation is the
  model's, not the code's. Read the model page before shipping anything built on it.
- **The SAM3 checkpoint is gated and non-redistributable** — every user must request access
  themselves before the pipeline will run at all.
- **An automated license scan of the pipeline venv will misreport SAM3.** Its package metadata
  advertises an `MIT License` classifier, while its `License` field and bundled `LICENSE` file
  both say `SAM License`. The `LICENSE` file governs.

Use of this pipeline may rely on third party components or models that you must download
separately. The components or models are subject to the applicable open source licenses or other
license terms, including any proprietary notices, disclaimers, requirements, and extended use
rights.

---

## 9. Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull
request. Every commit must carry a `Signed-off-by` line certifying the Developer Certificate of
Origin — `git commit -s` adds it. Pull requests with unsigned commits will not be merged.

To report a security vulnerability, follow [SECURITY.md](SECURITY.md) rather than opening an
issue.
