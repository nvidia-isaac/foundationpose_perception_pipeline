# FoundationPose Perception Pipeline

depth → SAM3 → FoundationPose evaluation on BOP-format datasets.

Per scene the pipeline predicts depth from the stereo pair, runs text-prompted SAM3 instance
segmentation at a size derived from that depth map, registers each proposal with FoundationPose,
optionally refines and reranks the proposal set, and scores everything against occlusion-aware
ground truth. Depth is first because SAM3's scene state is built at the depth map's resolution —
see [ARCHITECTURE.md](ARCHITECTURE.md) for the call sequence.

This file covers **installing, configuring and running** it. For how it is put together — module
layout, the inference/evaluation split, the pose metric, the optional refinement and rerank
stages, and the output artifacts — see
**[ARCHITECTURE.md](ARCHITECTURE.md)**. Those two files are the whole of the documentation; every
other explanation lives in a comment beside the code it describes.

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
- **GPU memory: 24 GB is a sensible floor, 32 GB is what this was tested on** — but treat that as
  a starting point and measure your own, because peak memory is a property of your data, not of
  the pipeline. On the captures it was profiled against it peaked at **16.8–17.4 GiB** (sampled every 200 ms
  through complete runs on an RTX 5000 Ada). Three things move that number:
  - **Instances in flight per scene** — the largest term, and the one your dataset controls.
    FoundationPose sets the peak and carries state per proposal, so a crowded scene costs
    materially more than a sparse one.
  - **Depth engine input shape** — TensorRT scratch scales with it, so a wider rectified image
    costs more (`depth.foundation_stereo_max_width`, default `800`).
  - **`--fp-n-hypotheses` / `--fp-prepare-batch` / `--fp-n-refine`** (`64` / `64` / `3`) —
    lowering these trades pose accuracy for memory, and is the first knob to reach for if a run
    will not fit.

  Size for FoundationPose, not for depth: the depth engine is released between scenes, so its
  scratch never coincides with the pose stage. To measure your own, run your densest scene under
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 200`.

  Note a **dynamic-profile** depth engine reserves ~3 GB more scratch than a static one and has
  been seen to fail allocating its execution context on a dense scene set even on a 32 GB card —
  §2.4 builds a static engine sized to your rig, which is the supported path.
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
  sam3/
  foundation-pose-inference-library/
  models/              the Hugging Face depth export (§2.4) — a directory of files, not a checkout
  <your datasets>/     wherever you like; `dataset.root` in the profile points at it (§5)
```

---

## 2. Install

### 2.1 uv and the pipeline environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

cd pipeline
uv venv --python=3.12
uv sync --extra foundationpose   # exactly what uv.lock pins: torch cu128 + the FP runtime libs

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

### 2.2 SAM3

```bash
cd ..
git clone https://github.com/facebookresearch/sam3
cd sam3
uv pip install --python ../pipeline/.venv/bin/python -e .
cd ../pipeline
```

The SAM3 checkpoint is **gated**. Request access at
<https://huggingface.co/facebook/sam3>, then authenticate:

```bash
hf auth login
```

The checkpoint (`sam3.pt`, 3.45 GB) downloads into `~/.cache/huggingface/hub/` on first use.
`sam3` is deliberately not a declared dependency — it is a sibling checkout, and a bare `sam3` on
PyPI is a different package.

**Which revision has been tested on.** SAM3 has no version this project can
pin through `uv.lock`, so it is the one input that can change while the lockfile stays fixed — and
it moves results: two checkouts a fortnight apart produced detection counts one proposal apart on
identical images. This pipeline has been tested on:

| | |
|---|---|
| `facebookresearch/sam3` commit | `96914d2425f90a64f45ca977c2b5165418099543` |
| `facebook/sam3` checkpoint revision | `3c879f39826c281e95690f02c7821c4de09afae7` |
| `sam3.pt` sha256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

Nothing checks this and nothing is enforced — a newer revision is a reasonable thing to run, and
`tools/verify_sam3.py` prints the commit it found so you can record yours.

> **After this point, always use `uv sync --inexact`.** A plain `uv sync` makes the environment
> match `uv.lock` *exactly*, and because sam3 is installed out-of-band it counts as extraneous —
> the sync uninstalls it along with ~17 of its dependencies, and the next run fails with
> `ModuleNotFoundError: No module named 'sam3'`. `--inexact` keeps packages that are not in the
> lock, so sam3 survives. Deleting `.venv` still requires re-running the editable install above,
> since that removes sam3 from disk.

### 2.3 FoundationPose

The FoundationPose Inference Library is open source (Apache-2.0). Its download script fetches
the ONNX weights from [nvidia/foundationpose on Hugging Face](https://huggingface.co/nvidia/foundationpose):

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

`uv sync --extra foundationpose` in §2.1 already installed those (`tensorrt-cu13`,
`nvidia-cuda-runtime`). If you synced without the extra, add it now:

```bash
cd ../pipeline
uv sync --inexact --extra foundationpose
```

Two details, both of which bite otherwise:

- **`--extra foundationpose`, not a bare `uv pip install`.** These libraries are declared in
  `pyproject.toml` so they are lock-managed; installing them ad-hoc means the next `uv sync`
  prunes them and the pose stage breaks.
- **`--inexact`.** Without it this very command uninstalls the sam3 you just installed in §2.2.


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

Depth comes from a TAO `deployable_*` export, run as a TensorRT engine through
[TAO Deploy](https://github.com/NVIDIA-TAO/tao-deploy) in this venv and this process — no
FoundationStereo source checkout and no second environment. The model carries the
[Hugging Face model page](https://huggingface.co/nvidia/c-foundationstereo-s)'s terms;
see [ARCHITECTURE.md → The environments](ARCHITECTURE.md#the-environments) for how it fits
together.

Two packages to install, and an engine to build once per machine.

**Prerequisite: a CUDA toolkit**, not just the pip CUDA runtime wheels. `pycuda` ships as an
sdist only, so pip compiles it; its extension binds the CUDA driver API and needs `cuda.h` and
the link libraries. Its `setup.py` finds them by locating `nvcc` on `PATH`, so:

```bash
nvcc --version    # if this fails, install a CUDA toolkit or set CUDA_ROOT=/usr/local/cuda-XX.Y
```

```bash
# --no-deps is required, not cautious: nvidia-tao-deploy pins scipy==1.17.1, which requires
# numpy>=2 and would break sam3. That conflict is also why these are NOT a `uv` extra -- see the
# comment above `[project.optional-dependencies]` in pyproject.toml.
uv pip install --python .venv/bin/python --no-deps nvidia-tao-deploy==7.1.0
uv pip install --python .venv/bin/python pycuda
```

`--no-deps` skips all 38 of its pins, two of which it genuinely imports at module scope
(`omegaconf`, `matplotlib`). Those are project dependencies, so §2.1's `uv sync` already installed
them — nothing extra to run. If the first depth call fails with `ModuleNotFoundError: No module
named 'omegaconf'` from inside `nvidia_tao_deploy`, run `uv sync --inexact --extra foundationpose`
once.

Fetch a deployable export from the
[Hugging Face model page](https://huggingface.co/nvidia/c-foundationstereo-s).
Choose the export and precision together; fixed and dynamic exports have different constraints.

| Export | Build with | Trade-off |
|---|---|---|
| **dynamic** (`deployable_foundation_stereo_s_dynamic.onnx`) | `--shape-from-scene <scene_dir> --precision fp32` | The repository's baseline. Its input dims are free, so the engine can be built for the size *your* rig rectifies to. |
| **fixed-shape** (`deployable_foundationstereo_small_320x736_v2.0.onnx`, `deployable_foundationstereo_small_576x960_v2.0.onnx`) | `--shape 320x736` or `--shape 576x960`, respectively | The size is baked into the export. Build at that size; the pipeline resamples rectified pairs and may crop their height. |

Where you put the file is up to you: nothing resolves it by convention, it is only the `--onnx`
argument below. These commands assume a `models/` directory beside this repo.

Then build the engine **once per machine**:

```bash
python tools/build_tao_engine.py \
    --onnx ../models/deployable_foundation_stereo_s_dynamic.onnx \
    --shape-from-scene ../<your-dataset>/<split>/000000 \
    --precision fp32
```

Exactly one of `--shape HxW`, `--shape-from-scene <scene_dir>` or `--min`/`--opt`/`--max` is
required — the tool will not guess a size. For a fixed export, use its matching `--shape`;
scene-derived shapes cannot override its dimensions. `<split>` is the profile's `dataset.split`: the
flag takes a path rather than resolving one, so `--config` supplies the width, not the scene.

`--shape-from-scene` runs the real rectification to find the input size this dataset produces, so
the engine is built for what it will actually be fed. Expect a few minutes. The engine is written
beside the ONNX unless `--out-dir` says otherwise, and its path is printed at the end.

A TensorRT engine is **not portable** — it is specific to the GPU architecture, the TensorRT
version, the precision and the input shape. The filename encodes all of these and a sidecar
records the source ONNX's hash, so a stale one is refused rather than used silently. Do not commit
engines; rebuild after any TensorRT change, including one driven by the FoundationPose Inference Library, which
pins the same `tensorrt-cu13` version.

**Then tell the pipeline where it is.** No engine path is committed — one is wrong for every
machine but the one that built it — so this is a step you have to do, and a run without it stops
at launch saying so. Two ways, and the first is the one to prefer:

**Update the config profile.** Uncomment the `engine:` line in `config/<dataset>.yaml`'s
`overrides:` block and point it at what the build printed. In `config/tless.yaml` it is already
there, commented, with the surrounding comment explaining the shape:

```yaml
overrides:
  depth:
    engine: ../../models/<the .engine path printed above>
```

The path resolves against the **config file's** directory, not your shell's, so `../../models/`
is the sibling `models/` directory in §1's layout. Per-dataset rather than in `defaults.yaml`,
because the shape is a property of the rig. Once set, every run uses the commercial model without
a flag.

---

## 3. Verify the install

Each check is independent, and only the last one needs a dataset.

```bash
python tools/verify_sam3.py
```
→ ends with `SAM3 OK: model loaded from HF, forward pass ran, output tensors well-formed.`
The printed proposal confidence is low; that is expected, the test image is a synthetic doodle.
If it fails with `SAM3 checkpoint access failed`, request access and re-run `hf auth login`.

```bash
python tools/verify_foundationpose.py
```
→ ends with `FoundationPose OK: library loaded, build info printed above, CUDA device
synchronized.`

| Failure | Cause |
|---|---|
| `... cannot open shared object file: No such file or directory` | A transitive `.so` is **missing from the search path**. Run the `ldd \| grep "not found"` check and extend `LD_LIBRARY_PATH`. |
| `libc.so.6: version GLIBC_2.38 not found` / `libstdc++.so.6: version GLIBCXX_3.4.31 not found` | **A different failure entirely, despite surfacing through the same `ldd` check.** The library is found; the host's C/C++ runtime is too old. `LD_LIBRARY_PATH` cannot fix this — no path on the machine contains the needed glibc, and no wheel ships one. The host OS is below the floor in §1; use Ubuntu 24.04 or newer. Distinguish the two by the word **`version`** in the message. |
| `CUDA driver version is insufficient` | Host driver older than 580. Check `nvidia-smi`, upgrade, reboot. |
| `FoundationPose checkout not found` | `FOUNDATIONPOSE_ROOT` unset. |
| `FoundationPose root does not exist` / missing library or weights | `run_dev.sh build` or `download_weights.sh` did not finish. |

```bash
python tools/verify_foundationstereo.py --config <name> --engine <path>.engine
```
→ ends with `FoundationStereo OK: engine loaded, forward pass ran, recovered the synthetic
disparity to within N px.` Needs the engine but **no dataset**: it runs one synthetic stereo pair
with a known disparity through the engine and checks the answer. A working install lands within
~0.01 px. This is what tells you the TensorRT engine, TAO Deploy and the pycuda context are all
working, as opposed to merely installed.

`--engine` can be omitted once `depth.engine` is set in the config profile ([§5](#5-configuration));
until then it is required, and leaving it off reports `No engine to verify` rather than guessing.

One more for the depth stage, and the only check that needs a dataset:

```bash
python test/check_engine_depth_smoke.py --config <name> --engine <path>
```

- `check_engine_depth_smoke.py --engine` is the only check that loads the engine, in this process,
  the way the pipeline does. Expect `backend=tao`, `normalization=imagenet`, and a plausible
  valid fraction.
- **`--dataset` is optional and defaults to the profile's `regression.clean_dataset`** — which is
  why the command above names no dataset. Pass it when the one you are about to run is not that
  one: an engine is built for a rectified size, and the whole point of this check is to confirm
  the engine matches the rig you are about to spend an hour on. A profile that sets no
  `clean_dataset` says so rather than guessing.
- **Re-run it after any change to the depth stage.** It needs one scene and the engine, and it is
  the only check that exercises the depth path end to end, in this process, the way a run does.
  The three `tools/verify_*.py` commands stay valid as install checks and cost seconds; re-run
  those after any environment change.

**None of the checks on this page is an accuracy gate.** They establish that a stage runs, not
that it is right. Accuracy is established by running a dataset and comparing `pose_summary.json`
against a saved baseline, and there is no substitute: FoundationPose is not bit-reproducible run
to run, so only `raw` detection counts can be compared exactly.

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
| `defaults.yaml` | repo | Algorithm behaviour: thresholds, rerank, refinement, depth. Read on every run. |
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

### Selecting a profile

Every entry point that reads a profile takes `--config`, which accepts a bare name or a path.
That now includes `tools/build_tao_engine.py`, which takes its `--max-width` default from
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

**3. Build a depth engine and point the profile at it.** No engine ships and none is committed
(§2.4) — `config/tless.yaml` carries the `engine:` line commented out. Build one for this dataset
and uncomment it, or the run below stops at launch:

```bash
python tools/build_tao_engine.py --onnx ../models/<deployable>.onnx \
    --shape-from-scene ../bop_adapted/tless/test/000000
$EDITOR config/tless.yaml     # uncomment `engine:` under overrides: depth:, paste the path
```

Take the shape from `--shape-from-scene` rather than choosing one. T-LESS rectifies to 720x540,
whose width is not a multiple of 32, and the runtime resizes to the engine's width rather than
padding to it — so the height follows from the padded width. Rounding both dimensions
independently gives a shape 32 rows short and every frame is silently cropped.

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
| Engine not set in the profile | `... --foundation-stereo-model <path>.engine --depth-backend commercial` |
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
| FoundationPose Inference Library | Apache-2.0, public on [GitHub](https://github.com/nvidia-isaac/foundation-pose-inference-library) | separate Hugging Face artifact — the [model page](https://huggingface.co/nvidia/foundationpose)'s terms, **not** Apache-2.0 |
| SAM3 | `LicenseRef-Meta-SAM` (Meta's custom SAM License, not OSI-approved) | same license, and the checkpoint is **gated** — request access at <https://huggingface.co/facebook/sam3> |
| FoundationStereo (TAO `deployable_*`) | executed as a TensorRT engine; no source is imported | separate Hugging Face artifact — the [model page](https://huggingface.co/nvidia/c-foundationstereo-s)'s terms |

Three things to know before shipping anything built on this:

- **The depth model is a Hugging Face artifact under the model page's terms.** Nothing here imports
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
