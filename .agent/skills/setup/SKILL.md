---
name: setup
description: Installs and verifies the perception pipeline environment — the pipeline's own venv, SAM3, the FoundationPose Inference Library, and FoundationStereo run through TAO Deploy as a TensorRT engine. Use when asked to set up, install, provision, bootstrap, or repair this repo's environment, or when a run fails with a missing-dependency, venv, CUDA, TensorRT, pycuda, or LD_LIBRARY_PATH error.
---

# Pipeline setup

Full narrative and rationale: [README.md](../../../README.md) — Requirements, Install and Verify.
This skill is the condensed, ordered checklist to execute; go read the referenced section when a
command's *why* matters — e.g. before deviating from it or explaining a failure to the user.

Four things get installed, and depth is the one with a shape to it: the NGC `deployable_*`
FoundationStereo export is built into a TensorRT engine and run through TAO Deploy **in this venv
and in this process**. No second environment, no FoundationStereo *source* checkout — only a
directory holding the model files — and depth is a function call rather than a subprocess.

## 0. Preflight

Sizing the machine is part of this step and is not covered here: README.md section 1 carries the
GPU-memory floor, what moves it, and the disk cost per scene. Read it before provisioning — two of
the three are properties of your data rather than of the pipeline, so they cannot be answered from
this document.

- **`ldd --version` → glibc must be ≥ 2.38, and `libstdc++` must carry `GLIBCXX_3.4.31`.** In
  practice that means **Ubuntu 24.04 or newer**. Check this FIRST — it is the only prerequisite
  here that cannot be installed:
  ```bash
  ldd --version | head -1
  strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -c GLIBCXX_3.4.31   # want 1
  ```
  The FoundationPose Inference Library ships a prebuilt `libfoundation_pose_nvidia.so` linked against those versions. On
  Ubuntu 22.04 (glibc 2.35 / GLIBCXX 3.4.30) every step below succeeds — including the FoundationPose Inference Library build
  in step 3 — and the pose stage then cannot load what it just built, failing with
  `libc.so.6: version GLIBC_2.38 not found`. **No wheel, venv or `LD_LIBRARY_PATH` can fix it**;
  extending the search path is the natural next move and it cannot work, because no path on the
  machine contains a newer glibc. If the host is below the floor, stop and tell the user the OS
  must change — do not proceed and do not try to work around it.
- `nvidia-smi` → Driver Version must be **≥ 580** (top-right). This is the FoundationPose floor
  and is stricter than the CUDA-12.8 floor SAM3/torch need.
- `nvcc --version` → a **CUDA toolkit**, not just the pip CUDA runtime wheels. `pycuda` (step 4)
  ships as an sdist only, so pip compiles it, and its extension needs `cuda.h` and the link
  libraries. Its `setup.py` finds them by locating `nvcc` on `PATH`; `CUDA_ROOT=/usr/local/cuda-XX.Y`
  is the escape hatch. It is the one prerequisite that is not a Python package, so check it
  *before* the long steps rather than discovering it during them.
- `uv --version`, `docker --version`, `git --version` — all required; stop and tell the user if
  any is missing rather than trying to install them yourself.
- **Docker must be able to reach the GPU, and your user must be able to reach Docker.** Step 3
  builds the FoundationPose Inference Library through `./run_dev.sh`, whose compose override gives the build service
  `gpus: all`, so a working `docker --version` is not enough on its own — the NVIDIA Container
  Toolkit has to be installed and the daemon restarted, and your account has to be in the `docker`
  group. Both fail late and confusingly if skipped: the first as a container that cannot see the
  GPU, the second as `permission denied` on the Docker socket.
  ```bash
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi   # must print the GPU table
  ```
  If it does not, install the toolkit per
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>,
  then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. If
  the failure is `permission denied` instead, `sudo usermod -aG docker $USER` and then **`newgrp
  docker`** — group membership does not reach a shell that was already open, which is the part
  that usually costs the time.
- **`wget` and `unzip`** — hard prerequisites of `scripts/download_weights.sh` in step 3, which
  exits with `error: 'unzip' is required but not installed` without them. Neither is present on a
  minimal Ubuntu 24.04 image, so check here rather than discovering it mid-step:
  ```bash
  sudo apt-get update && sudo apt-get install -y wget unzip
  ```
- Checkout layout: this repo's directory must sit **next to** the sibling checkouts, not contain
  them: `<parent>/<this repo>/`, `<parent>/sam3/`, `<parent>/foundation-pose-inference-library/`,
  `<parent>/models/` (the depth export -- files, not a checkout), `<parent>/<datasets>/`. The repo
  does not need to be literally named `pipeline` — only the sibling relationship matters, since
  `FOUNDATIONPOSE_ROOT` and the config profile's dataset paths are resolved relative to it.

```bash
REPO_DIR=$(basename "$PWD")   # this repo's directory name, used below
```

## 1. Pipeline venv (Python 3.12)

```bash
uv venv --python=3.12
uv sync --extra foundationpose      # torch cu128 + tensorrt-cu13 + nvidia-cuda-runtime, pinned by uv.lock
```

Run `uv sync` without `--inexact` **only this once, on an empty venv**. From step 2 onward this venv holds
packages that are not in `uv.lock` — sam3, and then `nvidia-tao-deploy`/`pycuda` — so every later
sync must add `--inexact` or it uninstalls them as extraneous.

Verify the numpy pin survived:

```bash
.venv/bin/python -c "import numpy, cv2, scipy; print(numpy.__version__, cv2.__version__, scipy.__version__)"
```

Must print numpy `1.26.x`. If it prints `2.x`: `uv pip install "numpy<2" "opencv-python<5" "scipy<1.16"`.

## 2. SAM3 (sibling checkout, editable install, gated checkpoint)

```bash
cd .. && git clone https://github.com/facebookresearch/sam3 && cd sam3
uv pip install --python "../$REPO_DIR/.venv/bin/python" -e .
cd "../$REPO_DIR"
```

The checkpoint is **gated** at <https://huggingface.co/facebook/sam3> — this step needs the user.
Confirm they have requested and been granted access, then `./.venv/bin/hf auth login` (or confirm
`~/.cache/huggingface/token` already holds a token with access). Call it by path: `hf` arrives as a
transitive dependency of the `sam3` install one command above, it is not a declared dependency of
this repo, and nothing has activated the venv yet — a bare `hf` is `command not found` here. Do not
push past this without
confirming access — a missing grant only surfaces later, deep inside `verify_sam3.py`, as
`SAM3 checkpoint access failed`.

**From here on, every `uv sync` for this repo is `uv sync --inexact --extra foundationpose`** —
never bare `uv sync` again.

## 3. FoundationPose Inference Library (sibling checkout, Docker build, weights)

```bash
cd .. && git clone https://github.com/nvidia-isaac/foundation-pose-inference-library.git
cd foundation-pose-inference-library
cp .env.example .env
sed -i "s/^FP_UID.*/FP_UID=$(id -u)/" .env
sed -i "s/^FP_GID.*/FP_GID=$(id -g)/" .env
# WHICH CONSUMER READS WHICH: `.env` is read by docker compose ONLY. The pipeline never reads it
# -- `pose.py`'s ensure_foundationpose_paths resolves <FOUNDATIONPOSE_ROOT>/weights/ directly. The
# two therefore agree only by coincidence of both pointing at ./weights, and editing FP_WEIGHTS_DIR
# moves where the DOWNLOAD lands without moving where the pipeline LOOKS.
# Only FP_UID/FP_GID need editing. LEAVE FP_WEIGHTS_DIR ALONE: the shipped default ./weights
# resolves to <FoundationPose Inference Library checkout>/weights, which is exactly where the pipeline looks for the ONNX
# weights when no CLI flag is given (pose.py's ensure_foundationpose_paths resolves
# <FOUNDATIONPOSE_ROOT>/weights/refiner_net.onnx and score_net.onnx). Point it anywhere else and
# the build still succeeds, download_weights.sh still succeeds, and the run fails much later with
# "Missing FoundationPose refine model: <path>" -- unless every run passes
# --fp-refine-model-path and --fp-score-model-path. `--fp-library` is the third flag with this
# same FOUNDATIONPOSE_ROOT coupling -- it defaults to
# <FOUNDATIONPOSE_ROOT>/build/libfoundation_pose_nvidia.so -- so if you
# relocate any part of the FoundationPose Inference Library, all three move together.
# Create the bind-mount targets first, or docker creates them as root inside your checkout and
# undoes the FP_UID/FP_GID lines above:
mkdir -p data weights engine_cache
./run_dev.sh build
./run_dev.sh run --rm build
scripts/download_weights.sh          # ONNX weights into ./weights -- NOT into $FP_WEIGHTS_DIR:
                                     # the script never sources .env, it falls back to ./weights
cd "../$REPO_DIR"
```

`download_weights.sh` needs **no NGC credential** — the FoundationPose weights are public, which
the script says in its own comments and error text (`NGC_API_KEY` is honoured if set, but is
strictly optional; its only hard prerequisites are `wget` and `unzip`). Measured: the weights
download on a host with no NGC configuration at all. So if it fails, treat it as a
**network/proxy problem and retry** — do not send the user hunting for credentials they do not
need. (Contrast SAM3 in §2, which is genuinely gated and does need the user.)

The pipeline calls FoundationPose in-process through its Python bindings, not through
`run_dev.sh`, so the venv needs its own copies of what the built `.so` links against:

```bash
uv sync --inexact --extra foundationpose
ldd ../foundation-pose-inference-library/build/libfoundation_pose_nvidia.so | grep "not found"
```

Anything `ldd` prints means a wheel under `.venv/lib/python3.12/site-packages/` owns the missing
`.so` — find it and add its directory to `LD_LIBRARY_PATH` in step 5.

## 4. Commercial depth: TAO Deploy + the engine

### 4.1 Two packages, installed out-of-band

```bash
uv pip install --python .venv/bin/python --no-deps nvidia-tao-deploy==7.1.0
uv pip install --python .venv/bin/python pycuda
```

- **`--no-deps` is required, not cautious.** `nvidia-tao-deploy` declares 38 `==` pins, several of
  which break this venv rather than merely annoy it: `scipy==1.17.1` requires `numpy>=2` while
  sam3 pins `numpy<2`, so a plain install resolves cleanly and then `import sam3` dies. It also
  pulls mpi4py, PyInstaller, pyarmor and `nvidia-eff` (not on public PyPI).
- **There is deliberately no `tao` extra to sync.** It existed until 2026-08-17 and made the
  whole project unresolvable: an extra's pins are part of the project's requirements even when
  nobody asks for that extra, so `scipy==1.17.1` vs the project's `scipy<1.16` broke every
  `uv sync` and `uv lock` on a clean checkout. The two commands above are the supported install;
  the pins are documented in `pyproject.toml` as a comment.
- **`--no-deps` leaves two real holes, and they are already plugged.** TAO Deploy imports
  `omegaconf` and `matplotlib` at module scope on the depth path, so skipping its pins skips
  those too. Both are declared as project dependencies, so `uv sync` installs them and there is
  nothing extra to run here. A venv built before 2026-08-17 predates that and needs one
  `uv sync --inexact --extra foundationpose`; the symptom is `ModuleNotFoundError: No module
  named 'omegaconf'` raised from inside `nvidia_tao_deploy`, at the first depth call rather than
  at install time. **Do not fix it by installing omegaconf by hand** — re-sync, or the next
  `uv sync` without `--inexact` removes it again.
- **The version pin matters.** `cv/depth_net` is not usable in every release — 6.25.10 ships no
  `nvidia_tao_deploy/config/` at all.
- Both packages are now out-of-band, exactly like sam3: `uv sync --inexact` from here on, always.

### 4.2 The model files

Fetch a `deployable_*` export from the
[NGC model page](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo). The NGC
model-page terms govern the weights.

**No credential of any kind is needed — the artifact is public.** The recipe, because the obvious
routes do not work and rediscovering this costs an hour on every new machine:

```bash
# 1. The /zip endpoint that works for the FoundationPose weights 404s for this model version,
#    so the NGC CLI is required. It is not installed by anything else in this setup.
curl -sSL -o ngccli.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.34.10/files/ngccli_linux.zip
python3 -m zipfile -e ngccli.zip .       # `unzip` is not present on a minimal 24.04, and Ubuntu
                                         # ships no bare `python` — both spellings matter here
chmod +x ngc-cli/ngc

# 2. Download into the sibling models/ directory from README.md's layout.
mkdir -p ../models && cd ../models
<path-to>/ngc-cli/ngc registry model download-version \
    nvidia/tao/foundationstereo:deployable_foundation_stereo_s_dynamic_v2.0
```

> **Do NOT set `NGC_CLI_ORG` / `NGC_CLI_TEAM`.** It is the intuitive step and it is what *breaks*
> the download: the CLI treats a configured org as an assertion of identity and refuses it from an
> anonymous caller (`Invalid org - If not Authenticated, org cannot be set`). With both unset the
> transfer completes in seconds. The org and team are already carried by the fully-qualified
> target name, so the variables are redundant as well as fatal.

To verify identity before downloading — also public, also no auth:

```bash
curl -sS https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationstereo/versions/deployable_foundation_stereo_s_dynamic_v2.0/files
# -> totalSizeInBytes 346531163, sha256_digest a001a7bc0512a0bc3b3218194e924784e58b20656c6f1ea2c151024e555cfd64
```

**Do not tell the user to clone FoundationStereo, and do not assume a directory of that name
exists.** Nothing on this path imports FoundationStereo's source, and nothing resolves the model
by convention: the ONNX is only ever the `--onnx` argument, and the built engine is only ever a
config value. Any directory will do. A clean install that follows the layout in `README.md` §1
has a `models/` directory beside the repo and no FoundationStereo checkout, so use that unless
the user already keeps the export somewhere else — in which case pass their path through
unchanged rather than relocating it.

Prefer a **dynamic** export (`*_dynamic_*.onnx`): its input dims are free, so §4.3 can build the
engine at whatever size this rig rectifies to. The page also carries fixed-shape exports
(`*_320x736_*`, `*_576x960_*`, …); those are usable, but the size is baked in, so build with
`--shape 320x736` to match the export and expect the pipeline to resample every pair to reach it.
Do not substitute one for the other silently — ask, or use whichever the user already has.

### 4.3 Build the engine (once per machine, per shape, per precision)

```bash
./.venv/bin/python tools/build_tao_engine.py \
    --onnx ../models/<the deployable export>.onnx \
    --shape-from-scene <dataset_root>/<dataset>/<split>/000000
```

`<split>` is the profile's `dataset.split` — `test` for every profile shipped here.

Expect a few minutes. `--shape-from-scene` runs the real pair selection and rectification to find
the padded input size *this rig* produces at the configured `--max-width`, so the engine is built
for what it will actually be fed. Do not guess a shape — the answer is not the raw image size.

That scene has to exist first, and it must be an ADAPTED one: a BOP tree as downloaded is not in
the layout the pair selection reads. Adapt the dataset before building the engine — the "pipeline"
skill's first section covers it — rather than discovering the ordering from a missing-scene error
here.

Four things about the result, each of which bites otherwise:

- **Not portable.** An engine is specific to the GPU architecture, the TensorRT version, the
  precision and the input shape. The filename encodes all of them and a sidecar `.json` records
  the source ONNX's sha256, so a stale one is refused rather than used silently. Never commit
  engines. Rebuild after any TensorRT change — including one driven by a FoundationPose Inference Library
  upgrade, since both extras pin the same `tensorrt-cu13`.
- **FP32 is the default and is what we want.** Precision is fixed at build time. Treat `--precision
  fp16` as a measured experiment against the fp32 engine, never as a default arrived at by
  copy-paste from TAO's spec template — a 5 mm pose bar has no room for a silent precision change.
- **Static `min=opt=max` is deliberate.** TAO Deploy allocates its buffers at the profile's MAX
  shape, so a generous dynamic profile costs memory on every scene.
- **Shape mismatch does not fail loudly.** Feeding a static engine a differently-sized rectified
  pair rescales it (and crops, with a warning, if it is still too tall). So build the engine for
  the width the runs will use, and rebuild if `depth.foundation_stereo_max_width` changes.
  `--shape-from-scene` already accounts for this: the runtime RESIZES to the engine's width
  rather than padding to it, so the height it needs follows from the padded width, not from the
  rectified height. A `cropping N rows` warning at run time therefore means the engine was built
  at a shape that did not come from `--shape-from-scene` — treat it as a build error and rebuild,
  not as a note, since a crop silently removes the bottom of every frame.

### 4.4 Point the config at it

So that a bare run uses the commercial model rather than needing the flag every time, set the
engine in the dataset profile's `overrides:` block (`config/<dataset>.yaml`). The shipped profiles
carry that line **commented out**, because an engine path is machine-specific and no committed one
can be right on your host — so this is an uncomment-and-edit, not an addition:

```yaml
overrides:
  depth:
    engine: ../../models/<...>.engine
```

Per-dataset, not in `defaults.yaml`: the shape is a property of the rig. `depth.engine` is unset
by default, and a run with nothing to fall back on fails at launch — so wiring the profile is the
difference between "the engine by configuration" and "the engine only if someone remembers the
flag on every command".

## 5. Environment variables (every run needs these)

```bash
source .venv/bin/activate
export FOUNDATIONPOSE_ROOT=$(realpath ../foundation-pose-inference-library)
SITE=$(realpath .venv/lib/python3.12/site-packages)
export LD_LIBRARY_PATH="${SITE}/tensorrt_libs:${SITE}/nvidia/cu13/lib:${LD_LIBRARY_PATH}"
```

**Both paths are absolutised on purpose.** A relative `LD_LIBRARY_PATH` is resolved against the
working directory at load time, not at export time, so it silently stops pointing anywhere the
moment a command runs from somewhere else — and §4.2 has you `cd ../models`. The failure is
`libcudart.so.13: cannot open shared object file`, raised deep in the pose stage after depth has
already run for every scene, which is an expensive way to learn it.

`FOUNDATIONPOSE_ROOT` is read at import time, before argparse runs — `--foundationpose-root` alone
does not substitute for it; set both consistently if that flag is used. No `PYTHONPATH`:
`foundationpose_perception_pipeline` is an installed package, and a `ModuleNotFoundError` for it means
`uv pip install -e .`, not a path hack. Do **not** put another venv's libraries on this path.

## 6. Verify

In order, cheapest first — each is independent. The first three need a GPU (and the engine, for
the last of them); only the final one needs a dataset:

```bash
python tools/verify_sam3.py                 # -> "SAM3 OK: ..."
python tools/verify_foundationpose.py       # -> "FoundationPose OK: ..."
python tools/verify_foundationstereo.py --config <name> --engine <the .engine path>   # -> "FoundationStereo OK: ..."; no dataset
python test/check_engine_depth_smoke.py --config <name> --engine <the .engine path>   # needs a dataset + the engine
```

Both `--engine` flags may be dropped once §4.4 has put the path in the profile — they default to
`depth.engine`, which is unset until then. Do not read "No engine to verify" as a broken install.

The one specific to this backend and worth understanding:

- `check_engine_depth_smoke.py --engine` is the only check that loads the engine, in this process,
  the way the pipeline does. Expect it to report the engine's fixed shape, `backend=tao`,
  `normalization=imagenet`, and a plausible valid fraction (what counts as plausible depends
  on how much of the frame survives rectification).

The failure → cause table lives in README.md's Verify section — read it rather than re-deriving causes from the
raw error text. Two additions specific to this path:

| Failure | Cause |
|---|---|
| `pycuda` fails to build during install | No CUDA toolkit. §0 — install one or set `CUDA_ROOT`. |
| `invalid resource handle` from a CUDA call | pycuda's context leaking outside `tao_context()` in `inference/stereo/tao.py`. That boundary is where to look, not the caller. |
| Engine refused as stale / sidecar mismatch | TensorRT or GPU changed, or the ONNX did. Rebuild (§4.3); do not `--force` past it without knowing why. |

## Checklist to report back to the user

- [ ] CUDA toolkit present (`nvcc`), driver ≥ 580
- [ ] Pipeline venv synced, numpy pinned to `1.26.x`
- [ ] SAM3 installed editable; checkpoint access confirmed (user-gated step)
- [ ] Host meets the glibc ≥ 2.38 / GLIBCXX ≥ 3.4.31 floor (§0) — check before anything else
- [ ] FoundationPose Inference Library built, weights downloaded (public, no NGC credential), `ldd` clean
- [ ] `nvidia-tao-deploy==7.1.0` (`--no-deps`) + `pycuda` installed
- [ ] Deployable ONNX fetched from NGC (public, no credential); engine built fp32 + static, with
      `--shape-from-scene` when a dataset is present. **Installing before the data arrives is
      normal**, and the placeholder shape is the one number this document cannot give you, because
      it is a property of the rig's stereo geometry: derive it as `--max-width` for the width and
      that rig's rectified height rounded up to a multiple of 32 for the height, which at the
      default 800 px lands at `--shape 480x800` on a 16:9-ish rig. Treat whatever you pick as
      **provisional** and rebuild with `--shape-from-scene` before any accuracy or regression
      figure is taken — a static engine silently rescales a differently-sized pair rather than
      refusing it, so a wrong placeholder never announces itself.
- [ ] `depth.engine` set in the dataset profile's `overrides:`
- [ ] All four verify commands pass, including `verify_foundationstereo.py` and
      `check_engine_depth_smoke.py --engine`
- [ ] That last one logged **no** `cropping N rows` warning. It is the check that answers whether
      the engine matches this rig, because it is the only one that runs YOUR engine against YOUR
      scene the way a run does. A crop is not a note: the engine is the wrong shape and every
      frame loses its bottom rows silently. Rebuild with `--shape-from-scene` rather than
      accepting it — see §4.3.
