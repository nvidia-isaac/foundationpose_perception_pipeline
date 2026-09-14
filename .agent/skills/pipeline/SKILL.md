---
name: pipeline
description: Runs one dataset end to end (depth → SAM3 → FoundationPose) on the COMMERCIAL depth model — the native TensorRT stereo engine — and reads out the pose metrics, including the flags a dataset without collected depth needs and how to prove which model actually ran. Covers adapting a BOP dataset into the pipeline's layout first, which the engine build depends on. Use when asked to run, evaluate, benchmark, or re-run a dataset, to adapt or convert a BOP dataset, to regenerate depth for one, or to compare pose results before and after a change. Environment installation is out of scope — that is the "setup" skill.
---

# Run one dataset end to end, on the commercial engine

Full narrative: [README.md](../../../README.md) — Quickstart, Running, and the FoundationStereo
section for the depth backend; outputs and stage flow in
[ARCHITECTURE.md](../../../ARCHITECTURE.md). This skill is the ordered checklist plus the failure
modes that are not obvious from those. Section *titles* rather than numbers, deliberately: the
numbering is not stable across README edits.

Depth comes from the NGC `deployable_*` export, compiled and executed by native TensorRT
in this process. Stereo is released after each scene and its plan is reloaded as needed.
`run_pipeline.py` generates depth itself; there is no separate depth step to run first.

Commands below are for an authorized run on the deployment machine. If the user asks for
documentation or prohibits execution, describe these steps without running them.

## 0. Environment (every shell, before anything else)

```bash
cd <repo root>
P=$PWD/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$P/tensorrt_libs:$P/nvidia/cu13/lib:$LD_LIBRARY_PATH"
export FOUNDATIONPOSE_ROOT=$(realpath ../foundation-pose-inference-library)
```

Missing runtime libraries can fail during model loading:
`OSError: libcudart.so.13: cannot open shared object file`. Verify cheaply before a long run:

```bash
./.venv/bin/python -c "import ctypes; ctypes.CDLL('libcudart.so.13'); print('ok')"
```

Do **not** add any other venv's libraries here.

## 1. Adapt the dataset (before scene-derived prebuilding or inference)

Skip only if the dataset is already in the pipeline's layout — `test/<scene>/rgb/<im_id>.png`
with one `scene_camera.json` whose im_ids are the cameras of a rig, im_id 0 being the base. A BOP
tree as downloaded is usually not.

```bash
./.venv/bin/python tools/bop_adapt/adapt.py --config <name> --src <dataset as downloaded>
```

The dataset module is selected by the profile's `dataset.name`, and the output paths come from
the same profile, so the adapter cannot write somewhere the run will not look. `--help` shows the
selected dataset's own flags. If no module is registered for the name, it says so and lists the
ones that are.

**Do this before §2, not after.** `build_stereo_engine.py --shape-from-scene` needs an adapted scene
directory to measure the rectified size from, so the engine cannot be built first. The ordering
is a real dependency, not a preference.

Two things worth reading in the output rather than skimming past:

- **Scenes out ≠ scenes in.** Where a dataset's cameras are *frames* of a static scene, one
  adapted scene is one (source scene, base frame) pair — the pipeline scores one frame per scene,
  so this is what stops a 20-scene capture yielding 20 scored frames.
- **Skipped base frames are normal.** The adapter runs the depth stage's own rectification check
  and drops any base frame left with no partner that passes it. That count is reported. With a
  narrow baseline band it is routinely a large fraction, and it is the reason the depth stage
  does not later raise `StereoDepthError` on a scene it cannot handle.

`--baseline-min/--baseline-max` bound which partners may be offered, and on a static-scene dataset
that is the single biggest lever on depth accuracy — a cliff rather than a slope. The shipped
values are measured; the profile header records the grouping that set them. Re-adapting with a
different band is a new dataset for caching purposes: rebuild the GT cache (§3) and rerun with
`--overwrite-depth`.

## 2. Confirm the engine before spending an hour

Set `overrides.models_dir` in the profile (or `MODELS_DIR`). Leave `depth.engine: null`
for conventional stereo ONNX/plan resolution; set it to an explicit ONNX or plan to override.
SAM3 uses that same root, unless `--sam3-models-dir` overrides it.

```text
<models_dir>/*.onnx
<models_dir>/bpe_simple_vocab_16e6.txt.gz
<models_dir>/engine_cache/*.plan
```

Named `<model-stem>.plan` overrides win; otherwise ONNX is fingerprinted and a matching
plan is reused or built. A first run or new stereo shape may compile for a long time.
For predictable startup, prebuild stereo and select the printed plan:

```bash
./.venv/bin/python tools/build_stereo_engine.py --config <name> --shape-from-scene <scene-dir>
./.venv/bin/python test/check_engine_depth_smoke.py --config <name> --dataset <name> --engine /path/to/printed.plan
```

Expect `backend=tensorrt`, `normalization=imagenet` and plausible depth for that scene.
Supplied plans must match the target GPU/TRT and remain your responsibility after re-export.
The runtime rejects an insufficient fixed-plan height instead of cropping. Full lookup rules
and optional SAM3 prebuild commands are in README. These checks do not establish accuracy parity.

## 3. Ground-truth cache (once per dataset)

```bash
./.venv/bin/python script/build_gt_cache.py --dataset <name>
./.venv/bin/python script/build_gt_cache.py --config <name> --all   # every dataset. --all
# names none, so there is nothing to infer the profile from and --config becomes required.
```

Skipping this costs ~12–14 s per target re-rasterizing the same z-buffer.

## 4. Run it

The examples below select an explicit plan. Omit `--foundation-stereo-model` to use the
configured model directory, or pass an ONNX path to compile on demand. A warmup that only
runs SAM3 text prompts does not build the lazy box decoder; first refinement can still compile.


**Dataset with collected sensor depth** (a `<collected_depth_root>/<name>` directory exists):

```bash
./.venv/bin/python script/run_pipeline.py --dataset <name> \
    --foundation-stereo-model <path>.plan --depth-backend commercial \
    --overwrite-depth --overwrite-results
```

**Dataset without it** — add `--no-depth-metrics`:

```bash
./.venv/bin/python script/run_pipeline.py --dataset <name> \
    --foundation-stereo-model <path>.plan --depth-backend commercial \
    --no-depth-metrics --overwrite-depth --overwrite-results
```

Without that flag the run aborts immediately, before doing any work, with
`FileNotFoundError: Missing collected-depth directory`. Check first rather than discovering it at
launch — the root is `dataset.collected_depth_root` in the profile:

```bash
ls -d /path/to/collected_depth/<name>
```

`--no-depth-metrics` only skips *scoring* predicted depth against the collected map. Detection and
pose metrics are unaffected — they come from `scene_gt.json` and the GT cache — and the report's
depth table reads `n/a` rather than disappearing, so a run without the comparison cannot be
mistaken for one that made it.

### `--depth-backend commercial` is the licence assertion

It selects nothing — an explicit ONNX/TRT model path selects the backend, or an unset path
selects the shipped conventional stereo model. The flag rejects a conflicting backend name;
it does not inspect the supplied model's contents or prove its license.

### Which overwrite flag

- `--overwrite-depth` regenerates depth. Needed whenever the engine, `--foundation-stereo-max-width`,
  the CLAHE settings or the working-distance bounds change — cached depth was made with the old ones.
- `--overwrite-results` reruns SAM3 + pose only.
- Omit `--overwrite-depth` to reuse existing depth — the right move when the pose stage crashed and
  depth already completed. Confirm what is cached before relying on it:

```bash
./.venv/bin/python -c "
import json,glob
ps=sorted(glob.glob('output/<name>/depth/*/metadata.json'))
m=json.load(open(ps[0]))
print(len(ps),'scenes  backend=',m['backend'],' norm=',m['normalization'],
      ' model_hw=',m.get('model_fixed_hw'),' model=',m['model'].split('/')[-1])"
```

### Depth settings: the tuned values are the committed defaults — do not pass them

`config/defaults.yaml` carries the tuned configuration: `clahe_clip_limit: 3.0`,
`clahe_detail_boost: 1.5`, `min/max_working_distance_m: 0.55/1.20`,
`foundation_stereo_max_width: 800`. Passing them explicitly on the command line is noise at best and drift at worst; change
them in the profile's `overrides:` block if a dataset needs different ones.

Two that are not free to move:

- **`--foundation-stereo-max-width` is tied to the engine.** A static engine fed a different
  rectified size is fitted by width and padded; insufficient height raises an error. Automatic
  ONNX mode can create another plan for another padded shape. Record the actual plan used.
- **Working-distance bounds are both-or-neither.** They drive the disparity pre-shift together;
  a half-specified volume silently disables it.

## 5. Prove what actually ran

Inspect requested configuration and resolved per-scene metadata separately:

```bash
./.venv/bin/python -c "
import json; c=json.load(open('output/<name>/inference_config.json'))
print(c['foundation_stereo_model']); print('max_width', c['foundation_stereo_max_width'])"

./.venv/bin/python -c "
import json; m=json.load(open('output/<name>/depth/000000/metadata.json'))
print(m['backend'], m['normalization'], m['model_fixed_hw'], m['model'])"
```

`metadata.json` records the resolved plan in `model`, including automatically generated paths.
`inference_config.json` can retain an unset requested model in automatic mode. A stale depth
cache can still contain old backend metadata: use `--overwrite-depth` after changing the model.
The backend label records execution, not proof of artifact licensing or numerical correctness.

## 6. Read the results

`output/<name>/` gets `report.md` plus the CSV/JSON summaries
([ARCHITECTURE.md → Outputs](../../../ARCHITECTURE.md#outputs)). The headline numbers:

```bash
./.venv/bin/python -c "
import json
d=json.load(open('output/<name>/pose_summary.json'))['overall']
for k in ['matched_predictions','max_vertex_error_within_threshold_rate','max_vertex_error_mm_median',
          'max_vertex_error_mm_p90','max_vertex_error_mm_p99','add_or_adds_mm_mean',
          'rotation_error_deg_mean','max_vertex_error_meets_required_rate']:
    print(f'{k:34} {d.get(k)}')"
```

`by_visibility` repeats all of it per visibility band (`0.0` and `0.9` by default), and
`by_object` per object.

**Compare `matched_predictions` first.** If it moved, the accuracy metrics are not comparable
across runs — a run that matches fewer, easier instances scores better while being worse.

**Read rates and means as separate signals.** Rates up with means up means the bulk improved while
a few instances regressed into the tail; that is a real and normal outcome here, not a
contradiction. `rotation_error_deg_mean` is the most sensitive indicator of a few poses having
flipped to a worse local minimum — a single 180° flip moves it far more than it moves the median.

Save a baseline before changing anything, since `--overwrite-results` destroys the previous run:

```bash
cp output/<name>/pose_summary.json /tmp/<name>.baseline.json
```

Compare the two `pose_summary.json` files rather than trusting a single headline number: a
change that moves `<=5mm` by a point can be several objects improving and several regressing.

**Do not re-run the pipeline to change a scoring parameter.** `predictions.jsonl` keeps every
proposal with its scores, so a different IoU threshold, visibility band or rerank cutoff is
arithmetic over that file:

```bash
./.venv/bin/python script/evaluate.py --config <name> --run output/<name> --rerank-cutoff 4.5
./.venv/bin/python tools/sweep_rerank_cutoff.py --config <name> --results-root output --datasets <name>
```

Both load no model and touch no GPU. Only a change that affects *inference* — depth settings, the
engine, the refinement policy, the confidence threshold — needs `run_pipeline.py` again.

## 7. All datasets

```bash
./.venv/bin/python script/run_batch_eval.py --config <name> --output-root output/<run_name> \
    --foundation-stereo-model <path>.plan --depth-backend commercial
```

Writes per-dataset outputs plus `summary.json` / `report.md` / `run_status.jsonl` under
`--output-root`. `--depth-backend commercial` is worth more here than on a single run: a batch
with an unexpected registered backend is harder to diagnose hours later. Add `--continue-on-error`
if one dataset failing should not end the sweep, and `--no-depth-metrics` to drop the collected
tree from both scoring *and* dataset discovery (without it, discovery intersects against that tree
and can select nothing).

## 8. Depth quality

Measure the depth itself rather than inferring it from pose. A normal run already did it, as long
as it was made *without* `--no-depth-metrics` on a dataset that has collected depth:

```bash
./.venv/bin/python -c "
import json; print(json.load(open('output/<name>/depth_summary.json'))['overall'])"
```

Read the **object** rows, not `all` — FoundationPose only ever consumes depth inside the SAM3
mask, so whole-image agreement is dominated by the table and the bin, which any stereo model
matches easily.

For "does this depth change help?", comparing two full pipeline runs is the wrong instrument: it
costs ~10 minutes per dataset and buries the answer in pose noise. Score depth directly instead —
`scene_depth` can be called on a loaded engine scene by scene, which is seconds per scene.

`test/check_engine_depth_smoke.py --engine` remains the cheap "did the backend run at all" check; it
is deliberately not an accuracy test.

## 9. What counts as a gate

`check_engine_depth_smoke.py --engine` needs one scene and a model. Run it when validation is
authorized; an ONNX cache miss may compile first. Never report unexecuted checks as passed.

None of them is an *accuracy* gate. Pose accuracy is only ever established by running a dataset
and comparing `pose_summary.json` against a saved baseline, which is section 6.

## Timing and failure notes

- Per-scene cost is dominated by the pose stage and scales with the number of proposals it is
  handed, so it varies widely between datasets — a scene dense with instances costs several times
  a sparse one. Time a short run before sizing a long one rather than assuming a rate. The depth
  stage also re-deserializes the engine each scene, deliberately, so TensorRT's scratch is not
  resident while SAM3 and FoundationPose run. A depth-only caller that owns the GPU can keep it
  loaded instead, by calling `scene_depth` directly.
- Run datasets sequentially unless GPU memory and concurrency have been measured. SAM3's box
  decoder adds separate weights/context memory.
- The logs are tqdm progress bars written with `\r`. Piping through `tr '\r' '\n'` first is what
  makes `grep` and `tail` behave.
- For CUDA handle/binding failures, inspect the configured device and resource lifetime in
  `inference/trt.py`. Borrowed features expire after their encoder runs again or is released;
  engines are sequential, and views cannot cross GPUs.
- A depth map that looks plausible and is ~2× wrong means the input convention, not the model.
  Check `normalization: imagenet` in the scene metadata, and in the depth smoke check's output —
  a deployable export fed raw 0–255 is roughly twice as wrong without failing anything.
