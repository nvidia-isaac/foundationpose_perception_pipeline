# FoundationStereo engine construction

Use the TAO `deployable_*` ONNX from the
[Hugging Face model page](https://huggingface.co/nvidia/c-foundationstereo-s).
Its model-page terms apply separately from the pipeline's code license. No FoundationStereo
source checkout or second Python environment is used.

## Obtain the export

Reuse a supplied ONNX path. For a new download, let the user choose an export from the
[Hugging Face model files](https://huggingface.co/nvidia/c-foundationstereo-s/tree/main/onnx).

Prefer a dynamic ONNX export for measuring this rig's shape. A fixed-shape export requires the
exact baked-in dimensions, such as 320x736, and may require resampling. Identify which export is
present before selecting build arguments.

## Measure, build, and configure

The scene must first be adapted to the pipeline's rig layout. `--shape-from-scene` measures pair
selection and rectification; raw image dimensions are not the TensorRT input dimensions.
Use the dataset profile's split (shipped profiles use `test`):

```bash
./.venv/bin/python tools/build_tao_engine.py \
  --config <profile> \
  --onnx <actual-downloaded-onnx> \
  --shape-from-scene <dataset-root>/<dataset>/<split>/<scene> \
  --max-width <profile-depth-foundation_stereo_max_width>
```

FP32 and a static min=opt=max profile are the defaults. Keep them for accuracy comparisons.
TAO allocates at the profile's maximum shape; a broad dynamic engine wastes GPU memory. If data
has not arrived, defer the shape-dependent build. A user-requested provisional engine must be
labelled provisional and rebuilt from an adapted scene before accuracy evaluation.

Record the generated filename and its JSON sidecar. The engine depends on GPU architecture,
TensorRT version, precision, input shape, and source ONNX hash. Rebuild when these change;
never commit engines or bypass stale-sidecar checks just to get a run to start.

Edit the dataset profile's existing commented engine entry, resolving relative paths from that
profile's directory:

```yaml
overrides:
  depth:
    engine: ../../models/<generated-engine>.engine
```

Keep this machine-specific setting local. `depth.engine` is deliberately unset in shipped
profiles. The runtime model path selects the backend; `--depth-backend commercial` asserts the
expected path and is not a model selector or proof of license approval.

Run `tools/verify_foundationstereo.py` and `test/check_engine_depth_smoke.py` as in SKILL.md.
Require `backend=tao`, `normalization=imagenet`, and no cropping warning. Static engines can
rescale and crop instead of failing on mismatched dimensions, so changing
`foundation_stereo_max_width` requires rebuilding for that width and regenerating cached depth.
