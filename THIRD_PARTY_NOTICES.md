# Third-Party Notices

FoundationPose Perception Pipeline
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE) in the root of this
repository, or <https://www.apache.org/licenses/LICENSE-2.0>).

This notice is provided in accordance with Section 4(d) of the Apache License, Version 2.0, and
with the redistribution and attribution terms of the upstream licenses listed below. Each
third-party component remains subject to its own license. In the event of any conflict, the
third-party license terms govern that component's use, reproduction, and distribution.

---

## 1. Third-party code this project derives from

### Isaac Lab

- **Project:** <https://github.com/isaac-sim/IsaacLab>
- **License:** BSD 3-Clause "New" or "Revised" License (`BSD-3-Clause`)
- **Copyright:** Copyright (c) 2022-2026, The Isaac Lab Project Developers
- **Used in:**
  - `internal/gt_depth_generation/build_scene.py`
  - `internal/gt_depth_generation/convert_ply_to_usd.py`
  - `internal/gt_depth_generation/render_all_scenes.py`

These files are derived from Isaac Lab and remain subject to the license below. They have been
modified by NVIDIA.

```
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### SAM3

- **Project:** <https://github.com/facebookresearch/sam3>
- **License:** `LicenseRef-Meta-SAM` (Meta's SAM License, not OSI-approved) —
  <https://raw.githubusercontent.com/facebookresearch/sam3/96914d2425f90a64f45ca977c2b5165418099543/LICENSE>
- **Copyright:** Copyright (c) Meta Platforms, Inc. and affiliates.
- **Used in:**
  - `tools/export_sam3_to_onnx.py` — `_exportable_geometry_forward` is an adaptation of
    SAM3's own geometry-prompt forward pass, rewritten so the graph traces under
    `torch.onnx.export`. Pinned to upstream revision `96914d2`.

That file carries the Meta SAM License header and remains subject to it. The rest of this
repository is Apache-2.0.

The NumPy tokenizer in `src/foundationpose_perception_pipeline/inference/sam3/tokenizer.py`
is an independent reimplementation written against SAM3's published tokenizer behaviour — the
BPE vocabulary file it loads is a separately obtained SAM3 artifact under the license above.
Upstream's tokenizer descends from OpenCLIP and OpenAI CLIP, both MIT.

---

## 2. Runtime dependencies

Declared in `[project].dependencies` in [pyproject.toml](pyproject.toml), and installed from PyPI
or from PyTorch's own index. They are not redistributed with this code.

| Component | License | Copyright | Project |
|---|---|---|---|
| numpy | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | NumPy Developers | <https://github.com/numpy/numpy> |
| scipy | BSD-3-Clause | SciPy Developers; Enthought, Inc. | <https://github.com/scipy/scipy> |
| opencv-python | Apache-2.0; packaging wrapper scripts MIT | OpenCV team; Olli-Pekka Heinisuo and contributors | <https://github.com/opencv/opencv-python> |
| pyyaml | MIT | Ingy döt Net; Kirill Simonov | <https://github.com/yaml/pyyaml> |
| torch | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | Meta Platforms, Inc. and affiliates, and the PyTorch contributors | <https://github.com/pytorch/pytorch> |
| torchvision | BSD-3-Clause | Soumith Chintala and the TorchVision contributors | <https://github.com/pytorch/vision> |
| pillow | `MIT-CMU` | Jeffrey A. Clark and contributors; Secret Labs AB; Fredrik Lundh | <https://github.com/python-pillow/Pillow> |
| tqdm | `MPL-2.0 AND MIT` | tqdm developers; noamraph | <https://github.com/tqdm/tqdm> |
| einops | MIT | Alex Rogozhnikov | <https://github.com/arogozhnikov/einops> |
| pycocotools | BSD-2-Clause | Piotr Dollar and Tsung-Yi Lin | <https://github.com/ppwwyyxx/cocoapi> |
| psutil | BSD-3-Clause | Giampaolo Rodola; Jay Loden; Dave Daeschler | <https://github.com/giampaolo/psutil> |
| regex | [Apache-2.0 additions; inherited Python license terms](https://github.com/mrabarnett/mrab-regex/blob/hg/LICENSE.txt) | regex contributors; inherited CPython/Secret Labs notices | <https://github.com/mrabarnett/mrab-regex> |
| ftfy | Apache-2.0 | Robyn Speer | <https://github.com/rspeer/python-ftfy> |
| setuptools | MIT | Jason R. Coombs and the Setuptools contributors | <https://github.com/pypa/setuptools> |

---

## 3. Optional and separately installed dependencies

Not installed by a default `uv sync`. TensorRT, the CUDA runtime and the CUDA bindings are no
longer optional -- they are declared in `[project].dependencies` and listed in section 2, because
every inference stage is now a TensorRT engine. See [README.md](README.md) for installation.

| Component | License | Project |
|---|---|---|
| markdown | BSD-3-Clause | <https://github.com/Python-Markdown/markdown> |
| ruff | MIT | <https://github.com/astral-sh/ruff> |
| cuda-bindings | License terms supplied with the installed NVIDIA CUDA Python distribution | <https://github.com/NVIDIA/cuda-python> |
| tensorrt-cu13 | NVIDIA TensorRT license terms, supplied with the package | <https://developer.nvidia.com/tensorrt> |
| nvidia-cuda-runtime | NVIDIA CUDA Toolkit EULA | <https://docs.nvidia.com/cuda/eula/> |

---

## 4. Models and Libs obtained separately

None of these are redistributed with this code, and none are covered by this repository's
Apache-2.0 license. Code and weights are licensed separately for every one of them.

| Component | Code | Weights / checkpoint |
|---|---|---|
| FoundationPose Inference Library | Apache-2.0, public on [GitHub](https://github.com/nvidia-isaac/foundation-pose-inference-library) | separate NGC artifact — `nvidia/tao/foundationpose:deployable_v1.0` terms, not Apache-2.0 |
| SAM3 | `LicenseRef-Meta-SAM` (Meta's SAM License, not OSI-approved) | same license; the checkpoint is gated — request access at <https://huggingface.co/facebook/sam3> |
| FoundationStereo (TAO `deployable_*`) | executed as a TensorRT engine; no source is imported | separate NGC artifact — the [model page](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationstereo)'s terms |

SAM3 source/checkpoint access is needed for ONNX export; runtime uses the exported graphs or
TensorRT plans and the copied vocabulary. See §1 for the one place upstream SAM3 source is
adapted rather than merely used. The local tokenizer reuses upstream's BPE vocabulary and
splitting pattern and reproduces its text-cleaning chain, `ftfy` repair included, so prompts
tokenize to the same ids.
Exporting or compiling model artifacts does not replace their upstream terms. Section 8 of
[README.md](README.md) covers these artifacts.

Use of this pipeline may rely on third party components or models that you must download
separately. Those components or models are subject to the applicable open source licenses or other
license terms, including any proprietary notices, disclaimers, requirements, and extended use
rights.

---

## Notes

- This file lists only **direct** dependencies. Transitive dependencies installed by `pip`/`uv`
  carry their own license metadata in their respective distributions and are not re-listed here.
- Several components bundle third-party code inside their own distributions — `torch`, `scipy`,
  and `opencv-python` among them. Consult each project's own notice files for the
  components it embeds.
