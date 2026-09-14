#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: LicenseRef-Meta-SAM
#
# UNLIKE THE REST OF THIS REPOSITORY, THIS FILE IS NOT APACHE-2.0.
#
# `_exportable_geometry_forward` below is adapted from SAM3's own geometry-prompt forward pass
# (facebookresearch/sam3 @ 96914d2425f90a64f45ca977c2b5165418099543), rewritten so the graph
# traces cleanly under `torch.onnx.export`. That derivation puts this file under Meta's SAM
# License, which is not OSI-approved and is not Apache-2.0:
#   https://raw.githubusercontent.com/facebookresearch/sam3/96914d2425f90a64f45ca977c2b5165418099543/LICENSE
# Keep adapted upstream code confined to this file so the rest of the tree stays Apache-2.0.
"""Export batch-one SAM3 vision, text, text decoder and single-box decoder to ONNX.

Targets facebookresearch/sam3 revision 96914d2425f90a64f45ca977c2b5165418099543.
Uses upstream forward_grounding, including geometry encoding and presence scores.
The two decoder graphs preserve the distinct empty-prompt and one-box branches;
no dynamic batching or unsupported FP16 conversion is advertised.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foundationpose_perception_pipeline.config import add_config_argument, models_dir_default

DEFAULT_OPSET_VERSION: int = 17
DEFAULT_SAM3_IMAGE_SIZE: int = 1008
DEFAULT_SAM3_CONTEXT_LENGTH: int = 32

VISION_NAMES = ["feat0", "feat1", "feat2", "pos2"]
TEXT_NAMES = ["lang_feat", "lang_mask"]
PREDICTION_NAMES = ["pred_boxes", "pred_logits", "pred_masks", "presence_logit"]


class Sam3VisionEncoderExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, image):
        output = self.backbone.forward_image(image)
        features = output["backbone_fpn"]
        if len(features) != 3:
            raise ValueError("Expected the pinned SAM3 image backbone with three FPN levels")
        return *features, output["vision_pos_enc"][-1]


class Sam3TextEncoderExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language = model.backbone.language_backbone

    def forward(self, input_ids):
        _, memory = self.language.encoder(input_ids)
        # Upstream attention uses sequence-first features and True for padding.
        return self.language.resizer(memory.transpose(0, 1)), input_ids == 0


class Sam3GroundingDecoderExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat0, feat1, feat2, pos2, lang_feat, lang_mask,
                prompt_boxes=None, prompt_labels=None):
        from sam3.model.geometry_encoders import Prompt

        backbone = {
            "backbone_fpn": [feat0, feat1, feat2],
            "vision_pos_enc": [pos2],  # SAM3 grounding uses the last feature level.
            "language_features": lang_feat,
            "language_mask": lang_mask,
        }
        if self.model.num_feature_levels != 1:
            raise ValueError("Expected the pinned SAM3 single-level grounding encoder")
        ids = torch.zeros(1, dtype=torch.long, device=feat0.device)
        find_input = SimpleNamespace(img_ids=ids, text_ids=ids)
        if prompt_boxes is None:
            geometry = self.model._get_dummy_prompt()
        else:
            geometry = Prompt(box_embeddings=prompt_boxes, box_labels=prompt_labels)
        output = self.model.forward_grounding(
            backbone_out=backbone, find_input=find_input, find_target=None, geometric_prompt=geometry
        )
        return tuple(output[name] for name in ("pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"))


def export_component(wrapper, inputs, path, input_names, output_names, skip_existing=False):
    """Export from real upstream output shapes.

    `torch.onnx.export` already validates the graph it writes, so no separate `onnx.checker`
    pass runs here and the `onnx` package is not needed to export.
    """
    wrapper.eval()
    with torch.inference_mode():
        expected = wrapper(*inputs)
        if skip_existing and path.is_file() and path.stat().st_size > 0:
            return expected
        torch.onnx.export(
            wrapper, inputs, str(path), input_names=input_names, output_names=output_names,
            opset_version=DEFAULT_OPSET_VERSION, dynamo=False, do_constant_folding=True,
        )
    return expected


def _exportable_addmm(activation, linear, value):
    # Upstream fused BF16 addmm has no ONNX symbolic. Use ordinary FP32 ops for export.
    if activation not in (F.relu, nn.ReLU, F.gelu, nn.GELU):
        raise ValueError(f"Unsupported activation: {activation}")
    result = linear(value.to(linear.weight.dtype))
    return F.relu(result) if activation in (F.relu, nn.ReLU) else F.gelu(result)


def _exportable_geometry_forward(self, geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
    from sam3.model.geometry_encoders import concat_padded_sequences

    points = geo_prompt.point_embeddings
    points_mask = geo_prompt.point_mask
    points_labels = geo_prompt.point_labels
    boxes = geo_prompt.box_embeddings
    boxes_mask = geo_prompt.box_mask
    boxes_labels = geo_prompt.box_labels
    seq_first_img_feats = img_feats[-1]
    seq_first_img_pos_embeds = (
        img_pos_embeds[-1]
        if img_pos_embeds is not None
        else torch.zeros_like(seq_first_img_feats)
    )

    if self.points_pool_project or self.boxes_pool_project:
        assert len(img_feats) == len(img_sizes)
        cur_img_feat = img_feats[-1]
        cur_img_feat = self.img_pre_norm(cur_img_feat)
        H, W = img_sizes[-1]
        assert cur_img_feat.shape[0] == H * W
        N, C = cur_img_feat.shape[-2:]
        cur_img_feat = cur_img_feat.permute(1, 2, 0).view(N, C, H, W)
        img_feats = cur_img_feat

    bs = boxes.shape[1] if boxes is not None else 1
    if boxes.shape[0] == 0:
        boxes_embeds = torch.zeros(0, bs, self.d_model, device=img_feats.device)
        boxes_mask = torch.zeros(bs, 0, device=img_feats.device, dtype=torch.bool)
    elif not self.encode_boxes_as_points:
        boxes_embeds, boxes_mask = self._encode_boxes(
            boxes=boxes,
            boxes_mask=boxes_mask,
            boxes_labels=boxes_labels,
            img_feats=img_feats,
        )
    else:
        # Upstream can route boxes through the point encoder. The pinned checkpoint does not,
        # and that path is untested here, so refuse rather than export a graph nobody checked.
        raise NotImplementedError(
            "encode_boxes_as_points=True is not supported by this exporter. The pinned SAM3 "
            "revision leaves it off; this checkpoint sets it. Export the decoder with upstream "
            "tooling, or extend this function and validate the result against upstream yourself."
        )

    if points.shape[0] == 0:
        final_embeds, final_mask = boxes_embeds, boxes_mask
    else:
        final_embeds, final_mask = self._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=img_feats,
        )
        if boxes.shape[0] != 0:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, boxes_embeds, boxes_mask
            )

    bs = final_embeds.shape[1]
    if self.cls_embed is not None:
        cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
        cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
        if final_embeds.shape[0] == 0:
            final_embeds, final_mask = cls, cls_mask
        else:
            final_embeds, final_mask = concat_padded_sequences(
                final_embeds, final_mask, cls, cls_mask
            )

    if self.final_proj is not None:
        final_embeds = self.norm(self.final_proj(final_embeds))

    if self.encode is not None:
        for lay in self.encode:
            final_embeds = lay(
                tgt=final_embeds,
                memory=seq_first_img_feats,
                tgt_key_padding_mask=final_mask,
                pos=seq_first_img_pos_embeds,
            )
        final_embeds = self.encode_norm(final_embeds)
    return final_embeds, final_mask


def export_sam3_models(
    checkpoint_path: Path, output_dir: Path, device: str = "cuda", skip_existing: bool = False
):
    import sam3
    import sam3.model.geometry_encoders as geo_enc
    import sam3.model.vitdet as vitdet
    import sam3.model_builder as builder

    output_dir.mkdir(parents=True, exist_ok=True)
    # Scope export-only adaptations; do not mutate the caller's SAM3 implementation permanently.
    with patch.object(builder, "_create_vit_backbone", partial(builder._create_vit_backbone, use_rope_real=True)), \
         patch.object(vitdet, "addmm_act", _exportable_addmm), \
         patch.object(geo_enc.SequenceGeometryEncoder, "forward", _exportable_geometry_forward):
        model = builder.build_sam3_image_model(
            checkpoint_path=str(checkpoint_path), load_from_HF=False, device=device, eval_mode=True
        ).float().eval()
        vision = Sam3VisionEncoderExportWrapper(model).eval()
        text = Sam3TextEncoderExportWrapper(model).eval()
        decoder = Sam3GroundingDecoderExportWrapper(model).eval()
        # The actual forwards supply feature dimensions; no hand-written feature shapes.
        image = torch.rand(1, 3, DEFAULT_SAM3_IMAGE_SIZE, DEFAULT_SAM3_IMAGE_SIZE, device=device) * 2 - 1
        tokens = model.backbone.language_backbone.tokenizer(["metal part"], context_length=DEFAULT_SAM3_CONTEXT_LENGTH).to(device)
        export = partial(export_component, skip_existing=skip_existing)
        features = export(vision, (image,), output_dir / "sam3_vision_encoder.onnx",
                                    ["image"], VISION_NAMES)
        language = export(text, (tokens,), output_dir / "sam3_text_encoder.onnx",
                                    ["input_ids"], TEXT_NAMES)
        inputs = (*features, *language)
        export(decoder, inputs, output_dir / "sam3_mask_decoder.onnx",
                         VISION_NAMES + TEXT_NAMES, PREDICTION_NAMES)
        boxes = torch.tensor([[[0.5, 0.5, 0.3, 0.3]]], device=device)
        labels = torch.ones((1, 1), dtype=torch.long, device=device)
        export(decoder, (*inputs, boxes, labels), output_dir / "sam3_box_decoder.onnx",
                         VISION_NAMES + TEXT_NAMES + ["prompt_boxes", "prompt_labels"], PREDICTION_NAMES)
        # Keep the vocabulary identical to the upstream model used during export.
        vocab = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        shutil.copyfile(vocab, output_dir / vocab.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=models_dir_default())
    parser.add_argument("--skip-existing", action="store_true", help="Skip re-exporting existing ONNX files")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        # Upstream SAM3 builds its position-encoding and boxRPB caches with a hardcoded
        # device="cuda" in __init__, so there is no CPU export path to offer.
        raise SystemExit("Export needs a CUDA device. Select one with CUDA_VISIBLE_DEVICES.")
    export_sam3_models(
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
