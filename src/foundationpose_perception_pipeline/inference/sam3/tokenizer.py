#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BPE tokenizer for SAM 3 text prompts, reimplemented in NumPy.

Its own module, not part of `trt_processor`, because nothing here touches TensorRT, CUDA or
torch: this is string handling, and keeping it separate is what lets it be read and reviewed
without the inference stack alongside it.

Upstream's tokenizer (`sam3/model/tokenizer_ve.py`, revision `96914d2`) is deliberately not
reused. It imports `torch` and returns tensors, which would put torch back on a runtime path
where every stage is otherwise a TensorRT engine; it needs `iopath` purely to open a local
`.gz`; it cannot be installed from PyPI, so reusing it means shipping the gated `sam3` checkout
to every deployment box; and it is covered by the Meta SAM License rather than Apache-2.0, so
vendoring it would put a Meta-licensed file on the runtime path of an Apache-2.0 library. This
is therefore an independent reimplementation written against SAM3's published tokenizer
behaviour, and the only SAM3 artifact involved is the BPE vocabulary itself, obtained separately
and disclosed in THIRD_PARTY_NOTICES.md §1.

The consequence to keep in mind: because this is a reimplementation rather than a copy, upstream
drift is silent. It does not raise -- it shifts token ids, and therefore detections.
"""

from __future__ import annotations

import gzip
import html
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import ftfy
import numpy as np
import regex as re

#: Vocabulary shipped with SAM3 and copied next to the exported engines during ONNX export.
BPE_VOCAB_FILENAME: str = "bpe_simple_vocab_16e6.txt.gz"

DEFAULT_CONTEXT_LENGTH: int = 32
BPE_VOCAB_SIZE: int = 49152
NUM_SPECIAL_TOKENS: int = 2
NUM_BYTE_TOKENS: int = 256
BPE_MERGES_END: int = BPE_VOCAB_SIZE - NUM_BYTE_TOKENS - NUM_SPECIAL_TOKENS + 1
BYTE_ENCODER_SIZE: int = 256


@lru_cache
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(BYTE_ENCODER_SIZE):
        if b not in bs:
            bs.append(b)
            cs.append(BYTE_ENCODER_SIZE + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


def _basic_clean(text: str) -> str:
    """Repair mojibake, then undo two rounds of HTML escaping.

    Mirrors `basic_clean` in upstream `sam3/model/tokenizer_ve.py`. The double `html.unescape` is
    upstream's, not a typo: it undoes text that was escaped twice (`&amp;amp;` -> `&`).
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    """Collapse every whitespace run to a single space. Mirrors upstream `whitespace_clean`."""
    return re.sub(r"\s+", " ", text).strip()


def _clean_lower(text: str) -> str:
    """Upstream's `clean="lower"` chain, which is the mode SAM3 builds its tokenizer in.

    `sam3.model_builder._create_text_encoder` constructs `SimpleTokenizer(bpe_path=bpe_path)` with
    no `clean=` argument, so the constructor default `"lower"` applies. The other two upstream
    modes (`"canonicalize"`, `"whitespace"`) are unreachable for this model and are not ported.
    """
    return _whitespace_clean(_basic_clean(text)).lower()


class Sam3TextTokenizer:
    """NumPy BPE tokenizer for SAM3 text prompts, matching upstream's ids exactly.

    Corresponds to upstream's `SimpleTokenizer` (`sam3/model/tokenizer_ve.py`), reimplemented
    rather than copied for the reasons in this module's docstring. It loads upstream's
    `bpe_simple_vocab_16e6.txt.gz`, applies the same `regex` split pattern, and runs the same
    cleaning chain (`_clean_lower`), so the same prompt produces the same ids -- returned as an
    `np.int64` array instead of a `torch.Tensor`.

    `context_length` defaults to 32 to match upstream: `VETextEncoder.__init__` defaults to 32 and
    `sam3.model_builder._create_text_encoder` does not override it, so `VETextEncoder.forward`
    always calls the tokenizer with `context_length=32`. The `DEFAULT_CONTEXT_LENGTH = 77` in
    upstream's tokenizer module is a fallback that SAM3's image model never reaches.
    """

    def __init__(self, bpe_path: Path | str, context_length: int = DEFAULT_CONTEXT_LENGTH):
        self.byte_encoder = _bytes_to_unicode()
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:BPE_MERGES_END]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab += [v + "</w>" for v in vocab]
        for m in merges:
            vocab.append("".join(m))
        vocab += ["<start_of_text>", "<end_of_text>"]
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.bpe_ranks = {token: index for index, token in enumerate(merges)}
        self.cache: dict[str, str] = {"<start_of_text>": "<start_of_text>", "<end_of_text>": "<end_of_text>"}
        self.pat = re.compile(r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.I)
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = (*token[:-1], token[-1] + "</w>")
        pairs = set(pairwise(word))
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = set(pairwise(word))
        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, texts: str | Sequence[str], context_length: int | None = None) -> np.ndarray:
        context_length = self.context_length if context_length is None else context_length
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        for text in texts:
            cleaned = _clean_lower(text)
            tokens = [self.sot_token_id]
            for token in re.findall(self.pat, cleaned):
                token_bpe = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                tokens.extend(self.encoder[t] for t in self.bpe(token_bpe).split(" "))
            tokens.append(self.eot_token_id)
            all_tokens.append(tokens)
        result = np.zeros((len(all_tokens), context_length), dtype=np.int64)
        for i, tokens in enumerate(all_tokens):
            tok = list(tokens)
            if len(tok) > context_length:
                tok = tok[:context_length]
                tok[-1] = self.eot_token_id
            result[i, : len(tok)] = tok
        return result


__all__ = [
    "BPE_VOCAB_FILENAME",
    "DEFAULT_CONTEXT_LENGTH",
    "Sam3TextTokenizer",
]
