"""Deterministic n-gram hash embeddings (no model download). Optional BGE later."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

DIM = 384


def _ngrams(text: str, n: int) -> list[str]:
    s = "".join(text.lower().split())
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def embed_texts(texts: Sequence[str], dim: int = DIM) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, raw in enumerate(texts):
        text = raw or ""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = list(text.lower().split())
        grams: list[str] = []
        grams.extend(tokens)
        grams.extend(_ngrams(text, 1))
        grams.extend(_ngrams(text, 2))
        grams.extend(_ngrams(text, 3))
        for g in grams:
            digest = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec /= nrm
        out[i] = vec
    return out
