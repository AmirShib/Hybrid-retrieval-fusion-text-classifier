"""Shared test doubles for the text-classifier test suite.

Single source of truth for HashingEncoder and make_synthetic — imported by
both tests/ and scripts/demo.py so there is no copy-paste divergence.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Sequence

import numpy as np

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.domain import TextEncoder


class HashingEncoder(TextEncoder):
    """Deterministic bag-of-hashed-tokens embeddings, L2-normalized.

    Shared tokens produce higher cosine similarity, making retrieval
    meaningful enough to smoke-test the pipeline offline without a real
    bi-encoder or network access.

    Token hashing uses ``hashlib.sha256`` rather than the builtin ``hash()`` so
    that embeddings are byte-for-byte identical across Python processes,
    versions, and platforms — no ``PYTHONHASHSEED`` pinning required.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    @staticmethod
    def _bucket_and_sign(token: str) -> tuple[int, float]:
        """Map a token to a (bucket_index_seed, signed_weight) pair via SHA-256.

        First 4 bytes (little-endian) seed the bucket; bit 0 of byte 4 picks the
        sign. SHA-256 is stable everywhere, so this is fully reproducible.
        """
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:4], "little")
        sign = 1.0 if digest[4] & 1 else -1.0
        return h, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                h, sign = self._bucket_and_sign(tok)
                out[i, h % self.dim] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-8, None)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "hashing_encoder.json"), "w") as fh:
            json.dump({"dim": self.dim}, fh)

    @classmethod
    def load(cls, path: str, batch_size: int = 64, device=None) -> "HashingEncoder":
        """Signature matches SentenceTransformerEncoder.load so it can be patched in."""
        try:
            with open(os.path.join(path, "hashing_encoder.json")) as fh:
                return cls(dim=json.load(fh)["dim"])
        except FileNotFoundError:
            return cls()


def make_synthetic(
    n_classes: int = 40, per_class: int = 60, seed: int = 0
) -> tuple[LabelSpace, list[LabeledItem]]:
    """Return (LabelSpace, list[LabeledItem]) built from synthetic data.

    Deliberately imbalanced: classes divisible by 7 get `per_class` items;
    all others get per_class//5.  Fixed seed makes outputs reproducible.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(600)]
    definitions: list[ClassDefinition] = []
    items: list[LabeledItem] = []
    for c in range(n_classes):
        theme = rng.choice(vocab, size=8, replace=False)
        definitions.append(ClassDefinition(
            key=f"CLS{c:03d}",
            description=f"class about {' '.join(theme[:4])} and related {' '.join(theme[4:])}",
        ))
        count = per_class if c % 7 else max(8, per_class // 5)
        for _ in range(count):
            k = int(rng.integers(4, 9))
            words = list(rng.choice(theme, size=min(k, len(theme)), replace=True))
            words += list(rng.choice(vocab, size=2, replace=True))
            rng.shuffle(words)
            items.append(LabeledItem(text=" ".join(words), label=f"CLS{c:03d}"))
    rng.shuffle(items)
    return LabelSpace(definitions), items
