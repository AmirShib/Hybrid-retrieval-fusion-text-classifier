"""Encoder adapters behind the TextEncoder port.

- ``SentenceTransformerEncoder``: wraps a SentenceTransformer (torch + a model
  download) and supports fine-tuning with MultipleNegativesSymmetricRankingLoss
  over (item, class-description) pairs.
- ``TfidfEncoder``: a torch-free, air-gap-friendly alternative built on sklearn's
  TfidfVectorizer. Its vocabulary/IDF are corpus-dependent, so it is *fit* on a
  training corpus rather than loaded pretrained.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Sequence

import numpy as np

from ..config import EncoderConfig
from ..domain import LabeledItem, LabelSpace, TextEncoder


class SentenceTransformerEncoder(TextEncoder):
    """Adapter producing L2-normalized float32 embeddings."""

    def __init__(self, model: "SentenceTransformer", batch_size: int = 64):  # type: ignore[name-defined]
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def load(cls, model_name_or_path: str, batch_size: int = 64, device=None) -> "SentenceTransformerEncoder":
        from sentence_transformers import SentenceTransformer

        return cls(SentenceTransformer(model_name_or_path, device=device), batch_size)

    @property
    def model(self):
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def save(self, directory: str) -> None:
        self._model.save(directory)


class TfidfEncoder(TextEncoder):
    """Torch-free TextEncoder: dense, L2-normalized TF-IDF vectors.

    The vocabulary and IDF weights depend on the training corpus, so the encoder
    must be fit before use (``fit`` / ``fit_on``). Rows are L2-normalized so that
    dot product == cosine, honoring the core invariant; an empty or fully-OOV
    text maps to an all-zero (finite, never NaN) vector.

    Persistence is a pickled fitted vectorizer — stdlib + sklearn only, no torch
    and no download, so a model directory stays portable to an air-gapped host.
    """

    _PICKLE_NAME = "tfidf.pkl"

    def __init__(self, vectorizer=None, tfidf_kwargs: dict | None = None):  # type: ignore[no-untyped-def]
        self._vectorizer = vectorizer
        self._kwargs = dict(tfidf_kwargs or {})

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "TfidfEncoder":
        """Build an *unfitted* encoder carrying the configured vectorizer kwargs."""
        return cls(tfidf_kwargs=config.params)

    def fit(self, texts: Sequence[str]) -> "TfidfEncoder":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(**self._kwargs)
        vec.fit(list(texts))
        self._vectorizer = vec
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TfidfEncoder must be fit before encode()")
        X = np.asarray(self._vectorizer.transform(list(texts)).todense(), dtype=np.float32)
        # Enforce unit norm regardless of the vectorizer's own `norm` setting;
        # zero rows (empty/OOV) stay zero rather than becoming NaN.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, 1e-8, None)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, self._PICKLE_NAME), "wb") as fh:
            pickle.dump(self._vectorizer, fh)

    @classmethod
    def load(cls, directory: str, batch_size: int = 64, device=None) -> "TfidfEncoder":
        """Signature mirrors SentenceTransformerEncoder.load for registry parity."""
        with open(os.path.join(directory, cls._PICKLE_NAME), "rb") as fh:
            return cls(vectorizer=pickle.load(fh))


def fit_tfidf_encoder(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: EncoderConfig,
) -> TfidfEncoder:
    """Build + fit a TfidfEncoder on the given items' text.

    Mirrors ``train_encoder``'s signature so the registry can dispatch corpus
    fitting uniformly. Crucially, this is called with a *fold's training rows
    only* in the OOF loop, which is what keeps the vocabulary leakage-free.
    """
    return TfidfEncoder.from_config(config).fit([it.text for it in items])


def train_encoder(
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    config: EncoderConfig,
    output_path: str | None = None,
) -> SentenceTransformerEncoder:
    """Fine-tune a bi-encoder so items sit near their class description.

    Uses MultipleNegativesSymmetricRankingLoss on (item_text, description) pairs.
    NoDuplicatesDataLoader keeps two items of the same class out of one batch,
    which is what prevents the symmetric in-batch negatives from treating an
    item's own description as a negative for a same-class sibling.
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    model = SentenceTransformer(config.model_name_or_path, device=config.device)
    examples = [
        InputExample(texts=[it.text, label_space.descriptions[label_space.index_of(it.label)]])
        for it in items
    ]
    loader = NoDuplicatesDataLoader(examples, batch_size=config.train_batch_size)
    loss = losses.MultipleNegativesSymmetricRankingLoss(model)

    steps_per_epoch = max(1, len(loader))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=config.train_epochs,
        warmup_steps=int(steps_per_epoch * config.train_epochs * config.warmup_ratio),
        output_path=output_path,
        show_progress_bar=False,
    )
    return SentenceTransformerEncoder(model, config.encode_batch_size)
