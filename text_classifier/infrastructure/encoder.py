"""Encoder adapter: wraps a SentenceTransformer behind the TextEncoder port and
provides fine-tuning with MultipleNegativesSymmetricRankingLoss over
(item, class-description) pairs.
"""
from __future__ import annotations

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
