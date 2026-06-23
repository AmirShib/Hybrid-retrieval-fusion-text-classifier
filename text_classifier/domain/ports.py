"""Ports: the abstract boundaries the application layer depends on. Infrastructure
adapters implement these; the domain never imports a concrete ML library.

All array shapes are documented as (rows, cols). `b` = query batch size,
`C` = number of classes, `k` = neighbors, `d` = embedding dim, `n` = pool size.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np


class TextEncoder(ABC):
    """Maps text to L2-normalized embeddings (so dot product == cosine)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:  # (n, d) float32
        ...


class DenseRetriever(ABC):
    """Semantic signals from a bi-encoder over an example pool + class set."""

    @abstractmethod
    def knn_example_labels(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_class_indices (b, k) int, similarities (b, k) float)."""

    @abstractmethod
    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        """(b, C) cosine to each class prototype; NaN column for absent classes."""

    @abstractmethod
    def description_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        """(b, C) cosine to each class-description embedding."""

    @property
    @abstractmethod
    def class_freq(self) -> np.ndarray:  # (C,) int
        ...


class LexicalRetriever(ABC):
    """Lexical signals from BM25 over the same example pool + class descriptions."""

    @abstractmethod
    def knn_example_labels(self, query_texts: Sequence[str], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_class_indices (b, k) int with -1 padding,
        scores (b, k) float with NaN padding)."""

    @abstractmethod
    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        """(b, C) BM25 score against each class description; 0 where no overlap."""


class FusionModel(ABC):
    """Pointwise model scoring P(candidate is the true class). Must tolerate NaN
    features (the 'not retrieved' encoding)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # (n,) P(class==1)
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "FusionModel": ...


class ConfidenceCalibrator(ABC):
    """Maps raw fusion scores onto calibrated P(correct)."""

    @abstractmethod
    def fit(self, scores: np.ndarray, correct: np.ndarray) -> None: ...

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ConfidenceCalibrator": ...
