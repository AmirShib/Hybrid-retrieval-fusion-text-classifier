"""Ports: the abstract boundaries the application layer depends on. Infrastructure
adapters implement these; the domain never imports a concrete ML library.

All array shapes are documented as (rows, cols). `b` = query batch size,
`C` = number of classes, `k` = neighbors, `d` = embedding dim, `n` = pool size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import LabeledItem, LabelSpace


class TextEncoder(ABC):
    """Maps text to L2-normalized embeddings (so dot product == cosine)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:  # (n, d) float32
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:  # (n, d) float32
        """Encode texts in the *query* role (the items being classified).

        Defaults to symmetric ``encode``, so existing adapters need no change;
        instruction-tuned adapters (E5/BGE-style prompts) override this. The
        pipelines route every encode call by role."""
        return self.encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:  # (n, d) float32
        """Encode texts in the *document* role (the example pool + class
        descriptions queries are matched against). Defaults to symmetric
        ``encode``; see ``encode_queries``."""
        return self.encode(texts)

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist to a directory (encoders may write several files). The
        matching loader is registered per kind (see ``EncoderSpec.load``);
        FusionModel/ConfidenceCalibrator declare the same contract."""


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
    def knn_example_labels(
        self, query_texts: Sequence[str], k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_class_indices (b, k) int with -1 padding,
        scores (b, k) float with NaN padding)."""

    @abstractmethod
    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        """(b, C) BM25 score against each class description; 0 where no overlap."""


class FusionModel(ABC):
    """Model scoring P(candidate is the true class). Must tolerate NaN features
    (the 'not retrieved' encoding).

    ``NEEDS_GROUPS`` flags learning-to-rank backends (e.g. XGBRanker) that need a
    per-query ``groups`` array at fit time. Pointwise backends leave it False and
    ignore ``groups``; the training pipeline only computes/passes groups when a
    model declares it needs them, so existing call sites stay ``fit(X, y)``."""

    NEEDS_GROUPS: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
        """Fit on features ``X`` and binary labels ``y``. ``groups`` (one count
        per query, summing to ``len(X)``) is required only when
        ``NEEDS_GROUPS`` is True; pointwise models ignore it."""

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


@dataclass(frozen=True)
class FeatureContext:
    """The per-(item, candidate) grid a ``FeatureProvider`` computes over.

    A provider produces a value for every *candidate row* — one (query, class)
    pair that survived candidate selection. ``rows`` and ``cols`` are the parallel
    index arrays into that grid: candidate ``i`` is query ``rows[i]`` paired with
    class index ``cols[i]``. A provider builds whatever per-query or per-class
    intermediate it needs (typically a ``(n_queries, n_classes)`` matrix) and
    gathers the answer with ``M[ctx.rows, ctx.cols]`` — vectorized, no per-row
    Python loop on the hot path (CLAUDE.md convention).

    Everything here is already computed by ``FeatureAssembler`` for the same chunk,
    so exposing it costs nothing: ``query_texts``/``query_emb`` are the queries in
    this chunk (``query_emb`` is L2-normalized, so dot products are cosines), and
    ``label_space`` owns the canonical column↔class-key map. Retrieval indices are
    deliberately *not* exposed here — that is a separate, later capability (T79)."""

    query_texts: Sequence[str]  # (n_queries,)
    query_emb: np.ndarray  # (n_queries, dim), L2-normalized
    rows: np.ndarray  # (n_candidates,) int: candidate -> query index
    cols: np.ndarray  # (n_candidates,) int: candidate -> class index
    label_space: LabelSpace

    @property
    def n_queries(self) -> int:
        return len(self.query_texts)

    @property
    def n_candidates(self) -> int:
        return int(self.rows.shape[0])

    @property
    def n_classes(self) -> int:
        return self.label_space.size


class FeatureProvider(ABC):
    """A pluggable source of extra fusion features (T70).

    The five retrieval signals → ~28 core features are fixed; a provider adds
    columns *beyond* them (text length, a domain-lexicon hit, an external score).
    The assembler concatenates the core columns with every active provider's
    columns; the composed, ordered name list is persisted into ``meta.json`` so
    inference rebuilds the identical column order. Getting that order wrong makes
    train and infer silently disagree on what a column means, so the contract is
    strict:

    - ``names()`` returns this provider's column names — **stable and unique**,
      the same list before and after ``fit`` (they are composed into the schema
      from an *unfitted* instance at train time).
    - ``compute(ctx)`` returns ``{name: (n_candidates,) float array}`` for exactly
      the names in ``names()``, one value per candidate row of ``ctx``. It must be
      vectorized (gather over ``ctx.rows``/``ctx.cols``). "This provider did not
      fire for this (item, candidate)" is emitted as ``NaN`` — never a true ``0``
      — so XGBoost consumes it as missing (CLAUDE.md invariant).
    - ``save``/``load`` round-trip any fitted state through a directory using
      portable formats only (stdlib pickle + numpy + json + native model
      formats), so the model dir ships to an air-gapped host.

    A provider that derives state from training data overrides ``fit``; the
    training pipeline fits it **per fold on the other folds' rows** (exactly like
    prototypes/indices), so an item never sees itself in state built from its own
    fold. A stateless provider leaves ``fit`` as the identity default."""

    @abstractmethod
    def names(self) -> List[str]:
        """Stable, unique column names this provider contributes."""

    def fit(self, items: Sequence[LabeledItem], label_space: LabelSpace) -> "FeatureProvider":
        """Learn any training-data-derived state from ``items`` and return self.

        The default is the identity (stateless providers need no fit). Providers
        with learned state override this; the training pipeline calls it per fold
        on training rows only, keeping provider features leakage-free."""
        return self

    @abstractmethod
    def compute(self, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Return ``{name: (n_candidates,) float array}`` for the names in
        ``names()``. ``NaN`` marks "did not fire" (never a true 0)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist fitted state to directory ``path`` (portable formats only)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "FeatureProvider":
        """Reload a provider persisted by ``save`` — no labels, no network."""
