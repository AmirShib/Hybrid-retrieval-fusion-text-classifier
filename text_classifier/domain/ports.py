"""Ports: the abstract boundaries the application layer depends on. Infrastructure
adapters implement these; the domain never imports a concrete ML library.

All array shapes are documented as (rows, cols). `b` = query batch size,
`C` = number of classes, `k` = neighbors, `d` = embedding dim, `n` = pool size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .models import LabeledItem, LabelSpace


class ArrayOps(ABC):
    """Narrow array-backend seam (T84) for the numeric kernels in feature
    assembly and dense retrieval. A numpy backend is the only one registered
    today; a torch backend (T85) implements the same surface so those kernels
    run device-resident without a second, drifting implementation.

    Deliberately narrow: only the primitives the existing kernels actually
    call, not an array-API reimplementation. ``to_host`` is the *only*
    sanctioned exit to a plain ``numpy.ndarray`` — every other method may
    return a backend-native array — so every transfer off-device is one
    greppable call."""

    name: str  # "numpy" | "torch"

    @abstractmethod
    def asarray(self, x: Any, dtype: Optional[Any] = None) -> Any: ...

    @abstractmethod
    def to_host(self, x: Any) -> np.ndarray:
        """Materialize ``x`` as a plain ``numpy.ndarray`` on the host. The only
        sanctioned exit from backend-native arrays."""

    @abstractmethod
    def zeros(self, shape: Any, dtype: Any) -> Any: ...

    @abstractmethod
    def full(self, shape: Any, value: Any, dtype: Any) -> Any: ...

    @abstractmethod
    def where(self, cond: Any, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def isnan(self, x: Any) -> Any: ...

    @abstractmethod
    def isfinite(self, x: Any) -> Any: ...

    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def log1p(self, x: Any) -> Any: ...

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def topk(self, x: Any, k: int, axis: int = -1) -> Tuple[Any, Any]:
        """Top-``k`` values and indices along ``axis``, best-first."""

    @abstractmethod
    def argsort(self, x: Any, axis: int = -1) -> Any: ...

    @abstractmethod
    def argpartition(self, x: Any, k: int, axis: int = -1) -> Any: ...

    @abstractmethod
    def nanmin(self, x: Any, axis: Optional[int] = None) -> Any: ...

    @abstractmethod
    def nanmax(self, x: Any, axis: Optional[int] = None) -> Any: ...

    @abstractmethod
    def scatter_add(self, target: Any, rows: Any, cols: Any, values: Any) -> Any:
        """``target[rows[i], cols[i]] += values[i]`` for every ``i``, summing
        duplicates. Returns the updated array (backends need not mutate
        in place); ``target`` is 2-D, ``rows``/``cols``/``values`` are 1-D and
        the same length."""

    @abstractmethod
    def scatter_max(self, target: Any, rows: Any, cols: Any, values: Any) -> Any:
        """``target[rows[i], cols[i]] = max(target[rows[i], cols[i]], values[i])``
        for every ``i``. Same shape contract as ``scatter_add``."""

    @abstractmethod
    def gather(self, M: Any, rows: Any, cols: Any) -> Any:
        """``(n,)``: ``M[rows[i], cols[i]]`` for every ``i``."""


class TextEncoder(ABC):
    """Maps text to L2-normalized embeddings (so dot product == cosine).

    The return type is ``(n, d)`` float32 in whatever array type the
    encoder's configured backend uses (T85): numpy for every encoder kind by
    default, or a resident torch tensor for ``SentenceTransformerEncoder``
    with ``array_backend="torch"`` (see its docstring). L2-normalization is
    the invariant that never changes; the container is not."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Any:  # (n, d) float32
        ...

    def encode_queries(self, texts: Sequence[str]) -> Any:  # (n, d) float32
        """Encode texts in the *query* role (the items being classified).

        Defaults to symmetric ``encode``, so existing adapters need no change;
        instruction-tuned adapters (E5/BGE-style prompts) override this. The
        pipelines route every encode call by role."""
        return self.encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> Any:  # (n, d) float32
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
    def knn_example_labels(
        self, query_emb: np.ndarray, k: int, exclude_idx: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_class_indices (b, k) int, similarities (b, k) float).

        ``exclude_idx`` (b,) gives, per query, one example-pool index to drop from
        that query's neighbors (or a value < 0 to drop nothing) — the leave-one-out
        self-mask, so a query that is itself in the pool never retrieves itself.
        ``None`` (the default) excludes nothing, the ordinary retrieval path."""

    @abstractmethod
    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        """(b, C) cosine to each class prototype; NaN column for absent classes."""

    def loo_prototype_similarity(self, query_emb: np.ndarray, self_idx: np.ndarray) -> np.ndarray:
        """Leave-one-out prototype similarity for queries that are themselves in
        the example pool.

        Like ``prototype_similarity``, but for each query ``i`` whose own example
        index is ``self_idx[i] >= 0``, that query's own-class prototype is
        recomputed with example ``self_idx[i]`` removed (NaN if it was the class's
        only example). Every other class column is the ordinary prototype. Queries
        with ``self_idx[i] < 0`` (not in the pool) fall back to the ordinary value.

        Optional capability: only retrievers that back leave-one-out training
        (``n_folds=1``) implement it; the default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support leave-one-out prototype similarity"
        )

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
        self, query_texts: Sequence[str], k: int, exclude_idx: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (neighbor_class_indices (b, k) int with -1 padding,
        scores (b, k) float with NaN padding).

        ``exclude_idx`` (b,) gives, per query, one example-pool index to drop from
        that query's neighbors (or a value < 0 to drop nothing) — the leave-one-out
        self-mask. ``None`` (the default) excludes nothing."""

    @abstractmethod
    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        """(b, C) BM25 score against each class description; 0 where no overlap."""


@dataclass(frozen=True)
class CandidateView:
    """The shortlist, handed to a *second-stage* ``SignalProvider`` (T33).

    Candidate selection splits signal computation into two rounds. Round one is
    every ordinary provider: it scores all ``C`` classes and its top-n join the
    candidate union. Round two is providers that declare ``needs_candidates`` —
    a reranker is the motivating case, being far too slow to score every class,
    so it can only run once the shortlist exists.

    - ``mask``: the ``(b, C)`` boolean candidate mask.
    - ``rows``/``cols``: its ``np.nonzero`` decomposition — candidate ``i`` is
      query ``rows[i]`` paired with class ``cols[i]``, the same parallel-index
      convention ``FeatureContext`` uses.
    - ``signals``: round one's ``{node: (b, C) value}`` matrices, read-only. A
      second-stage provider needs these to decide *which* candidates are worth
      its cost (e.g. rerank the top-k by ``"dense.desc"``) rather than paying
      for the whole shortlist. Look nodes up with ``.get`` and degrade
      gracefully: which signals ran is a config decision, so a node is not
      guaranteed to be present.
    """

    mask: np.ndarray  # (b, C) bool
    rows: np.ndarray  # (n_candidates,) int -> query index
    cols: np.ndarray  # (n_candidates,) int -> class index
    signals: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SignalContext:
    """The per-chunk inputs a ``SignalProvider.build`` computes over — the same
    chunk ``FeatureAssembler._assemble_chunk`` is assembling, so a provider that
    wraps a retriever sees exactly the arguments the retriever ports already take.

    ``self_ids`` mirrors ``FeatureAssembler.assemble``'s leave-one-out mode
    (``n_folds=1``): when given, each query is itself in the pool being
    retrieved against, and a provider whose signal can self-match (kNN, a
    prototype) must mask its own index out, exactly like
    ``DenseRetriever.knn_example_labels``/``loo_prototype_similarity``.

    ``label_space`` is the canonical column<->class map, mirroring
    ``FeatureContext``'s. A provider that scores against *class-side text*
    (descriptions, taxonomy views) reads it from here rather than holding a copy,
    which is what keeps such a provider stateless with respect to the taxonomy —
    added classes are picked up automatically and it needs no
    ``rewrap_signal_providers`` case.

    ``candidates`` is ``None`` in round one and populated in round two (see
    ``CandidateView``). It is one field rather than several loose optionals so
    the two rounds cannot be half-read: a provider either has the shortlist or
    it does not."""

    texts: Sequence[str]
    q_emb: np.ndarray  # (b, dim), L2-normalized
    k: int
    n_classes: int
    label_space: LabelSpace
    self_ids: Optional[np.ndarray] = None
    candidates: Optional[CandidateView] = None


@dataclass
class SignalMatrix:
    """One ``(b, C)`` intermediate a ``SignalProvider`` contributes for one query
    chunk, plus how it participates in candidate selection and the *generic*
    per-signal derivations ``FeatureAssembler`` already computes uniformly for
    any matrix by name (raw value, rank, min-max norm, missing flag, margin +
    per-query gap — see ``application/features.py``'s ``_row_rank``/
    ``_row_minmax``/``_row_margin`` helpers).

    The five built-in signals are *not* symmetric (``d_desc_sim`` has no
    missing-flag column; ``d_proto_sim`` has no rank/norm at all), so which
    derivations apply and what each derived column is named is declared here,
    per matrix, rather than inferred from a fixed naming convention.

    - ``node``: the ``FEATURE_DEPS``/candidate-mask name (e.g. ``"dense.desc"``).
    - ``value``: the ``(b, C)`` matrix itself. NaN means "this signal did not
      retrieve this class" (never a true 0) — the CLAUDE.md invariant.
    - ``derive``: subset of ``{"raw", "rank", "norm", "missing", "margin"}`` —
      which generic derivations to compute for this matrix. ``"margin"`` also
      produces the paired per-query top1-top2 gap when ``gap_column`` is set.
    - ``columns``: derivation name -> output column name, for every entry in
      ``derive`` (e.g. ``{"raw": "d_desc_sim", "rank": "rank_d_desc"}``).
    - ``gap_column``: output column name for the per-query top1-top2 gap paired
      with the ``"margin"`` derivation, or ``None`` if this signal has no
      ``q_gap_*`` column (e.g. ``dense.proto``/``bm25.knn`` today).
    - ``top1_idx``/``top1_column``: an ``is_<x>_top1`` column naming the (b,)
      class-index array that defines "this signal's pick" for the query — not
      always ``argmax(value)`` (a kNN-style signal's top1 is its single nearest
      neighbor's own class, not the arg-max of its aggregated per-class sum).
      ``top1_column`` is ``None`` when this matrix contributes no top1 column.
    - ``top1_check_valid``: AND the top1 comparison with ``top1_idx >= 0`` (an
      argmax-derived top1 can be "no valid pick", encoded as -1).
    - ``extra_columns``: additional ``(b, C)`` matrices gathered directly at the
      same ``(rows, cols)`` grid, under their own explicit names, that are not
      one of the generic derivations (e.g. ``d_knn_max``/``d_knn_count`` sit
      beside ``d_knn_sum``'s node without being a rank/norm/margin of it).
    - ``extra_scalars``: additional per-query ``(b,)`` columns broadcast across
      every candidate row of that query (e.g. ``abs_top_dense_sim``).
    - ``topn_positive_only``: whether this matrix's contribution to the
      candidate top-n union drops non-positive values (``bm25.desc`` today).
    """

    node: str
    value: np.ndarray
    derive: FrozenSet[str] = frozenset()
    columns: Dict[str, str] = field(default_factory=dict)
    gap_column: Optional[str] = None
    top1_idx: Optional[np.ndarray] = None
    top1_column: Optional[str] = None
    top1_check_valid: bool = False
    extra_columns: Dict[str, np.ndarray] = field(default_factory=dict)
    extra_scalars: Dict[str, np.ndarray] = field(default_factory=dict)
    topn_positive_only: bool = False

    def column_for(self, derivation: str) -> Optional[str]:
        """The output column name for ``derivation`` on this matrix, or ``None``
        when this matrix does not declare it.

        ``derive`` and ``columns`` answer two halves of one question — "does
        this derivation apply here" and "what is it called" — and a derivation
        is only real when *both* agree. Asking it as one method keeps the
        assembler from re-deriving the conjunction per derivation, and makes a
        matrix that lists a derivation in ``derive`` without naming its column
        (or vice versa) a uniformly ignored no-op rather than a ``KeyError`` in
        one code path and silence in another.
        """
        if derivation not in self.derive:
            return None
        return self.columns.get(derivation)


class SignalProvider(ABC):
    """A pluggable source of retrieval *signals* (T34 phase 2) — the layer below
    ``FeatureProvider`` (T70): a ``FeatureProvider`` appends fusion columns after
    the core schema and never joins candidate selection; a ``SignalProvider``
    contributes one or more ``(b, C)`` matrices that *do* join the top-n
    candidate union, exactly like the five built-in retrieval signals.

    The built-in ``DenseSignalProvider``/``LexicalSignalProvider`` (see
    ``infrastructure/signals.py``) wrap an already-built/loaded
    ``DenseRetriever``/``LexicalRetriever`` — a ``SignalProvider`` is not itself
    a retriever and does not re-implement retrieval; ``TrainingPipeline`` builds
    the underlying retriever per fold (the existing leakage-free discipline) and
    wraps it. A provider with its own learned state (not wrapping a retriever)
    persists that state directly in ``save``/``load``.

    - ``name``: unique prefix identifying this provider (registry key).
    - ``needs_candidates``: run in round *two*, after candidate selection, with
      ``SignalContext.candidates`` populated (see ``CandidateView``). Default
      ``False`` — round one, the only behaviour before T33. A round-two provider
      **must** return an empty ``candidate_features()``: it cannot select what it
      consumes, and the assembler rejects the combination rather than resolving
      it in some order-dependent way. That restriction is also what lets the
      assembler skip a round-two provider entirely when every one of its columns
      is pruned (T87) — with no contribution to the candidate mask, not running
      it cannot change any surviving column's value.
    - ``candidate_features``: which of this provider's ``SignalMatrix.node``
      names join the top-n candidate union for this query chunk. Every node
      returned by ``build`` that is *not* named here still computes its generic
      derivations but never contributes to candidate selection.
    - ``build(ctx)``: compute this provider's ``SignalMatrix`` list for one
      query chunk. Must be vectorized (CLAUDE.md convention) and must emit NaN,
      never a true 0, for "did not retrieve" cells.
    - ``save``/``load``: persist any fitted state through a directory using
      portable formats only (numpy + json + native model formats; no pickle),
      so the model dir ships to an air-gapped host.
    """

    name: str
    needs_candidates: bool = False

    @abstractmethod
    def candidate_features(self) -> Sequence[str]:
        """Node names (``SignalMatrix.node``) that join the top-n candidate
        union. A node this provider computes but omits here still gets its
        generic derivations, just never selects candidates on its own."""

    @abstractmethod
    def column_names(self) -> List[str]:
        """Every feature-column name this provider will contribute, independent
        of any query batch -- the ``SignalMatrix``-level ``SignalProvider``
        analogue of ``FeatureProvider.names()``, needed to compose the full
        schema (for training-column selection and the persisted ``meta.json``
        schema check) before ``build`` ever runs against real data. Must stay
        stable and match ``build``'s actual output columns exactly.

        The two built-in providers' ``column_names()`` reconstruct the relevant
        slice of the core ``FEATURE_NAMES`` schema; the assembler special-cases
        them (by ``name in ("dense", "lexical")``) so the *default* config's
        schema stays exactly ``FEATURE_NAMES`` rather than double-declaring it."""

    @abstractmethod
    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        """Compute this provider's signal matrices for one query chunk."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist fitted state to directory ``path`` (portable formats only)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SignalProvider":
        """Reload a provider persisted by ``save`` — no labels, no network."""


class PairwiseReranker(ABC):
    """Scores ``(query_text, document_text)`` pairs *jointly* (T33).

    The contrast with ``TextEncoder`` is the whole point: an encoder embeds each
    side independently, so each is compressed without knowing what it will be
    compared against. A reranker attends across both texts at once, which is why
    it ranks better and why it costs ~100x more — it cannot precompute anything,
    so its cost is per *pair*, not per text.

    Deliberately minimal, and deliberately not a retriever: this port answers
    "how well do these two texts go together" and nothing else. *Which* pairs to
    score, how the document side is composed, and how scores become feature
    columns are all the calling ``SignalProvider``'s decisions (see
    ``infrastructure/reranker.py``). That split is what lets the same port back a
    stock cross-encoder, a fine-tuned one, and an instructed LLM judge.

    ``score`` returns raw model outputs — logits for a typical cross-encoder,
    unbounded and *not* calibrated probabilities. The fusion model learns the
    scale, so no backend should squash them to look comparable; the isotonic
    calibration stage downstream is what makes scores mean something.
    """

    @abstractmethod
    def score(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """``(n,)`` float32 relevance score per ``(query, document)`` pair.

        Must preserve input order and return exactly one score per pair.
        Batching is the implementation's business; callers hand over whole
        chunks precisely so a backend can batch them."""

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist to ``directory`` using portable formats only (no pickle) —
        the same air-gapped-portability contract ``TextEncoder.save`` carries."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs: Any) -> "PairwiseReranker":
        """Reload a reranker persisted by ``save`` — no network."""


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

    def set_device(self, device: Optional[str]) -> None:
        """Pin inference to a specific device (e.g. "cuda", "cpu"), overriding
        auto-detection. ``None`` restores auto-detection. No-op (the default)
        for backends with no device concept -- e.g. LightGBM's CPU-only wheel."""
        return None

    def predict_contribs(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Optional per-feature contributions toward the *raw* (pre-calibration)
        score, for prediction explanations.

        Returns ``(n, n_features + 1)``: columns ``0..n_features-1`` align to the
        fusion feature columns in order, and the trailing column is the bias/base
        value, so each row *sums to the raw margin* — the additive-feature identity
        the gradient-boosted-tree backends provide. Returns ``None`` (the default)
        when the backend cannot decompose its score additively — e.g. a ranker
        whose isotonic head breaks additivity. Callers must treat ``None`` as "no
        attribution available" and degrade gracefully. The default keeps the port
        additive: existing and custom backends need no change."""
        return None

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "FusionModel": ...


class ConfidenceCalibrator(ABC):
    """Maps raw fusion scores onto calibrated P(correct).

    ``classes`` (optional, ``(n,)`` int, one candidate class index per score) lets
    a class-aware backend (e.g. ``PerClassCalibrator``) fit/apply a per-class
    curve. ``None`` (the default) is the class-blind path every existing
    calibrator implements; a backend that ignores ``classes`` behaves exactly as
    it did before this parameter existed."""

    @abstractmethod
    def fit(
        self, scores: np.ndarray, correct: np.ndarray, *, classes: Optional[np.ndarray] = None
    ) -> None: ...

    @abstractmethod
    def transform(
        self, scores: np.ndarray, *, classes: Optional[np.ndarray] = None
    ) -> np.ndarray: ...

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
    deliberately *not* exposed here — that is a separate, later capability."""

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
    """A pluggable source of extra fusion features.

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
