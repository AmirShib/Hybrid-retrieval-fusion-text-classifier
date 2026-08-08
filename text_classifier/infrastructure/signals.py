"""Built-in ``SignalProvider``s (T34 phase 2).

``DenseSignalProvider``/``LexicalSignalProvider`` wrap an already-built/loaded
``DenseRetriever``/``LexicalRetriever`` (T34 phase 1's registry) and reproduce
exactly the five signal matrices ``FeatureAssembler`` used to compute directly:
``dense.desc``/``dense.proto``/``dense.knn`` and ``bm25.desc``/``bm25.knn``.

These two providers are the byte-for-byte-identical default (``PipelineConfig
.signals == ["dense", "lexical"]``); nothing here changes the numerics that
used to live inline in ``application/features.py``, only where they run.

``_scatter_knn``/``_argmax_or_missing`` live here (not in
``application/features.py``) because a ``SignalProvider`` is an infrastructure
adapter and must not depend on the application layer; ``application/features.py``
imports and re-exports both so existing test imports
(``from text_classifier.application.features import _scatter_knn``) keep working.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..domain import (
    ArrayOps,
    DenseRetriever,
    LexicalRetriever,
    SignalContext,
    SignalMatrix,
    SignalProvider,
)
from .array_ops import NumpyArrayOps


def _topn_mask(
    M: np.ndarray, n: int, positive_only: bool = False, ops: Optional[ArrayOps] = None
) -> np.ndarray:
    """Boolean (b, C) mask of each row's top-n columns. NaN ranks last; -inf
    selections (all-missing) are dropped. Ties may admit slightly more than n.

    Lives here rather than in ``application/features.py`` for the same reason
    ``_scatter_knn`` does (see the module docstring): candidate selection is the
    assembler's own use, but a second-stage ``SignalProvider`` (T33) needs the
    identical top-n rule to pick which candidates are worth its cost, and an
    infrastructure adapter must not import the application layer.

    Fully routed through ``ArrayOps``. It previously reached past the port twice
    -- ``M.astype`` (which torch tensors do not have) and ``np.partition`` on
    what may be a tensor, whose numpy result then failed to compare against the
    tensor it was derived from -- so this kernel, and therefore candidate
    selection itself, raised under ``array_backend="torch"``. ``topk``'s
    nth-best value is the same threshold ``partition``'s pivot gave (both are
    the nth largest in the row), so the numpy backend is bit-identical."""
    ops = ops or NumpyArrayOps()
    C = M.shape[1]
    n = min(n, C)
    Mf = ops.where(ops.isnan(M), -np.inf, ops.asarray(M, dtype=np.float64))
    if positive_only:
        Mf = ops.where(Mf > 0, Mf, -np.inf)
    best, _ = ops.topk(Mf, n, axis=1)
    kth = best[:, n - 1 : n]  # (b, 1): the nth-best value in each row
    return (Mf >= kth) & ops.isfinite(Mf)


def _scatter_knn(
    labels: np.ndarray, scores: np.ndarray, n_classes: int, ops: Optional[ArrayOps] = None
):
    """Aggregate (b, k) neighbor labels/scores into per-class (b, C) sum/max/count.
    Missing entries (label < 0 or NaN score) are ignored; sum/max are NaN where
    count == 0 so that 'not retrieved' stays distinct from a true zero.

    Sum/max are scattered via ``ArrayOps.scatter_add``/``scatter_max`` (T84)
    rather than ``np.add.at``/``np.maximum.at`` -- the latter is numpy's
    unbuffered, slowest scatter. Both process the same (row, col, value)
    triples in the same order at the same float64 precision the ``.at`` calls
    used, so the numpy backend's result is bit-for-bit identical."""
    ops = ops or NumpyArrayOps()
    b = labels.shape[0]
    ksum = ops.zeros((b, n_classes), dtype=np.float64)
    kcnt = ops.zeros((b, n_classes), dtype=np.float64)
    kmax = ops.full((b, n_classes), -np.inf, dtype=np.float64)

    rows = np.repeat(np.arange(b), labels.shape[1])
    L = labels.ravel()
    S = scores.ravel().astype(np.float64)
    valid = (L >= 0) & ~ops.isnan(S)
    r, c, sv = rows[valid], L[valid], S[valid]

    ksum = ops.to_host(ops.scatter_add(ksum, r, c, sv))
    kcnt = ops.to_host(ops.scatter_add(kcnt, r, c, np.ones_like(sv)))
    kmax = ops.to_host(ops.scatter_max(kmax, r, c, sv))

    empty = kcnt == 0
    ksum[empty] = np.nan
    kmax[empty] = np.nan
    return ksum, kmax, kcnt


def _argmax_or_missing(M: np.ndarray, require_positive: bool = False) -> np.ndarray:
    Mf = np.where(np.isnan(M), -np.inf, M)
    a = np.argmax(Mf, axis=1)
    best = Mf[np.arange(M.shape[0]), a]
    invalid = ~np.isfinite(best) | (require_positive & (best <= 0))
    return np.where(invalid, -1, a)


def rewrap_signal_providers(
    providers: Sequence[SignalProvider],
    dense: DenseRetriever,
    lexical: LexicalRetriever,
) -> List[SignalProvider]:
    """Rebuild ``providers`` onto a new ``dense``/``lexical`` pair, in order.

    A ``DenseSignalProvider``/``LexicalSignalProvider`` wraps a *specific*
    retriever instance by reference. Any operation that replaces ``dense``/
    ``lexical`` with an extended copy (``with_added_classes``, ``update``'s
    added-examples path) must rewrap the corresponding built-in providers onto
    the new instances, or a later ``assemble()`` call would keep scoring
    against the stale, pre-extension retriever and silently miss the new
    classes/examples. Any other (non-wrapping, e.g. a stateless custom) provider
    is returned unchanged -- it holds no dense/lexical reference to go stale."""
    out: List[SignalProvider] = []
    for provider in providers:
        if isinstance(provider, DenseSignalProvider):
            out.append(DenseSignalProvider(dense, provider._ops))
        elif isinstance(provider, LexicalSignalProvider):
            out.append(LexicalSignalProvider(lexical, provider._ops))
        else:
            out.append(provider)
    return out


class DenseSignalProvider(SignalProvider):
    """Wraps a fitted ``DenseRetriever``, producing the ``dense.desc``/
    ``dense.proto``/``dense.knn`` nodes exactly as ``FeatureAssembler`` computed
    them before T34 phase 2 — same values, same NaN-for-missing semantics, same
    top1/rank/margin/gap columns."""

    name = "dense"

    def __init__(self, retriever: DenseRetriever, ops: Optional[ArrayOps] = None):
        self._retriever = retriever
        self._ops = ops or NumpyArrayOps()

    def candidate_features(self) -> Sequence[str]:
        return ("dense.desc", "dense.proto", "dense.knn")

    def column_names(self) -> List[str]:
        # Exactly the dense-signal slice of FEATURE_NAMES -- the assembler
        # never actually consults this for the default config (it special-cases
        # name == "dense"), but it must match `build`'s real output columns.
        return [
            "d_desc_sim",
            "d_proto_sim",
            "d_knn_sum",
            "d_knn_max",
            "d_knn_count",
            "abs_top_dense_sim",
            "is_d_desc_top1",
            "is_d_proto_top1",
            "is_d_knn_top1",
            "d_knn_missing",
            "rank_d_desc",
            "rank_d_knn",
            "norm_d_desc",
            "margin_d_desc",
            "margin_d_proto",
            "margin_d_knn",
            "q_gap_d_desc",
            "q_gap_d_knn",
        ]

    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        dense = self._retriever
        q_emb = ctx.q_emb
        desc_d = np.asarray(dense.description_similarity(q_emb), dtype=np.float64)
        if ctx.self_ids is None:
            proto = np.asarray(dense.prototype_similarity(q_emb), dtype=np.float64)
            dn_lab, dn_sim = dense.knn_example_labels(q_emb, ctx.k)
        else:
            proto = np.asarray(
                dense.loo_prototype_similarity(q_emb, ctx.self_ids), dtype=np.float64
            )
            dn_lab, dn_sim = dense.knn_example_labels(q_emb, ctx.k, ctx.self_ids)
        d_sum, d_max, d_cnt = _scatter_knn(dn_lab, dn_sim, ctx.n_classes, self._ops)

        a_desc = np.argmax(np.where(np.isnan(desc_d), -np.inf, desc_d), axis=1)
        a_proto = _argmax_or_missing(proto)
        a_dknn = dn_lab[:, 0]
        abs_top_dense = dn_sim[:, 0].astype(np.float64)

        return [
            SignalMatrix(
                node="dense.desc",
                value=desc_d,
                derive=frozenset({"raw", "rank", "norm", "margin"}),
                columns={
                    "raw": "d_desc_sim",
                    "rank": "rank_d_desc",
                    "norm": "norm_d_desc",
                    "margin": "margin_d_desc",
                },
                gap_column="q_gap_d_desc",
                top1_idx=a_desc,
                top1_column="is_d_desc_top1",
                top1_check_valid=False,
            ),
            SignalMatrix(
                node="dense.proto",
                value=proto,
                derive=frozenset({"raw", "margin"}),
                columns={"raw": "d_proto_sim", "margin": "margin_d_proto"},
                gap_column=None,
                top1_idx=a_proto,
                top1_column="is_d_proto_top1",
                top1_check_valid=True,
            ),
            SignalMatrix(
                node="dense.knn",
                value=d_sum,
                derive=frozenset({"raw", "missing", "rank", "margin"}),
                columns={
                    "raw": "d_knn_sum",
                    "missing": "d_knn_missing",
                    "rank": "rank_d_knn",
                    "margin": "margin_d_knn",
                },
                gap_column="q_gap_d_knn",
                top1_idx=a_dknn,
                top1_column="is_d_knn_top1",
                top1_check_valid=False,
                extra_columns={"d_knn_max": d_max, "d_knn_count": d_cnt},
                extra_scalars={"abs_top_dense_sim": abs_top_dense},
            ),
        ]

    def save(self, path: str) -> None:
        # State already persists as `dense.npz` via the wrapped DenseRetriever's
        # own registered spec (T34 phase 1) — nothing extra to write.
        return None

    @classmethod
    def load(cls, path: str) -> "DenseSignalProvider":
        raise NotImplementedError(
            "DenseSignalProvider wraps an already-built/loaded DenseRetriever; "
            "construct it directly (DenseSignalProvider(dense_retriever)) rather "
            "than via .load(path) -- see infrastructure.registry.build_signal_providers."
        )


class LexicalSignalProvider(SignalProvider):
    """Wraps a fitted ``LexicalRetriever``, producing the ``bm25.desc``/
    ``bm25.knn`` nodes exactly as ``FeatureAssembler`` computed them before T34
    phase 2."""

    name = "lexical"

    def __init__(self, retriever: LexicalRetriever, ops: Optional[ArrayOps] = None):
        self._retriever = retriever
        self._ops = ops or NumpyArrayOps()

    def candidate_features(self) -> Sequence[str]:
        return ("bm25.desc", "bm25.knn")

    def column_names(self) -> List[str]:
        # Exactly the lexical-signal slice of FEATURE_NAMES -- see
        # DenseSignalProvider.column_names()'s docstring note.
        return [
            "b_desc_sim",
            "b_knn_sum",
            "b_knn_max",
            "b_knn_count",
            "abs_top_bm25",
            "is_b_desc_top1",
            "is_b_knn_top1",
            "b_desc_missing",
            "b_knn_missing",
            "rank_b_desc",
            "rank_b_knn",
            "norm_b_desc",
            "margin_b_desc",
            "margin_b_knn",
            "q_gap_b_desc",
        ]

    def build(self, ctx: SignalContext) -> List[SignalMatrix]:
        lexical = self._retriever
        texts = ctx.texts
        if ctx.self_ids is None:
            bn_lab, bn_sco = lexical.knn_example_labels(texts, ctx.k)
        else:
            bn_lab, bn_sco = lexical.knn_example_labels(texts, ctx.k, ctx.self_ids)
        bdesc_raw = np.asarray(lexical.description_score(texts), dtype=np.float64)
        bdesc = np.where(bdesc_raw > 0, bdesc_raw, np.nan)  # 0 overlap == missing
        b_sum, b_max, b_cnt = _scatter_knn(bn_lab, bn_sco, ctx.n_classes, self._ops)

        a_bdesc = _argmax_or_missing(bdesc, require_positive=True)
        a_bknn = bn_lab[:, 0]
        with np.errstate(invalid="ignore"):
            abs_top_bm25 = np.nanmax(np.where(np.isnan(bn_sco), -np.inf, bn_sco), axis=1)
        abs_top_bm25 = np.where(np.isfinite(abs_top_bm25), abs_top_bm25, 0.0)

        return [
            SignalMatrix(
                node="bm25.desc",
                value=bdesc,
                derive=frozenset({"raw", "missing", "rank", "norm", "margin"}),
                columns={
                    "raw": "b_desc_sim",
                    "missing": "b_desc_missing",
                    "rank": "rank_b_desc",
                    "norm": "norm_b_desc",
                    "margin": "margin_b_desc",
                },
                gap_column="q_gap_b_desc",
                top1_idx=a_bdesc,
                top1_column="is_b_desc_top1",
                top1_check_valid=True,
                topn_positive_only=True,
            ),
            SignalMatrix(
                node="bm25.knn",
                value=b_sum,
                derive=frozenset({"raw", "missing", "rank", "margin"}),
                columns={
                    "raw": "b_knn_sum",
                    "missing": "b_knn_missing",
                    "rank": "rank_b_knn",
                    "margin": "margin_b_knn",
                },
                gap_column=None,
                top1_idx=a_bknn,
                top1_column="is_b_knn_top1",
                top1_check_valid=True,
                extra_columns={"b_knn_max": b_max, "b_knn_count": b_cnt},
                extra_scalars={"abs_top_bm25": abs_top_bm25},
            ),
        ]

    def save(self, path: str) -> None:
        # State already persists as `lexical.npz`/`lexical.json` via the wrapped
        # LexicalRetriever's own registered spec (T34 phase 1).
        return None

    @classmethod
    def load(cls, path: str) -> "LexicalSignalProvider":
        raise NotImplementedError(
            "LexicalSignalProvider wraps an already-built/loaded LexicalRetriever; "
            "construct it directly (LexicalSignalProvider(lexical_retriever)) rather "
            "than via .load(path) -- see infrastructure.registry.build_signal_providers."
        )
