"""Feature assembly (application service).

Turns the five retrieval signals into one feature row per (item, candidate class).
Everything is vectorized: each signal becomes a (batch, n_classes) matrix, the
candidate set is a boolean mask, and feature columns are gathered with fancy
indexing. Queries are processed in chunks to bound peak memory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import warnings

import numpy as np
import pandas as pd

from ..domain import (
    ArrayOps,
    CandidatePolicy,
    DenseRetriever,
    FEATURE_NAMES,
    FeatureContext,
    FeatureProvider,
    LabelSpace,
    LexicalRetriever,
    SignalContext,
    SignalMatrix,
    SignalProvider,
    composed_feature_names,
    feature_closure,
)
from ..infrastructure.array_ops import NumpyArrayOps
from ..infrastructure.signals import (
    DenseSignalProvider,
    LexicalSignalProvider,
    _argmax_or_missing,  # noqa: F401 -- re-exported, see __all__ note below
    _scatter_knn,  # noqa: F401 -- re-exported, see __all__ note below
)

# Re-exported for callers that reach for it via the assembly module; the
# canonical definition lives in the domain schema (``domain/services.py``).
# `_scatter_knn`/`_argmax_or_missing` are re-exported for backward compatibility
# — their implementation lives in `infrastructure/signals.py` (T34 phase 2), an
# infrastructure adapter, so the two built-in `SignalProvider`s can use them
# without `infrastructure` importing `application`.
__all__ = ["FeatureAssembler", "composed_feature_names"]


def _effective_names(
    providers: Sequence[FeatureProvider],
    requested: Optional[Sequence[str]],
    signal_providers: Sequence[SignalProvider] = (),
) -> list:
    """The columns this call will actually produce: the full composed schema
    when ``requested`` is ``None`` (every existing caller, byte-for-byte
    unchanged), else the composed schema narrowed to ``feature_closure(requested)``
    plus any provider whose own names overlap ``requested``."""
    names = composed_feature_names(providers, signal_providers)
    if requested is None:
        return names
    needed = feature_closure(requested)
    req = set(requested)
    provider_names = {n for p in providers for n in p.names()}
    core_needed = {n for n in names if n not in provider_names and n in needed}
    kept_providers = {n for n in provider_names if n in req}
    return [n for n in names if n in core_needed or n in kept_providers]


def _topn_mask(
    M: np.ndarray, n: int, positive_only: bool = False, ops: Optional[ArrayOps] = None
) -> np.ndarray:
    """Boolean (b, C) mask of each row's top-n columns. NaN ranks last; -inf
    selections (all-missing) are dropped. Ties may admit slightly more than n."""
    ops = ops or NumpyArrayOps()
    b, C = M.shape
    n = min(n, C)
    Mf = ops.where(ops.isnan(M), -np.inf, M.astype(np.float64))
    if positive_only:
        Mf = ops.where(Mf > 0, Mf, -np.inf)
    kth = np.partition(Mf, C - n, axis=1)[:, C - n][:, None]
    return (Mf >= kth) & ops.isfinite(Mf)


def _row_rank(M: np.ndarray, cand_mask: np.ndarray, ops: Optional[ArrayOps] = None) -> np.ndarray:
    """Dense descending rank (1 = best) within each row's candidate set."""
    ops = ops or NumpyArrayOps()
    Mf = ops.where(cand_mask, M, np.nan)
    Mf = ops.where(ops.isnan(Mf), -np.inf, Mf)
    order = ops.argsort(-Mf, axis=1)
    ranks = np.empty(M.shape, dtype=np.float64)
    rows = np.arange(M.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, M.shape[1] + 1)[None, :]
    return ranks


def _row_minmax(M: np.ndarray, cand_mask: np.ndarray, ops: Optional[ArrayOps] = None) -> np.ndarray:
    """Per-row min-max of M over candidates (NaN preserved for all-missing rows)."""
    ops = ops or NumpyArrayOps()
    Mc = ops.where(cand_mask, M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN (intended)
        lo = ops.nanmin(Mc, axis=1)
        hi = ops.nanmax(Mc, axis=1)
    rng = ops.where(hi > lo, hi - lo, 1.0)
    return (M - lo[:, None]) / rng[:, None]


def _row_margin(
    M: np.ndarray, cand_mask: np.ndarray, ops: Optional[ArrayOps] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Competition features for one signal: per-candidate margins and the per-row
    top1-top2 gap.

    Returns ``(margin (b, C), gap (b,))``.

    ``margin[r, c]`` is ``M[r, c]`` minus the best *other* candidate's value in
    row ``r`` — so the row's leader gets ``top1 - top2`` (positive, its winning
    margin) and every other candidate gets ``value - top1`` (non-positive, its
    deficit). ``gap[r]`` is that same ``top1 - top2``, carried as a per-query
    column so trailing candidates also see how contested the lead is.

    Only candidates (``cand_mask``) with a non-NaN value compete; NaN is "this
    signal did not retrieve this class" and must never be read as a low score.
    Both outputs are NaN where no margin is *defined*:

    - ``margin`` is NaN wherever ``M`` is NaN (the signal did not fire here), and
      NaN for the leader of a row with only one scored candidate — there is no
      competitor to measure against, which is a different statement from a
      margin of 0.0 (a tie).
    - ``gap`` is NaN for a row with fewer than two scored candidates.

    Ties are handled the obvious way: two candidates tied at the top both get
    margin 0.0, and the row's gap is 0.0.
    """
    ops = ops or NumpyArrayOps()
    b, C = M.shape
    Mc = ops.where(cand_mask & ~ops.isnan(M), M, -np.inf)
    if C == 1:
        # A single class: it is its own row's leader and has no competitor ever.
        top1 = Mc[:, 0]
        top2 = np.full(b, -np.inf)
        best_other = top2[:, None]
    else:
        # Top-2 by partition (O(C)) rather than a full sort — only the two best
        # values in each row matter here.
        part = ops.argpartition(-Mc, 1, axis=1)[:, :2]
        rows = np.arange(b)[:, None]
        vals = Mc[rows, part]
        swap = vals[:, 0] < vals[:, 1]
        leader = ops.where(swap, part[:, 1], part[:, 0])
        top1 = ops.where(swap, vals[:, 1], vals[:, 0])
        top2 = ops.where(swap, vals[:, 0], vals[:, 1])
        # The leader competes against #2; everyone else competes against #1.
        is_leader = np.arange(C)[None, :] == leader[:, None]
        best_other = ops.where(is_leader, top2[:, None], top1[:, None])

    with np.errstate(invalid="ignore"):  # -inf - -inf on all-missing rows -> NaN
        margin = ops.where(ops.isfinite(best_other) & ~ops.isnan(M), M - best_other, np.nan)
        gap = ops.where(ops.isfinite(top1) & ops.isfinite(top2), top1 - top2, np.nan)
    return margin, gap


class FeatureAssembler:
    """Builds the (item, candidate) feature table for a batch of queries."""

    def __init__(
        self,
        label_space: LabelSpace,
        candidate_policy: CandidatePolicy,
        array_ops: Optional[ArrayOps] = None,
    ):
        self._space = label_space
        self._policy = candidate_policy
        self._ops = array_ops or NumpyArrayOps()

    def assemble(
        self,
        query_texts: Sequence[str],
        query_emb: np.ndarray,
        dense: DenseRetriever,
        lexical: Optional[LexicalRetriever],
        k_neighbors: int,
        query_ids: Union[Sequence[Any], np.ndarray],
        query_labels: Optional[np.ndarray] = None,
        chunk: int = 4096,
        providers: Sequence[FeatureProvider] = (),
        self_ids: Optional[np.ndarray] = None,
        requested: Optional[Sequence[str]] = None,
        signal_providers: Optional[Sequence[SignalProvider]] = None,
    ) -> pd.DataFrame:
        """Assemble the (item, candidate) feature table.

        ``providers`` contribute extra columns appended after the core ~28,
        in provider order; with none configured the output is exactly the core
        schema. Each provider must already be fitted (the caller fits per fold to
        stay leakage-free).

        ``self_ids`` (leave-one-out mode, ``n_folds=1``) is the example-pool index
        of each query when the queries *are* the pool being retrieved against: the
        dense/BM25 kNN drop that self-neighbor and the dense prototype leaves the
        query's own vector out of its own class. ``None`` (the default) is ordinary
        featurization — every existing caller — and is byte-for-byte unchanged.

        ``requested`` (T87) is the caller's demand: the column names it actually
        needs (typically ``fusion_feature_names(...)`` to fit/score the model, or
        ``composed_feature_names(...)`` for a diagnostic that reads core columns
        by name). ``None`` (the default) means "everything" and reproduces the
        original, unconditional-computation behaviour exactly. A non-``None``
        request only *narrows* what appears in the output — the candidate mask
        itself is never pruned (every signal that feeds it still runs), and every
        surviving column's value is identical to the unpruned computation.

        ``signal_providers`` (T34 phase 2) is the ordered list of ``SignalProvider``s
        that compute the five (now pluggable) retrieval signals. ``None`` (the
        default, every existing caller) builds the two built-in providers wrapping
        ``dense``/``lexical`` — byte-for-byte the previous, hardcoded behaviour.
        A caller that configures extra signals passes its own list here; ``dense``/
        ``lexical`` are still required for ``class_freq`` and remain the objects the
        default providers wrap when ``signal_providers`` is left unset."""
        frames = []
        ids = np.asarray(query_ids)
        sids = None if self_ids is None else np.asarray(self_ids)
        for s in range(0, len(query_texts), chunk):
            sl = slice(s, s + chunk)
            frames.append(
                self._assemble_chunk(
                    list(query_texts[sl]),
                    query_emb[sl],
                    dense,
                    lexical,
                    k_neighbors,
                    ids[sl],
                    None if query_labels is None else np.asarray(query_labels)[sl],
                    providers,
                    None if sids is None else sids[sl],
                    requested,
                    signal_providers,
                )
            )
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=_effective_names(providers, requested, signal_providers or ()))

    def _assemble_chunk(
        self,
        texts,
        q_emb,
        dense,
        lexical,
        k,
        ids,
        labels,
        providers=(),
        self_ids=None,
        requested: Optional[Sequence[str]] = None,
        signal_providers: Optional[Sequence[SignalProvider]] = None,
    ) -> pd.DataFrame:
        C = self._space.size
        n = self._policy.top_n_per_signal
        class_freq = dense.class_freq

        # T87: which columns this call actually needs. `None` means "everything"
        # (every pre-T87 caller) and is deliberately not narrowed to a concrete
        # set — `_want` below then always answers True, reproducing the original
        # unconditional computation exactly, including its exact column order.
        needed: Optional[Set[str]] = None if requested is None else feature_closure(requested)

        def _want(name: Optional[str]) -> bool:
            return name is not None and (needed is None or name in needed)

        if signal_providers is None:
            # The byte-for-byte-identical default (T34 phase 2): the two
            # built-in providers wrapping `dense`/`lexical`, reproducing exactly
            # the five signal matrices `_assemble_chunk` used to compute inline.
            signal_providers = (
                DenseSignalProvider(dense, self._ops),
                LexicalSignalProvider(lexical, self._ops),
            )

        # ---- run every signal provider unconditionally: the candidate mask is
        # an unconditional dependency of the whole frame (T87's "candidates is
        # never pruned" rule), so every signal that feeds it must run regardless
        # of `requested`. Only each matrix's *generic derivations* below are
        # gated by `_want`. ----
        ctx = SignalContext(texts=texts, q_emb=q_emb, k=k, n_classes=C, self_ids=self_ids)
        sig_matrices: Dict[str, SignalMatrix] = {}
        candidate_values = []
        for provider in signal_providers:
            for sm in provider.build(ctx):
                if sm.node in sig_matrices:
                    raise ValueError(
                        f"duplicate signal node {sm.node!r}: both an earlier provider and "
                        f"{provider.name!r} produced it. Signal-provider node names must "
                        "be unique across every active provider."
                    )
                sig_matrices[sm.node] = sm
                if sm.node in provider.candidate_features():
                    candidate_values.append((sm.value, sm.topn_positive_only))

        # ---- candidate set = union of each declared candidate matrix's top-n ----
        mask = np.zeros((len(texts), C), dtype=bool)
        for value, positive_only in candidate_values:
            mask |= _topn_mask(value, n, positive_only=positive_only, ops=self._ops)
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            empty_cols = _effective_names(providers, requested, signal_providers) + (
                ["is_true"] if labels is not None else []
            )
            return pd.DataFrame(columns=empty_cols)

        # ---- gather one value per (row, col) ----
        def g(M):  # gather helper
            return self._ops.gather(M, rows, cols)

        data: Dict[str, np.ndarray] = {}

        # ---- generic per-matrix derivations, uniform across every signal by
        # name (raw value, rank, min-max norm, missing flag, margin + q_gap,
        # is-top1) — driven by each SignalMatrix's own declaration, so the
        # asymmetric built-in schema (e.g. `d_desc_sim` has no missing flag,
        # `d_proto_sim` has no rank/norm at all) is reproduced exactly. ----
        for sm in sig_matrices.values():
            M = sm.value
            raw_col = sm.columns.get("raw") if "raw" in sm.derive else None
            if raw_col is not None and _want(raw_col):
                data[raw_col] = g(M)
            missing_col = sm.columns.get("missing") if "missing" in sm.derive else None
            if missing_col is not None and _want(missing_col):
                data[missing_col] = np.isnan(g(M)).astype(np.float64)
            rank_col = sm.columns.get("rank") if "rank" in sm.derive else None
            if rank_col is not None and _want(rank_col):
                data[rank_col] = g(_row_rank(M, mask, self._ops))
            norm_col = sm.columns.get("norm") if "norm" in sm.derive else None
            if norm_col is not None and _want(norm_col):
                data[norm_col] = g(_row_minmax(M, mask, self._ops))
            margin_col = sm.columns.get("margin") if "margin" in sm.derive else None
            want_margin = margin_col is not None and _want(margin_col)
            want_gap = sm.gap_column is not None and _want(sm.gap_column)
            if margin_col is not None and (want_margin or want_gap):
                mg, gap = _row_margin(M, mask, self._ops)
                if want_margin:
                    data[margin_col] = g(mg)
                if want_gap:
                    data[sm.gap_column] = gap[rows]
            if sm.top1_column is not None and _want(sm.top1_column):
                idx = sm.top1_idx
                hit = cols == idx[rows]
                if sm.top1_check_valid:
                    hit = hit & (idx[rows] >= 0)
                data[sm.top1_column] = hit.astype(np.float64)
            for name, M2 in sm.extra_columns.items():
                if _want(name):
                    data[name] = g(M2)
            for name, arr in sm.extra_scalars.items():
                if _want(name):
                    data[name] = arr[rows]

        # ---- cross-signal features: owned by the assembler, not by any one
        # signal provider (T34 phase 2 explicitly scopes agreement/gap
        # participation to the default two providers, looked up here by node
        # name — a third-party provider does not automatically join them). ----
        if _want("desc_proto_gap"):
            dd = sig_matrices.get("dense.desc")
            dp = sig_matrices.get("dense.proto")
            if dd is not None and dp is not None:
                data["desc_proto_gap"] = g(dd.value) - g(dp.value)
        if _want("class_log_freq"):
            data["class_log_freq"] = self._ops.log1p(class_freq[cols].astype(np.float64))
        if _want("n_signal_agreement"):
            agreement_nodes = ("dense.desc", "dense.proto", "bm25.desc", "dense.knn", "bm25.knn")
            idxs = [sig_matrices[node].top1_idx for node in agreement_nodes if node in sig_matrices]
            if len(idxs) == len(agreement_nodes):
                argstack = np.stack(idxs, axis=1)
                n_agree = np.fromiter(
                    (len(agreement_nodes) - len({v for v in r if v >= 0}) for r in argstack),
                    dtype=np.float64,
                    count=argstack.shape[0],
                )
                data["n_signal_agreement"] = n_agree[rows]

        # Any signal provider beyond the two built-ins ("dense"/"lexical") may
        # contribute columns outside FEATURE_NAMES; append them in provider
        # order (each provider's own `column_names()` order), matching
        # `composed_feature_names`'s ordering exactly -- the default config
        # contributes none here, so `core_cols` is unchanged from before T34
        # phase 2 in that case.
        extra_signal_cols = [
            name
            for provider in signal_providers
            if provider.name not in ("dense", "lexical")
            for name in provider.column_names()
            if name in data
        ]
        core_cols = [c for c in FEATURE_NAMES if c in data] + extra_signal_cols
        df = pd.DataFrame({col: np.asarray(data[col], dtype=np.float32) for col in core_cols})
        # Custom providers append their columns after the core ~28. Each
        # gathers over the same (rows, cols) grid; a provider that "did not fire"
        # for a candidate emits NaN, which XGBoost consumes as missing. A
        # provider whose columns are all pruned by ``needed`` is not called at
        # all (T87) — the whole point for a provider that calls out to an
        # external service or a reranker.
        for col, values in self._provider_columns(
            providers, texts, q_emb, rows, cols, needed
        ).items():
            df[col] = values
        df["item_id"] = ids[rows]
        df["candidate"] = cols.astype(np.int64)
        if labels is not None:
            df["is_true"] = (cols == labels[rows]).astype(np.int64)
        return df

    def _provider_columns(self, providers, texts, q_emb, rows, cols, needed=None) -> dict:
        """Run each provider over the candidate grid and collect its columns as
        float32 arrays, validating the contract (declared names, one value per
        candidate row). Returns an insertion-ordered ``{name: (n_candidates,)}``.

        ``needed`` is the T87 demand set (``None`` = everything, the pre-T87
        behaviour): a provider whose declared ``names()`` share nothing with it
        has ``compute`` skipped entirely — the demand-driven half of the
        provider contract, so a provider that calls an external service pays
        nothing when every one of its columns is dropped."""
        if not providers:
            return {}
        ctx = FeatureContext(
            query_texts=texts,
            query_emb=q_emb,
            rows=rows,
            cols=cols,
            label_space=self._space,
        )
        n = rows.shape[0]
        out: dict = {}
        for provider in providers:
            declared = provider.names()
            if needed is not None and not any(name in needed for name in declared):
                continue
            produced = provider.compute(ctx)
            missing = [nm for nm in declared if nm not in produced]
            if missing:
                raise ValueError(
                    f"{type(provider).__name__}.compute did not return column(s) {missing} "
                    f"that {type(provider).__name__}.names() declares"
                )
            for name in declared:
                if needed is not None and name not in needed:
                    continue
                arr = np.asarray(produced[name])
                if arr.shape != (n,):
                    raise ValueError(
                        f"{type(provider).__name__} column {name!r} has shape {arr.shape}; "
                        f"expected one value per candidate row, i.e. ({n},)"
                    )
                out[name] = arr.astype(np.float32)
        return out
