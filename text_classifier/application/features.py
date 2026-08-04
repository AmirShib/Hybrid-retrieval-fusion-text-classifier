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
    CandidatePolicy,
    DenseRetriever,
    FEATURE_NAMES,
    FeatureContext,
    FeatureProvider,
    LabelSpace,
    LexicalRetriever,
    composed_feature_names,
    feature_closure,
)

# Re-exported for callers that reach for it via the assembly module; the
# canonical definition lives in the domain schema (``domain/services.py``).
__all__ = ["FeatureAssembler", "composed_feature_names"]


def _effective_names(
    providers: Sequence[FeatureProvider], requested: Optional[Sequence[str]]
) -> list:
    """The columns this call will actually produce: the full composed schema
    when ``requested`` is ``None`` (every existing caller, byte-for-byte
    unchanged), else the composed schema narrowed to ``feature_closure(requested)``
    plus any provider whose own names overlap ``requested``."""
    names = composed_feature_names(providers)
    if requested is None:
        return names
    needed = feature_closure(requested)
    req = set(requested)
    provider_names = {n for p in providers for n in p.names()}
    core_needed = {n for n in FEATURE_NAMES if n in needed}
    kept_providers = {n for n in provider_names if n in req}
    return [n for n in names if n in core_needed or n in kept_providers]


def _scatter_knn(labels: np.ndarray, scores: np.ndarray, n_classes: int):
    """Aggregate (b, k) neighbor labels/scores into per-class (b, C) sum/max/count.
    Missing entries (label < 0 or NaN score) are ignored; sum/max are NaN where
    count == 0 so that 'not retrieved' stays distinct from a true zero."""
    b = labels.shape[0]
    ksum = np.zeros((b, n_classes), dtype=np.float64)
    kcnt = np.zeros((b, n_classes), dtype=np.float64)
    kmax = np.full((b, n_classes), -np.inf, dtype=np.float64)

    rows = np.repeat(np.arange(b), labels.shape[1])
    L = labels.ravel()
    S = scores.ravel().astype(np.float64)
    valid = (L >= 0) & ~np.isnan(S)
    r, c, sv = rows[valid], L[valid], S[valid]

    np.add.at(ksum, (r, c), sv)
    np.add.at(kcnt, (r, c), 1.0)
    np.maximum.at(kmax, (r, c), sv)

    empty = kcnt == 0
    ksum[empty] = np.nan
    kmax[empty] = np.nan
    return ksum, kmax, kcnt


def _topn_mask(M: np.ndarray, n: int, positive_only: bool = False) -> np.ndarray:
    """Boolean (b, C) mask of each row's top-n columns. NaN ranks last; -inf
    selections (all-missing) are dropped. Ties may admit slightly more than n."""
    b, C = M.shape
    n = min(n, C)
    Mf = np.where(np.isnan(M), -np.inf, M.astype(np.float64))
    if positive_only:
        Mf = np.where(Mf > 0, Mf, -np.inf)
    kth = np.partition(Mf, C - n, axis=1)[:, C - n][:, None]
    return (Mf >= kth) & np.isfinite(Mf)


def _row_rank(M: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    """Dense descending rank (1 = best) within each row's candidate set."""
    Mf = np.where(cand_mask, M, np.nan)
    Mf = np.where(np.isnan(Mf), -np.inf, Mf)
    order = np.argsort(-Mf, axis=1)
    ranks = np.empty(M.shape, dtype=np.float64)
    rows = np.arange(M.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, M.shape[1] + 1)[None, :]
    return ranks


def _row_minmax(M: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    """Per-row min-max of M over candidates (NaN preserved for all-missing rows)."""
    Mc = np.where(cand_mask, M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN (intended)
        lo = np.nanmin(Mc, axis=1)
        hi = np.nanmax(Mc, axis=1)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (M - lo[:, None]) / rng[:, None]


def _row_margin(M: np.ndarray, cand_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    b, C = M.shape
    Mc = np.where(cand_mask & ~np.isnan(M), M, -np.inf)
    if C == 1:
        # A single class: it is its own row's leader and has no competitor ever.
        top1 = Mc[:, 0]
        top2 = np.full(b, -np.inf)
        best_other = top2[:, None]
    else:
        # Top-2 by partition (O(C)) rather than a full sort — only the two best
        # values in each row matter here.
        part = np.argpartition(-Mc, 1, axis=1)[:, :2]
        rows = np.arange(b)[:, None]
        vals = Mc[rows, part]
        swap = vals[:, 0] < vals[:, 1]
        leader = np.where(swap, part[:, 1], part[:, 0])
        top1 = np.where(swap, vals[:, 1], vals[:, 0])
        top2 = np.where(swap, vals[:, 0], vals[:, 1])
        # The leader competes against #2; everyone else competes against #1.
        is_leader = np.arange(C)[None, :] == leader[:, None]
        best_other = np.where(is_leader, top2[:, None], top1[:, None])

    with np.errstate(invalid="ignore"):  # -inf - -inf on all-missing rows -> NaN
        margin = np.where(np.isfinite(best_other) & ~np.isnan(M), M - best_other, np.nan)
        gap = np.where(np.isfinite(top1) & np.isfinite(top2), top1 - top2, np.nan)
    return margin, gap


def _argmax_or_missing(M: np.ndarray, require_positive: bool = False) -> np.ndarray:
    Mf = np.where(np.isnan(M), -np.inf, M)
    a = np.argmax(Mf, axis=1)
    best = Mf[np.arange(M.shape[0]), a]
    invalid = ~np.isfinite(best) | (require_positive & (best <= 0))
    return np.where(invalid, -1, a)


class FeatureAssembler:
    """Builds the (item, candidate) feature table for a batch of queries."""

    def __init__(self, label_space: LabelSpace, candidate_policy: CandidatePolicy):
        self._space = label_space
        self._policy = candidate_policy

    def assemble(
        self,
        query_texts: Sequence[str],
        query_emb: np.ndarray,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        k_neighbors: int,
        query_ids: Union[Sequence[Any], np.ndarray],
        query_labels: Optional[np.ndarray] = None,
        chunk: int = 4096,
        providers: Sequence[FeatureProvider] = (),
        self_ids: Optional[np.ndarray] = None,
        requested: Optional[Sequence[str]] = None,
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
        surviving column's value is identical to the unpruned computation."""
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
                )
            )
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=_effective_names(providers, requested))

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
    ) -> pd.DataFrame:
        C = self._space.size
        n = self._policy.top_n_per_signal
        class_freq = dense.class_freq

        # T87: which columns this call actually needs. `None` means "everything"
        # (every pre-T87 caller) and is deliberately not narrowed to a concrete
        # set — `_want` below then always answers True, reproducing the original
        # unconditional computation exactly, including its exact column order.
        needed: Optional[Set[str]] = None if requested is None else feature_closure(requested)

        def _want(name: str) -> bool:
            return needed is None or name in needed

        # ---- signals as (b, C) matrices ----
        # In leave-one-out mode (self_ids given) each query is itself in the pool,
        # so its own kNN self-match and its own contribution to its class prototype
        # are masked out — the same leakage-free discipline as an out-of-fold index.
        desc_d = np.asarray(dense.description_similarity(q_emb), dtype=np.float64)
        if self_ids is None:
            # Ordinary path — call the two-argument kNN form so retriever doubles
            # that predate the leave-one-out param keep working unchanged.
            proto = np.asarray(dense.prototype_similarity(q_emb), dtype=np.float64)
            dn_lab, dn_sim = dense.knn_example_labels(q_emb, k)
            bn_lab, bn_sco = lexical.knn_example_labels(texts, k)
        else:
            proto = np.asarray(dense.loo_prototype_similarity(q_emb, self_ids), dtype=np.float64)
            dn_lab, dn_sim = dense.knn_example_labels(q_emb, k, self_ids)
            bn_lab, bn_sco = lexical.knn_example_labels(texts, k, self_ids)
        d_sum, d_max, d_cnt = _scatter_knn(dn_lab, dn_sim, C)

        bdesc_raw = np.asarray(lexical.description_score(texts), dtype=np.float64)
        bdesc = np.where(bdesc_raw > 0, bdesc_raw, np.nan)  # 0 overlap == missing
        b_sum, b_max, b_cnt = _scatter_knn(bn_lab, bn_sco, C)

        # ---- candidate set = union of each signal's top-n ----
        mask = (
            _topn_mask(desc_d, n)
            | _topn_mask(proto, n)
            | _topn_mask(bdesc, n, positive_only=True)
            | _topn_mask(d_sum, n)
            | _topn_mask(b_sum, n)
        )
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            empty_cols = _effective_names(providers, requested) + (
                ["is_true"] if labels is not None else []
            )
            return pd.DataFrame(columns=empty_cols)

        # ---- per-query scalars ----
        a_desc = np.argmax(np.where(np.isnan(desc_d), -np.inf, desc_d), axis=1)
        a_proto = _argmax_or_missing(proto)
        a_bdesc = _argmax_or_missing(bdesc, require_positive=True)
        a_dknn = dn_lab[:, 0]
        a_bknn = bn_lab[:, 0]
        abs_top_dense = dn_sim[:, 0].astype(np.float64)
        with np.errstate(invalid="ignore"):
            abs_top_bm25 = np.nanmax(np.where(np.isnan(bn_sco), -np.inf, bn_sco), axis=1)
        abs_top_bm25 = np.where(np.isfinite(abs_top_bm25), abs_top_bm25, 0.0)

        # ---- ranks / norms / margins over candidate sets (T87: leaf columns,
        # each a full (b, C) sort or top-2 partition — skip the ones nobody
        # asked for). A margin call also produces the paired q_gap_* column, so
        # it runs when *either* is wanted and each column is only added to
        # ``data`` below if it was individually requested. ----
        want_rank_d_desc = _want("rank_d_desc")
        want_rank_b_desc = _want("rank_b_desc")
        want_rank_d_knn = _want("rank_d_knn")
        want_rank_b_knn = _want("rank_b_knn")
        want_norm_d_desc = _want("norm_d_desc")
        want_norm_b_desc = _want("norm_b_desc")
        want_agreement = _want("n_signal_agreement")
        want_margin_d_desc = _want("margin_d_desc") or _want("q_gap_d_desc")
        want_margin_d_proto = _want("margin_d_proto")
        want_margin_d_knn = _want("margin_d_knn") or _want("q_gap_d_knn")
        want_margin_b_desc = _want("margin_b_desc") or _want("q_gap_b_desc")
        want_margin_b_knn = _want("margin_b_knn")

        rk_dd = _row_rank(desc_d, mask) if want_rank_d_desc else None
        rk_bd = _row_rank(bdesc, mask) if want_rank_b_desc else None
        rk_dk = _row_rank(d_sum, mask) if want_rank_d_knn else None
        rk_bk = _row_rank(b_sum, mask) if want_rank_b_knn else None
        nm_dd = _row_minmax(desc_d, mask) if want_norm_d_desc else None
        nm_bd = _row_minmax(bdesc, mask) if want_norm_b_desc else None

        mg_dd, gap_dd = _row_margin(desc_d, mask) if want_margin_d_desc else (None, None)
        mg_dp = _row_margin(proto, mask)[0] if want_margin_d_proto else None
        mg_dk, gap_dk = _row_margin(d_sum, mask) if want_margin_d_knn else (None, None)
        mg_bd, gap_bd = _row_margin(bdesc, mask) if want_margin_b_desc else (None, None)
        mg_bk = _row_margin(b_sum, mask)[0] if want_margin_b_knn else None

        if want_agreement:
            argstack = np.stack([a_desc, a_proto, a_bdesc, a_dknn, a_bknn], axis=1)
            n_agree = np.fromiter(
                (5 - len({v for v in r if v >= 0}) for r in argstack),
                dtype=np.float64,
                count=argstack.shape[0],
            )
        else:
            n_agree = None

        # ---- gather one value per (row, col) ----
        def g(M):  # gather helper
            return M[rows, cols]

        data: Dict[str, np.ndarray] = {}
        if _want("d_desc_sim"):
            data["d_desc_sim"] = g(desc_d)
        if _want("d_proto_sim"):
            data["d_proto_sim"] = g(proto)
        if _want("d_knn_sum"):
            data["d_knn_sum"] = g(d_sum)
        if _want("d_knn_max"):
            data["d_knn_max"] = g(d_max)
        if _want("d_knn_count"):
            data["d_knn_count"] = g(d_cnt)
        if _want("b_desc_sim"):
            data["b_desc_sim"] = g(bdesc)
        if _want("b_knn_sum"):
            data["b_knn_sum"] = g(b_sum)
        if _want("b_knn_max"):
            data["b_knn_max"] = g(b_max)
        if _want("b_knn_count"):
            data["b_knn_count"] = g(b_cnt)
        if _want("desc_proto_gap"):
            data["desc_proto_gap"] = g(desc_d) - g(proto)
        if _want("class_log_freq"):
            data["class_log_freq"] = np.log1p(class_freq[cols].astype(np.float64))
        if _want("abs_top_dense_sim"):
            data["abs_top_dense_sim"] = abs_top_dense[rows]
        if _want("abs_top_bm25"):
            data["abs_top_bm25"] = abs_top_bm25[rows]
        if _want("is_d_desc_top1"):
            data["is_d_desc_top1"] = (cols == a_desc[rows]).astype(np.float64)
        if _want("is_d_proto_top1"):
            data["is_d_proto_top1"] = ((cols == a_proto[rows]) & (a_proto[rows] >= 0)).astype(
                np.float64
            )
        if _want("is_b_desc_top1"):
            data["is_b_desc_top1"] = ((cols == a_bdesc[rows]) & (a_bdesc[rows] >= 0)).astype(
                np.float64
            )
        if _want("is_d_knn_top1"):
            data["is_d_knn_top1"] = (cols == a_dknn[rows]).astype(np.float64)
        if _want("is_b_knn_top1"):
            data["is_b_knn_top1"] = ((cols == a_bknn[rows]) & (a_bknn[rows] >= 0)).astype(
                np.float64
            )
        if _want("b_desc_missing"):
            data["b_desc_missing"] = np.isnan(g(bdesc)).astype(np.float64)
        if _want("b_knn_missing"):
            data["b_knn_missing"] = np.isnan(g(b_sum)).astype(np.float64)
        if _want("d_knn_missing"):
            data["d_knn_missing"] = np.isnan(g(d_sum)).astype(np.float64)
        if rk_dd is not None and _want("rank_d_desc"):
            data["rank_d_desc"] = g(rk_dd)
        if rk_bd is not None and _want("rank_b_desc"):
            data["rank_b_desc"] = g(rk_bd)
        if rk_dk is not None and _want("rank_d_knn"):
            data["rank_d_knn"] = g(rk_dk)
        if rk_bk is not None and _want("rank_b_knn"):
            data["rank_b_knn"] = g(rk_bk)
        if nm_dd is not None and _want("norm_d_desc"):
            data["norm_d_desc"] = g(nm_dd)
        if nm_bd is not None and _want("norm_b_desc"):
            data["norm_b_desc"] = g(nm_bd)
        if n_agree is not None:
            data["n_signal_agreement"] = n_agree[rows]
        if mg_dd is not None and _want("margin_d_desc"):
            data["margin_d_desc"] = g(mg_dd)
        if mg_dp is not None:
            data["margin_d_proto"] = g(mg_dp)
        if mg_dk is not None and _want("margin_d_knn"):
            data["margin_d_knn"] = g(mg_dk)
        if mg_bd is not None and _want("margin_b_desc"):
            data["margin_b_desc"] = g(mg_bd)
        if mg_bk is not None:
            data["margin_b_knn"] = g(mg_bk)
        if gap_dd is not None and _want("q_gap_d_desc"):
            data["q_gap_d_desc"] = gap_dd[rows]
        if gap_dk is not None and _want("q_gap_d_knn"):
            data["q_gap_d_knn"] = gap_dk[rows]
        if gap_bd is not None and _want("q_gap_b_desc"):
            data["q_gap_b_desc"] = gap_bd[rows]

        core_cols = [c for c in FEATURE_NAMES if c in data]
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
