"""Feature assembly (application service).

Turns the five retrieval signals into one feature row per (item, candidate class).
Everything is vectorized: each signal becomes a (batch, n_classes) matrix, the
candidate set is a boolean mask, and feature columns are gathered with fancy
indexing. Queries are processed in chunks to bound peak memory.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import warnings

import numpy as np
import pandas as pd

from ..domain import (
    CandidatePolicy,
    DenseRetriever,
    FEATURE_NAMES,
    LabelSpace,
    LexicalRetriever,
)


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
        query_ids: Sequence[Any],
        query_labels: Optional[np.ndarray] = None,
        chunk: int = 4096,
    ) -> pd.DataFrame:
        frames = []
        ids = np.asarray(query_ids)
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
                )
            )
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FEATURE_NAMES)
        )

    def _assemble_chunk(self, texts, q_emb, dense, lexical, k, ids, labels) -> pd.DataFrame:
        C = self._space.size
        n = self._policy.top_n_per_signal
        class_freq = dense.class_freq

        # ---- signals as (b, C) matrices ----
        desc_d = np.asarray(dense.description_similarity(q_emb), dtype=np.float64)
        proto = np.asarray(dense.prototype_similarity(q_emb), dtype=np.float64)
        dn_lab, dn_sim = dense.knn_example_labels(q_emb, k)
        d_sum, d_max, d_cnt = _scatter_knn(dn_lab, dn_sim, C)

        bdesc_raw = np.asarray(lexical.description_score(texts), dtype=np.float64)
        bdesc = np.where(bdesc_raw > 0, bdesc_raw, np.nan)  # 0 overlap == missing
        bn_lab, bn_sco = lexical.knn_example_labels(texts, k)
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
            return pd.DataFrame(columns=FEATURE_NAMES + (["is_true"] if labels is not None else []))

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

        argstack = np.stack([a_desc, a_proto, a_bdesc, a_dknn, a_bknn], axis=1)
        n_agree = np.fromiter(
            (5 - len({v for v in r if v >= 0}) for r in argstack),
            dtype=np.float64,
            count=argstack.shape[0],
        )

        # ---- ranks / norms over candidate sets ----
        rk_dd = _row_rank(desc_d, mask)
        rk_bd = _row_rank(bdesc, mask)
        rk_dk = _row_rank(d_sum, mask)
        rk_bk = _row_rank(b_sum, mask)
        nm_dd = _row_minmax(desc_d, mask)
        nm_bd = _row_minmax(bdesc, mask)

        # ---- gather one value per (row, col) ----
        def g(M):  # gather helper
            return M[rows, cols]

        data = {
            "d_desc_sim": g(desc_d),
            "d_proto_sim": g(proto),
            "d_knn_sum": g(d_sum),
            "d_knn_max": g(d_max),
            "d_knn_count": g(d_cnt),
            "b_desc_sim": g(bdesc),
            "b_knn_sum": g(b_sum),
            "b_knn_max": g(b_max),
            "b_knn_count": g(b_cnt),
            "desc_proto_gap": g(desc_d) - g(proto),
            "class_log_freq": np.log1p(class_freq[cols].astype(np.float64)),
            "abs_top_dense_sim": abs_top_dense[rows],
            "abs_top_bm25": abs_top_bm25[rows],
            "is_d_desc_top1": (cols == a_desc[rows]).astype(np.float64),
            "is_d_proto_top1": ((cols == a_proto[rows]) & (a_proto[rows] >= 0)).astype(np.float64),
            "is_b_desc_top1": ((cols == a_bdesc[rows]) & (a_bdesc[rows] >= 0)).astype(np.float64),
            "is_d_knn_top1": (cols == a_dknn[rows]).astype(np.float64),
            "is_b_knn_top1": ((cols == a_bknn[rows]) & (a_bknn[rows] >= 0)).astype(np.float64),
            "b_desc_missing": np.isnan(g(bdesc)).astype(np.float64),
            "b_knn_missing": np.isnan(g(b_sum)).astype(np.float64),
            "d_knn_missing": np.isnan(g(d_sum)).astype(np.float64),
            "rank_d_desc": g(rk_dd),
            "rank_b_desc": g(rk_bd),
            "rank_d_knn": g(rk_dk),
            "rank_b_knn": g(rk_bk),
            "norm_d_desc": g(nm_dd),
            "norm_b_desc": g(nm_bd),
            "n_signal_agreement": n_agree[rows],
        }
        df = pd.DataFrame({col: np.asarray(data[col], dtype=np.float32) for col in FEATURE_NAMES})
        df["item_id"] = ids[rows]
        df["candidate"] = cols.astype(np.int64)
        if labels is not None:
            df["is_true"] = (cols == labels[rows]).astype(np.int64)
        return df
