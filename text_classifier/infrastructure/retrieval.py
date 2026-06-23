"""Retrieval adapters.

BM25Index uses a precomputed per-(doc, term) weight matrix W so that scoring a
batch of queries is a single sparse mat-mul: because we ignore query-term
frequency, BM25(query, doc) = sum over distinct query terms t of W[doc, t],
which is exactly (Q_binary @ W.T). This replaces the per-query Python loop.

DenseRetrieverAdapter holds example embeddings, class prototypes (mean of
example embeddings), and class-description embeddings, and answers kNN via a
query-chunked cosine mat-mul.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..config import RetrievalConfig
from ..domain import DenseRetriever, LabelSpace, LexicalRetriever, TextEncoder


# --------------------------------------------------------------------------- BM25
class BM25Index:
    """Okapi BM25 (Lucene IDF variant) with a precomputed weight matrix."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, **cv_kwargs: Any):
        self.k1, self.b, self.cv_kwargs = k1, b, cv_kwargs

    def fit(self, corpus: Sequence[str]) -> "BM25Index":
        self.vectorizer = CountVectorizer(**self.cv_kwargs)
        counts = self.vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
        self.n_docs = counts.shape[0]

        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avg = float(doc_len.mean()) if self.n_docs else 0.0
        avg = avg or 1.0
        len_norm = 1.0 - self.b + self.b * (doc_len / avg)

        # W[doc, t] = idf_t * tf*(k1+1) / (tf + k1 * len_norm_doc)
        coo = counts.tocoo()
        tf = coo.data
        w = self.idf[coo.col] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm[coo.row])
        W = sparse.coo_matrix((w.astype(np.float32), (coo.row, coo.col)), shape=counts.shape)
        self._Wt = W.tocsc().T.tocsr()  # (vocab, n_docs)
        return self

    def _query_incidence(self, texts: Sequence[str]) -> sparse.csr_matrix:
        q = self.vectorizer.transform(list(texts))
        q.data[:] = 1.0  # binary incidence: ignore query-term frequency
        return q.astype(np.float32)

    def score_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Dense (b, n_docs) score block. Use for small doc sets (descriptions)."""
        return np.asarray((self._query_incidence(texts) @ self._Wt).todense(), dtype=np.float32)

    def top_k(self, texts: Sequence[str], k: int, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """(idx (b, k) int with -1 pad, score (b, k) float with NaN pad). Only
        strictly-positive scores are returned; the rest is padding."""
        b = len(texts)
        kk = min(k, self.n_docs)
        out_idx = np.full((b, k), -1, dtype=np.int64)
        out_score = np.full((b, k), np.nan, dtype=np.float32)
        Qbin = self._query_incidence(texts)
        for s in range(0, b, chunk):
            S = np.asarray((Qbin[s:s + chunk] @ self._Wt).todense(), dtype=np.float32)
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            rows = np.arange(part.shape[0])[:, None]
            part_s = S[rows, part]
            order = np.argsort(-part_s, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
            sc = np.take_along_axis(part_s, order, axis=1)
            bad = sc <= 0
            idx = np.where(bad, -1, idx)
            sc = np.where(bad, np.nan, sc)
            out_idx[s:s + chunk, :kk] = idx
            out_score[s:s + chunk, :kk] = sc
        return out_idx, out_score


# ----------------------------------------------------------------- lexical adapter
class LexicalRetrieverAdapter(LexicalRetriever):
    def __init__(self, example_bm25: BM25Index, example_labels: np.ndarray, desc_bm25: BM25Index, k_chunk: int = 256):
        self._examples = example_bm25
        self._labels = example_labels.astype(np.int64)
        self._descriptions = desc_bm25
        self._k_chunk = k_chunk

    @classmethod
    def build(cls, texts: Sequence[str], labels: np.ndarray, label_space: LabelSpace,
              cfg: RetrievalConfig) -> "LexicalRetrieverAdapter":
        ex = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(texts)
        desc = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(label_space.descriptions)
        return cls(ex, np.asarray(labels), desc, cfg.dense_chunk)

    def knn_example_labels(self, query_texts: Sequence[str], k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx, score = self._examples.top_k(query_texts, k, self._k_chunk)
        labels = np.where(idx >= 0, self._labels[np.clip(idx, 0, None)], -1)
        return labels.astype(np.int64), score

    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        return self._descriptions.score_matrix(query_texts)


# ------------------------------------------------------------------- dense adapter
def _dense_topk(Q: np.ndarray, X: np.ndarray, k: int, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    k = min(k, n)
    out_idx = np.empty((Q.shape[0], k), dtype=np.int64)
    out_sim = np.empty((Q.shape[0], k), dtype=np.float32)
    Xt = np.ascontiguousarray(X.T)
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s:s + chunk] @ Xt
        part = np.argpartition(sims, -k, axis=1)[:, -k:]
        rows = np.arange(part.shape[0])[:, None]
        part_sims = sims[rows, part]
        order = np.argsort(-part_sims, axis=1)
        out_idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        out_sim[s:s + chunk] = np.take_along_axis(part_sims, order, axis=1)
    return out_idx, out_sim


@dataclass
class DenseState:
    """Serializable numeric state of the dense retriever."""
    example_emb: np.ndarray
    example_labels: np.ndarray
    prototypes: np.ndarray
    description_emb: np.ndarray
    class_freq: np.ndarray


class DenseRetrieverAdapter(DenseRetriever):
    def __init__(self, state: DenseState, chunk: int = 256):
        self._s = state
        self._chunk = chunk

    @classmethod
    def build(cls, encoder: TextEncoder, texts: Sequence[str], labels: np.ndarray,
              label_space: LabelSpace, cfg: RetrievalConfig) -> "DenseRetrieverAdapter":
        emb = encoder.encode(texts)
        dim = emb.shape[1]
        C = label_space.size
        proto = np.full((C, dim), np.nan, dtype=np.float32)
        freq = np.zeros(C, dtype=np.int64)
        labels = np.asarray(labels)
        for c in range(C):
            mask = labels == c
            freq[c] = int(mask.sum())
            if freq[c]:
                v = emb[mask].mean(axis=0)
                norm = np.linalg.norm(v)
                if norm > 0:
                    proto[c] = (v / norm).astype(np.float32)
        desc = encoder.encode(label_space.descriptions)
        return cls(DenseState(emb, labels.astype(np.int64), proto, desc, freq), cfg.dense_chunk)

    @property
    def state(self) -> DenseState:
        return self._s

    @property
    def class_freq(self) -> np.ndarray:
        return self._s.class_freq

    def knn_example_labels(self, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx, sim = _dense_topk(query_emb, self._s.example_emb, k, self._chunk)
        return self._s.example_labels[idx], sim

    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return query_emb @ self._s.prototypes.T

    def description_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return query_emb @ self._s.description_emb.T
