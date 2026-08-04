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

import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..config import RetrievalConfig
from ..domain import DenseRetriever, LabelSpace, LexicalRetriever, TextEncoder


def _exclude_self(
    idx: np.ndarray, score: np.ndarray, exclude: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the per-row self-match from best-first neighbor lists and return
    exactly ``(b, k)``.

    ``idx``/``score`` are ``(b, m)`` neighbor indices/scores sorted best-first
    (``-1``/NaN pad allowed). ``exclude`` is ``(b,)``: for row ``r`` the example
    index ``exclude[r]`` is removed from that row's neighbors (a value ``< 0``
    removes nothing — a query not present in the pool). Real non-self neighbors
    are kept best-first, everything else is nulled to ``(-1, NaN)``, and the
    result is padded/trimmed to width ``k``. This is the leave-one-out self-mask:
    fetch one extra neighbor upstream (``k + 1``) so ``k`` real ones always remain.
    """
    b, m = idx.shape
    exclude = np.asarray(exclude)
    self_hit = (idx == exclude[:, None]) & (exclude[:, None] >= 0)
    valid = (idx >= 0) & ~self_hit
    # Stable sort by ~valid: kept neighbours (valid) stay first in their existing
    # best-first order; self-matches and padding sink to the end.
    order = np.argsort(~valid, axis=1, kind="stable")
    idx_s = np.take_along_axis(idx, order, axis=1)
    score_s = np.take_along_axis(score, order, axis=1)
    valid_s = np.take_along_axis(valid, order, axis=1)
    idx_s = np.where(valid_s, idx_s, -1)
    score_s = np.where(valid_s, score_s, np.nan)
    if m >= k:
        return idx_s[:, :k], score_s[:, :k]
    pad_i = np.full((b, k - m), -1, dtype=idx_s.dtype)
    pad_s = np.full((b, k - m), np.nan, dtype=score_s.dtype)
    return np.concatenate([idx_s, pad_i], axis=1), np.concatenate([score_s, pad_s], axis=1)


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

    def top_k(
        self, texts: Sequence[str], k: int, chunk: int = 256, exclude: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(idx (b, k) int with -1 pad, score (b, k) float with NaN pad). Only
        strictly-positive scores are returned; the rest is padding.

        ``exclude`` (b,), when given, drops one example index per query from that
        query's neighbors (a value < 0 drops nothing) — the leave-one-out
        self-mask. One extra neighbor is fetched so ``k`` real ones survive."""
        b = len(texts)
        # Fetch one extra when self-masking so k real neighbours remain after drop.
        width = k + 1 if exclude is not None else k
        fetch = min(width, self.n_docs)
        out_idx = np.full((b, width), -1, dtype=np.int64)
        out_score = np.full((b, width), np.nan, dtype=np.float32)
        if fetch > 0:
            Qbin = self._query_incidence(texts)
            for s in range(0, b, chunk):
                S = np.asarray((Qbin[s : s + chunk] @ self._Wt).todense(), dtype=np.float32)
                part = np.argpartition(-S, fetch - 1, axis=1)[:, :fetch]
                rows = np.arange(part.shape[0])[:, None]
                part_s = S[rows, part]
                order = np.argsort(-part_s, axis=1)
                idx = np.take_along_axis(part, order, axis=1)
                sc = np.take_along_axis(part_s, order, axis=1)
                bad = sc <= 0
                idx = np.where(bad, -1, idx)
                sc = np.where(bad, np.nan, sc)
                out_idx[s : s + chunk, :fetch] = idx
                out_score[s : s + chunk, :fetch] = sc
        if exclude is not None:
            return _exclude_self(out_idx, out_score, exclude, k)
        return out_idx, out_score

    def to_state(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Split this index into npz-able arrays and a JSON-clean meta dict.

        ``cv_kwargs`` must itself be JSON-clean (str/int/float/bool/list/dict of
        those) — a custom callable analyzer can't round-trip through this format
        (it never round-tripped safely under pickle-by-value semantics either)."""
        try:
            json.dumps(self.cv_kwargs)
        except TypeError as exc:
            raise ValueError(
                f"BM25Index.cv_kwargs must be JSON-serializable to persist without "
                f"pickle; got {self.cv_kwargs!r}"
            ) from exc
        Wt = self._Wt.tocsr()
        arrays = {
            "Wt_data": Wt.data.astype(np.float32),
            "Wt_indices": Wt.indices.astype(np.int64),
            "Wt_indptr": Wt.indptr.astype(np.int64),
        }
        meta = {
            "Wt_shape": list(Wt.shape),
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "cv_kwargs": self.cv_kwargs,
            "vocabulary": {term: int(col) for term, col in self.vectorizer.vocabulary_.items()},
        }
        return arrays, meta

    @classmethod
    def from_state(cls, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> "BM25Index":
        obj = cls(meta["k1"], meta["b"], **meta["cv_kwargs"])
        obj.n_docs = meta["n_docs"]
        # A fixed vocabulary means CountVectorizer.transform() works without a
        # prior fit() call, so this reproduces fit()'s analyzer exactly.
        # `_validate_vocabulary()` eagerly populates `vocabulary_` (otherwise it
        # is set lazily on first transform()), so a from_state() index that is
        # immediately re-persisted without ever scoring a query still has it.
        obj.vectorizer = CountVectorizer(vocabulary=meta["vocabulary"], **meta["cv_kwargs"])
        obj.vectorizer._validate_vocabulary()
        shape = tuple(meta["Wt_shape"])
        obj._Wt = sparse.csr_matrix(
            (arrays["Wt_data"], arrays["Wt_indices"], arrays["Wt_indptr"]), shape=shape
        )
        return obj


# ----------------------------------------------------------------- lexical adapter
class LexicalRetrieverAdapter(LexicalRetriever):
    def __init__(
        self,
        example_bm25: BM25Index,
        example_labels: np.ndarray,
        desc_bm25: BM25Index,
        k_chunk: int = 256,
    ):
        self._examples = example_bm25
        self._labels = example_labels.astype(np.int64)
        self._descriptions = desc_bm25
        self._k_chunk = k_chunk

    @classmethod
    def build(
        cls, texts: Sequence[str], labels: np.ndarray, label_space: LabelSpace, cfg: RetrievalConfig
    ) -> "LexicalRetrieverAdapter":
        ex = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(texts)
        desc = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(label_space.descriptions)
        return cls(ex, np.asarray(labels), desc, cfg.dense_chunk)

    def knn_example_labels(
        self, query_texts: Sequence[str], k: int, exclude_idx: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx, score = self._examples.top_k(query_texts, k, self._k_chunk, exclude=exclude_idx)
        labels = np.where(idx >= 0, self._labels[np.clip(idx, 0, None)], -1)
        return labels.astype(np.int64), score

    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        return self._descriptions.score_matrix(query_texts)

    def with_added_descriptions(self, all_descriptions: Sequence[str]) -> "LexicalRetrieverAdapter":
        """Return a copy whose description BM25 is refit over ``all_descriptions``
        (the full, extended class-description list, existing classes first then
        the new ones).

        Used to widen a trained model's label space at inference time. BM25
        IDF is corpus-global, so the description side must be *refit* over every
        description, not have a row appended — hence the full list. The example
        index and its labels are reused verbatim: a class added this way is
        description-only (no example support), so nothing on the example side
        changes. The new BM25 keeps the same ``k1``/``b``/tokenizer kwargs as the
        original, so existing classes score identically."""
        old = self._descriptions
        desc = BM25Index(old.k1, old.b, **old.cv_kwargs).fit(all_descriptions)
        return LexicalRetrieverAdapter(self._examples, self._labels, desc, self._k_chunk)

    def to_state(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Split this adapter into npz-able arrays and a JSON-clean meta dict,
        prefixing each BM25 sub-index's arrays so both pack into one npz file."""
        ex_arrays, ex_meta = self._examples.to_state()
        desc_arrays, desc_meta = self._descriptions.to_state()
        arrays: Dict[str, np.ndarray] = {f"examples_{k}": v for k, v in ex_arrays.items()}
        arrays.update({f"descriptions_{k}": v for k, v in desc_arrays.items()})
        arrays["example_labels"] = self._labels
        meta = {"examples": ex_meta, "descriptions": desc_meta, "k_chunk": self._k_chunk}
        return arrays, meta

    @classmethod
    def from_state(
        cls, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]
    ) -> "LexicalRetrieverAdapter":
        ex_arrays = {k[len("examples_") :]: v for k, v in arrays.items() if k.startswith("examples_")}
        desc_arrays = {
            k[len("descriptions_") :]: v for k, v in arrays.items() if k.startswith("descriptions_")
        }
        ex = BM25Index.from_state(ex_arrays, meta["examples"])
        desc = BM25Index.from_state(desc_arrays, meta["descriptions"])
        return cls(ex, arrays["example_labels"], desc, meta["k_chunk"])


# ------------------------------------------------------------------- dense adapter
def _dense_topk(
    Q: np.ndarray, X: np.ndarray, k: int, chunk: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k nearest examples by dot product, always shaped ``(n_queries, k)``.

    The result is padded to the requested ``k`` even when the corpus is smaller
    (``k > n_examples``): the first ``min(k, n)`` columns hold real neighbours in
    descending-similarity order and the remainder are ``-1`` / ``NaN`` padding,
    mirroring ``BM25Index.top_k``. An empty query batch returns ``(0, k)`` arrays.
    """
    n = X.shape[0]
    k_eff = min(k, n)
    out_idx = np.full((Q.shape[0], k), -1, dtype=np.int64)
    out_sim = np.full((Q.shape[0], k), np.nan, dtype=np.float32)
    if Q.shape[0] == 0 or k_eff == 0:
        return out_idx, out_sim
    Xt = np.ascontiguousarray(X.T)
    for s in range(0, Q.shape[0], chunk):
        sims = Q[s : s + chunk] @ Xt
        part = np.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
        rows = np.arange(part.shape[0])[:, None]
        part_sims = sims[rows, part]
        order = np.argsort(-part_sims, axis=1)
        out_idx[s : s + chunk, :k_eff] = np.take_along_axis(part, order, axis=1)
        out_sim[s : s + chunk, :k_eff] = np.take_along_axis(part_sims, order, axis=1)
    return out_idx, out_sim


def _prototypes_and_freq(
    emb: np.ndarray, labels: np.ndarray, n_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class prototype (L2-normalized mean example embedding) and example
    count, over ``n_classes`` classes. A class with no examples gets an
    all-``NaN`` prototype row (XGBoost reads NaN as "missing"). Shared by
    ``DenseRetrieverAdapter.build`` (fresh) and ``with_added_examples`` (merged
    pool) so both compute prototypes identically."""
    dim = emb.shape[1]
    proto = np.full((n_classes, dim), np.nan, dtype=np.float32)
    freq = np.zeros(n_classes, dtype=np.int64)
    labels = np.asarray(labels)
    for c in range(n_classes):
        mask = labels == c
        freq[c] = int(mask.sum())
        if freq[c]:
            v = emb[mask].mean(axis=0)
            norm = np.linalg.norm(v)
            if norm > 0:
                proto[c] = (v / norm).astype(np.float32)
    return proto, freq


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
    def build(
        cls,
        encoder: TextEncoder,
        texts: Sequence[str],
        labels: np.ndarray,
        label_space: LabelSpace,
        cfg: RetrievalConfig,
    ) -> "DenseRetrieverAdapter":
        # The example pool and class descriptions are the *document* side of
        # retrieval; asymmetric encoders (E5/BGE prompts) encode them with the
        # document prompt so query embeddings land in the matching space.
        emb = encoder.encode_documents(texts)
        labels = np.asarray(labels)
        desc = encoder.encode_documents(label_space.descriptions)
        return cls.build_from_embeddings(emb, labels, desc, label_space, cfg)

    @classmethod
    def build_from_embeddings(
        cls,
        example_emb: np.ndarray,
        labels: np.ndarray,
        description_emb: np.ndarray,
        label_space: LabelSpace,
        cfg: RetrievalConfig,
    ) -> "DenseRetrieverAdapter":
        """Build from already-encoded document embeddings (T88): the encode step
        is a pure function of the text for a frozen shared encoder, so a caller
        that has already encoded the full pool/description set once (e.g. the
        training pipeline's per-fold loop) can slice and hand in embeddings
        instead of paying `encode_documents` again per fold. ``build`` still
        encodes internally and delegates here, so every existing caller and test
        double is untouched."""
        labels = np.asarray(labels)
        proto, freq = _prototypes_and_freq(example_emb, labels, label_space.size)
        return cls(
            DenseState(example_emb, labels.astype(np.int64), proto, description_emb, freq),
            cfg.dense_chunk,
        )

    @property
    def state(self) -> DenseState:
        return self._s

    @property
    def class_freq(self) -> np.ndarray:
        return self._s.class_freq

    def knn_example_labels(
        self, query_emb: np.ndarray, k: int, exclude_idx: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Fetch one extra when self-masking so k real neighbours survive the drop.
        fetch = k + 1 if exclude_idx is not None else k
        idx, sim = _dense_topk(query_emb, self._s.example_emb, fetch, self._chunk)
        if exclude_idx is not None:
            idx, sim = _exclude_self(idx, sim, exclude_idx, k)
        # idx == -1 marks padding (k > n_examples); keep it as -1 rather than
        # letting np indexing wrap around to a real label.
        labels = np.where(idx >= 0, self._s.example_labels[np.clip(idx, 0, None)], -1)
        return labels.astype(np.int64), sim

    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return query_emb @ self._s.prototypes.T

    def loo_prototype_similarity(self, query_emb: np.ndarray, self_idx: np.ndarray) -> np.ndarray:
        """Prototype similarity with each query's own example left out of its own
        class prototype (see the port docstring).

        Only the own-class column of each in-pool query (``self_idx >= 0``) is
        recomputed: its class prototype is the normalized mean of that class's
        embeddings *minus* the query's own, and cosine to it (query embeddings are
        L2-normalized, so ``q·mean/‖mean‖`` == cosine and the count cancels). A
        query that was its class's only example gets NaN (no prototype without it),
        which XGBoost reads as "did not retrieve" — the same as an absent class.
        Every other column, and every out-of-pool query, keeps the ordinary value.
        """
        base = self.prototype_similarity(query_emb)
        self_idx = np.asarray(self_idx)
        rows = np.nonzero(self_idx >= 0)[0]
        if rows.size == 0:
            return base
        E = self._s.example_emb.astype(np.float64)
        y = self._s.example_labels
        C = base.shape[1]
        class_sum = np.zeros((C, E.shape[1]), dtype=np.float64)
        class_cnt = np.zeros(C, dtype=np.float64)
        np.add.at(class_sum, y, E)
        np.add.at(class_cnt, y, 1.0)

        s = self_idx[rows]
        c = y[s]  # own class of each in-pool query
        loo_vec = class_sum[c] - E[s]  # class sum with the query's own vector removed
        loo_cnt = class_cnt[c] - 1.0
        norm = np.linalg.norm(loo_vec, axis=1)
        q = np.asarray(query_emb, dtype=np.float64)[rows]
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.einsum("md,md->m", q, loo_vec) / norm
        sim = np.where((loo_cnt > 0) & (norm > 0), sim, np.nan)
        out = base.copy()
        out[rows, c] = sim.astype(out.dtype)
        return out

    def description_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return query_emb @ self._s.description_emb.T

    def with_added_classes(
        self, encoder: TextEncoder, new_descriptions: Sequence[str]
    ) -> "DenseRetrieverAdapter":
        """Return a copy extended with new, example-free classes.

        Each new class is appended at the end so existing class indices stay
        stable. Its description is encoded with the frozen deployment encoder
        (``encode_documents`` — the document role, matching how the original
        descriptions were embedded), and it gets an all-``NaN`` prototype row and
        ``class_freq = 0``: it has no training examples, so it is description-only
        and carries no prototype/kNN support. The example pool is reused verbatim.
        """
        new_descriptions = list(new_descriptions)
        s = self._s
        if not new_descriptions:
            return DenseRetrieverAdapter(s, self._chunk)
        new_desc = np.asarray(encoder.encode_documents(new_descriptions), dtype=np.float32)
        dim = s.description_emb.shape[1]
        if new_desc.shape[1] != dim:
            raise ValueError(
                f"encoder produced {new_desc.shape[1]}-dim embeddings but the model's "
                f"dense index is {dim}-dim; the same encoder must be used to extend it"
            )
        m = len(new_descriptions)
        prototypes = np.concatenate(
            [s.prototypes, np.full((m, dim), np.nan, dtype=np.float32)], axis=0
        )
        class_freq = np.concatenate([s.class_freq, np.zeros(m, dtype=s.class_freq.dtype)])
        description_emb = np.concatenate([s.description_emb, new_desc], axis=0)
        extended = DenseState(
            s.example_emb, s.example_labels, prototypes, description_emb, class_freq
        )
        return DenseRetrieverAdapter(extended, self._chunk)

    def with_updated_descriptions(
        self, encoder: TextEncoder, edits: dict
    ) -> "DenseRetrieverAdapter":
        """Return a copy with the description embeddings at ``edits``' class
        indices re-encoded: ``{class_index: new_description_text}``, for
        editing an *existing* class's description in place. A brand-new class's
        description is added via ``with_added_classes``, not this method. Every
        row not named in ``edits`` is untouched."""
        if not edits:
            return DenseRetrieverAdapter(self._s, self._chunk)
        s = self._s
        idxs = list(edits.keys())
        new_rows = np.asarray(encoder.encode_documents([edits[i] for i in idxs]), dtype=np.float32)
        description_emb = s.description_emb.copy()
        description_emb[idxs] = new_rows
        updated = DenseState(
            s.example_emb, s.example_labels, s.prototypes, description_emb, s.class_freq
        )
        return DenseRetrieverAdapter(updated, self._chunk)

    def with_added_examples(
        self,
        encoder: TextEncoder,
        new_texts: Sequence[str],
        new_labels: np.ndarray,
        n_classes: int,
    ) -> "DenseRetrieverAdapter":
        """Return a copy whose example pool is extended with ``new_texts``/
        ``new_labels``. Only ``new_texts`` is encoded — the existing
        ``example_emb`` is reused verbatim (the encoder is frozen, so
        re-encoding it would reproduce the same vectors at needless cost, and
        for an expensive sentence-transformer encoder that cost is the whole
        point of avoiding a retrain). Prototypes and ``class_freq`` are
        recomputed over the *merged* pool for all ``n_classes`` classes — an
        added example can only change its own class's prototype, but computing
        every class the same way ``build`` does keeps the numerics provably
        identical to a from-scratch build over the same merged corpus.
        Description embeddings are untouched; see ``with_added_classes``/
        ``with_updated_descriptions`` for those."""
        s = self._s
        new_texts = list(new_texts)
        if new_texts:
            new_emb = np.asarray(encoder.encode_documents(new_texts), dtype=np.float32)
        else:
            new_emb = np.zeros((0, s.example_emb.shape[1]), dtype=s.example_emb.dtype)
        merged_emb = np.concatenate([s.example_emb, new_emb], axis=0)
        merged_labels = np.concatenate([s.example_labels, np.asarray(new_labels, dtype=np.int64)])
        proto, freq = _prototypes_and_freq(merged_emb, merged_labels, n_classes)
        updated = DenseState(merged_emb, merged_labels, proto, s.description_emb, freq)
        return DenseRetrieverAdapter(updated, self._chunk)
