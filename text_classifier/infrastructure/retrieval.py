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
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from ..config import RetrievalConfig
from ..domain import ArrayOps, DenseRetriever, LabelSpace, LexicalRetriever, TextEncoder
from .array_ops import NumpyArrayOps


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


def bm25_prunes_vocab(cv_kwargs: Dict[str, Any]) -> bool:
    """Whether ``cv_kwargs`` prunes ``CountVectorizer``'s vocabulary by corpus
    statistics (``min_df``/``max_df``/``max_features``) — T32 A2's guard.

    Those three prune against whatever corpus they are fit on, so a
    full-corpus vocabulary and a per-fold vocabulary genuinely differ when any
    is set; a caller that wants to tokenize once and slice per fold (rather
    than refitting ``CountVectorizer`` per fold) must fall back to the
    ordinary per-fold path when this is ``True``."""
    return any(k in cv_kwargs for k in ("min_df", "max_df", "max_features"))


def _sparse_row_topk(S: sparse.csr_matrix, fetch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row-wise top-``fetch`` largest entries of a sparse CSR matrix whose
    explicit nonzeros are all strictly positive (true of a BM25 weight product:
    every ``idf`` and ``tf`` term is positive), without ever densifying the
    block (T32 B/overlap-with-A: the ``(chunk, n_docs)`` dense intermediate this
    replaces is what blows up memory at scale, and it also forced scanning
    every discarded zero to find the top-k).

    Rows with fewer than ``fetch`` explicit nonzeros are padded with ``-1``
    index / ``NaN`` score. Vectorized via one lexsort over every explicit
    nonzero — no per-row Python loop (CLAUDE.md convention)."""
    n_rows = S.shape[0]
    out_idx = np.full((n_rows, fetch), -1, dtype=np.int64)
    out_score = np.full((n_rows, fetch), np.nan, dtype=np.float32)
    if S.nnz == 0 or fetch == 0:
        return out_idx, out_score

    indptr = S.indptr
    counts = np.diff(indptr)
    row_ids = np.repeat(np.arange(n_rows), counts)
    cols = S.indices
    vals = S.data.astype(np.float32)

    # Sort by row ascending (primary), value descending (secondary). lexsort's
    # primary key is its *last* argument. Because `row_ids` is already
    # nondecreasing, the sorted output's row-blocks appear in the same
    # positions (`indptr`) as before the sort — only the order *within* each
    # block changes — so `rank` below is a plain arithmetic offset, not a
    # second grouping pass.
    order = np.lexsort((-vals, row_ids))
    row_ids_s = row_ids[order]
    cols_s = cols[order]
    vals_s = vals[order]

    group_start = np.repeat(indptr[:-1], counts)
    rank = np.arange(row_ids_s.size) - group_start
    keep = rank < fetch
    kr, kc, kv, kk = row_ids_s[keep], cols_s[keep], vals_s[keep], rank[keep]
    out_idx[kr, kk] = kc
    out_score[kr, kk] = kv

    # Defensive, not load-bearing today: every explicit nonzero here is a sum
    # of positive idf*tf terms, so this never actually fires, but it keeps the
    # positive-scores-only contract explicit rather than assumed.
    bad = ~(out_score > 0)
    out_idx = np.where(bad, -1, out_idx)
    out_score = np.where(bad, np.nan, out_score)
    return out_idx, out_score


# --------------------------------------------------------------------------- BM25
class BM25Index:
    """Okapi BM25 (Lucene IDF variant) with a precomputed weight matrix."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_df_ratio: Optional[float] = None,
        **cv_kwargs: Any,
    ):
        self.k1, self.b, self.max_df_ratio, self.cv_kwargs = k1, b, max_df_ratio, cv_kwargs

    @staticmethod
    def tokenize_corpus(
        corpus: Sequence[str], **cv_kwargs: Any
    ) -> Tuple[sparse.csr_matrix, CountVectorizer]:
        """Tokenize ``corpus`` into a raw counts matrix + fitted vectorizer
        (T32 A2), split out of ``fit`` so a caller that needs several
        ``BM25Index`` instances over slices of the *same* corpus (one per
        training fold, each with its own fold-local IDF) can tokenize once and
        reuse the counts/vectorizer via ``fit_from_counts``, instead of paying
        ``CountVectorizer.fit_transform``'s per-document regex analysis again
        for every instance."""
        vectorizer = CountVectorizer(**cv_kwargs)
        # `fit_transform` is one opaque C-backed call with no progress hook, so
        # this can't show fractional progress -- the bar just brackets it with
        # a start marker and an elapsed-time readout at completion.
        # `disable=None` (not the tqdm default of `False`): auto-silence when
        # stdout isn't a TTY (redirected to a file, CI logs) instead of
        # printing a bar per refresh tick.
        with tqdm(total=1, desc="BM25: tokenizing corpus", unit="corpus", disable=None) as bar:
            counts = vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
            bar.update(1)
        return counts, vectorizer

    def fit(self, corpus: Sequence[str]) -> "BM25Index":
        counts, vectorizer = self.tokenize_corpus(corpus, **self.cv_kwargs)
        return self.fit_from_counts(counts, vectorizer)

    def fit_from_counts(
        self, counts: sparse.csr_matrix, vectorizer: CountVectorizer
    ) -> "BM25Index":
        """Build the weight matrix from an already-tokenized ``counts`` matrix
        (rows = documents, columns = ``vectorizer``'s vocabulary). ``fit`` is
        exactly ``tokenize_corpus`` + this. A caller may pass a *row slice* of
        a larger corpus's counts (e.g. one training fold's rows): document
        frequency, length normalization and IDF are recomputed from the slice
        alone, which is correct — IDF is corpus-global and legitimately differs
        per fold — while the tokenization work (the counts themselves) is
        reused verbatim (T32 A2)."""
        counts = counts.tocsr().astype(np.float32)
        self.vectorizer = vectorizer
        self.n_docs = counts.shape[0]

        # Every step here is one vectorized numpy/scipy call over the whole
        # matrix (no per-row loop to report finer-grained progress over), so
        # the bar advances one tick per stage rather than a fake percentage.
        with tqdm(total=4, desc="BM25: fitting weight matrix", unit="step", disable=None) as bar:
            df = np.asarray((counts > 0).sum(axis=0)).ravel()
            self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
            bar.update(1)

            doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
            avg = float(doc_len.mean()) if self.n_docs else 0.0
            avg = avg or 1.0
            len_norm = 1.0 - self.b + self.b * (doc_len / avg)
            bar.update(1)

            # W[doc, t] = idf_t * tf*(k1+1) / (tf + k1 * len_norm_doc)
            coo = counts.tocoo()
            tf, row, col = coo.data, coo.row, coo.col
            if self.max_df_ratio is not None and self.n_docs:
                # T32 A4 (opt-in, lossy): drop entries whose term exceeds the df
                # ratio *before* building W, so the matrix actually shrinks rather
                # than merely carrying more near-zero weights.
                keep = (df[col] / self.n_docs) <= self.max_df_ratio
                tf, row, col = tf[keep], row[keep], col[keep]
            w = self.idf[col] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm[row])
            W = sparse.coo_matrix((w.astype(np.float32), (row, col)), shape=counts.shape)
            bar.update(1)

            self._Wt = W.tocsc().T.tocsr()  # (vocab, n_docs)
            bar.update(1)
        return self

    def _query_incidence(self, texts: Sequence[str]) -> sparse.csr_matrix:
        q = self.vectorizer.transform(list(texts))
        q.data[:] = 1.0  # binary incidence: ignore query-term frequency
        return q.astype(np.float32)

    def score_matrix(
        self, texts: Sequence[str], max_block_elems: Optional[int] = None
    ) -> np.ndarray:
        """Dense (b, n_docs) score block. For small doc sets (class
        descriptions) only — the example pool must go through the chunked,
        sparse ``top_k`` path instead. When ``max_block_elems`` is set (T32 B),
        raises rather than silently allocating a block over the configured
        cap, instead of leaving that guarantee to caller discipline."""
        b = len(texts)
        if max_block_elems is not None and b * self.n_docs > max_block_elems:
            raise ValueError(
                f"BM25Index.score_matrix would densify a ({b}, {self.n_docs}) block "
                f"({b * self.n_docs} elements), over the configured cap of "
                f"{max_block_elems} (RetrievalConfig.bm25_max_block_elems). "
                "score_matrix is for small document sets (e.g. class descriptions); "
                "a large example pool must go through the chunked, sparse top_k path."
            )
        return np.asarray((self._query_incidence(texts) @ self._Wt).todense(), dtype=np.float32)

    def top_k(
        self,
        texts: Sequence[str],
        k: int,
        chunk: int = 256,
        exclude: Any = None,
        n_jobs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(idx (b, k) int with -1 pad, score (b, k) float with NaN pad). Only
        strictly-positive scores are returned; the rest is padding.

        ``exclude`` (b,), when given, drops one example index per query from that
        query's neighbors (a value < 0 drops nothing) — the leave-one-out
        self-mask. One extra neighbor is fetched so ``k`` real ones survive.

        The per-chunk score block (``Qbin_chunk @ self._Wt``) is kept sparse
        end to end (T32 B): the product's only explicit nonzeros are candidates
        that share at least one term with the query, and the top-k is read
        directly off that sparse structure (``_sparse_row_topk``) rather than
        densifying to a ``(chunk, n_docs)`` block and discarding everything
        below the cut — the dense intermediate this used to allocate no longer
        exists, so there is nothing left to bound with a block-size cap.

        ``n_jobs`` (T-large-corpus): chunks are independent (each writes a
        disjoint row range of ``out_idx``/``out_score``), and scipy's sparse
        ``@`` releases the GIL during the C-level multiply, so ``n_jobs != 1``
        runs the chunk loop across a thread pool instead of serially — no
        pickling of ``self._Wt`` across a process boundary, which for a
        (vocab, n_docs) matrix at real corpus sizes would dwarf the per-chunk
        compute it's meant to parallelize. ``1`` (default) is the original
        serial loop, byte-for-byte; ``-1`` uses ``os.cpu_count()``."""
        b = len(texts)
        # Fetch one extra when self-masking so k real neighbours remain after drop.
        width = k + 1 if exclude is not None else k
        fetch = min(width, self.n_docs)
        out_idx = np.full((b, width), -1, dtype=np.int64)
        out_score = np.full((b, width), np.nan, dtype=np.float32)
        if fetch > 0:
            Qbin = self._query_incidence(texts)
            starts = list(range(0, b, chunk))
            # A single chunk finishes before a bar would ever render anything
            # useful; only bother above that (also keeps tiny/test-sized calls
            # quiet even when `disable=None` would otherwise let it through
            # on an interactive terminal).
            show_progress = len(starts) > 1

            def _run_chunk(s: int) -> None:
                S = (Qbin[s : s + chunk] @ self._Wt).tocsr()
                idx, sc = _sparse_row_topk(S, fetch)
                out_idx[s : s + chunk, :fetch] = idx
                out_score[s : s + chunk, :fetch] = sc

            workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
            if workers == 1 or len(starts) <= 1:
                iterator = (
                    tqdm(starts, desc="BM25: scoring queries", unit="chunk", disable=None)
                    if show_progress
                    else starts
                )
                for s in iterator:
                    _run_chunk(s)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = pool.map(_run_chunk, starts)
                    if show_progress:
                        results = tqdm(
                            results,
                            total=len(starts),
                            desc="BM25: scoring queries",
                            unit="chunk",
                            disable=None,
                        )
                    for _ in results:
                        pass
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
            "max_df_ratio": self.max_df_ratio,
            "n_docs": self.n_docs,
            "cv_kwargs": self.cv_kwargs,
            "vocabulary": {term: int(col) for term, col in self.vectorizer.vocabulary_.items()},
        }
        return arrays, meta

    @classmethod
    def from_state(cls, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> "BM25Index":
        # `.get` (rather than `[...]`): a directory saved before T32 has no
        # `max_df_ratio` key; absence must mean "off", the byte-identical
        # legacy behaviour, not a KeyError on load.
        obj = cls(meta["k1"], meta["b"], max_df_ratio=meta.get("max_df_ratio"), **meta["cv_kwargs"])
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
        max_block_elems: Optional[int] = None,
        n_jobs: int = 1,
    ):
        self._examples = example_bm25
        self._labels = example_labels.astype(np.int64)
        self._descriptions = desc_bm25
        self._k_chunk = k_chunk
        self._max_block_elems = max_block_elems
        self._n_jobs = n_jobs

    @classmethod
    def build(
        cls, texts: Sequence[str], labels: np.ndarray, label_space: LabelSpace, cfg: RetrievalConfig
    ) -> "LexicalRetrieverAdapter":
        ex = BM25Index(
            cfg.k1, cfg.b, max_df_ratio=cfg.bm25_max_df_ratio, **cfg.bm25_token_kwargs
        ).fit(texts)
        desc = BM25Index(
            cfg.k1, cfg.b, max_df_ratio=cfg.bm25_max_df_ratio, **cfg.bm25_token_kwargs
        ).fit(label_space.descriptions)
        return cls(
            ex, np.asarray(labels), desc, cfg.dense_chunk, cfg.bm25_max_block_elems, cfg.bm25_n_jobs
        )

    @classmethod
    def build_from_counts(
        cls,
        example_counts: sparse.csr_matrix,
        example_vectorizer: CountVectorizer,
        labels: np.ndarray,
        desc_bm25: BM25Index,
        cfg: RetrievalConfig,
    ) -> "LexicalRetrieverAdapter":
        """Build from an already-tokenized example corpus + a pre-built
        description index (T32 A1/A2). The training pipeline tokenizes the
        whole example pool and fits the description BM25 once per run, then
        reuses both here — once per fold, with ``example_counts`` a row slice
        of the full corpus, and once whole for the deployment index — instead
        of repeating ``CountVectorizer.fit_transform`` for every one of those
        calls. ``fit_from_counts`` recomputes document frequency and IDF from
        whatever rows it is given, so a fold's weights are still fold-local;
        only the tokenization is shared."""
        ex = BM25Index(cfg.k1, cfg.b, max_df_ratio=cfg.bm25_max_df_ratio, **cfg.bm25_token_kwargs)
        ex.fit_from_counts(example_counts, example_vectorizer)
        return cls(
            ex,
            np.asarray(labels),
            desc_bm25,
            cfg.dense_chunk,
            cfg.bm25_max_block_elems,
            cfg.bm25_n_jobs,
        )

    @classmethod
    def build_with_shared_descriptions(
        cls,
        texts: Sequence[str],
        labels: np.ndarray,
        desc_bm25: BM25Index,
        cfg: RetrievalConfig,
    ) -> "LexicalRetrieverAdapter":
        """Fit a fresh example index (the ordinary per-fold path, e.g. when
        ``bm25_token_kwargs`` prunes vocabulary by corpus statistics and
        ``build_from_counts``'s shared tokenization is unsafe — T32 A2's
        guard), but reuse a pre-built description index (T32 A1) rather than
        refitting it. A1 and A2 are independent: the description corpus is
        never row-sliced, so nothing about A2's per-fold-vocabulary concern
        applies to it — it is always safe to share, even when the example side
        must fall back to fitting fresh."""
        ex = BM25Index(
            cfg.k1, cfg.b, max_df_ratio=cfg.bm25_max_df_ratio, **cfg.bm25_token_kwargs
        ).fit(texts)
        return cls(
            ex,
            np.asarray(labels),
            desc_bm25,
            cfg.dense_chunk,
            cfg.bm25_max_block_elems,
            cfg.bm25_n_jobs,
        )

    def knn_example_labels(
        self, query_texts: Sequence[str], k: int, exclude_idx: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx, score = self._examples.top_k(
            query_texts, k, self._k_chunk, exclude=exclude_idx, n_jobs=self._n_jobs
        )
        labels = np.where(idx >= 0, self._labels[np.clip(idx, 0, None)], -1)
        return labels.astype(np.int64), score

    def description_score(self, query_texts: Sequence[str]) -> np.ndarray:
        return self._descriptions.score_matrix(query_texts, max_block_elems=self._max_block_elems)

    def with_added_descriptions(self, all_descriptions: Sequence[str]) -> "LexicalRetrieverAdapter":
        """Return a copy whose description BM25 is refit over ``all_descriptions``
        (the full, extended class-description list, existing classes first then
        the new ones).

        Used to widen a trained model's label space at inference time. BM25
        IDF is corpus-global, so the description side must be *refit* over every
        description, not have a row appended — hence the full list. The example
        index and its labels are reused verbatim: a class added this way is
        description-only (no example support), so nothing on the example side
        changes. The new BM25 keeps the same ``k1``/``b``/``max_df_ratio``/tokenizer
        kwargs as the original, so existing classes score identically."""
        old = self._descriptions
        desc = BM25Index(old.k1, old.b, max_df_ratio=old.max_df_ratio, **old.cv_kwargs).fit(
            all_descriptions
        )
        return LexicalRetrieverAdapter(
            self._examples, self._labels, desc, self._k_chunk, self._max_block_elems, self._n_jobs
        )

    def to_state(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Split this adapter into npz-able arrays and a JSON-clean meta dict,
        prefixing each BM25 sub-index's arrays so both pack into one npz file."""
        ex_arrays, ex_meta = self._examples.to_state()
        desc_arrays, desc_meta = self._descriptions.to_state()
        arrays: Dict[str, np.ndarray] = {f"examples_{k}": v for k, v in ex_arrays.items()}
        arrays.update({f"descriptions_{k}": v for k, v in desc_arrays.items()})
        arrays["example_labels"] = self._labels
        meta = {
            "examples": ex_meta,
            "descriptions": desc_meta,
            "k_chunk": self._k_chunk,
            "max_block_elems": self._max_block_elems,
            "n_jobs": self._n_jobs,
        }
        return arrays, meta

    @classmethod
    def from_state(
        cls, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]
    ) -> "LexicalRetrieverAdapter":
        ex_arrays = {
            k[len("examples_") :]: v for k, v in arrays.items() if k.startswith("examples_")
        }
        desc_arrays = {
            k[len("descriptions_") :]: v for k, v in arrays.items() if k.startswith("descriptions_")
        }
        ex = BM25Index.from_state(ex_arrays, meta["examples"])
        desc = BM25Index.from_state(desc_arrays, meta["descriptions"])
        # `.get`: a directory saved before T32/this change has no
        # `max_block_elems`/`n_jobs` key; absence must mean "unbounded"/"1",
        # the byte-identical legacy behaviour.
        return cls(
            ex,
            arrays["example_labels"],
            desc,
            meta["k_chunk"],
            meta.get("max_block_elems"),
            meta.get("n_jobs", 1),
        )


# ------------------------------------------------------------------- dense adapter
def _dense_topk(
    Q: np.ndarray, X: np.ndarray, k: int, chunk: int = 256, ops: Optional[ArrayOps] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k nearest examples by dot product, always shaped ``(n_queries, k)``.

    The result is padded to the requested ``k`` even when the corpus is smaller
    (``k > n_examples``): the first ``min(k, n)`` columns hold real neighbours in
    descending-similarity order and the remainder are ``-1`` / ``NaN`` padding,
    mirroring ``BM25Index.top_k``. An empty query batch returns ``(0, k)`` arrays.
    """
    ops = ops or NumpyArrayOps()
    n = X.shape[0]
    k_eff = min(k, n)
    out_idx = np.full((Q.shape[0], k), -1, dtype=np.int64)
    out_sim = np.full((Q.shape[0], k), np.nan, dtype=np.float32)
    if Q.shape[0] == 0 or k_eff == 0:
        return out_idx, out_sim
    Xt = np.ascontiguousarray(X.T)
    for s in range(0, Q.shape[0], chunk):
        sims = ops.matmul(Q[s : s + chunk], Xt)
        part = ops.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
        rows = np.arange(part.shape[0])[:, None]
        part_sims = ops.gather(sims, rows, part)
        order = ops.argsort(-part_sims, axis=1)
        out_idx[s : s + chunk, :k_eff] = np.take_along_axis(part, order, axis=1)
        out_sim[s : s + chunk, :k_eff] = np.take_along_axis(part_sims, order, axis=1)
    return out_idx, out_sim


def _prototypes_and_freq(
    emb: np.ndarray, labels: np.ndarray, n_classes: int, ops: Optional[ArrayOps] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class prototype (L2-normalized mean example embedding) and example
    count, over ``n_classes`` classes. A class with no examples gets an
    all-``NaN`` prototype row (XGBoost reads NaN as "missing"). Shared by
    ``DenseRetrieverAdapter.build`` (fresh) and ``with_added_examples`` (merged
    pool) so both compute prototypes identically.

    Replaces the ``for c in range(n_classes)`` masked-mean loop with a single
    ``scatter_add`` (class-sum) + norm: on a large label space that loop is
    thousands of masked reductions over the full embedding matrix, which is
    both the "no per-row Python loop" convention and, per T83, one of the
    largest CPU stages at high class count. The per-class sum is accumulated
    in float64 regardless of ``emb``'s dtype (matching ``scatter_add``'s
    contract), so results are numerically equal to the loop version to
    float32 precision, not bit-for-bit identical -- IEEE754 addition is not
    associative, and the loop's ``.mean(axis=0)`` sums each group in a
    different order (numpy's pairwise summation) than a flat scatter does."""
    ops = ops or NumpyArrayOps()
    dim = emb.shape[1]
    labels = np.asarray(labels)
    n = labels.shape[0]
    freq = (
        np.bincount(labels, minlength=n_classes).astype(np.int64)
        if n
        else np.zeros(n_classes, dtype=np.int64)
    )
    rows = np.repeat(labels, dim)
    cols = np.tile(np.arange(dim), n)
    values = np.asarray(emb, dtype=np.float64).ravel()
    class_sum = ops.to_host(
        ops.scatter_add(ops.zeros((n_classes, dim), dtype=np.float64), rows, cols, values)
    )
    counts = freq.astype(np.float64)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = class_sum / counts
    norm = np.linalg.norm(mean, axis=1)
    has_proto = (freq > 0) & (norm > 0)
    safe_norm = np.where(norm > 0, norm, 1.0)
    proto = np.where(has_proto[:, None], (mean / safe_norm[:, None]).astype(np.float32), np.nan)
    return proto.astype(np.float32), freq


@dataclass
class DenseState:
    """Serializable numeric state of the dense retriever."""

    example_emb: np.ndarray
    example_labels: np.ndarray
    prototypes: np.ndarray
    description_emb: np.ndarray
    class_freq: np.ndarray


class DenseRetrieverAdapter(DenseRetriever):
    def __init__(self, state: DenseState, chunk: int = 256, array_ops: Optional[ArrayOps] = None):
        self._s = state
        self._chunk = chunk
        self._ops = array_ops or NumpyArrayOps()

    @classmethod
    def build(
        cls,
        encoder: TextEncoder,
        texts: Sequence[str],
        labels: np.ndarray,
        label_space: LabelSpace,
        cfg: RetrievalConfig,
        array_ops: Optional[ArrayOps] = None,
    ) -> "DenseRetrieverAdapter":
        # The example pool and class descriptions are the *document* side of
        # retrieval; asymmetric encoders (E5/BGE prompts) encode them with the
        # document prompt so query embeddings land in the matching space.
        emb = encoder.encode_documents(texts)
        labels = np.asarray(labels)
        desc = encoder.encode_documents(label_space.descriptions)
        return cls.build_from_embeddings(emb, labels, desc, label_space, cfg, array_ops)

    @classmethod
    def build_from_embeddings(
        cls,
        example_emb: np.ndarray,
        labels: np.ndarray,
        description_emb: np.ndarray,
        label_space: LabelSpace,
        cfg: RetrievalConfig,
        array_ops: Optional[ArrayOps] = None,
    ) -> "DenseRetrieverAdapter":
        """Build from already-encoded document embeddings (T88): the encode step
        is a pure function of the text for a frozen shared encoder, so a caller
        that has already encoded the full pool/description set once (e.g. the
        training pipeline's per-fold loop) can slice and hand in embeddings
        instead of paying `encode_documents` again per fold. ``build`` still
        encodes internally and delegates here, so every existing caller and test
        double is untouched."""
        labels = np.asarray(labels)
        ops = array_ops or NumpyArrayOps()
        proto, freq = _prototypes_and_freq(example_emb, labels, label_space.size, ops)
        return cls(
            DenseState(example_emb, labels.astype(np.int64), proto, description_emb, freq),
            cfg.dense_chunk,
            ops,
        )

    @property
    def state(self) -> DenseState:
        return self._s

    @property
    def class_freq(self) -> np.ndarray:
        return self._s.class_freq

    def to_state(self) -> Dict[str, np.ndarray]:
        """Split this adapter into npz-able arrays (T34 phase 1: symmetric with
        ``LexicalRetrieverAdapter.to_state``, so a registered dense-retriever
        spec's save/load can be generic). Keys/layout match ``dense.npz`` as
        written by ``persistence.py`` before this method existed, byte-for-byte —
        existing model dirs load unchanged."""
        s = self._s
        return {
            "example_emb": s.example_emb,
            "example_labels": s.example_labels,
            "prototypes": s.prototypes,
            "description_emb": s.description_emb,
            "class_freq": s.class_freq,
        }

    @classmethod
    def from_state(
        cls,
        arrays: Dict[str, np.ndarray],
        chunk: int = 256,
        array_ops: Optional[ArrayOps] = None,
    ) -> "DenseRetrieverAdapter":
        return cls(
            DenseState(
                arrays["example_emb"],
                arrays["example_labels"],
                arrays["prototypes"],
                arrays["description_emb"],
                arrays["class_freq"],
            ),
            chunk,
            array_ops,
        )

    def knn_example_labels(
        self, query_emb: np.ndarray, k: int, exclude_idx: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Fetch one extra when self-masking so k real neighbours survive the drop.
        fetch = k + 1 if exclude_idx is not None else k
        idx, sim = _dense_topk(query_emb, self._s.example_emb, fetch, self._chunk, self._ops)
        if exclude_idx is not None:
            idx, sim = _exclude_self(idx, sim, exclude_idx, k)
        # idx == -1 marks padding (k > n_examples); keep it as -1 rather than
        # letting np indexing wrap around to a real label.
        labels = np.where(idx >= 0, self._s.example_labels[np.clip(idx, 0, None)], -1)
        return labels.astype(np.int64), sim

    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return self._ops.matmul(query_emb, self._s.prototypes.T)

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
        dim = E.shape[1]
        n = y.shape[0]
        scatter_rows = np.repeat(y, dim)
        scatter_cols = np.tile(np.arange(dim), n)
        class_sum = self._ops.to_host(
            self._ops.scatter_add(
                self._ops.zeros((C, dim), dtype=np.float64), scatter_rows, scatter_cols, E.ravel()
            )
        )
        class_cnt = np.bincount(y, minlength=C).astype(np.float64)

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
        return self._ops.matmul(query_emb, self._s.description_emb.T)

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
            return DenseRetrieverAdapter(s, self._chunk, self._ops)
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
        return DenseRetrieverAdapter(extended, self._chunk, self._ops)

    def with_updated_descriptions(
        self, encoder: TextEncoder, edits: dict
    ) -> "DenseRetrieverAdapter":
        """Return a copy with the description embeddings at ``edits``' class
        indices re-encoded: ``{class_index: new_description_text}``, for
        editing an *existing* class's description in place. A brand-new class's
        description is added via ``with_added_classes``, not this method. Every
        row not named in ``edits`` is untouched."""
        if not edits:
            return DenseRetrieverAdapter(self._s, self._chunk, self._ops)
        s = self._s
        idxs = list(edits.keys())
        new_rows = np.asarray(encoder.encode_documents([edits[i] for i in idxs]), dtype=np.float32)
        description_emb = s.description_emb.copy()
        description_emb[idxs] = new_rows
        updated = DenseState(
            s.example_emb, s.example_labels, s.prototypes, description_emb, s.class_freq
        )
        return DenseRetrieverAdapter(updated, self._chunk, self._ops)

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
        proto, freq = _prototypes_and_freq(merged_emb, merged_labels, n_classes, self._ops)
        updated = DenseState(merged_emb, merged_labels, proto, s.description_emb, freq)
        return DenseRetrieverAdapter(updated, self._chunk, self._ops)
