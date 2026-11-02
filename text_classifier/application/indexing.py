"""Retrieval-index construction for one training run (application service).

``TrainingPipeline`` needs dense + lexical retrieval state over two different
slices of the same corpus: each fold's *training* rows (the leakage-free
out-of-fold loop) and, separately, *all* rows (the deployment index that ships
in the model and scores any external val/test split). Those are the same
construction under a different row selection, but several optimizations make
the "how" conditional:

- **T88** — a frozen shared encoder's ``encode_documents`` is a pure function of
  the text, so the whole example pool and every class description are encoded
  *once per run* and sliced per fold, instead of re-encoded per fold.
- **T89** — held-out items may then take their *query* embeddings from that same
  cache rather than being encoded a second time, when (and only when) the
  encoder treats the two roles identically.
- **T32 A1/A2** — BM25 tokenization is independent of the encoder, so the
  example corpus is tokenized once and the class-description index built once,
  both reused across folds. A2's guard: vocabulary-pruning kwargs make
  full-corpus and per-fold vocabularies genuinely differ, so the example side
  falls back to a per-fold refit.
- **T34 phase 1** — all of the above reach into the *built-in* adapters'
  internals, so a non-default ``dense_kind``/``lexical_kind`` goes through the
  plain registry build and loses the sharing.

Holding that policy in one object is the point. It was previously spelled out
twice — once in the fold loop, once in the deployment build — and the two had to
be kept in agreement by hand; a backend that changed one and not the other would
silently train the fusion model on features from a differently-built index than
the one it ships against. Here the caches *are* the object's state, so "shared
once per run" is an invariant rather than a convention, and each caller just
names the rows it wants.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ..config import PipelineConfig
from ..domain import ArrayOps, DenseRetriever, LabelSpace, LexicalRetriever, TextEncoder
from ..infrastructure import (
    BM25Index,
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
    bm25_prunes_vocab,
    build_dense_retriever,
    build_lexical_retriever,
)

log = logging.getLogger(__name__)

# `rows=None` throughout this module means "the whole corpus", distinct from an
# empty selection. Slicing helpers below are the only place that distinction is
# interpreted.
Rows = Optional[np.ndarray]


class RetrievalIndexBuilder:
    """Builds dense + lexical retrieval state over row slices of one corpus,
    reusing whatever can legitimately be shared across those slices.

    Construct one per ``TrainingPipeline.run`` call and keep it for the whole
    run: the caches are keyed to *this* corpus and *this* encoder, and are
    populated lazily by whichever caller asks first (the fold loop on the k-fold
    path, the deployment build on the leave-one-out path).

    ``shared_encoder`` is the frozen encoder used for every slice, or ``None``
    for the per-fold-encoder mode (a fine-tuned or corpus-dependent encoder,
    where a fold's embeddings are stale the moment the next fold refits). In
    that mode no embedding cache is ever populated and every dense build goes
    through the ordinary per-slice path; the *lexical* caches still apply, since
    BM25 tokenization does not depend on the encoder.
    """

    def __init__(
        self,
        config: PipelineConfig,
        label_space: LabelSpace,
        texts: Sequence[str],
        labels: np.ndarray,
        array_ops: ArrayOps,
        *,
        shared_encoder: Optional[TextEncoder] = None,
    ):
        self._cfg = config
        self._space = label_space
        self._texts = list(texts)
        self._y = labels
        self._ops = array_ops
        self._shared_encoder = shared_encoder

        retrieval = config.retrieval
        # Whether each side's sharing optimizations apply at all (T34 phase 1:
        # they are built-in-adapter-specific). Computed once here rather than
        # re-derived at each build site — that duplication is what this class
        # exists to remove.
        self._dense_shareable = shared_encoder is not None and retrieval.dense_kind == "exact"
        self.lexical_enabled = "lexical" in config.signals
        self._lexical_shareable = self.lexical_enabled and retrieval.lexical_kind == "bm25"

        # T88 caches: the full example-pool and class-description embeddings.
        # `Any`, not `np.ndarray`: a resident torch tensor when the run resolved
        # the torch backend (T85). Both are set together or not at all.
        self._pool_emb: Any = None
        self._desc_emb: Any = None
        # T32 A1/A2 caches: the example-corpus tokenization and the description
        # BM25 index.
        self._example_counts: Optional[sparse.csr_matrix] = None
        self._example_vectorizer: Optional[CountVectorizer] = None
        self._desc_bm25: Optional[BM25Index] = None

    # ------------------------------------------------------------- introspection
    @property
    def pool_embeddings_cached(self) -> bool:
        """Whether the corpus has been encoded once and cached (T88). ``False``
        on the per-fold-encoder path and for a non-built-in ``dense_kind``,
        where every slice encodes for itself."""
        return self._pool_emb is not None

    @property
    def example_counts_cached(self) -> bool:
        """Whether the example corpus was tokenized once and cached (T32 A2).
        ``False`` when vocabulary-pruning kwargs forced the per-fold fallback."""
        return self._example_counts is not None

    @property
    def description_index_cached(self) -> bool:
        """Whether the class-description BM25 index is shared across slices
        (T32 A1). Independent of ``example_counts_cached``: A2 can fall back
        while A1 still applies, since the description index is never row-sliced."""
        return self._desc_bm25 is not None

    # ------------------------------------------------------------------ slicing
    def _texts_for(self, rows: Rows) -> List[str]:
        return self._texts if rows is None else [self._texts[i] for i in rows]

    def _labels_for(self, rows: Rows) -> np.ndarray:
        return self._y if rows is None else self._y[rows]

    # ------------------------------------------------------------------ caches
    def _document_embeddings(self) -> Tuple[Any, Any]:
        """The full example-pool and class-description embeddings, encoded once
        per run (T88). Valid only under a frozen shared encoder — guarded by
        ``_dense_shareable`` at every call site."""
        if self._pool_emb is None:
            encoder = self._shared_encoder
            assert encoder is not None  # guaranteed by `_dense_shareable`
            self._pool_emb = encoder.encode_documents(self._texts)
            self._desc_emb = encoder.encode_documents(self._space.descriptions)
        return self._pool_emb, self._desc_emb

    def _lexical_state(
        self,
    ) -> Tuple[Optional[Tuple[sparse.csr_matrix, CountVectorizer]], BM25Index]:
        """The example-corpus tokenization + description BM25 index, built once
        per run (T32 A1/A2).

        The description index is always cached and reused verbatim: the class
        descriptions do not vary by slice and ``BM25Index`` is immutable after
        ``fit``. The example state is ``None`` when ``bm25_token_kwargs`` prunes
        vocabulary by corpus statistics, because a full-corpus vocabulary and a
        per-fold one genuinely differ then (A2's guard).
        """
        cfg = self._cfg.retrieval
        if self._desc_bm25 is None:
            self._desc_bm25 = BM25Index(
                cfg.k1, cfg.b, max_df_ratio=cfg.bm25_max_df_ratio, **cfg.bm25_token_kwargs
            ).fit(self._space.descriptions)
            if not bm25_prunes_vocab(cfg.bm25_token_kwargs):
                self._example_counts, self._example_vectorizer = BM25Index.tokenize_corpus(
                    self._texts, **cfg.bm25_token_kwargs
                )
        example_state = (
            None
            if self._example_counts is None
            else (self._example_counts, self._example_vectorizer)
        )
        return example_state, self._desc_bm25

    # ------------------------------------------------------------------- public
    def build(
        self, encoder: TextEncoder, rows: Rows = None
    ) -> Tuple[DenseRetriever, Optional[LexicalRetriever]]:
        """The dense and lexical retrieval state over ``rows`` (``None`` = the
        whole corpus), built with ``encoder`` wherever an encode is actually
        needed.

        ``encoder`` is the encoder for *this* slice: the shared one on the
        frozen path, this fold's freshly fitted one on the per-fold path. It is
        unused when the embedding cache applies, since the cache was produced by
        the shared encoder and slicing it is what T88 buys.

        The lexical half is ``None`` when ``"lexical"`` is not a configured
        signal — nothing downstream queries it, so tokenizing the corpus and
        fitting a weight matrix for it would be pure waste.
        """
        return self._build_dense(encoder, rows), self._build_lexical(encoder, rows)

    def _build_dense(self, encoder: TextEncoder, rows: Rows) -> DenseRetriever:
        if self._dense_shareable:
            pool_emb, desc_emb = self._document_embeddings()
            return DenseRetrieverAdapter.build_from_embeddings(
                pool_emb if rows is None else pool_emb[rows],
                self._labels_for(rows),
                desc_emb,
                self._space,
                self._cfg.retrieval,
                self._ops,
            )
        return build_dense_retriever(
            self._cfg.retrieval,
            encoder,
            self._texts_for(rows),
            self._labels_for(rows),
            self._space,
            self._ops,
        )

    def _build_lexical(self, encoder: TextEncoder, rows: Rows) -> Optional[LexicalRetriever]:
        if not self.lexical_enabled:
            return None
        cfg = self._cfg.retrieval
        labels = self._labels_for(rows)
        if not self._lexical_shareable:
            return build_lexical_retriever(cfg, self._texts_for(rows), labels, self._space)
        example_state, desc_bm25 = self._lexical_state()
        if example_state is not None:
            counts, vectorizer = example_state
            return LexicalRetrieverAdapter.build_from_counts(
                counts if rows is None else counts[rows], vectorizer, labels, desc_bm25, cfg
            )
        # A2's guard fell back (vocab-pruning kwargs): the example side must
        # refit per slice, but the description index (A1) is still shared — it
        # is never row-sliced, so nothing about A2's concern applies to it.
        return LexicalRetrieverAdapter.build_with_shared_descriptions(
            self._texts_for(rows), labels, desc_bm25, cfg
        )

    def query_embeddings(self, encoder: TextEncoder, rows: Rows = None) -> Any:
        """Query embeddings for ``rows``, taken from the shared document cache
        when that is legitimate (T89) and encoded otherwise.

        The cache holds ``encode_documents`` output. Substituting it for
        ``encode_queries`` is only correct when the encoder treats the two roles
        identically — with E5/BGE-style prompts configured it is a correctness
        bug, not an optimization, so the default is to reuse only when the
        encoder advertises role symmetry (see ``_may_reuse``).

        Row order is preserved: the cache is indexed with the same ``rows`` used
        to slice the pool for the retriever, so this returns exactly the rows a
        re-encode would have produced, in the same order — and it keeps working
        when the cache holds a resident torch tensor rather than numpy (T85).
        """
        if self._pool_emb is not None and self._may_reuse(encoder):
            return self._pool_emb if rows is None else self._pool_emb[rows]
        return encoder.encode_queries(self._texts_for(rows))

    def _may_reuse(self, encoder: TextEncoder) -> bool:
        """Whether ``encoder``'s document embeddings may stand in as query
        embeddings (T89).

        Policy only — the correctness precondition (that a frozen shared encoder
        produced the cache) is structural: the cache is populated exclusively
        under ``_dense_shareable``, which requires a shared encoder, so every
        mode below is a no-op on the per-fold path.

        The capability is probed with ``getattr``, not ``isinstance``, and
        **absence means "do not reuse"**: a custom ``TextEncoder`` supplied
        through the port may distinguish the two roles internally without going
        through ``EncoderConfig``'s prompt fields, and reading those fields
        directly would silently hand it document vectors where it expects query
        vectors. Conservative by default; opt in with ``"always"``.
        """
        mode = self._cfg.encoder.reuse_query_embeddings
        if mode == "never":
            return False
        symmetric = bool(getattr(encoder, "roles_share_encoding", False))
        if mode == "always":
            if not symmetric:
                log.warning(
                    "encoder.reuse_query_embeddings='always' is reusing document embeddings "
                    "as query embeddings even though %s does not advertise role-symmetric "
                    "encoding. If this encoder really does encode queries and documents "
                    "differently (e.g. E5/BGE-style prompts), the reused vectors are wrong "
                    "and every retrieval signal built on them is invalid.",
                    type(encoder).__name__,
                )
            return True
        return symmetric
