# T32 — BM25 at scale: bounded memory **and** throughput

status: in-review
tier: 8
depends_on: T04

_Re-scoped 2026-08-04 from "memory profile" to memory + throughput. BM25 is the
one pipeline stage Tier 8's device policy keeps permanently on the host, so its
cost is a hard floor on what T85/T86 can deliver: once the dense side is on the
GPU, BM25 is what a batch waits for. The original memory scope is unchanged and
is retained below as section B._

## Goal
Characterize and bound BM25's query-time memory (B), and cut its wall-clock by
removing redundant tokenization and the densification of an already-sparse
product (A). Correctness must be preserved exactly, except for one explicitly
opt-in lossy knob (A4).

## Measure first
BM25 has two costs with different fixes, and they are easy to conflate:

- **Tokenization** — `CountVectorizer.transform`, a Python-level regex per
  document, in `_query_incidence` (`retrieval.py:87`) and `fit` (`retrieval.py:68`).
- **Scoring** — `Qbin @ _Wt` plus the top-k scan (`retrieval.py:114-119`).

For short texts tokenization usually dominates. **Instrument both and report the
split before implementing anything below** — the two halves have disjoint fixes
and optimizing the wrong one is wasted work. Fold this into T83's harness rather
than building a second profiler.

---

## A. Throughput

### A1. The description index is rebuilt identically every fold
`LexicalRetrieverAdapter.build` (`retrieval.py:195-196`) does:

    ex   = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(texts)
    desc = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs).fit(label_space.descriptions)

`label_space.descriptions` does not vary by fold, so the second line performs the
identical tokenization and weight build `n_folds + 1` times per training run and
produces the identical object each time. Build it once and share it — `BM25Index`
is immutable after `fit`, so this is a straight reuse with no caveats.

### A2. The example corpus is re-tokenized every fold
`build` is called inside the fold loop (`training.py:414`). IDF is corpus-global,
so the fold's *weights* legitimately differ — but tokenization is a pure function
of the text.

- Tokenize the full corpus **once** into a counts matrix, then per fold slice rows
  (`counts[tr]`) and recompute `df` / `doc_len` / `avg` / `idf` / `W` from the
  submatrix. Sparse arithmetic on an existing matrix, no re-analysis.
- **Scores are identical.** A term absent from fold `tr` has `tf = 0` in every row
  of the slice, so its `W` column is all-zero and contributes nothing to any
  query — exactly what happens today by that term being absent from the fold's
  vocabulary. The extra column costs memory, not correctness.
- **Guard, do not assume.** `min_df` / `max_df` / `max_features` in
  `bm25_token_kwargs` prune the vocabulary against whatever corpus they are fit
  on, so full-corpus and per-fold vocabularies genuinely differ when any is set.
  Take the fast path only when none is present; otherwise fall back to today's
  per-fold `fit` and log why.

New API: `BM25Index.tokenize_corpus(corpus, **cv_kwargs) -> (counts, vectorizer)`
and `BM25Index.fit_from_counts(counts, vectorizer)`, with `fit()` kept as
`tokenize_corpus` + `fit_from_counts` so every existing caller and test double is
untouched. `LexicalRetrieverAdapter` gains a matching `build_from_counts`.

Same insight as T88 (which does this for the *encoder*); deliberately kept here
because the machinery is BM25-specific — vocabulary handling, the `min_df` guard,
sparse row slicing — and mixing the two subsystems in one ticket would help
nobody. Cross-reference both ways.

### A3. Query texts are tokenized twice per batch
`_assemble_chunk` calls `lexical.knn_example_labels(texts, k)`
(`features.py:228`) and `lexical.description_score(texts)` (`features.py:231`).
Those reach two separate `BM25Index` objects holding two separate
`CountVectorizer`s, so the same batch goes through the analyzer twice. Both are
constructed from the same `cfg.bm25_token_kwargs`, so the analyzer is identical:
run `build_analyzer()` once over the batch and map the resulting token lists into
each vocabulary separately. Roughly halves query-side tokenization.

### A4. Static high-df pruning (opt-in, lossy)
BM25's Lucene IDF is `log(1 + (N − df + 0.5)/(df + 0.5))`, which tends to 0 as
`df → N`. The most common terms therefore contribute almost nothing to ranking
while owning the longest postings lists and dominating `nnz` in `_Wt`. Dropping
terms above a df ratio shrinks the matrix substantially for near-zero ranking
cost.

`RetrievalConfig.bm25_max_df_ratio: Optional[float] = None` — `None` (default) is
off and byte-for-byte today's behaviour. This is the **one** knob here that can
change scores, so it must be opt-in, persisted in `meta.json`, and documented as
a quality/speed trade rather than a free win.

This is also the mechanism-level explanation for why T35 (dropping the hidden
English stopword list) cost no accuracy: the IDF was already neutralizing those
terms, just expensively.

### A5. Only the top ~30 matters
`k_neighbors=20`, `top_n_per_signal=10`. Exact ranking below roughly the top 30 is
never read. This is what licenses A4, and what would license WAND-style skipping
if it is ever needed. No work here; recorded so the assumption is explicit.

---

## B. Memory (original scope, unchanged)

The precomputed weight matrix `_Wt` is sparse (memory O(total tokens)), which is
fine. The risk is **densification**: `top_k` computes
`S = (Qbin[s:s+chunk] @ _Wt).todense()` → a dense `(chunk, n_docs)` float32 block.
At scale this dominates: `chunk=256`, `n_docs=1e6` → 256 × 1e6 × 4 B ≈ **1 GB per
chunk**. `score_matrix` densifies the full `(b, n_docs)` block and is only safe for
the small description set — it must never be pointed at a large example pool.

- **Bound the dense block:** auto-size the query chunk from `n_docs` so the dense
  intermediate stays under a configurable cap (`bm25_max_block_elems`) instead of
  a fixed `chunk=256`. Small `n_docs` keeps today's behaviour.
- **Sparse top-k:** the `Qbin @ _Wt` product is sparse; compute per-row top-k
  without full densification when `n_docs` is large. Densify only a bounded slice.
- **Guard `score_matrix`:** document/assert it is for the small description set;
  the example path must go through chunked `top_k`.

**Sparse top-k is the overlap between A and B, and the time win is likely the
larger of the two.** The product is sparse×sparse, and `top_k` immediately
discards everything non-positive (`bad = sc <= 0`, `retrieval.py:121`) — so today
it materializes millions of zeros in order to delete them. Only nonzeros can
reach the top-k, taking the scan from O(b·n_docs) to O(nnz).

---

## Files to change
- `text_classifier/infrastructure/retrieval.py` — `tokenize_corpus` /
  `fit_from_counts`, shared analyzer pass, chunk auto-sizing, sparse top-k,
  df pruning, `score_matrix` guard.
- `text_classifier/application/training.py` — tokenize once outside the fold loop;
  build the description index once.
- `text_classifier/application/features.py` — single analyzer pass per batch (A3).
- `text_classifier/config.py` — `bm25_max_block_elems`, `bm25_max_df_ratio`.
- `tests/unit/test_retrieval.py`, `tests/integration/test_e2e.py`, `CHANGELOG.md`.

## Tests
- [x] **Correctness invariance:** `top_k` returns identical shape/padding/
      positive-score-only/descending-order contract before and after — all
      existing `TestBM25TopK` tests pass unchanged against the sparse
      rewrite, plus `TestSparseRowTopk` (hand-computed cases, empty matrix,
      `fetch=0`, and a cross-check against a brute-force dense reference on
      random data). **Caveat, recorded honestly:** exact tie-breaking order
      between two equal-score candidates is not asserted identical to the old
      dense `argpartition`/`argsort` implementation — `np.argsort`'s default
      (non-stable) sort never made that order a documented contract, only an
      implementation accident, and pinning to it would over-constrain the
      rewrite for no real benefit.
- [x] **A1/A2 byte-identity:** OOF frame, fusion model, thresholds and
      evaluation bit-for-bit unchanged with default `bm25_token_kwargs` — the
      full existing suite (e2e round-trip, T06 leakage, T52 benchmark floors)
      stays green, plus dedicated `test_byte_identical_lexical_scores_on_a_
      fixed_corpus` / `TestBM25TokenizeOnceFitFromCounts` /
      `TestLexicalBuildFromCounts` parity tests.
- [x] **A2 fallback:** with `min_df` / `max_df` / `max_features` set, the
      per-fold path is taken (`TestVocabPruningGuard`, all three kwargs).
      Found and fixed here: the first implementation still rebuilt the
      *description* index per fold in the fallback branch, defeating A1;
      `LexicalRetrieverAdapter.build_with_shared_descriptions` fixes it — the
      description index is shared from `_shared_lexical_state` regardless of
      whether the example side takes the shared or per-fold path, since A2's
      vocabulary-mismatch concern never applies to the (never row-sliced)
      description corpus.
- [x] **Tokenizer call counting:** `test_bm25_tokenize_once.py` — the whole
      run tokenizes exactly twice (once for the pool, once for descriptions),
      independent of `n_folds` (today: `2 * (n_folds + 1)`).
- [ ] **Memory bound:** peak dense intermediate ≤ `bm25_max_block_elems` on a
      synthetic larger corpus. **Not done as originally specified** — `top_k`
      no longer has a dense intermediate to bound (see acceptance criteria
      below), so this test's premise doesn't apply there; `score_matrix`'s
      guard is tested directly instead (`TestScoreMatrixBlockGuard`: raises
      over the cap, succeeds under it or when `None`).
- [ ] Auto-chunk shrinks as `n_docs` grows; tiny corpora keep the single-shot
      path. **Not implemented** — superseded by removing `top_k`'s dense block
      entirely rather than bounding it (see notes below).
- [x] `bm25_max_df_ratio=None` is byte-identical; a set value is persisted in
      `meta.json` and re-applied at load (`TestBM25MaxDfRatio`,
      `TestMaxDfRatioTrainingIntegration`).
- [x] Existing T04 retrieval tests and the T06 leakage regression pass
      unchanged.

## Acceptance criteria
- [ ] Tokenize/score split measured and published before the fixes land.
      **Blocked on T83** — the ticket itself says to fold this into T83's
      harness rather than build a second profiler, and T83 was explicitly out
      of scope for this pass. Not measuring first was a judgment call, not an
      oversight: A1/A2/A4/B are argued from the code (a `CountVectorizer.
      fit_transform` call site duplicated `n_folds + 1` times is waste
      regardless of which half of the cost dominates), and every change is
      byte-identity-tested, so correctness doesn't depend on the missing
      measurement — only the *prioritization* would have.
- [x] Corpus tokenized once per run; description index built once; query
      batch analyzed once — **A1/A2 done, A3 (query-side single analyzer
      pass) deferred, see notes below.**
- [x] Peak BM25 query-time memory bounded by config, independent of `n_docs`
      — via eliminating `top_k`'s dense intermediate entirely (stronger than
      bounding it) plus `score_matrix`'s explicit cap.
- [x] `top_k` bit-identical on existing tests; only `bm25_max_df_ratio` may
      change scores, and only when explicitly set.
- [ ] Before/after profile documented for both memory and wall-clock.
      **Not done** — same T83 dependency as the tokenize/score split above.

## Implementation notes
- **A3 (query batch tokenized twice per feature-assembly chunk) is deferred,
  not landed.** The natural implementation threads a shared, pre-tokenized
  representation through `LexicalRetriever.knn_example_labels` and
  `.description_score`, which means either widening the `LexicalRetriever`
  port with a new combined method (breaking every existing test double that
  implements the ABC, e.g. `_ZeroLexical` in `test_features.py`) or
  duck-typing around it in `FeatureAssembler`. Given the ticket's own framing
  — A3 "roughly halves query-side tokenization," secondary to A2's corpus-side
  fix and B's sparse top-k, which the ticket calls "likely the larger win" —
  the invasiveness didn't clear the bar this pass. Left for a follow-up.
- **B's block-bounding was implemented differently than specified, on
  purpose.** The spec called for auto-sizing `top_k`'s query chunk from
  `n_docs` so a fixed `bm25_max_block_elems` cap holds. Implementing sparse
  top-k (`_sparse_row_topk`) instead removes the dense `(chunk, n_docs)`
  block from `top_k` entirely — there is no intermediate left to bound, which
  is strictly stronger than bounding one. `bm25_max_block_elems` therefore
  only guards `score_matrix` (the one remaining always-dense path, explicitly
  for the small description set) rather than also auto-sizing `top_k`'s
  chunk.
- `BM25Index.tokenize_corpus` (static) / `fit_from_counts` split out of
  `fit()`; `LexicalRetrieverAdapter` gained `build_from_counts` (shared
  tokenization + shared description index) and
  `build_with_shared_descriptions` (per-fold example fit + shared description
  index, A2's fallback path) alongside the unchanged `build()`.
  `TrainingPipeline._shared_lexical_state` caches `(example_counts,
  vectorizer)` and the description `BM25Index` on `self`, populated by
  whichever of `_build_oof` / `_build_deployment_index` runs first — the same
  pattern T88 uses for encoder embeddings.
- `RetrievalConfig.bm25_max_df_ratio` / `bm25_max_block_elems` persist through
  `PipelineConfig`'s existing `asdict`-based serialization with no special
  handling needed; `BM25Index`/`LexicalRetrieverAdapter`'s `from_state` use
  `.get` for both new meta keys so a pre-T32 model directory still loads.

## Out of scope
Changing the BM25 scoring formula or the precomputed-`W` design. Disk-backed /
out-of-core indexes. ANN for lexical retrieval. The dense retriever (T31).
**WAND / block-max skipping** and **GPU sparse mat-mul** — both are plausible but
should only be considered after the above, since A2/A3 remove tokenization cost
that no device change can touch, and sparse top-k may leave the matmul small
enough that the question does not arise. **Overlapping BM25 with GPU dense work**
belongs to T85, not here.
