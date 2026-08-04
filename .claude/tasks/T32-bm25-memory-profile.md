# T32 — BM25 at scale: bounded memory **and** throughput

status: todo
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
- [ ] **Correctness invariance:** `top_k` returns identical `(idx, score)` (incl.
      `-1`/`NaN` padding, positive-score-only filter, descending order) before and
      after, across `k < n`, `k > n`, and empty batch.
- [ ] **A1/A2 byte-identity:** OOF frame, fusion model, thresholds and evaluation
      bit-for-bit unchanged with default `bm25_token_kwargs`.
- [ ] **A2 fallback:** with `min_df` / `max_df` / `max_features` set, the per-fold
      `fit` path is taken and results match today exactly.
- [ ] **Tokenizer call counting:** a counting analyzer double asserts one corpus
      tokenization per run (today: `n_folds + 1`) and one query pass per batch
      (today: two).
- [ ] **Memory bound:** peak dense intermediate ≤ `bm25_max_block_elems` on a
      synthetic larger corpus (assert instrumented block size, not RSS).
- [ ] Auto-chunk shrinks as `n_docs` grows; tiny corpora keep the single-shot path.
- [ ] `bm25_max_df_ratio=None` is byte-identical; a set value is persisted in
      `meta.json` and re-applied at load.
- [ ] Existing T04 retrieval tests and the T06 leakage regression pass unchanged.

## Acceptance criteria
- [ ] Tokenize/score split measured and published before the fixes land.
- [ ] Corpus tokenized once per run; description index built once; query batch
      analyzed once.
- [ ] Peak BM25 query-time memory bounded by config, independent of `n_docs`.
- [ ] `top_k` bit-identical on existing tests; only `bm25_max_df_ratio` may change
      scores, and only when explicitly set.
- [ ] Before/after profile documented for both memory and wall-clock.

## Out of scope
Changing the BM25 scoring formula or the precomputed-`W` design. Disk-backed /
out-of-core indexes. ANN for lexical retrieval. The dense retriever (T31).
**WAND / block-max skipping** and **GPU sparse mat-mul** — both are plausible but
should only be considered after the above, since A2/A3 remove tokenization cost
that no device change can touch, and sparse top-k may leave the matmul small
enough that the question does not arise. **Overlapping BM25 with GPU dense work**
belongs to T85, not here.
