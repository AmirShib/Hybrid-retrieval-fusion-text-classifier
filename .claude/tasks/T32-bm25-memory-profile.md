# T32 — BM25 memory profile for large corpora; chunk/sparsify as needed

status: todo
tier: 3
depends_on: T04

## Goal
Characterize and **bound** BM25 peak query-time memory for large example corpora,
then fix the one place that does not scale: the dense `(chunk, n_docs)` score
block materialized inside `BM25Index.top_k` / `score_matrix`. Correctness must be
preserved exactly.

## Why
The precomputed weight matrix `_Wt` is sparse (memory O(total tokens)), which is
fine. The risk is **densification**: `top_k` computes
`S = (Qbin[s:s+chunk] @ _Wt).todense()` → a dense `(chunk, n_docs)` float32 block.
At scale this dominates: e.g. `chunk=256`, `n_docs=1e6` → 256 × 1e6 × 4 B ≈
**1 GB per chunk**. `score_matrix` densifies the full `(b, n_docs)` block and is
only safe for the small description set — it must never be pointed at a large
example pool.

## Approach
- **Profile first:** add a small benchmark (script or test) that measures the
  intermediate block size / peak as `n_docs` and `chunk` grow, to document the
  blow-up and validate the fix.
- **Bound the dense block:** auto-size the query chunk from `n_docs` so the dense
  intermediate stays under a configurable cap (`bm25_max_block_elems`) instead of
  a fixed `chunk=256`. Small `n_docs` keeps today's behaviour.
- **Prefer sparse top-k where it pays:** the `Qbin @ _Wt` product is sparse;
  compute per-row top-k without full densification (operate on the sparse `csr`
  rows) when `n_docs` is large. Densify only a bounded slice.
- **Guard `score_matrix`:** document/assert it is for the small description set;
  the example path must go through chunked `top_k`.

## Files to change
- `text_classifier/infrastructure/retrieval.py` — chunk auto-sizing + optional
  sparse top-k in `BM25Index.top_k`; guard/doc on `score_matrix`.
- `text_classifier/config.py` — `RetrievalConfig.bm25_max_block_elems` (cap).
- `tests/unit/test_retrieval.py` — correctness + memory-bound tests; a small
  offline profiling helper.

## Tests
- [ ] **Correctness invariance:** `top_k` returns identical `(idx, score)` (incl.
      `-1`/`NaN` padding, positive-score-only filter, descending order) before and
      after the change, across `k < n`, `k > n`, and empty batch.
- [ ] **Memory bound:** on a synthetic larger corpus the peak dense intermediate
      ≤ `bm25_max_block_elems` (assert via instrumented block size, not wall-clock
      RSS, for determinism).
- [ ] Auto-chunk picks a smaller chunk as `n_docs` grows; tiny corpora keep the
      original single-shot path.
- [ ] Existing T04 retrieval tests pass unchanged.

## Acceptance criteria
- [ ] Peak BM25 query-time memory is bounded by config, independent of `n_docs`.
- [ ] `top_k` output is bit-identical to the current implementation on existing
      tests.
- [ ] A documented profile shows the before/after memory behaviour.

## Out of scope
Changing the BM25 scoring formula or the precomputed-`W` design. Disk-backed /
out-of-core indexes. ANN for lexical retrieval. The dense retriever (T31).
