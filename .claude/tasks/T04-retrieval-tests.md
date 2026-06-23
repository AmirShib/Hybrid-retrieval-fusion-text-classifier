# T04 — Retrieval tests (BM25 + dense adapters)

status: todo
tier: 1
depends_on: T01

## Goal
Test `infrastructure/retrieval.py`: the `BM25Index` math against a hand-computed reference,
its `top_k` padding/ordering contract, and the `DenseRetrieverAdapter` (prototypes, NaN for
empty classes, kNN ordering, similarity matrices).

## Why
The BM25 precomputed-weight-matrix trick (`Q_binary @ W.T`) is a clever optimization that is
easy to get subtly wrong (IDF variant, length normalization, query-frequency-ignored). The
dense adapter's prototype/NaN handling feeds the "missing" convention. Both are deterministic
given inputs → assert exact or near-exact values.

## Files under test
`text_classifier/infrastructure/retrieval.py`

## Part A — `BM25Index`

### Math correctness
- [ ] Build a tiny corpus (≈4 short docs) and compute BM25 by hand for a query using the
      **Lucene IDF variant** `idf = log(1 + (N - df + 0.5)/(df + 0.5))` and
      `score = idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))`, **summed over distinct
      query terms only** (query term frequency ignored). Assert `score_matrix` matches to
      `~1e-5`.
- [ ] **Query-frequency ignored:** a query with a repeated term ("foo foo bar") scores the
      same as ("foo bar") — confirm `_query_incidence` binarizes (`q.data[:] = 1`).
- [ ] **IDF on rare vs common terms:** a term in every doc has lower (or non-positive) idf
      than a rare term; verify ordering of contributions.
- [ ] **Length normalization:** with `b=0`, doc length must not affect score; with `b=1`,
      it fully normalizes. Construct docs of different lengths and assert the expected shift.
- [ ] Out-of-vocabulary query terms contribute 0 (not an error).
- [ ] `cv_kwargs` plumbing: `stop_words="english"` actually removes stopwords from the vocab.

### `top_k` contract
- [ ] Returns `(idx (b,k) int, score (b,k) float)`; indices sorted by descending score.
- [ ] **Positive-only:** zero/negative scores become `-1` index + `NaN` score padding.
- [ ] `k > n_docs` → only `n_docs` real entries, the rest padded.
- [ ] A query with **no term overlap** → entire row is `-1` / `NaN`.
- [ ] **Chunking equivalence:** `top_k(..., chunk=1)` == `top_k(..., chunk=1000)`.
- [ ] Tie handling: equal-score docs don't crash and fill `k` slots deterministically enough
      that scores (not necessarily idx) match across chunk sizes.

## Part B — `LexicalRetrieverAdapter`
- [ ] `build` constructs separate BM25 indices over examples and over class descriptions.
- [ ] `knn_example_labels` maps doc indices → **class labels**, preserving `-1` padding
      (padded slots stay `-1`, not `labels[0]` — guard the `np.clip(idx,0,None)` masking).
- [ ] `description_score` returns a `(b, C)` matrix aligned to `LabelSpace` column order.

## Part C — `DenseRetrieverAdapter` (`build` + queries)
Use `hashing_encoder` from T01.
- [ ] **Prototypes:** a class's prototype is the L2-normalized mean of its example embeddings
      (assert norm ≈ 1 and direction matches the manual mean for a class with known members).
- [ ] **Empty class:** a class present in `LabelSpace` but with zero training examples has an
      all-`NaN` prototype row and `class_freq == 0`. `prototype_similarity` yields NaN for it.
- [ ] `class_freq` counts examples per class correctly (including the imbalanced fixture).
- [ ] `knn_example_labels`: neighbors sorted by descending cosine; returns `(labels, sims)`
      with `labels = example_labels[idx]`; `k > n_examples` is clamped.
- [ ] `description_similarity` / `prototype_similarity` return `(b, C)` cosine matrices; since
      embeddings are L2-normalized, values lie in `[-1, 1]` (allow tiny float slack).
- [ ] `_dense_topk` chunking equivalence (`chunk=1` vs large).

## Acceptance criteria
- [ ] At least one BM25 case asserts hand-computed scores to ~1e-5.
- [ ] Padding/`-1`/NaN contracts for `top_k` and `knn_example_labels` are explicitly asserted.
- [ ] Empty-class NaN prototype path is covered.
- [ ] `infrastructure/retrieval.py` ≥ 90% line coverage.

## Out of scope
The encoder adapter / fine-tuning (needs sentence-transformers; cover via the port + double).
Fusion (T05).
