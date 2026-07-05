# T03 — Feature-assembly tests (numpy helpers + `assemble`)

status: done
tier: 1
depends_on: T01

## Goal
Test the feature engineering in `application/features.py` — first the pure numpy helpers in
isolation (exact values on tiny hand-built arrays), then the full `FeatureAssembler.assemble`
for schema, shape, NaN semantics, and `is_true` correctness.

## Why
This is the highest-risk module: subtle NaN/tie/empty semantics, fancy indexing, and the
"missing == NaN" convention all live here. A silent off-by-one or a broken mask corrupts
every downstream score with no error. These functions are pure → assert exact values.

## Files under test
`text_classifier/application/features.py`

## Part A — pure helpers (tiny hand-built arrays, exact assertions)

### `_scatter_knn(labels, scores, n_classes)`
- [ ] Basic aggregation: a `(2, 3)` labels/scores block where a class appears twice in a row
      → `ksum` is the sum, `kmax` is the max, `kcnt` is the count for that (row, class).
- [ ] **Missing semantics:** entries with `label < 0` or `NaN` score are ignored.
- [ ] **Empty class:** a class never retrieved for a row has `ksum == NaN`, `kmax == NaN`,
      `kcnt == 0` (sum/max NaN is the "not retrieved" encoding; count is a true 0).
- [ ] Mixed row where some entries valid, some `-1` padded → only valid ones counted.

### `_topn_mask(M, n, positive_only)`
- [ ] Selects exactly the top-n columns per row on a clean (no-tie) matrix.
- [ ] `n > C` is clamped to `C` (selects all finite columns).
- [ ] **NaN ranks last:** a row with NaNs never selects a NaN unless it runs out of finite
      values; all-NaN row selects nothing (mask all-False).
- [ ] `positive_only=True` drops columns `<= 0` even if they'd otherwise be top-n.
- [ ] **Ties:** with duplicate values at the cutoff, the mask may include slightly more than
      `n` (documented behavior) — assert it includes *all* tied-at-threshold columns, never
      fewer than `n` finite ones.

### `_row_rank(M, cand_mask)`
- [ ] Dense descending rank within the candidate set (1 = best); a known small matrix gives
      a known rank vector.
- [ ] Non-candidate columns (mask False) and NaNs are pushed to the worst ranks.

### `_row_minmax(M, cand_mask)`
- [ ] Min-max over candidates maps the row min→0 and max→1 for a known row.
- [ ] Degenerate row (all candidates equal) → uses range 1.0 (no div-by-zero); result finite.
- [ ] All-missing row → NaN preserved (no RuntimeWarning escapes; it's suppressed intentionally).

### `_argmax_or_missing(M, require_positive)`
- [ ] Returns the argmax column for a normal row.
- [ ] All-NaN row → returns `-1`.
- [ ] `require_positive=True` with a best value `<= 0` → returns `-1`.

## Part B — `FeatureAssembler.assemble` (use fixtures from T01)

Use the `hashing_encoder` + a small `synthetic_dataset`, build a `DenseRetrieverAdapter` and
`LexicalRetrieverAdapter` over a few items, then assemble features for a handful of queries.

- [ ] **Schema/order:** the produced DataFrame's feature columns equal `FEATURE_NAMES`
      *in order* (`list(df[FEATURE_NAMES].columns) == FEATURE_NAMES` and no extas beyond the
      bookkeeping cols `item_id`, `candidate`, and optional `is_true`).
- [ ] **dtype:** the 28 feature columns are float32.
- [ ] **`is_true` correctness:** when `query_labels` is provided, `is_true == 1` exactly on
      rows where `candidate == label[item]`, else 0. With labels omitted, no `is_true` column.
- [ ] **Candidate-row mapping:** for a query, the set of `candidate` values equals
      `np.nonzero(union_of_topn_masks)` for that row (reconstruct the mask and compare).
- [ ] **Empty-candidate path:** craft (or mock) signals so a query surfaces *no* candidates
      → assembler returns an empty frame with the right columns (no crash, no partial row).
- [ ] **Chunking equivalence:** `assemble(..., chunk=2)` and `assemble(..., chunk=10_000)`
      produce identical DataFrames (sort by `item_id, candidate` before comparing).
- [ ] **Derived features sanity:** `desc_proto_gap == d_desc_sim - d_proto_sim` per row;
      `class_log_freq == log1p(class_freq[candidate])`; `b_desc_missing` is 1 iff `b_desc_sim`
      is NaN. Assert these algebraically on the produced frame.
- [ ] **`n_signal_agreement`** is in `[0, 4]` and equals `5 - distinct_nonneg_argmaxes`.

## Acceptance criteria
- [ ] Every helper has at least the listed exact-value cases.
- [ ] `assemble` schema/order test guards `FEATURE_NAMES` drift (this is a load-bearing invariant).
- [ ] Chunking-equivalence test passes.
- [ ] `application/features.py` ≥ 95% line coverage.

## Out of scope
Retriever internals (T04). Leakage (T06).
