# T76 — Numpy-only inference path: drop pandas from the hot loop (and the serving install)

status: todo
tier: 7
depends_on: T34, T75

**Gated: do not start before T34 lands.** T34 reshapes `FeatureAssembler` and pays
for a golden-output test harness; doing this refactor first would mean paying for
that harness twice on the same code.

## Goal
Make `InferencePipeline.predict` run without pandas, so (a) the hot path drops
DataFrame construction/groupby overhead, and (b) an inference-only install can be
`numpy + scipy + scikit-learn + xgboost` — meaningfully leaner for the air-gapped
serving hosts this package targets. Pandas stays a dependency for training and
the CLIs (moves to an extra only if the numbers justify it — measure first).

## Why
Pandas is not just at the boundary; it is load-bearing on the hot path — and for
bookkeeping, not numerics:
- `FeatureAssembler._assemble_chunk` builds a `pd.DataFrame` per chunk and
  `assemble` concatenates them (`application/features.py:119, 207-212`); every
  numeric step before that is already pure numpy.
- `top_per_item` does pandas `sort_values` / `groupby.head(1)` / `.nth(1)` /
  `merge` (`application/scoring.py:23-33`) — expressible as
  `np.lexsort` + `np.unique(..., return_index=True)` + index arithmetic.
- `add_confidence` only extracts `to_numpy` and appends a column.

So the inference-side pandas usage is: group rows by `item_id`, take the top-1
and runner-up per group. That's ~30 lines of numpy. The wins: no per-batch
DataFrame allocation (matters at small-batch/online latency), and no pandas on
the serving host (pandas is the single heaviest non-ML dependency in the core
install).

## Design
- Introduce a plain **struct-of-arrays** feature block (a small frozen dataclass:
  `features: (n, F) float32`, `item_id: (n,) intp`, `candidate: (n,) int64`,
  optional `is_true`) as the internal currency of assembly → scoring → decision.
- `FeatureAssembler` produces the block natively (it already has the arrays —
  the DataFrame is constructed *from* them today). A thin `to_frame()` keeps the
  DataFrame available where it is genuinely wanted (training's OOF table with
  its `fold` column and `groupby` conveniences, T69's explain slicing).
- Numpy `top_per_item` / `top_k_per_item` (T65) on the block:
  `order = np.lexsort((-conf, item_id))` then segment boundaries via
  `np.unique(item_id[order], return_index=True)`; runner-up = boundary+1 where
  the segment has ≥ 2 rows. Must reproduce current tie-breaking (stable sort —
  pandas `sort_values(kind="stable")` semantics) exactly.
- Training keeps pandas (OOF concat, fold bookkeeping, calibration slicing are
  not hot and the code reads well); it consumes the same blocks via `to_frame()`.
- Import hygiene: `application/inference.py` and its transitive imports must not
  import pandas at module level; add a test that constructs `InferencePipeline`
  and predicts with pandas blocked (`sys.modules["pandas"] = None`-style guard).
- Only after all that lands and is measured: consider `pandas` → an extra with a
  lazy import error for CLI/training use. Separate decision, own changelog note.

## The bar to clear (why this is ranked last)
This is a wide refactor of correct, vectorized, well-tested code for a moderate
win. It must ship with: the T34 golden-output test proving byte-identical
features, a decision-level golden test (identical `Prediction`s on a fixed
corpus), and a before/after benchmark (batch-1 and batch-10k latency; install
footprint with/without pandas). If the measured win is small, close as wontfix
with the numbers attached — that's a valid outcome.

## Files to change
- `text_classifier/application/features.py` — emit the block; `to_frame()`.
- `text_classifier/application/scoring.py` — numpy top-1/top-k/runner-up.
- `text_classifier/application/inference.py` — consume blocks end-to-end.
- `text_classifier/application/training.py` — adapt via `to_frame()`.
- `tests/` — golden decision test, pandas-free import test, tie-break tests.

## Tests
- [ ] Golden: `predict` output (all `Prediction` fields) identical before/after
      on a fixed corpus, including tie cases (equal confidences) and
      single-candidate items.
- [ ] `top_per_item` numpy vs pandas implementations agree on randomized inputs
      (property test, including duplicate conf values).
- [ ] Inference works with pandas import blocked; training still works with it.
- [ ] T65/T69 paths (top-k, explain) still correct against the block/`to_frame`.

## Acceptance criteria
- [ ] Zero pandas imports on the predict path; identical decisions everywhere.
- [ ] Benchmark table in the PR (latency + install size); a wontfix-with-numbers
      close is acceptable if the win doesn't materialize.
- [ ] Training-side behaviour and artifacts byte-identical.

## Out of scope
Making pandas optional in this ticket (follow-up decision, needs the numbers);
touching retrieval/encoder internals (already numpy); polars as an internal
engine (protocols at the boundary — T75 — not a new internal dependency).
