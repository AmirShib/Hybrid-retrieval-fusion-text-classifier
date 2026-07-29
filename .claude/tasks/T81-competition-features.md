# T81 — Competition features: per-candidate margins + per-query top1−top2 gap

status: in-review
tier: 3
depends_on: T03

## Goal
Give the fusion model the one thing a pointwise ranker structurally cannot infer:
how a candidate compares to *the rest of its own query*. Eight new core columns —
five `margin_*` (per candidate) and three `q_gap_*` (per query).

## Why
The fusion model scores one (item, candidate) row at a time, independently. Any
fact about the competition inside a query has to be written into the row or the
model never sees it. The pre-existing schema does part of this — `rank_*` and the
min-max `norm_*` — but had no notion of *distance* to a rival:

| | item A | item B |
|---|---|---|
| `d_desc_sim` (leader) | 0.85 | 0.85 |
| runner-up | 0.84 | 0.40 |
| `rank_d_desc` | 1 | 1 |
| `is_d_desc_top1` | 1 | 1 |
| `abs_top_dense_sim` | equal | equal |

A is a coin flip, B is decided, and **every column above is identical**. The only
column carrying any of the difference was `norm_d_desc`, which is min-max — so it
is anchored to the *worst* candidate in the set and gets rescaled by any single
outlier, rather than measuring the contest at the top.

This matters most for the thing the package is built around: calibrated
abstention. Top1−top2 is the classic confidence signal, and it was absent.

Secondary reason: trees are axis-aligned. A gradient-boosted model cannot cheaply
represent `a − b` from two columns — it needs a deep staircase of splits to
approximate one. Handing it the subtraction directly is a real capacity win at
the data scale this package targets.

## The columns

Appended to `domain/services.py::FEATURE_NAMES` (28 → 36). Appended, not
interleaved, so the diff to the persisted order is a pure suffix.

**`margin_d_desc`, `margin_d_proto`, `margin_d_knn`, `margin_b_desc`,
`margin_b_knn`** — the candidate's value for that signal minus the best *other*
candidate's value. The signal's leader gets `top1 − top2` (positive: its winning
margin); everyone else gets `value − top1` (non-positive: its deficit). One
column expresses both "am I winning" and "by how much", and it is invariant to a
query-wide offset — which is exactly the hubness effect that makes raw cosines
incomparable across items.

**`q_gap_d_desc`, `q_gap_d_knn`, `q_gap_b_desc`** — the per-query top1−top2 gap,
constant across an item's rows. Redundant with `margin_*` on the leader's row
(they are equal there, and there is a test pinning that), but *new* information
on every other row: a trailing candidate's own margin says how far back it is,
not whether the lead ahead of it is contested.

Only three `q_gap_*`, not five: proto and BM25-kNN gaps are the weakest and most
frequently-missing of the five, and the goal was to add competition information
without doubling the schema.

## NaN discipline (the invariant)
`NaN` means "this signal did not retrieve this class" and must never be read as a
low score. Consequences, all tested:
- a candidate the signal did not score has a `NaN` margin;
- **NaN does not compete** — it is excluded from the top-2 that defines the
  margin, so one missing rival does not inflate another candidate's margin;
- a lone scored candidate has a `NaN` margin, not `0.0`. No competitor means the
  margin is *undefined*, which is a different statement from a tie (`0.0`), and
  conflating them would teach the model that "only class that fired" looks like
  "dead heat";
- a non-candidate column does not compete either (it lost candidate selection).

## Implementation
`application/features.py::_row_margin(M, cand_mask) -> (margin, gap)`, called
once per signal in `_assemble_chunk` and gathered over the same `(rows, cols)`
grid as everything else. Top-2 via `argpartition` (O(C)) rather than a full sort.
No new retrieval, no new state, nothing persisted beyond the schema — these are
pure transforms of the `(b, C)` signal matrices already computed in the chunk.

Leakage: none possible. Every input is already in the batch, and the transform is
identical at train and inference time.

## Measured effect
Offline quality benchmark (`tests/quality/test_benchmark.py`), same fixed seeded
task, A/B on one machine (baseline reproduced the recorded 2026-07-02 numbers
exactly, so these deltas are not platform drift):

| encoder | candidate recall | coverage | acc on accepted | accepted × correct |
|---|---|---|---|---|
| hashing — 28 feat | 0.9938 | 0.9812 | 0.8726 | 0.8562 |
| hashing — 36 feat | 0.9938 | 0.9812 | **0.8854** | **0.8688** |
| tfidf — 28 feat | 1.0000 | 0.9875 | 0.8354 | 0.8250 |
| tfidf — 36 feat | 1.0000 | **1.0000** | 0.8250 | 0.8250 |

Hashing: +1.3pp accuracy on accepted at identical coverage — a real gain.
TFIDF: a wash. Coverage rises to 1.0 and accuracy falls by ~1pp, and the product
is unchanged to four decimals — the same operating curve, a different point on
it, chosen by the threshold tuner.

Honest reading: one small synthetic benchmark (30 classes, items truncated to two
tokens). Its own docstring warns the five-signal ensemble is redundant enough to
mask single-signal changes. This is suggestive, not conclusive — **T40 (ablation
+ importance harness) is what would actually attribute the gain**, and should
land before the rest of the competition-feature backlog below.

## Compatibility
Model dirs trained before this change carry 36-name-mismatched `meta.json` and
will fail `_check_feature_schema` at load with its existing actionable error.
That is the guard working as designed — a schema change requires a retrain. No
silent-wrong-answer path.

## Follow-ups (the rest of the Tier-A analysis, not in this ticket)
1. Softmax / z-score normalization per signal, replacing reliance on min-max —
   removes the query-level offset distributionally, not just by subtraction.
2. Per-query dispersion: `entropy_*` over the candidate softmax, `n_candidates`.
3. Reciprocal-rank fusion `Σ 1/(60 + rank_s)` + per-candidate
   `n_signals_supporting` (evidence breadth — distinct from the query-level
   `n_signal_agreement`).
4. kNN shape: `d_knn_mean` (sum/count — a ratio a tree cannot form),
   `d_knn_frac`, rank-weighted neighbor votes (neighbor *position* is currently
   discarded by `_scatter_knn`).
5. Class-side priors: `class_desc_proto_agreement`, `class_proto_spread`,
   `class_confusability`. `class_log_freq` is currently the only class-level
   feature.
6. Query-side text properties via the `FeatureProvider` seam (T70) — notably
   `q_oov_rate`, which explains *why* a BM25 column is missing.

## Known defect found while doing this (not fixed here)
`application/features.py::_assemble_chunk` computes
`n_signal_agreement = 5 − len({v for v in row if v >= 0})`, which conflates
missing signals with agreeing ones: three signals agreeing with two absent gives
`{c}` → `5 − 1 = 4`, identical to all five agreeing. Absent reads as consensus.
Fix is to split it into `n_signals_present` and `n_agreeing_among_present`.
Deliberately left alone — it changes an existing column's meaning, which is a
separate ticket with its own before/after measurement.

## Acceptance
- [x] 8 columns in `FEATURE_NAMES`, appended, composed schema persisted as before
- [x] `_row_margin` unit tests: leader/deficit, ties, NaN-does-not-compete,
      lone candidate → NaN, all-missing rows, `C == 1`, row independence,
      shift-invariance, no escaping `RuntimeWarning`
- [x] Assembled-frame tests: margin NaN ⊇ signal NaN, ≤1 positive margin per
      item per signal, `q_gap_*` constant within an item, `q_gap_*` == leader
      margin, gap ≥ 0, and the contested-vs-decided regression case
- [x] Full suite green (540 → 571 passed, +31 tests), ruff check + format clean
      on touched files, mypy clean on touched files
- [x] Quality benchmark floors hold; A/B recorded above
