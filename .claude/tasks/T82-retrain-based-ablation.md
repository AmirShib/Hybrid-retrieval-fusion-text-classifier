# T82 — Retrain-based feature ablation: `drop_features` + a seeded sweep harness

status: in-review
tier: 4
depends_on: T40, T81

## Goal
Make "should this column be in the schema?" an answerable question. T40 gave a
masking ablation against a fixed model; this adds the missing half — the ability
to **train without a column** and compare, across seeds, with error bars.

## Why
T40 landed and was immediately pointed at T81's eight competition columns. It
produced two reports that flatly disagreed, and the disagreement was the finding:

- `margin_d_desc` was the **#1 column by attribution on tfidf (23% of all mass)
  and #2 on hashing** — yet masking it out never once hurt across 8 runs.

That is not a contradiction, it is a diagnosis: the model routes through
`margin_d_desc` because it is the cleanest single split available, but the
information is redundant with `rank_d_desc` / `norm_d_desc` / `d_desc_sim`, which
absorb the loss when it goes missing.

But the masking ablation *cannot* settle the schema question, and it is important
to be precise about why. Masking a column to `NaN` on an already-trained model
answers "what if this signal fails at inference?" — the model still has splits on
that column, and masked rows take the default branch. "Would the model be better
if this column had never existed?" requires fitting a model that never saw it.
Nothing in the codebase could do that: no way to select a feature subset, and no
multi-seed runner. T40's own follow-ups anticipated the first half.

## What was done

**1. The seam — `FusionConfig.drop_features`**
- `domain/services.py::fusion_feature_names(providers, drop)`: the composed
  schema minus `drop`, order preserved. Deliberately *separate* from
  `composed_feature_names`, which stays the **assembled** schema. Raises on an
  unknown column (a typo would otherwise drop nothing and silently invalidate an
  experiment) and on dropping everything.
- Threaded through training, inference, persistence and tuning. It rides in
  `meta.json`'s config block and is re-applied at load, so inference rebuilds
  the exact column list the model was fitted on — the same train/infer parity
  contract the composed schema already carries.
- `--drop-features a,b,c` on the train CLI.

**Key boundary:** dropping narrows *the model*, never the frame. The assembler
still computes every column, because `explain`, `signal_report` (documented as
running on `explain`'s output) and T40's masking ablation all read core columns
by name. `InferencePipeline` therefore carries two lists — `_feature_names`
(fitted, drives scoring + contribution alignment) and `_assembled_names` (every
column, drives the diagnostic surface). This was found by a test: `explain` had
silently inherited the narrowed list, which would have broken `signal_report` on
any subset-trained model.

**2. The harness — `application/retrain_ablation.py`**
`retrain_ablation(items, label_space, config, arms, seeds)` trains each arm plus
a no-drop baseline once per seed and reports:
- **paired deltas.** The baseline runs under the *same* seeds, and arms are
  compared seed-for-seed. Comparing means across independently-noisy runs
  attributes shared fold-split variance to the feature change; pairing removes it.
- **`accepted_correct`** (coverage x accuracy) alongside both. Coverage and
  accuracy slide against each other as the tuned threshold moves, so either alone
  can shift a point with no model change; their product does not.
- **a verdict** — `earns_place` / `redundant` / `inconclusive`, where the effect
  must exceed the spread of its own per-seed differences before it is called
  anything.

Plus `text-classifier-retrain-ablate` (`--group NAME=col,col`, repeatable).
Cost is `(arms + 1) x seeds` full training runs; it is a measurement tool, not a
hot path.

## What it says about T81

Benchmark task (30 classes, 2-token items), 6 seeds, 36 training runs per encoder:

| arm (columns **dropped**) | hashing paired Δ | tfidf paired Δ | verdict |
|---|---|---|---|
| all 8 T81 columns | −0.0021 ± 0.0164 | +0.0010 ± 0.0278 | inconclusive |
| the 5 `margin_*` | −0.0021 ± 0.0112 | −0.0104 ± 0.0283 | inconclusive |
| the 3 `q_gap_*` | −0.0042 ± 0.0176 | +0.0031 ± 0.0129 | inconclusive |
| `margin_d_desc` alone | +0.0021 ± 0.0179 | −0.0073 ± 0.0227 | inconclusive |
| the 4 zero-importance legacy columns | −0.0010 ± 0.0098 | **0.0000 ± 0.0000** | redundant (tfidf) |

**Dropping all eight T81 columns is indistinguishable from keeping them.** The
baseline's own seed-to-seed spread is ±0.0156 (hashing) / ±0.0278 (tfidf), which
is larger than every effect measured — so this benchmark **cannot resolve
feature-level effects below roughly 2-3pp**, and the +1.3pp originally recorded
for T81 sits inside that noise. It should not have been reported as a gain.

That is a statement about the benchmark's power, not proof the columns are
worthless: a 30-class synthetic task with two-token items is not the imbalanced
real-world taxonomy the package targets. The right next move is to run this
harness on real data, where the answer may well differ.

**The 4 zero-importance columns are provably inert.** `is_d_desc_top1`,
`b_desc_missing`, `b_knn_missing`, `d_knn_missing` carry exactly 0.00 attribution
(the model never splits on them), and on tfidf dropping them changes the outcome
by **exactly 0.0000 with exactly 0.0000 spread across all 6 seeds**. The three
`*_missing` flags are structurally redundant: they encode "this column is NaN",
which XGBoost's native missing-value handling already knows. Removing them is the
one schema change the current evidence actually supports — left for its own
ticket, since it touches the pre-existing 28 and forces a retrain.

## Bug found by using the tool
The first verdict rule read `abs(delta) <= std -> inconclusive`, which mapped the
exactly-inert case (delta 0.0, spread 0.0) to "inconclusive" — inverting the
strongest result in the sweep. A zero-spread sweep is now decided by sign, as the
docstring always claimed. Regression test:
`test_provably_inert_columns_are_redundant_not_inconclusive`.

## Acceptance
- [x] `fusion_feature_names` with validation; `composed_feature_names` unchanged
- [x] `FusionConfig.drop_features` validated, persisted, re-applied at load
- [x] `--drop-features` on train; `text-classifier-retrain-ablate` console script
- [x] `explain` keeps the full assembled schema (guards `signal_report`)
- [x] Paired-by-seed deltas, `accepted_correct`, conservative verdict
- [x] 620 passed (+38); ruff check/format and mypy clean on touched files
- [x] Empty `drop_features` is byte-for-byte the existing behaviour

## Follow-ups
- Drop the 3 `*_missing` flags + `is_d_desc_top1` (own ticket; touches the core 28)
- Re-run this harness on a real dataset before any further feature work
- Group-level arms seeded from `signal_report.SIGNALS` (T40's own follow-up)
