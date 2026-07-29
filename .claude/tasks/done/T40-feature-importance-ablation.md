# T40 — Feature ablation + importance reporting harness

status: done
tier: 4
depends_on: T61

## Goal
Give a data scientist a way to answer "which of the ~28+ fusion feature columns
are actually earning their place?" against an already-trained model, without a
retrain.

## Why
`T61` gave per-class metrics and calibration diagnostics; `signal_report`
(inside the `evaluate` CLI) gives per-*signal* diagnostics pre-fusion. Neither
answers a feature-column-level question: does `d_proto_sim` move the score at
all, and if it went missing tomorrow, would accuracy actually change? That gap
was called out explicitly as the gating item for the T81 follow-ups (softmax
normalization, dispersion features, RRF, kNN shape, class priors, `q_oov_rate`)
— before adding more ~28 columns, there should be a way to measure whether the
existing ones are pulling weight.

## What was done
- **`text_classifier/application/importance.py`** — two pure reports, both
  computed against the trained model with no retraining:
  - `global_feature_importance(fusion, X, feature_names)`: aggregates
    `FusionModel.predict_contribs` (the same additive attribution
    `explain_records(..., include_contributions=True)` exposes per row) across
    a dataset — mean `|contribution|` and each feature's `share` of the total.
    Returns `None` when the backend doesn't support `predict_contribs`
    (matches that port's existing "no attribution available" contract).
  - `ablation_report(feats, fusion, calibrator, abstention, feature_names,
    true_idx_by_item)`: for each feature, masks the column to `NaN` — the
    domain's own "signal did not retrieve this" encoding, which XGBoost/
    LightGBM-style backends already consume natively — and re-scores with the
    *unchanged* model, reporting the coverage/accuracy delta from baseline.
    Because masking reuses the model's native missing-value handling, this is
    a faithful ablation of "what if this signal weren't available", not a
    retrain-based approximation. Columns that are already all-`NaN` on the
    given set are skipped (masking a no-op column just restates baseline).
- **`InferencePipeline.importance_report(texts, true_keys)`** — orchestrates a
  single encode → assemble pass and returns `{"importance": [...] | None,
  "ablation": {"baseline": {...}, "ablations": [...]}}`.
- **`text-classifier-importance` CLI** (`text_classifier/cli/importance.py`,
  registered as a console script) — scores a labeled CSV against a trained
  model, prints the baseline + top-N importance/ablation tables, and optionally
  writes the full JSON report.
- **Tests**: `tests/unit/test_importance.py` (hand-built feature frame + fake
  fusion/calibrator/policy doubles — exact-value checks on the ablation
  arithmetic, since a real XGBoost model's numbers aren't hand-checkable) and
  `tests/integration/test_importance_cli.py` (end-to-end against a trained
  model, offline via the hashing/tfidf encoders, plus the CLI).

## Acceptance criteria
- [x] Importance report ranks features by mean `|contribution|` and shares sum
      to 1; returns `None` gracefully for a backend without `predict_contribs`.
- [x] Ablation report masks each present feature to `NaN`, rescoring with the
      unchanged model, and reports the accuracy/coverage delta from baseline;
      all-missing columns are skipped.
- [x] `text-classifier-importance` runs against a trained model dir and a
      labeled CSV, prints a summary, and can write a full JSON report.
- [x] Full test suite green; ruff check/format and mypy clean on touched files.

## Follow-ups (not in scope here)
- Group-level ablation (e.g. "drop the whole `bm25_knn` signal family, not one
  column") — the per-signal grouping already exists in `signal_report.SIGNALS`
  and could seed it.
- Feeding the ablation ranking back into the T81 follow-up prioritization.
