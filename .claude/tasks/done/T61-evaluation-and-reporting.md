# T61 — Evaluation metrics, persisted report + model card, evaluate CLI

status: done
tier: 6
depends_on: T07

## Goal
Give a data scientist the evidence needed to trust a trained model, and make
that evidence travel with the model and be reproducible on fresh data.

## Why
The system is a calibrated, abstaining classifier for imbalanced data, but it
only ever emitted four global scalars (coverage, accuracy-on-accepted,
accuracy-if-no-abstain, candidate recall), and even those were printed and then
discarded. There was no per-class view (the global number hides classes the
system silently abstains on), no calibration diagnostics (despite calibration
being the basis of the abstention threshold), and no way to re-score a deployed
model on a fresh labeled set.

## What was done
- **`text_classifier/application/evaluation.py`** — pure-numpy metrics, shared by
  training-time and standalone evaluation, JSON-clean (NaN/inf -> null):
  - overall coverage / accuracy-on-accepted / accuracy-if-no-abstain / candidate
    recall;
  - per-class precision (on accepted) / recall / coverage / support;
  - calibration: Brier score, expected calibration error, reliability table;
  - risk-coverage curve (accuracy vs. coverage at sampled thresholds);
  - `build_manifest` (package version, timestamp, data shape, config) and
    `render_model_card` / `write_evaluation_artifacts`.
- **Persisted artifacts**: `TrainingPipeline.run(output_dir=...)` now writes
  `evaluation.json` (full held-out report + manifest + abstention thresholds) and
  `model_card.md` (human summary) into the model directory.
- **`text-classifier-eval`** scores a trained model against a labeled CSV and
  prints/writes the same report — for pre-deploy validation and drift monitoring.
- **Tests**: `tests/unit/test_evaluation.py` (exact-value metric checks) and
  `tests/integration/test_evaluate_cli.py` (persistence + CLI + error paths).

## Acceptance criteria
- [x] Training writes `evaluation.json` and `model_card.md` into the model dir.
- [x] `evaluation.json` contains overall, per-class, calibration, risk-coverage,
      abstention, and manifest blocks and is valid JSON (no NaN tokens).
- [x] `text-classifier-eval` runs on a labeled CSV and writes a JSON report.
- [x] Metrics have exact-value unit tests.

## Follow-ups (not in scope here)
- Feature importance / ablation reporting (T40).
- Reliability-diagram / risk-coverage plot rendering (currently data only).
