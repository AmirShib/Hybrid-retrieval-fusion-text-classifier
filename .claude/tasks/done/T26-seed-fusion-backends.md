# T26 — Seed the fusion backends: two identical runs must produce identical models

status: done
tier: 2
depends_on: T01

## Goal
Make training end-to-end deterministic by seeding the fusion backends. Two
`TrainingPipeline.run` calls on identical inputs with identical config must produce
byte-identical fusion scores, calibration curves, thresholds, and coverage numbers.

## Why
Everything else in the pipeline is already carefully deterministic:
`StratifiedKFold` takes `TrainingConfig.random_state` (`config.py`), and T21 replaced
`hash()` with `hashlib.sha256` in the test double specifically so results reproduce
across processes. The one remaining leak is the fusion stage:

- `FusionConfig`'s default `xgb_params` set `subsample: 0.8` and
  `colsample_bytree: 0.8` but **no `random_state`** (`config.py:39-50`). Row/column
  subsampling with an unseeded RNG means two runs on the same data give different
  trees, hence different raw scores, a different isotonic fit, different tuned
  thresholds, and a different coverage report.
- `LightGBMFusionModel` and `XGBRankerFusionModel` pass user params straight through
  (`infrastructure/fusion.py`); neither defaults a seed either.

This silently undermines the project's reproducibility story ("air-gapped
reproducibility" is a headline feature) and makes any A/B comparison of a config
change noisy by default.

## Design
Two layers, so the fix holds even for params supplied by a user:

1. **Default config**: add `"random_state": 0` to `FusionConfig.xgb_params`' default
   dict in `config.py`. This fixes the common path and serializes into `meta.json`
   like every other param.
2. **Backend `setdefault`**: in each backend's `fit`
   (`XGBoostFusionModel`, `LightGBMFusionModel`, `XGBRankerFusionModel` in
   `infrastructure/fusion.py`), apply `params.setdefault("random_state", 0)` before
   constructing the estimator — mirroring how `LightGBMFusionModel.fit` already does
   `params.setdefault("verbosity", -1)`. Explicit user params always win.

Notes:
- `random_state` is the accepted kwarg on `XGBClassifier`, `XGBRanker`, and
  `LGBMClassifier` alike — no per-backend spelling needed.
- Do NOT touch `n_jobs`: xgboost's `hist` method is deterministic for a fixed seed
  regardless of thread count, and forcing single-threading would be a real
  performance cost for no gain.
- The calibrators (`IsotonicRegression`, `LogisticRegression` with lbfgs) are
  deterministic already; no change there.

## Files to change
- `text_classifier/config.py` — add `"random_state": 0` to the default `xgb_params`.
- `text_classifier/infrastructure/fusion.py` — `setdefault` in the three `fit`s.
- `tests/unit/test_fusion.py` (+ the lightgbm/xgbranker test modules) — extend.
- `tests/integration/test_e2e.py` — pipeline-level determinism check.

## Tests
- [ ] Unit, per backend: fit twice on the same `(X, y)` (and `groups` for the
      ranker), assert `predict_proba` outputs are exactly equal
      (`np.array_equal`, not `allclose`). LightGBM test skips when the package
      is absent, matching the existing suite convention.
- [ ] Unit: a user-supplied `random_state` in params is respected (two different
      seeds → allowed to differ; same seed → equal).
- [ ] Integration: two `TrainingPipeline.run` calls with the same config, items,
      and `HashingEncoder` produce identical `AbstentionPolicy` thresholds and
      identical `CoverageReport` fields.

## Acceptance criteria
- [ ] Same inputs + same config → identical model directory contents for the
      deterministic formats (fusion model scores, thresholds in `meta.json`).
- [ ] Default `PipelineConfig().to_dict()` now includes the seed, so it is recorded
      in every `meta.json` / `evaluation.json` manifest going forward.
- [ ] No behaviour change other than determinism (metrics on a fixed seed are the
      same class of numbers as before).

## Out of scope
Determinism of sentence-transformers fine-tuning (`use_per_fold_encoder=True` with
torch is a different beast — GPU nondeterminism, dataloader ordering); document it
as a known limitation in T08 instead.
