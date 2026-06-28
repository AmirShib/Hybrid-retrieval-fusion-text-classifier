# T41 — Alternative fusion model (LightGBM) behind the `FusionModel` port

status: done
tier: 4
depends_on: T05, T23

## Goal
Ship a second `FusionModel` implementation, `LightGBMFusionModel`, selectable via
`FusionConfig.kind="lightgbm"`, that honors every contract the XGBoost model does —
crucially **native NaN handling** — and round-trips through persistence.

## Why
XGBoost is currently the only fusion backend, and it's instantiated directly in
the pipeline. LightGBM is a common, often-faster gradient-boosting alternative
with its own native missing-value handling, so it's the natural proof that the
`FusionModel` port (now pluggable via T23) accepts a genuinely different library.

## Files to add/change
- `text_classifier/infrastructure/fusion.py` (or `fusion_lightgbm.py`)
  — `LightGBMFusionModel(FusionModel)` + `register_fusion("lightgbm")`
- `text_classifier/config.py` — `FusionConfig` already carries `kind` (T23);
  add a `lgbm_params` default block or reuse the generalized `params`
- `pyproject.toml` — add `lightgbm` as an **optional** dependency
  (`[project.optional-dependencies] lightgbm = ["lightgbm"]`); do NOT make it a
  hard dep (air-gapped hosts may not have it)
- `tests/unit/test_fusion_lightgbm.py` — NEW (skipif lightgbm not installed)

## Contract (mirror the XGBoost tests in T05)
- [ ] **Fits and predicts:** separable `(X, y)` → higher scores for the positive
      class; `predict_proba` shape `(n,)`, values in `[0, 1]`.
- [ ] **NaN tolerance (load-bearing):** inject NaN into feature columns — `fit` and
      `predict_proba` run without error and still separate the classes. LightGBM
      handles missing natively; assert we do **not** impute (pass NaN straight through).
- [ ] **Imbalance:** support `auto_scale_pos_weight` (LightGBM uses `scale_pos_weight`
      or `is_unbalance`/class weights) — with 1:20 data the minority recall improves
      vs. the unweighted variant. `pos == 0` falls back safely (no div-by-zero).
- [ ] **Save/load round-trip:** `fit → save → load → predict_proba` equal within ~1e-6
      using `tmp_path`. Use LightGBM's native text model format (portable, stdlib-loadable).
- [ ] `predict_proba` before `fit`/`load` raises the documented assertion.

## Registry / pipeline integration
- [ ] `build_fusion(FusionConfig(kind="lightgbm", ...))` returns the LightGBM model.
- [ ] End-to-end: `TrainingPipeline` with `kind="lightgbm"` trains, saves, loads, and
      predicts offline on the synthetic fixture (reuse the T07 harness; skip if lightgbm
      absent). `meta.json` records `components.fusion == "lightgbm"`.
- [ ] Persistence stores the LightGBM model under its own filename/format via the T23
      format-agnostic save (NOT hardcoded `fusion.json`).

## Acceptance criteria
- [ ] LightGBM trains + infers end-to-end via config alone, no edits to `TrainingPipeline`
      or `ArtifactRepository` (proves T23's seam).
- [ ] NaN-tolerance test present and passing (the core invariant).
- [ ] Save/load round-trip within tolerance.
- [ ] LightGBM is optional; the suite still passes (with the LightGBM tests skipped)
      when it is not installed.

## Out of scope
Hyperparameter tuning / benchmarking XGBoost vs LightGBM accuracy (T40 ablation).
Changing the feature set or calibration.
