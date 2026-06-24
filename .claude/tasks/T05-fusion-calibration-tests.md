# T05 — Fusion + calibration tests

status: done
tier: 1
depends_on: T01

## Goal
Test `infrastructure/fusion.py`: the `XGBoostFusionModel` (NaN tolerance, `scale_pos_weight`
on imbalance, save/load round-trip) and the `IsotonicCalibrator` (monotonicity, clipping,
save/load round-trip).

## Why
The fusion model is the heart of the system and the only component that *must* tolerate NaN
(the "not retrieved" encoding). Calibration is what makes the abstention threshold meaningful.
Both are persisted, so round-trip fidelity is a correctness requirement, not a nicety.

## Files under test
`text_classifier/infrastructure/fusion.py`

## Part A — `XGBoostFusionModel`

- [ ] **Fits and predicts:** on a separable synthetic `(X, y)` where one feature determines
      the label, `predict_proba` returns higher scores for the positive class. Shape is
      `(n,)` and values in `[0, 1]`.
- [ ] **NaN tolerance:** inject `NaN` into feature columns (the real "missing" encoding) and
      confirm `fit` and `predict_proba` run without error and still separate the classes.
      This guards the core invariant "do not impute NaN."
- [ ] **`auto_scale_pos_weight=True`:** with a heavily imbalanced `y` (e.g. 1:20), confirm
      the model sets `scale_pos_weight ≈ n_neg/n_pos` at fit time. Verify by inspecting the
      fitted estimator's param, or by behavioral contrast vs `auto_scale_pos_weight=False`
      (the weighted model assigns higher recall to the minority positive).
- [ ] **`pos == 0` guard:** all-negative `y` → `scale_pos_weight` falls back to `1.0`, no
      div-by-zero.
- [ ] **Save/load round-trip:** `fit → save → load → predict_proba` gives **bitwise-equal or
      ~1e-6** predictions to the in-memory model on the same `X`. Use a `tmp_path`.
- [ ] `predict_proba` before `fit`/`load` raises the documented assertion.

## Part B — `IsotonicCalibrator`

- [ ] **Monotonicity:** fit on `(scores, correct)`; `transform` is non-decreasing in the raw
      score (sort inputs, assert output is sorted/non-decreasing).
- [ ] **Range:** outputs lie in `[0, 1]`.
- [ ] **Out-of-bounds clipping:** scores below the min / above the max seen in `fit` are
      clipped (no extrapolation beyond `[0,1]`), per `out_of_bounds="clip"`.
- [ ] **Recovers a known mapping:** construct data where P(correct) is a known monotonic
      function of score (e.g. score ~ Uniform, correct ~ Bernoulli(score)); after fit, the
      calibrated values track the true probability within a tolerance on a held bin.
- [ ] **Save/load round-trip:** `fit → save → load → transform` reproduces outputs exactly.

## Acceptance criteria
- [ ] NaN-tolerance test present and passing (load-bearing invariant).
- [ ] Both adapters have a save/load round-trip test using `tmp_path`.
- [ ] Imbalance / `scale_pos_weight` behavior asserted (param or behavioral).
- [ ] `infrastructure/fusion.py` ≥ 95% line coverage.

## Notes
- XGBoost is a hard dependency already; no need to mock it. Keep `n_estimators` small in
  tests (e.g. 20) for speed — override the default `FusionConfig.xgb_params`.
