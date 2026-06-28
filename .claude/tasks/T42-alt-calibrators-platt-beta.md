# T42 — Alternative calibrators (Platt, beta) behind the `ConfidenceCalibrator` port

status: todo
tier: 4
depends_on: T23

## Goal
Ship two additional calibration backends — **Platt scaling** and **beta
calibration** — alongside the existing `isotonic` calibrator, each selectable via
`CalibrationConfig.kind`, trainable and persistable end-to-end through the
registry. Provide a lightweight comparison so a user can pick the calibrator that
best fits their reliability curve and calibration-set size.

## Why
Isotonic regression is non-parametric and excellent with ample calibration data,
but it can overfit a small calibration fold (it will happily fit a step function
to noise) and emits piecewise-constant outputs. Parametric calibrators are more
robust when the held-out calibration fold is small — common here, since exactly
one fold is reserved for calibration:
- **Platt scaling** (sigmoid): a logistic fit on the raw fusion score. Cheap,
  smooth, robust on small data; assumes a sigmoidal reliability curve.
- **Beta calibration** (Kull et al., 2017): a 3-parameter family generalizing
  Platt that handles asymmetric/skewed reliability curves — a good default when
  scores are already probability-like (as XGBoost's are) and the sigmoid is too
  rigid.

T23's registry makes this purely additive: no pipeline or persistence edits.

## Background: the port
`ConfidenceCalibrator` (domain/ports.py) is already generic — `fit(scores,
correct)`, `transform(scores)`, `save(path)`, `load(path)`. Each new backend
implements it and registers via `register_calibrator(name, CalibratorSpec(...))`.
`build_calibrator(cfg)` and `ArtifactRepository` already route through the spec
(including the per-backend filename + loader), so the seam exists today.

## Backends to add
### Platt (`kind="platt"`)
- Fit `sklearn.linear_model.LogisticRegression` on the 1-D score → `correct`.
  (Equivalent to Platt's sigmoid; LogisticRegression is the standard, stable
  implementation.)
- `transform` returns `predict_proba(...)[:, 1]`, clipped to `[0, 1]`.
- Persist via pickle of the fitted estimator (its own registry filename, distinct
  from isotonic's `calibrator.pkl`).

### Beta (`kind="beta"`)
- Beta calibration: fit `a, b, c` for
  `p = 1 / (1 + 1/(e^c · x^a / (1-x)^b))` via logistic regression on the two
  features `[ln(x), -ln(1-x)]` (the standard fitting trick from the paper).
- Clip `x` to `[eps, 1-eps]` before the log transform to guard {0, 1}.
- Prefer a self-contained ~20-line implementation (no new package) so air-gapped
  hosts aren't blocked; if the `betacal` package is used instead, make it optional
  like `lightgbm` and skip its tests when absent.
- Persist the three fitted params (json or pickle).

## Files to add/change
- `text_classifier/infrastructure/fusion.py` (or a new `calibration.py`) —
  `PlattCalibrator`, `BetaCalibrator`.
- `text_classifier/infrastructure/registry.py` — `register_calibrator("platt", ...)`
  and `register_calibrator("beta", ...)`.
- `text_classifier/config.py` — document `CalibrationConfig.kind ∈
  {"isotonic","platt","beta"}` and any `params`.
- `tests/unit/test_calibration.py` — NEW.
- `tests/integration/test_e2e.py` — parametrize the e2e / persistence round-trip
  over calibrator kind.

## Tests

### Unit (offline)
- [ ] Each calibrator: `fit` then `transform` returns finite values in `[0, 1]`.
- [ ] Monotonicity: Platt and beta are monotonic in the score — higher raw score
      → higher (or equal) calibrated `p`.
- [ ] Calibration improves a deliberately miscalibrated input (lower Brier/ECE
      after calibration than before) on a synthetic reliability curve.
- [ ] Degenerate inputs: all-positive / all-negative `correct`, constant scores,
      scores at exactly 0 and 1 → no NaN/inf (beta's log-guard exercised).
- [ ] `save` → `load` round-trip: identical outputs within ~1e-6.
- [ ] Registered under their kinds; `build_calibrator(CalibrationConfig(kind=...))`
      returns the right type; an unknown kind raises the registry's `ValueError`.

### Integration
- [ ] e2e train → save → load → predict with each calibrator kind, fully offline.
- [ ] Default (`isotonic`) behaviour unchanged; existing tests pass untouched.

## Comparison (lightweight)
- [ ] A small test or script reports ECE/Brier for isotonic vs platt vs beta on
      the calibration fold of the synthetic dataset, so the trade-off is visible.
      Reporting only — introduces no new runtime dependency.

## Acceptance criteria
- [ ] Two new calibrators select via config alone; no pipeline/persistence edits
      beyond registry entries.
- [ ] Beta calibration has a torch-free, package-free default implementation.
- [ ] Persistence round-trips identical within ~1e-6 for all three.
- [ ] Default behaviour and existing tests unchanged.

## Out of scope
Temperature scaling and histogram/binning calibrators (separate follow-ups if
wanted). Per-class calibration (thresholds already go per-class; calibration
stays global). Automatic calibrator selection — this ticket exposes the choice;
choosing for the user is future work.
