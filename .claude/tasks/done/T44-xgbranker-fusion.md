# T44 — Alternative fusion model (XGBRanker) behind the `FusionModel` port

status: done
tier: 4
depends_on: T23

## Goal
Ship a third `FusionModel` implementation, `XGBRankerFusionModel`, selectable via
`FusionConfig.kind="xgbranker"`, that trains a learning-to-rank model (XGBoost's
`XGBRanker`) rather than a pointwise classifier, calibrates its raw scores to
probabilities via an isotonic layer, and satisfies every contract the `XGBoostFusionModel`
does — including native NaN handling.

## Why
XGBRanker optimizes a ranking loss (LambdaMart / pairwise) rather than log-loss, so
it directly targets "rank the correct class higher" — the real objective — instead of
binary calibration. It is plausible that it achieves better ranking metrics (NDCG,
MRR) while still supporting the existing predict_proba interface via a post-hoc
isotonic calibration layer.

The `XGBRanker` API shares most of XGBoost's infrastructure (NaN as missing, tree
params, same `xgboost` package), so adding it is a low-risk proof that the T23 seam
accepts a qualitatively different training objective without modifying `TrainingPipeline`.

## Port widening (minimal, backward-compatible)

`FusionModel.fit` today is `fit(X, y)`. XGBRanker requires a `groups` array (one
integer per query, value = number of candidates in that group). The widening:

```python
class FusionModel(ABC):
    @abstractmethod
    def fit(self, X, y, *, groups=None): ...
```

`XGBoostFusionModel.fit` and `LightGBMFusionModel.fit` (T41) must accept and silently
ignore `groups=None` (existing call sites pass nothing). `XGBRankerFusionModel.fit`
requires `groups` and raises `ValueError` if it is `None`.

The training pipeline computes `groups` when `config.fusion.kind == "xgbranker"` (or
when `build_fusion` returns a type that declares `NEEDS_GROUPS = True`); for all other
kinds the call remains `fit(X, y)`.

## Files to add/change
- `text_classifier/domain/ports.py` — widen `FusionModel.fit` signature (keyword-only
  `groups=None`); existing implementations accept without error.
- `text_classifier/infrastructure/fusion.py` (or `fusion_xgbranker.py`)
  — `XGBRankerFusionModel(FusionModel)` + `register_fusion("xgbranker")`
- `text_classifier/application/training.py` — compute `groups` and pass when needed
- `text_classifier/config.py` — document `FusionConfig.kind="xgbranker"` + its params
- `tests/unit/test_fusion_xgbranker.py` — NEW

## Contract (mirror the XGBoost tests in T05)

- [ ] **Fits and predicts:** separable `(X, y, groups)` → `predict_proba` shape `(n,)`,
      values in `[0, 1]`, higher scores for the positive class. Calibration is mandatory
      (raw ranker scores are not probabilities).
- [ ] **NaN tolerance (load-bearing):** inject NaN into feature columns — `fit` and
      `predict_proba` run without error and still separate the classes. XGBRanker
      handles missing natively (same mechanism as XGBClassifier). Assert we do NOT
      impute before passing to the ranker.
- [ ] **Groups:** `groups` must sum to `len(X)`. Test with heterogeneous group sizes.
      Raise `ValueError` with a clear message when `groups is None`.
- [ ] **Imbalance:** within each group the positive label is minority; verify the
      ranker scores the positive item higher than at least half the negatives in the
      group (soft ranking assertion, not hard accuracy).
- [ ] **Save/load round-trip:** `fit → save → load → predict_proba` equal within ~1e-6
      using `tmp_path`. Use XGBoost's native `.ubj` or `.json` model format.
- [ ] `predict_proba` before `fit`/`load` raises the documented assertion.

## Registry / pipeline integration
- [ ] `build_fusion(FusionConfig(kind="xgbranker", ...))` returns the XGBRanker model.
- [ ] Training pipeline detects `NEEDS_GROUPS` and computes group sizes from the OOF
      fold structure: within a fold, each (item, class) set forms one group.
- [ ] `meta.json` records `components.fusion == "xgbranker"`.
- [ ] End-to-end: `TrainingPipeline` with `kind="xgbranker"` trains, saves, loads, and
      predicts offline on the synthetic fixture (reuse the T07 harness; mark slow).

## Acceptance criteria
- [ ] XGBRanker trains + infers end-to-end via config alone, no edits to
      `TrainingPipeline` or `ArtifactRepository` beyond the `groups` plumbing.
- [ ] NaN-tolerance test present and passing.
- [ ] Save/load round-trip within tolerance.
- [ ] Existing `XGBoostFusionModel` tests and default-config behaviour unchanged.
- [ ] Port widening (`groups=None`) is backward-compatible: all existing call sites
      pass without change.

## Out of scope
Benchmarking ranking metrics (NDCG/MRR) vs XGBoost classifier (T40 ablation).
Hyperparameter tuning. Changing the calibration scheme.
