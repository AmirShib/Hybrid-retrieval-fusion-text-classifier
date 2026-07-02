"""T41 — LightGBMFusionModel: a second FusionModel backend.

Mirrors the XGBoost contract from T05: fit/predict shape+range, separability,
native NaN tolerance (the load-bearing invariant), imbalance handling, and an
exact save/load round-trip via LightGBM's native text format.

Skipped entirely when lightgbm is not installed (it is an optional dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from text_classifier.config import FusionConfig
from text_classifier.infrastructure import build_fusion
from text_classifier.infrastructure.fusion import LightGBMFusionModel


_FAST = {"n_estimators": 30, "max_depth": 3, "random_state": 0, "verbosity": -1}


def _separable_xy(n: int = 200, nan_frac: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int32)
    X = np.zeros((n, 4), dtype=np.float32)
    X[:, 0] = y.astype(np.float32)
    X[:, 1] = rng.standard_normal(n).astype(np.float32)
    if nan_frac > 0:
        X[rng.random((n, 4)) < nan_frac] = np.nan
    return X, y


def test_fit_predict_shape_and_range():
    X, y = _separable_xy()
    m = LightGBMFusionModel(_FAST)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y),)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_separable_classes_score_higher():
    X, y = _separable_xy(n=300)
    m = LightGBMFusionModel(_FAST)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_nan_tolerance_fit_and_predict():
    """Core invariant: NaN == 'not retrieved'; LightGBM handles it natively."""
    X, y = _separable_xy(n=300, nan_frac=0.3)
    m = LightGBMFusionModel(_FAST)
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all(np.isfinite(proba))
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_fit_does_not_impute_nan():
    """Passing NaN straight through must change nothing vs. the model's own
    missing handling — i.e. we never replace NaN with a sentinel before fit."""
    X, y = _separable_xy(n=200, nan_frac=0.25, seed=1)
    m = LightGBMFusionModel(_FAST)
    m.fit(X, y)  # would raise if we tried to coerce NaN to int labels etc.
    assert np.isnan(X).any()  # the input genuinely contained NaN


def test_imbalance_scale_pos_weight_helps_minority():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 20, 400
    X = np.zeros((n_pos + n_neg, 3), dtype=np.float32)
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
    X[:, 0] = y + rng.normal(0, 0.3, n_pos + n_neg)  # weak but real signal
    weighted = LightGBMFusionModel(_FAST, auto_scale_pos_weight=True)
    weighted.fit(X, y)
    plain = LightGBMFusionModel(_FAST, auto_scale_pos_weight=False)
    plain.fit(X, y)
    # Up-weighting the minority raises its average score.
    assert weighted.predict_proba(X)[y == 1].mean() >= plain.predict_proba(X)[y == 1].mean()
    assert abs(weighted._scale_pos_weight - n_neg / n_pos) < 1e-6


def test_all_negative_no_div_zero():
    X = np.zeros((20, 2), dtype=np.float32)
    y = np.zeros(20, dtype=np.int32)
    m = LightGBMFusionModel(_FAST, auto_scale_pos_weight=True)
    m.fit(X, y)  # must not raise (no division by zero)
    assert m._scale_pos_weight == 1.0


def test_save_load_roundtrip(tmp_path):
    X, y = _separable_xy(n=300)
    m = LightGBMFusionModel(_FAST)
    m.fit(X, y)
    original = m.predict_proba(X)
    path = str(tmp_path / "fusion.txt")
    m.save(path)
    loaded = LightGBMFusionModel.load(path).predict_proba(X)
    np.testing.assert_allclose(original, loaded, atol=1e-6)


def test_predict_before_fit_raises():
    with pytest.raises(AssertionError):
        LightGBMFusionModel(_FAST).predict_proba(np.zeros((5, 4), dtype=np.float32))


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(AssertionError):
        LightGBMFusionModel(_FAST).save(str(tmp_path / "fusion.txt"))


def test_registry_builds_lightgbm():
    model = build_fusion(FusionConfig(kind="lightgbm", params=dict(_FAST)))
    assert isinstance(model, LightGBMFusionModel)


def test_fit_ignores_groups_kwarg():
    """Pointwise model accepts groups=None (and any groups) silently for the
    widened FusionModel.fit signature."""
    X, y = _separable_xy(n=100)
    LightGBMFusionModel(_FAST).fit(X, y, groups=None)  # no error


# ---------------------------------------------------------------------------
# Determinism (T26)
# ---------------------------------------------------------------------------


def test_unseeded_fits_deterministic():
    """random_state absent from params -> setdefault seeds it; two fits agree."""
    X, y = _separable_xy(n=200, nan_frac=0.1)
    unseeded = {
        "n_estimators": 30,
        "max_depth": 3,
        "verbosity": -1,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "feature_fraction": 0.5,
    }
    runs = []
    for _ in range(2):
        m = LightGBMFusionModel(dict(unseeded))
        m.fit(X, y)
        runs.append(m.predict_proba(X))
    np.testing.assert_array_equal(runs[0], runs[1])
