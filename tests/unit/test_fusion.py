"""T05 — Fusion + calibration tests.

Covers XGBoostFusionModel and IsotonicCalibrator in infrastructure/fusion.py.
"""

import numpy as np
import pytest

from text_classifier.infrastructure.fusion import IsotonicCalibrator, XGBoostFusionModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAST_PARAMS = {"n_estimators": 20, "max_depth": 3, "random_state": 0}


def _separable_xy(n: int = 100, nan_frac: float = 0.0, seed: int = 0):
    """Return (X, y) where feature 0 deterministically separates classes."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int32)
    X = np.zeros((n, 4), dtype=np.float32)
    # feature 0: positive class gets 1.0, negative gets 0.0
    X[:, 0] = y.astype(np.float32)
    # feature 1: noise
    X[:, 1] = rng.standard_normal(n).astype(np.float32)
    if nan_frac > 0:
        mask = rng.random((n, 4)) < nan_frac
        X[mask] = np.nan
    return X, y


# ---------------------------------------------------------------------------
# Part A — XGBoostFusionModel
# ---------------------------------------------------------------------------


class TestXGBoostFusionModel:
    def test_fit_predict_shape_and_range(self):
        X, y = _separable_xy()
        m = XGBoostFusionModel(_FAST_PARAMS)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y),)
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_separable_classes_score_higher(self):
        X, y = _separable_xy(n=200)
        m = XGBoostFusionModel(_FAST_PARAMS)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba[y == 1].mean() > proba[y == 0].mean()

    def test_predict_contribs_shape_and_additive_identity(self):
        """T69: per-feature contributions sum (with the trailing bias) to the raw
        margin — the SHAP additive identity XGBoost guarantees."""
        from xgboost import DMatrix

        X, y = _separable_xy(n=200)
        m = XGBoostFusionModel(_FAST_PARAMS)
        m.fit(X, y)
        contribs = m.predict_contribs(X)
        assert contribs is not None
        assert contribs.shape == (len(y), X.shape[1] + 1)  # +1 bias column
        raw_margin = m._model.get_booster().predict(DMatrix(X), output_margin=True)
        np.testing.assert_allclose(contribs.sum(axis=1), raw_margin, atol=1e-4)

    def test_nan_tolerance_fit_and_predict(self):
        """Core invariant: NaN == 'not retrieved'; model must not crash or impute."""
        X, y = _separable_xy(n=200, nan_frac=0.3)
        m = XGBoostFusionModel(_FAST_PARAMS)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y),)
        assert np.all(np.isfinite(proba))
        # Model still separates despite heavy NaN injection
        assert proba[y == 1].mean() > proba[y == 0].mean()

    def test_auto_scale_pos_weight_set(self):
        """With imbalanced data (1:20), scale_pos_weight ≈ 20."""
        n_pos, n_neg = 10, 200
        X = np.vstack(
            [
                np.ones((n_pos, 2), dtype=np.float32),
                np.zeros((n_neg, 2), dtype=np.float32),
            ]
        )
        y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
        m = XGBoostFusionModel(_FAST_PARAMS, auto_scale_pos_weight=True)
        m.fit(X, y)
        fitted_spw = m._model.get_params()["scale_pos_weight"]
        assert abs(fitted_spw - n_neg / n_pos) < 1e-6

    def test_auto_scale_pos_weight_false_no_override(self):
        """When auto_scale_pos_weight=False, scale_pos_weight is not injected by us."""
        X, y = _separable_xy()
        params = dict(_FAST_PARAMS)  # no scale_pos_weight key
        m = XGBoostFusionModel(params, auto_scale_pos_weight=False)
        m.fit(X, y)
        # XGBoost leaves scale_pos_weight as None (its own default) when we don't set it
        spw = m._model.get_params()["scale_pos_weight"]
        assert spw is None or spw == 1.0

    def test_all_negative_no_div_zero(self):
        """pos == 0 guard: scale_pos_weight falls back to 1.0."""
        X = np.zeros((20, 2), dtype=np.float32)
        y = np.zeros(20, dtype=np.int32)
        m = XGBoostFusionModel(_FAST_PARAMS, auto_scale_pos_weight=True)
        m.fit(X, y)  # must not raise
        fitted_spw = m._model.get_params()["scale_pos_weight"]
        assert fitted_spw == 1.0

    def test_save_load_roundtrip(self, tmp_path):
        X, y = _separable_xy(n=200)
        m = XGBoostFusionModel(_FAST_PARAMS)
        m.fit(X, y)
        original = m.predict_proba(X)

        path = str(tmp_path / "fusion.json")
        m.save(path)
        m2 = XGBoostFusionModel.load(path)
        loaded = m2.predict_proba(X)

        np.testing.assert_allclose(original, loaded, atol=1e-6)

    def test_predict_before_fit_raises(self):
        m = XGBoostFusionModel(_FAST_PARAMS)
        with pytest.raises(AssertionError):
            m.predict_proba(np.zeros((5, 4), dtype=np.float32))

    def test_save_before_fit_raises(self, tmp_path):
        m = XGBoostFusionModel(_FAST_PARAMS)
        with pytest.raises(AssertionError):
            m.save(str(tmp_path / "model.json"))


# ---------------------------------------------------------------------------
# Part B — IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def _fit_cal(self, n: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0, 1, n)
        correct = rng.binomial(1, scores).astype(np.float64)
        cal = IsotonicCalibrator()
        cal.fit(scores, correct)
        return cal, scores

    def test_monotonicity(self):
        cal, scores = self._fit_cal()
        sorted_scores = np.sort(scores)
        out = cal.transform(sorted_scores)
        # non-decreasing
        assert np.all(out[1:] >= out[:-1] - 1e-12)

    def test_output_range(self):
        cal, scores = self._fit_cal()
        out = cal.transform(scores)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_out_of_bounds_clipping(self):
        """Scores outside training range are clipped, not extrapolated."""
        cal, _ = self._fit_cal()
        extreme = np.array([-1.0, 2.0])
        out = cal.transform(extreme)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_known_mapping(self):
        """Calibrator tracks true probability on held-out bins."""
        rng = np.random.default_rng(7)
        n = 2000
        scores = rng.uniform(0, 1, n)
        correct = rng.binomial(1, scores).astype(np.float64)

        cal = IsotonicCalibrator()
        cal.fit(scores, correct)

        # Evaluate on mid-point of 5 equal bins
        bin_edges = np.linspace(0, 1, 6)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mid = (lo + hi) / 2.0
            calibrated = float(cal.transform(np.array([mid]))[0])
            # True P(correct | score=mid) ≈ mid; tolerance 0.2 with n=2000
            assert abs(calibrated - mid) < 0.20

    def test_save_load_roundtrip(self, tmp_path):
        cal, scores = self._fit_cal()
        original = cal.transform(scores)

        path = str(tmp_path / "calibrator.pkl")
        cal.save(path)
        cal2 = IsotonicCalibrator.load(path)
        loaded = cal2.transform(scores)

        np.testing.assert_array_equal(original, loaded)

    def test_save_writes_npz_not_pickle(self, tmp_path):
        """T67: fresh saves are inert numpy .npz, not a pickle stream."""
        cal, _ = self._fit_cal()
        path = str(tmp_path / "calibrator.npz")
        cal.save(path)
        with open(path, "rb") as fh:
            head = fh.read(1)
        assert head != b"\x80"  # 0x80 is pickle's PROTO opcode
        data = np.load(path)  # must parse as a valid npz archive
        assert "X_thresholds" in data and "y_thresholds" in data

    def test_legacy_pickle_fallback_loads_with_warning(self, tmp_path, caplog):
        """A pre-T67 model dir has only calibrator.pkl; load() must still work."""
        import logging
        import pickle

        cal, scores = self._fit_cal()
        original = cal.transform(scores)

        legacy_path = tmp_path / "calibrator.pkl"
        with open(legacy_path, "wb") as fh:
            pickle.dump(cal._iso, fh)

        new_path = str(tmp_path / "calibrator.npz")  # does not exist
        with caplog.at_level(logging.WARNING):
            loaded = IsotonicCalibrator.load(new_path)
        np.testing.assert_array_equal(original, loaded.transform(scores))
        assert any("legacy pickle" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Part C — Determinism (T26)
# ---------------------------------------------------------------------------

# Row/column subsampling engages the RNG; random_state deliberately absent so
# the backend's setdefault is what makes the runs reproducible.
_UNSEEDED_STOCHASTIC = {
    "n_estimators": 30,
    "max_depth": 3,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "n_jobs": 1,
}


class TestDeterminism:
    def test_unseeded_params_default_to_seed_zero(self):
        X, y = _separable_xy(n=200)
        m = XGBoostFusionModel(dict(_UNSEEDED_STOCHASTIC))
        m.fit(X, y)
        assert m._model.get_params()["random_state"] == 0

    def test_two_unseeded_fits_are_exactly_equal(self):
        X, y = _separable_xy(n=200, nan_frac=0.1)
        runs = []
        for _ in range(2):
            m = XGBoostFusionModel(dict(_UNSEEDED_STOCHASTIC))
            m.fit(X, y)
            runs.append(m.predict_proba(X))
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_explicit_user_seed_wins(self):
        X, y = _separable_xy(n=200)
        m = XGBoostFusionModel({**_UNSEEDED_STOCHASTIC, "random_state": 123})
        m.fit(X, y)
        assert m._model.get_params()["random_state"] == 123

    def test_fit_does_not_mutate_caller_params(self):
        params = dict(_UNSEEDED_STOCHASTIC)
        X, y = _separable_xy(n=100)
        XGBoostFusionModel(params).fit(X, y)
        assert "random_state" not in params
        assert "scale_pos_weight" not in params
