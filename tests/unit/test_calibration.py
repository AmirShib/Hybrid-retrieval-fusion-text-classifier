"""T42 — Platt + beta calibrator tests.

Covers PlattCalibrator and BetaCalibrator in infrastructure/fusion.py, alongside
the existing IsotonicCalibrator (tested in test_fusion.py). Fully offline.
"""

import numpy as np
import pandas as pd
import pytest

from text_classifier.config import CalibrationConfig
from text_classifier.domain import ConfidenceCalibrator
from text_classifier.infrastructure import build_calibrator
from text_classifier.infrastructure.fusion import (
    BetaCalibrator,
    IsotonicCalibrator,
    PerClassCalibrator,
    PlattCalibrator,
)

_PARAMETRIC = [PlattCalibrator, BetaCalibrator]


def _miscalibrated(n: int = 4000, seed: int = 0):
    """Systematically over-confident raw scores: the reported score is ``s**2``
    but the true P(correct | s) is ``s``. A good calibrator should pull ``s**2``
    back toward ``s`` and lower the Brier score."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, n)
    raw = s**2
    correct = rng.binomial(1, s).astype(np.float64)
    return raw, correct


def _brier(p: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((p - correct) ** 2))


@pytest.mark.parametrize("cls", _PARAMETRIC)
class TestParametricCalibrators:
    def _fit_monotone(self, cls, n: int = 4000, seed: int = 42):
        """Fit on data where correctness clearly increases with the score."""
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0, 1, n)
        correct = rng.binomial(1, scores).astype(np.float64)
        cal = cls()
        cal.fit(scores, correct)
        return cal, scores

    def test_is_a_calibrator(self, cls):
        assert isinstance(cls(), ConfidenceCalibrator)

    def test_output_shape_and_range(self, cls):
        cal, scores = self._fit_monotone(cls)
        out = cal.transform(scores)
        assert out.shape == scores.shape
        assert np.all(np.isfinite(out))
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_monotonic_nondecreasing(self, cls):
        cal, _ = self._fit_monotone(cls, seed=1)
        grid = np.linspace(0.0, 1.0, 200)
        out = cal.transform(grid)
        assert np.all(out[1:] >= out[:-1] - 1e-9)

    def test_lowers_brier_on_miscalibrated_input(self, cls):
        raw, correct = _miscalibrated()
        cal = cls()
        cal.fit(raw, correct)
        assert _brier(cal.transform(raw), correct) < _brier(raw, correct)

    def test_all_positive_correct_is_finite(self, cls):
        scores = np.linspace(0, 1, 50)
        cal = cls()
        cal.fit(scores, np.ones_like(scores))  # single-class -> constant fallback
        out = cal.transform(scores)
        assert np.all(np.isfinite(out)) and np.all((out >= 0.0) & (out <= 1.0))

    def test_all_negative_correct_is_finite(self, cls):
        scores = np.linspace(0, 1, 50)
        cal = cls()
        cal.fit(scores, np.zeros_like(scores))  # single-class -> constant fallback
        out = cal.transform(scores)
        assert np.all(np.isfinite(out)) and np.all((out >= 0.0) & (out <= 1.0))

    def test_constant_scores_is_finite(self, cls):
        scores = np.full(100, 0.7)
        correct = np.array([0, 1] * 50, dtype=np.float64)
        cal = cls()
        cal.fit(scores, correct)
        assert np.all(np.isfinite(cal.transform(scores)))

    def test_scores_at_zero_and_one_finite(self, cls):
        scores = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.5])
        correct = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
        cal = cls()
        cal.fit(scores, correct)
        out = cal.transform(np.array([0.0, 1.0, 0.5]))
        assert np.all(np.isfinite(out)) and np.all((out >= 0.0) & (out <= 1.0))

    def test_save_load_roundtrip(self, cls, tmp_path):
        cal, scores = self._fit_monotone(cls)
        original = cal.transform(scores)
        path = str(tmp_path / "calibrator.pkl")
        cal.save(path)
        loaded = cls.load(path)
        assert isinstance(loaded, cls)
        np.testing.assert_allclose(original, loaded.transform(scores), atol=1e-6)

    def test_save_load_roundtrip_constant_fallback(self, cls, tmp_path):
        scores = np.linspace(0, 1, 30)
        cal = cls()
        cal.fit(scores, np.ones_like(scores))  # exercises the constant branch
        original = cal.transform(scores)
        path = str(tmp_path / "calibrator.pkl")
        cal.save(path)
        np.testing.assert_allclose(original, cls.load(path).transform(scores), atol=1e-6)

    def test_save_writes_json_not_pickle(self, cls, tmp_path):
        """T67: fresh saves are inert JSON, not a pickle stream."""
        cal, scores = self._fit_monotone(cls)
        path = str(tmp_path / "calibrator.json")
        cal.save(path)
        with open(path, "rb") as fh:
            head = fh.read(1)
        assert head != b"\x80"  # 0x80 is pickle's PROTO opcode; JSON never starts with it
        import json

        with open(path) as fh:
            json.load(fh)  # must parse as JSON

    def test_legacy_pickle_fallback_loads_with_warning(self, cls, tmp_path, caplog):
        """A pre-T67 model dir has only calibrator.pkl; load() must still work."""
        import logging
        import pickle

        cal, scores = self._fit_monotone(cls)
        original = cal.transform(scores)

        legacy_path = tmp_path / "calibrator.pkl"
        with open(legacy_path, "wb") as fh:
            pickle.dump({"lr": cal._lr, "constant": cal._constant}, fh)

        new_path = str(tmp_path / "calibrator.json")  # does not exist
        with caplog.at_level(logging.WARNING):
            loaded = cls.load(new_path)
        np.testing.assert_allclose(original, loaded.transform(scores), atol=1e-6)
        assert any("legacy pickle" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_registry_builds_platt():
    assert isinstance(build_calibrator(CalibrationConfig(kind="platt")), PlattCalibrator)


def test_registry_builds_beta():
    assert isinstance(build_calibrator(CalibrationConfig(kind="beta")), BetaCalibrator)


# --------------------------------------------------------------------------- #
# Lightweight comparison: every calibrator improves on the raw score (T42 report)
# --------------------------------------------------------------------------- #
def test_all_calibrators_beat_raw_brier():
    raw, correct = _miscalibrated(seed=3)
    baseline = _brier(raw, correct)
    for cls in (IsotonicCalibrator, PlattCalibrator, BetaCalibrator):
        cal = cls()
        cal.fit(raw, correct)
        assert _brier(cal.transform(raw), correct) <= baseline


# --------------------------------------------------------------------------- #
# T45 — PerClassCalibrator
# --------------------------------------------------------------------------- #
def _two_class_data(n_per_class: int = 300, seed: int = 0):
    """Class 0 is well-calibrated (P(correct) == score); class 1 is badly
    skewed (P(correct) == score**3) -- a global curve averages the two and is
    wrong for both, so a per-class fit should do better on class 1."""
    rng = np.random.default_rng(seed)
    scores0 = rng.uniform(0, 1, n_per_class)
    correct0 = rng.binomial(1, scores0).astype(np.float64)
    scores1 = rng.uniform(0, 1, n_per_class)
    correct1 = rng.binomial(1, scores1**3).astype(np.float64)
    scores = np.concatenate([scores0, scores1])
    correct = np.concatenate([correct0, correct1])
    classes = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return scores, correct, classes, scores1, correct1


class TestPerClassCalibrator:
    def test_is_a_calibrator(self):
        assert isinstance(PerClassCalibrator(), ConfidenceCalibrator)

    def test_unknown_inner_raises(self):
        with pytest.raises(ValueError):
            PerClassCalibrator(inner="bogus")

    def test_classes_none_reproduces_global_at_fit_and_transform(self):
        scores, correct, classes, _, _ = _two_class_data()
        cal = PerClassCalibrator(inner="isotonic", min_support=1)
        cal.fit(scores, correct)  # no classes -> global-only fit

        direct = IsotonicCalibrator()
        direct.fit(scores, correct)

        np.testing.assert_allclose(cal.transform(scores), direct.transform(scores), atol=1e-9)
        # classes given at transform time but never fitted per-class -> still global
        np.testing.assert_allclose(
            cal.transform(scores, classes=classes), direct.transform(scores), atol=1e-9
        )

    def test_well_supported_class_beats_global_brier(self):
        scores, correct, classes, scores1, correct1 = _two_class_data()
        cal = PerClassCalibrator(inner="beta", min_support=50)
        cal.fit(scores, correct, classes=classes)

        global_only = BetaCalibrator()
        global_only.fit(scores, correct)

        per_class_out = cal.transform(scores1, classes=np.ones(len(scores1), dtype=int))
        global_out = global_only.transform(scores1)
        assert _brier(per_class_out, correct1) < _brier(global_out, correct1)

    def test_under_supported_class_equals_global(self):
        scores, correct, classes, _, _ = _two_class_data()
        cal = PerClassCalibrator(inner="beta", min_support=10_000)  # nothing qualifies
        cal.fit(scores, correct, classes=classes)

        global_only = BetaCalibrator()
        global_only.fit(scores, correct)

        np.testing.assert_allclose(
            cal.transform(scores, classes=classes), global_only.transform(scores), atol=1e-9
        )

    def test_degenerate_class_slice_is_finite(self):
        """A class slice that is all-correct or all-incorrect must not raise or
        produce NaN/inf -- exercises _ParametricCalibrator's base-rate fallback
        per class."""
        classes = np.array([0] * 20 + [1] * 20)
        scores = np.tile(np.linspace(0, 1, 20), 2)
        correct = np.concatenate([np.ones(20), np.zeros(20)])
        cal = PerClassCalibrator(inner="beta", min_support=5)
        cal.fit(scores, correct, classes=classes)
        out = cal.transform(scores, classes=classes)
        assert np.all(np.isfinite(out)) and np.all((out >= 0.0) & (out <= 1.0))

    def test_unseen_class_at_transform_uses_global(self):
        scores, correct, classes, _, _ = _two_class_data()
        cal = PerClassCalibrator(inner="beta", min_support=5)
        cal.fit(scores, correct, classes=classes)  # only classes 0, 1 seen

        global_only = BetaCalibrator()
        global_only.fit(scores, correct)

        probe = np.linspace(0, 1, 10)
        unseen_classes = np.full(10, 99)
        np.testing.assert_allclose(
            cal.transform(probe, classes=unseen_classes), global_only.transform(probe), atol=1e-6
        )

    def test_save_load_roundtrip(self, tmp_path):
        scores, correct, classes, _, _ = _two_class_data()
        cal = PerClassCalibrator(inner="beta", min_support=50)
        cal.fit(scores, correct, classes=classes)
        original = cal.transform(scores, classes=classes)

        path = str(tmp_path / "per_class")
        cal.save(path)
        loaded = PerClassCalibrator.load(path)
        assert isinstance(loaded, PerClassCalibrator)
        np.testing.assert_allclose(original, loaded.transform(scores, classes=classes), atol=1e-6)


def test_registry_builds_per_class():
    cal = build_calibrator(CalibrationConfig(kind="per-class", params={"inner": "beta", "min_support": 10}))
    assert isinstance(cal, PerClassCalibrator)


def test_registry_per_class_unknown_inner_raises():
    with pytest.raises(ValueError):
        build_calibrator(CalibrationConfig(kind="per-class", params={"inner": "bogus"}))


# --------------------------------------------------------------------------- #
# T45 follow-up — fit_calibration_and_abstention actually threads `classes`
# --------------------------------------------------------------------------- #
class _RawScoreFusion:
    """A FusionModel double whose raw score is just the `f1` feature column,
    so the calibrator sees exactly the scores this test constructs."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]


def test_fit_calibration_and_abstention_wires_classes_into_per_class_fit():
    from text_classifier.application.training import fit_calibration_and_abstention

    rng = np.random.default_rng(0)
    n_per_class = 200
    scores0 = rng.uniform(0, 1, n_per_class)
    correct0 = rng.binomial(1, scores0).astype(np.float64)
    scores1 = rng.uniform(0, 1, n_per_class)
    correct1 = rng.binomial(1, scores1**3).astype(np.float64)

    ca = pd.DataFrame(
        {
            "item_id": np.arange(2 * n_per_class),
            "candidate": np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int),
            "f1": np.concatenate([scores0, scores1]),
            "is_true": np.concatenate([correct0, correct1]),
        }
    )

    calibrator, _ = fit_calibration_and_abstention(
        ca,
        _RawScoreFusion(),
        CalibrationConfig(kind="per-class", params={"inner": "beta", "min_support": 50}),
        feature_names=["f1"],
        target_precision=0.5,
        per_class_min_support=1,
    )

    assert isinstance(calibrator, PerClassCalibrator)
    # Both classes cleared min_support -> each got its own inner calibrator,
    # not just the global fallback.
    assert set(calibrator._by_class) == {0, 1}
