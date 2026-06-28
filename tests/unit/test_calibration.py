"""T42 — Platt + beta calibrator tests.

Covers PlattCalibrator and BetaCalibrator in infrastructure/fusion.py, alongside
the existing IsotonicCalibrator (tested in test_fusion.py). Fully offline.
"""
import numpy as np
import pytest

from text_classifier.config import CalibrationConfig
from text_classifier.domain import ConfidenceCalibrator
from text_classifier.infrastructure import build_calibrator
from text_classifier.infrastructure.fusion import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)

_PARAMETRIC = [PlattCalibrator, BetaCalibrator]


def _miscalibrated(n: int = 4000, seed: int = 0):
    """Systematically over-confident raw scores: the reported score is ``s**2``
    but the true P(correct | s) is ``s``. A good calibrator should pull ``s**2``
    back toward ``s`` and lower the Brier score."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, n)
    raw = s ** 2
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
