"""Scoring and ablation must not duplicate the feature table to read it.

``add_confidence`` deep-copied the whole frame to append one column, and
``ablation_report`` copied the frame *per feature* before calling into
``add_confidence``, which copied it again — roughly 72 full copies of the
feature table for one report on the core schema. These tests pin the cheaper
shape and, more importantly, that it did not change any number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from text_classifier.application.importance import ablation_report
from text_classifier.application.scoring import add_confidence
from text_classifier.domain import AbstentionPolicy

FEATURES = ["f0", "f1", "f2"]


class StubFusion:
    """Deterministic, order-sensitive: the score depends on every column, so a
    masked column provably changes the output (an ablation that silently did
    nothing would pass a weaker double)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isnan(X), 0.0, X)
        weights = np.array([0.5, 0.3, 0.2])[: X.shape[1]]
        return 1.0 / (1.0 + np.exp(-(filled @ weights)))


class IdentityCalibrator:
    def transform(self, raw: np.ndarray, classes=None) -> np.ndarray:
        return np.asarray(raw, dtype=np.float64)


@pytest.fixture
def feats() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 12
    frame = pd.DataFrame(
        {name: rng.standard_normal(n).astype(np.float32) for name in FEATURES},
    )
    frame["item_id"] = np.repeat(np.arange(n // 3), 3)
    frame["candidate"] = np.tile(np.arange(3), n // 3)
    # A column that never fires on this set — ablating it must be skipped.
    frame["f2"] = np.nan
    return frame


class TestAddConfidence:
    def test_caller_frame_is_not_given_a_conf_column(self, feats):
        """The shallow copy must still leave the input frame alone — the
        contract a deep copy provided."""
        before = list(feats.columns)
        out = add_confidence(feats, StubFusion(), IdentityCalibrator(), FEATURES)
        assert list(feats.columns) == before
        assert "conf" not in feats.columns
        assert "conf" in out.columns

    def test_feature_columns_are_shared_not_duplicated(self, feats):
        """Appending a column does not require copying the other 36. Verified
        by identity of the underlying buffers, not by timing."""
        out = add_confidence(feats, StubFusion(), IdentityCalibrator(), FEATURES)
        assert np.shares_memory(out["f0"].to_numpy(), feats["f0"].to_numpy())

    def test_values_are_unchanged_by_the_shallow_copy(self, feats):
        out = add_confidence(feats, StubFusion(), IdentityCalibrator(), FEATURES)
        expected = StubFusion().predict_proba(feats[FEATURES].to_numpy(np.float32))
        np.testing.assert_allclose(out["conf"].to_numpy(), expected)


class TestAblationReport:
    def _report(self, feats, X=None):
        return ablation_report(
            feats,
            StubFusion(),
            IdentityCalibrator(),
            AbstentionPolicy(global_threshold=0.5, per_class={}),
            FEATURES,
            true_idx_by_item=np.zeros(4, dtype=np.intp),
            X=X,
        )

    def test_shared_matrix_is_restored_column_by_column(self, feats):
        """The caller may hand in the matrix it also used for importance, so
        every masked column has to come back exactly as it went in."""
        X = feats[FEATURES].to_numpy(dtype=np.float32, copy=True)
        original = X.copy()
        self._report(feats, X=X)
        np.testing.assert_array_equal(np.isnan(X), np.isnan(original))
        np.testing.assert_array_equal(X[~np.isnan(X)], original[~np.isnan(original)])

    def test_passing_a_matrix_matches_building_one(self, feats):
        built = self._report(feats)
        shared = self._report(feats, X=feats[FEATURES].to_numpy(dtype=np.float32, copy=True))
        assert built == shared

    def test_input_frame_is_never_mutated(self, feats):
        snapshot = feats.copy(deep=True)
        self._report(feats)
        pd.testing.assert_frame_equal(feats, snapshot)

    def test_all_nan_column_is_skipped(self, feats):
        """`f2` is entirely NaN here: masking a column that never fired would
        only restate the baseline."""
        report = self._report(feats)
        assert [row["feature"] for row in report["ablations"]] == ["f0", "f1"] or [
            row["feature"] for row in report["ablations"]
        ] == ["f1", "f0"]

    def test_masking_actually_changes_the_decisions(self, feats):
        """Guards the in-place mask: if `X[:, j] = np.nan` were applied to a
        throwaway copy, every ablation would equal the baseline and the report
        would be quietly meaningless."""
        report = self._report(feats)
        baseline = report["baseline"]["accuracy_if_no_abstain"]
        assert any(row["accuracy_if_no_abstain"] != baseline for row in report["ablations"]), (
            "no ablation moved the metric; the column mask is not reaching the model"
        )
