"""Unit tests for application/importance.py: pure functions over a small,
hand-built (item, candidate) feature frame with fake fusion/calibrator/policy
doubles, so the arithmetic is checkable by hand instead of trusting XGBoost
internals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from text_classifier.application.importance import ablation_report, global_feature_importance

FEATURES = ["f_helpful", "f_useless"]


class _SumFusion:
    """predict_proba = sigmoid-free stand-in: sum of present features (NaN->0),
    scaled to stay in [0, 1]. predict_contribs returns each feature's raw
    contribution (itself, NaN->0) plus a zero bias column, so contributions sum
    to the same "raw score" predict_proba is derived from — enough to unit-test
    the aggregation math without needing a real additive model.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.nan_to_num(X, nan=0.0).sum(axis=1)
        return np.clip(raw, 0.0, 1.0)

    def predict_contribs(self, X: np.ndarray) -> np.ndarray:
        contribs = np.nan_to_num(X, nan=0.0)
        bias = np.zeros((X.shape[0], 1))
        return np.concatenate([contribs, bias], axis=1)


class _NoContribFusion(_SumFusion):
    def predict_contribs(self, X: np.ndarray) -> None:
        return None


class _IdentityCalibrator:
    def transform(self, raw: np.ndarray, *, classes=None) -> np.ndarray:
        return raw


class _FixedThreshold:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def accept(self, confidence: np.ndarray, class_index: np.ndarray) -> np.ndarray:
        return confidence >= self.threshold


def _feats() -> pd.DataFrame:
    # Two items, two candidates each, candidate 0 is always the true class.
    # `f_helpful` decides the winner on its own (0.5 vs 0.1 comfortably beats
    # `f_useless`'s spread); `f_useless` alone would flip both items to the
    # wrong candidate, so masking `f_helpful` should visibly hurt accuracy
    # while masking `f_useless` should not move it at all.
    return pd.DataFrame(
        {
            "item_id": [0, 0, 1, 1],
            "candidate": [0, 1, 0, 1],
            "f_helpful": [0.5, 0.1, 0.5, 0.1],
            "f_useless": [0.1, 0.45, 0.05, 0.4],
        }
    )


def test_global_feature_importance_ranks_and_shares_sum_to_one():
    feats = _feats()
    X = feats[FEATURES].to_numpy(dtype=np.float32)
    rows = global_feature_importance(_SumFusion(), X, FEATURES)
    assert rows is not None
    assert [r["feature"] for r in rows] == ["f_helpful", "f_useless"]
    assert pytest.approx(sum(r["share"] for r in rows), abs=1e-9) == 1.0
    for r in rows:
        assert r["mean_abs_contribution"] >= 0


def test_global_feature_importance_none_when_backend_lacks_contribs():
    feats = _feats()
    X = feats[FEATURES].to_numpy(dtype=np.float32)
    assert global_feature_importance(_NoContribFusion(), X, FEATURES) is None


def test_ablation_masking_helpful_feature_drops_accuracy():
    feats = _feats()
    true_idx_by_item = np.array([0, 0], dtype=np.intp)
    fusion, calibrator = _SumFusion(), _IdentityCalibrator()
    # threshold low enough that everything is accepted; isolates the accuracy
    # effect from the abstention effect.
    abstention = _FixedThreshold(threshold=0.0)

    report = ablation_report(feats, fusion, calibrator, abstention, FEATURES, true_idx_by_item)

    assert report["baseline"]["accuracy_if_no_abstain"] == 1.0
    by_feature = {r["feature"]: r for r in report["ablations"]}

    # f_useless alone flips both items to the wrong candidate, so masking the
    # feature that was actually deciding the vote drops accuracy to zero.
    assert by_feature["f_helpful"]["delta_accuracy_if_no_abstain"] == pytest.approx(-1.0)
    assert by_feature["f_helpful"]["accuracy_if_no_abstain"] == 0.0
    assert by_feature["f_helpful"]["n_rows_masked"] == 4

    # f_useless was never decisive, so ablating it changes nothing.
    assert by_feature["f_useless"]["delta_accuracy_if_no_abstain"] == pytest.approx(0.0)

    # The most damaging removal is reported first.
    assert report["ablations"][0]["feature"] == "f_helpful"


def test_ablation_skips_all_missing_feature():
    feats = _feats()
    feats["f_always_missing"] = np.nan
    true_idx_by_item = np.array([0, 0], dtype=np.intp)
    report = ablation_report(
        feats,
        _SumFusion(),
        _IdentityCalibrator(),
        _FixedThreshold(threshold=0.0),
        [*FEATURES, "f_always_missing"],
        true_idx_by_item,
    )
    assert "f_always_missing" not in {r["feature"] for r in report["ablations"]}
