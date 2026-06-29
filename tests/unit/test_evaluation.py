"""Unit tests for the evaluation metrics.

Exact-value assertions on small hand-built arrays: these are deterministic pure
numpy, so the math is pinned (calibration, risk-coverage, per-class breakdown).
"""
from __future__ import annotations

import numpy as np
import pytest

from text_classifier.application.evaluation import (
    _json_safe,
    brier_score,
    evaluate_decisions,
    expected_calibration_error,
    per_class_table,
    reliability_table,
    risk_coverage_curve,
)


def test_brier_score_basic():
    conf = np.array([1.0, 0.0, 1.0, 0.0])
    correct = np.array([1, 0, 0, 1])
    # (0 + 0 + 1 + 1) / 4
    assert brier_score(conf, correct) == pytest.approx(0.5)


def test_brier_and_ece_on_two_bins():
    conf = np.array([0.2, 0.2, 0.8, 0.8])
    correct = np.array([0, 0, 1, 1])
    assert brier_score(conf, correct) == pytest.approx(0.04)
    # bin [0.2,0.3): conf 0.2 acc 0 -> gap .2 ; bin [0.8,0.9): conf .8 acc 1 -> gap .2
    assert expected_calibration_error(conf, correct, n_bins=10) == pytest.approx(0.2)
    table = reliability_table(conf, correct, n_bins=10)
    assert len(table) == 2
    assert {r["count"] for r in table} == {2}


def test_empty_inputs_return_none():
    assert brier_score(np.array([]), np.array([])) is None
    assert expected_calibration_error(np.array([]), np.array([])) is None
    assert risk_coverage_curve(np.array([]), np.array([])) == []


def test_risk_coverage_curve_monotone_prefix():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    curve = risk_coverage_curve(conf, correct, n_points=4)
    cov = [p["coverage"] for p in curve]
    acc = [p["accuracy"] for p in curve]
    assert cov == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert acc == pytest.approx([1.0, 1.0, 2 / 3, 0.5])
    # most-confident-first: thresholds fall as coverage rises
    assert [p["threshold"] for p in curve] == pytest.approx([0.9, 0.8, 0.7, 0.6])


def test_per_class_table_precision_recall_coverage():
    keys = ["a", "b", "c"]
    true_idx = np.array([0, 0, 1, 1, 2])
    pred_idx = np.array([0, 1, 1, 1, -1])
    accepted = np.array([True, True, True, False, False])
    correct = np.array([True, False, True, True, False])

    rows = {r["key"]: r for r in per_class_table(pred_idx, true_idx, accepted, correct, keys)}

    # class a: items 0,1 are true=a; item0 predicted a (correct), item1 predicted b (a miss)
    assert rows["a"]["support"] == 2
    assert rows["a"]["precision_on_accepted"] == pytest.approx(1.0)  # only item0 predicted a, correct
    assert rows["a"]["coverage"] == pytest.approx(1.0)              # both true-a items accepted
    assert rows["a"]["recall_on_accepted"] == pytest.approx(0.5)    # only item0 accepted+correct

    assert rows["b"]["support"] == 2
    assert rows["b"]["precision_on_accepted"] == pytest.approx(0.5)  # items 1(F),2(T) accepted
    assert rows["b"]["coverage"] == pytest.approx(0.5)               # item3 abstained
    assert rows["b"]["recall_on_accepted"] == pytest.approx(0.5)

    assert rows["c"]["support"] == 1
    assert rows["c"]["precision_on_accepted"] is None                # never predicted+accepted
    assert rows["c"]["coverage"] == pytest.approx(0.0)


def test_per_class_table_sorted_by_support():
    keys = ["a", "b", "c"]
    true_idx = np.array([0, 0, 1, 1, 2])
    pred_idx = np.array([0, 1, 1, 1, -1])
    accepted = np.ones(5, dtype=bool)
    correct = np.ones(5, dtype=bool)
    rows = per_class_table(pred_idx, true_idx, accepted, correct, keys)
    supports = [r["support"] for r in rows]
    assert supports == sorted(supports, reverse=True)


def test_evaluate_decisions_shape_and_overall():
    keys = ["a", "b"]
    conf = np.array([0.9, 0.4, 0.95, 0.1])
    correct = np.array([True, False, True, False])
    accepted = np.array([True, False, True, False])
    pred_idx = np.array([0, 1, 1, 0])
    true_idx = np.array([0, 0, 1, 1])

    rep = evaluate_decisions(
        confidence=conf, correct=correct, accepted=accepted,
        pred_idx=pred_idx, true_idx=true_idx, keys=keys, candidate_recall=0.75,
    )
    o = rep["overall"]
    assert o["n_items"] == 4
    assert o["n_accepted"] == 2
    assert o["coverage"] == pytest.approx(0.5)
    assert o["accuracy_on_accepted"] == pytest.approx(1.0)   # both accepted are correct
    assert o["accuracy_if_no_abstain"] == pytest.approx(0.5)
    assert o["candidate_recall"] == pytest.approx(0.75)
    assert set(rep) >= {"overall", "calibration", "risk_coverage_curve", "per_class"}


def test_json_safe_replaces_nonfinite_and_numpy():
    cleaned = _json_safe({"a": np.float64("nan"), "b": np.int64(3), "c": [np.float32(1.5)]})
    assert cleaned == {"a": None, "b": 3, "c": [1.5]}
