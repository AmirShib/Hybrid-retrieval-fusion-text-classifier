"""T02 — Domain unit tests.

The domain layer is pure and deterministic (no IO, no randomness), so these
tests assert *exact* values and *exact* exceptions per the determinism contract
in tests/conftest.py.

Covers:
  - text_classifier/domain/models.py  (LabelSpace + frozen value objects)
  - text_classifier/domain/services.py (CandidatePolicy, AbstentionPolicy,
    ThresholdTuner, FEATURE_NAMES)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from text_classifier.domain import (
    AbstentionPolicy,
    CandidatePolicy,
    ClassDefinition,
    CoverageReport,
    FEATURE_NAMES,
    LabeledItem,
    LabelSpace,
    Prediction,
    ThresholdTuner,
)


# --------------------------------------------------------------------------- #
# LabelSpace
# --------------------------------------------------------------------------- #
PAIRS = [("alpha", "the alpha class"), ("beta", "the beta class"), ("gamma", "the gamma class")]


def _space_from_pairs() -> LabelSpace:
    return LabelSpace.from_pairs(PAIRS)


def _space_from_defs() -> LabelSpace:
    return LabelSpace([ClassDefinition(k, d) for k, d in PAIRS])


def test_from_pairs_equivalent_to_constructor():
    a, b = _space_from_pairs(), _space_from_defs()
    assert a.keys == b.keys
    assert a.descriptions == b.descriptions
    assert a.size == b.size


def test_len_size_keys_descriptions_order():
    s = _space_from_pairs()
    assert len(s) == 3
    assert s.size == 3
    assert s.keys == ["alpha", "beta", "gamma"]
    assert s.descriptions == ["the alpha class", "the beta class", "the gamma class"]


def test_index_key_round_trip():
    s = _space_from_pairs()
    for i in range(s.size):
        assert s.index_of(s.key_at(i)) == i
    for k in s.keys:
        assert s.key_at(s.index_of(k)) == k


def test_encode_labels_order_preserved():
    s = _space_from_pairs()
    assert s.encode_labels(["gamma", "alpha", "beta", "alpha"]) == [2, 0, 1, 0]


def test_empty_definitions_raises_value_error():
    with pytest.raises(ValueError):
        LabelSpace([])


def test_duplicate_keys_raises_value_error():
    with pytest.raises(ValueError):
        LabelSpace.from_pairs([("dup", "a"), ("dup", "b")])


def test_index_of_unknown_key_raises_key_error():
    s = _space_from_pairs()
    with pytest.raises(KeyError):
        s.index_of("does-not-exist")


def test_key_at_out_of_range_raises_index_error():
    s = _space_from_pairs()
    with pytest.raises(IndexError):
        s.key_at(99)


def test_encode_labels_unknown_key_raises_key_error():
    s = _space_from_pairs()
    with pytest.raises(KeyError):
        s.encode_labels(["alpha", "ghost"])


# --------------------------------------------------------------------------- #
# Frozen value objects
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "obj, attr, value",
    [
        (ClassDefinition("k", "d"), "key", "other"),
        (LabeledItem("text", "label"), "label", "x"),
        (Prediction(top_key="t", confidence=0.5, abstained=False), "confidence", 0.9),
        (
            CoverageReport(
                coverage=1.0,
                accuracy_on_accepted=1.0,
                accuracy_if_no_abstain=1.0,
                candidate_recall=1.0,
                n_items=10,
            ),
            "n_items",
            0,
        ),
    ],
)
def test_value_objects_are_frozen(obj, attr, value):
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, value)


def test_prediction_optional_defaults():
    p = Prediction(top_key="t", confidence=0.5, abstained=True)
    assert p.predicted_key is None
    assert p.runner_up_key is None
    assert p.margin is None


# --------------------------------------------------------------------------- #
# CandidatePolicy
# --------------------------------------------------------------------------- #
def test_candidate_policy_default_and_custom():
    assert CandidatePolicy().top_n_per_signal == 10
    assert CandidatePolicy(top_n_per_signal=3).top_n_per_signal == 3


def test_candidate_policy_is_frozen():
    with pytest.raises(FrozenInstanceError):
        CandidatePolicy().top_n_per_signal = 5


# --------------------------------------------------------------------------- #
# AbstentionPolicy
# --------------------------------------------------------------------------- #
def test_threshold_for_global_and_override():
    policy = AbstentionPolicy(global_threshold=0.5, per_class={1: 0.8})
    assert policy.threshold_for(0) == 0.5  # falls back to global
    assert policy.threshold_for(1) == 0.8  # per-class override


def test_threshold_for_coerces_numpy_int():
    policy = AbstentionPolicy(global_threshold=0.5, per_class={1: 0.8})
    assert policy.threshold_for(np.int64(1)) == 0.8


def test_accept_is_elementwise_with_mixed_overrides():
    policy = AbstentionPolicy(global_threshold=0.5, per_class={1: 0.8})
    confidence = np.array([0.6, 0.7, 0.9, 0.5])
    class_index = np.array([0, 1, 1, 2])
    # thresholds: 0.5, 0.8, 0.8, 0.5  ->  [T, F, T, T] (>= inclusive)
    mask = policy.accept(confidence, class_index)
    np.testing.assert_array_equal(mask, np.array([True, False, True, True]))


def test_accept_boundary_is_inclusive():
    policy = AbstentionPolicy(global_threshold=0.5)
    confidence = np.array([0.5])
    class_index = np.array([0])
    np.testing.assert_array_equal(policy.accept(confidence, class_index), np.array([True]))


# --------------------------------------------------------------------------- #
# ThresholdTuner.threshold_for_precision
# --------------------------------------------------------------------------- #
def test_empty_input_returns_one():
    assert ThresholdTuner.threshold_for_precision(np.array([]), np.array([]), target=0.9) == 1.0


def test_all_correct_returns_lowest_confidence():
    conf = np.array([0.9, 0.5, 0.7])
    corr = np.array([1, 1, 1])
    # everything is correct -> accept all -> deepest point is the lowest confidence
    assert ThresholdTuner.threshold_for_precision(conf, corr, target=0.9) == 0.5


def test_nothing_meets_target_accepts_nothing():
    conf = np.array([0.9, 0.8])
    corr = np.array([0, 0])
    thr = ThresholdTuner.threshold_for_precision(conf, corr, target=0.99)
    assert thr == pytest.approx(0.9 + 1e-6)
    # feed back through the acceptance rule: zero items accepted
    assert int(np.sum(conf >= thr)) == 0


def test_mixed_case_exact_threshold():
    # running accuracy over sorted prefix: [1.0, 1.0, 0.667, 0.75]
    # deepest index meeting >= 0.66 is the last -> 0.6
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    corr = np.array([1, 1, 0, 1])
    assert ThresholdTuner.threshold_for_precision(conf, corr, target=0.66) == 0.6


def test_ordering_independence():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    corr = np.array([1, 1, 0, 1])
    expected = ThresholdTuner.threshold_for_precision(conf, corr, target=0.66)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(conf))
    shuffled = ThresholdTuner.threshold_for_precision(conf[perm], corr[perm], target=0.66)
    assert shuffled == expected == 0.6


def test_ties_in_confidence_do_not_crash():
    conf = np.array([0.7, 0.7, 0.7])
    corr = np.array([1, 0, 1])
    thr = ThresholdTuner.threshold_for_precision(conf, corr, target=0.6)
    # running acc: [1.0, 0.5, 0.667]; only prefixes 0 and 2 meet >=0.6, deepest is 0.7
    assert thr == 0.7


# --------------------------------------------------------------------------- #
# FEATURE_NAMES
# --------------------------------------------------------------------------- #
def test_feature_names_nonempty_and_unique():
    assert len(FEATURE_NAMES) > 0
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
