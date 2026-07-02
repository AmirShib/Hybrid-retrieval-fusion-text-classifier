"""T44 — XGBRankerFusionModel: a learning-to-rank FusionModel backend.

XGBRanker optimizes a pairwise ranking loss; an isotonic head turns raw scores
into probabilities so it satisfies the same predict_proba contract. Mirrors the
T05 contract plus the groups plumbing and the widened fit signature.

xgboost is a hard dependency, so these always run.
"""
from __future__ import annotations

import numpy as np
import pytest

from text_classifier.config import FusionConfig
from text_classifier.domain import FusionModel
from text_classifier.infrastructure import build_fusion
from text_classifier.infrastructure.fusion import XGBRankerFusionModel


_FAST = {"n_estimators": 30, "max_depth": 3, "random_state": 0}


def _grouped_data(n_groups: int = 40, size: int = 5, nan_frac: float = 0.0, seed: int = 0):
    """Return (X, y, groups): one positive per group; feature 0 marks it."""
    rng = np.random.default_rng(seed)
    groups = np.full(n_groups, size, dtype=np.int64)
    n = int(groups.sum())
    X = np.zeros((n, 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    idx = 0
    for _ in range(n_groups):
        pos = int(rng.integers(0, size))
        for j in range(size):
            X[idx, 0] = 1.0 if j == pos else 0.0
            X[idx, 1] = rng.standard_normal()
            y[idx] = 1 if j == pos else 0
            idx += 1
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    return X, y, groups


def test_needs_groups_flag():
    assert XGBRankerFusionModel.NEEDS_GROUPS is True


def test_fit_predict_shape_and_range():
    X, y, g = _grouped_data()
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=g)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y),)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_positive_scores_higher_overall():
    X, y, g = _grouped_data(n_groups=60)
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=g)
    proba = m.predict_proba(X)
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_nan_tolerance():
    X, y, g = _grouped_data(n_groups=60, nan_frac=0.2)
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=g)
    proba = m.predict_proba(X)
    assert np.all(np.isfinite(proba))
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_ranks_positive_above_half_of_negatives_in_group():
    X, y, g = _grouped_data(n_groups=50, size=5, seed=3)
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=g)
    raw = m.predict_proba(X)
    wins, total = 0, 0
    start = 0
    for size in g:
        block = raw[start:start + size]
        labels = y[start:start + size]
        pos_score = block[labels == 1][0]
        negs = block[labels == 0]
        wins += int((pos_score > negs).sum())
        total += len(negs)
        start += size
    # The positive beats well over half of the negatives across groups.
    assert wins / total > 0.5


def test_groups_none_raises():
    X, y, _ = _grouped_data()
    with pytest.raises(ValueError, match="requires `groups`"):
        XGBRankerFusionModel(_FAST).fit(X, y)


def test_groups_wrong_sum_raises():
    X, y, _ = _grouped_data(n_groups=10, size=5)  # len 50
    with pytest.raises(ValueError, match="must sum to len"):
        XGBRankerFusionModel(_FAST).fit(X, y, groups=np.array([5, 5, 5]))  # sums to 15


def test_heterogeneous_group_sizes():
    rng = np.random.default_rng(1)
    sizes = np.array([2, 5, 3, 8, 4], dtype=np.int64)
    n = int(sizes.sum())
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    start = 0
    for s in sizes:  # one positive per group
        y[start] = 1
        X[start, 0] = 5.0
        start += s
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=sizes)
    assert m.predict_proba(X).shape == (n,)


def test_save_load_roundtrip(tmp_path):
    X, y, g = _grouped_data(n_groups=60)
    m = XGBRankerFusionModel(_FAST)
    m.fit(X, y, groups=g)
    original = m.predict_proba(X)
    d = str(tmp_path / "fusion_ranker")
    m.save(d)
    loaded = XGBRankerFusionModel.load(d).predict_proba(X)
    np.testing.assert_allclose(original, loaded, atol=1e-6)


def test_predict_before_fit_raises():
    with pytest.raises(AssertionError):
        XGBRankerFusionModel(_FAST).predict_proba(np.zeros((5, 3), dtype=np.float32))


def test_registry_builds_xgbranker():
    model = build_fusion(FusionConfig(kind="xgbranker", params=dict(_FAST)))
    assert isinstance(model, XGBRankerFusionModel)
    assert isinstance(model, FusionModel)


# ---------------------------------------------------------------------------
# Determinism (T26)
# ---------------------------------------------------------------------------


def test_unseeded_fits_deterministic():
    """random_state absent from params -> setdefault seeds it; two fits agree."""
    X, y, g = _grouped_data(nan_frac=0.1)
    unseeded = {"n_estimators": 30, "max_depth": 3, "subsample": 0.5, "n_jobs": 1}
    runs = []
    for _ in range(2):
        m = XGBRankerFusionModel(dict(unseeded))
        m.fit(X, y, groups=g)
        runs.append(m.predict_proba(X))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert m._model.get_params()["random_state"] == 0


def test_explicit_seed_wins_ranker():
    X, y, g = _grouped_data()
    m = XGBRankerFusionModel({"n_estimators": 10, "random_state": 123})
    m.fit(X, y, groups=g)
    assert m._model.get_params()["random_state"] == 123
