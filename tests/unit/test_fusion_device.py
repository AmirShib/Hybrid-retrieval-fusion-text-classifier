"""XGBoost prediction keeps the data on the booster's device.

On a CUDA booster, handing XGBoost a host array logs "Falling back to prediction
using DMatrix due to mismatched devices" on every call. These tests pin the
routing (no GPU needed: the GPU routes are exercised with stand-ins) and that the
CPU route is numerically identical to the previous ``predict_proba`` path."""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from text_classifier.infrastructure.fusion import (  # noqa: E402
    XGBoostFusionModel,
    xgb_predict_values,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan  # NaN = "signal did not retrieve"
    y = (np.nan_to_num(X[:, 0]) + 0.3 * np.nan_to_num(X[:, 1]) > 0).astype(int)
    m = XGBoostFusionModel(
        {"n_estimators": 30, "max_depth": 3, "tree_method": "hist", "device": "cpu"}
    )
    m.fit(X, y)
    return m, X


def test_cpu_route_matches_sklearn_predict_proba_exactly(fitted):
    m, X = fitted
    m.set_device("cpu")
    expected = m._model.predict_proba(X)[:, 1]
    np.testing.assert_array_equal(m.predict_proba(X), expected.astype(np.float64))


class _Booster:
    def __init__(self):
        self.calls = []

    def inplace_predict(self, data, predict_type="value"):
        self.calls.append(("inplace", type(data).__name__))
        return np.zeros(len(data))

    def predict(self, dmatrix):
        self.calls.append(("dmatrix", type(dmatrix).__name__))
        return np.zeros(dmatrix.num_row())


class _FakeCupy:
    class ndarray(np.ndarray):
        pass

    @staticmethod
    def asarray(x):
        return np.asarray(x).view(_FakeCupy.ndarray)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)


def test_cuda_with_cupy_moves_data_to_the_gpu_first():
    b = _Booster()
    xgb_predict_values(b, np.ones((5, 3)), "cuda:0", cupy=_FakeCupy)
    assert b.calls == [("inplace", "ndarray")]  # the device array, not host data


def test_cuda_without_cupy_uses_an_explicit_dmatrix(monkeypatch):
    import text_classifier.infrastructure.fusion as fusion

    monkeypatch.setattr(fusion, "_optional_cupy", lambda: None)
    b = _Booster()
    out = xgb_predict_values(b, np.ones((5, 3)), "cuda")
    assert b.calls == [("dmatrix", "DMatrix")]  # no host array to inplace_predict
    assert out.dtype == np.float64 and out.shape == (5,)


def test_cpu_never_builds_a_dmatrix():
    b = _Booster()
    xgb_predict_values(b, np.ones((5, 3)), "cpu")
    assert b.calls == [("inplace", "ndarray")]


def test_calibration_ties_are_broken_by_the_raw_fusion_score():
    """Isotonic plateaus give an item's top candidates the same calibrated
    confidence; the reported top-1 must be the one the fusion model scored
    higher, not the lower class index."""
    import pandas as pd

    from text_classifier.application.scoring import rank_candidates, top_per_item

    scored = pd.DataFrame(
        {
            "item_id": [0, 0, 0],
            "candidate": [0, 1, 2],  # class 0 would win a pure index tie-break
            "conf": [0.9, 0.9, 0.2],
            "raw_score": [0.71, 0.88, 0.10],
        }
    )
    assert int(top_per_item(scored)["candidate"].iloc[0]) == 1
    assert rank_candidates(scored)["candidate"].tolist() == [1, 0, 2]
