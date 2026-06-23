"""Fusion + calibration adapters.

XGBoostFusionModel implements the pointwise ranking model. XGBoost treats NaN as
'missing' natively, which is exactly how we encode 'signal did not retrieve this
class', so no imputation is needed.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..domain import ConfidenceCalibrator, FusionModel


class XGBoostFusionModel(FusionModel):
    def __init__(self, params: Dict[str, Any], auto_scale_pos_weight: bool = True):
        self._params = dict(params)
        self._auto_spw = auto_scale_pos_weight
        self._model = None  # lazily created in fit/load

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from xgboost import XGBClassifier

        params = dict(self._params)
        if self._auto_spw:
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0
        self._model = XGBClassifier(**params)
        self._model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "fusion model not fitted"
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]

    def save(self, path: str) -> None:
        assert self._model is not None
        self._model.save_model(path)  # native JSON

    @classmethod
    def load(cls, path: str) -> "XGBoostFusionModel":
        from xgboost import XGBClassifier

        obj = cls(params={}, auto_scale_pos_weight=False)
        model = XGBClassifier()
        model.load_model(path)
        obj._model = model
        return obj


class IsotonicCalibrator(ConfidenceCalibrator):
    """Monotonic map from raw fusion score to P(correct)."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, correct: np.ndarray) -> None:
        self._iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(correct, dtype=np.float64))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._iso.transform(np.asarray(scores, dtype=np.float64))

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self._iso, fh)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        obj = cls()
        with open(path, "rb") as fh:
            obj._iso = pickle.load(fh)
        return obj
