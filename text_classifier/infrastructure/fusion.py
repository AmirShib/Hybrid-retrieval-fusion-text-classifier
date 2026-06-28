"""Fusion + calibration adapters.

Three pluggable FusionModel backends (selected by FusionConfig.kind via the
registry): XGBoostFusionModel (pointwise classifier, default), LightGBMFusionModel
(pointwise, optional dependency), and XGBRankerFusionModel (learning-to-rank with
an isotonic head). All treat NaN as 'missing' natively — exactly how we encode
'signal did not retrieve this class' — so no imputation is ever needed.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..domain import ConfidenceCalibrator, FusionModel


class XGBoostFusionModel(FusionModel):
    def __init__(self, params: Dict[str, Any], auto_scale_pos_weight: bool = True):
        self._params = dict(params)
        self._auto_spw = auto_scale_pos_weight
        self._model = None  # lazily created in fit/load

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
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


class LightGBMFusionModel(FusionModel):
    """Pointwise gradient-boosting fusion on LightGBM (optional dependency).

    Mirrors XGBoostFusionModel's contract, including native NaN-as-missing and
    `auto_scale_pos_weight` for imbalance. Persists via LightGBM's portable native
    text model format. Prediction goes through the trained Booster so save/load is
    exact.
    """

    def __init__(self, params: Dict[str, Any], auto_scale_pos_weight: bool = True):
        self._params = dict(params)
        self._auto_spw = auto_scale_pos_weight
        self._booster = None  # lightgbm.Booster, set in fit/load
        self._scale_pos_weight: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
        from lightgbm import LGBMClassifier

        params = dict(self._params)
        params.setdefault("verbosity", -1)
        if self._auto_spw:
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            self._scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            params["scale_pos_weight"] = self._scale_pos_weight
        model = LGBMClassifier(**params)
        model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        self._booster = model.booster_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._booster is not None, "fusion model not fitted"
        # For binary objective the Booster yields P(class==1) directly.
        return np.asarray(self._booster.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def save(self, path: str) -> None:
        assert self._booster is not None
        self._booster.save_model(path)  # native text format

    @classmethod
    def load(cls, path: str) -> "LightGBMFusionModel":
        import lightgbm as lgb

        obj = cls(params={}, auto_scale_pos_weight=False)
        obj._booster = lgb.Booster(model_file=path)
        return obj


class XGBRankerFusionModel(FusionModel):
    """Learning-to-rank fusion: XGBoost's XGBRanker optimizes a pairwise ranking
    loss ('rank the true class above the others') instead of log-loss.

    Raw ranker scores are not probabilities, so an isotonic head maps them onto
    [0, 1] to satisfy `predict_proba`. Requires a per-query `groups` array at fit
    time (one count per item, summing to len(X)); raises if it is missing.

    Persists to a directory: the native XGBoost model plus the pickled isotonic
    head, so save/load round-trips exactly.
    """

    NEEDS_GROUPS = True

    _MODEL_NAME = "ranker.ubj"
    _ISO_NAME = "isotonic.pkl"

    def __init__(self, params: Dict[str, Any], auto_scale_pos_weight: bool = True):
        # auto_scale_pos_weight is accepted for signature parity but unused: a
        # ranking loss handles within-group imbalance directly.
        self._params = dict(params)
        self._model = None       # xgboost.XGBRanker
        self._iso: Optional[IsotonicRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
        if groups is None:
            raise ValueError(
                "XGBRankerFusionModel.fit requires `groups` (one count per query, "
                "summing to len(X)); none was provided"
            )
        from xgboost import XGBRanker

        groups = np.asarray(groups)
        if int(groups.sum()) != len(X):
            raise ValueError(
                f"`groups` must sum to len(X)={len(X)}, got {int(groups.sum())}"
            )
        X = np.asarray(X, dtype=np.float32)
        params = dict(self._params)
        params.setdefault("objective", "rank:pairwise")
        self._model = XGBRanker(**params)
        self._model.fit(X, np.asarray(y), group=groups)

        raw = np.asarray(self._model.predict(X), dtype=np.float64)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw, np.asarray(y, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None and self._iso is not None, "fusion model not fitted"
        raw = np.asarray(self._model.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64)
        return np.clip(self._iso.transform(raw), 0.0, 1.0)

    def save(self, path: str) -> None:
        assert self._model is not None and self._iso is not None
        os.makedirs(path, exist_ok=True)
        self._model.save_model(os.path.join(path, self._MODEL_NAME))
        with open(os.path.join(path, self._ISO_NAME), "wb") as fh:
            pickle.dump(self._iso, fh)

    @classmethod
    def load(cls, path: str) -> "XGBRankerFusionModel":
        from xgboost import XGBRanker

        obj = cls(params={})
        model = XGBRanker()
        model.load_model(os.path.join(path, cls._MODEL_NAME))
        obj._model = model
        with open(os.path.join(path, cls._ISO_NAME), "rb") as fh:
            obj._iso = pickle.load(fh)
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
