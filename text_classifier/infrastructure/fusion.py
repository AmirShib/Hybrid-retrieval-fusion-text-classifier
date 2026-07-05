"""Fusion + calibration adapters.

Three pluggable FusionModel backends (selected by FusionConfig.kind via the
registry): XGBoostFusionModel (pointwise classifier, default), LightGBMFusionModel
(pointwise, optional dependency), and XGBRankerFusionModel (learning-to-rank with
an isotonic head). All treat NaN as 'missing' natively — exactly how we encode
'signal did not retrieve this class' — so no imputation is ever needed.

Three pluggable ConfidenceCalibrator backends (selected by CalibrationConfig.kind):
IsotonicCalibrator (non-parametric, default), PlattCalibrator (sigmoid), and
BetaCalibrator (three-parameter beta calibration). The parametric pair share a
base class and tend to be more robust than isotonic on a small calibration fold.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..domain import ConfidenceCalibrator, FusionModel


class XGBoostFusionModel(FusionModel):
    def __init__(self, params: Dict[str, Any], auto_scale_pos_weight: bool = True):
        self._params = dict(params)
        self._auto_spw = auto_scale_pos_weight
        self._model: Any = None  # lazily created in fit/load (xgboost.XGBClassifier)

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
        from xgboost import XGBClassifier

        params = dict(self._params)
        # Deterministic by default: subsample/colsample draw from an RNG, and an
        # unseeded run cannot be reproduced. An explicit user seed always wins.
        params.setdefault("random_state", 0)
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
        self._booster: Any = None  # lightgbm.Booster, set in fit/load
        self._scale_pos_weight: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, groups: Optional[np.ndarray] = None) -> None:
        from lightgbm import LGBMClassifier

        params = dict(self._params)
        params.setdefault("verbosity", -1)
        params.setdefault("random_state", 0)
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
        self._model: Any = None  # xgboost.XGBRanker
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
            raise ValueError(f"`groups` must sum to len(X)={len(X)}, got {int(groups.sum())}")
        X = np.asarray(X, dtype=np.float32)
        params = dict(self._params)
        params.setdefault("objective", "rank:pairwise")
        params.setdefault("random_state", 0)
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


class _ParametricCalibrator(ConfidenceCalibrator):
    """Shared machinery for the logistic-regression-based calibrators.

    A subclass only has to map raw scores to the design matrix it regresses on
    (``_features``); fitting, the single-class fallback, prediction, and
    persistence are common. The single-class fallback matters because exactly
    one fold is held out for calibration: if that fold happens to contain only
    correct (or only incorrect) decisions, ``LogisticRegression`` cannot fit, so
    we fall back to the constant empirical base rate rather than raising.
    """

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None
        self._constant: Optional[float] = None

    def _features(self, scores: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, scores: np.ndarray, correct: np.ndarray) -> None:
        y = np.asarray(correct)
        if np.unique(y).size < 2:
            # Only one class observed → no logistic fit is possible; the honest
            # estimate of P(correct) is the constant base rate.
            self._constant = float(y.mean()) if y.size else 0.5
            self._lr = None
            return
        self._lr = LogisticRegression(max_iter=1000)
        self._lr.fit(self._features(scores), y)
        self._constant = None

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if self._constant is not None:
            return np.full(scores.shape, self._constant, dtype=np.float64)
        assert self._lr is not None, "calibrator not fitted"
        p = self._lr.predict_proba(self._features(scores))[:, 1]
        return np.clip(p, 0.0, 1.0)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"lr": self._lr, "constant": self._constant}, fh)

    @classmethod
    def load(cls, path: str) -> "_ParametricCalibrator":
        obj = cls()
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj._lr = state["lr"]
        obj._constant = state["constant"]
        return obj


class PlattCalibrator(_ParametricCalibrator):
    """Platt scaling: a logistic (sigmoid) fit from the raw fusion score to
    P(correct). Parametric and smooth, so it is more robust than isotonic on a
    small calibration fold — at the cost of assuming a sigmoidal reliability
    curve. Monotonic in the score by construction (single non-negative slope on
    well-behaved data)."""

    def _features(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64).reshape(-1, 1)


class BetaCalibrator(_ParametricCalibrator):
    """Beta calibration (Kull et al., 2017): a three-parameter generalization of
    Platt that regresses on the features ``[ln(x), -ln(1-x)]``. It fits
    asymmetric/skewed reliability curves the plain sigmoid cannot, and stays
    monotonic when the fitted coefficients are non-negative (the usual case when
    the score correlates with correctness).

    Self-contained — no dependency beyond scikit-learn (already required), so it
    works on air-gapped hosts. Scores are clipped away from {0, 1} before the log
    transform so 0/1 inputs stay finite."""

    _EPS = 1e-12

    def _features(self, scores: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(scores, dtype=np.float64), self._EPS, 1.0 - self._EPS)
        # -log1p(-x) == -ln(1 - x), evaluated stably near x = 0.
        return np.column_stack([np.log(x), -np.log1p(-x)])
