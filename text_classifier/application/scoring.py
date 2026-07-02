"""Scoring helpers shared by the training (evaluation) and inference pipelines:
turn a feature table into calibrated confidences and collapse to one decision
per item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..domain import FEATURE_NAMES, ConfidenceCalibrator, FusionModel


def add_confidence(
    features: pd.DataFrame, fusion: FusionModel, calibrator: ConfidenceCalibrator
) -> pd.DataFrame:
    """Append a calibrated `conf` column = P(candidate is correct)."""
    X = features[FEATURE_NAMES].to_numpy(dtype=np.float32)
    raw = fusion.predict_proba(X)
    out = features.copy()
    out["conf"] = calibrator.transform(raw)
    return out


def top_per_item(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per item: best candidate, its confidence, the runner-up margin,
    and (if present) whether the top candidate was correct."""
    scored = scored.sort_values(["item_id", "conf"], ascending=[True, False])
    grp = scored.groupby("item_id", sort=False)
    top = grp.head(1).copy()

    runner = grp.nth(1)[["item_id", "conf"]].rename(columns={"conf": "second_conf"})
    top = top.merge(runner, on="item_id", how="left")
    top["margin"] = top["conf"] - top["second_conf"].fillna(0.0)
    return top.reset_index(drop=True)
