"""Scoring helpers shared by the training (evaluation) and inference pipelines:
turn a feature table into calibrated confidences and collapse to one decision
per item.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..domain import FEATURE_NAMES, ConfidenceCalibrator, FusionModel


def select_feature_columns(df: pd.DataFrame, names: Sequence[str], *, context: str) -> pd.DataFrame:
    """``df[names]``, but a name absent from ``df.columns`` raises a clear,
    actionable error instead of pandas' bare ``KeyError: "[...] not in index"``.

    A mismatch here means the assembled frame doesn't carry every column the
    fusion model expects to fit/score on -- e.g. a ``SignalProvider``'s
    ``column_names()``/a ``FeatureProvider``'s ``names()`` declared a column its
    ``build()``/``transform()`` didn't actually produce, or ``names`` was
    computed against a different ``signals``/provider configuration than the
    one that actually ran the assembly. Both are configuration/contract bugs,
    not an expected runtime state, so this fails fast naming exactly what's
    missing rather than surfacing pandas' indexer traceback."""
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: {len(missing)} expected feature column(s) are missing from the "
            f"assembled frame: {missing}. This means a SignalProvider/FeatureProvider "
            "declared a column its build/transform step didn't actually produce, or the "
            "requested feature schema doesn't match the signals/providers that ran "
            f"(check `signals`/`drop_features` in the config). Columns present: "
            f"{list(df.columns)}"
        )
    return df[list(names)]


def add_confidence(
    features: pd.DataFrame,
    fusion: FusionModel,
    calibrator: ConfidenceCalibrator,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append a calibrated `conf` column = P(candidate is correct).

    ``feature_names`` is the effective feature schema (core + any custom-provider
    columns); it must match the order the fusion model was trained on. Defaults to
    the core ``FEATURE_NAMES`` so callers with no custom providers are unaffected."""
    cols = list(feature_names) if feature_names is not None else FEATURE_NAMES
    X = select_feature_columns(features, cols, context="add_confidence").to_numpy(dtype=np.float32)
    raw = fusion.predict_proba(X)
    out = features.copy()
    out["conf"] = calibrator.transform(raw, classes=features["candidate"].to_numpy())
    # Kept for ranking: isotonic calibration is a step function, so an item's
    # top candidates often share one calibrated value. Ties are then broken by
    # the fusion model's own score, not by class index (see `_RANK_KEYS`).
    out[RAW_SCORE] = np.asarray(raw, dtype=np.float64)
    return out


RAW_SCORE = "raw_score"


def _rank_order(scored: pd.DataFrame) -> pd.DataFrame:
    """Sort by item, then calibrated confidence, then raw fusion score (desc).

    Without the raw-score key, candidates tied on calibrated confidence kept
    their assembly order -- class index ascending -- so the reported top-1 for a
    tie was the alphabetically-first label rather than the one the fusion model
    scored higher (measured: ~1% of items on the Hebrew COICOP data). A frame
    without ``raw_score`` (built elsewhere) falls back to the old order."""
    keys = ["item_id", "conf"] + ([RAW_SCORE] if RAW_SCORE in scored.columns else [])
    asc = [True] + [False] * (len(keys) - 1)
    return scored.sort_values(keys, ascending=asc, kind="stable")


def rank_candidates(scored: pd.DataFrame, k: Optional[int] = None) -> pd.DataFrame:
    """``scored`` sorted by (item, confidence descending) with a 1-based ``rank``
    column added per item, optionally truncated to each item's best ``k``.

    Every consumer of a scored frame — ``top_k_per_item``, ``explain``,
    ``explain_records`` — needs exactly this, and each used to spell it out
    again. The tie-breaking is part of the contract, not an accident: equal
    calibrated confidences are ordered by the raw fusion score, and only an exact
    tie on both keeps assembly order (candidate-index ascending, stable sort).
    One shared ordering (``_rank_order``) is what makes ``rank`` and the reported
    top-1 agree across every call site. The index is reset so
    positional row numbers align with any array computed from the result (the
    contributions matrix in ``explain_records`` relies on this).
    """
    ranked = _rank_order(scored).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("item_id", sort=False).cumcount() + 1
    if k is not None:
        ranked = ranked[ranked["rank"] <= k].reset_index(drop=True)
    return ranked


def top_per_item(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per item: best candidate, its confidence, the runner-up's
    identity + margin, and (if present) whether the top candidate was correct.

    ``second_candidate`` is NaN (not an int) for single-candidate items — the
    merge is ``how="left"`` — so it stays a float column; consumers must check
    for NaN before indexing into it.
    """
    scored = _rank_order(scored)
    grp = scored.groupby("item_id", sort=False)
    top = grp.head(1).copy()

    runner = grp.nth(1)[["item_id", "candidate", "conf"]].rename(
        columns={"candidate": "second_candidate", "conf": "second_conf"}
    )
    top = top.merge(runner, on="item_id", how="left")
    top["margin"] = top["conf"] - top["second_conf"].fillna(0.0)
    return top.reset_index(drop=True)


def top_k_per_item(scored: pd.DataFrame, k: int) -> pd.DataFrame:
    """Long format: up to ``k`` rows per item, ranked by confidence descending.
    Columns: ``item_id``, ``rank`` (1-based), ``candidate``, ``conf``. Items
    with fewer than ``k`` scored candidates contribute fewer rows."""
    return rank_candidates(scored, k)[["item_id", "rank", "candidate", "conf"]]
