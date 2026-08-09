"""Feature importance + ablation reporting (application service).

Two complementary answers to "does this column matter?", both computed against
an *already-trained* model on a fresh labeled set — no retraining:

- **importance** (``global_feature_importance``): aggregates the same additive
  per-prediction contributions ``explain_records(..., include_contributions=True)``
  exposes for one row, across a whole dataset. Answers "how much does this
  column move the raw fusion score, on average, when it fires?".
- **ablation** (``ablation_report``): masks one feature column to ``NaN`` — the
  domain's own "signal did not retrieve this" encoding — and re-scores with the
  unchanged model. Answers "how much does removing this column actually cost in
  accuracy/coverage?". Because XGBoost/LightGBM-style backends consume ``NaN``
  natively as "missing", this is a faithful ablation, not an approximation: it is
  exactly the input the model would see if that signal genuinely failed to fire.

The two can disagree — a column can swing scores a lot on the rows it covers
while rarely flipping a decision, or the reverse — and that disagreement is
itself evidence, not a bug in either report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..domain import AbstentionPolicy, ConfidenceCalibrator, FusionModel
from .scoring import select_feature_columns, top_per_item


def global_feature_importance(
    fusion: FusionModel, X: np.ndarray, feature_names: Sequence[str]
) -> Optional[List[Dict[str, Any]]]:
    """Mean absolute per-feature contribution toward the raw (pre-calibration)
    fusion score over the rows of ``X``.

    Reuses ``FusionModel.predict_contribs`` — the same additive attribution
    ``explain_records`` exposes per prediction. Returns ``None`` when the backend
    cannot decompose its score additively (``predict_contribs`` returns
    ``None``), matching that port's "no attribution available" contract; callers
    must degrade gracefully, same as the explain path does.

    Rows are sorted by descending mean absolute contribution. ``share`` is each
    feature's fraction of the total attribution mass (bias column excluded), so
    it reads as "N% of what moves the score comes from this column".
    """
    contribs = fusion.predict_contribs(X)
    if contribs is None:
        return None
    feature_contribs = contribs[:, : len(feature_names)]
    mean_abs = np.abs(feature_contribs).mean(axis=0)
    total = float(mean_abs.sum())
    rows: List[Dict[str, Any]] = [
        {
            "feature": name,
            "mean_abs_contribution": float(mean_abs[i]),
            "share": float(mean_abs[i] / total) if total > 0 else 0.0,
        }
        for i, name in enumerate(feature_names)
    ]
    rows.sort(key=lambda r: r["mean_abs_contribution"], reverse=True)
    return rows


def _score_matrix(
    X: np.ndarray,
    item_id: np.ndarray,
    candidate: np.ndarray,
    fusion: FusionModel,
    calibrator: ConfidenceCalibrator,
    abstention: AbstentionPolicy,
    true_idx_by_item: np.ndarray,
) -> Dict[str, Any]:
    """Decision quality for one already-assembled feature matrix.

    Takes ``X`` plus the two identity columns rather than the full feature
    frame, so an ablation sweep can re-score a *mutated copy of one column*
    instead of duplicating the whole table per feature. The decision layer
    still runs through ``top_per_item`` on a three-column frame — those are the
    only columns it reads, so the collapse (including its tie-breaking) is the
    same one ``predict`` performs, not a reimplementation that could drift."""
    raw = fusion.predict_proba(X)
    conf = calibrator.transform(raw, classes=candidate)
    decided = top_per_item(pd.DataFrame({"item_id": item_id, "candidate": candidate, "conf": conf}))
    item_ids = decided["item_id"].to_numpy(dtype=np.intp)
    candidates = decided["candidate"].to_numpy(dtype=np.intp)
    confidences = decided["conf"].to_numpy(dtype=np.float64)
    accepted = abstention.accept(confidences, candidates)
    correct = candidates == true_idx_by_item[item_ids]
    n = len(item_ids)
    n_acc = int(accepted.sum())
    return {
        "n_items": n,
        "coverage": (n_acc / n) if n else None,
        "accuracy_on_accepted": float(correct[accepted].mean()) if n_acc else None,
        "accuracy_if_no_abstain": float(correct.mean()) if n else None,
    }


def ablation_report(
    feats: pd.DataFrame,
    fusion: FusionModel,
    calibrator: ConfidenceCalibrator,
    abstention: AbstentionPolicy,
    feature_names: Sequence[str],
    true_idx_by_item: np.ndarray,
    *,
    X: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Per-feature ablation against the already-trained ``fusion``/``calibrator``.

    ``feats`` is the raw assembled (item, candidate) frame the ``predict`` path
    builds: one row per surfaced candidate, an integer ``item_id`` column, an
    integer ``candidate`` (class index) column, plus the feature columns.
    ``true_idx_by_item[item_id]`` is the ground-truth class index for that item.

    For each feature, rows where it is already all-``NaN`` (never fired on this
    set) are skipped — masking a no-op column would just restate the baseline.
    Otherwise the column is set to ``NaN`` for every row, the frame is rescored
    with the unchanged model, and the resulting decision quality is compared
    against the unablated baseline.

    Returns ``{"baseline": {...}, "ablations": [...]}``; ``ablations`` is sorted
    by ``delta_accuracy_if_no_abstain`` ascending — the most damaging removals
    (largest accuracy drop) first.

    ``X`` lets a caller that has already materialized the feature matrix (in
    ``feature_names`` order, float32) hand it over instead of having this build
    a second one — ``InferencePipeline.importance_report`` needs the identical
    matrix for ``global_feature_importance``. It is masked and restored column
    by column here, so it is returned to the caller unchanged; pass ``None``
    (the default) to have it built internally.

    One matrix, masked in place, is the whole point: the previous shape of this
    loop copied the entire feature frame per feature and then copied it again
    inside scoring, which on the ~36-column core schema meant ~72 full copies of
    the table to produce one report.
    """
    # Fail fast, naming exactly what's missing, rather than a bare KeyError on
    # the column selection below.
    selected = select_feature_columns(feats, feature_names, context="ablation_report")
    if X is None:
        # copy=True: the sweep mutates this in place, and `to_numpy` is free to
        # hand back a view onto the frame's own block for a single-dtype frame.
        X = selected.to_numpy(dtype=np.float32, copy=True)
    item_id = feats["item_id"].to_numpy()
    candidate = feats["candidate"].to_numpy()

    baseline = _score_matrix(
        X, item_id, candidate, fusion, calibrator, abstention, true_idx_by_item
    )
    base_acc = baseline["accuracy_if_no_abstain"] or 0.0
    base_acc_on_acc = baseline["accuracy_on_accepted"] or 0.0
    base_cov = baseline["coverage"] or 0.0

    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        column = X[:, j]
        n_present = int(np.count_nonzero(~np.isnan(column)))
        if n_present == 0:
            continue
        saved = column.copy()
        X[:, j] = np.nan
        ablated = _score_matrix(
            X, item_id, candidate, fusion, calibrator, abstention, true_idx_by_item
        )
        X[:, j] = saved
        acc = ablated["accuracy_if_no_abstain"] or 0.0
        acc_on_acc = ablated["accuracy_on_accepted"] or 0.0
        cov = ablated["coverage"] or 0.0
        rows.append(
            {
                "feature": name,
                "n_rows_masked": n_present,
                "coverage": ablated["coverage"],
                "accuracy_on_accepted": ablated["accuracy_on_accepted"],
                "accuracy_if_no_abstain": ablated["accuracy_if_no_abstain"],
                "delta_accuracy_if_no_abstain": acc - base_acc,
                "delta_accuracy_on_accepted": acc_on_acc - base_acc_on_acc,
                "delta_coverage": cov - base_cov,
            }
        )
    rows.sort(key=lambda r: r["delta_accuracy_if_no_abstain"])
    return {"baseline": baseline, "ablations": rows}
