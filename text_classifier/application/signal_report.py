"""Per-signal diagnostics (application service).

Pre-fusion insight into *which retrieval techniques actually carry information on
this dataset*. The five signals each nominate candidate classes and score them;
the fusion model then learns how to weigh them. Before that combination a data
scientist wants the ground truth about each signal on its own: how often would it
be right if you trusted it alone, how often does it fire at all, and how much do
the signals agree (a proxy for how much *independent* evidence fusion actually has).

This report answers that straight off the assembled feature table — no model
internals. It is designed to run on the leakage-free out-of-fold rows the fusion
model is evaluated on (each item's signals come from an index built on *other*
folds), so the numbers describe the signals as the model really sees them. It also
runs on any (item, candidate) table that carries ``item_id``, ``is_true``, and the
core signal columns (e.g. the frame ``InferencePipeline.explain`` produces plus a
ground-truth flag), so the same report serves training and standalone evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Each core retrieval signal, mapped to the assembled columns that expose it: the
# ``top1`` indicator (is this candidate that signal's own #1 pick for its item?),
# a representative raw score, and — for signals that can fail to retrieve — the
# 'missing' flag. The five keys are the five techniques a data scientist reasons
# about; the columns are the source of truth in ``domain/services.py::FEATURE_NAMES``.
SIGNALS: Dict[str, Dict[str, str]] = {
    "dense_description": {"top1": "is_d_desc_top1", "score": "d_desc_sim"},
    "dense_prototype": {"top1": "is_d_proto_top1", "score": "d_proto_sim"},
    "dense_knn": {"top1": "is_d_knn_top1", "score": "d_knn_max", "missing": "d_knn_missing"},
    "bm25_description": {
        "top1": "is_b_desc_top1",
        "score": "b_desc_sim",
        "missing": "b_desc_missing",
    },
    "bm25_knn": {"top1": "is_b_knn_top1", "score": "b_knn_max", "missing": "b_knn_missing"},
}


def _empty_report() -> Dict[str, Any]:
    return {"n_items": 0, "n_candidate_rows": 0, "per_signal": [], "agreement": {}}


def signal_report(features: pd.DataFrame) -> Dict[str, Any]:
    """Per-signal standalone diagnostics over an (item, candidate) feature table.

    ``features`` must carry ``item_id``, ``is_true`` (1 where the candidate is the
    item's true class), and the core signal columns (see ``SIGNALS``). The natural
    input is the training pipeline's out-of-fold frame; a standalone-evaluation
    frame with the same columns works identically.

    Returns a JSON-clean dict:

    - ``n_items`` / ``n_candidate_rows`` — the sample the report is computed on.
    - ``per_signal`` — one entry per signal, sorted best-first, with:
        - ``top1_accuracy``: fraction of *all* items where this signal's own top pick
          is the true class — "how good is this technique alone on your data";
        - ``fired_rate``: fraction of items where the signal nominated a top pick at
          all (a signal that rarely retrieves can still be a strong tie-breaker);
        - ``top1_precision_when_fired``: ``top1_accuracy`` restricted to items where
          the signal fired;
        - ``candidate_missing_rate`` (signals that can miss only): fraction of
          candidate rows the signal did not score (its ``NaN`` "not retrieved" rate).
    - ``agreement`` — ``mean_distinct_top_classes`` (1 = every firing signal points
      at the same class, up to 5 = all disagree) and ``consensus_rate`` (fraction of
      items where the firing signals unanimously agree).

    Empty input yields the zero-filled structure rather than raising.
    """
    if "is_true" not in features.columns:
        raise ValueError(
            "signal_report needs the ground-truth 'is_true' column "
            "(1 where a candidate is its item's true class)"
        )
    if not len(features):
        return _empty_report()

    n_items = int(features["item_id"].nunique())
    is_true = features["is_true"].to_numpy().astype(bool)

    per_signal: List[Dict[str, Any]] = []
    for name, cols in SIGNALS.items():
        flag_col = cols["top1"]
        if flag_col not in features.columns:
            continue
        # Exactly one candidate per item carries the flag — the signal's own top
        # pick — and none when the signal did not fire for that item.
        flag = features[flag_col].to_numpy().astype(bool)
        n_fired = int(flag.sum())
        n_correct = int((flag & is_true).sum())
        entry: Dict[str, Any] = {
            "signal": name,
            "top1_accuracy": n_correct / n_items,
            "fired_rate": n_fired / n_items,
            "top1_precision_when_fired": (n_correct / n_fired) if n_fired else None,
        }
        miss = cols.get("missing")
        if miss and miss in features.columns:
            entry["candidate_missing_rate"] = float(features[miss].to_numpy().mean())
        per_signal.append(entry)
    per_signal.sort(key=lambda e: e["top1_accuracy"], reverse=True)

    agreement: Dict[str, Any] = {}
    if "n_signal_agreement" in features.columns:
        # n_signal_agreement = 5 - (distinct classes the firing signals point at),
        # constant across an item's candidate rows, so one value per item.
        per_item = features.groupby("item_id", sort=False)["n_signal_agreement"].first().to_numpy()
        distinct = 5.0 - per_item.astype(np.float64)
        agreement = {
            "mean_distinct_top_classes": float(distinct.mean()),
            "consensus_rate": float((distinct <= 1.0).mean()),
        }

    return {
        "n_items": n_items,
        "n_candidate_rows": int(len(features)),
        "per_signal": per_signal,
        "agreement": agreement,
    }
