"""Evaluation metrics for the abstaining classifier (application service).

Produces the evidence a data scientist needs to *trust* a trained model, beyond
the single coverage/accuracy headline:

- an **overall** summary (coverage, accuracy on accepted, accuracy with no
  abstention, candidate recall);
- a **per-class** breakdown — essential on imbalanced data, where a healthy
  global number can hide classes the system silently abstains on or
  systematically confuses;
- **calibration** diagnostics (Brier score, expected calibration error, and a
  reliability table) — calibrated confidence is exactly what the abstention
  threshold is set on, so its quality is a first-class concern;
- a **risk-coverage curve** — the central trade-off for a human-in-the-loop
  system: how accuracy on accepted items rises as coverage falls.

Everything operates on plain numpy arrays, so the same code serves both the
training pipeline's held-out test fold and the standalone ``evaluate`` use case
on a freshly labeled set. Outputs are JSON-clean (NaN/inf -> ``None``) so they
persist next to a model directory and round-trip through standard JSON.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .._version import __version__


# --------------------------------------------------------------------------- #
# JSON hygiene
# --------------------------------------------------------------------------- #
def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy scalars to Python and non-finite floats to None.

    Standard JSON has no representation for NaN/Infinity; emitting ``None`` keeps
    the persisted report valid JSON that any consumer can parse.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def brier_score(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    """Mean squared error between confidence and the 0/1 correctness outcome.

    Lower is better; 0 is perfect. Sensitive to both calibration and sharpness.
    Returns ``None`` for an empty input.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.size == 0:
        return None
    return float(np.mean((confidence - correct) ** 2))


def reliability_table(confidence: np.ndarray, correct: np.ndarray,
                      n_bins: int = 10) -> List[Dict[str, Any]]:
    """Bin items by confidence and report mean confidence vs. observed accuracy.

    A well-calibrated model has ``mean_confidence ≈ accuracy`` in every bin.
    Empty bins are omitted. Bins are the equal-width intervals of [0, 1].
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize on the interior edges so values land in 0..n_bins-1; the closed
    # right end (conf == 1.0) is folded into the last bin.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    rows: List[Dict[str, Any]] = []
    for b in range(n_bins):
        m = idx == b
        count = int(m.sum())
        if count == 0:
            continue
        mean_conf = float(confidence[m].mean())
        acc = float(correct[m].mean())
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": acc - mean_conf,
        })
    return rows


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> Optional[float]:
    """Support-weighted mean absolute gap between confidence and accuracy.

    The standard ECE: bin by confidence, take ``|accuracy - mean_confidence|``
    per bin, weight by bin size. 0 is perfectly calibrated. Returns ``None`` for
    an empty input.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.size == 0:
        return None
    table = reliability_table(confidence, correct, n_bins)
    n = confidence.size
    return float(sum(r["count"] / n * abs(r["gap"]) for r in table))


# --------------------------------------------------------------------------- #
# Risk-coverage trade-off
# --------------------------------------------------------------------------- #
def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                        n_points: int = 20) -> List[Dict[str, float]]:
    """Accuracy on accepted items as a function of coverage.

    Sort items by descending confidence; accepting the most confident ``m`` of
    ``n`` items gives coverage ``m/n`` and accuracy ``mean(correct[:m])``. The
    curve is the model-independent ceiling on what any single global threshold
    can achieve — the shape a reviewer reads to choose an operating point.

    Sampled at up to ``n_points`` coverage levels. ``threshold`` is the
    confidence of the least-confident accepted item at that coverage.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = confidence.size
    if n == 0:
        return []
    order = np.argsort(-confidence)
    conf_sorted = confidence[order]
    cum_acc = np.cumsum(correct[order]) / np.arange(1, n + 1)
    targets = np.unique(np.linspace(1, n, min(n_points, n)).round().astype(int))
    return [
        {
            "coverage": float(m / n),
            "accuracy": float(cum_acc[m - 1]),
            "threshold": float(conf_sorted[m - 1]),
        }
        for m in targets
    ]


# --------------------------------------------------------------------------- #
# Per-class breakdown
# --------------------------------------------------------------------------- #
def per_class_table(pred_idx: np.ndarray, true_idx: np.ndarray,
                    accepted: np.ndarray, correct: np.ndarray,
                    keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-class precision / recall / coverage over the accepted decisions.

    Parameters
    ----------
    pred_idx:
        Predicted class index per item (``-1`` for "no candidate").
    true_idx:
        Ground-truth class index per item.
    accepted:
        Boolean accept/abstain mask per item.
    correct:
        Boolean "top prediction matches truth" per item.
    keys:
        Class keys, indexed by class id.

    Returns
    -------
    list of dict
        One row per class that is either a true label or a prediction for some
        item, sorted by descending support. ``precision_on_accepted`` is over
        items *predicted* as the class and accepted; ``recall_on_accepted`` and
        ``coverage`` are over items whose *true* label is the class.
    """
    pred_idx = np.asarray(pred_idx)
    true_idx = np.asarray(true_idx)
    accepted = np.asarray(accepted, dtype=bool)
    correct = np.asarray(correct, dtype=bool)

    rows: List[Dict[str, Any]] = []
    for c, key in enumerate(keys):
        is_true_c = true_idx == c
        is_pred_c = pred_idx == c
        support = int(is_true_c.sum())
        n_pred = int(is_pred_c.sum())
        if support == 0 and n_pred == 0:
            continue

        accepted_pred_c = is_pred_c & accepted
        n_pred_acc = int(accepted_pred_c.sum())
        precision = float(correct[accepted_pred_c].mean()) if n_pred_acc else None

        if support:
            n_true_acc = int((is_true_c & accepted).sum())
            coverage = n_true_acc / support
            recall = int((is_true_c & accepted & correct).sum()) / support
        else:
            coverage = None
            recall = None

        rows.append({
            "key": key,
            "support": support,
            "n_predicted": n_pred,
            "n_accepted_as_class": n_pred_acc,
            "precision_on_accepted": precision,
            "recall_on_accepted": recall,
            "coverage": coverage,
        })
    rows.sort(key=lambda r: r["support"], reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# Top-level assembly
# --------------------------------------------------------------------------- #
def evaluate_decisions(
    *,
    confidence: np.ndarray,
    correct: np.ndarray,
    accepted: np.ndarray,
    pred_idx: np.ndarray,
    true_idx: np.ndarray,
    keys: Sequence[str],
    candidate_recall: Optional[float] = None,
    n_bins: int = 10,
    n_curve_points: int = 20,
) -> Dict[str, Any]:
    """Assemble the full evaluation report from per-item decision arrays.

    All arrays are aligned, one entry per item. ``confidence`` is the calibrated
    confidence of the top prediction; ``correct`` is whether that prediction
    matches the truth; ``accepted`` is the accept/abstain decision.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    n = int(confidence.size)
    n_acc = int(accepted.sum())

    overall = {
        "n_items": n,
        "n_accepted": n_acc,
        "n_abstained": n - n_acc,
        "coverage": (n_acc / n) if n else None,
        "accuracy_on_accepted": float(correct[accepted].mean()) if n_acc else None,
        "accuracy_if_no_abstain": float(correct.mean()) if n else None,
        "candidate_recall": (float(candidate_recall) if candidate_recall is not None else None),
    }
    return {
        "overall": overall,
        "calibration": {
            "brier_score": brier_score(confidence, correct),
            "expected_calibration_error": expected_calibration_error(confidence, correct, n_bins),
            "n_bins": n_bins,
            "reliability_table": reliability_table(confidence, correct, n_bins),
        },
        "risk_coverage_curve": risk_coverage_curve(confidence, correct, n_curve_points),
        "per_class": per_class_table(pred_idx, true_idx, accepted, correct, keys),
    }


def build_manifest(n_training_items: int, n_classes: int, config: Any,
                   n_evaluated: Optional[int] = None) -> Dict[str, Any]:
    """A provenance record: package version, timestamp, data shape, and config.

    Persisted alongside the metrics so a trained model can be audited later —
    what was it trained on, when, with which version and settings.
    """
    return {
        "package_version": __version__,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_training_items": int(n_training_items),
        "n_classes": int(n_classes),
        "n_evaluated": (int(n_evaluated) if n_evaluated is not None else None),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
    }


def render_model_card(manifest: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    """A short, human-readable Markdown summary of a trained model."""
    o = evaluation.get("overall", {})
    cal = evaluation.get("calibration", {})
    cfg = manifest.get("config", {}) or {}
    enc = (cfg.get("encoder") or {}).get("kind", "?")
    fus = (cfg.get("fusion") or {}).get("kind", "?")
    cal_kind = (cfg.get("calibration") or {}).get("kind", "?")
    target = (cfg.get("training") or {}).get("target_precision", "?")

    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def num(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    abst = evaluation.get("abstention", {}) or {}
    lines = [
        "# Model card",
        "",
        f"- **Package version:** {manifest.get('package_version', '?')}",
        f"- **Generated:** {manifest.get('generated_at', '?')}",
        f"- **Classes:** {manifest.get('n_classes', '?')}",
        f"- **Training items:** {manifest.get('n_training_items', '?')}",
        f"- **Encoder / fusion / calibrator:** {enc} / {fus} / {cal_kind}",
        f"- **Target accuracy on accepted:** {target}",
        "",
        "## Headline metrics (held-out)",
        "",
        f"- **Coverage:** {pct(o.get('coverage'))} "
        f"({o.get('n_accepted', '?')} accepted of {o.get('n_items', '?')})",
        f"- **Accuracy on accepted:** {pct(o.get('accuracy_on_accepted'))}",
        f"- **Accuracy if never abstaining:** {pct(o.get('accuracy_if_no_abstain'))}",
        f"- **Candidate recall (accuracy ceiling):** {pct(o.get('candidate_recall'))}",
        f"- **Expected calibration error:** {num(cal.get('expected_calibration_error'))}",
        f"- **Brier score:** {num(cal.get('brier_score'))}",
    ]
    if abst:
        lines += [
            "",
            "## Abstention thresholds",
            "",
            f"- **Global threshold:** {num(abst.get('global_threshold'))}",
            f"- **Per-class thresholds:** {abst.get('n_per_class_thresholds', 0)} "
            "class(es) had enough calibration support for their own threshold; "
            "the rest fall back to the global one.",
        ]
    lines += [
        "",
        "See `evaluation.json` for the per-class breakdown, reliability table, "
        "and risk-coverage curve.",
        "",
    ]
    return "\n".join(lines)


def write_evaluation_artifacts(directory: str, evaluation: Dict[str, Any],
                               manifest: Dict[str, Any]) -> None:
    """Write ``evaluation.json`` and ``model_card.md`` into a model directory."""
    os.makedirs(directory, exist_ok=True)
    payload = {"manifest": manifest, **evaluation}
    with open(os.path.join(directory, "evaluation.json"), "w") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
    with open(os.path.join(directory, "model_card.md"), "w") as fh:
        fh.write(render_model_card(manifest, evaluation))
