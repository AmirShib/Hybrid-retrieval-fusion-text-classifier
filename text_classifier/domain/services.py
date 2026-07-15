"""Domain services and policies: the system's *decision rules*, independent of
how the numbers were produced. These operate on plain numpy arrays (a numeric
primitive, not a framework) and contain no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for type hints
    from .ports import FeatureProvider

# Canonical, ordered schema of the *core* features — the five retrieval signals'
# ~28 columns. This is the "core provider's" ``names()`` (T70): with no custom
# FeatureProviders configured it is the entire schema, byte-for-byte as before.
# When providers are active the *effective* schema is composed at runtime (core +
# each provider's names, in order) and persisted into ``meta.json``; that composed
# list — not this constant alone — is the source of truth inference rebuilds from.
# Adding/removing/reordering a core feature touches this list *and*
# ``application/features.py``.
FEATURE_NAMES: List[str] = [
    "d_desc_sim",
    "d_proto_sim",
    "d_knn_sum",
    "d_knn_max",
    "d_knn_count",
    "b_desc_sim",
    "b_knn_sum",
    "b_knn_max",
    "b_knn_count",
    "desc_proto_gap",
    "class_log_freq",
    "abs_top_dense_sim",
    "abs_top_bm25",
    "is_d_desc_top1",
    "is_d_proto_top1",
    "is_b_desc_top1",
    "is_d_knn_top1",
    "is_b_knn_top1",
    "b_desc_missing",
    "b_knn_missing",
    "d_knn_missing",
    "rank_d_desc",
    "rank_b_desc",
    "rank_d_knn",
    "rank_b_knn",
    "norm_d_desc",
    "norm_b_desc",
    "n_signal_agreement",
]


def composed_feature_names(providers: Sequence["FeatureProvider"] = ()) -> List[str]:
    """The effective, ordered feature schema: the core ~28 columns (``FEATURE_NAMES``)
    followed by each provider's ``names()``, in provider order.

    This is *the* column order the fusion model is trained and scored on, and it is
    persisted into ``meta.json`` at save time so inference rebuilds the identical
    order (T70). With no providers it is exactly ``FEATURE_NAMES`` — byte-for-byte
    the schema before custom features existed. Raises ``ValueError`` on a name
    collision (a provider colliding with a core column, or two providers colliding),
    because a duplicate column name would silently overwrite data in the assembled
    frame — the worst kind of train/infer disagreement."""
    names: List[str] = list(FEATURE_NAMES)
    seen = set(names)
    for provider in providers:
        for name in provider.names():
            if name in seen:
                raise ValueError(
                    f"feature name collision: {name!r} is contributed by "
                    f"{type(provider).__name__} but is already in the schema. Provider "
                    "column names must be unique across the core features and all providers."
                )
            seen.add(name)
            names.append(name)
    return names


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    """How many classes each signal may nominate. The candidate set is the union
    across signals, so the true ceiling on accuracy is candidate recall."""

    top_n_per_signal: int = 10


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    """A global confidence threshold plus optional per-class overrides. A class
    falls back to the global threshold when it lacked calibration support."""

    global_threshold: float
    per_class: Dict[int, float] = field(default_factory=dict)

    def threshold_for(self, class_index: int) -> float:
        return self.per_class.get(int(class_index), self.global_threshold)

    def accept(self, confidence: np.ndarray, class_index: np.ndarray) -> np.ndarray:
        thr = np.fromiter(
            (self.threshold_for(c) for c in class_index.tolist()),
            dtype=np.float64,
            count=len(class_index),
        )
        return confidence >= thr


class ThresholdTuner:
    """Picks the lowest threshold (max coverage) that still meets a target
    accuracy on accepted items."""

    @staticmethod
    def threshold_for_precision(
        confidence: np.ndarray, correct: np.ndarray, target: float
    ) -> float:
        if len(confidence) == 0:
            return 1.0
        order = np.argsort(-confidence)
        conf = confidence[order]
        corr = correct[order].astype(np.float64)
        running_acc = np.cumsum(corr) / np.arange(1, len(corr) + 1)
        acceptable = np.where(running_acc >= target)[0]
        if acceptable.size == 0:
            return float(conf[0] + 1e-6)  # nothing meets target -> accept nothing
        return float(conf[acceptable[-1]])  # deepest acceptable point
