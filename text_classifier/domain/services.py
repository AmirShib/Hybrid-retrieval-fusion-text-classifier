"""Domain services and policies: the system's *decision rules*, independent of
how the numbers were produced. These operate on plain numpy arrays (a numeric
primitive, not a framework) and contain no IO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for type hints
    from .ports import FeatureProvider

# Canonical, ordered schema of the *core* features — the five retrieval signals'
# ~36 columns. This is the "core provider's" ``names()``: with no custom
# FeatureProviders configured it is the entire schema. When providers are active
# the *effective* schema is composed at runtime (core + each provider's names, in
# order) and persisted into ``meta.json``; that composed list — not this constant
# alone — is the source of truth inference rebuilds from. Adding/removing/
# reordering a core feature touches this list *and* ``application/features.py``.
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
    # --- competition features (T81) ---------------------------------------
    # The fusion model is *pointwise*: one row per (item, candidate), scored
    # independently. Anything about how a candidate compares to the rest of its
    # own query has to be written into the row, or the model cannot see it. The
    # rank/min-max columns above do part of that job; these do the part they
    # cannot. ``d_desc_sim=0.85`` against a runner-up of 0.84 and the same 0.85
    # against a runner-up of 0.40 are identical in every column above — and one
    # is a coin flip while the other is decided.
    #
    # ``margin_*``: this candidate's signal value minus the best *other*
    # candidate's value for the same signal. Positive only for that signal's
    # leader, where it equals the top1-top2 gap; negative elsewhere, measuring
    # how far behind the leader this candidate sits. NaN when the signal did not
    # retrieve this candidate, and NaN for a lone candidate (no competitor, so
    # no margin is defined) — both are "missing" to XGBoost.
    "margin_d_desc",
    "margin_d_proto",
    "margin_d_knn",
    "margin_b_desc",
    "margin_b_knn",
    # ``q_gap_*``: per-*query* top1-top2 gap for a signal, identical across that
    # query's rows. Redundant with ``margin_*`` on the leader's row, but new
    # information on every other row: a trailing candidate's own margin says how
    # far back it is, not whether the lead itself is contested. This is the
    # classic abstention feature — how decided is this query, before asking
    # anything about the candidate at hand.
    "q_gap_d_desc",
    "q_gap_d_knn",
    "q_gap_b_desc",
]


def composed_feature_names(providers: Sequence["FeatureProvider"] = ()) -> List[str]:
    """The effective, ordered feature schema: the core ~36 columns (``FEATURE_NAMES``)
    followed by each provider's ``names()``, in provider order.

    This is *the* column order the fusion model is trained and scored on, and it is
    persisted into ``meta.json`` at save time so inference rebuilds the identical
    order. With no providers it is exactly ``FEATURE_NAMES``. Raises ``ValueError`` on a name
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


# --------------------------------------------------------------- encoder epochs
# Metrics an encoder fine-tune can be *scored* on after each epoch, so a
# multi-epoch run can return its best epoch instead of blindly its last. All are
# "higher is better" and all are measured on a held-out slice of the fine-tuning
# items (see ``encoder_retrieval_metrics``):
#
#   desc_acc@1   -- fraction of items whose nearest class *description* is their
#                   true class. The direct analogue of the `d_desc_sim` signal,
#                   and the closest cheap proxy for downstream accuracy.
#   desc_mrr     -- mean reciprocal rank of the true class among descriptions.
#                   Smoother than acc@1, so it separates epochs on small holdouts
#                   where acc@1 plateaus.
#   desc_pos_sim -- mean cosine to the item's own class description. Moves even
#                   when no ranking changes; useful as a diagnostic, weak as a
#                   selection target (it can rise while ranking degrades).
#   knn_acc@1    -- fraction whose nearest *example* in the fine-tuning pool
#                   shares its label — the analogue of the `d_knn_*` signals.
#                   Requires re-encoding the example pool each epoch, so it is
#                   only computed when a pool is supplied.
ENCODER_SELECTION_METRICS: Tuple[str, ...] = (
    "desc_acc@1",
    "desc_mrr",
    "desc_pos_sim",
    "knn_acc@1",
)


def encoder_retrieval_metrics(
    query_emb: np.ndarray,
    desc_emb: np.ndarray,
    true_idx: np.ndarray,
    *,
    pool_emb: Optional[np.ndarray] = None,
    pool_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Score an encoder's retrieval quality on a labeled holdout.

    ``query_emb`` (n, d) are the holdout items, ``desc_emb`` (C, d) the class
    descriptions in ``LabelSpace`` order, and ``true_idx`` (n,) each item's true
    class index. All embeddings are expected L2-normalized (the package-wide
    invariant), so the dot products below *are* cosines. ``pool_emb`` (m, d) with
    ``pool_labels`` (m,) optionally adds the nearest-example metric; the pool must
    not contain the holdout items themselves, or every query self-retrieves and
    ``knn_acc@1`` reads 1.0.

    Returns a ``{metric: value}`` dict over ``ENCODER_SELECTION_METRICS`` (the
    ``knn_`` key only when a pool is given). An empty holdout yields NaNs rather
    than raising — the caller decides whether an unscoreable epoch is fatal.

    Ranks are *optimistic* on ties (``rank = #{strictly better} + 1``), while
    ``desc_acc@1`` follows ``argmax`` and so breaks ties by lowest class index.
    The two therefore disagree slightly on a degenerate encoder that maps
    everything to the same vector; neither is used to make claims about such a
    model beyond "this epoch is not the one to keep".
    """
    q = np.asarray(query_emb, dtype=np.float64)
    d = np.asarray(desc_emb, dtype=np.float64)
    t = np.asarray(true_idx, dtype=np.intp)
    metrics: Dict[str, float] = {}
    if q.shape[0] == 0 or d.shape[0] == 0:
        metrics.update({name: float("nan") for name in ENCODER_SELECTION_METRICS[:3]})
        if pool_emb is not None:
            metrics["knn_acc@1"] = float("nan")
        return metrics

    sims = q @ d.T  # (n, C) cosine to every class description
    own = sims[np.arange(sims.shape[0]), t]  # (n,) cosine to the true class
    rank = (sims > own[:, None]).sum(axis=1) + 1  # optimistic rank of the true class
    metrics["desc_acc@1"] = float((sims.argmax(axis=1) == t).mean())
    metrics["desc_mrr"] = float((1.0 / rank).mean())
    metrics["desc_pos_sim"] = float(own.mean())

    if pool_emb is not None:
        p = np.asarray(pool_emb, dtype=np.float64)
        labels = np.asarray(pool_labels, dtype=np.intp)
        if p.shape[0] == 0:
            metrics["knn_acc@1"] = float("nan")
        else:
            nearest = (q @ p.T).argmax(axis=1)  # (n,) index of the closest example
            metrics["knn_acc@1"] = float((labels[nearest] == t).mean())
    return metrics


@dataclass(frozen=True, slots=True)
class EpochSelectionPolicy:
    """Which epoch of a multi-epoch fit to keep, given the per-epoch scores.

    A pure decision rule over a history of ``{metric: value}`` dicts (one per
    epoch, in order) — it holds no model state and does no IO, so the same rule
    drives selection during training and can be re-derived from a persisted
    history afterwards.

    - ``metric`` names the entry to select on (one of ``ENCODER_SELECTION_METRICS``).
    - ``min_delta`` is how much a later epoch must beat the incumbent by to be
      considered an improvement — it suppresses churn from noise on a small
      holdout, and (with ``patience``) defines what "no progress" means.
    - ``patience`` > 0 requests early stopping after that many consecutive
      non-improving epochs; 0 trains every epoch and just picks the best.

    Ties go to the *earlier* epoch: equal measured quality from less training is
    the cheaper, less-overfit model.
    """

    metric: str = "desc_acc@1"
    min_delta: float = 0.0
    patience: int = 0

    def score(self, metrics: Mapping[str, float]) -> float:
        """This policy's metric out of one epoch's record.

        Raises ``KeyError`` naming what was available: a missing metric means the
        history was produced under a different measurement (e.g. selecting on
        ``knn_acc@1`` with no example pool supplied), which must not silently
        degrade into "no epoch is better than any other".
        """
        try:
            return float(metrics[self.metric])
        except KeyError:
            raise KeyError(
                f"epoch metric {self.metric!r} was not measured; recorded metrics: "
                f"{sorted(metrics)}"
            ) from None

    def best_epoch(self, history: Sequence[Mapping[str, float]]) -> int:
        """The 1-based epoch to keep, or 0 if no epoch produced a usable score.

        0 is a real answer, not an error: it says "selection has nothing to go
        on" (an empty history, or every score NaN), and the caller falls back to
        whatever the fit produced last.
        """
        best_epoch, best_score = 0, -math.inf
        for epoch, metrics in enumerate(history, start=1):
            score = self.score(metrics)
            if math.isnan(score):
                continue
            if best_epoch == 0 or score > best_score + self.min_delta:
                best_epoch, best_score = epoch, score
        return best_epoch

    def should_stop(self, history: Sequence[Mapping[str, float]]) -> bool:
        """Whether ``patience`` non-improving epochs have elapsed since the best."""
        if self.patience <= 0 or not history:
            return False
        best = self.best_epoch(history)
        if best == 0:
            return False
        return len(history) - best >= self.patience
