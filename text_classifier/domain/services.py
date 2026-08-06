"""Domain services and policies: the system's *decision rules*, independent of
how the numbers were produced. These operate on plain numpy arrays (a numeric
primitive, not a framework) and contain no IO.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for type hints
    from .ports import FeatureProvider, SignalProvider

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


# What each core column (or intermediate) is built from (T87). Intermediates are
# the five signal matrices plus ``candidates`` (the top-n union mask, computed
# from all five and therefore an unconditional dependency of the whole frame —
# see ``FeatureAssembler``'s "candidates is never pruned" rule). A column with an
# empty tuple is itself a terminal producer, not something built from something
# else. ``FEATURE_NAMES`` stays the source of truth for *order*; this is the
# source of truth for what triggers what. A schema-completeness test asserts
# every ``FEATURE_NAMES`` entry has a key here and every referenced node is
# itself a key (possibly with an empty tuple) — so a new feature cannot be added
# without declaring its inputs.
FEATURE_DEPS: Dict[str, Tuple[str, ...]] = {
    # ---- intermediates -----------------------------------------------------
    "dense.desc": (),
    "dense.proto": (),
    "dense.knn": (),
    "bm25.desc": (),
    "bm25.knn": (),
    "candidates": ("dense.desc", "dense.proto", "bm25.desc", "dense.knn", "bm25.knn"),
    # ---- core columns -------------------------------------------------------
    "d_desc_sim": ("dense.desc",),
    "d_proto_sim": ("dense.proto",),
    "d_knn_sum": ("dense.knn",),
    "d_knn_max": ("dense.knn",),
    "d_knn_count": ("dense.knn",),
    "b_desc_sim": ("bm25.desc",),
    "b_knn_sum": ("bm25.knn",),
    "b_knn_max": ("bm25.knn",),
    "b_knn_count": ("bm25.knn",),
    "desc_proto_gap": ("dense.desc", "dense.proto"),
    "class_log_freq": ("candidates",),
    "abs_top_dense_sim": ("dense.knn",),
    "abs_top_bm25": ("bm25.knn",),
    "is_d_desc_top1": ("dense.desc", "candidates"),
    "is_d_proto_top1": ("dense.proto", "candidates"),
    "is_b_desc_top1": ("bm25.desc", "candidates"),
    "is_d_knn_top1": ("dense.knn", "candidates"),
    "is_b_knn_top1": ("bm25.knn", "candidates"),
    "b_desc_missing": ("bm25.desc", "candidates"),
    "b_knn_missing": ("bm25.knn", "candidates"),
    "d_knn_missing": ("dense.knn", "candidates"),
    "rank_d_desc": ("dense.desc", "candidates"),
    "rank_b_desc": ("bm25.desc", "candidates"),
    "rank_d_knn": ("dense.knn", "candidates"),
    "rank_b_knn": ("bm25.knn", "candidates"),
    "norm_d_desc": ("dense.desc", "candidates"),
    "norm_b_desc": ("bm25.desc", "candidates"),
    "n_signal_agreement": (
        "dense.desc",
        "dense.proto",
        "bm25.desc",
        "dense.knn",
        "bm25.knn",
        "candidates",
    ),
    "margin_d_desc": ("dense.desc", "candidates"),
    "margin_d_proto": ("dense.proto", "candidates"),
    "margin_d_knn": ("dense.knn", "candidates"),
    "margin_b_desc": ("bm25.desc", "candidates"),
    "margin_b_knn": ("bm25.knn", "candidates"),
    "q_gap_d_desc": ("dense.desc", "candidates"),
    "q_gap_d_knn": ("dense.knn", "candidates"),
    "q_gap_b_desc": ("bm25.desc", "candidates"),
}


def _direct_signal_reqs(name: str) -> Set[str]:
    """Which of the two built-in signals (``"dense"``/``"lexical"``) a core
    column's *own* ``FEATURE_DEPS`` entry names directly (a ``dense.*``/
    ``bm25.*`` node), ignoring the shared ``"candidates"`` node.

    ``"candidates"`` is deliberately excluded: every candidate-gated column
    depends on it, but the *mask itself* is only ever built from whichever
    signals are actually active (``FeatureAssembler``'s candidate union), not
    both unconditionally — so it carries no signal-specific information for
    this purpose. Derived from ``FEATURE_DEPS`` (not a second hardcoded list)
    so a new dense/bm25 column is classified automatically, the same "single
    source of truth" discipline ``FEATURE_DEPS`` itself documents."""
    reqs: Set[str] = set()
    for dep in FEATURE_DEPS.get(name, ()):
        if dep.startswith("dense."):
            reqs.add("dense")
        elif dep.startswith("bm25."):
            reqs.add("lexical")
    return reqs


def core_feature_names(active_signals: Optional[Iterable[str]] = None) -> List[str]:
    """``FEATURE_NAMES`` narrowed to the columns computable from
    ``active_signals`` (a name set that may include ``"dense"``/``"lexical"``
    plus any third-party signal kind -- only the two built-ins matter here).

    ``None`` (the default) means both built-ins are active and returns
    ``FEATURE_NAMES`` verbatim -- the byte-for-byte-identical default schema.
    A column that needs a signal absent from ``active_signals`` (e.g.
    ``b_desc_sim`` without ``"lexical"``) is dropped; a column needing neither
    (``class_log_freq``) always survives; ``n_signal_agreement`` needs both,
    so it drops whenever either built-in is disabled -- with only one signal
    active, "how many signals agree" carries no information."""
    if active_signals is None:
        return list(FEATURE_NAMES)
    active = set(active_signals)
    return [n for n in FEATURE_NAMES if _direct_signal_reqs(n) <= active]


def feature_closure(requested: Iterable[str]) -> Set[str]:
    """The transitive closure of ``requested`` over ``FEATURE_DEPS``, plus the
    unconditional ``candidates`` dependency (and everything it pulls in).

    The result mixes column names and intermediate node names — callers that
    gate a column's computation just test membership (``"rank_d_desc" in
    needed``); callers that gate a whole signal matrix test the node
    (``"dense.desc" in needed``). ``FeatureProvider`` columns are not core
    features and never appear here — a provider is gated by whether any of its
    own declared names is in ``requested``, checked directly by the caller.
    """
    needed: Set[str] = set()
    stack: List[str] = list(requested) + ["candidates"]
    while stack:
        node = stack.pop()
        if node in needed:
            continue
        needed.add(node)
        stack.extend(FEATURE_DEPS.get(node, ()))
    return needed


def composed_feature_names(
    providers: Sequence["FeatureProvider"] = (),
    signal_providers: Sequence["SignalProvider"] = (),
) -> List[str]:
    """The effective, ordered feature schema: the core ~36 columns (``FEATURE_NAMES``),
    then any *non-default* ``SignalProvider``'s ``column_names()`` (T34 phase 2 --
    a signal beyond the two built-ins), then each ``FeatureProvider``'s ``names()``,
    in provider order.

    This is *the* column order the fusion model is trained and scored on, and it is
    persisted into ``meta.json`` at save time so inference rebuilds the identical
    order. With no providers it is exactly ``FEATURE_NAMES``. The built-in ``"dense"``/
    ``"lexical"`` signal providers are skipped in the loop below (by ``name``): their
    columns are ``core_feature_names``' output, already present, so declaring them
    again would just be a duplicate of what this function already starts from.

    ``core_feature_names`` narrows that starting list to whichever of ``"dense"``/
    ``"lexical"`` actually appear (by ``name``) in ``signal_providers`` — an empty
    ``signal_providers`` (every pre-T34-phase-2 caller, and any caller that
    deliberately wants the full legacy schema) means "assume both", the
    byte-for-byte-identical default; a non-empty list missing one of the two
    (e.g. only a ``DenseSignalProvider``) drops that signal's columns, because
    nothing built them.

    Raises ``ValueError`` on a name collision (a provider colliding with a core
    column, or two providers colliding), because a duplicate column name would
    silently overwrite data in the assembled frame — the worst kind of
    train/infer disagreement."""
    active_builtin = {sp.name for sp in signal_providers if sp.name in ("dense", "lexical")}
    names: List[str] = core_feature_names(active_builtin if signal_providers else None)
    seen = set(names)
    for sp in signal_providers:
        if sp.name in ("dense", "lexical"):
            continue
        for name in sp.column_names():
            if name in seen:
                raise ValueError(
                    f"feature name collision: {name!r} is contributed by signal provider "
                    f"{sp.name!r} but is already in the schema. Signal-provider column "
                    "names must be unique across the core features and all providers."
                )
            seen.add(name)
            names.append(name)
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


def fusion_feature_names(
    providers: Sequence["FeatureProvider"] = (),
    drop: Sequence[str] = (),
    signal_providers: Sequence["SignalProvider"] = (),
) -> List[str]:
    """The columns the *fusion model* is trained and scored on: the composed
    schema (``composed_feature_names``) minus ``drop``, order otherwise preserved.

    This is deliberately a different question from ``composed_feature_names``:
    the *schema* (this function, and ``composed_feature_names``) is unaffected
    by ``drop`` — the assembler is always capable of producing the full column
    list. What a given assembly call actually computes is a separate,
    per-call decision (``FeatureAssembler.assemble``'s ``requested`` parameter,
    see ``feature_closure`` / T87): training and scoring request exactly this
    narrowed list, while ``explain``/``signal_report``/the ablation report
    request the full ``composed_feature_names`` in their own pass, because they
    read core columns by name. ``drop`` is what makes a *retrain-based*
    ablation possible: train without a column, and compare against a model
    trained with it.

    ``drop`` is persisted (via ``FusionConfig.drop_features`` inside ``meta.json``)
    and re-applied at load, so inference reconstructs the identical column list
    the model was fitted on — the same train/infer parity contract the composed
    schema already carries.

    Raises ``ValueError`` on a name that is not in the schema (a typo would
    otherwise silently drop nothing and quietly invalidate an experiment) and on
    a ``drop`` that would empty the schema.
    """
    names = composed_feature_names(providers, signal_providers)
    if not drop:
        return names
    known = set(names)
    unknown = sorted(set(drop) - known)
    if unknown:
        raise ValueError(
            f"cannot drop unknown feature(s) {unknown}: not in the composed schema. "
            f"Available columns: {names}"
        )
    dropped = set(drop)
    kept = [n for n in names if n not in dropped]
    if not kept:
        raise ValueError(
            "cannot drop every feature: the fusion model needs at least one column "
            f"(tried to drop all {len(names)})"
        )
    return kept


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
