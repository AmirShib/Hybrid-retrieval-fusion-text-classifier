"""Re-tune calibration + abstention thresholds on a deployed model.

The operating point (target precision -> thresholds) is normally baked in at
train time (``TrainingPipeline._fit_fusion``). Moving the coverage/precision
knob — or responding to drift the ``eval`` CLI surfaced — costs a full retrain
today, even though refitting the calibrator and re-tuning thresholds is seconds
of work on arrays the model already produces.

``retune`` refits only that decision layer: it featurizes a *fresh* labeled set
against the model's existing (frozen) encoder and retrieval indices — the exact
path ``evaluate``/``explain`` use — gets raw fusion scores, and reuses
``fit_calibration_and_abstention`` (the same threshold logic ``TrainingPipeline``
applies) to produce a new calibrator and ``AbstentionPolicy``. The encoder,
retrieval indices, and fusion model are never touched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..domain import (
    AbstentionPolicy,
    CandidatePolicy,
    ConfidenceCalibrator,
    LabeledItem,
    LabelSpace,
    fusion_feature_names,
)
from ..infrastructure import DeployedArtifacts
from .evaluation import evaluate_decisions
from .features import FeatureAssembler
from .scoring import add_confidence, top_per_item
from .training import fit_calibration_and_abstention

log = logging.getLogger(__name__)

# A tune-set item whose nearest indexed example sits at (near-)identical cosine
# similarity is almost certainly the same text the encoder has already seen —
# the encoder is deterministic, so re-encoding the same string reproduces the
# same (L2-normalized) vector. This can't detect paraphrases or genuinely new
# near-duplicates, only exact-or-near-exact text reuse; it exists to catch the
# one costly mistake this tool warns against (accidentally re-pointing it at
# the training set), not to replace an exact-text corpus check.
_OVERLAP_SIMILARITY_THRESHOLD = 0.999


def count_likely_overlap(artifacts: DeployedArtifacts, q_emb: np.ndarray) -> int:
    """Count tune-set items whose nearest indexed example is a (near-)exact
    embedding match — a cheap, conservative proxy for "this text is already in
    the training corpus" (see module docstring for why it's a proxy, not exact
    text matching)."""
    if artifacts.dense.state.example_emb.shape[0] == 0:
        return 0
    _, sim = artifacts.dense.knn_example_labels(q_emb, k=1)
    best = np.nan_to_num(sim[:, 0], nan=-1.0)
    return int(np.sum(best >= _OVERLAP_SIMILARITY_THRESHOLD))


def retune(
    artifacts: DeployedArtifacts,
    items: Sequence[LabeledItem],
    label_space: LabelSpace,
    target_precision: float,
    per_class_min_support: int,
) -> Tuple[AbstentionPolicy, ConfidenceCalibrator, Dict[str, Any]]:
    """Refit the calibrator and re-tune abstention thresholds on ``items``.

    Returns ``(abstention, calibrator, evaluation)`` — ``artifacts`` is not
    mutated; the caller decides whether/how to persist the result (see
    ``ArtifactRepository.update_decision_layer``).

    ``items`` must be a *fresh* labeled set: an item that was in the original
    training set sits inside the deployed retrieval indices, retrieves itself
    as a perfect match, and draws an optimistically inflated confidence — the
    retuned threshold would then under-abstain in production. This function
    logs a best-effort warning (``count_likely_overlap``) when tune-set items
    look like near-duplicates of an indexed example; it cannot check exact text
    identity without the training corpus itself (a later capability).
    """
    items = list(items)
    if not items:
        raise ValueError("retune requires a non-empty labeled set")

    known = set(label_space.keys)
    unknown = sorted({it.label for it in items if it.label not in known})
    if unknown:
        shown = unknown[:10]
        suffix = " ..." if len(unknown) > 10 else ""
        raise ValueError(
            f"{len(unknown)} tune-set label(s) are not in the model's label space: {shown}{suffix}"
        )

    feature_names = fusion_feature_names(
        artifacts.feature_providers,
        artifacts.config.fusion.drop_features,
        artifacts.signal_providers,
    )
    assembler = FeatureAssembler(label_space, CandidatePolicy(artifacts.config.candidate_top_n))
    texts = [it.text for it in items]
    y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)

    q_emb = artifacts.encoder.encode_queries(texts)
    n_overlap = count_likely_overlap(artifacts, q_emb)
    if n_overlap:
        log.warning(
            "%d of %d tune-set item(s) are a (near-)exact embedding match to an example "
            "already in the deployed index — likely the same text the model was trained "
            "on. If this labeled set overlaps the training data, retuned thresholds will "
            "be optimistic and under-abstain in production; use a genuinely held-out set.",
            n_overlap,
            len(items),
        )

    feats = assembler.assemble(
        texts,
        q_emb,
        artifacts.dense,
        artifacts.lexical,
        artifacts.config.retrieval.k_neighbors,
        query_ids=list(range(len(texts))),
        query_labels=y,
        chunk=artifacts.config.retrieval.feature_chunk,
        providers=artifacts.feature_providers,
        signal_providers=artifacts.signal_providers,
    )
    if not len(feats):
        raise ValueError(
            "no candidates were retrieved for any tune-set item against the deployed "
            "index (candidate recall is 0); cannot retune on this set"
        )

    calibrator, abstention = fit_calibration_and_abstention(
        feats,
        artifacts.fusion,
        artifacts.config.calibration,
        feature_names,
        target_precision,
        per_class_min_support,
    )

    decided = top_per_item(add_confidence(feats, artifacts.fusion, calibrator, feature_names))
    item_ids = decided["item_id"].to_numpy(dtype=np.intp)
    pred_idx = decided["candidate"].to_numpy(dtype=np.intp)
    conf = decided["conf"].to_numpy(dtype=np.float64)
    correct = decided["is_true"].to_numpy().astype(bool)
    accept = abstention.accept(conf, pred_idx)
    true_idx = y[item_ids]
    recall = float(feats.groupby("item_id")["is_true"].max().mean())

    evaluation = evaluate_decisions(
        confidence=conf,
        correct=correct,
        accepted=accept,
        pred_idx=pred_idx,
        true_idx=true_idx,
        keys=label_space.keys,
        candidate_recall=recall,
    )
    evaluation["abstention"] = {
        "global_threshold": abstention.global_threshold,
        "n_per_class_thresholds": len(abstention.per_class),
        "per_class": {label_space.key_at(c): thr for c, thr in abstention.per_class.items()},
    }
    return abstention, calibrator, evaluation
