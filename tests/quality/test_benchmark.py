"""T52 — Offline quality-regression benchmark.

Trains the full pipeline on a fixed, seeded synthetic task and asserts the
headline metrics stay above known floors. This catches the class of regression
the unit suite cannot: a change that is *correct code* but a *worse model*.

What this net catches (verified by sabotage when the floors were set):
- candidate-generation breakage (narrowing each signal's nominations to top-1
  dropped candidate recall to 0.944 -> trips FLOOR_RECALL);
- decision-path breakage (picking the worst candidate per item drove coverage
  to 0.0 -> trips FLOOR_COVERAGE);
- gross signal destruction and calibration/threshold-tuner bugs (accuracy /
  coverage floors).

What it deliberately does NOT catch: degradation of a *single* retrieval
signal. The five-signal ensemble is redundant enough that the fusion model
compensates (zeroing d_desc_sim, or even feeding the dense kNN shuffled
neighbor labels, moved accuracy-on-accepted by < 0.03). Exactness of
individual signals is the job of the golden-output tests specified in T34,
not of this benchmark.

Items are truncated to their first two tokens to pull the task off the
ceiling (full texts score ~0.99 accuracy, where floors can't discriminate).

Runs fully offline (hashing/tfidf encoders, no torch, no network) in a few
seconds. Excluded from the default CI test job via ``-m "not quality"``; the
dedicated CI quality job runs ``-m quality``.
"""
from __future__ import annotations

import json
import os

import pytest

from text_classifier import LabeledItem, PipelineConfig, TrainingPipeline
from text_classifier.config import FusionConfig, RetrievalConfig, TrainingConfig
from text_classifier.datasets import make_synthetic

# Floors sit well under the observed values (recorded 2026-07-02, T26 seeding
# in place, so run-to-run variance is zero on a fixed platform; the margin
# absorbs cross-platform/library-version drift only):
#   hashing: recall=0.994  coverage=0.981  acc_on_accepted=0.873
#   tfidf:   recall=1.000  coverage=0.988  acc_on_accepted=0.835
FLOOR_RECALL = 0.95
FLOOR_COVERAGE = 0.85
FLOOR_ACC_ON_ACCEPTED = 0.75


def _benchmark_task():
    """Fixed task: 30 imbalanced classes, items truncated to 2 tokens."""
    label_space, items = make_synthetic(n_classes=30, per_class=24, seed=7)
    items = [LabeledItem(" ".join(it.text.split()[:2]), it.label) for it in items]
    return label_space, items


def _benchmark_config(encoder_kind: str) -> PipelineConfig:
    cfg = PipelineConfig(candidate_top_n=8)
    cfg.encoder.kind = encoder_kind
    cfg.training = TrainingConfig(n_folds=4, target_precision=0.85,
                                  per_class_min_support=100)
    cfg.fusion = FusionConfig(xgb_params={
        "n_estimators": 60, "max_depth": 4, "n_jobs": 1, "random_state": 0,
    })
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    return cfg


@pytest.mark.quality
@pytest.mark.parametrize("encoder_kind", ["hashing", "tfidf"])
def test_headline_metrics_meet_floors(encoder_kind, tmp_path):
    label_space, items = _benchmark_task()
    out_dir = str(tmp_path / f"model_{encoder_kind}")
    _, report = TrainingPipeline(_benchmark_config(encoder_kind)).run(
        items, label_space, output_dir=out_dir
    )

    assert report.candidate_recall >= FLOOR_RECALL, (
        f"candidate recall {report.candidate_recall:.4f} < floor {FLOOR_RECALL} "
        f"({encoder_kind}): candidate generation has regressed"
    )
    assert report.coverage >= FLOOR_COVERAGE, (
        f"coverage {report.coverage:.4f} < floor {FLOOR_COVERAGE} "
        f"({encoder_kind}): calibration/threshold tuning has regressed"
    )
    assert report.accuracy_on_accepted >= FLOOR_ACC_ON_ACCEPTED, (
        f"accuracy on accepted {report.accuracy_on_accepted:.4f} < floor "
        f"{FLOOR_ACC_ON_ACCEPTED} ({encoder_kind}): the model has regressed"
    )

    # Cross-check: the persisted evaluation carries the same headline numbers
    # the pipeline reported (ties T61's artifacts to the in-memory report).
    with open(os.path.join(out_dir, "evaluation.json")) as fh:
        evaluation = json.load(fh)
    overall = evaluation["overall"]
    assert overall["candidate_recall"] == pytest.approx(report.candidate_recall)
    assert overall["coverage"] == pytest.approx(report.coverage)
    assert overall["accuracy_on_accepted"] == pytest.approx(report.accuracy_on_accepted)
