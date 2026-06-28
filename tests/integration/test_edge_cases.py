"""T22 — Edge cases at the pipeline level.

Bounds/invariant assertions only (per the determinism contract in
tests/conftest.py): pipeline outputs depend on XGBoost internals, so we never
pin exact floats here.

Covers:
  - a label space with a class that has zero training examples trains end-to-end
    and yields a finite CoverageReport;
  - InferencePipeline.predict([]) returns [] (re-confirmed after T22 changes).
"""
from __future__ import annotations

import numpy as np

from text_classifier import ClassDefinition, LabelSpace
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from tests._doubles import HashingEncoder, make_synthetic


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, use_per_fold_encoder=False,
        target_precision=0.5, per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(xgb_params={
        "n_estimators": 30, "max_depth": 3, "random_state": 0, "n_jobs": 1,
    })
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    return cfg


def test_training_with_zero_example_class_produces_finite_report():
    """A class declared in the LabelSpace but absent from the items is legal:
    it never enters a fold and simply carries an all-NaN prototype."""
    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=1)
    # Append a class with zero examples.
    defs = [ClassDefinition(k, d) for k, d in zip(label_space.keys, label_space.descriptions)]
    defs.append(ClassDefinition("EMPTY", "a declared class that has no training items"))
    augmented = LabelSpace(defs)

    enc = HashingEncoder(dim=64)
    artifacts, report = TrainingPipeline(_cfg(), shared_encoder=enc).run(items, augmented)

    assert report.n_items > 0
    assert 0.0 <= report.coverage <= 1.0
    assert 0.0 <= report.candidate_recall <= 1.0
    assert np.isfinite(report.candidate_recall)
    # The empty class is preserved in the deployed label space.
    assert "EMPTY" in artifacts.label_space.keys
    assert artifacts.dense.class_freq[augmented.index_of("EMPTY")] == 0


def test_predict_empty_batch_returns_empty_list():
    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=2)
    enc = HashingEncoder(dim=64)
    artifacts, _ = TrainingPipeline(_cfg(), shared_encoder=enc).run(items, label_space)
    assert InferencePipeline(artifacts).predict([]) == []
