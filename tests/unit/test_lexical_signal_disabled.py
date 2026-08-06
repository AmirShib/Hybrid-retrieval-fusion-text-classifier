"""``signals=["dense"]`` must skip building the BM25/lexical index entirely,
not just skip using it -- tokenizing the corpus and fitting a per-fold BM25
weight matrix is dead work when nothing ever queries it. Covers the full
lifecycle a real model dir goes through: train -> save -> load -> predict ->
explain -> update -> retune, with ``lexical`` staying ``None`` throughout.
"""

from __future__ import annotations

import json
import os

from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.application.tuning import retune
from text_classifier.application.updating import update
from text_classifier.config import PipelineConfig, RetrievalConfig, TrainingConfig
from text_classifier.domain import ClassDefinition, LabeledItem
from text_classifier.infrastructure import ArtifactRepository
from tests._doubles import make_synthetic


def _train(cfg, seed=5, n_classes=6, per_class=15):
    label_space, items = make_synthetic(n_classes=n_classes, per_class=per_class, seed=seed)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    artifacts, report = TrainingPipeline(cfg).run(items, label_space)
    return artifacts, report, items


def test_dense_only_signals_builds_no_lexical_index():
    cfg = PipelineConfig()
    cfg.signals = ["dense"]
    artifacts, report, _ = _train(cfg)
    assert report.n_items > 0
    assert artifacts.lexical is None
    assert [type(p).__name__ for p in artifacts.signal_providers] == ["DenseSignalProvider"]


def test_dense_only_signals_round_trips_and_predicts(tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense"]
    artifacts, _, items = _train(cfg)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    # No lexical files were written at all.
    assert not os.path.exists(os.path.join(model_dir, "lexical.npz"))
    assert not os.path.exists(os.path.join(model_dir, "lexical.json"))

    with open(f"{model_dir}/meta.json") as fh:
        meta = json.load(fh)
    assert meta["components"]["lexical"] is None
    assert meta["components"]["signals"] == ["dense"]

    texts = [it.text for it in items[:10]]
    before = InferencePipeline(artifacts).predict(texts)

    loaded = ArtifactRepository().load(model_dir)
    assert loaded.lexical is None
    after = InferencePipeline(loaded).predict(texts)
    assert [p.top_key for p in before] == [p.top_key for p in after]

    # explain()'s neighbor evidence tolerates the missing lexical index.
    records = InferencePipeline(loaded).explain_records(texts[:3])
    for rec in records:
        assert rec["neighbors"]["lexical"] == []


def test_dense_only_signals_survives_update_and_retune(tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense"]
    cfg.training.store_corpus = True
    artifacts, _, items = _train(cfg)

    label_space = artifacts.label_space
    classes = [
        ClassDefinition(k, d) for k, d in zip(label_space.keys, label_space.descriptions)
    ] + [ClassDefinition("new_class", "a brand new class")]
    new_items = [LabeledItem(items[0].text, items[0].label)]

    updated, updated_corpus = update(artifacts, items, classes=classes, new_items=new_items)
    assert updated.lexical is None
    assert updated_corpus is not None

    policy, calibrator, evaluation = retune(
        updated, items[:20], updated.label_space, target_precision=0.5, per_class_min_support=1
    )
    assert "abstention" in evaluation
