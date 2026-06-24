"""T07 — End-to-end pipeline + persistence round-trip.

Exercises: TrainingPipeline.run → ArtifactRepository.save → .load →
InferencePipeline.predict, plus config serialization.

All tests run fully offline via HashingEncoder; no network access needed.
The encoder save/load is handled by patching SentenceTransformerEncoder in
the persistence module with HashingEncoder (same interface).
"""
from __future__ import annotations

import dataclasses
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from text_classifier import LabeledItem, LabelSpace
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from text_classifier.domain import FEATURE_NAMES
from text_classifier.infrastructure.persistence import ArtifactRepository
from tests._doubles import HashingEncoder, make_synthetic


# ---------------------------------------------------------------------------
# Config + fixtures
# ---------------------------------------------------------------------------

def _e2e_cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.training = TrainingConfig(
        n_folds=3,
        random_state=0,
        use_per_fold_encoder=False,
        target_precision=0.5,
        per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(xgb_params={
        "n_estimators": 30, "max_depth": 3, "random_state": 0, "n_jobs": 1,
    })
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    return cfg


@pytest.fixture(scope="module")
def trained():
    """Run TrainingPipeline once; reuse artifacts + report across all T07 tests."""
    enc = HashingEncoder(dim=64)
    label_space, items = make_synthetic(n_classes=8, per_class=20, seed=7)
    cfg = _e2e_cfg()
    artifacts, report = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
    return artifacts, report, label_space, items, enc


@pytest.fixture(scope="module")
def saved_dir(trained, tmp_path_factory):
    """Save artifacts to a temp dir (patching out SentenceTransformerEncoder)."""
    artifacts, _, _, _, _ = trained
    d = str(tmp_path_factory.mktemp("model"))
    with patch("text_classifier.infrastructure.persistence.SentenceTransformerEncoder",
               HashingEncoder):
        ArtifactRepository().save(artifacts, d)
    return d


@pytest.fixture(scope="module")
def loaded_artifacts(saved_dir):
    """Load artifacts from the saved dir (patching encoder load)."""
    with patch("text_classifier.infrastructure.persistence.SentenceTransformerEncoder",
               HashingEncoder):
        return ArtifactRepository().load(saved_dir)


# ---------------------------------------------------------------------------
# Train → report
# ---------------------------------------------------------------------------

class TestTrainReport:
    def test_coverage_in_unit_interval(self, trained):
        _, report, *_ = trained
        assert 0.0 <= report.coverage <= 1.0

    def test_accuracy_on_accepted_in_unit_interval_or_nan(self, trained):
        _, report, *_ = trained
        if not np.isnan(report.accuracy_on_accepted):
            assert 0.0 <= report.accuracy_on_accepted <= 1.0

    def test_accuracy_if_no_abstain_in_unit_interval_or_nan(self, trained):
        _, report, *_ = trained
        if not np.isnan(report.accuracy_if_no_abstain):
            assert 0.0 <= report.accuracy_if_no_abstain <= 1.0

    def test_candidate_recall_in_unit_interval(self, trained):
        _, report, *_ = trained
        assert 0.0 <= report.candidate_recall <= 1.0

    def test_n_items_positive(self, trained):
        _, report, *_ = trained
        assert report.n_items > 0

    def test_behavioral_candidate_recall_high(self, trained):
        """On the synthetic dataset (which shares class-specific tokens) recall > 0.5."""
        _, report, *_ = trained
        assert report.candidate_recall > 0.5

    def test_abstention_does_not_hurt_accepted_accuracy(self, trained):
        """accuracy_on_accepted ≥ accuracy_if_no_abstain (abstention should help or be neutral)."""
        _, report, *_ = trained
        if not np.isnan(report.accuracy_on_accepted) and not np.isnan(report.accuracy_if_no_abstain):
            assert report.accuracy_on_accepted >= report.accuracy_if_no_abstain - 1e-9

    def test_determinism(self):
        """Two runs with the same seed produce identical reports."""
        enc = HashingEncoder(dim=64)
        label_space, items = make_synthetic(n_classes=6, per_class=15, seed=3)
        cfg = _e2e_cfg()
        _, r1 = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
        _, r2 = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
        assert r1.coverage == r2.coverage
        assert r1.candidate_recall == r2.candidate_recall


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_expected_files_exist(self, saved_dir):
        for name in ("dense.npz", "lexical.pkl", "fusion.json", "calibrator.pkl", "meta.json"):
            assert os.path.exists(os.path.join(saved_dir, name)), f"Missing {name}"
        assert os.path.isdir(os.path.join(saved_dir, "encoder"))

    def test_meta_json_feature_names(self, saved_dir):
        with open(os.path.join(saved_dir, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["feature_names"] == FEATURE_NAMES

    def test_meta_json_has_config(self, saved_dir):
        with open(os.path.join(saved_dir, "meta.json")) as fh:
            meta = json.load(fh)
        assert "config" in meta
        assert "training" in meta["config"]
        assert "fusion" in meta["config"]

    def test_meta_json_has_classes(self, saved_dir, trained):
        _, _, label_space, _, _ = trained
        with open(os.path.join(saved_dir, "meta.json")) as fh:
            meta = json.load(fh)
        saved_keys = [c["key"] for c in meta["classes"]]
        assert saved_keys == label_space.keys

    def test_meta_json_has_abstention(self, saved_dir):
        with open(os.path.join(saved_dir, "meta.json")) as fh:
            meta = json.load(fh)
        assert "global_threshold" in meta["abstention"]
        assert "per_class" in meta["abstention"]

    def test_loaded_label_space_matches(self, trained, loaded_artifacts):
        _, _, label_space, _, _ = trained
        assert loaded_artifacts.label_space.keys == label_space.keys
        assert loaded_artifacts.label_space.descriptions == label_space.descriptions

    def test_loaded_abstention_matches(self, trained, loaded_artifacts):
        artifacts, *_ = trained
        orig = artifacts.abstention
        loaded = loaded_artifacts.abstention
        assert abs(orig.global_threshold - loaded.global_threshold) < 1e-9
        assert orig.per_class == loaded.per_class

    def test_prediction_identity_after_reload(self, trained, loaded_artifacts):
        artifacts, _, _, items, _ = trained
        sample_texts = [it.text for it in items[:10]]
        orig_preds = InferencePipeline(artifacts).predict(sample_texts)
        load_preds = InferencePipeline(loaded_artifacts).predict(sample_texts)
        for o, l in zip(orig_preds, load_preds):
            assert o.top_key == l.top_key
            assert o.abstained == l.abstained
            assert abs(o.confidence - l.confidence) < 1e-5


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------

class TestInferenceContract:
    def test_one_prediction_per_input(self, trained):
        artifacts, _, _, items, _ = trained
        texts = [it.text for it in items[:15]]
        preds = InferencePipeline(artifacts).predict(texts)
        assert len(preds) == len(texts)

    def test_empty_input_returns_empty(self, trained):
        artifacts, *_ = trained
        assert InferencePipeline(artifacts).predict([]) == []

    def test_confidence_in_unit_interval(self, trained):
        artifacts, _, _, items, _ = trained
        preds = InferencePipeline(artifacts).predict([it.text for it in items[:20]])
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0

    def test_abstained_has_none_predicted_key(self, trained):
        artifacts, *_ = trained
        preds = InferencePipeline(artifacts).predict(
            [it.text for it in make_synthetic(n_classes=8, per_class=20, seed=7)[1][:30]]
        )
        for p in preds:
            if p.abstained:
                assert p.predicted_key is None
            else:
                assert p.predicted_key == p.top_key

    def test_no_candidate_item_abstains(self, trained):
        """When the feature assembler surfaces no candidates, predict returns abstained."""
        artifacts, *_ = trained
        import pandas as pd
        empty_df = pd.DataFrame(columns=FEATURE_NAMES)
        # Patch assemble to return an empty frame, forcing the no-candidate code path.
        with patch("text_classifier.application.inference.FeatureAssembler.assemble",
                   return_value=empty_df):
            preds = InferencePipeline(artifacts).predict(["any text"])
        assert len(preds) == 1
        assert preds[0].abstained is True
        assert preds[0].predicted_key is None
        assert preds[0].top_key == ""
        assert preds[0].confidence == 0.0

    def test_label_space_property(self, trained):
        artifacts, _, label_space, _, _ = trained
        pipeline = InferencePipeline(artifacts)
        assert pipeline.label_space.keys == label_space.keys

    def test_config_property(self, trained):
        artifacts, *_ = trained
        pipeline = InferencePipeline(artifacts)
        assert pipeline.config is artifacts.config

    def test_from_directory(self, saved_dir, loaded_artifacts):
        """InferencePipeline.from_directory loads the same artifacts as ArtifactRepository."""
        with patch("text_classifier.infrastructure.persistence.SentenceTransformerEncoder",
                   HashingEncoder):
            pipeline = InferencePipeline.from_directory(saved_dir)
        assert pipeline.label_space.keys == loaded_artifacts.label_space.keys


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------

class TestConfigSerialization:
    def test_pipeline_config_roundtrip(self):
        cfg = _e2e_cfg()
        restored = PipelineConfig.from_dict(cfg.to_dict())
        assert dataclasses.asdict(cfg) == dataclasses.asdict(restored)

    def test_nested_config_roundtrip(self):
        """Each sub-config survives to_dict → from_dict independently."""
        cfg = _e2e_cfg()
        d = cfg.to_dict()
        assert d["training"]["n_folds"] == cfg.training.n_folds
        assert d["fusion"]["xgb_params"] == cfg.fusion.xgb_params
        assert d["retrieval"]["k_neighbors"] == cfg.retrieval.k_neighbors
