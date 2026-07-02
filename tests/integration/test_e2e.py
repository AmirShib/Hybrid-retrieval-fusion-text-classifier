"""T07 — End-to-end pipeline + persistence round-trip.

Exercises: TrainingPipeline.run → ArtifactRepository.save → .load →
InferencePipeline.predict, plus config serialization.

All tests run fully offline via HashingEncoder; no network access needed.
The encoder save/load is handled by patching SentenceTransformerEncoder in
the persistence module with HashingEncoder (same interface).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from text_classifier import LabeledItem, LabelSpace
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    CalibrationConfig,
    EncoderConfig,
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
    cfg.encoder.kind = "hashing"   # offline encoder via the registry seam (T23)
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
    """Save artifacts to a temp dir. No patching: the encoder kind is "hashing",
    so the registry dispatches save/load to HashingEncoder."""
    artifacts, _, _, _, _ = trained
    d = str(tmp_path_factory.mktemp("model"))
    ArtifactRepository().save(artifacts, d)
    return d


@pytest.fixture(scope="module")
def loaded_artifacts(saved_dir):
    """Load artifacts from the saved dir via the registry seam."""
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
        pipeline = InferencePipeline.from_directory(saved_dir)
        assert pipeline.label_space.keys == loaded_artifacts.label_space.keys


# ---------------------------------------------------------------------------
# Config serialization
# ---------------------------------------------------------------------------

class TestEncoderBackends:
    """T24 — the whole pipeline trains/saves/loads/predicts over multiple
    encoder backends selected purely by config, fully offline."""

    @pytest.mark.parametrize("encoder_kind", ["hashing", "tfidf"])
    def test_full_pipeline_round_trip(self, encoder_kind, tmp_path):
        label_space, items = make_synthetic(n_classes=6, per_class=15, seed=11)
        cfg = _e2e_cfg()
        cfg.encoder = EncoderConfig(kind=encoder_kind)
        # "hashing" is data-independent and injected; "tfidf" is corpus-dependent
        # and fit per fold by the pipeline (no injection).
        shared = HashingEncoder(dim=64) if encoder_kind == "hashing" else None

        artifacts, report = TrainingPipeline(cfg, shared_encoder=shared).run(items, label_space)
        assert report.n_items > 0
        assert 0.0 <= report.coverage <= 1.0

        model_dir = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, model_dir)

        with open(os.path.join(model_dir, "meta.json")) as fh:
            assert json.load(fh)["components"]["encoder"] == encoder_kind

        loaded = ArtifactRepository().load(model_dir)
        texts = [it.text for it in items[:8]]
        before = InferencePipeline(artifacts).predict(texts)
        after = InferencePipeline(loaded).predict(texts)
        assert len(before) == len(after) == 8
        for a, b in zip(before, after):
            assert a.top_key == b.top_key
            assert a.abstained == b.abstained
            assert abs(a.confidence - b.confidence) < 1e-5


_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None


class TestFusionBackends:
    """T41 + T44 — the whole pipeline trains/saves/loads/predicts over multiple
    fusion backends selected purely by config, fully offline."""

    @pytest.mark.parametrize("fusion_kind", ["xgboost", "lightgbm", "xgbranker"])
    def test_full_pipeline_round_trip(self, fusion_kind, tmp_path):
        if fusion_kind == "lightgbm" and not _HAS_LGBM:
            pytest.skip("lightgbm not installed")

        label_space, items = make_synthetic(n_classes=6, per_class=15, seed=13)
        cfg = _e2e_cfg()  # encoder kind="hashing"
        if fusion_kind == "lightgbm":
            cfg.fusion = FusionConfig(kind="lightgbm", params={
                "n_estimators": 20, "max_depth": 3, "random_state": 0, "verbosity": -1,
            })
        elif fusion_kind == "xgbranker":
            cfg.fusion = FusionConfig(kind="xgbranker", params={
                "n_estimators": 20, "max_depth": 3, "random_state": 0,
            })
        # else: keep the default xgboost fusion from _e2e_cfg

        enc = HashingEncoder(dim=64)
        artifacts, report = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
        assert report.n_items > 0
        assert 0.0 <= report.coverage <= 1.0

        model_dir = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, model_dir)
        with open(os.path.join(model_dir, "meta.json")) as fh:
            assert json.load(fh)["components"]["fusion"] == fusion_kind

        loaded = ArtifactRepository().load(model_dir)
        texts = [it.text for it in items[:8]]
        before = InferencePipeline(artifacts).predict(texts)
        after = InferencePipeline(loaded).predict(texts)
        assert len(before) == len(after) == 8
        for a, b in zip(before, after):
            assert a.top_key == b.top_key
            assert a.abstained == b.abstained
            assert abs(a.confidence - b.confidence) < 1e-5


class TestCalibratorBackends:
    """T42 — the whole pipeline trains/saves/loads/predicts over multiple
    calibrator backends selected purely by config, fully offline."""

    @pytest.mark.parametrize("calibrator_kind", ["isotonic", "platt", "beta"])
    def test_full_pipeline_round_trip(self, calibrator_kind, tmp_path):
        label_space, items = make_synthetic(n_classes=6, per_class=15, seed=17)
        cfg = _e2e_cfg()  # encoder kind="hashing", xgboost fusion
        cfg.calibration = CalibrationConfig(kind=calibrator_kind)

        enc = HashingEncoder(dim=64)
        artifacts, report = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
        assert report.n_items > 0
        assert 0.0 <= report.coverage <= 1.0

        model_dir = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, model_dir)
        with open(os.path.join(model_dir, "meta.json")) as fh:
            assert json.load(fh)["components"]["calibrator"] == calibrator_kind

        loaded = ArtifactRepository().load(model_dir)
        texts = [it.text for it in items[:8]]
        before = InferencePipeline(artifacts).predict(texts)
        after = InferencePipeline(loaded).predict(texts)
        assert len(before) == len(after) == 8
        for a, b in zip(before, after):
            assert a.top_key == b.top_key
            assert a.abstained == b.abstained
            assert abs(a.confidence - b.confidence) < 1e-5


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

    def test_component_kinds_roundtrip(self):
        """The T23 `kind` selectors (encoder/fusion/calibration) survive serialization."""
        cfg = _e2e_cfg()
        d = cfg.to_dict()
        assert d["encoder"]["kind"] == "hashing"
        assert d["fusion"]["kind"] == "xgboost"
        assert d["calibration"]["kind"] == "isotonic"
        restored = PipelineConfig.from_dict(d)
        assert restored.encoder.kind == "hashing"
        assert restored.calibration.kind == "isotonic"

    def test_from_dict_without_calibration_uses_default(self):
        """A config serialized before `calibration` existed still loads."""
        d = _e2e_cfg().to_dict()
        del d["calibration"]
        restored = PipelineConfig.from_dict(d)
        assert restored.calibration.kind == "isotonic"


class TestVectorizedPredict:
    """T30 — predict() is vectorized: a single accept() call per batch, and
    results stay aligned to input order (no per-row pandas loop)."""

    def test_accept_called_once_per_batch(self, trained):
        from text_classifier.domain.services import AbstentionPolicy

        artifacts, _, _, items, _ = trained
        texts = [it.text for it in items[:12]]

        calls = []
        original = AbstentionPolicy.accept

        def counting_accept(self, confidence, class_index):
            calls.append(len(confidence))
            return original(self, confidence, class_index)

        with patch.object(AbstentionPolicy, "accept", counting_accept):
            preds = InferencePipeline(artifacts).predict(texts)

        assert len(preds) == len(texts)
        assert len(calls) == 1  # one batched call, not one per item

    def test_results_track_input_order(self, trained):
        """Each output corresponds to its input position and is independent of
        batch order — reversing the inputs reverses the outputs."""
        artifacts, _, _, items, _ = trained
        texts = [it.text for it in items[:6]]
        pipeline = InferencePipeline(artifacts)

        forward = pipeline.predict(texts)
        reversed_ = pipeline.predict(texts[::-1])
        n = len(texts)
        for i in range(n):
            mirror = reversed_[n - 1 - i]
            assert forward[i].top_key == mirror.top_key
            assert forward[i].predicted_key == mirror.predicted_key
            assert forward[i].abstained == mirror.abstained
            assert abs(forward[i].confidence - mirror.confidence) < 1e-9


# ---------------------------------------------------------------------------
# Determinism (T26)
# ---------------------------------------------------------------------------


def test_training_is_deterministic_end_to_end():
    """Two identical runs produce identical thresholds and coverage numbers.

    xgb_params deliberately omit random_state and engage subsampling: the
    backend's seed default is what has to make the runs reproducible.
    """
    label_space, items = make_synthetic(n_classes=8, per_class=20, seed=7)

    def run():
        cfg = _e2e_cfg()
        cfg.fusion = FusionConfig(xgb_params={
            "n_estimators": 30, "max_depth": 3, "subsample": 0.5,
            "colsample_bytree": 0.5, "n_jobs": 1,
        })
        enc = HashingEncoder(dim=64)
        return TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)

    (art1, rep1), (art2, rep2) = run(), run()
    assert art1.abstention.global_threshold == art2.abstention.global_threshold
    assert art1.abstention.per_class == art2.abstention.per_class
    # NaN-tolerant field-wise equality (accuracy fields may be NaN by contract)
    np.testing.assert_equal(dataclasses.asdict(rep1), dataclasses.asdict(rep2))
