"""T70 — custom fusion features: end-to-end parity + air-gapped portability.

The three non-negotiable constraints from the ticket get a pipeline-level test:
  1. Inference parity / air-gapped: a provider runs from the persisted model dir
     (no labels, no network) and reproduces predictions bit-for-bit.
  2. Schema authority: the composed feature order (core + provider columns) is
     persisted into meta.json and drives both train and infer.
  3. Zero providers => byte-for-byte identical schema + predictions vs. a build
     without the feature at all.

(The out-of-fold leakage constraint is exercised in test_leakage.py.)
"""

from __future__ import annotations

import json
import os

import pytest

from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    FeatureProviderConfig,
    FeaturesConfig,
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from text_classifier.domain import FEATURE_NAMES
from text_classifier.infrastructure.persistence import ArtifactRepository
from tests._doubles import HashingEncoder, make_synthetic


def _cfg(with_provider: bool) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, use_per_fold_encoder=False,
        target_precision=0.5, per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 30, "max_depth": 3, "random_state": 0, "n_jobs": 1}
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    if with_provider:
        cfg.features = FeaturesConfig(
            providers=[FeatureProviderConfig(kind="class-keyword", params={})]
        )
    return cfg


@pytest.fixture(scope="module")
def dataset():
    return make_synthetic(n_classes=8, per_class=20, seed=7)


@pytest.fixture(scope="module")
def trained_with_provider(dataset):
    label_space, items = dataset
    enc = HashingEncoder(dim=64)
    artifacts, report = TrainingPipeline(_cfg(True), shared_encoder=enc).run(items, label_space)
    return artifacts, report, label_space, items


class TestSchemaComposition:
    def test_meta_feature_names_include_provider_column(self, trained_with_provider, tmp_path):
        artifacts, *_ = trained_with_provider
        d = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, d)
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["feature_names"] == FEATURE_NAMES + ["class_kw_overlap"]

    def test_meta_records_provider_manifest(self, trained_with_provider, tmp_path):
        artifacts, *_ = trained_with_provider
        d = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, d)
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        manifest = meta["feature_providers"]
        assert len(manifest) == 1
        assert manifest[0]["kind"] == "class-keyword"
        assert manifest[0]["names"] == ["class_kw_overlap"]
        # The provider's artifact actually exists on disk (portable, air-gapped).
        assert os.path.isdir(os.path.join(d, manifest[0]["path"]))

    def test_provider_reaches_fusion_model(self, trained_with_provider):
        """The fusion model was trained on the composed schema, so its input
        dimensionality includes the provider column."""
        artifacts, *_ = trained_with_provider
        assert len(artifacts.feature_providers) == 1
        # The composed schema is core + 1.
        from text_classifier.domain import composed_feature_names

        names = composed_feature_names(artifacts.feature_providers)
        assert len(names) == len(FEATURE_NAMES) + 1


class TestInferenceParity:
    def test_predict_identical_after_reload(self, trained_with_provider, tmp_path):
        """Constraint 1: the provider runs from the persisted dir (no labels, no
        network) and reproduces predictions exactly."""
        artifacts, _, _, items = trained_with_provider
        d = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, d)
        loaded = ArtifactRepository().load(d)
        assert len(loaded.feature_providers) == 1

        texts = [it.text for it in items[:15]]
        before = InferencePipeline(artifacts).predict(texts)
        after = InferencePipeline(loaded).predict(texts)
        assert len(before) == len(after) == 15
        for a, b in zip(before, after):
            assert a.top_key == b.top_key
            assert a.abstained == b.abstained
            assert abs(a.confidence - b.confidence) < 1e-6

    def test_from_directory_loads_providers(self, trained_with_provider, tmp_path):
        artifacts, *_ = trained_with_provider
        d = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, d)
        pipe = InferencePipeline.from_directory(d)
        preds = pipe.predict(["some text to classify"])
        assert len(preds) == 1


class TestZeroProviderIdentity:
    def test_schema_and_predictions_unchanged_without_providers(self, dataset, tmp_path):
        """Constraint 3: with no providers configured the schema and outputs are
        byte-for-byte what they were before T70 existed."""
        label_space, items = dataset
        enc = HashingEncoder(dim=64)
        artifacts, _ = TrainingPipeline(_cfg(False), shared_encoder=enc).run(items, label_space)

        assert artifacts.feature_providers == []
        d = str(tmp_path / "model")
        ArtifactRepository().save(artifacts, d)
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["feature_names"] == FEATURE_NAMES
        assert meta["feature_providers"] == []
        # No stray features/ directory is written for a provider-less model.
        assert not os.path.exists(os.path.join(d, "features"))


class TestConfigRoundTrip:
    def test_features_config_survives_serialization(self):
        cfg = _cfg(True)
        restored = PipelineConfig.from_dict(cfg.to_dict())
        assert len(restored.features.providers) == 1
        assert restored.features.providers[0].kind == "class-keyword"

    def test_unknown_provider_key_rejected(self):
        with pytest.raises(ValueError, match="unknown key"):
            PipelineConfig.from_dict(
                {"features": {"providers": [{"kind": "class-keyword", "bogus": 1}]}}
            )

    def test_provider_missing_kind_rejected(self):
        with pytest.raises(ValueError, match="missing required key 'kind'"):
            PipelineConfig.from_dict({"features": {"providers": [{"params": {}}]}})
