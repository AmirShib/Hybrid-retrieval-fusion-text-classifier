"""T23 — Component registry + factory DI.

Proves the seam works without touching TrainingPipeline or ArtifactRepository:
a backend is wired in purely by registering a spec and selecting its kind in
config. Also pins the error contract and the persistence back-compat path.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from text_classifier.config import CalibrationConfig, EncoderConfig, FusionConfig
from text_classifier.domain import ConfidenceCalibrator, FusionModel
from text_classifier.infrastructure import (
    CalibratorSpec,
    FusionSpec,
    build_calibrator,
    build_encoder,
    build_fusion,
    register_calibrator,
    register_fusion,
)
from text_classifier.infrastructure.persistence import ArtifactRepository


# --------------------------------------------------------------------------- #
# Built-ins resolve
# --------------------------------------------------------------------------- #
def test_builtin_fusion_resolves():
    model = build_fusion(FusionConfig(kind="xgboost"))
    assert isinstance(model, FusionModel)


def test_builtin_calibrator_resolves():
    cal = build_calibrator(CalibrationConfig(kind="isotonic"))
    assert isinstance(cal, ConfidenceCalibrator)


# --------------------------------------------------------------------------- #
# Unknown kind → clear error listing registered names
# --------------------------------------------------------------------------- #
def test_unknown_fusion_kind_lists_registered():
    with pytest.raises(ValueError, match="unknown fusion kind 'nope'") as exc:
        build_fusion(FusionConfig(kind="nope"))
    assert "xgboost" in str(exc.value)  # registered names are surfaced


def test_unknown_encoder_kind_lists_registered():
    with pytest.raises(ValueError, match="unknown encoder kind") as exc:
        build_encoder(EncoderConfig(kind="ghost"))
    assert "sentence-transformers" in str(exc.value)


def test_unknown_calibrator_kind_raises():
    with pytest.raises(ValueError, match="unknown calibrator kind"):
        build_calibrator(CalibrationConfig(kind="ghost"))


# --------------------------------------------------------------------------- #
# A custom backend plugs in with no pipeline edits
# --------------------------------------------------------------------------- #
class _ConstantFusion(FusionModel):
    """Trivial in-test fusion double: predicts a constant probability."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def fit(self, X, y):  # noqa: D102 - no learning needed
        self._n_features = X.shape[1]

    def predict_proba(self, X):
        return np.full(X.shape[0], self.value, dtype=np.float64)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"value": self.value}, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(value=json.load(fh)["value"])


def test_register_and_build_custom_fusion():
    register_fusion(
        "constant-test",
        FusionSpec(
            build=lambda cfg: _ConstantFusion(cfg.params.get("value", 0.5)),
            filename="constant.json",
            load=_ConstantFusion.load,
        ),
    )
    model = build_fusion(FusionConfig(kind="constant-test", params={"value": 0.7}))
    assert isinstance(model, _ConstantFusion)
    assert model.value == 0.7


def test_custom_fusion_save_load_roundtrip(tmp_path):
    register_fusion(
        "constant-rt",
        FusionSpec(
            build=lambda cfg: _ConstantFusion(),
            filename="constant.json",
            load=_ConstantFusion.load,
        ),
    )
    from text_classifier.infrastructure.registry import fusion_spec

    spec = fusion_spec("constant-rt")
    path = str(tmp_path / spec.filename)
    _ConstantFusion(0.9).save(path)
    restored = spec.load(path)
    assert restored.value == 0.9


# --------------------------------------------------------------------------- #
# Persistence component resolution (incl. legacy back-compat)
# --------------------------------------------------------------------------- #
def test_components_from_meta_reads_explicit_block():
    meta = {"components": {"encoder": "hashing", "fusion": "lightgbm", "calibrator": "platt"}}
    got = ArtifactRepository._components_from_meta(meta)
    assert got == {"encoder": "hashing", "fusion": "lightgbm", "calibrator": "platt"}


def test_components_from_meta_falls_back_to_config_kinds():
    meta = {"config": {"encoder": {"kind": "hashing"}, "fusion": {"kind": "xgboost"},
                       "calibration": {"kind": "isotonic"}}}
    got = ArtifactRepository._components_from_meta(meta)
    assert got == {"encoder": "hashing", "fusion": "xgboost", "calibrator": "isotonic"}


def test_components_from_meta_legacy_defaults():
    """A model dir written before T23 has neither block → built-in defaults."""
    got = ArtifactRepository._components_from_meta({"config": {}})
    assert got == {
        "encoder": "sentence-transformers",
        "fusion": "xgboost",
        "calibrator": "isotonic",
    }


# --------------------------------------------------------------------------- #
# End-to-end: a config-selected custom backend trains, persists, and predicts
# with no edits to TrainingPipeline or ArtifactRepository.
# --------------------------------------------------------------------------- #
def test_end_to_end_with_custom_fusion(tmp_path):
    from text_classifier.application.inference import InferencePipeline
    from text_classifier.application.training import TrainingPipeline
    from text_classifier.config import PipelineConfig, RetrievalConfig, TrainingConfig
    from tests._doubles import make_synthetic

    register_fusion(
        "constant-e2e",
        FusionSpec(
            build=lambda cfg: _ConstantFusion(cfg.params.get("value", 0.6)),
            filename="constant.json",
            load=_ConstantFusion.load,
        ),
    )

    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=5)
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"                                   # encoder seam
    cfg.fusion = FusionConfig(kind="constant-e2e", params={"value": 0.6})  # fusion seam
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1,
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)

    # No shared_encoder injected: the encoder is built from config too.
    artifacts, report = TrainingPipeline(cfg).run(items, label_space)
    assert report.n_items > 0
    assert isinstance(artifacts.fusion, _ConstantFusion)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    with open(f"{model_dir}/meta.json") as fh:
        meta = json.load(fh)
    assert meta["components"] == {
        "encoder": "hashing", "fusion": "constant-e2e", "calibrator": "isotonic",
    }

    loaded = ArtifactRepository().load(model_dir)
    assert isinstance(loaded.fusion, _ConstantFusion)
    preds = InferencePipeline(loaded).predict([it.text for it in items[:5]])
    assert len(preds) == 5
