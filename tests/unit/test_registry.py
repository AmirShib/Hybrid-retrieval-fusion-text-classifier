"""T23 — Component registry + factory DI.

Proves the seam works without touching TrainingPipeline or ArtifactRepository:
a backend is wired in purely by registering a spec and selecting its kind in
config. Also pins the error contract and the persistence back-compat path.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from text_classifier.config import CalibrationConfig, EncoderConfig, FusionConfig, RetrievalConfig
from text_classifier.domain import (
    ConfidenceCalibrator,
    DenseRetriever,
    FusionModel,
    LexicalRetriever,
)
from text_classifier.infrastructure import (
    DenseRetrieverSpec,
    FusionSpec,
    LexicalRetrieverSpec,
    build_calibrator,
    build_dense_retriever,
    build_encoder,
    build_fusion,
    build_lexical_retriever,
    register_dense_retriever,
    register_fusion,
    register_lexical_retriever,
)
from text_classifier.infrastructure.persistence import ArtifactRepository
from text_classifier.infrastructure.retrieval import DenseRetrieverAdapter, LexicalRetrieverAdapter


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
    meta = {
        "components": {
            "encoder": "hashing",
            "fusion": "lightgbm",
            "calibrator": "platt",
            "dense": "exact",
            "lexical": "bm25",
            "signals": ["dense", "lexical"],
        }
    }
    got = ArtifactRepository._components_from_meta(meta)
    assert got == {
        "encoder": "hashing",
        "fusion": "lightgbm",
        "calibrator": "platt",
        "dense": "exact",
        "lexical": "bm25",
        "signals": ["dense", "lexical"],
    }


def test_components_from_meta_falls_back_to_config_kinds():
    meta = {
        "config": {
            "encoder": {"kind": "hashing"},
            "fusion": {"kind": "xgboost"},
            "calibration": {"kind": "isotonic"},
            "retrieval": {"dense_kind": "exact", "lexical_kind": "bm25"},
            "signals": ["dense", "lexical"],
        }
    }
    got = ArtifactRepository._components_from_meta(meta)
    assert got == {
        "encoder": "hashing",
        "fusion": "xgboost",
        "calibrator": "isotonic",
        "dense": "exact",
        "lexical": "bm25",
        "signals": ["dense", "lexical"],
    }


def test_components_from_meta_legacy_defaults():
    """A model dir written before T23 (or before T34, for dense/lexical/signals)
    has neither block → built-in defaults."""
    got = ArtifactRepository._components_from_meta({"config": {}})
    assert got == {
        "encoder": "sentence-transformers",
        "fusion": "xgboost",
        "calibrator": "isotonic",
        "dense": "exact",
        "lexical": "bm25",
        "signals": ["dense", "lexical"],
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
    cfg.encoder.kind = "hashing"  # encoder seam
    cfg.fusion = FusionConfig(kind="constant-e2e", params={"value": 0.6})  # fusion seam
    cfg.training = TrainingConfig(
        n_folds=3,
        random_state=0,
        target_precision=0.5,
        per_class_min_support=1,
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
        "encoder": "hashing",
        "fusion": "constant-e2e",
        "calibrator": "isotonic",
        "dense": "exact",
        "lexical": "bm25",
        "signals": ["dense", "lexical"],
    }

    loaded = ArtifactRepository().load(model_dir)
    assert isinstance(loaded.fusion, _ConstantFusion)
    preds = InferencePipeline(loaded).predict([it.text for it in items[:5]])
    assert len(preds) == 5


# --------------------------------------------------------------------------- #
# T34 phase 1 — dense/lexical retriever builders behind the registry
#
# Both dummy backends below are thin wrappers around the built-in adapters:
# the point of these tests is proving the registry seam (build/persist/load
# dispatch by `cfg.retrieval.dense_kind`/`lexical_kind`), not reimplementing
# kNN/BM25 math.
# --------------------------------------------------------------------------- #
class _EchoDenseRetriever(DenseRetriever):
    def __init__(self, inner: DenseRetrieverAdapter):
        self._inner = inner

    def knn_example_labels(self, query_emb, k, exclude_idx=None):
        return self._inner.knn_example_labels(query_emb, k, exclude_idx)

    def prototype_similarity(self, query_emb):
        return self._inner.prototype_similarity(query_emb)

    def description_similarity(self, query_emb):
        return self._inner.description_similarity(query_emb)

    @property
    def class_freq(self):
        return self._inner.class_freq

    @classmethod
    def build(cls, encoder, texts, labels, label_space, cfg, array_ops=None):
        return cls(DenseRetrieverAdapter.build(encoder, texts, labels, label_space, cfg, array_ops))

    def to_state(self):
        return self._inner.to_state()

    @classmethod
    def from_state(cls, arrays, chunk=256, array_ops=None):
        return cls(DenseRetrieverAdapter.from_state(arrays, chunk=chunk, array_ops=array_ops))


class _EchoLexicalRetriever(LexicalRetriever):
    def __init__(self, inner: LexicalRetrieverAdapter):
        self._inner = inner

    def knn_example_labels(self, query_texts, k, exclude_idx=None):
        return self._inner.knn_example_labels(query_texts, k, exclude_idx)

    def description_score(self, query_texts):
        return self._inner.description_score(query_texts)

    @classmethod
    def build(cls, texts, labels, label_space, cfg):
        return cls(LexicalRetrieverAdapter.build(texts, labels, label_space, cfg))

    def to_state(self):
        return self._inner.to_state()

    @classmethod
    def from_state(cls, arrays, meta):
        return cls(LexicalRetrieverAdapter.from_state(arrays, meta))


def _load_echo_dense(directory, cfg):
    arrays = dict(np.load(os.path.join(directory, "echo_dense.npz")))
    return _EchoDenseRetriever.from_state(arrays, chunk=cfg.dense_chunk)


def _load_echo_lexical(directory):
    arrays = dict(np.load(os.path.join(directory, "echo_lexical.npz")))
    with open(os.path.join(directory, "echo_lexical.json")) as fh:
        meta = json.load(fh)
    return _EchoLexicalRetriever.from_state(arrays, meta)


def _register_echo_retrievers():
    register_dense_retriever(
        "echo-dense",
        DenseRetrieverSpec(
            build=_EchoDenseRetriever.build,
            filename="echo_dense.npz",  # deliberately not "dense.npz" -- proves the
            load=_load_echo_dense,  # filename is honoured, not hardcoded downstream
        ),
    )
    register_lexical_retriever(
        "echo-bm25",
        LexicalRetrieverSpec(
            build=_EchoLexicalRetriever.build,
            filename="echo_lexical.npz",
            load=_load_echo_lexical,
        ),
    )


def test_register_and_build_custom_dense_retriever():
    _register_echo_retrievers()
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=6, per_class=10, seed=1)
    from text_classifier.infrastructure import HashingEncoder

    encoder = HashingEncoder()
    texts = [it.text for it in items]
    labels = np.array(label_space.encode_labels([it.label for it in items]))
    retriever = build_dense_retriever(
        RetrievalConfig(dense_kind="echo-dense"), encoder, texts, labels, label_space
    )
    assert isinstance(retriever, _EchoDenseRetriever)
    assert retriever.class_freq.shape == (label_space.size,)


def test_register_and_build_custom_lexical_retriever():
    _register_echo_retrievers()
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=6, per_class=10, seed=1)
    texts = [it.text for it in items]
    labels = np.array(label_space.encode_labels([it.label for it in items]))
    retriever = build_lexical_retriever(
        RetrievalConfig(lexical_kind="echo-bm25"), texts, labels, label_space
    )
    assert isinstance(retriever, _EchoLexicalRetriever)


def test_unknown_dense_kind_lists_registered():
    with pytest.raises(ValueError, match="unknown dense retriever kind 'nope'") as exc:
        build_dense_retriever(RetrievalConfig(dense_kind="nope"), None, [], np.array([]), None)
    assert "exact" in str(exc.value)


def test_unknown_lexical_kind_lists_registered():
    with pytest.raises(ValueError, match="unknown lexical retriever kind 'nope'") as exc:
        build_lexical_retriever(RetrievalConfig(lexical_kind="nope"), [], np.array([]), None)
    assert "bm25" in str(exc.value)


def test_dense_and_lexical_kind_selected_persisted_and_loaded_end_to_end(tmp_path):
    """A config-selected custom dense/lexical retriever trains, persists under
    its own filenames, records its kind in meta.json, and loads back to score --
    mirroring test_end_to_end_with_custom_fusion for the retriever seam."""
    from text_classifier.application.inference import InferencePipeline
    from text_classifier.application.training import TrainingPipeline
    from text_classifier.config import PipelineConfig, TrainingConfig
    from tests._doubles import make_synthetic

    _register_echo_retrievers()

    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=5)
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.retrieval = RetrievalConfig(
        k_neighbors=10, dense_kind="echo-dense", lexical_kind="echo-bm25"
    )
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )

    artifacts, report = TrainingPipeline(cfg).run(items, label_space)
    assert report.n_items > 0
    assert isinstance(artifacts.dense, _EchoDenseRetriever)
    assert isinstance(artifacts.lexical, _EchoLexicalRetriever)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    assert os.path.isfile(os.path.join(model_dir, "echo_dense.npz"))
    assert os.path.isfile(os.path.join(model_dir, "echo_lexical.npz"))
    assert os.path.isfile(os.path.join(model_dir, "echo_lexical.json"))
    assert not os.path.exists(os.path.join(model_dir, "dense.npz"))
    assert not os.path.exists(os.path.join(model_dir, "lexical.npz"))

    with open(f"{model_dir}/meta.json") as fh:
        meta = json.load(fh)
    assert meta["components"]["dense"] == "echo-dense"
    assert meta["components"]["lexical"] == "echo-bm25"

    loaded = ArtifactRepository().load(model_dir)
    assert isinstance(loaded.dense, _EchoDenseRetriever)
    assert isinstance(loaded.lexical, _EchoLexicalRetriever)
    preds = InferencePipeline(loaded).predict([it.text for it in items[:5]])
    assert len(preds) == 5


def test_dense_lexical_defaults_recorded_in_components():
    """The default kinds ("exact"/"bm25") are what a plain `save()` records --
    the byte-identical-with-before contract for a model trained with no
    dense_kind/lexical_kind override."""
    meta = ArtifactRepository._components_from_meta({"config": {}})
    assert meta["dense"] == "exact"
    assert meta["lexical"] == "bm25"
