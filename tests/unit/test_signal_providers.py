"""T34 phase 2 — SignalProvider registry, config-driven selection, and the
schema-drift contract for the (now pluggable) retrieval signals.

Mirrors ``test_registry.py``'s style for the dense/lexical retriever seam
(T34 phase 1): a toy third-party signal provider is registered purely via
``register_signal_provider`` + ``PipelineConfig.signals``, with zero edits to
``FeatureAssembler``/``TrainingPipeline``/``ArtifactRepository``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, RetrievalConfig, TrainingConfig
from text_classifier.domain import SignalContext, SignalMatrix, SignalProvider
from text_classifier.infrastructure import (
    ArtifactRepository,
    SignalProviderSpec,
    register_signal_provider,
)
from tests._doubles import make_synthetic


# --------------------------------------------------------------------------- #
# A toy, fully stateless third-party signal: one-hot on `len(text) % n_classes`.
# Deliberately trivial (no fit/learned state) so persistence is a formality —
# the point of the test is the wiring, not the signal's predictive value.
# --------------------------------------------------------------------------- #
class _TextLengthSignalProvider(SignalProvider):
    name = "textlen"

    def candidate_features(self):
        return ("textlen.score",)

    def column_names(self):
        return ["textlen_score"]

    def build(self, ctx: SignalContext):
        b = len(ctx.texts)
        C = ctx.n_classes
        picks = np.array([len(t) % C for t in ctx.texts], dtype=np.int64)
        M = np.full((b, C), np.nan, dtype=np.float64)
        M[np.arange(b), picks] = 1.0
        return [
            SignalMatrix(
                node="textlen.score",
                value=M,
                derive=frozenset({"raw"}),
                columns={"raw": "textlen_score"},
            )
        ]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "textlen.json"), "w") as fh:
            json.dump({"stateless": True}, fh)

    @classmethod
    def load(cls, path: str) -> "_TextLengthSignalProvider":
        return cls()


def _register_textlen_signal():
    register_signal_provider(
        "textlen",
        SignalProviderSpec(
            build=lambda cfg, dense, lexical, ops: _TextLengthSignalProvider(),
            load=lambda directory, cfg: _TextLengthSignalProvider.load(
                os.path.join(directory, "signals", "textlen")
            ),
        ),
    )


def _train(cfg, seed=5, n_classes=6, per_class=15):
    label_space, items = make_synthetic(n_classes=n_classes, per_class=per_class, seed=seed)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    artifacts, report = TrainingPipeline(cfg).run(items, label_space)
    return artifacts, report, items


# --------------------------------------------------------------------------- #
# Default config: two built-in providers, byte-for-byte identical schema.
# --------------------------------------------------------------------------- #
def test_default_signals_config_is_dense_and_lexical():
    assert PipelineConfig().signals == ["dense", "lexical"]


# --------------------------------------------------------------------------- #
# End-to-end: a third signal joins the candidate union, adds a column, and
# round-trips through train -> save -> load -> predict with identical scores.
# --------------------------------------------------------------------------- #
def test_toy_signal_provider_end_to_end(tmp_path):
    _register_textlen_signal()

    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "textlen"]
    artifacts, report, items = _train(cfg)
    assert report.n_items > 0

    names = [type(p).__name__ for p in artifacts.signal_providers]
    assert names == ["DenseSignalProvider", "LexicalSignalProvider", "_TextLengthSignalProvider"]

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    with open(f"{model_dir}/meta.json") as fh:
        meta = json.load(fh)
    assert meta["components"]["signals"] == ["dense", "lexical", "textlen"]
    assert "textlen_score" in meta["feature_names"]
    assert os.path.isfile(os.path.join(model_dir, "signals", "textlen", "textlen.json"))
    # The two built-ins have nothing extra to persist -- no signals/dense or
    # signals/lexical directory is written.
    assert not os.path.exists(os.path.join(model_dir, "signals", "dense"))
    assert not os.path.exists(os.path.join(model_dir, "signals", "lexical"))

    texts = [it.text for it in items[:10]]
    before = InferencePipeline(artifacts).predict(texts)

    loaded = ArtifactRepository().load(model_dir)
    loaded_names = [type(p).__name__ for p in loaded.signal_providers]
    assert loaded_names == [
        "DenseSignalProvider",
        "LexicalSignalProvider",
        "_TextLengthSignalProvider",
    ]
    after = InferencePipeline(loaded).predict(texts)

    assert [p.top_key for p in before] == [p.top_key for p in after]
    assert [p.abstained for p in before] == [p.abstained for p in after]
    np.testing.assert_allclose(
        [p.confidence for p in before], [p.confidence for p in after], rtol=0, atol=1e-9
    )

    # The extra column actually reached the fusion model and the assembled frame.
    explained = InferencePipeline(loaded).explain(texts)
    assert "textlen_score" in explained.columns


# --------------------------------------------------------------------------- #
# Schema drift: a model dir naming an unregistered signal kind fails clearly.
# --------------------------------------------------------------------------- #
def test_unknown_signal_kind_lists_registered():
    from text_classifier.infrastructure import signal_provider_spec

    with pytest.raises(ValueError, match="unknown signal provider kind 'nope'") as exc:
        signal_provider_spec("nope")
    assert "dense" in str(exc.value) and "lexical" in str(exc.value)


def test_loading_model_with_unregistered_signal_provider_fails_clearly(tmp_path):
    """A ``meta.json`` naming a signal kind the running code has never
    registered (e.g. a plugin that was uninstalled, or a typo'd custom kind)
    must fail with a message naming the missing kind -- the same schema-drift
    contract as an unrecognized encoder/fusion/calibrator/dense/lexical kind."""
    _register_textlen_signal()
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "textlen"]
    artifacts, _, _ = _train(cfg)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["components"]["signals"] = ["dense", "lexical", "a-signal-that-does-not-exist"]
    meta["config"]["signals"] = meta["components"]["signals"]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    with pytest.raises(ValueError, match="unknown signal provider kind") as exc:
        ArtifactRepository().load(model_dir)
    assert "a-signal-that-does-not-exist" in str(exc.value)


def test_legacy_model_dir_defaults_signals_to_dense_and_lexical(tmp_path):
    """A model dir saved before ``components.signals`` existed (no key at all)
    still loads, defaulting to the two built-ins -- the same legacy-default
    contract ``dense_kind``/``lexical_kind`` already have."""
    cfg = PipelineConfig()
    artifacts, _, items = _train(cfg, n_classes=5, per_class=10)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)

    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    del meta["components"]["signals"]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    loaded = ArtifactRepository().load(model_dir)
    names = [p.name for p in loaded.signal_providers]
    assert names == ["dense", "lexical"]
    # And it still predicts.
    preds = InferencePipeline(loaded).predict([it.text for it in items[:5]])
    assert len(preds) == 5


# --------------------------------------------------------------------------- #
# SignalMatrix.column_for
# --------------------------------------------------------------------------- #
class TestSignalMatrixColumnFor:
    """`derive` and `columns` answer two halves of one question — does this
    derivation apply here, and what is it called. `column_for` is where they are
    combined, so the assembler's table-driven loop can ask once per derivation.
    """

    def _matrix(self, **kwargs):
        return SignalMatrix(node="toy.sig", value=np.zeros((2, 3)), **kwargs)

    def test_returns_the_column_name_for_a_declared_derivation(self):
        sm = self._matrix(derive=frozenset({"raw", "rank"}), columns={"raw": "x", "rank": "r_x"})
        assert sm.column_for("raw") == "x"
        assert sm.column_for("rank") == "r_x"

    def test_returns_none_for_a_derivation_this_matrix_does_not_declare(self):
        sm = self._matrix(derive=frozenset({"raw"}), columns={"raw": "x"})
        assert sm.column_for("norm") is None

    def test_a_derivation_named_in_derive_but_unnamed_in_columns_is_a_no_op(self):
        """Half a declaration is not a derivation: it must read as absent rather
        than raise in one code path and be silently skipped in another."""
        sm = self._matrix(derive=frozenset({"raw", "margin"}), columns={"raw": "x"})
        assert sm.column_for("margin") is None

    def test_a_column_named_without_being_derived_is_also_a_no_op(self):
        sm = self._matrix(derive=frozenset({"raw"}), columns={"raw": "x", "norm": "n_x"})
        assert sm.column_for("norm") is None

    def test_defaults_declare_nothing(self):
        assert self._matrix().column_for("raw") is None


def test_builtin_providers_declare_every_column_they_name():
    """Each built-in `SignalMatrix`'s declared derivations must resolve to a
    name, and those names must be exactly what `column_names()` advertises —
    the contract `composed_feature_names` and the persisted schema rely on."""
    cfg = PipelineConfig()
    artifacts, _, items = _train(cfg)
    texts = [it.text for it in items[:3]]

    for provider in artifacts.signal_providers:
        declared = set(provider.column_names())
        ctx = SignalContext(
            texts=texts,
            q_emb=artifacts.encoder.encode_queries(texts),
            k=cfg.retrieval.k_neighbors,
            n_classes=artifacts.label_space.size,
        )
        for sm in provider.build(ctx):
            for derivation in sm.derive:
                name = sm.column_for(derivation)
                assert name is not None, f"{sm.node} declares {derivation} with no column name"
                assert name in declared, f"{sm.node}.{derivation} -> {name} not in column_names()"
