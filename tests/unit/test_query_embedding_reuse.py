"""T89 — reuse the pooled document embeddings as query embeddings.

The optimization's whole claim is that it changes *nothing* except how many
times the encoder runs, so the load-bearing test here is bit-identity between
`reuse_query_embeddings="never"` (the pre-T89 path) and `"auto"` (the new
default). Everything else guards the conditions under which reuse is *not*
allowed — which is where a mistake would be silent and scientifically wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    EncoderConfig,
    FusionConfig,
    PipelineConfig,
    TrainingConfig,
)
from text_classifier.domain import TextEncoder
from text_classifier.infrastructure.encoder import SentenceTransformerEncoder
from tests._doubles import HashingEncoder, make_synthetic


class RoleCountingEncoder(HashingEncoder):
    """Symmetric double that counts how many texts went through each role."""

    def __init__(self, dim: int = 64):
        super().__init__(dim)
        self.n_query_texts = 0
        self.n_document_texts = 0

    def encode_queries(self, texts):
        self.n_query_texts += len(list(texts))
        return super().encode(texts)

    def encode_documents(self, texts):
        self.n_document_texts += len(list(texts))
        return super().encode(texts)


class AsymmetricEncoder(RoleCountingEncoder):
    """A custom encoder that genuinely encodes the two roles differently, and
    says so. Reusing a document embedding as a query embedding here is a
    correctness bug, not an optimization."""

    roles_share_encoding = False

    def encode_queries(self, texts):
        self.n_query_texts += len(list(texts))
        return -super(RoleCountingEncoder, self).encode(texts)


class UndeclaredEncoder(TextEncoder):
    """A third-party encoder implementing only the port — no capability
    attribute at all, which is what an outside implementation actually looks
    like. The pipeline must not assume it is symmetric."""

    def __init__(self, dim: int = 64):
        self._inner = HashingEncoder(dim)
        self.n_query_texts = 0

    def encode(self, texts):
        return self._inner.encode(texts)

    def encode_queries(self, texts):
        self.n_query_texts += len(list(texts))
        return self._inner.encode(texts)

    def save(self, directory):  # pragma: no cover - never persisted in this test
        self._inner.save(directory)


def _cfg(**encoder_kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder = EncoderConfig(kind="hashing", **encoder_kwargs)
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 10, "max_depth": 3, "random_state": 0, "n_jobs": 1}
    )
    return cfg


def _data():
    return make_synthetic(n_classes=4, per_class=9, seed=5)


# Derived from the fixture rather than hardcoded: across the folds the held-out
# sets partition the items, so the pre-T89 path encodes each item exactly once
# in the query role.
N_ITEMS = len(_data()[1])


def _run(mode: str, encoder=None):
    label_space, items = _data()
    enc = encoder if encoder is not None else RoleCountingEncoder()
    cfg = _cfg(reuse_query_embeddings=mode)
    artifacts, report = TrainingPipeline(cfg, shared_encoder=enc).run(items, label_space)
    return enc, artifacts, report


# --------------------------------------------------------------------------- #
# The load-bearing test: reuse changes nothing but the encode count
# --------------------------------------------------------------------------- #
def test_auto_is_bit_identical_to_never():
    """The default path must produce identical numbers, not merely close ones."""
    enc_never, _, report_never = _run("never")
    enc_auto, _, report_auto = _run("auto")

    assert report_auto == report_never

    # ...and the point of the exercise: strictly fewer texts encoded.
    assert enc_auto.n_query_texts < enc_never.n_query_texts
    assert enc_auto.n_document_texts == enc_never.n_document_texts


def test_auto_skips_the_query_encode_entirely_on_the_shared_path():
    enc, _, _ = _run("auto")
    assert enc.n_query_texts == 0


def test_never_reproduces_the_pre_t89_query_encode_volume():
    """Across the folds the held-out sets partition the items, so the pre-T89
    path encodes each item exactly once in the query role."""
    enc, _, _ = _run("never")
    assert enc.n_query_texts == N_ITEMS


# --------------------------------------------------------------------------- #
# Where reuse must NOT happen
# --------------------------------------------------------------------------- #
def test_asymmetric_encoder_still_re_encodes_under_auto():
    enc, _, _ = _run("auto", encoder=AsymmetricEncoder())
    assert enc.n_query_texts == N_ITEMS, "an asymmetric encoder must not reuse document embeddings"


def test_encoder_without_the_capability_does_not_reuse():
    enc, _, _ = _run("auto", encoder=UndeclaredEncoder())
    assert enc.n_query_texts == N_ITEMS, "reuse must be opt-in, not assumed for unknown encoders"


def test_per_fold_encoder_populates_no_cache_to_reuse():
    """The correctness guard for a *fine-tuned* encoder is structural, not a
    policy check: embeddings computed before a fold's encoder is trained are
    stale, and the pipeline never caches any on that path. Asserted directly —
    even `"always"`, the most permissive mode, has nothing it could reuse.

    Uses the corpus-fitted `tfidf` kind because it is the fittable encoder the
    torch-free test suite has; `hashing` carries no weights and cannot be fit.
    """
    label_space, items = _data()
    cfg = _cfg(reuse_query_embeddings="always")
    cfg.encoder.kind = "tfidf"
    cfg.training.use_per_fold_encoder = True
    pipeline = TrainingPipeline(cfg)
    pipeline.run(items, label_space)
    assert pipeline._use_per_fold_encoder() is True
    assert not pipeline._indexes.pool_embeddings_cached, (
        "a per-fold fine-tuned encoder must leave no shared embedding cache; "
        "reusing one would feed folds embeddings from the wrong weights"
    )


# --------------------------------------------------------------------------- #
# The "always" override
# --------------------------------------------------------------------------- #
def test_always_forces_reuse_against_a_detected_asymmetry_and_warns(caplog):
    with caplog.at_level("WARNING"):
        enc, _, _ = _run("always", encoder=AsymmetricEncoder())
    assert enc.n_query_texts == 0
    assert any("reuse_query_embeddings='always'" in r.getMessage() for r in caplog.records)


def test_always_does_not_warn_for_a_symmetric_encoder(caplog):
    with caplog.at_level("WARNING"):
        _run("always")
    assert not any("reuse_query_embeddings" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# The capability itself
# --------------------------------------------------------------------------- #
class _StubModel:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, True),
        ({"query_prompt": "query: ", "document_prompt": "passage: "}, False),
        ({"query_prompt": "same: ", "document_prompt": "same: "}, True),
        ({"query_prompt_name": "q", "document_prompt_name": "d"}, False),
        ({"query_prompt_name": "same", "document_prompt_name": "same"}, True),
        ({"query_prompt": "x: ", "query_prompt_name": "ignored", "document_prompt": "x: "}, True),
    ],
)
def test_roles_share_encoding_mirrors_encode_precedence(kwargs, expected):
    """The last case is the subtle one: `_encode` gives an explicit prompt
    precedence over a prompt_name, so a role with both set ignores its
    prompt_name — comparing the raw fields would report a false difference."""
    enc = SentenceTransformerEncoder(_StubModel(), **kwargs)
    assert enc.roles_share_encoding is expected


def test_builtin_torch_free_encoders_declare_symmetry():
    assert HashingEncoder(dim=8).roles_share_encoding is True


def test_mode_is_validated():
    cfg = _cfg(reuse_query_embeddings="sometimes")
    with pytest.raises(ValueError, match="reuse_query_embeddings"):
        cfg.validate()


def test_config_round_trip_and_backward_compat():
    cfg = PipelineConfig()
    cfg.encoder.reuse_query_embeddings = "never"
    assert PipelineConfig.from_dict(cfg.to_dict()).encoder.reuse_query_embeddings == "never"

    # A meta.json written before T89 has no such key; the default applies.
    d = PipelineConfig().to_dict()
    d["encoder"].pop("reuse_query_embeddings")
    assert PipelineConfig.from_dict(d).encoder.reuse_query_embeddings == "auto"


# --------------------------------------------------------------------------- #
# Leave-one-out path (n_folds == 1)
# --------------------------------------------------------------------------- #
def test_loo_path_is_bit_identical_and_skips_the_pool_re_encode():
    label_space, items = _data()
    val = [items[0].__class__("val text %d" % i, items[i].label) for i in range(4)]
    test = [items[0].__class__("test text %d" % i, items[i].label) for i in range(4)]

    frames = {}
    counts = {}
    for mode in ("never", "auto"):
        enc = RoleCountingEncoder()
        cfg = _cfg(reuse_query_embeddings=mode)
        cfg.training.n_folds = 1
        _, report = TrainingPipeline(cfg, shared_encoder=enc).run(
            items, label_space, val_items=val, test_items=test
        )
        frames[mode] = report
        counts[mode] = enc.n_query_texts

    assert frames["auto"] == frames["never"]
    # The whole pool is skipped; the 8 external val/test items are still encoded
    # as queries (they were never in the pool, so there is nothing to reuse).
    assert counts["never"] - counts["auto"] == N_ITEMS
    assert counts["auto"] == len(val) + len(test)
