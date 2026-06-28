"""T24 — TfidfEncoder: a torch-free TextEncoder backend.

Validates the encode contract (shape/dtype/unit-norm), retrieval meaningfulness
(shared tokens → higher cosine), graceful empty/OOV handling, persistence
round-trip, registry wiring, and the per-fold leakage rule for corpus fitting.

All tests run offline (sklearn only, no torch, no download).
"""
from __future__ import annotations

import numpy as np
import pytest

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.config import EncoderConfig
from text_classifier.infrastructure import build_encoder, encoder_is_corpus_dependent, fit_encoder
from text_classifier.infrastructure.encoder import TfidfEncoder


CORPUS = [
    "apple banana cherry",
    "apple apple date",
    "banana date elderberry",
    "fig grape honeydew",
    "grape kiwi lemon",
]


def _fitted(**params) -> TfidfEncoder:
    return TfidfEncoder.from_config(EncoderConfig(kind="tfidf", params=params)).fit(CORPUS)


# --------------------------------------------------------------------------- #
# encode contract
# --------------------------------------------------------------------------- #
def test_encode_shape_dtype_and_unit_norm():
    enc = _fitted()
    out = enc.encode(["apple banana", "grape kiwi"])
    assert out.ndim == 2 and out.shape[0] == 2
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(2), atol=1e-5)


def test_shared_tokens_score_higher_than_disjoint():
    enc = _fitted()
    a = enc.encode(["apple banana"])[0]
    shared = enc.encode(["apple date"])[0]      # shares 'apple'
    disjoint = enc.encode(["fig grape"])[0]      # no overlap
    assert float(a @ shared) > float(a @ disjoint)


def test_empty_and_oov_text_is_finite_zero_vector():
    enc = _fitted()
    out = enc.encode(["", "zzz qqq wholly-unknown"])
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.zeros(2), atol=1e-6)


def test_encode_before_fit_raises():
    enc = TfidfEncoder.from_config(EncoderConfig(kind="tfidf"))
    with pytest.raises(RuntimeError, match="must be fit"):
        enc.encode(["anything"])


def test_vectorizer_params_are_honored():
    # max_features caps the vocabulary dimensionality.
    enc = _fitted(max_features=3)
    assert enc.encode(["apple banana cherry"]).shape[1] == 3


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_identical(tmp_path):
    enc = _fitted()
    before = enc.encode(["apple banana", "grape kiwi lemon"])
    d = str(tmp_path / "encoder")
    enc.save(d)
    restored = TfidfEncoder.load(d)
    after = restored.encode(["apple banana", "grape kiwi lemon"])
    np.testing.assert_allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# registry wiring
# --------------------------------------------------------------------------- #
def test_registered_under_tfidf_and_corpus_dependent():
    cfg = EncoderConfig(kind="tfidf")
    built = build_encoder(cfg)            # unfitted instance from the registry
    assert isinstance(built, TfidfEncoder)
    assert encoder_is_corpus_dependent(cfg) is True


def test_fit_encoder_via_registry_returns_fitted_tfidf():
    cfg = EncoderConfig(kind="tfidf")
    ls = LabelSpace([ClassDefinition("a", "alpha"), ClassDefinition("b", "beta")])
    items = [LabeledItem(t, "a") for t in CORPUS]
    enc = fit_encoder(cfg, items, ls)
    assert isinstance(enc, TfidfEncoder)
    assert np.isclose(np.linalg.norm(enc.encode(["apple banana"])[0]), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# leakage rule: a fold's vocabulary is fit on training rows only
# --------------------------------------------------------------------------- #
def test_fold_vocabulary_excludes_held_out_tokens():
    """The per-fold fit must not see validation rows. A token unique to a held-out
    item must be absent from the encoder fit on the remaining (training) items."""
    cfg = EncoderConfig(kind="tfidf")
    ls = LabelSpace([ClassDefinition("a", "alpha"), ClassDefinition("b", "beta")])
    train_items = [LabeledItem(t, "a") for t in CORPUS]
    held_out = LabeledItem("uniqueheldouttoken something", "b")

    enc = fit_encoder(cfg, train_items, ls)  # train rows only — held_out excluded
    vocab = enc._vectorizer.vocabulary_       # type: ignore[attr-defined]
    assert "uniqueheldouttoken" not in vocab
    # The held-out item still encodes (to a finite vector) against the train vocab.
    v = enc.encode([held_out.text])[0]
    assert np.all(np.isfinite(v))
