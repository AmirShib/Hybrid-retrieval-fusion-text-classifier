"""T63 — torch-optional install: a missing `sentence-transformers` extra must
raise a clear, actionable error, not a raw ImportError, from every entry point
that needs the package. Simulates the extra being absent by making the import
fail regardless of whether it is actually installed in this environment.
"""

from __future__ import annotations

import builtins
import sys

import pytest

from text_classifier.config import EncoderConfig
from text_classifier.infrastructure.encoder import SentenceTransformerEncoder, train_encoder
from text_classifier.infrastructure.registry import build_encoder


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Force every `import sentence_transformers` (direct or `from ... import`)
    to fail, whether or not the real package is installed here."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)


def _assert_actionable(exc: ImportError) -> None:
    msg = str(exc)
    assert "pip install text-classifier[sentence-transformers]" in msg
    assert "tfidf" in msg and "hashing" in msg


def test_build_encoder_raises_actionable_import_error(no_sentence_transformers):
    with pytest.raises(ImportError) as exc:
        build_encoder(EncoderConfig(kind="sentence-transformers"))
    _assert_actionable(exc.value)


def test_sentence_transformer_encoder_load_raises_actionable_import_error(
    no_sentence_transformers,
):
    with pytest.raises(ImportError) as exc:
        SentenceTransformerEncoder.load("sentence-transformers/all-MiniLM-L6-v2")
    _assert_actionable(exc.value)


def test_train_encoder_raises_actionable_import_error(no_sentence_transformers):
    with pytest.raises(ImportError) as exc:
        train_encoder([], None, EncoderConfig())  # type: ignore[arg-type]
    _assert_actionable(exc.value)
