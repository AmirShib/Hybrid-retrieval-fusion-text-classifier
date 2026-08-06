"""T85 -- SentenceTransformerEncoder's array_backend: "numpy" (default, every
existing caller) stays exactly as before; "torch" hands back a resident,
still-L2-normalized tensor instead of forcing a numpy conversion. Runs
against a stubbed model (no real download), consistent with the rest of the
encoder test suite; skipped entirely when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from text_classifier.infrastructure.encoder import SentenceTransformerEncoder  # noqa: E402


class _StubSTModel:
    """Mirrors SentenceTransformer.encode: honors normalize_embeddings, and
    convert_to_numpy=False/convert_to_tensor=True like the real thing."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append(dict(kwargs))
        rng = np.random.default_rng(len(texts))
        emb = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        if kwargs.get("normalize_embeddings"):
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        if kwargs.get("convert_to_tensor"):
            return torch.as_tensor(emb)
        assert kwargs.get("convert_to_numpy") is True
        return emb


def _adapter() -> tuple[SentenceTransformerEncoder, _StubSTModel]:
    stub = _StubSTModel()
    return SentenceTransformerEncoder(stub, batch_size=8), stub  # type: ignore[arg-type]


def test_default_array_backend_is_numpy_unchanged():
    enc, stub = _adapter()
    out = enc.encode(["a", "b"])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert stub.calls[0]["convert_to_numpy"] is True
    assert "convert_to_tensor" not in stub.calls[0]


def test_torch_backend_returns_resident_normalized_tensor():
    enc, stub = _adapter()
    enc.set_array_backend("torch")
    out = enc.encode_documents(["a", "b", "c"])
    assert isinstance(out, torch.Tensor)
    call = stub.calls[0]
    assert call["convert_to_numpy"] is False
    assert call["convert_to_tensor"] is True
    assert call["normalize_embeddings"] is True
    norms = torch.linalg.norm(out, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_switching_backend_affects_every_subsequent_encode_call():
    enc, stub = _adapter()
    assert isinstance(enc.encode(["x"]), np.ndarray)
    enc.set_array_backend("torch")
    assert isinstance(enc.encode(["x"]), torch.Tensor)
    enc.set_array_backend("numpy")
    assert isinstance(enc.encode(["x"]), np.ndarray)


def test_convert_kwargs_cannot_be_overridden_via_encode_kwargs():
    stub = _StubSTModel()
    enc = SentenceTransformerEncoder(
        stub,
        batch_size=8,
        encode_kwargs={"convert_to_numpy": False, "convert_to_tensor": True},
    )
    out = enc.encode(["a"])
    assert isinstance(out, np.ndarray)  # protected keys win regardless of user kwargs
