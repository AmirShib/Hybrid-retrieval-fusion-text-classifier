"""T28 — Encode-time kwargs + asymmetric query/document encoding.

Part A: SentenceTransformerEncoder against a stubbed model (no torch, no
        download): prompt logic, encode_kwargs merge, forced-invariant keys,
        unit-norm output.
Part B: the TextEncoder port's default role methods delegate to encode(),
        so symmetric encoders need no changes.
Part C: role routing through both pipelines with a recording double:
        queries (items) vs documents (example pool + class descriptions).
Part D: the new EncoderConfig fields survive to_dict/from_dict.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import EncoderConfig, FusionConfig, PipelineConfig, TrainingConfig
from text_classifier.domain import TextEncoder
from text_classifier.infrastructure.encoder import SentenceTransformerEncoder
from tests._doubles import HashingEncoder, make_synthetic


# --------------------------------------------------------------------------- #
# Part A — SentenceTransformerEncoder with a stubbed model
# --------------------------------------------------------------------------- #
class _StubSTModel:
    """Mirrors the SentenceTransformer.encode signature: records every call's
    (texts, kwargs) and honors normalize_embeddings like the real thing."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        rng = np.random.default_rng(len(texts))
        emb = (rng.normal(size=(len(texts), self.dim)) * 5.0).astype(np.float32)
        if kwargs.get("normalize_embeddings"):
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


def _adapter(**kw) -> tuple[SentenceTransformerEncoder, _StubSTModel]:
    stub = _StubSTModel()
    return SentenceTransformerEncoder(stub, batch_size=64, **kw), stub  # type: ignore[arg-type]


class TestStAdapterPrompts:
    def test_default_encode_args_unchanged(self):
        """No new config -> the exact pre-T28 call: same kwargs, texts untouched."""
        enc, stub = _adapter()
        enc.encode(["hello world"])
        texts, kwargs = stub.calls[0]
        assert texts == ["hello world"]
        assert kwargs == {
            "batch_size": 64,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }

    def test_no_prompts_means_all_roles_encode_identically(self):
        enc, stub = _adapter()
        enc.encode(["x"])
        enc.encode_queries(["x"])
        enc.encode_documents(["x"])
        assert stub.calls[0] == stub.calls[1] == stub.calls[2]

    def test_query_prompt_prepended_exactly_once_and_only_to_queries(self):
        enc, stub = _adapter(query_prompt="query: ", document_prompt="passage: ")
        enc.encode_queries(["hello"])
        enc.encode_documents(["hello"])
        enc.encode(["hello"])
        assert stub.calls[0][0] == ["query: hello"]
        assert stub.calls[1][0] == ["passage: hello"]
        assert stub.calls[2][0] == ["hello"]  # role-less encode stays bare

    def test_prompt_name_passed_through_when_no_literal_prompt(self):
        enc, stub = _adapter(query_prompt_name="s2s_query")
        enc.encode_queries(["hello"])
        texts, kwargs = stub.calls[0]
        assert texts == ["hello"]
        assert kwargs["prompt_name"] == "s2s_query"

    def test_literal_prompt_wins_over_prompt_name(self):
        enc, stub = _adapter(query_prompt="query: ", query_prompt_name="s2s_query")
        enc.encode_queries(["hello"])
        texts, kwargs = stub.calls[0]
        assert texts == ["query: hello"]
        assert "prompt_name" not in kwargs


class TestStAdapterEncodeKwargs:
    def test_user_kwargs_passed_through(self):
        enc, stub = _adapter(encode_kwargs={"truncate_dim": 4, "precision": "float32"})
        enc.encode(["a"])
        _, kwargs = stub.calls[0]
        assert kwargs["truncate_dim"] == 4
        assert kwargs["precision"] == "float32"

    def test_user_kwargs_win_over_our_defaults(self):
        enc, stub = _adapter(encode_kwargs={"batch_size": 8, "show_progress_bar": True})
        enc.encode(["a"])
        _, kwargs = stub.calls[0]
        assert kwargs["batch_size"] == 8
        assert kwargs["show_progress_bar"] is True

    @pytest.mark.parametrize("key", ["normalize_embeddings", "convert_to_numpy"])
    def test_invariant_keys_cannot_be_overridden(self, key, caplog):
        with caplog.at_level(logging.WARNING):
            enc, stub = _adapter(encode_kwargs={key: False})
        assert key in caplog.text  # the ignored key is named in the warning
        enc.encode(["a"])
        _, kwargs = stub.calls[0]
        assert kwargs[key] is True

    def test_output_is_unit_norm_under_any_kwargs(self):
        enc, _ = _adapter(
            encode_kwargs={"truncate_dim": 8, "normalize_embeddings": False},
            query_prompt="query: ",
        )
        for emb in (enc.encode(["a", "bb"]), enc.encode_queries(["a", "bb"])):
            npt.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)
            assert emb.dtype == np.float32


# --------------------------------------------------------------------------- #
# Part B — port defaults: symmetric encoders need no changes
# --------------------------------------------------------------------------- #
def test_port_default_role_methods_delegate_to_encode():
    class Minimal(TextEncoder):
        def encode(self, texts):
            return np.full((len(texts), 2), 0.5, dtype=np.float32)

        def save(self, directory):  # pragma: no cover
            pass

    enc = Minimal()
    npt.assert_array_equal(enc.encode_queries(["x"]), enc.encode(["x"]))
    npt.assert_array_equal(enc.encode_documents(["x"]), enc.encode(["x"]))


# --------------------------------------------------------------------------- #
# Part C — pipelines route by role
# --------------------------------------------------------------------------- #
class RoleRecordingEncoder(HashingEncoder):
    """Hashing double that records which role each batch was encoded under."""

    def __init__(self, dim: int = 64):
        super().__init__(dim)
        self.query_batches: list[list[str]] = []
        self.document_batches: list[list[str]] = []

    def encode_queries(self, texts):
        self.query_batches.append(list(texts))
        return super().encode(texts)

    def encode_documents(self, texts):
        self.document_batches.append(list(texts))
        return super().encode(texts)


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 10, "max_depth": 3, "random_state": 0, "n_jobs": 1}
    )
    return cfg


class TestPipelineRoleRouting:
    def test_training_routes_examples_and_descriptions_as_documents(self):
        enc = RoleRecordingEncoder()
        label_space, items = make_synthetic(n_classes=4, per_class=9, seed=5)
        TrainingPipeline(_cfg(), shared_encoder=enc).run(items, label_space)

        # Every fold's validation items went through the query role...
        assert enc.query_batches, "no query-role encodes recorded during training"
        all_texts = {it.text for it in items}
        for batch in enc.query_batches:
            assert set(batch) <= all_texts
        # ...and the class descriptions only ever through the document role.
        assert any(set(batch) == set(label_space.descriptions) for batch in enc.document_batches), (
            "class descriptions were not encoded as documents"
        )
        for batch in enc.query_batches:
            assert not set(batch) & set(label_space.descriptions)

    def test_inference_routes_inputs_as_queries_only(self):
        enc = RoleRecordingEncoder()
        label_space, items = make_synthetic(n_classes=4, per_class=9, seed=5)
        artifacts, _ = TrainingPipeline(_cfg(), shared_encoder=enc).run(items, label_space)

        enc.query_batches.clear()
        enc.document_batches.clear()
        InferencePipeline(artifacts).predict(["some new item"])
        assert enc.query_batches == [["some new item"]]
        assert enc.document_batches == []  # the index is already built


# --------------------------------------------------------------------------- #
# Part D — config round-trip
# --------------------------------------------------------------------------- #
def test_encoder_prompt_fields_survive_config_round_trip():
    cfg = PipelineConfig()
    cfg.encoder = EncoderConfig(
        query_prompt="query: ",
        document_prompt="passage: ",
        query_prompt_name="qn",
        document_prompt_name="dn",
        encode_kwargs={"truncate_dim": 128},
    )
    restored = PipelineConfig.from_dict(cfg.to_dict())
    assert restored.encoder == cfg.encoder


def test_old_serialized_config_without_new_fields_still_loads():
    """A meta.json written before T28 has no prompt fields; defaults apply."""
    d = PipelineConfig().to_dict()
    for key in (
        "encode_kwargs",
        "query_prompt",
        "document_prompt",
        "query_prompt_name",
        "document_prompt_name",
    ):
        d["encoder"].pop(key, None)
    restored = PipelineConfig.from_dict(d)
    assert restored.encoder.query_prompt is None
    assert restored.encoder.encode_kwargs == {}
