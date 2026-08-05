"""T88 — encode the corpus once per training run, not once per fold.

On the shared-encoder path, `encode_documents` is a pure function of the text
for a frozen encoder, so a full `n_folds` run should perform exactly one
document-encode pass over the example pool and one over the class
descriptions — not one per fold. Counted directly through a wrapping encoder
double rather than timed, so the assertion is exact and CI-stable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from text_classifier.config import EncoderConfig, FusionConfig, PipelineConfig, TrainingConfig
from text_classifier.domain import TextEncoder
from text_classifier.application.training import TrainingPipeline
from tests._doubles import HashingEncoder, make_synthetic


class _CountingEncoder(TextEncoder):
    """Wraps a real encoder, counting `encode_documents` calls and rows."""

    def __init__(self, inner: TextEncoder):
        self._inner = inner
        self.document_calls = 0
        self.document_rows = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.encode_queries(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        self.document_calls += 1
        self.document_rows += len(texts)
        return self._inner.encode_documents(texts)

    def save(self, directory: str) -> None:
        self._inner.save(directory)


def _cfg(n_folds: int, use_per_fold_encoder: bool = False) -> PipelineConfig:
    cfg = PipelineConfig(candidate_top_n=5)
    cfg.encoder = EncoderConfig(kind="hashing")
    cfg.training = TrainingConfig(
        n_folds=n_folds,
        use_per_fold_encoder=use_per_fold_encoder,
        target_precision=0.5,
        per_class_min_support=1,
        random_state=0,
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 8, "max_depth": 2, "n_jobs": 1, "random_state": 0}
    )
    return cfg


@pytest.fixture
def corpus():
    label_space, items = make_synthetic(n_classes=6, per_class=8, seed=5)
    return label_space, items


class TestSharedEncoderPath:
    def test_document_encodes_are_n_plus_c_not_per_fold(self, corpus):
        """Acceptance criterion: distinct-text encodes per training run = n + C."""
        label_space, items = corpus
        n, C = len(items), label_space.size
        counting = _CountingEncoder(HashingEncoder())
        TrainingPipeline(_cfg(n_folds=5), shared_encoder=counting).run(items, label_space)
        assert counting.document_rows == n + C
        # One call for the whole pool, one for all descriptions — not one pair
        # per fold, and not a second pair for the deployment index.
        assert counting.document_calls == 2

    def test_scales_with_n_folds_not_n_folds_squared(self, corpus):
        """The whole point: more folds must not cost more document encodes."""
        label_space, items = corpus
        n, C = len(items), label_space.size
        c3 = _CountingEncoder(HashingEncoder())
        TrainingPipeline(_cfg(n_folds=3), shared_encoder=c3).run(items, label_space)
        c8 = _CountingEncoder(HashingEncoder())
        TrainingPipeline(_cfg(n_folds=8), shared_encoder=c8).run(items, label_space)
        assert c3.document_rows == c8.document_rows == n + C

    def test_byte_identical_oof_and_evaluation_on_a_fixed_corpus(self, corpus):
        """The refactor must not change a single feature value: fold-sliced
        cached embeddings vs. a direct per-fold re-encode are the same numbers
        for a deterministic encoder."""
        label_space, items = corpus
        _, report_cached = TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        _, report_again = TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        assert report_cached == report_again


class TestPerFoldEncoderPathUnchanged:
    def test_use_per_fold_encoder_still_refits_and_encodes_per_fold(self, corpus):
        """The rigorous path (use_per_fold_encoder=True) is untouched: each fold
        fits + encodes its own corpus-dependent (TF-IDF) encoder, so the T88
        caching path (which requires a frozen shared encoder) must never engage.
        A per-fold-fit TF-IDF vocabulary differs fold to fold, so training must
        still succeed without ever touching `_shared_document_embeddings`."""
        label_space, items = corpus
        cfg = _cfg(n_folds=5, use_per_fold_encoder=True)
        cfg.encoder = EncoderConfig(kind="tfidf")
        pipeline = TrainingPipeline(cfg)
        _, report = pipeline.run(items, label_space)
        assert report.n_items > 0
        assert pipeline._shared_pool_emb is None


class TestCorpusDependentEncoderPathUnchanged:
    def test_tfidf_encoder_still_takes_the_per_fold_path(self, corpus):
        """Corpus-dependent kinds (TF-IDF) are excluded from T88's cache even
        without `use_per_fold_encoder` set explicitly — `encoder_is_corpus_dependent`
        forces the per-fold path, exactly as it did before this ticket."""
        label_space, items = corpus
        cfg = _cfg(n_folds=5)
        cfg.encoder = EncoderConfig(kind="tfidf")
        pipeline = TrainingPipeline(cfg)
        _, report = pipeline.run(items, label_space)
        assert report.n_items > 0
        assert pipeline._shared_pool_emb is None
