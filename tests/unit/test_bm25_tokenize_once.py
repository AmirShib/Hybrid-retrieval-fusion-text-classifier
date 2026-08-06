"""T32 A1/A2 — tokenize the BM25 corpus once per training run, build the
description index once, not once per fold.

`CountVectorizer.fit_transform` is a pure function of the text (given fixed
`bm25_token_kwargs`), so a full `n_folds` run should tokenize the example pool
exactly once and build the description BM25 index exactly once — not one pair
per fold plus a second pair for the deployment index. Counted directly through
monkeypatched `BM25Index` methods rather than timed, so the assertion is exact
and CI-stable.
"""

from __future__ import annotations

import pytest

import text_classifier.infrastructure.retrieval as retrieval_mod
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import FusionConfig, PipelineConfig, RetrievalConfig, TrainingConfig
from tests._doubles import make_synthetic


def _cfg(n_folds: int, retrieval: RetrievalConfig = None) -> PipelineConfig:
    cfg = PipelineConfig(candidate_top_n=5)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=n_folds,
        target_precision=0.5,
        per_class_min_support=1,
        random_state=0,
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 8, "max_depth": 2, "n_jobs": 1, "random_state": 0}
    )
    if retrieval is not None:
        cfg.retrieval = retrieval
    return cfg


@pytest.fixture
def corpus():
    label_space, items = make_synthetic(n_classes=6, per_class=8, seed=9)
    return label_space, items


class _Counters:
    def __init__(self):
        self.tokenize_calls = 0
        self.tokenize_rows = 0
        self.desc_fit_calls = 0


@pytest.fixture
def counting_bm25(monkeypatch):
    """Wraps BM25Index.tokenize_corpus (A2) and BM25Index.fit (A1's
    description-index build) to count calls without changing behaviour."""
    counters = _Counters()
    real_tokenize = retrieval_mod.BM25Index.tokenize_corpus
    real_fit = retrieval_mod.BM25Index.fit

    def counting_tokenize(corpus_arg, **kw):
        counters.tokenize_calls += 1
        counters.tokenize_rows += len(corpus_arg)
        return real_tokenize(corpus_arg, **kw)

    def counting_fit(self, corpus_arg):
        counters.desc_fit_calls += 1
        return real_fit(self, corpus_arg)

    monkeypatch.setattr(retrieval_mod.BM25Index, "tokenize_corpus", staticmethod(counting_tokenize))
    monkeypatch.setattr(retrieval_mod.BM25Index, "fit", counting_fit)
    return counters


class TestSharedBM25TokenizationPath:
    def test_corpus_tokenized_once_and_description_index_built_once(self, corpus, counting_bm25):
        """`tokenize_calls` counts every `tokenize_corpus` call, including the
        one `BM25Index.fit` makes internally — so the shared path makes
        exactly two: one directly (the example pool, via `tokenize_corpus`
        called from `_shared_lexical_state`) and one via `.fit()` (the
        description index, A1). Neither scales with `n_folds`; today's code
        makes `2 * (n_folds + 1)`."""
        label_space, items = corpus
        n, C = len(items), label_space.size
        TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        assert counting_bm25.tokenize_calls == 2
        assert counting_bm25.tokenize_rows == n + C
        # The description index is the only thing built via `.fit()` — every
        # fold and the deployment index reuse it via `build_from_counts`
        # (example side) and the cached `BM25Index` object (description side).
        assert counting_bm25.desc_fit_calls == 1

    def test_scales_with_n_folds_not_n_folds_squared(self, corpus, counting_bm25):
        label_space, items = corpus
        TrainingPipeline(_cfg(n_folds=3)).run(items, label_space)
        calls_at_3 = counting_bm25.tokenize_calls
        TrainingPipeline(_cfg(n_folds=8)).run(items, label_space)
        calls_at_8 = counting_bm25.tokenize_calls - calls_at_3
        assert calls_at_3 == calls_at_8 == 2

    def test_byte_identical_lexical_scores_on_a_fixed_corpus(self, corpus):
        """The refactor must not change a single feature value."""
        label_space, items = corpus
        _, report_a = TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        _, report_b = TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        assert report_a == report_b


class TestVocabPruningGuard:
    def test_min_df_falls_back_to_per_fold_example_tokenization(self, corpus, counting_bm25):
        """A2's guard: `min_df` prunes vocabulary by corpus statistics, so a
        full-corpus tokenization would differ from a per-fold one. The example
        side must fall back to fitting fresh per fold (`n_folds` calls) plus
        once more for the deployment index — but the description index (A1)
        stays shared regardless, since A2's concern never applies to it."""
        label_space, items = corpus
        cfg = _cfg(n_folds=3, retrieval=RetrievalConfig(bm25_token_kwargs={"min_df": 1}))
        pipeline = TrainingPipeline(cfg)
        _, report = pipeline.run(items, label_space)
        assert report.n_items > 0
        assert not pipeline._indexes.example_counts_cached
        assert pipeline._indexes.description_index_cached  # A1 still shared
        # One `.fit()` (and its internal `tokenize_corpus`) for the shared
        # description index, plus one per fold and one for the deployment
        # index on the example side — none of those reach `build_from_counts`.
        n_folds = cfg.training.n_folds
        assert counting_bm25.desc_fit_calls == n_folds + 2
        assert counting_bm25.tokenize_calls == n_folds + 2

    @pytest.mark.parametrize("kwarg", ["max_df", "max_features"])
    def test_other_vocab_pruning_kwargs_also_fall_back(self, corpus, kwarg):
        label_space, items = corpus
        value = 0.9 if kwarg == "max_df" else 50
        cfg = _cfg(n_folds=3, retrieval=RetrievalConfig(bm25_token_kwargs={kwarg: value}))
        pipeline = TrainingPipeline(cfg)
        _, report = pipeline.run(items, label_space)
        assert report.n_items > 0
        assert not pipeline._indexes.example_counts_cached


class TestMaxDfRatioTrainingIntegration:
    def test_none_is_byte_identical_to_default(self, corpus):
        label_space, items = corpus
        _, report_default = TrainingPipeline(_cfg(n_folds=5)).run(items, label_space)
        _, report_explicit_none = TrainingPipeline(
            _cfg(n_folds=5, retrieval=RetrievalConfig(bm25_max_df_ratio=None))
        ).run(items, label_space)
        assert report_default == report_explicit_none

    def test_set_ratio_persists_and_still_trains(self, tmp_path, corpus):
        label_space, items = corpus
        cfg = _cfg(n_folds=5, retrieval=RetrievalConfig(bm25_max_df_ratio=0.9))
        out = str(tmp_path / "model")
        TrainingPipeline(cfg).run(items, label_space, output_dir=out)

        import json
        import os

        with open(os.path.join(out, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["config"]["retrieval"]["bm25_max_df_ratio"] == 0.9
