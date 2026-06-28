"""T22 — Edge cases at the retrieval / feature-assembly layer.

Degenerate-but-legal inputs that the T01–T07 synthetic fixture never exercises:
a single-class label space, a class declared with zero training examples, a
``k`` larger than the corpus, and an empty query batch. Each must either produce
a sensible result or raise a clear error — never a silent NaN frame or an
index-out-of-bounds crash.

All tests run offline via HashingEncoder.
"""
from __future__ import annotations

import numpy as np
import pytest

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.application.features import FeatureAssembler
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, RetrievalConfig, TrainingConfig
from text_classifier.domain import CandidatePolicy, FEATURE_NAMES
from text_classifier.infrastructure.retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
)
from tests._doubles import HashingEncoder


DIM = 32


def _enc() -> HashingEncoder:
    return HashingEncoder(dim=DIM)


def _cfg() -> RetrievalConfig:
    # Empty token kwargs: keep every content word so tiny corpora are controllable.
    return RetrievalConfig(bm25_token_kwargs={})


def _assembler(label_space: LabelSpace) -> FeatureAssembler:
    return FeatureAssembler(label_space, CandidatePolicy(10))


# --------------------------------------------------------------------------- #
# Part A — Single-class label space
# --------------------------------------------------------------------------- #
class TestSingleClass:
    def setup_method(self):
        self.ls = LabelSpace.from_pairs([("only", "the only class apple orange")])
        self.texts = ["apple orange", "apple fruit"]
        self.labels = np.array([0, 0])
        self.dense = DenseRetrieverAdapter.build(_enc(), self.texts, self.labels, self.ls, _cfg())
        self.lex = LexicalRetrieverAdapter.build(self.texts, self.labels, self.ls, _cfg())

    def test_prototype_similarity_is_single_column(self):
        sim = self.dense.prototype_similarity(_enc().encode(self.texts))
        assert sim.shape == (2, 1)

    def test_knn_labels_all_zero(self):
        lab, _ = self.dense.knn_example_labels(_enc().encode(self.texts), k=3)
        assert np.all(lab[lab >= 0] == 0)

    def test_description_score_is_single_column(self):
        assert self.lex.description_score(["apple"]).shape == (1, 1)

    def test_feature_assembler_returns_nonempty_frame(self):
        q = _enc().encode(self.texts)
        feats = self._assemble(q)
        assert len(feats) > 0
        for name in FEATURE_NAMES:
            assert name in feats.columns

    def _assemble(self, q):
        return _assembler(self.ls).assemble(
            self.texts, q, self.dense, self.lex, _cfg().k_neighbors,
            query_ids=[0, 1], query_labels=self.labels,
        )

    def test_training_with_single_class_raises_clear_error(self):
        # Chosen behaviour: refuse, because a one-class problem has no negatives
        # to calibrate against.
        items = [LabeledItem(f"apple orange item {i}", "only") for i in range(6)]
        cfg = PipelineConfig()
        cfg.training = TrainingConfig(n_folds=3)
        with pytest.raises(ValueError, match="at least 2 classes"):
            TrainingPipeline(cfg).run(items, self.ls)


# --------------------------------------------------------------------------- #
# Part B — Class declared with zero training examples
# --------------------------------------------------------------------------- #
class TestZeroExampleClass:
    def setup_method(self):
        # class index 2 ("empty") is declared but has no examples.
        self.ls = LabelSpace([
            ClassDefinition("a", "alpha apple"),
            ClassDefinition("b", "beta banana"),
            ClassDefinition("empty", "gamma grape never seen"),
        ])
        self.texts = ["apple", "apple fruit", "banana"]
        self.labels = np.array([0, 0, 1])
        self.dense = DenseRetrieverAdapter.build(_enc(), self.texts, self.labels, self.ls, _cfg())
        self.lex = LexicalRetrieverAdapter.build(self.texts, self.labels, self.ls, _cfg())

    def test_freq_zero_and_prototype_nan(self):
        assert self.dense.class_freq[2] == 0
        assert np.all(np.isnan(self.dense.state.prototypes[2]))

    def test_prototype_similarity_nan_for_empty_class(self):
        sim = self.dense.prototype_similarity(_enc().encode(self.texts))
        assert np.all(np.isnan(sim[:, 2]))

    def test_knn_never_returns_empty_class(self):
        lab, _ = self.dense.knn_example_labels(_enc().encode(self.texts), k=3)
        assert np.all(lab[lab >= 0] != 2)

    def test_features_mark_empty_class_missing(self):
        q = _enc().encode(self.texts)
        feats = _assembler(self.ls).assemble(
            self.texts, q, self.dense, self.lex, _cfg().k_neighbors,
            query_ids=[0, 1, 2], query_labels=self.labels,
        )
        empty_rows = feats[feats["candidate"] == 2]
        assert len(empty_rows) > 0  # surfaces as a candidate via description sim
        assert empty_rows["d_knn_missing"].eq(1.0).all()
        assert empty_rows["d_proto_sim"].isna().all()


# --------------------------------------------------------------------------- #
# Part C — k > number of documents/examples
# --------------------------------------------------------------------------- #
class TestKLargerThanCorpus:
    def test_bm25_topk_pads_to_k(self):
        idx = BM25Index(k1=1.5, b=0.75).fit(["apple", "banana", "cherry", "date", "fig"])
        out_idx, out_score = idx.top_k(["apple banana cherry date fig"], k=1000)
        assert out_idx.shape == (1, 1000)
        assert int((out_idx[0] >= 0).sum()) == 5  # only 5 real docs
        assert int(np.isfinite(out_score[0]).sum()) == 5

    def test_dense_knn_pads_to_k(self):
        ls = LabelSpace.from_pairs([("a", "alpha"), ("b", "beta")])
        texts = ["alpha one", "beta two", "alpha three"]
        dense = DenseRetrieverAdapter.build(_enc(), texts, np.array([0, 1, 0]), ls, _cfg())
        lab, sim = dense.knn_example_labels(_enc().encode(["alpha one"]), k=50)
        assert lab.shape == (1, 50)
        assert int((lab[0] >= 0).sum()) == 3
        assert int(np.isfinite(sim[0]).sum()) == 3

    def test_feature_assembler_k_exceeds_examples(self):
        ls = LabelSpace.from_pairs([("a", "alpha apple"), ("b", "beta banana")])
        texts = ["apple", "banana", "apple banana"]
        labels = np.array([0, 1, 0])
        dense = DenseRetrieverAdapter.build(_enc(), texts, labels, ls, _cfg())
        lex = LexicalRetrieverAdapter.build(texts, labels, ls, _cfg())
        feats = _assembler(ls).assemble(
            texts, _enc().encode(texts), dense, lex, k_neighbors=50,
            query_ids=[0, 1, 2], query_labels=labels,
        )
        assert len(feats) > 0
        assert not feats[FEATURE_NAMES].isnull().all().all()  # not an all-NaN frame


# --------------------------------------------------------------------------- #
# Part D — Empty query batch
# --------------------------------------------------------------------------- #
class TestEmptyBatch:
    def setup_method(self):
        self.ls = LabelSpace.from_pairs([("a", "alpha apple"), ("b", "beta banana")])
        self.texts = ["apple", "banana"]
        self.labels = np.array([0, 1])
        self.dense = DenseRetrieverAdapter.build(_enc(), self.texts, self.labels, self.ls, _cfg())
        self.lex = LexicalRetrieverAdapter.build(self.texts, self.labels, self.ls, _cfg())

    def test_dense_knn_empty_batch(self):
        lab, sim = self.dense.knn_example_labels(np.zeros((0, DIM), dtype=np.float32), k=5)
        assert lab.shape == (0, 5)
        assert sim.shape == (0, 5)

    def test_dense_prototype_similarity_empty_batch(self):
        sim = self.dense.prototype_similarity(np.zeros((0, DIM), dtype=np.float32))
        assert sim.shape == (0, self.ls.size)

    def test_lexical_knn_empty_batch(self):
        lab, sim = self.lex.knn_example_labels([], k=4)
        assert lab.shape == (0, 4)
        assert sim.shape == (0, 4)

    def test_feature_assembler_empty_batch(self):
        feats = _assembler(self.ls).assemble(
            [], np.zeros((0, DIM), dtype=np.float32), self.dense, self.lex,
            _cfg().k_neighbors, query_ids=[], query_labels=None,
        )
        assert len(feats) == 0
        assert list(feats.columns) == FEATURE_NAMES
