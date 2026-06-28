"""T04 — Retrieval tests (BM25Index, LexicalRetrieverAdapter, DenseRetrieverAdapter).

Part A: BM25Index — math against Lucene formula, query-freq binarization,
        IDF ordering, length normalization, OOV terms, stop_words plumbing,
        top_k padding/ordering/chunking contracts.
Part B: LexicalRetrieverAdapter — build, label mapping, -1 padding guard.
Part C: DenseRetrieverAdapter — prototypes, empty-class NaN, class_freq,
        kNN ordering, similarity ranges, chunking equivalence.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.config import RetrievalConfig
from text_classifier.domain import LabelSpace, ClassDefinition
from text_classifier.infrastructure.retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
    _dense_topk,
)
from tests._doubles import HashingEncoder


# =========================================================================== #
#  Part A — BM25Index
# =========================================================================== #

class TestBM25MathCorrectness:
    def test_single_doc_single_term_exact_value(self):
        """Simplest possible case: N=1, df=1, avgdl=dl → len_norm=1.
        Formula collapses to: idf * tf*(k1+1)/(tf+k1) = log(4/3) * 2.5/2.5 = log(4/3).
        """
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(["apple"])                    # 1 doc, 1 term, tf=1, dl=1
        # idf = log(1 + (1−1+0.5)/(1+0.5)) = log(1+1/3) = log(4/3)
        # len_norm = 1 (dl == avgdl, b cancels)
        # W = log(4/3) * 1 * 2.5 / (1 + 1.5) = log(4/3)
        expected = math.log(4 / 3)
        sm = idx.score_matrix(["apple"])
        assert sm.shape == (1, 1)
        assert abs(float(sm[0, 0]) - expected) < 1e-4

    def test_four_doc_corpus_lucene_formula(self):
        """4-doc corpus; hand-computed scores via the Lucene BM25 formula.

        corpus:
            doc0: "apple"              dl=1
            doc1: "apple banana"       dl=2
            doc2: "banana banana"      dl=2
            doc3: "cherry"             dl=1

        N=4, avgdl=1.5, k1=1.5, b=0.75
        vocab (alphabetical): apple=0, banana=1, cherry=2
        df: apple=2, banana=2, cherry=1
        idf: apple=banana=log(2), cherry=log(10/3)
        len_norm: doc0/doc3=0.75, doc1/doc2=1.25

        query "apple cherry" → binary terms {apple, cherry}
        """
        k1, b = 1.5, 0.75
        N, avgdl = 4, 1.5
        log2 = math.log(2)
        log10_3 = math.log(10 / 3)

        len_norm = [0.75, 1.25, 1.25, 0.75]   # 1 - b + b*(dl/avgdl)

        # BM25 weight for (doc, term):
        def w(idf, tf, ln): return idf * tf * (k1 + 1) / (tf + k1 * ln)

        expected = np.array([
            w(log2, 1, len_norm[0]),   # doc0: apple tf=1
            w(log2, 1, len_norm[1]),   # doc1: apple tf=1
            0.0,                       # doc2: neither apple nor cherry
            w(log10_3, 1, len_norm[3]),# doc3: cherry tf=1
        ], dtype=np.float32)

        idx = BM25Index(k1=k1, b=b)
        idx.fit(["apple", "apple banana", "banana banana", "cherry"])
        sm = idx.score_matrix(["apple cherry"])
        npt.assert_allclose(sm[0], expected, atol=1e-4)

    def test_query_frequency_is_ignored(self):
        """'foo foo bar' and 'foo bar' must produce identical scores (binary incidence)."""
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(["foo bar", "foo baz"])
        s1 = idx.score_matrix(["foo bar"])
        s2 = idx.score_matrix(["foo foo bar"])
        npt.assert_allclose(s1, s2, atol=1e-6)

    def test_rare_term_has_higher_idf_than_common_term(self):
        """A term in every doc should contribute less per unit tf than a rare term."""
        # "common" appears in all 3 docs; "rare" only in doc0.
        idx = BM25Index(k1=1.5, b=0.0)   # b=0 removes length effect
        idx.fit(["rare common", "common", "common"])
        # For doc0, query "rare common":
        # rare contribution > common contribution because idf_rare >> idf_common
        sm = idx.score_matrix(["rare"])
        sm_common = idx.score_matrix(["common"])
        # doc0 score on "rare" should exceed doc0 score on "common" (same tf, higher idf)
        assert float(sm[0, 0]) > float(sm_common[0, 0])

    def test_b_zero_length_invariant(self):
        """With b=0, two docs with the same term (tf=1) but different lengths score equally."""
        idx = BM25Index(k1=1.5, b=0.0)
        idx.fit(["cat", "cat dog bird fish"])   # doc0: dl=1, doc1: dl=4, both cat tf=1
        sm = idx.score_matrix(["cat"])
        # b=0 → len_norm=1.0 always → identical BM25 weight
        assert abs(float(sm[0, 0]) - float(sm[0, 1])) < 1e-5

    def test_b_one_shorter_doc_scores_higher(self):
        """With b=1 (full normalization), the shorter doc scores higher for same tf."""
        idx = BM25Index(k1=1.5, b=1.0)
        idx.fit(["cat", "cat dog bird fish"])   # doc0: dl=1, doc1: dl=4, both cat tf=1
        sm = idx.score_matrix(["cat"])
        # b=1 → shorter doc (lower len_norm) gets higher weight
        assert float(sm[0, 0]) > float(sm[0, 1])

    def test_oov_query_term_contributes_zero(self):
        """Terms absent from the vocabulary add 0 and cause no error."""
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(["apple orange"])
        sm_oov = idx.score_matrix(["banana"])         # banana not in vocab
        sm_normal = idx.score_matrix(["apple"])
        npt.assert_allclose(sm_oov, np.zeros_like(sm_oov), atol=1e-8)
        assert float(sm_normal[0, 0]) > 0

    def test_stop_words_plumbing(self):
        """stop_words='english' removes common English words from the vocabulary."""
        idx = BM25Index(k1=1.5, b=0.75, stop_words="english")
        idx.fit(["the quick brown fox"])
        vocab = set(idx.vectorizer.vocabulary_.keys())
        # 'the' is an English stop word and must not appear
        assert "the" not in vocab
        # at least one content word should survive
        assert len(vocab) > 0


class TestBM25TopK:
    @pytest.fixture
    def idx4(self):
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(["apple", "apple banana", "banana banana", "cherry"])
        return idx

    def test_topk_shape_and_sorted_descending(self, idx4):
        out_idx, out_score = idx4.top_k(["apple"], k=3)
        assert out_idx.shape == (1, 3)
        assert out_score.shape == (1, 3)
        # Positive scores should be sorted in descending order (NaN pads go to end)
        scores = out_score[0]
        pos = scores[~np.isnan(scores)]
        assert np.all(pos[:-1] >= pos[1:])

    def test_topk_positive_only_zero_scores_become_padding(self, idx4):
        """Docs with zero BM25 score become -1 / NaN padding."""
        # "cherry" only matches doc3; docs without overlap get score 0
        out_idx, out_score = idx4.top_k(["cherry"], k=4)
        # doc0/1/2 have 0 overlap with "cherry" → must be padded
        mask = out_idx[0] == -1
        assert np.all(np.isnan(out_score[0, mask]))
        # exactly 1 non-padded slot (doc3)
        assert int((out_idx[0] >= 0).sum()) == 1

    def test_topk_k_larger_than_ndocs_pads_remainder(self, idx4):
        """k > n_docs: real results fill min(k,n_docs) slots; the rest are padded."""
        out_idx, out_score = idx4.top_k(["apple"], k=10)
        assert out_idx.shape == (1, 10)
        n_real = int((out_idx[0] >= 0).sum())
        # At most n_docs=4 real entries
        assert n_real <= 4

    def test_topk_no_overlap_all_padding(self, idx4):
        """A query with no vocabulary overlap produces an all-padded row."""
        out_idx, out_score = idx4.top_k(["xyzzy"], k=3)
        npt.assert_array_equal(out_idx[0], [-1, -1, -1])
        assert np.all(np.isnan(out_score[0]))

    def test_topk_chunking_equivalence(self, idx4):
        """chunk=1 and chunk=1000 return the same scores."""
        queries = ["apple", "banana cherry", "xyzzy"]
        idx_a, sc_a = idx4.top_k(queries, k=3, chunk=1)
        idx_b, sc_b = idx4.top_k(queries, k=3, chunk=1000)
        # Scores must match (idx may differ on ties)
        npt.assert_allclose(
            np.where(np.isnan(sc_a), 0, sc_a),
            np.where(np.isnan(sc_b), 0, sc_b),
            atol=1e-5,
        )


# =========================================================================== #
#  Part B — LexicalRetrieverAdapter
# =========================================================================== #

@pytest.fixture
def lex_env():
    """Small 3-class environment for lexical adapter tests."""
    label_space = LabelSpace([
        ClassDefinition("fruits", "apple orange fruit"),
        ClassDefinition("fish", "salmon trout ocean"),
        ClassDefinition("birds", "eagle falcon sky"),
    ])
    texts = [
        "apple apple fruit",   # label 0
        "orange fruit",        # label 0
        "salmon fish",         # label 1
        "trout ocean fish",    # label 1
        "eagle falcon",        # label 2
    ]
    labels = np.array([0, 0, 1, 1, 2])
    cfg = RetrievalConfig(bm25_token_kwargs={})   # no stop_words for full control
    adapter = LexicalRetrieverAdapter.build(texts, labels, label_space, cfg)
    return adapter, label_space, labels


def test_lexical_description_score_shape(lex_env):
    adapter, label_space, _ = lex_env
    sm = adapter.description_score(["apple fruit", "eagle sky"])
    assert sm.shape == (2, label_space.size)


def test_lexical_description_score_values_plausible(lex_env):
    adapter, label_space, _ = lex_env
    sm = adapter.description_score(["apple fruit"])
    # "apple fruit" should score highest on class 0 ("apple orange fruit")
    assert int(np.argmax(sm[0])) == 0


def test_lexical_knn_label_mapping(lex_env):
    adapter, label_space, _ = lex_env
    labels_out, scores = adapter.knn_example_labels(["apple apple fruit"], k=3)
    # All returned (non-padded) labels must be class 0 or valid class indices
    valid = labels_out[0][labels_out[0] >= 0]
    assert np.all(valid < label_space.size)
    # The top hit for "apple apple fruit" should map to class 0
    assert labels_out[0, 0] == 0


def test_lexical_knn_padding_preserved(lex_env):
    """Padded doc slots (idx=-1) must stay -1 — np.clip must not corrupt them."""
    adapter, _, _ = lex_env
    labels_out, scores = adapter.knn_example_labels(["xyzzy"], k=5)
    # "xyzzy" has no overlap → all padding
    npt.assert_array_equal(labels_out[0], [-1, -1, -1, -1, -1])
    assert np.all(np.isnan(scores[0]))


# =========================================================================== #
#  Part C — DenseRetrieverAdapter
# =========================================================================== #

@pytest.fixture
def dense_env():
    """3-class env with a deliberate empty class (index 2 has no examples)."""
    label_space = LabelSpace([
        ClassDefinition("alpha", "alpha description one"),
        ClassDefinition("beta",  "beta description two"),
        ClassDefinition("empty", "empty class no examples"),
    ])
    texts = ["alpha one", "alpha two", "alpha three", "beta one", "beta two"]
    labels = np.array([0, 0, 0, 1, 1])
    enc = HashingEncoder(dim=64)
    cfg = RetrievalConfig(dense_chunk=256)
    adapter = DenseRetrieverAdapter.build(enc, texts, labels, label_space, cfg)
    return adapter, label_space, enc, texts, labels


def test_dense_prototype_is_l2_normalized(dense_env):
    adapter, label_space, *_ = dense_env
    proto = adapter.state.prototypes
    # Classes 0 and 1 have examples; check their prototype norms
    for c in [0, 1]:
        row = proto[c]
        if not np.any(np.isnan(row)):
            npt.assert_allclose(np.linalg.norm(row), 1.0, atol=1e-5)


def test_dense_prototype_direction_single_example(dense_env):
    """A class with one example: prototype must equal that example's embedding."""
    enc = HashingEncoder(dim=64)
    label_space = LabelSpace([
        ClassDefinition("solo", "solo"),
        ClassDefinition("other", "other"),
    ])
    texts = ["solo item"]
    labels = np.array([0])
    cfg = RetrievalConfig(dense_chunk=256)
    adapter = DenseRetrieverAdapter.build(enc, texts, labels, label_space, cfg)
    emb = enc.encode(["solo item"])
    npt.assert_allclose(adapter.state.prototypes[0], emb[0], atol=1e-6)


def test_dense_empty_class_has_nan_prototype(dense_env):
    adapter, *_ = dense_env
    proto = adapter.state.prototypes
    # Class index 2 ("empty") has no training examples
    assert np.all(np.isnan(proto[2]))


def test_dense_empty_class_freq_is_zero(dense_env):
    adapter, *_ = dense_env
    assert adapter.class_freq[2] == 0


def test_dense_empty_class_gives_nan_similarity(dense_env):
    adapter, _, enc, texts, _ = dense_env
    q_emb = enc.encode([texts[0]])
    proto_sim = adapter.prototype_similarity(q_emb)
    assert proto_sim.shape == (1, 3)
    assert np.isnan(proto_sim[0, 2])   # empty class column must be NaN


def test_dense_class_freq_counts_correctly(dense_env):
    adapter, _, _, _, labels = dense_env
    freq = adapter.class_freq
    npt.assert_array_equal(freq[0], int((labels == 0).sum()))   # 3
    npt.assert_array_equal(freq[1], int((labels == 1).sum()))   # 2
    npt.assert_array_equal(freq[2], 0)                          # empty class


def test_dense_knn_sorted_by_descending_similarity(dense_env):
    adapter, _, enc, texts, _ = dense_env
    q_emb = enc.encode([texts[0]])
    _, sims = adapter.knn_example_labels(q_emb, k=3)
    # Similarities must be non-increasing
    assert np.all(sims[0, :-1] >= sims[0, 1:])


def test_dense_knn_k_larger_than_n_examples_pads_remainder(dense_env):
    adapter, _, enc, texts, _ = dense_env
    q_emb = enc.encode([texts[0]])
    lab, sim = adapter.knn_example_labels(q_emb, k=1000)
    n_examples = len(texts)   # 5
    # shape is padded to the requested k (mirrors BM25Index.top_k)...
    assert lab.shape == (1, 1000)
    assert sim.shape == (1, 1000)
    # ...with exactly n_examples real neighbours and the rest -1 / NaN padding.
    assert int((lab[0] >= 0).sum()) == n_examples
    assert int(np.isfinite(sim[0]).sum()) == n_examples
    npt.assert_array_equal(lab[0, n_examples:], -1)
    assert np.all(np.isnan(sim[0, n_examples:]))


def test_dense_similarity_values_in_range(dense_env):
    adapter, _, enc, texts, _ = dense_env
    q_emb = enc.encode(texts)
    desc_sim = adapter.description_similarity(q_emb)
    proto_sim = adapter.prototype_similarity(q_emb)
    # L2-normalized embeddings → dot product ∈ [-1, 1]; allow tiny float slack
    finite_d = desc_sim[np.isfinite(desc_sim)]
    finite_p = proto_sim[np.isfinite(proto_sim)]
    assert np.all(finite_d >= -1.001) and np.all(finite_d <= 1.001)
    assert np.all(finite_p >= -1.001) and np.all(finite_p <= 1.001)


def test_dense_description_similarity_shape(dense_env):
    adapter, label_space, enc, texts, _ = dense_env
    q_emb = enc.encode(texts[:3])
    desc_sim = adapter.description_similarity(q_emb)
    assert desc_sim.shape == (3, label_space.size)


def test_dense_topk_chunking_equivalence(dense_env):
    """_dense_topk with chunk=1 and chunk=1000 return identical results."""
    adapter, _, enc, texts, _ = dense_env
    q_emb = enc.encode(texts)
    X = adapter.state.example_emb
    idx_a, sim_a = _dense_topk(q_emb, X, k=3, chunk=1)
    idx_b, sim_b = _dense_topk(q_emb, X, k=3, chunk=1000)
    npt.assert_allclose(sim_a, sim_b, atol=1e-5)
    # Indices may differ on ties, but their retrieved similarities match
    # (sorting: same scores if same top-k)
