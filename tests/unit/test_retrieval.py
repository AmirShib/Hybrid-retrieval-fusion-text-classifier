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

from scipy import sparse

from text_classifier.config import RetrievalConfig
from text_classifier.domain import LabelSpace, ClassDefinition
from text_classifier.infrastructure.retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
    _dense_topk,
    _prototypes_and_freq,
    _sparse_row_topk,
    bm25_prunes_vocab,
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
        idx.fit(["apple"])  # 1 doc, 1 term, tf=1, dl=1
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
        # N=4 docs, avgdl=1.5 (baked into idf/len_norm below)
        log2 = math.log(2)
        log10_3 = math.log(10 / 3)

        len_norm = [0.75, 1.25, 1.25, 0.75]  # 1 - b + b*(dl/avgdl)

        # BM25 weight for (doc, term):
        def w(idf, tf, ln):
            return idf * tf * (k1 + 1) / (tf + k1 * ln)

        expected = np.array(
            [
                w(log2, 1, len_norm[0]),  # doc0: apple tf=1
                w(log2, 1, len_norm[1]),  # doc1: apple tf=1
                0.0,  # doc2: neither apple nor cherry
                w(log10_3, 1, len_norm[3]),  # doc3: cherry tf=1
            ],
            dtype=np.float32,
        )

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
        idx = BM25Index(k1=1.5, b=0.0)  # b=0 removes length effect
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
        idx.fit(["cat", "cat dog bird fish"])  # doc0: dl=1, doc1: dl=4, both cat tf=1
        sm = idx.score_matrix(["cat"])
        # b=0 → len_norm=1.0 always → identical BM25 weight
        assert abs(float(sm[0, 0]) - float(sm[0, 1])) < 1e-5

    def test_b_one_shorter_doc_scores_higher(self):
        """With b=1 (full normalization), the shorter doc scores higher for same tf."""
        idx = BM25Index(k1=1.5, b=1.0)
        idx.fit(["cat", "cat dog bird fish"])  # doc0: dl=1, doc1: dl=4, both cat tf=1
        sm = idx.score_matrix(["cat"])
        # b=1 → shorter doc (lower len_norm) gets higher weight
        assert float(sm[0, 0]) > float(sm[0, 1])

    def test_oov_query_term_contributes_zero(self):
        """Terms absent from the vocabulary add 0 and cause no error."""
        idx = BM25Index(k1=1.5, b=0.75)
        idx.fit(["apple orange"])
        sm_oov = idx.score_matrix(["banana"])  # banana not in vocab
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


class TestBM25LanguageNeutralDefaults:
    """T35 — no hidden English-stopword assumption; it's opt-in."""

    def test_retrieval_config_default_has_no_stop_words(self):
        cfg = RetrievalConfig()
        assert cfg.bm25_token_kwargs == {}

    def test_default_config_keeps_stopwords_in_vocabulary(self):
        """Wiring RetrievalConfig()'s default kwargs into BM25Index keeps
        common English words -- the old behaviour must be an explicit opt-in."""
        cfg = RetrievalConfig()
        idx = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs)
        idx.fit(["the quick brown fox"])
        vocab = set(idx.vectorizer.vocabulary_.keys())
        assert "the" in vocab

    def test_old_behaviour_recoverable_via_kwarg(self):
        """english stopwords can still be requested explicitly."""
        cfg = RetrievalConfig(bm25_token_kwargs={"stop_words": "english"})
        idx = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs)
        idx.fit(["the quick brown fox"])
        vocab = set(idx.vectorizer.vocabulary_.keys())
        assert "the" not in vocab

    def test_hebrew_unicode_corpus_retrieves_under_default(self):
        """A non-English/Unicode micro-corpus scores correctly with no
        stopword filtering and the default (Unicode-aware) token pattern."""
        cfg = RetrievalConfig()
        idx = BM25Index(cfg.k1, cfg.b, **cfg.bm25_token_kwargs)
        idx.fit(["חלב טרי", "לחם אחיד", "גבינה צהובה"])
        sm = idx.score_matrix(["חלב"])
        # matches the milk document, not the bread or cheese ones
        assert float(sm[0, 0]) > float(sm[0, 1])
        assert float(sm[0, 0]) > float(sm[0, 2])


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
    label_space = LabelSpace(
        [
            ClassDefinition("fruits", "apple orange fruit"),
            ClassDefinition("fish", "salmon trout ocean"),
            ClassDefinition("birds", "eagle falcon sky"),
        ]
    )
    texts = [
        "apple apple fruit",  # label 0
        "orange fruit",  # label 0
        "salmon fish",  # label 1
        "trout ocean fish",  # label 1
        "eagle falcon",  # label 2
    ]
    labels = np.array([0, 0, 1, 1, 2])
    cfg = RetrievalConfig(bm25_token_kwargs={})  # no stop_words for full control
    adapter = LexicalRetrieverAdapter.build(texts, labels, label_space, cfg)
    return adapter, label_space, labels


def test_lexical_description_score_shape(lex_env):
    adapter, label_space, _ = lex_env
    sm = adapter.description_score(["apple fruit", "eagle sky"])
    assert sm.shape == (2, label_space.size)


# --------------------------------------------------------------- persistence (T67)
def test_bm25_state_roundtrip_reproduces_scores():
    texts = ["apple apple fruit", "orange fruit", "salmon fish ocean", "eagle falcon sky"]
    idx = BM25Index(1.5, 0.75).fit(texts)
    queries = ["apple orange", "salmon", "falcon eagle sky", "unseen gibberish"]
    before = idx.score_matrix(queries)

    arrays, meta = idx.to_state()
    restored = BM25Index.from_state(arrays, meta)
    after = restored.score_matrix(queries)

    npt.assert_array_equal(before, after)
    assert restored.k1 == idx.k1 and restored.b == idx.b and restored.n_docs == idx.n_docs


def test_bm25_state_rejects_non_json_clean_cv_kwargs():
    idx = BM25Index(1.5, 0.75, analyzer=lambda t: t.split())
    idx.fit(["a b c", "d e f"])
    with pytest.raises(ValueError):
        idx.to_state()


def test_lexical_adapter_state_roundtrip_reproduces_everything(lex_env):
    adapter, label_space, _ = lex_env
    queries = ["apple orange", "salmon trout", "eagle sky", "gibberish unseen"]

    before_desc = adapter.description_score(queries)
    before_knn = adapter.knn_example_labels(queries, k=2)

    arrays, meta = adapter.to_state()
    restored = LexicalRetrieverAdapter.from_state(arrays, meta)

    npt.assert_array_equal(restored.description_score(queries), before_desc)
    after_labels, after_scores = restored.knn_example_labels(queries, k=2)
    npt.assert_array_equal(after_labels, before_knn[0])
    npt.assert_array_equal(after_scores, before_knn[1])


def test_lexical_adapter_state_roundtrip_via_npz(lex_env, tmp_path):
    """The exact round-trip persistence.py performs: pack to npz, reload from disk."""
    adapter, _, _ = lex_env
    queries = ["apple orange", "salmon trout"]
    before = adapter.description_score(queries)

    arrays, meta = adapter.to_state()
    npz_path = tmp_path / "lexical.npz"
    np.savez_compressed(npz_path, **arrays)
    import json

    json_path = tmp_path / "lexical.json"
    json_path.write_text(json.dumps(meta))

    loaded_arrays = dict(np.load(npz_path))
    loaded_meta = json.loads(json_path.read_text())
    restored = LexicalRetrieverAdapter.from_state(loaded_arrays, loaded_meta)
    npt.assert_array_equal(restored.description_score(queries), before)


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
    label_space = LabelSpace(
        [
            ClassDefinition("alpha", "alpha description one"),
            ClassDefinition("beta", "beta description two"),
            ClassDefinition("empty", "empty class no examples"),
        ]
    )
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
    label_space = LabelSpace(
        [
            ClassDefinition("solo", "solo"),
            ClassDefinition("other", "other"),
        ]
    )
    texts = ["solo item"]
    labels = np.array([0])
    cfg = RetrievalConfig(dense_chunk=256)
    adapter = DenseRetrieverAdapter.build(enc, texts, labels, label_space, cfg)
    emb = enc.encode(["solo item"])
    npt.assert_allclose(adapter.state.prototypes[0], emb[0], atol=1e-6)


class TestBuildFromEmbeddings:
    """T88: `build_from_embeddings` is the seam the training pipeline slices
    cached, whole-pool embeddings into. `build()` delegates to it, so the two
    must produce byte-identical adapters for the same inputs."""

    def test_matches_build_byte_for_byte(self, dense_env):
        adapter, label_space, enc, texts, labels = dense_env
        cfg = RetrievalConfig(dense_chunk=256)
        example_emb = enc.encode_documents(texts)
        desc_emb = enc.encode_documents(label_space.descriptions)
        via_embeddings = DenseRetrieverAdapter.build_from_embeddings(
            example_emb, labels, desc_emb, label_space, cfg
        )
        npt.assert_array_equal(adapter.state.example_emb, via_embeddings.state.example_emb)
        npt.assert_array_equal(adapter.state.example_labels, via_embeddings.state.example_labels)
        npt.assert_array_equal(adapter.state.description_emb, via_embeddings.state.description_emb)
        npt.assert_array_equal(adapter.state.class_freq, via_embeddings.state.class_freq)
        proto_a, proto_b = adapter.state.prototypes, via_embeddings.state.prototypes
        both_nan = np.isnan(proto_a) & np.isnan(proto_b)
        npt.assert_array_equal(np.isnan(proto_a), np.isnan(proto_b))
        npt.assert_allclose(proto_a[~both_nan], proto_b[~both_nan], atol=1e-7)

    def test_sliced_embeddings_match_re_encoding_the_subset(self, dense_env):
        """The exact operation T88 performs: encode the whole pool once, then
        slice for a fold, versus encoding just that fold's subset directly."""
        _, label_space, enc, texts, labels = dense_env
        cfg = RetrievalConfig(dense_chunk=256)
        idx = np.array([0, 2, 3])  # a "fold"

        whole_emb = enc.encode_documents(texts)
        desc_emb = enc.encode_documents(label_space.descriptions)
        sliced = DenseRetrieverAdapter.build_from_embeddings(
            whole_emb[idx], labels[idx], desc_emb, label_space, cfg
        )

        direct = DenseRetrieverAdapter.build(
            enc, [texts[i] for i in idx], labels[idx], label_space, cfg
        )
        npt.assert_array_equal(sliced.state.example_emb, direct.state.example_emb)
        both_nan = np.isnan(sliced.state.prototypes) & np.isnan(direct.state.prototypes)
        npt.assert_array_equal(np.isnan(sliced.state.prototypes), np.isnan(direct.state.prototypes))
        npt.assert_allclose(
            sliced.state.prototypes[~both_nan], direct.state.prototypes[~both_nan], atol=1e-7
        )


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
    assert np.isnan(proto_sim[0, 2])  # empty class column must be NaN


def test_dense_class_freq_counts_correctly(dense_env):
    adapter, _, _, _, labels = dense_env
    freq = adapter.class_freq
    npt.assert_array_equal(freq[0], int((labels == 0).sum()))  # 3
    npt.assert_array_equal(freq[1], int((labels == 1).sum()))  # 2
    npt.assert_array_equal(freq[2], 0)  # empty class


def _prototypes_and_freq_loop_reference(emb, labels, n_classes):
    """The pre-T84 ``for c in range(n_classes)`` masked-mean loop, kept here to
    verify the scatter-based replacement against it directly."""
    dim = emb.shape[1]
    proto = np.full((n_classes, dim), np.nan, dtype=np.float32)
    freq = np.zeros(n_classes, dtype=np.int64)
    labels = np.asarray(labels)
    for c in range(n_classes):
        mask = labels == c
        freq[c] = int(mask.sum())
        if freq[c]:
            v = emb[mask].mean(axis=0)
            norm = np.linalg.norm(v)
            if norm > 0:
                proto[c] = (v / norm).astype(np.float32)
    return proto, freq


class TestPrototypesAndFreqLoopFree:
    """T84: the scatter-based ``_prototypes_and_freq`` against the loop it
    replaced. Not asserted bit-for-bit -- IEEE754 addition is not associative,
    so a flat scatter-sum and a per-group ``.mean(axis=0)`` (numpy's pairwise
    summation) legitimately round differently once a class has more than a
    couple of examples -- but must agree to float32 precision, including the
    NaN/zero-frequency edge cases."""

    def test_matches_loop_reference_including_empty_classes(self):
        rng = np.random.default_rng(0)
        n_classes, dim, n = 12, 6, 150
        emb = rng.standard_normal((n, dim)).astype(np.float32)
        # class 3 gets no examples at all (n_classes-1 possible labels used)
        labels = rng.integers(0, n_classes - 1, n).astype(np.int64)

        proto, freq = _prototypes_and_freq(emb, labels, n_classes)
        want_proto, want_freq = _prototypes_and_freq_loop_reference(emb, labels, n_classes)

        npt.assert_array_equal(freq, want_freq)
        assert freq[n_classes - 1] == 0
        assert np.all(np.isnan(proto[n_classes - 1]))
        valid = freq > 0
        npt.assert_allclose(proto[valid], want_proto[valid], rtol=1e-5, atol=1e-6)

    def test_empty_pool_all_nan(self):
        emb = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
        proto, freq = _prototypes_and_freq(emb, labels, 3)
        npt.assert_array_equal(freq, np.zeros(3, dtype=np.int64))
        assert np.all(np.isnan(proto))

    def test_single_example_class_matches_normalized_vector(self):
        emb = np.array([[3.0, 4.0]], dtype=np.float32)  # norm 5
        labels = np.array([0], dtype=np.int64)
        proto, freq = _prototypes_and_freq(emb, labels, 1)
        assert freq[0] == 1
        npt.assert_allclose(proto[0], [0.6, 0.8], rtol=1e-6)


# =========================================================================== #
#  Part D — leave-one-out self-masking (n_folds=1)
# =========================================================================== #


class TestSelfMaskExcludeHelper:
    """`_exclude_self` drops the per-row self index, keeps real neighbours
    best-first, and returns exactly width k."""

    def test_drops_self_and_promotes_next(self):
        from text_classifier.infrastructure.retrieval import _exclude_self

        # Row 0's self is index 10 (its top neighbour); row 1 has no self (-1).
        idx = np.array([[10, 11, 12], [20, 21, 22]])
        score = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)
        out_idx, out_score = _exclude_self(idx, score, np.array([10, -1]), k=2)
        npt.assert_array_equal(out_idx[0], [11, 12])  # self (10) removed, rest promoted
        npt.assert_allclose(out_score[0], [0.8, 0.7])
        npt.assert_array_equal(out_idx[1], [20, 21])  # no self -> ordinary top-2
        assert out_idx.shape == (2, 2)

    def test_small_corpus_pads_after_removing_self(self):
        from text_classifier.infrastructure.retrieval import _exclude_self

        # Two neighbours, one is self -> one real neighbour, padded to width 3.
        idx = np.array([[5, 6]])
        score = np.array([[0.9, 0.8]], dtype=np.float32)
        out_idx, out_score = _exclude_self(idx, score, np.array([5]), k=3)
        npt.assert_array_equal(out_idx[0], [6, -1, -1])
        assert np.isnan(out_score[0, 1]) and np.isnan(out_score[0, 2])


def test_dense_knn_excludes_self(dense_env):
    """Querying the pool's own items with exclude_idx never returns the self index;
    without it, the top neighbour is the item itself (perfect self-match)."""
    from text_classifier.infrastructure.retrieval import _exclude_self

    adapter, _, enc, texts, _ = dense_env
    q = enc.encode_queries(texts)
    self_ids = np.arange(len(texts))

    # Ordinary retrieval: each item is its own nearest neighbour (perfect match).
    idx_plain, sim_plain = _dense_topk(q, adapter.state.example_emb, 3, 256)
    npt.assert_array_equal(idx_plain[:, 0], self_ids)
    npt.assert_allclose(sim_plain[:, 0], 1.0, atol=1e-5)

    # Leave-one-out: the self index is gone from every row's neighbour list. Fetch
    # k+1 and apply the same mask the adapter uses, then assert self is absent.
    ex_idx, ex_sim = _dense_topk(q, adapter.state.example_emb, 4, 256)
    ex_idx, _ = _exclude_self(ex_idx, ex_sim, self_ids, 3)
    for i in range(len(texts)):
        assert i not in set(ex_idx[i].tolist())


def test_lexical_knn_excludes_self():
    """BM25 kNN with exclude never self-retrieves the query's own document."""
    label_space = LabelSpace(
        [ClassDefinition("a", "fruit pastry"), ClassDefinition("b", "fruit loaf")]
    )
    texts = ["apple pie", "apple tart", "banana bread", "banana cake"]
    labels = np.array([0, 0, 1, 1])
    cfg = RetrievalConfig()
    adapter = LexicalRetrieverAdapter.build(texts, labels, label_space, cfg)
    self_ids = np.arange(len(texts))
    idx, score = adapter._examples.top_k(texts, 3, exclude=self_ids)
    for i in range(len(texts)):
        assert i not in set(idx[i].tolist())


def test_loo_prototype_leaves_self_out():
    """LOO prototype for a 2-item class equals cosine to the *other* item; a
    1-item class yields NaN (no prototype once its only example is removed)."""
    enc = HashingEncoder(dim=64)
    label_space = LabelSpace(
        [ClassDefinition("pair", "pair"), ClassDefinition("solo", "solo")]
    )
    texts = ["pair one", "pair two", "solo only"]
    labels = np.array([0, 0, 1])
    cfg = RetrievalConfig(dense_chunk=256)
    adapter = DenseRetrieverAdapter.build(enc, texts, labels, label_space, cfg)
    emb = enc.encode_documents(texts)
    q = enc.encode_queries(texts)
    self_ids = np.arange(len(texts))

    loo = adapter.loo_prototype_similarity(q, self_ids)
    # Item 0's own-class (0) LOO prototype is just item 1's embedding (normalized).
    expected01 = float(q[0] @ (emb[1] / np.linalg.norm(emb[1])))
    npt.assert_allclose(loo[0, 0], expected01, atol=1e-5)
    # Item 2 is the only member of class 1: its LOO own-class prototype is NaN.
    assert np.isnan(loo[2, 1])
    # A column that is not the query's own class is unchanged from the plain value.
    plain = adapter.prototype_similarity(q)
    npt.assert_allclose(loo[0, 1], plain[0, 1], atol=1e-6)


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
    n_examples = len(texts)  # 5
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


# =========================================================================== #
#  T32 — BM25 at scale: bounded memory and throughput
# =========================================================================== #


class TestSparseRowTopk:
    """`_sparse_row_topk` is the replacement for BM25 `top_k`'s dense
    argpartition/argsort — it must never densify, but must still return the
    same shape, padding, and descending-order contract."""

    def test_basic_topk_matches_hand_computation(self):
        # row0: cols {0: 3, 2: 1, 3: 5}; row1: cols {1: 2}
        S = sparse.csr_matrix(
            ([3.0, 1.0, 5.0, 2.0], ([0, 0, 0, 1], [0, 2, 3, 1])), shape=(2, 4)
        )
        idx, score = _sparse_row_topk(S, fetch=2)
        assert idx.shape == (2, 2) and score.shape == (2, 2)
        npt.assert_array_equal(idx[0], [3, 0])  # descending: 5 (col3), 3 (col0)
        npt.assert_allclose(score[0], [5.0, 3.0])
        assert idx[1, 0] == 1
        npt.assert_allclose(score[1, 0], 2.0)
        # row1 has only one nonzero: the second slot is padding.
        assert idx[1, 1] == -1
        assert np.isnan(score[1, 1])

    def test_empty_matrix_is_all_padding(self):
        S = sparse.csr_matrix((3, 5), dtype=np.float32)
        idx, score = _sparse_row_topk(S, fetch=2)
        assert np.all(idx == -1)
        assert np.all(np.isnan(score))

    def test_fetch_zero_returns_empty_width(self):
        S = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))
        idx, score = _sparse_row_topk(S, fetch=0)
        assert idx.shape == (1, 0) and score.shape == (1, 0)

    def test_row_with_more_nonzeros_than_fetch_keeps_the_largest(self):
        S = sparse.csr_matrix(([1.0, 5.0, 3.0, 2.0], ([0, 0, 0, 0], [0, 1, 2, 3])), shape=(1, 4))
        idx, score = _sparse_row_topk(S, fetch=2)
        npt.assert_array_equal(idx[0], [1, 2])  # cols with values 5, 3
        npt.assert_allclose(score[0], [5.0, 3.0])

    def test_matches_dense_argpartition_reference(self):
        """Cross-check against a brute-force dense computation on a random
        sparse matrix — the property that actually matters (same top-k values,
        same descending order), not incidental tie-breaking."""
        rng = np.random.default_rng(0)
        dense = rng.random((6, 9)).astype(np.float32)
        dense[dense < 0.5] = 0.0  # make it genuinely sparse
        S = sparse.csr_matrix(dense)
        fetch = 3
        idx, score = _sparse_row_topk(S, fetch)
        for r in range(dense.shape[0]):
            expected_vals = np.sort(dense[r][dense[r] > 0])[::-1][:fetch]
            got_vals = score[r][~np.isnan(score[r])]
            npt.assert_allclose(np.sort(got_vals)[::-1], expected_vals, atol=1e-6)


class TestBM25TokenizeOnceFitFromCounts:
    """T32 A2: `fit()` == `tokenize_corpus()` + `fit_from_counts()`, and a row
    slice of the shared counts matrix reproduces fitting a fresh index on just
    that subset."""

    CORPUS = [
        "apple apple fruit",
        "orange fruit",
        "salmon fish ocean",
        "trout ocean fish",
        "eagle falcon sky",
    ]

    def test_fit_from_counts_matches_fit(self):
        direct = BM25Index(1.5, 0.75).fit(self.CORPUS)
        counts, vectorizer = BM25Index.tokenize_corpus(self.CORPUS)
        via_counts = BM25Index(1.5, 0.75).fit_from_counts(counts, vectorizer)

        queries = ["apple orange", "salmon trout", "unseen gibberish"]
        npt.assert_allclose(direct.score_matrix(queries), via_counts.score_matrix(queries))

    def test_sliced_counts_matches_fitting_the_subset_directly(self):
        """The exact operation A2 performs: tokenize the whole corpus once,
        then slice rows for a fold, versus tokenizing just that fold's texts."""
        idx = [0, 2, 4]
        counts, vectorizer = BM25Index.tokenize_corpus(self.CORPUS)
        sliced = BM25Index(1.5, 0.75).fit_from_counts(counts[idx], vectorizer)

        subset_texts = [self.CORPUS[i] for i in idx]
        direct = BM25Index(1.5, 0.75).fit(subset_texts)

        queries = ["apple fruit", "salmon eagle"]
        npt.assert_allclose(sliced.score_matrix(queries), direct.score_matrix(queries), atol=1e-6)

    def test_bm25_prunes_vocab_detects_corpus_pruning_kwargs(self):
        assert bm25_prunes_vocab({}) is False
        assert bm25_prunes_vocab({"stop_words": "english"}) is False
        assert bm25_prunes_vocab({"min_df": 2}) is True
        assert bm25_prunes_vocab({"max_df": 0.9}) is True
        assert bm25_prunes_vocab({"max_features": 100}) is True


class TestBM25MaxDfRatio:
    """T32 A4: opt-in, lossy high-df pruning."""

    CORPUS = ["common apple", "common banana", "common cherry", "rare apple"]

    def test_none_is_byte_identical_to_no_pruning(self):
        a = BM25Index(1.5, 0.75).fit(self.CORPUS)
        b = BM25Index(1.5, 0.75, max_df_ratio=None).fit(self.CORPUS)
        npt.assert_array_equal(a.score_matrix(["common"]), b.score_matrix(["common"]))

    def test_pruning_a_high_df_term_zeroes_its_contribution(self):
        # "common" appears in 3/4 docs (df ratio 0.75); pruning at 0.5 drops it.
        pruned = BM25Index(1.5, 0.75, max_df_ratio=0.5).fit(self.CORPUS)
        sm = pruned.score_matrix(["common"])
        npt.assert_allclose(sm, np.zeros_like(sm), atol=1e-8)

    def test_pruning_shrinks_the_weight_matrix_nnz(self):
        full = BM25Index(1.5, 0.75).fit(self.CORPUS)
        pruned = BM25Index(1.5, 0.75, max_df_ratio=0.5).fit(self.CORPUS)
        assert pruned._Wt.nnz < full._Wt.nnz

    def test_persisted_and_reapplied_at_load(self):
        idx = BM25Index(1.5, 0.75, max_df_ratio=0.5).fit(self.CORPUS)
        arrays, meta = idx.to_state()
        assert meta["max_df_ratio"] == 0.5
        restored = BM25Index.from_state(arrays, meta)
        assert restored.max_df_ratio == 0.5
        npt.assert_array_equal(idx.score_matrix(["common"]), restored.score_matrix(["common"]))

    def test_legacy_state_without_the_key_loads_as_off(self):
        """A directory saved before T32 has no `max_df_ratio` key in meta."""
        idx = BM25Index(1.5, 0.75).fit(self.CORPUS)
        arrays, meta = idx.to_state()
        del meta["max_df_ratio"]
        restored = BM25Index.from_state(arrays, meta)
        assert restored.max_df_ratio is None


class TestScoreMatrixBlockGuard:
    """T32 B: score_matrix must refuse to densify past a configured cap."""

    def test_none_is_unbounded(self):
        idx = BM25Index(1.5, 0.75).fit(["ab cd", "ef gh"])
        idx.score_matrix(["ab"], max_block_elems=None)  # must not raise

    def test_under_cap_succeeds(self):
        idx = BM25Index(1.5, 0.75).fit(["ab cd", "ef gh"])
        idx.score_matrix(["ab"], max_block_elems=100)  # 1 * 2 = 2 elements

    def test_over_cap_raises(self):
        idx = BM25Index(1.5, 0.75).fit(["ab cd", "ef gh"])
        with pytest.raises(ValueError, match="score_matrix"):
            idx.score_matrix(["ab", "cd", "ef"], max_block_elems=1)  # 3 * 2 = 6 > 1


class TestLexicalBuildFromCounts:
    """T32 A1/A2 at the adapter level: `build_from_counts` (pre-tokenized
    example corpus + a pre-built description index) matches `build()`."""

    def test_matches_build_byte_for_byte(self, lex_env):
        adapter, label_space, labels = lex_env
        texts = [
            "apple apple fruit",
            "orange fruit",
            "salmon fish",
            "trout ocean fish",
            "eagle falcon",
        ]
        cfg = RetrievalConfig(bm25_token_kwargs={})
        counts, vectorizer = BM25Index.tokenize_corpus(texts)
        desc_bm25 = BM25Index(cfg.k1, cfg.b).fit(label_space.descriptions)
        via_counts = LexicalRetrieverAdapter.build_from_counts(
            counts, vectorizer, labels, desc_bm25, cfg
        )

        queries = ["apple orange", "salmon trout", "eagle sky"]
        npt.assert_array_equal(
            adapter.description_score(queries), via_counts.description_score(queries)
        )
        a_labels, a_scores = adapter.knn_example_labels(queries, k=2)
        b_labels, b_scores = via_counts.knn_example_labels(queries, k=2)
        npt.assert_array_equal(a_labels, b_labels)
        npt.assert_array_equal(a_scores, b_scores)

    def test_sliced_counts_matches_building_the_subset_directly(self, lex_env):
        _, label_space, _ = lex_env
        texts = [
            "apple apple fruit",
            "orange fruit",
            "salmon fish",
            "trout ocean fish",
            "eagle falcon",
        ]
        labels = np.array([0, 0, 1, 1, 2])
        cfg = RetrievalConfig(bm25_token_kwargs={})
        idx = np.array([0, 1, 3])

        counts, vectorizer = BM25Index.tokenize_corpus(texts)
        desc_bm25 = BM25Index(cfg.k1, cfg.b).fit(label_space.descriptions)
        sliced = LexicalRetrieverAdapter.build_from_counts(
            counts[idx], vectorizer, labels[idx], desc_bm25, cfg
        )
        direct = LexicalRetrieverAdapter.build(
            [texts[i] for i in idx], labels[idx], label_space, cfg
        )

        queries = ["apple orange", "trout fish"]
        npt.assert_allclose(
            sliced.description_score(queries), direct.description_score(queries), atol=1e-6
        )
