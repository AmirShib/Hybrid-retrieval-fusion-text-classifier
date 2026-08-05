"""T03 — Feature-assembly tests.

Part A: pure numpy helpers (_scatter_knn, _topn_mask, _row_rank, _row_minmax,
        _argmax_or_missing) — tiny hand-built arrays, exact assertions.
Part B: FeatureAssembler.assemble — schema/order, dtypes, NaN semantics,
        is_true, derived-feature identities, chunking equivalence, and the
        empty-input / empty-candidate paths.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.application.features import (
    FeatureAssembler,
    _argmax_or_missing,
    _row_margin,
    _row_minmax,
    _row_rank,
    _scatter_knn,
    _topn_mask,
)
from text_classifier.config import RetrievalConfig
from text_classifier.domain import (
    CandidatePolicy,
    DenseRetriever,
    FEATURE_NAMES,
    LexicalRetriever,
)
from text_classifier.infrastructure.retrieval import (
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
)


# =========================================================================== #
#  Stubs: all-NaN / all-zero signals → no candidate survives the union mask.
# =========================================================================== #


class _NaNDense(DenseRetriever):
    def __init__(self, n_classes: int) -> None:
        self._C = n_classes

    @property
    def class_freq(self) -> np.ndarray:
        return np.zeros(self._C, dtype=np.int64)

    def knn_example_labels(self, query_emb: np.ndarray, k: int):
        b = query_emb.shape[0]
        return np.full((b, k), -1, np.int64), np.full((b, k), np.nan, np.float32)

    def prototype_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return np.full((query_emb.shape[0], self._C), np.nan, np.float32)

    def description_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        return np.full((query_emb.shape[0], self._C), np.nan, np.float32)


class _ZeroLexical(LexicalRetriever):
    def __init__(self, n_classes: int) -> None:
        self._C = n_classes

    def knn_example_labels(self, query_texts, k: int):
        b = len(query_texts)
        return np.full((b, k), -1, np.int64), np.full((b, k), np.nan, np.float32)

    def description_score(self, query_texts) -> np.ndarray:
        # 0 scores → converted to NaN by the assembler (0 overlap == missing)
        return np.zeros((len(query_texts), self._C), np.float32)


# =========================================================================== #
#  Part A — pure helper tests
# =========================================================================== #


class TestScatterKnn:
    def test_basic_aggregation(self):
        # Row 0: class 0 twice (positions 0,2), class 1 once.
        # Row 1: class 2 once, class 1 once, entry 2 is -1 (invalid).
        labels = np.array([[0, 1, 0], [2, 1, -1]])
        scores = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.1]])
        ksum, kmax, kcnt = _scatter_knn(labels, scores, n_classes=3)

        npt.assert_allclose(ksum[0, 0], 1.6)  # 0.9 + 0.7
        npt.assert_allclose(kmax[0, 0], 0.9)
        assert kcnt[0, 0] == 2
        npt.assert_allclose(ksum[0, 1], 0.8)
        assert kcnt[0, 1] == 1
        assert np.isnan(ksum[0, 2]) and np.isnan(kmax[0, 2])
        assert kcnt[0, 2] == 0

        npt.assert_allclose(ksum[1, 2], 0.6)
        npt.assert_allclose(ksum[1, 1], 0.5)
        assert np.isnan(ksum[1, 0])  # class 0 never retrieved in row 1

    def test_nan_score_is_ignored(self):
        labels = np.array([[0, 1]])
        scores = np.array([[0.9, np.nan]])
        ksum, kmax, kcnt = _scatter_knn(labels, scores, n_classes=2)
        npt.assert_allclose(ksum[0, 0], 0.9)
        assert kcnt[0, 0] == 1
        # class 1: NaN score → ignored → treated as not retrieved
        assert np.isnan(ksum[0, 1])
        assert np.isnan(kmax[0, 1])
        assert kcnt[0, 1] == 0

    def test_negative_label_is_ignored(self):
        labels = np.array([[-1, 0]])
        scores = np.array([[0.5, 0.8]])
        ksum, _, kcnt = _scatter_knn(labels, scores, n_classes=2)
        assert kcnt[0, 0] == 1
        assert np.isnan(ksum[0, 1])  # class 1 never validly retrieved

    def test_empty_class_uses_nan_not_zero(self):
        """sum/max NaN for count==0 — the 'not retrieved' encoding."""
        labels = np.array([[0, 0]])
        scores = np.array([[0.4, 0.6]])
        ksum, kmax, kcnt = _scatter_knn(labels, scores, n_classes=3)
        # class 2 never appears
        assert np.isnan(ksum[0, 2])
        assert np.isnan(kmax[0, 2])
        assert kcnt[0, 2] == 0  # count is a true 0, not NaN


class TestTopnMask:
    def test_selects_top_n_columns(self):
        M = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]])
        mask = _topn_mask(M, n=2)
        npt.assert_array_equal(mask[0], [True, False, True])  # 0.9, 0.5
        npt.assert_array_equal(mask[1], [False, True, True])  # 0.8, 0.3

    def test_n_larger_than_c_selects_all_finite(self):
        M = np.array([[0.9, 0.5, 0.1]])
        mask = _topn_mask(M, n=10)
        npt.assert_array_equal(mask[0], [True, True, True])

    def test_all_nan_row_selects_nothing(self):
        M = np.array([[np.nan, np.nan, np.nan]])
        mask = _topn_mask(M, n=2)
        npt.assert_array_equal(mask[0], [False, False, False])

    def test_nan_never_beats_finite(self):
        M = np.array([[np.nan, 0.5, 0.3]])
        mask = _topn_mask(M, n=1)
        # NaN ranks last; top-1 is 0.5 (col 1)
        npt.assert_array_equal(mask[0], [False, True, False])

    def test_positive_only_drops_nonpositive_columns(self):
        M = np.array([[0.9, -0.1, 0.5, 0.0]])
        mask = _topn_mask(M, n=4, positive_only=True)
        # only cols 0 and 2 are strictly positive
        npt.assert_array_equal(mask[0], [True, False, True, False])

    def test_ties_include_all_tied_at_threshold(self):
        # n=2; cutoff lands on 0.5 (tied between cols 1 and 2)
        M = np.array([[0.9, 0.5, 0.5]])
        mask = _topn_mask(M, n=2)
        # documented: ties admit more than n → both tied columns included
        assert mask[0, 1] and mask[0, 2]
        assert int(mask.sum()) >= 2


class TestRowRank:
    def test_descending_rank_within_candidates(self):
        M = np.array([[0.9, 0.5, 0.1]])
        cand = np.array([[True, True, True]])
        ranks = _row_rank(M, cand)
        npt.assert_array_equal(ranks[0], [1, 2, 3])

    def test_non_candidate_pushed_to_worst_rank(self):
        M = np.array([[0.9, 0.5, 0.1]])
        cand = np.array([[True, True, False]])
        ranks = _row_rank(M, cand)
        assert ranks[0, 0] == 1
        assert ranks[0, 1] == 2
        assert ranks[0, 2] == 3  # non-candidate → worst

    def test_nan_value_pushed_to_worst_rank(self):
        M = np.array([[0.9, np.nan, 0.1]])
        cand = np.array([[True, True, True]])
        ranks = _row_rank(M, cand)
        assert ranks[0, 0] == 1
        assert ranks[0, 2] == 2
        assert ranks[0, 1] == 3  # NaN → worst


class TestRowMinmax:
    def test_min_maps_to_zero_max_maps_to_one(self):
        M = np.array([[0.4, 0.8, 0.6]])
        cand = np.array([[True, True, True]])
        result = _row_minmax(M, cand)
        npt.assert_allclose(result[0], [0.0, 1.0, 0.5], atol=1e-6)

    def test_degenerate_all_equal_produces_finite_result(self):
        M = np.array([[0.5, 0.5, 0.5]])
        cand = np.array([[True, True, True]])
        result = _row_minmax(M, cand)
        assert np.all(np.isfinite(result)), "degenerate row should not produce NaN/inf"

    def test_no_candidates_returns_nan(self):
        M = np.array([[0.4, 0.8]])
        cand = np.array([[False, False]])
        result = _row_minmax(M, cand)
        assert np.all(np.isnan(result))

    def test_all_nan_candidates_returns_nan(self):
        M = np.array([[np.nan, np.nan]])
        cand = np.array([[True, True]])
        result = _row_minmax(M, cand)
        assert np.all(np.isnan(result))

    def test_runtimewarning_is_suppressed(self):
        """The internal RuntimeWarning for all-NaN rows must not escape."""
        M = np.array([[np.nan, np.nan]])
        cand = np.array([[True, True]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → error
            result = _row_minmax(M, cand)  # must not raise
        assert np.all(np.isnan(result))


class TestRowMargin:
    def test_leader_gets_top1_top2_gap_others_get_deficit(self):
        M = np.array([[0.9, 0.5, 0.1]])
        cand = np.array([[True, True, True]])
        margin, gap = _row_margin(M, cand)
        npt.assert_allclose(margin[0], [0.4, -0.4, -0.8], atol=1e-9)
        npt.assert_allclose(gap, [0.4], atol=1e-9)

    def test_leader_margin_equals_gap(self):
        M = np.array([[0.2, 0.7, 0.65], [0.9, 0.1, 0.3]])
        cand = np.full((2, 3), True)
        margin, gap = _row_margin(M, cand)
        npt.assert_allclose(margin.max(axis=1), gap, atol=1e-9)

    def test_only_the_leader_is_positive(self):
        M = np.array([[0.2, 0.7, 0.65]])
        cand = np.full((1, 3), True)
        margin, _ = _row_margin(M, cand)
        assert int((margin[0] > 0).sum()) == 1

    def test_nan_value_stays_nan(self):
        """NaN means 'signal did not retrieve this class' — never a low score."""
        M = np.array([[0.9, np.nan, 0.1]])
        cand = np.array([[True, True, True]])
        margin, gap = _row_margin(M, cand)
        assert np.isnan(margin[0, 1])
        # NaN does not compete: the gap is 0.9 - 0.1, not 0.9 - NaN.
        npt.assert_allclose(gap, [0.8], atol=1e-9)

    def test_non_candidate_does_not_compete(self):
        M = np.array([[0.9, 0.5, 0.1]])
        cand = np.array([[True, False, True]])
        margin, gap = _row_margin(M, cand)
        # 0.5 is masked out of the competition, so the runner-up is 0.1.
        npt.assert_allclose(margin[0, 0], 0.8, atol=1e-9)
        npt.assert_allclose(gap, [0.8], atol=1e-9)

    def test_single_scored_candidate_has_no_defined_margin(self):
        """One competitor-free candidate: undefined margin (NaN), not 0.0 (a tie)."""
        M = np.array([[0.9, np.nan, np.nan]])
        cand = np.array([[True, True, True]])
        margin, gap = _row_margin(M, cand)
        assert np.isnan(margin[0, 0])
        assert np.isnan(gap[0])

    def test_all_missing_row_is_all_nan(self):
        M = np.array([[np.nan, np.nan]])
        cand = np.array([[True, True]])
        margin, gap = _row_margin(M, cand)
        assert np.all(np.isnan(margin))
        assert np.isnan(gap[0])

    def test_no_candidates_row_is_all_nan(self):
        M = np.array([[0.4, 0.8]])
        cand = np.array([[False, False]])
        margin, gap = _row_margin(M, cand)
        assert np.all(np.isnan(margin))
        assert np.isnan(gap[0])

    def test_tie_at_the_top_gives_zero_margin_to_both(self):
        M = np.array([[0.7, 0.7, 0.1]])
        cand = np.full((1, 3), True)
        margin, gap = _row_margin(M, cand)
        npt.assert_allclose(margin[0, 0], 0.0, atol=1e-9)
        npt.assert_allclose(margin[0, 1], 0.0, atol=1e-9)
        npt.assert_allclose(gap, [0.0], atol=1e-9)

    def test_single_class_label_space(self):
        """C == 1: the lone class never has a competitor."""
        M = np.array([[0.9]])
        cand = np.array([[True]])
        margin, gap = _row_margin(M, cand)
        assert np.isnan(margin[0, 0])
        assert np.isnan(gap[0])

    def test_runtimewarning_is_suppressed(self):
        """The internal -inf arithmetic on all-missing rows must not escape."""
        M = np.array([[np.nan, np.nan]])
        cand = np.array([[True, True]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → error
            margin, gap = _row_margin(M, cand)  # must not raise
        assert np.all(np.isnan(margin)) and np.isnan(gap[0])

    def test_rows_are_independent(self):
        M = np.array([[0.9, 0.5], [0.1, 0.4]])
        cand = np.full((2, 2), True)
        margin, gap = _row_margin(M, cand)
        npt.assert_allclose(margin[0], [0.4, -0.4], atol=1e-9)
        npt.assert_allclose(margin[1], [-0.3, 0.3], atol=1e-9)
        npt.assert_allclose(gap, [0.4, 0.3], atol=1e-9)

    def test_margin_is_invariant_to_a_row_wide_shift(self):
        """The point of a margin: it removes the query-level offset that makes raw
        cosines incomparable across items."""
        M = np.array([[0.9, 0.5, 0.1]])
        cand = np.full((1, 3), True)
        base, base_gap = _row_margin(M, cand)
        shifted, shifted_gap = _row_margin(M + 0.05, cand)
        npt.assert_allclose(base, shifted, atol=1e-9)
        npt.assert_allclose(base_gap, shifted_gap, atol=1e-9)


class TestArgmaxOrMissing:
    def test_returns_argmax_for_normal_row(self):
        M = np.array([[0.1, 0.9, 0.5]])
        npt.assert_array_equal(_argmax_or_missing(M), [1])

    def test_all_nan_returns_minus_one(self):
        M = np.array([[np.nan, np.nan]])
        npt.assert_array_equal(_argmax_or_missing(M), [-1])

    def test_require_positive_zero_best_returns_minus_one(self):
        M = np.array([[0.0, -0.5]])
        npt.assert_array_equal(_argmax_or_missing(M, require_positive=True), [-1])

    def test_require_positive_positive_best_returned(self):
        M = np.array([[-0.5, 0.3]])
        npt.assert_array_equal(_argmax_or_missing(M, require_positive=True), [1])

    def test_require_positive_false_allows_nonpositive(self):
        M = np.array([[-0.1, -0.9]])
        npt.assert_array_equal(_argmax_or_missing(M, require_positive=False), [0])


# =========================================================================== #
#  Part B — FeatureAssembler integration tests
# =========================================================================== #


@pytest.fixture
def fenv(hashing_encoder):
    """5-class × 10-item environment; first 8 items used as queries."""
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=5, per_class=10, seed=7)
    texts = [it.text for it in items]
    label_idx = np.array(label_space.encode_labels([it.label for it in items]))
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build(hashing_encoder, texts, label_idx, label_space, cfg)
    lexical = LexicalRetrieverAdapter.build(texts, label_idx, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3))
    q_items = items[:8]
    q_texts = [it.text for it in q_items]
    q_emb = hashing_encoder.encode(q_texts)
    q_labels = np.array(label_space.encode_labels([it.label for it in q_items]))
    return dict(
        assembler=assembler,
        dense=dense,
        lexical=lexical,
        q_texts=q_texts,
        q_emb=q_emb,
        q_labels=q_labels,
        label_space=label_space,
    )


def _assemble(fenv, with_labels=False, chunk=4096):
    e = fenv
    return e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        query_labels=e["q_labels"] if with_labels else None,
        chunk=chunk,
    )


class TestAssembleSchema:
    def test_feature_columns_match_feature_names_in_order(self, fenv):
        df = _assemble(fenv)
        feature_cols = [c for c in df.columns if c not in ("item_id", "candidate", "is_true")]
        assert feature_cols == FEATURE_NAMES

    def test_no_extra_columns(self, fenv):
        df = _assemble(fenv)
        assert set(df.columns) - set(FEATURE_NAMES) - {"item_id", "candidate"} == set()

    def test_feature_dtype_is_float32(self, fenv):
        df = _assemble(fenv)
        bad = [c for c in FEATURE_NAMES if df[c].dtype != np.float32]
        assert bad == [], f"wrong dtype for columns: {bad}"

    def test_candidate_values_are_valid_class_indices(self, fenv):
        df = _assemble(fenv)
        C = fenv["label_space"].size
        assert df["candidate"].between(0, C - 1).all()

    def test_is_true_absent_without_labels(self, fenv):
        assert "is_true" not in _assemble(fenv, with_labels=False).columns

    def test_is_true_present_with_labels(self, fenv):
        assert "is_true" in _assemble(fenv, with_labels=True).columns

    def test_is_true_correct_values(self, fenv):
        df = _assemble(fenv, with_labels=True)
        q_labels = fenv["q_labels"]
        expected = (df["candidate"].values == q_labels[df["item_id"].values]).astype(np.int64)
        npt.assert_array_equal(df["is_true"].values, expected)


class TestAssembleDerivedFeatures:
    def test_desc_proto_gap_identity(self, fenv):
        df = _assemble(fenv)
        expected = df["d_desc_sim"].values.astype(np.float64) - df["d_proto_sim"].values.astype(
            np.float64
        )
        npt.assert_allclose(df["desc_proto_gap"].values.astype(np.float64), expected, atol=1e-5)

    def test_class_log_freq_identity(self, fenv):
        df = _assemble(fenv)
        freq = fenv["dense"].class_freq
        expected = np.log1p(freq[df["candidate"].values.astype(int)].astype(np.float64)).astype(
            np.float32
        )
        npt.assert_allclose(df["class_log_freq"].values, expected, atol=1e-5)

    def test_b_desc_missing_iff_b_desc_sim_nan(self, fenv):
        df = _assemble(fenv)
        expected = np.isnan(df["b_desc_sim"].values).astype(np.float32)
        npt.assert_array_equal(df["b_desc_missing"].values, expected)

    def test_b_knn_missing_iff_b_knn_sum_nan(self, fenv):
        df = _assemble(fenv)
        expected = np.isnan(df["b_knn_sum"].values).astype(np.float32)
        npt.assert_array_equal(df["b_knn_missing"].values, expected)

    def test_d_knn_missing_iff_d_knn_sum_nan(self, fenv):
        df = _assemble(fenv)
        expected = np.isnan(df["d_knn_sum"].values).astype(np.float32)
        npt.assert_array_equal(df["d_knn_missing"].values, expected)

    def test_n_signal_agreement_in_range(self, fenv):
        df = _assemble(fenv)
        vals = df["n_signal_agreement"].values
        assert np.all(vals >= 0) and np.all(vals <= 4)


class TestAssembleCompetitionFeatures:
    """The margin / top1-top2 columns, checked against the assembled frame."""

    MARGINS = {
        "margin_d_desc": "d_desc_sim",
        "margin_d_proto": "d_proto_sim",
        "margin_d_knn": "d_knn_sum",
        "margin_b_desc": "b_desc_sim",
        "margin_b_knn": "b_knn_sum",
    }
    GAPS = {
        "q_gap_d_desc": "margin_d_desc",
        "q_gap_d_knn": "margin_d_knn",
        "q_gap_b_desc": "margin_b_desc",
    }

    @pytest.mark.parametrize("margin_col,score_col", sorted(MARGINS.items()))
    def test_margin_is_nan_exactly_where_its_signal_is_nan_or_uncontested(
        self, fenv, margin_col, score_col
    ):
        """NaN-as-missing: a margin exists only where the signal scored the
        candidate *and* the candidate had a rival."""
        df = _assemble(fenv)
        signal_nan = np.isnan(df[score_col].values)
        assert np.all(np.isnan(df.loc[signal_nan, margin_col].values))

    @pytest.mark.parametrize("margin_col,score_col", sorted(MARGINS.items()))
    def test_at_most_one_positive_margin_per_item(self, fenv, margin_col, score_col):
        """Only a signal's own leader can be ahead of the field."""
        df = _assemble(fenv)
        n_positive = df.assign(pos=df[margin_col].values > 0).groupby("item_id")["pos"].sum()
        assert n_positive.max() <= 1

    @pytest.mark.parametrize("gap_col,margin_col", sorted(GAPS.items()))
    def test_gap_is_constant_within_an_item(self, fenv, gap_col, margin_col):
        """A per-query column: every candidate row of an item carries the same value."""
        df = _assemble(fenv)
        spread = df.groupby("item_id")[gap_col].nunique(dropna=False)
        assert spread.max() == 1

    @pytest.mark.parametrize("gap_col,margin_col", sorted(GAPS.items()))
    def test_gap_equals_the_leader_margin(self, fenv, gap_col, margin_col):
        """top1 - top2 is exactly what the winning candidate's margin measures."""
        df = _assemble(fenv)
        per_item = df.groupby("item_id")
        best_margin = per_item[margin_col].max()
        gap = per_item[gap_col].first()
        both = ~(np.isnan(best_margin.values) | np.isnan(gap.values))
        npt.assert_allclose(best_margin.values[both], gap.values[both], atol=1e-5)

    def test_gap_is_never_negative(self, fenv):
        df = _assemble(fenv)
        for gap_col in self.GAPS:
            vals = df[gap_col].values
            assert np.all(vals[~np.isnan(vals)] >= -1e-6), gap_col

    def test_margin_separates_a_contested_lead_from_a_decided_one(self, fenv):
        """The regression these features exist to prevent: two candidates with the
        same raw similarity but different competition must not look identical."""
        e = fenv
        C = e["label_space"].size
        assembler = e["assembler"]

        class _Fixed(DenseRetriever):
            """Item 0: leader 0.85 over a 0.84 rival. Item 1: 0.85 over 0.40."""

            def __init__(self, desc):
                self._desc = desc

            @property
            def class_freq(self):
                return np.ones(C, dtype=np.int64)

            def knn_example_labels(self, query_emb, k):
                b = query_emb.shape[0]
                return np.full((b, k), -1, np.int64), np.full((b, k), np.nan, np.float32)

            def prototype_similarity(self, query_emb):
                return np.full((query_emb.shape[0], C), np.nan, np.float32)

            def description_similarity(self, query_emb):
                return self._desc

        desc = np.full((2, C), 0.0, dtype=np.float32)
        desc[0, 0], desc[0, 1] = 0.85, 0.84  # contested
        desc[1, 0], desc[1, 1] = 0.85, 0.40  # decided
        df = assembler.assemble(
            e["q_texts"][:2],
            e["q_emb"][:2],
            _Fixed(desc),
            _ZeroLexical(C),
            k_neighbors=3,
            query_ids=[0, 1],
        )
        lead = df[df["candidate"] == 0].set_index("item_id")
        # Identical raw similarity …
        npt.assert_allclose(lead.loc[0, "d_desc_sim"], lead.loc[1, "d_desc_sim"], atol=1e-6)
        # … but the margin and the query gap tell them apart.
        npt.assert_allclose(lead.loc[0, "margin_d_desc"], 0.01, atol=1e-5)
        npt.assert_allclose(lead.loc[1, "margin_d_desc"], 0.45, atol=1e-5)
        npt.assert_allclose(lead.loc[0, "q_gap_d_desc"], 0.01, atol=1e-5)
        npt.assert_allclose(lead.loc[1, "q_gap_d_desc"], 0.45, atol=1e-5)


class TestAssembleChunking:
    def test_chunk_size_does_not_affect_output(self, fenv):
        big = (
            _assemble(fenv, chunk=10_000)
            .sort_values(["item_id", "candidate"])
            .reset_index(drop=True)
        )
        small = (
            _assemble(fenv, chunk=2).sort_values(["item_id", "candidate"]).reset_index(drop=True)
        )
        for col in FEATURE_NAMES:
            npt.assert_allclose(
                big[col].values.astype(np.float64),
                small[col].values.astype(np.float64),
                atol=1e-6,
                err_msg=f"chunking changed values for column {col!r}",
            )
        npt.assert_array_equal(big["candidate"].values, small["candidate"].values)
        npt.assert_array_equal(big["item_id"].values, small["item_id"].values)


class TestAssembleEdgeCases:
    def test_empty_query_list_returns_valid_empty_frame(self, fenv):
        e = fenv
        df = e["assembler"].assemble(
            [],
            np.empty((0, 128), dtype=np.float32),
            e["dense"],
            e["lexical"],
            k_neighbors=3,
            query_ids=[],
        )
        assert len(df) == 0
        assert set(FEATURE_NAMES).issubset(df.columns)

    def test_empty_candidate_path_no_crash(self, fenv):
        """All-NaN signals → no candidate survives; empty frame with right columns."""
        e = fenv
        C = e["label_space"].size
        df = e["assembler"].assemble(
            e["q_texts"][:2],
            e["q_emb"][:2],
            _NaNDense(C),
            _ZeroLexical(C),
            k_neighbors=3,
            query_ids=[0, 1],
        )
        assert len(df) == 0
        assert set(FEATURE_NAMES).issubset(df.columns)
