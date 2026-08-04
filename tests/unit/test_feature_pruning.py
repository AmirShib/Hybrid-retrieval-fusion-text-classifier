"""T87 — feature dependency graph + demand-driven computation.

The safety property this ticket exists to protect: pruning is value-preserving.
For any requested column subset, every surviving column's value is identical to
the value the unpruned (``requested=None``) computation produces. That is
fuzzed directly against the real assembler here, rather than trusted by
inspection of ``FEATURE_DEPS``.
"""

from __future__ import annotations

import random

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.application.features import FeatureAssembler
from text_classifier.config import RetrievalConfig
from text_classifier.domain import (
    FEATURE_DEPS,
    FEATURE_NAMES,
    CandidatePolicy,
    FeatureContext,
    FeatureProvider,
    composed_feature_names,
    feature_closure,
    fusion_feature_names,
)
from text_classifier.infrastructure.retrieval import DenseRetrieverAdapter, LexicalRetrieverAdapter


# =========================================================================== #
#  Schema completeness: every column declares its inputs, every input exists.
# =========================================================================== #


class TestFeatureDepsCompleteness:
    def test_every_core_column_has_a_deps_entry(self):
        missing = [n for n in FEATURE_NAMES if n not in FEATURE_DEPS]
        assert missing == [], f"FEATURE_NAMES entries with no FEATURE_DEPS: {missing}"

    def test_every_referenced_node_is_itself_declared(self):
        referenced = {dep for deps in FEATURE_DEPS.values() for dep in deps}
        dangling = referenced - set(FEATURE_DEPS)
        assert dangling == set(), f"nodes referenced but never declared: {dangling}"

    def test_candidates_is_never_pruned(self):
        assert "candidates" in feature_closure([])
        assert "candidates" in feature_closure(["d_desc_sim"])

    def test_closure_of_full_schema_is_a_superset_of_the_schema(self):
        assert set(FEATURE_NAMES) <= feature_closure(FEATURE_NAMES)


# =========================================================================== #
#  Pruning parity, fuzzed against the real assembler.
# =========================================================================== #


@pytest.fixture
def fenv(hashing_encoder):
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=6, per_class=10, seed=11)
    texts = [it.text for it in items]
    label_idx = np.array(label_space.encode_labels([it.label for it in items]))
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build(hashing_encoder, texts, label_idx, label_space, cfg)
    lexical = LexicalRetrieverAdapter.build(texts, label_idx, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3))
    q_items = items[:10]
    q_texts = [it.text for it in q_items]
    q_emb = hashing_encoder.encode(q_texts)
    return dict(assembler=assembler, dense=dense, lexical=lexical, q_texts=q_texts, q_emb=q_emb)


def _assemble(fenv, requested=None):
    e = fenv
    return e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        requested=requested,
    )


class TestPruningParity:
    def test_full_schema_request_is_byte_identical_to_the_default(self, fenv):
        default = _assemble(fenv, requested=None)
        full = _assemble(fenv, requested=list(FEATURE_NAMES))
        default = default.sort_values(["item_id", "candidate"]).reset_index(drop=True)
        full = full.sort_values(["item_id", "candidate"]).reset_index(drop=True)
        assert list(default.columns) == list(full.columns)
        for col in FEATURE_NAMES:
            npt.assert_array_equal(default[col].values, full[col].values, err_msg=col)

    def test_candidate_mask_is_identical_under_every_pruning_subset(self, fenv):
        """The (item_id, candidate) pairs that survive selection must not move,
        no matter what's requested — pruning narrows columns, never rows."""
        baseline = _assemble(fenv, requested=None)
        key = lambda df: set(zip(df["item_id"].tolist(), df["candidate"].tolist()))
        base_keys = key(baseline)
        rng = random.Random(0)
        for _ in range(8):
            subset = rng.sample(FEATURE_NAMES, k=rng.randint(1, len(FEATURE_NAMES)))
            pruned = _assemble(fenv, requested=subset)
            assert key(pruned) == base_keys, subset

    @pytest.mark.parametrize("seed", range(12))
    def test_random_subset_values_match_the_full_computation(self, fenv, seed):
        baseline = _assemble(fenv, requested=None).sort_values(["item_id", "candidate"])
        baseline = baseline.reset_index(drop=True)
        rng = random.Random(seed)
        subset = rng.sample(FEATURE_NAMES, k=rng.randint(1, len(FEATURE_NAMES)))
        pruned = _assemble(fenv, requested=subset)
        pruned = pruned.sort_values(["item_id", "candidate"]).reset_index(drop=True)

        npt.assert_array_equal(baseline["item_id"].values, pruned["item_id"].values)
        npt.assert_array_equal(baseline["candidate"].values, pruned["candidate"].values)
        for col in subset:
            assert col in pruned.columns, f"requested column {col!r} missing from pruned output"
            npt.assert_allclose(
                baseline[col].values.astype(np.float64),
                pruned[col].values.astype(np.float64),
                equal_nan=True,
                atol=1e-6,
                err_msg=f"pruning changed the value of requested column {col!r}",
            )

    def test_unrequested_columns_are_absent_not_nan(self, fenv):
        pruned = _assemble(fenv, requested=["margin_d_desc"])
        assert "margin_d_desc" in pruned.columns
        assert "rank_d_desc" not in pruned.columns
        assert "q_gap_d_knn" not in pruned.columns

    def test_shared_margin_computation_still_respects_individual_demand(self, fenv):
        """margin_d_desc and q_gap_d_desc share one _row_margin call; requesting
        only one must not leak the other into the output."""
        only_margin = _assemble(fenv, requested=["margin_d_desc"])
        assert "margin_d_desc" in only_margin.columns
        assert "q_gap_d_desc" not in only_margin.columns

        only_gap = _assemble(fenv, requested=["q_gap_d_desc"])
        assert "q_gap_d_desc" in only_gap.columns
        assert "margin_d_desc" not in only_gap.columns

    def test_empty_batch_respects_requested_columns(self, fenv):
        e = fenv
        df = e["assembler"].assemble(
            [],
            np.empty((0, e["q_emb"].shape[1]), dtype=np.float32),
            e["dense"],
            e["lexical"],
            k_neighbors=3,
            query_ids=[],
            requested=["margin_d_desc"],
        )
        assert len(df) == 0
        assert "margin_d_desc" in df.columns
        assert "rank_d_desc" not in df.columns


# =========================================================================== #
#  Provider gating: a fully-dropped provider's compute() is never called.
# =========================================================================== #


class _SpyProvider(FeatureProvider):
    def __init__(self):
        self.calls = 0

    def names(self):
        return ["spy_a", "spy_b"]

    def compute(self, ctx: FeatureContext):
        self.calls += 1
        n = ctx.rows.shape[0]
        return {"spy_a": np.ones(n), "spy_b": np.zeros(n)}

    def save(self, path: str) -> None:  # pragma: no cover - not exercised here
        pass

    @classmethod
    def load(cls, path: str) -> "FeatureProvider":  # pragma: no cover
        raise NotImplementedError


class TestProviderDemandGating:
    def test_compute_never_called_when_every_provider_column_is_dropped(self, fenv):
        spy = _SpyProvider()
        df = _assemble_with_providers(fenv, [spy], requested=list(FEATURE_NAMES))
        assert spy.calls == 0
        assert "spy_a" not in df.columns and "spy_b" not in df.columns

    def test_compute_runs_when_one_column_survives(self, fenv):
        spy = _SpyProvider()
        df = _assemble_with_providers(fenv, [spy], requested=["spy_a"])
        assert spy.calls == 1
        assert "spy_a" in df.columns
        assert "spy_b" not in df.columns

    def test_full_request_none_runs_every_provider(self, fenv):
        spy = _SpyProvider()
        df = _assemble_with_providers(fenv, [spy], requested=None)
        assert spy.calls == 1
        assert {"spy_a", "spy_b"} <= set(df.columns)


def _assemble_with_providers(fenv, providers, requested):
    e = fenv
    return e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        providers=providers,
        requested=requested,
    )


# =========================================================================== #
#  fusion_feature_names / composed_feature_names as realistic `requested` values.
# =========================================================================== #


class TestRealisticRequestedValues:
    def test_drop_features_narrows_the_assembled_frame(self, fenv):
        drop = ["margin_d_desc", "rank_b_knn"]
        requested = fusion_feature_names(drop=drop)
        df = _assemble(fenv, requested=requested)
        for name in drop:
            assert name not in df.columns

    def test_composed_feature_names_reproduces_the_full_schema(self, fenv):
        requested = composed_feature_names()
        df = _assemble(fenv, requested=requested)
        assert set(FEATURE_NAMES) <= set(df.columns)


# =========================================================================== #
#  Acceptance criterion: drop_features measurably reduces assembly work.
# =========================================================================== #


class TestPruningReducesWork:
    """Counts calls into the expensive helpers instead of timing wall-clock —
    deterministic, not flaky under CI load, and a direct measurement of "did
    fewer sorts/partitions run", which is the actual claim."""

    def test_dropping_every_rank_and_margin_column_skips_their_computation(self, fenv):
        import text_classifier.application.features as features_mod

        counts = {"rank": 0, "margin": 0, "minmax": 0}
        real_rank, real_margin, real_minmax = (
            features_mod._row_rank,
            features_mod._row_margin,
            features_mod._row_minmax,
        )

        def counting_rank(*a, **k):
            counts["rank"] += 1
            return real_rank(*a, **k)

        def counting_margin(*a, **k):
            counts["margin"] += 1
            return real_margin(*a, **k)

        def counting_minmax(*a, **k):
            counts["minmax"] += 1
            return real_minmax(*a, **k)

        features_mod._row_rank = counting_rank
        features_mod._row_margin = counting_margin
        features_mod._row_minmax = counting_minmax
        try:
            _assemble(fenv, requested=None)
            full_counts = dict(counts)

            counts["rank"] = counts["margin"] = counts["minmax"] = 0
            # Request only columns with no rank/margin/norm dependency.
            skip_prefixes = ("rank_", "margin_", "norm_", "q_gap_")
            narrow = [n for n in FEATURE_NAMES if not n.startswith(skip_prefixes)]
            narrow = [n for n in narrow if n != "n_signal_agreement"]
            _assemble(fenv, requested=narrow)
            narrow_counts = dict(counts)
        finally:
            features_mod._row_rank = real_rank
            features_mod._row_margin = real_margin
            features_mod._row_minmax = real_minmax

        assert full_counts["rank"] == 4  # rank_d_desc, rank_b_desc, rank_d_knn, rank_b_knn
        assert full_counts["margin"] == 5
        assert full_counts["minmax"] == 2
        assert narrow_counts == {"rank": 0, "margin": 0, "minmax": 0}
