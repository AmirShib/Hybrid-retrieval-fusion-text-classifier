"""T70 — custom fusion-feature providers: the seam + the sample provider.

Covers the ``FeatureProvider`` contract in isolation (no pipeline):
  * ``composed_feature_names`` — ordering + collision detection.
  * ``ClassKeywordOverlapProvider`` — names/compute shape, NaN-as-missing, the
    save/load round-trip (portable, no labels), and that the assembler appends
    provider columns after the core ~28.

Pipeline-level parity + leakage live in tests/integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier.application.features import FeatureAssembler, composed_feature_names
from text_classifier.config import RetrievalConfig
from text_classifier.domain import (
    CandidatePolicy,
    FEATURE_NAMES,
    FeatureContext,
    LabeledItem,
    LabelSpace,
)
from text_classifier.infrastructure import ClassKeywordOverlapProvider, build_feature_providers
from text_classifier.config import FeatureProviderConfig, FeaturesConfig
from text_classifier.infrastructure.retrieval import (
    DenseRetrieverAdapter,
    LexicalRetrieverAdapter,
)
from tests._doubles import make_synthetic


# --------------------------------------------------------------------------- #
# composed_feature_names
# --------------------------------------------------------------------------- #
class TestComposedFeatureNames:
    def test_no_providers_is_exactly_core(self):
        assert composed_feature_names() == FEATURE_NAMES
        assert composed_feature_names([]) == FEATURE_NAMES

    def test_appends_provider_names_in_order(self):
        p1 = ClassKeywordOverlapProvider(column="alpha")
        p2 = ClassKeywordOverlapProvider(column="beta")
        assert composed_feature_names([p1, p2]) == FEATURE_NAMES + ["alpha", "beta"]

    def test_collision_with_core_raises(self):
        clash = ClassKeywordOverlapProvider(column=FEATURE_NAMES[0])
        with pytest.raises(ValueError, match="collision"):
            composed_feature_names([clash])

    def test_collision_between_providers_raises(self):
        a = ClassKeywordOverlapProvider(column="dup")
        b = ClassKeywordOverlapProvider(column="dup")
        with pytest.raises(ValueError, match="collision"):
            composed_feature_names([a, b])


# --------------------------------------------------------------------------- #
# ClassKeywordOverlapProvider — fit / compute
# --------------------------------------------------------------------------- #
@pytest.fixture
def fitted_provider():
    ls = LabelSpace.from_pairs(
        [("sport", "games and athletics"), ("food", "cooking and cuisine")]
    )
    items = [
        LabeledItem("football tennis soccer", "sport"),
        LabeledItem("running marathon race", "sport"),
        LabeledItem("pasta pizza risotto", "food"),
        LabeledItem("bread cheese wine", "food"),
    ]
    provider = ClassKeywordOverlapProvider().fit(items, ls)
    return provider, ls


def _ctx(texts, ls, rows, cols):
    q_emb = np.zeros((len(texts), 4), dtype=np.float32)  # unused by this provider
    return FeatureContext(
        query_texts=texts,
        query_emb=q_emb,
        rows=np.asarray(rows),
        cols=np.asarray(cols),
        label_space=ls,
    )


class TestProviderContract:
    def test_names_is_stable_before_and_after_fit(self):
        ls = LabelSpace.from_pairs([("a", "aaa"), ("b", "bbb")])
        p = ClassKeywordOverlapProvider()
        before = p.names()
        p.fit([LabeledItem("aaa word", "a")], ls)
        assert p.names() == before == ["class_kw_overlap"]

    def test_custom_column_name(self):
        assert ClassKeywordOverlapProvider(column="kw").names() == ["kw"]

    def test_compute_returns_one_value_per_candidate(self, fitted_provider):
        provider, ls = fitted_provider
        ctx = _ctx(["football pizza"], ls, rows=[0, 0], cols=[0, 1])
        out = provider.compute(ctx)
        assert set(out) == {"class_kw_overlap"}
        assert out["class_kw_overlap"].shape == (2,)

    def test_overlap_fraction_is_correct(self, fitted_provider):
        provider, ls = fitted_provider
        # "football risotto": 2 in-vocab tokens; football∈sport lexicon,
        # risotto∈food lexicon. sport overlap = 1/2, food overlap = 1/2.
        ctx = _ctx(["football risotto"], ls, rows=[0, 0], cols=[0, 1])
        vals = provider.compute(ctx)["class_kw_overlap"]
        np.testing.assert_allclose(vals, [0.5, 0.5])

    def test_missing_is_nan_not_zero_for_no_overlap(self, fitted_provider):
        provider, ls = fitted_provider
        # A token in the sport lexicon but not food: food overlap is a true 0/1,
        # i.e. 0.0 — NOT NaN (the class fired, the fraction is genuinely zero).
        ctx = _ctx(["football tennis"], ls, rows=[0], cols=[1])
        vals = provider.compute(ctx)["class_kw_overlap"]
        np.testing.assert_allclose(vals, [0.0])

    def test_empty_query_tokens_yield_nan(self, fitted_provider):
        provider, ls = fitted_provider
        # No in-vocabulary tokens -> no denominator -> NaN (did not fire).
        ctx = _ctx(["zzz_unknown_token"], ls, rows=[0, 0], cols=[0, 1])
        vals = provider.compute(ctx)["class_kw_overlap"]
        assert np.isnan(vals).all()

    def test_class_without_lexicon_is_nan(self):
        # 'empty' class has no training examples -> no lexicon -> NaN.
        ls = LabelSpace.from_pairs([("a", "alpha"), ("empty", "nothing here")])
        provider = ClassKeywordOverlapProvider().fit([LabeledItem("alpha beta", "a")], ls)
        ctx = _ctx(["alpha beta"], ls, rows=[0, 0], cols=[0, 1])
        vals = provider.compute(ctx)["class_kw_overlap"]
        assert not np.isnan(vals[0])  # class 'a' fired
        assert np.isnan(vals[1])  # class 'empty' did not

    def test_out_of_range_class_index_is_nan(self, fitted_provider):
        """Robust to a label space widened after fit (added classes, T78)."""
        provider, ls = fitted_provider
        ctx = _ctx(["football"], ls, rows=[0], cols=[99])  # class 99 doesn't exist
        vals = provider.compute(ctx)["class_kw_overlap"]
        assert np.isnan(vals).all()


class TestProviderPersistence:
    def test_save_load_round_trip_reproduces_compute(self, fitted_provider, tmp_path):
        provider, ls = fitted_provider
        d = str(tmp_path / "prov")
        provider.save(d)
        loaded = ClassKeywordOverlapProvider.load(d)

        ctx = _ctx(["football pizza wine"], ls, rows=[0, 0], cols=[0, 1])
        before = provider.compute(ctx)["class_kw_overlap"]
        after = loaded.compute(ctx)["class_kw_overlap"]
        np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
        np.testing.assert_allclose(before[~np.isnan(before)], after[~np.isnan(after)])

    def test_load_rejects_wrong_type(self, tmp_path):
        import pickle
        import os

        os.makedirs(str(tmp_path / "bad"))
        with open(str(tmp_path / "bad" / "provider.pkl"), "wb") as fh:
            pickle.dump({"not": "a provider"}, fh)
        with pytest.raises(TypeError):
            ClassKeywordOverlapProvider.load(str(tmp_path / "bad"))


class TestRegistryFactory:
    def test_build_feature_providers_from_config(self):
        cfg = FeaturesConfig(
            providers=[FeatureProviderConfig(kind="class-keyword", params={"column": "kw"})]
        )
        built = build_feature_providers(cfg)
        assert len(built) == 1
        assert isinstance(built[0], ClassKeywordOverlapProvider)
        assert built[0].names() == ["kw"]

    def test_empty_config_builds_nothing(self):
        assert build_feature_providers(FeaturesConfig()) == []

    def test_unknown_kind_raises(self):
        cfg = FeaturesConfig(providers=[FeatureProviderConfig(kind="does-not-exist")])
        with pytest.raises(ValueError, match="feature provider"):
            build_feature_providers(cfg)


# --------------------------------------------------------------------------- #
# Assembler composition
# --------------------------------------------------------------------------- #
class TestAssemblerComposition:
    def _env(self, hashing_encoder, providers):
        ls, items = make_synthetic(n_classes=4, per_class=8, seed=5)
        texts = [it.text for it in items]
        y = np.array(ls.encode_labels([it.label for it in items]))
        cfg = RetrievalConfig()
        dense = DenseRetrieverAdapter.build(hashing_encoder, texts, y, ls, cfg)
        lexical = LexicalRetrieverAdapter.build(texts, y, ls, cfg)
        for p in providers:
            p.fit(items, ls)
        assembler = FeatureAssembler(ls, CandidatePolicy(top_n_per_signal=3))
        q_emb = hashing_encoder.encode(texts[:6])
        feats = assembler.assemble(
            texts[:6], q_emb, dense, lexical, k_neighbors=3,
            query_ids=list(range(6)), providers=providers,
        )
        return feats

    def test_provider_columns_appended_after_core(self, hashing_encoder):
        provider = ClassKeywordOverlapProvider()
        feats = self._env(hashing_encoder, [provider])
        feature_cols = [c for c in feats.columns if c not in ("item_id", "candidate", "is_true")]
        assert feature_cols == FEATURE_NAMES + ["class_kw_overlap"]

    def test_no_providers_matches_core_exactly(self, hashing_encoder):
        feats = self._env(hashing_encoder, [])
        feature_cols = [c for c in feats.columns if c not in ("item_id", "candidate", "is_true")]
        assert feature_cols == FEATURE_NAMES

    def test_provider_column_is_float32(self, hashing_encoder):
        feats = self._env(hashing_encoder, [ClassKeywordOverlapProvider()])
        assert feats["class_kw_overlap"].dtype == np.float32

    def test_wrong_length_column_raises(self, hashing_encoder):
        class _BadProvider(ClassKeywordOverlapProvider):
            def compute(self, ctx):
                return {"class_kw_overlap": np.zeros(ctx.n_candidates + 1)}

        with pytest.raises(ValueError, match="expected one value per candidate"):
            self._env(hashing_encoder, [_BadProvider()])
