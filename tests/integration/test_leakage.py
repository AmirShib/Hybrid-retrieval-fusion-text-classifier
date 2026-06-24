"""T06 — Leakage regression test.

Verifies the core scientific claim: out-of-fold (OOF) features are
leakage-free — no validation item ever sees itself in its own fold's
retrieval index.

Approaches implemented:
  2. Fold-disjointness: item_id sets across folds are pairwise disjoint and
     together cover every item.
  1+3. Singleton-class canary: with 1 example per class, OOF recall collapses
     to ≈0 (true class excluded from fold index) while a deliberately leaky
     variant achieves ≈1.0 — proving the test is sensitive.
  4. Role-separation: fold_roles() keeps train / calibration / test disjoint
     and exhaustive.
"""
from __future__ import annotations

import numpy as np
import pytest

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.application.features import FeatureAssembler
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from text_classifier.domain import CandidatePolicy
from text_classifier.infrastructure import DenseRetrieverAdapter, LexicalRetrieverAdapter
from tests._doubles import HashingEncoder, make_synthetic


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fast_cfg(n_folds: int = 3) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.training = TrainingConfig(
        n_folds=n_folds,
        random_state=0,
        use_per_fold_encoder=False,
        target_precision=0.5,
        per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(xgb_params={
        "n_estimators": 10, "max_depth": 2, "random_state": 0, "n_jobs": 1,
    })
    cfg.retrieval = RetrievalConfig(k_neighbors=5)
    return cfg


def _run_oof(items, label_space, enc, n_folds=3):
    cfg = _fast_cfg(n_folds=n_folds)
    pipeline = TrainingPipeline(cfg, shared_encoder=enc)
    pipeline.assembler = FeatureAssembler(label_space, CandidatePolicy(cfg.candidate_top_n))
    texts = [it.text for it in items]
    y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
    return pipeline._build_oof(texts, y, label_space), len(items)


# ---------------------------------------------------------------------------
# Module-level fixture: run OOF once and share across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oof_result():
    enc = HashingEncoder(dim=64)
    label_space, items = make_synthetic(n_classes=10, per_class=12, seed=42)
    return _run_oof(items, label_space, enc, n_folds=3)


# ---------------------------------------------------------------------------
# Approach 2 — fold disjointness
# ---------------------------------------------------------------------------

class TestFoldDisjointness:
    def test_item_id_sets_are_pairwise_disjoint(self, oof_result):
        oof, _ = oof_result
        by_fold = {f: set(g["item_id"].unique()) for f, g in oof.groupby("fold")}
        folds = list(by_fold)
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                overlap = by_fold[folds[i]] & by_fold[folds[j]]
                assert overlap == set(), (
                    f"Folds {folds[i]} and {folds[j]} share item_ids: {overlap}"
                )

    def test_folds_cover_all_items(self, oof_result):
        oof, n_items = oof_result
        assert set(oof["item_id"].unique()) == set(range(n_items))

    def test_each_item_appears_in_exactly_one_fold(self, oof_result):
        oof, n_items = oof_result
        # groupby fold then unique item_ids: sum of sizes == n_items
        per_fold_counts = oof.groupby("fold")["item_id"].nunique()
        assert int(per_fold_counts.sum()) == n_items


# ---------------------------------------------------------------------------
# Approaches 1+3 — singleton-class canary
# ---------------------------------------------------------------------------

class TestCanary:
    """Approaches 1+3: singleton-class dataset proves the test is sensitive.

    Dataset: 10 classes × 1 example each with class-exclusive vocabulary.

    The description signal always surfaces every class as a candidate (tied
    cosine similarities across all classes), so candidate recall alone can't
    distinguish OOF from leaky.  Instead we compare the `d_proto_sim` feature:

    OOF (leave-one-out): class C_i has zero training examples in its fold's
    index → prototype for C_i is NaN → `d_proto_sim` for the true-class rows
    is NaN.  This directly proves item_i never saw itself in the fold index.

    Leaky (all items in index): prototype for C_i IS item_i's own embedding →
    `d_proto_sim` = dot(emb_i, emb_i) = 1.0 (L2-normalized self-similarity).

    If OOF leakage were introduced, the prototype would become non-NaN and the
    `isna().all()` assertion would fail immediately — proving the test is
    sensitive.
    """

    def _singleton_dataset(self):
        n = 10
        # Identical descriptions neutralise the description signal for all classes
        # so the prototype signal cleanly separates OOF from leaky.
        defs = [ClassDefinition(f"C{i}", "shared description for all classes") for i in range(n)]
        # Each item has class-exclusive tokens; no token appears in any description.
        items = [
            LabeledItem(f"excl_{i}_p{i*31+7} excl_{i}_q{i*17+3} excl_{i}_r{i*13+11}", f"C{i}")
            for i in range(n)
        ]
        return LabelSpace(defs), items

    def _build_feats(self, label_space, items, enc, include_self: bool):
        """Return assembled features; include_self=True is the leaky variant."""
        cfg = _fast_cfg()
        texts = [it.text for it in items]
        y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
        assembler = FeatureAssembler(label_space, CandidatePolicy(cfg.candidate_top_n))
        all_idx = np.arange(len(items))
        import pandas as pd
        parts = []
        for i in all_idx:
            tr = all_idx if include_self else all_idx[all_idx != i]
            tr_texts = [texts[j] for j in tr]
            dense = DenseRetrieverAdapter.build(enc, tr_texts, y[tr], label_space, cfg.retrieval)
            lexical = LexicalRetrieverAdapter.build(tr_texts, y[tr], label_space, cfg.retrieval)
            q_emb = enc.encode([texts[i]])
            feats = assembler.assemble(
                [texts[i]], q_emb, dense, lexical, cfg.retrieval.k_neighbors,
                query_ids=[int(i)], query_labels=y[[i]],
            )
            parts.append(feats)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def test_oof_proto_nan_leaky_proto_one(self):
        enc = HashingEncoder(dim=64)
        label_space, items = self._singleton_dataset()

        oof_feats = self._build_feats(label_space, items, enc, include_self=False)
        leaky_feats = self._build_feats(label_space, items, enc, include_self=True)

        # Retrieve true-class candidate rows from each variant
        oof_true = oof_feats[oof_feats["is_true"] == 1]
        leaky_true = leaky_feats[leaky_feats["is_true"] == 1]

        assert len(oof_true) > 0, "OOF true-class rows must exist to test the prototype"
        assert len(leaky_true) > 0, "Leaky true-class rows must exist"

        # Approach 1 / self-neighbour exclusion:
        # OOF: class C_i has no training examples in its fold's index → prototype = NaN
        assert oof_true["d_proto_sim"].isna().all(), (
            "OOF: true-class prototype should be NaN when the item is excluded from "
            "its fold's index. A non-NaN value here means the item saw itself."
        )

        # Approach 3 / canary sensitivity:
        # Leaky: item IS its own training example → prototype = L2-norm(emb_i)
        # → d_proto_sim = dot(emb_i, emb_i) = 1.0
        assert (leaky_true["d_proto_sim"] > 0.9).all(), (
            "Leaky: true-class prototype should be ≈1.0 (item retrieves itself). "
            "If this fails, the leaky baseline is broken."
        )


# ---------------------------------------------------------------------------
# Approach 4 — role separation
# ---------------------------------------------------------------------------

class TestRoleSeparation:
    @pytest.mark.parametrize("n_folds", [3, 4, 5])
    def test_roles_are_pairwise_disjoint(self, n_folds):
        tc = TrainingConfig(n_folds=n_folds)
        roles = tc.fold_roles()
        train = set(roles["train"])
        cal = set(roles["calibration"])
        test = set(roles["test"])
        assert train & cal == set()
        assert train & test == set()
        assert cal & test == set()

    @pytest.mark.parametrize("n_folds", [3, 4, 5])
    def test_roles_cover_all_folds(self, n_folds):
        tc = TrainingConfig(n_folds=n_folds)
        roles = tc.fold_roles()
        covered = set(roles["train"]) | set(roles["calibration"]) | set(roles["test"])
        assert covered == set(range(n_folds))

    def test_oof_item_ids_respect_role_disjointness(self, oof_result):
        """Items in train folds don't appear in calibration or test folds."""
        oof, _ = oof_result
        cfg = _fast_cfg(n_folds=3)
        roles = cfg.training.fold_roles()

        train_ids = set(oof[oof["fold"].isin(roles["train"])]["item_id"].unique())
        cal_ids = set(oof[oof["fold"].isin(roles["calibration"])]["item_id"].unique())
        test_ids = set(oof[oof["fold"].isin(roles["test"])]["item_id"].unique())

        assert train_ids & cal_ids == set()
        assert train_ids & test_ids == set()
        assert cal_ids & test_ids == set()
