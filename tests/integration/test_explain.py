"""InferencePipeline.explain — the per-(item, candidate) signal table behind predict.

Runs fully offline via HashingEncoder. Integration tests assert invariants and
consistency with `predict`, never exact floats (XGBoost internals vary).
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier import PipelineConfig, TrainingPipeline
from text_classifier.application.inference import InferencePipeline
from text_classifier.domain import FEATURE_NAMES
from tests._doubles import HashingEncoder, make_synthetic


@pytest.fixture(scope="module")
def pipeline_and_sample():
    label_space, items = make_synthetic(n_classes=8, per_class=20, seed=7)
    cfg = PipelineConfig(candidate_top_n=6)
    cfg.encoder.kind = "hashing"
    cfg.training.n_folds = 3
    cfg.training.target_precision = 0.5
    cfg.training.per_class_min_support = 1
    cfg.retrieval.k_neighbors = 10
    artifacts, _ = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space
    )
    return InferencePipeline(artifacts), [it.text for it in items[:6]]


@pytest.fixture
def pipeline(pipeline_and_sample) -> InferencePipeline:
    return pipeline_and_sample[0]


@pytest.fixture
def sample(pipeline_and_sample):
    return pipeline_and_sample[1]


class TestSchema:
    def test_leading_columns_then_full_feature_schema(self, pipeline, sample):
        df = pipeline.explain(sample)
        assert list(df.columns)[:5] == ["item_id", "text", "rank", "candidate_key", "conf"]
        # Every core signal column is present, after the leading columns.
        for name in FEATURE_NAMES:
            assert name in df.columns

    def test_empty_input_returns_empty_framed_table(self, pipeline):
        df = pipeline.explain([])
        assert len(df) == 0
        assert list(df.columns)[:5] == ["item_id", "text", "rank", "candidate_key", "conf"]


class TestRankingAndConsistency:
    def test_rank_is_one_based_and_conf_descending_per_item(self, pipeline, sample):
        df = pipeline.explain(sample)
        for _, g in df.groupby("item_id"):
            g = g.sort_values("rank")
            assert g["rank"].tolist() == list(range(1, len(g) + 1))
            confs = g["conf"].to_numpy()
            assert np.all(np.diff(confs) <= 1e-9)  # non-increasing

    def test_rank1_matches_predict_decision(self, pipeline, sample):
        df = pipeline.explain(sample)
        preds = pipeline.predict(sample)
        top1 = df[df["rank"] == 1].set_index("item_id")
        for i, pred in enumerate(preds):
            if i not in top1.index:
                continue  # item retrieved no candidate; predict abstains with top_key ""
            row = top1.loc[i]
            assert row["candidate_key"] == pred.top_key
            assert row["conf"] == pytest.approx(pred.confidence, abs=1e-9)

    def test_top_k_bounds_rows_per_item(self, pipeline, sample):
        df = pipeline.explain(sample, top_k=2)
        assert df.groupby("item_id").size().max() <= 2
        assert (df["rank"] <= 2).all()

    def test_item_id_indexes_into_inputs_and_text_matches(self, pipeline, sample):
        df = pipeline.explain(sample)
        assert df["item_id"].min() >= 0
        assert df["item_id"].max() < len(sample)
        for _, row in df.iterrows():
            assert row["text"] == sample[int(row["item_id"])]


class TestSignalDetail:
    def test_not_retrieved_is_nan_not_zero(self, pipeline, sample):
        # Across a batch, at least one lexical signal fails to retrieve some
        # candidate: that is encoded as NaN (missing), never a true 0.
        df = pipeline.explain(sample)
        assert df[["b_desc_sim", "b_knn_sum", "d_knn_sum"]].isna().any().any()
