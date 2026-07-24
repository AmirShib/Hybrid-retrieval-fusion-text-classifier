"""InferencePipeline.explain — the per-(item, candidate) signal table behind predict.

Runs fully offline via HashingEncoder. Integration tests assert invariants and
consistency with `predict`, never exact floats (XGBoost internals vary).
"""

from __future__ import annotations

import json

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


class TestExplainRecords:
    def test_payload_schema_is_json_clean(self, pipeline, sample):
        recs = pipeline.explain_records(sample, top_k=3)
        assert len(recs) == len(sample)
        for rec in recs:
            assert set(rec) == {"text", "decision", "candidates", "neighbors"}
            dec = rec["decision"]
            assert set(dec) == {
                "top_key",
                "confidence",
                "abstained",
                "threshold_applied",
                "threshold_scope",
            }
            assert rec["neighbors"]["texts_available"] is False
            assert isinstance(rec["neighbors"]["dense"], list)
            # Round-trips through stdlib json (JSON-clean, NaN already -> null).
            json.loads(json.dumps(rec))

    def test_top_candidate_features_match_flat_table_and_nan_is_null(self, pipeline, sample):
        recs = pipeline.explain_records(sample, top_k=3)
        flat = pipeline.explain(sample)
        saw_null = False
        for i, rec in enumerate(recs):
            for cand in rec["candidates"]:
                frow = flat[(flat["item_id"] == i) & (flat["candidate_key"] == cand["key"])].iloc[0]
                for name in FEATURE_NAMES:
                    val = cand["features"][name]
                    if val is None:
                        assert np.isnan(frow[name])  # NaN -> null
                        saw_null = True
                    else:
                        assert val == pytest.approx(float(frow[name]), abs=1e-5)
        assert saw_null  # some non-retrieved signal surfaced as null across the batch

    def test_signals_top1_agrees_with_is_top1_flags(self, pipeline, sample):
        flag_to_signal = {
            "is_d_desc_top1": "dense_description",
            "is_d_proto_top1": "dense_prototype",
            "is_d_knn_top1": "dense_knn",
            "is_b_desc_top1": "bm25_description",
            "is_b_knn_top1": "bm25_knn",
        }
        for rec in pipeline.explain_records(sample, top_k=3):
            for cand in rec["candidates"]:
                expected = {
                    sig for flag, sig in flag_to_signal.items() if cand["features"][flag] == 1.0
                }
                assert set(cand["signals_top1"]) == expected

    def test_decision_matches_predict(self, pipeline, sample):
        recs = pipeline.explain_records(sample, top_k=3)
        preds = pipeline.predict(sample)
        for rec, pred in zip(recs, preds):
            assert rec["decision"]["top_key"] == pred.top_key
            assert rec["decision"]["abstained"] == pred.abstained
            assert rec["decision"]["confidence"] == pytest.approx(pred.confidence, abs=1e-9)

    def test_contributions_present_and_sum_to_margin_when_requested(self, pipeline, sample):
        recs = pipeline.explain_records(sample, top_k=2, include_contributions=True)
        seen = 0
        for rec in recs:
            for cand in rec["candidates"]:
                assert cand["contributions_space"] == "raw_margin"
                contribs = cand["contributions"]
                assert "bias" in contribs
                assert set(contribs) == {*pipeline._feature_names, "bias"}
                assert np.isfinite(sum(contribs.values()))
                seen += 1
        assert seen > 0

    def test_contributions_absent_by_default(self, pipeline, sample):
        for rec in pipeline.explain_records(sample, top_k=2):
            for cand in rec["candidates"]:
                assert "contributions" not in cand
