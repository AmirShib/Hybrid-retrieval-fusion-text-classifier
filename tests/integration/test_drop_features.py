"""T82 — end-to-end parity for a model trained on a feature subset.

The risk this guards: `drop_features` narrows what the fusion model is fitted on,
so inference must reconstruct the *identical* column list from the persisted
model dir. Getting it wrong feeds XGBoost mislabelled columns and produces
silently wrong scores — the same failure mode the composed-schema check exists
to prevent.
"""

from __future__ import annotations

import json
import os

import pytest

from text_classifier import InferencePipeline, LabeledItem, PipelineConfig, TrainingPipeline
from text_classifier.config import FusionConfig, TrainingConfig
from text_classifier.domain import FEATURE_NAMES, composed_feature_names

DROPPED = ["margin_d_desc", "q_gap_d_knn", "b_knn_count"]


def _config(drop):
    cfg = PipelineConfig(candidate_top_n=5)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(n_folds=3, target_precision=0.5, per_class_min_support=100)
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 12, "max_depth": 3, "n_jobs": 1, "random_state": 0},
        drop_features=list(drop),
    )
    return cfg


@pytest.fixture
def task():
    """Items truncated to two tokens, the same trick the quality benchmark uses:
    on the full texts the task is so easy every confidence saturates at 1.0 and
    no feature change is observable."""
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=8, per_class=10, seed=3)
    return label_space, [LabeledItem(" ".join(it.text.split()[:2]), it.label) for it in items]


def _train(tmp_path, task, drop, name="m"):
    label_space, items = task
    out = str(tmp_path / name)
    TrainingPipeline(_config(drop)).run(items, label_space, output_dir=out)
    return out


class TestDropFeaturesPersistence:
    def test_meta_records_the_narrowed_schema(self, tmp_path, task):
        out = _train(tmp_path, task, DROPPED)
        with open(os.path.join(out, "meta.json")) as fh:
            meta = json.load(fh)
        assert len(meta["feature_names"]) == len(FEATURE_NAMES) - len(DROPPED)
        for name in DROPPED:
            assert name not in meta["feature_names"]

    def test_drop_list_travels_in_the_config_block(self, tmp_path, task):
        out = _train(tmp_path, task, DROPPED)
        with open(os.path.join(out, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["config"]["fusion"]["drop_features"] == DROPPED

    def test_loaded_pipeline_scores_on_the_same_columns(self, tmp_path, task):
        out = _train(tmp_path, task, DROPPED)
        pipe = InferencePipeline.from_directory(out)
        with open(os.path.join(out, "meta.json")) as fh:
            saved = json.load(fh)["feature_names"]
        assert pipe._feature_names == saved

    def test_load_and_predict_round_trips(self, tmp_path, task):
        """The end the user actually sees: a subset-trained model dir loads and
        classifies without a schema-mismatch error."""
        _, items = task
        out = _train(tmp_path, task, DROPPED)
        preds = InferencePipeline.from_directory(out).predict([it.text for it in items[:5]])
        assert len(preds) == 5

    def test_assembler_still_produces_every_column(self, tmp_path, task):
        """Dropping narrows the *model*, not the frame: explain/signal_report and
        the masking ablation all read core columns by name."""
        _, items = task
        out = _train(tmp_path, task, DROPPED)
        frame = InferencePipeline.from_directory(out).explain([it.text for it in items[:3]])
        for name in DROPPED:
            assert name in frame.columns, f"{name} vanished from the assembled frame"


class TestDropFeaturesBehaviour:
    def test_no_drop_is_unchanged_from_the_full_schema(self, tmp_path, task):
        out = _train(tmp_path, task, [])
        with open(os.path.join(out, "meta.json")) as fh:
            meta = json.load(fh)
        assert meta["feature_names"] == composed_feature_names()

    def test_the_subset_actually_reaches_the_fusion_model(self, tmp_path, task):
        """The load-bearing claim: a dropped column is not merely filtered out of
        the frame, the model is *fitted on fewer columns*.

        Asserted through the fusion model's own width (``predict_contribs``
        returns one column per feature plus a bias), not through predicted
        confidence: on a task small enough to run in CI the isotonic calibrator
        quantizes both models to the same handful of values, so confidence
        cannot witness this."""
        import numpy as np

        drop = ["d_desc_sim", "b_desc_sim", "d_proto_sim"]
        full = InferencePipeline.from_directory(_train(tmp_path, task, [], "full"))
        thin = InferencePipeline.from_directory(_train(tmp_path, task, drop, "thin"))

        assert len(thin._feature_names) == len(full._feature_names) - len(drop)
        n_full, n_thin = len(full._feature_names), len(thin._feature_names)
        c_full = full.artifacts.fusion.predict_contribs(np.zeros((3, n_full), dtype=np.float32))
        c_thin = thin.artifacts.fusion.predict_contribs(np.zeros((3, n_thin), dtype=np.float32))
        assert c_full is not None and c_thin is not None
        assert c_full.shape[1] == n_full + 1  # +1 = bias column
        assert c_thin.shape[1] == n_thin + 1

    def test_unknown_column_fails_fast_at_train_time(self, tmp_path, task):
        label_space, items = task
        with pytest.raises(ValueError, match="not_a_column"):
            TrainingPipeline(_config(["not_a_column"])).run(items, label_space)


class TestRetrainAblationEndToEnd:
    def test_runs_a_small_sweep_against_the_real_trainer(self, task):
        """The harness on the genuine training pipeline (not the stub): two arms
        x two seeds + baseline, offline via the hashing encoder."""
        from text_classifier.application.retrain_ablation import AblationArm, retrain_ablation

        label_space, items = task
        rep = retrain_ablation(
            items,
            label_space,
            _config([]),
            [AblationArm("margins", ("margin_d_desc",)), AblationArm("gaps", ("q_gap_d_knn",))],
            seeds=[0, 1],
        )
        assert rep["n_runs"] == 6
        assert [a["name"] for a in rep["arms"]] == ["margins", "gaps"]
        for arm in rep["arms"]:
            assert arm["verdict"] in {"earns_place", "redundant", "inconclusive"}
            assert len(arm["per_seed"]) == 2

    def test_baseline_arm_carries_no_drop(self, task):
        from text_classifier.application.retrain_ablation import AblationArm, retrain_ablation

        label_space, items = task
        rep = retrain_ablation(
            items, label_space, _config([]), [AblationArm("m", ("margin_d_desc",))], seeds=[0]
        )
        assert rep["baseline"]["drop"] == []
