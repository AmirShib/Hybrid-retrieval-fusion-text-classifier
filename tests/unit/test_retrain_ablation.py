"""T82 — retrain-based feature ablation.

Part A: ``fusion_feature_names`` — the seam that lets a model be trained on a
        subset of the schema, with the validation that stops a typo silently
        invalidating an experiment.
Part B: ``retrain_ablation`` aggregation — paired deltas and the verdict rule,
        driven through a stubbed trainer so the arithmetic is exact and the test
        does not train 20 models.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.application import retrain_ablation as ra
from text_classifier.application.retrain_ablation import (
    AblationArm,
    arms_from_groups,
    retrain_ablation,
)
from text_classifier.config import PipelineConfig
from text_classifier.domain import FEATURE_NAMES, composed_feature_names, fusion_feature_names


# =========================================================================== #
#  Part A — the feature-subset seam
# =========================================================================== #


class TestFusionFeatureNames:
    def test_no_drop_is_the_composed_schema(self):
        assert fusion_feature_names() == composed_feature_names() == FEATURE_NAMES

    def test_drop_removes_only_the_named_columns(self):
        kept = fusion_feature_names(drop=["margin_d_desc", "q_gap_d_knn"])
        assert "margin_d_desc" not in kept and "q_gap_d_knn" not in kept
        assert len(kept) == len(FEATURE_NAMES) - 2

    def test_order_of_survivors_is_preserved(self):
        """Column order is the schema contract — dropping must not permute it."""
        kept = fusion_feature_names(drop=["d_proto_sim", "rank_b_knn"])
        expected = [n for n in FEATURE_NAMES if n not in {"d_proto_sim", "rank_b_knn"}]
        assert kept == expected

    def test_drop_order_does_not_matter(self):
        a = fusion_feature_names(drop=["q_gap_d_desc", "margin_b_knn"])
        b = fusion_feature_names(drop=["margin_b_knn", "q_gap_d_desc"])
        assert a == b

    def test_unknown_column_raises_naming_it(self):
        """A typo must not silently drop nothing and quietly invalidate a run."""
        with pytest.raises(ValueError, match="margin_d_dsec"):
            fusion_feature_names(drop=["margin_d_dsec"])

    def test_dropping_everything_raises(self):
        with pytest.raises(ValueError, match="at least one column"):
            fusion_feature_names(drop=list(FEATURE_NAMES))

    def test_composed_schema_is_unaffected_by_drop(self):
        """The assembler keeps producing every column — signal_report, explain and
        the masking ablation all read core columns by name."""
        fusion_feature_names(drop=["b_desc_sim"])
        assert composed_feature_names() == FEATURE_NAMES


class TestDropFeaturesConfig:
    def test_default_is_empty_and_validates(self):
        cfg = PipelineConfig()
        assert cfg.fusion.drop_features == []
        cfg.validate()

    def test_duplicates_rejected(self):
        cfg = PipelineConfig()
        cfg.fusion.drop_features = ["d_desc_sim", "d_desc_sim"]
        with pytest.raises(ValueError, match="duplicates"):
            cfg.validate()

    def test_blank_name_rejected(self):
        cfg = PipelineConfig()
        cfg.fusion.drop_features = ["  "]
        with pytest.raises(ValueError, match="non-empty"):
            cfg.validate()

    def test_round_trips_through_serialization(self):
        """It rides in meta.json's config block, so inference rebuilds the same
        column list the model was fitted on."""
        cfg = PipelineConfig()
        cfg.fusion.drop_features = ["margin_d_desc"]
        assert PipelineConfig.from_dict(cfg.to_dict()).fusion.drop_features == ["margin_d_desc"]


# =========================================================================== #
#  Part B — aggregation, on a stubbed trainer
# =========================================================================== #


@pytest.fixture
def fake_runs(monkeypatch):
    """Replace the training run with a lookup table so the report arithmetic is
    exactly checkable. Keyed by (frozenset(drop), seed)."""
    table: dict = {}

    def _fake(items, label_space, config, drop, seed):
        cov, acc = table[(frozenset(drop), seed)]
        return {
            "candidate_recall": 1.0,
            "coverage": cov,
            "accuracy_on_accepted": acc,
            "accepted_correct": cov * acc,
        }

    monkeypatch.setattr(ra, "_run_one", _fake)
    return table


def _report(table, arms, seeds):
    return retrain_ablation([], None, PipelineConfig(), arms, seeds=seeds)


class TestRetrainAblationAggregation:
    def test_paired_delta_is_the_mean_of_per_seed_differences(self, fake_runs):
        # baseline accepted_correct: 0.80, 0.90 ; arm: 0.70, 0.85
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.80),
                (frozenset(), 1): (1.0, 0.90),
                (frozenset({"d_desc_sim"}), 0): (1.0, 0.70),
                (frozenset({"d_desc_sim"}), 1): (1.0, 0.85),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("d_desc_sim",))], [0, 1])
        arm = rep["arms"][0]
        # per-seed diffs: -0.10, -0.05 -> mean -0.075
        npt.assert_allclose(arm["paired_delta_accepted_correct"], -0.075, atol=1e-9)

    def test_baseline_is_trained_under_the_same_seeds(self, fake_runs):
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.8),
                (frozenset(), 7): (1.0, 0.9),
                (frozenset({"x"}), 0): (1.0, 0.8),
                (frozenset({"x"}), 7): (1.0, 0.9),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 7])
        assert rep["seeds"] == [0, 7]
        assert len(rep["baseline"]["per_seed"]) == 2

    def test_n_runs_counts_baseline_plus_every_arm(self, fake_runs):
        for drop in (frozenset(), frozenset({"x"}), frozenset({"y"})):
            for seed in (0, 1):
                fake_runs[(drop, seed)] = (1.0, 0.8)
        rep = _report(fake_runs, [AblationArm("a", ("x",)), AblationArm("b", ("y",))], [0, 1])
        assert rep["n_runs"] == (2 + 1) * 2

    def test_accepted_correct_is_coverage_times_accuracy(self, fake_runs):
        fake_runs.update({(frozenset(), 0): (0.5, 0.8), (frozenset({"x"}), 0): (0.5, 0.8)})
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0])
        npt.assert_allclose(rep["baseline"]["accepted_correct"]["mean"], 0.4, atol=1e-9)

    def test_summaries_report_spread_not_just_a_mean(self, fake_runs):
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.70),
                (frozenset(), 1): (1.0, 0.90),
                (frozenset({"x"}), 0): (1.0, 0.70),
                (frozenset({"x"}), 1): (1.0, 0.90),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        s = rep["baseline"]["accuracy_on_accepted"]
        npt.assert_allclose(s["mean"], 0.80, atol=1e-9)
        npt.assert_allclose(s["std"], 0.10, atol=1e-9)
        npt.assert_allclose([s["min"], s["max"]], [0.70, 0.90], atol=1e-9)


class TestVerdict:
    def test_consistent_harm_from_dropping_earns_its_place(self, fake_runs):
        # dropping costs 0.10 every seed -> zero spread, unambiguous
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.90),
                (frozenset(), 1): (1.0, 0.80),
                (frozenset({"x"}), 0): (1.0, 0.80),
                (frozenset({"x"}), 1): (1.0, 0.70),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        assert rep["arms"][0]["verdict"] == "earns_place"

    def test_consistent_gain_from_dropping_is_redundant(self, fake_runs):
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.80),
                (frozenset(), 1): (1.0, 0.70),
                (frozenset({"x"}), 0): (1.0, 0.90),
                (frozenset({"x"}), 1): (1.0, 0.80),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        assert rep["arms"][0]["verdict"] == "redundant"

    def test_provably_inert_columns_are_redundant_not_inconclusive(self, fake_runs):
        """Dropping changes *nothing* on every seed (delta 0.0, spread 0.0). That
        is the strongest possible evidence the columns cost nothing to remove —
        reporting it as 'inconclusive' would invert the finding."""
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.80),
                (frozenset(), 1): (1.0, 0.90),
                (frozenset({"x"}), 0): (1.0, 0.80),
                (frozenset({"x"}), 1): (1.0, 0.90),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        arm = rep["arms"][0]
        assert arm["paired_delta_accepted_correct"] == 0.0
        assert arm["paired_delta_accepted_correct_std"] == 0.0
        assert arm["verdict"] == "redundant"

    def test_effect_smaller_than_its_own_spread_is_inconclusive(self, fake_runs):
        # diffs +0.10 and -0.08: mean +0.01, std 0.09 -> noise
        fake_runs.update(
            {
                (frozenset(), 0): (1.0, 0.80),
                (frozenset(), 1): (1.0, 0.80),
                (frozenset({"x"}), 0): (1.0, 0.90),
                (frozenset({"x"}), 1): (1.0, 0.72),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        assert rep["arms"][0]["verdict"] == "inconclusive"

    def test_verdict_reads_accepted_correct_not_accuracy_alone(self, fake_runs):
        """Accuracy rising because coverage collapsed is not an improvement."""
        fake_runs.update(
            {
                (frozenset(), 0): (1.00, 0.80),  # accepted_correct 0.80
                (frozenset(), 1): (1.00, 0.80),
                (frozenset({"x"}), 0): (0.50, 0.95),  # accuracy up, product 0.475
                (frozenset({"x"}), 1): (0.50, 0.95),
            }
        )
        rep = _report(fake_runs, [AblationArm("a", ("x",))], [0, 1])
        arm = rep["arms"][0]
        assert arm["paired_delta_accuracy_on_accepted"] > 0  # accuracy improved
        assert arm["paired_delta_accepted_correct"] < 0  # but throughput fell
        assert arm["verdict"] == "earns_place"


class TestRetrainAblationValidation:
    def test_empty_seeds_rejected(self):
        with pytest.raises(ValueError, match="at least one seed"):
            retrain_ablation([], None, PipelineConfig(), [AblationArm("a", ("x",))], seeds=[])

    def test_duplicate_arm_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            retrain_ablation(
                [], None, PipelineConfig(), [AblationArm("a", ("x",)), AblationArm("a", ("y",))]
            )

    def test_baseline_is_a_reserved_arm_name(self):
        with pytest.raises(ValueError, match="reserved"):
            retrain_ablation([], None, PipelineConfig(), [AblationArm("baseline", ("x",))])

    def test_progress_callback_fires_once_per_run(self, fake_runs):
        for drop in (frozenset(), frozenset({"x"})):
            for seed in (0, 1):
                fake_runs[(drop, seed)] = (1.0, 0.8)
        seen = []
        retrain_ablation(
            [],
            None,
            PipelineConfig(),
            [AblationArm("a", ("x",))],
            seeds=[0, 1],
            progress=lambda name, seed, i, total: seen.append((name, seed, i, total)),
        )
        assert [s[0] for s in seen] == ["baseline", "baseline", "a", "a"]
        assert [s[2] for s in seen] == [1, 2, 3, 4]
        assert all(s[3] == 4 for s in seen)


class TestArmsFromGroups:
    def test_builds_arms_in_iteration_order(self):
        arms = arms_from_groups({"m": ["margin_d_desc"], "g": ["q_gap_d_knn", "q_gap_b_desc"]})
        assert [a.name for a in arms] == ["m", "g"]
        assert arms[1].drop == ("q_gap_d_knn", "q_gap_b_desc")


class TestSummarize:
    def test_all_nan_yields_nan_rather_than_raising(self):
        s = ra._summarize([float("nan"), float("nan")])
        assert all(np.isnan(s[k]) for k in ("mean", "std", "min", "max"))

    def test_nan_values_are_excluded_not_propagated(self):
        s = ra._summarize([0.5, float("nan"), 0.7])
        npt.assert_allclose(s["mean"], 0.6, atol=1e-9)
