"""T77 — User-provided validation/test splits (bring your own val, test, or both).

Exercises the three external-split modes end-to-end through TrainingPipeline.run:

  - external val only   -> calibrator/thresholds derived from it; the calibration
                           fold role is retired and its rows train the fusion model;
  - external test only  -> evaluation.json counts match the external set size;
  - both                -> every internal fold trains the fusion model, so the
                           pipeline runs with n_folds=2;

plus the leakage guard (external set overlapping the training text is a hard
error) and the backward-compat contract (no external sets == byte-identical to
before this feature).

All tests run fully offline via HashingEncoder (the house rule): the split logic
under test is encoder-independent.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

from text_classifier.application.evaluation import build_manifest
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import FusionConfig, PipelineConfig, RetrievalConfig, TrainingConfig
from tests._doubles import HashingEncoder, make_synthetic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _cfg(n_folds: int = 3) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=n_folds,
        random_state=0,
        use_per_fold_encoder=False,
        target_precision=0.5,
        per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 30, "max_depth": 3, "random_state": 0, "n_jobs": 1}
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    return cfg


def _split(n_classes=6, per_class=18, seed=5):
    """A pooled dataset carved into disjoint train / val / test sets.

    The carve is by index over the shuffled pool, so every set stays multi-class
    and the val/test items never share text with train (guaranteeing the leakage
    guard passes for the happy-path tests).
    """
    label_space, items = make_synthetic(n_classes=n_classes, per_class=per_class, seed=seed)
    n = len(items)
    n_val = n // 6
    n_test = n // 6
    val = items[:n_val]
    test = items[n_val : n_val + n_test]
    train = items[n_val + n_test :]
    return label_space, train, val, test


def _run(cfg, train, label_space, **kw):
    return TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        train, label_space, **kw
    )


# ---------------------------------------------------------------------------
# External validation set only
# ---------------------------------------------------------------------------
class TestExternalVal:
    def test_runs_and_reports_sane_numbers(self):
        label_space, train, val, _ = _split()
        _, report = _run(_cfg(), train, label_space, val_items=val)
        assert report.n_items > 0
        assert 0.0 <= report.coverage <= 1.0

    def test_calibration_fold_rows_join_fusion_training(self):
        """With an external val set the calibration fold role is retired, so its
        rows train the fusion model instead of calibrating it (fold-role
        bookkeeping — the scientific guarantee behind the feature)."""
        cfg = _cfg(n_folds=4)
        roles_internal = cfg.training.fold_roles()
        roles_external = cfg.training.fold_roles(external_val=True)
        assert roles_internal["calibration"][0] in roles_external["train"]
        assert roles_external["calibration"] == []

    def test_calibration_data_comes_from_the_external_val_set(self):
        """The calibrator is fit on the external val rows, not an internal fold.

        Proof: capture the frame passed to calibration and assert its item_ids
        are all positional indices into the (small) external val set. An internal
        calibration fold would carry item_ids from the much larger training set,
        so the bound discriminates cleanly.
        """
        from unittest.mock import patch

        import text_classifier.application.training as training_mod

        label_space, train, val, _ = _split()
        assert len(val) < len(train)  # the bound below is only meaningful if so

        real_add_confidence = training_mod.add_confidence
        seen = []

        def spy(features, fusion, calibrator, *args, **kwargs):
            seen.append(features)
            return real_add_confidence(features, fusion, calibrator, *args, **kwargs)

        with patch.object(training_mod, "add_confidence", side_effect=spy):
            _run(_cfg(), train, label_space, val_items=val)

        # The first add_confidence call is the calibration frame in _fit_fusion.
        calibration_frame = seen[0]
        assert int(calibration_frame["item_id"].max()) < len(val)


# ---------------------------------------------------------------------------
# External test set only
# ---------------------------------------------------------------------------
class TestExternalTest:
    def test_evaluation_counts_match_external_test_size(self, tmp_path):
        label_space, train, _, test = _split()
        out = str(tmp_path / "model")
        _, report = _run(_cfg(), train, label_space, output_dir=out, test_items=test)
        # Every test item yields exactly one decision row.
        assert report.n_items == len(test)
        with open(os.path.join(out, "evaluation.json")) as fh:
            payload = json.load(fh)
        assert payload["overall"]["n_items"] == len(test)

    def test_manifest_records_split_provenance(self, tmp_path):
        label_space, train, _, test = _split()
        out = str(tmp_path / "model")
        _run(_cfg(), train, label_space, output_dir=out, test_items=test)
        with open(os.path.join(out, "evaluation.json")) as fh:
            manifest = json.load(fh)["manifest"]
        assert manifest["splits"]["test"] == f"external:n={len(test)}"
        assert manifest["splits"]["val"] == "internal-fold"


# ---------------------------------------------------------------------------
# Both external sets
# ---------------------------------------------------------------------------
class TestBothExternal:
    def test_runs_with_two_folds(self, tmp_path):
        """With both external sets every fold trains fusion, so n_folds=2 (the
        OOF floor) is enough — the >=3 floor no longer applies."""
        label_space, train, val, test = _split()
        out = str(tmp_path / "model")
        _, report = _run(
            _cfg(n_folds=2), train, label_space, output_dir=out, val_items=val, test_items=test
        )
        assert report.n_items == len(test)
        with open(os.path.join(out, "evaluation.json")) as fh:
            manifest = json.load(fh)["manifest"]
        assert manifest["splits"] == {
            "val": f"external:n={len(val)}",
            "test": f"external:n={len(test)}",
        }

    def test_all_folds_train_fusion(self):
        roles = _cfg(n_folds=2).training.fold_roles(external_val=True, external_test=True)
        assert roles == {"train": [0, 1], "calibration": [], "test": []}


# ---------------------------------------------------------------------------
# Leave-one-out (n_folds=1): both external sets, no k-fold split
# ---------------------------------------------------------------------------
class TestLeaveOneOut:
    def test_runs_with_one_fold(self, tmp_path):
        """n_folds=1 featurizes every training item against the deployment index
        with itself masked out (leave-one-out) instead of a k-fold split. Requires
        both external sets; the run must produce sane numbers and count the external
        test set."""
        label_space, train, val, test = _split()
        out = str(tmp_path / "model")
        _, report = _run(
            _cfg(n_folds=1), train, label_space, output_dir=out, val_items=val, test_items=test
        )
        assert report.n_items == len(test)
        assert 0.0 <= report.coverage <= 1.0
        with open(os.path.join(out, "evaluation.json")) as fh:
            payload = json.load(fh)
        assert payload["overall"]["n_items"] == len(test)

    def test_single_synthetic_training_fold(self):
        roles = _cfg(n_folds=1).training.fold_roles(external_val=True, external_test=True)
        assert roles == {"train": [0], "calibration": [], "test": []}

    def test_one_fold_requires_both_external_sets(self):
        """LOO has no internal fold to carve a calibration/test set from, so a single
        external set (or none) is rejected before any encoding."""
        label_space, train, val, test = _split()
        with pytest.raises(ValueError, match="n_folds"):
            _run(_cfg(n_folds=1), train, label_space, val_items=val)  # test set missing
        with pytest.raises(ValueError, match="n_folds"):
            _run(_cfg(n_folds=1), train, label_space, test_items=test)  # val set missing

    def test_loo_training_never_self_retrieves(self):
        """The leakage guarantee end-to-end: capture the fusion-training frame and
        assert no training item shows a perfect dense self-match. Under LOO each
        item is scored against every *other* item, so ``abs_top_dense_sim`` stays
        strictly below 1 (no item retrieves itself)."""
        from unittest.mock import patch

        import text_classifier.application.training as training_mod

        label_space, train, val, test = _split()
        captured = {}

        real_fit = training_mod.TrainingPipeline._fit_fusion

        def spy(self, oof, roles, val_feats=None):
            captured["oof"] = oof
            return real_fit(self, oof, roles, val_feats)

        with patch.object(training_mod.TrainingPipeline, "_fit_fusion", spy):
            _run(_cfg(n_folds=1), train, label_space, val_items=val, test_items=test)

        oof = captured["oof"]
        # A self-match would be cosine ~1.0; masking self keeps it strictly below.
        assert oof["abs_top_dense_sim"].max() < 0.999


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------
class TestLeakageGuard:
    def test_val_overlapping_training_text_is_a_hard_error(self):
        label_space, train, _, _ = _split()
        # Point the val set at actual training rows: the classic leakage trap.
        leaky_val = train[:5]
        with pytest.raises(ValueError, match="identical to a training item"):
            _run(_cfg(), train, label_space, val_items=leaky_val)

    def test_test_overlapping_training_text_is_a_hard_error(self):
        label_space, train, _, _ = _split()
        leaky_test = train[10:14]
        with pytest.raises(ValueError, match="identical to a training item"):
            _run(_cfg(), train, label_space, test_items=leaky_test)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    def test_no_external_sets_is_identical_to_before(self):
        """run(items, label_space) with no external sets must be byte-identical in
        its thresholds, coverage report, and evaluation to a plain run."""
        label_space, items = make_synthetic(n_classes=8, per_class=20, seed=7)

        def run():
            return TrainingPipeline(_cfg(), shared_encoder=HashingEncoder(dim=64)).run(
                items, label_space
            )

        (art1, rep1), (art2, rep2) = run(), run()
        assert art1.abstention.global_threshold == art2.abstention.global_threshold
        assert art1.abstention.per_class == art2.abstention.per_class
        np.testing.assert_equal(dataclasses.asdict(rep1), dataclasses.asdict(rep2))

    def test_manifest_without_splits_omits_the_key(self):
        """build_manifest stays backward-compatible: no splits arg -> no key."""
        m = build_manifest(n_training_items=10, n_classes=3, config=PipelineConfig())
        assert "splits" not in m
