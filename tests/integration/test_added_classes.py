"""T78 — Widen a trained model's label space at inference time, without retraining.

Exercises ``DeployedArtifacts.with_added_classes`` /
``InferencePipeline.with_added_classes`` end-to-end:

  - a class added purely at inference is **retrievable** and wins as the top
    candidate for a query that matches its description (from the description
    alone — no examples, frozen encoder);
  - adding classes is **index-stable**: new classes append at the end, so every
    existing item's prediction is byte-identical before and after the extension
    (the guarantee that lets the trained fusion model be reused verbatim);
  - the extended model **round-trips** through the artifact repository;
  - the input-contract errors (empty, key collision, duplicate new keys);
  - the evaluate CLI's ``--classes`` path scores a test set whose label space is
    larger than the training one.

All tests run fully offline via HashingEncoder (the house rule): HashingEncoder
embeds any token, so a brand-new class description is retrievable without a real
model or network.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from text_classifier import ClassDefinition, InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.cli import evaluate as evaluate_cli
from text_classifier.config import FusionConfig, PipelineConfig, RetrievalConfig, TrainingConfig
from text_classifier.infrastructure import ArtifactRepository
from tests._doubles import HashingEncoder, make_synthetic

# A class whose vocabulary is disjoint from make_synthetic's ``w{i}`` themes, so
# it only ever retrieves for queries built from its own tokens.
NEW_KEY = "NEWCLS"
NEW_TOKENS = "zzq zzr zzs zzt zzu zzv"
NEW_CLASS = ClassDefinition(NEW_KEY, f"a class about {NEW_TOKENS}")
MATCHING_QUERY = "zzq zzr zzs zzt"


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


def _train(seed: int = 5):
    label_space, items = make_synthetic(n_classes=6, per_class=18, seed=seed)
    artifacts, _ = TrainingPipeline(_cfg(), shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space
    )
    return InferencePipeline(artifacts), label_space, items


# --------------------------------------------------------------------------- #
# Retrievability
# --------------------------------------------------------------------------- #
class TestRetrievable:
    def test_added_class_wins_for_a_matching_query(self):
        pipeline, _, _ = _train()
        extended = pipeline.with_added_classes([NEW_CLASS])

        assert NEW_KEY in extended.label_space.keys
        assert NEW_KEY not in pipeline.label_space.keys  # original left untouched

        pred = extended.predict([MATCHING_QUERY])[0]
        assert pred.top_key == NEW_KEY  # retrievable from its description alone

    def test_added_class_appears_in_topk(self):
        pipeline, _, _ = _train()
        extended = pipeline.with_added_classes([(NEW_KEY, f"about {NEW_TOKENS}")])
        topk = extended.predict_topk([MATCHING_QUERY], k=3)[0]
        assert NEW_KEY in [key for key, _ in topk]

    def test_description_only_class_has_no_example_support(self):
        """The new class ships description-only: NaN prototype, zero class_freq.
        This is what makes the fusion model assign it low confidence."""
        pipeline, _, _ = _train()
        extended = pipeline.with_added_classes([NEW_CLASS]).artifacts
        new_idx = extended.label_space.index_of(NEW_KEY)
        assert extended.dense.class_freq[new_idx] == 0
        assert np.all(np.isnan(extended.dense.state.prototypes[new_idx]))


# --------------------------------------------------------------------------- #
# Index stability — the guarantee that lets the trained model be reused as-is
# --------------------------------------------------------------------------- #
class TestIndexStability:
    def test_existing_predictions_are_unchanged(self):
        """Appending classes at the end must not perturb any existing item's
        decision: the trained fusion model/calibrator are reused verbatim, and no
        existing column index moves."""
        pipeline, _, items = _train()
        probe = [it.text for it in items[:40]]
        before = pipeline.predict(probe)

        extended = pipeline.with_added_classes([NEW_CLASS])
        after = extended.predict(probe)

        for b, a in zip(before, after):
            assert a.top_key == b.top_key
            assert a.predicted_key == b.predicted_key
            assert a.abstained == b.abstained
            assert a.confidence == pytest.approx(b.confidence, abs=1e-9)

    def test_existing_class_indices_are_preserved(self):
        pipeline, label_space, _ = _train()
        extended = pipeline.with_added_classes([NEW_CLASS])
        for key in label_space.keys:
            assert extended.label_space.index_of(key) == label_space.index_of(key)


# --------------------------------------------------------------------------- #
# Round-trip: an extended model is a first-class, shippable artifact
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_extended_model_saves_and_reloads(self, tmp_path):
        pipeline, _, _ = _train()
        extended = pipeline.with_added_classes([NEW_CLASS])
        out = str(tmp_path / "extended_model")
        ArtifactRepository().save(extended.artifacts, out)

        reloaded = InferencePipeline.from_directory(out)
        assert reloaded.label_space.keys == extended.label_space.keys
        # The reloaded model still retrieves the added class.
        assert reloaded.predict([MATCHING_QUERY])[0].top_key == NEW_KEY


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #
class TestErrors:
    def test_empty_is_rejected(self):
        pipeline, _, _ = _train()
        with pytest.raises(ValueError, match="at least one new class"):
            pipeline.with_added_classes([])

    def test_collision_with_existing_key_is_rejected(self):
        pipeline, label_space, _ = _train()
        existing = label_space.keys[0]
        with pytest.raises(ValueError, match="already in the label space"):
            pipeline.with_added_classes([(existing, "some other description")])

    def test_duplicate_new_keys_are_rejected(self):
        pipeline, _, _ = _train()
        with pytest.raises(ValueError, match="unique"):
            pipeline.with_added_classes([(NEW_KEY, "one"), (NEW_KEY, "two")])


# --------------------------------------------------------------------------- #
# Evaluate CLI: score a test set whose label space is larger than training's
# --------------------------------------------------------------------------- #
def _run_cli(module, argv) -> None:
    with patch.object(sys, "argv", argv):
        module.main()


class TestEvaluateCliWiderLabelSpace:
    def _write_model_and_data(self, tmp_path):
        pipeline, label_space, items = _train()
        out = str(tmp_path / "model")
        ArtifactRepository().save(pipeline.artifacts, out)

        # A classes CSV that lists every training class PLUS the new one, and a
        # labeled set that references the new label (unknown to the model).
        classes_csv = tmp_path / "classes.csv"
        pd.DataFrame(
            {
                "key": label_space.keys + [NEW_KEY],
                "description": label_space.descriptions + [NEW_CLASS.description],
            }
        ).to_csv(classes_csv, index=False)

        labeled_csv = tmp_path / "labeled.csv"
        rows = [(it.text, it.label) for it in items[:20]]
        rows += [(MATCHING_QUERY, NEW_KEY), (f"{NEW_TOKENS} extra", NEW_KEY)]
        pd.DataFrame(rows, columns=["text", "label"]).to_csv(labeled_csv, index=False)
        return out, str(classes_csv), str(labeled_csv), len(rows)

    def test_new_labels_rejected_without_classes_flag(self, tmp_path):
        out, _, labeled_csv, _ = self._write_model_and_data(tmp_path)
        with pytest.raises(SystemExit, match="not in the model"):
            _run_cli(evaluate_cli, ["eval", "--model", out, "--input", labeled_csv])

    def test_classes_flag_widens_and_scores_end_to_end(self, tmp_path, capsys):
        out, classes_csv, labeled_csv, n_rows = self._write_model_and_data(tmp_path)
        report_path = str(tmp_path / "report.json")
        _run_cli(
            evaluate_cli,
            [
                "eval",
                "--model",
                out,
                "--input",
                labeled_csv,
                "--classes",
                classes_csv,
                "--output",
                report_path,
            ],
        )
        printed = capsys.readouterr().out
        assert "added 1 class" in printed and NEW_KEY in printed

        assert os.path.isfile(report_path)
        import json

        with open(report_path) as fh:
            report = json.load(fh)
        # Every labeled row (including the two new-class rows) was scored.
        assert report["overall"]["n_items"] == n_rows
        # The report is over the widened label space (6 trained + 1 added).
        assert report["manifest"]["n_classes"] == 7
