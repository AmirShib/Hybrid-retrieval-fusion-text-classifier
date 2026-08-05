"""Integration tests for the re-tune CLI (T66): refit calibration + abstention
thresholds on a trained model without retraining the encoder, indices, or
fusion model.

Runs fully offline via the torch-free tfidf encoder, following the same
train-via-CLI fixture pattern as ``test_evaluate_cli.py``. The "fresh" tune set
is a disjoint per-class split of one synthetic generation (not a second,
differently-seeded ``make_synthetic`` call) so its class vocabulary actually
overlaps the training distribution -- two independent seeds draw unrelated
per-class themes and would make retrieval (and therefore the retuned
thresholds) meaningless.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.train as train_cli
from text_classifier.cli import tune as tune_cli
from text_classifier.datasets import make_synthetic


def _split_items(items):
    """Disjoint ~50/50 per-class split so every class appears in both halves."""
    by_label = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)
    left, right = [], []
    for group in by_label.values():
        cut = len(group) // 2
        left.extend(group[:cut])
        right.extend(group[cut:])
    return left, right


def _write_items_csv(path, items) -> str:
    pd.DataFrame({"text": [it.text for it in items], "label": [it.label for it in items]}).to_csv(
        path, index=False
    )
    return str(path)


def _run(module, argv) -> None:
    with patch.object(sys, "argv", argv):
        module.main()


def _train_model(tmp_path, target_precision="0.5"):
    """Train a model on one half of a synthetic generation; return
    (model_dir, train_items_csv, tune_items_csv) where the tune set is the
    disjoint other half (same classes/vocabulary, different concrete items)."""
    label_space, items = make_synthetic(n_classes=4, per_class=24, seed=3)
    train_items, tune_items = _split_items(items)

    items_csv = _write_items_csv(tmp_path / "items.csv", train_items)
    tune_csv = _write_items_csv(tmp_path / "tune.csv", tune_items)
    classes_csv = tmp_path / "classes.csv"
    pd.DataFrame({"key": label_space.keys, "description": label_space.descriptions}).to_csv(
        classes_csv, index=False
    )
    out = str(tmp_path / "model")
    _run(
        train_cli,
        [
            "train",
            "--items",
            items_csv,
            "--classes",
            str(classes_csv),
            "--out",
            out,
            "--encoder-kind",
            "tfidf",
            "--folds",
            "3",
            "--target-precision",
            target_precision,
            "--candidate-top-n",
            "8",
            "--k-neighbors",
            "10",
        ],
    )
    return out, items_csv, tune_csv, len(tune_items)


def _file_hashes(directory: str) -> dict:
    hashes = {}
    for path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                hashes[os.path.relpath(path, directory)] = hashlib.sha256(fh.read()).hexdigest()
    return hashes


def _retune_dir(model_dir, tmp_path, name, tune_csv, target_precision):
    """Retune a fresh copy of `model_dir` at `target_precision`; return its
    resulting (global_threshold, evaluation dict). Copying rather than reusing
    one directory across trials isolates each target-precision retune onto the
    same starting model."""
    copy_dir = str(tmp_path / name)
    shutil.copytree(model_dir, copy_dir)
    _run(
        tune_cli,
        [
            "tune",
            "--model",
            copy_dir,
            "--input",
            tune_csv,
            "--target-precision",
            str(target_precision),
            "--per-class-min-support",
            "3",
        ],
    )
    with open(os.path.join(copy_dir, "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(copy_dir, "evaluation.json")) as fh:
        evaluation = json.load(fh)
    return meta["abstention"]["global_threshold"], evaluation


def test_higher_target_precision_raises_threshold_and_cannot_raise_coverage(tmp_path):
    """`ThresholdTuner.threshold_for_precision` is monotonic in its target: for
    the same confidence distribution, a stricter target can only pick an equal
    or higher threshold, which can only keep or shrink the accepted set. Retune
    the same trained model at a low and a high target and check that guarantee
    holds end-to-end through the CLI (not just at the domain-service level)."""
    out, _, tune_csv, _ = _train_model(tmp_path, target_precision="0.5")

    low_threshold, low_eval = _retune_dir(out, tmp_path, "model_low", tune_csv, 0.05)
    high_threshold, high_eval = _retune_dir(out, tmp_path, "model_high", tune_csv, 0.95)

    assert high_threshold >= low_threshold
    assert high_eval["overall"]["coverage"] <= low_eval["overall"]["coverage"]


def test_meta_and_calibrator_change_other_artifacts_untouched(tmp_path):
    out, _, tune_csv, n_tune = _train_model(tmp_path)
    before = _file_hashes(out)

    _run(
        tune_cli,
        ["tune", "--model", out, "--input", tune_csv, "--target-precision", "0.9"],
    )
    after = _file_hashes(out)

    for changed in ("calibrator.npz", "meta.json", "evaluation.json", "model_card.md"):
        assert before[changed] != after[changed], f"{changed} should have changed"

    # everything else (encoder/, dense.npz, lexical.npz/.json, fusion file) untouched
    untouched = set(before) - {"calibrator.npz", "meta.json", "evaluation.json", "model_card.md"}
    assert untouched, "expected other artifacts to exist"
    for path in untouched:
        assert before[path] == after[path], f"{path} should be byte-identical after retune"

    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["retunes"][0]["n_items"] == n_tune


def test_reloaded_pipeline_uses_new_thresholds(tmp_path):
    from text_classifier import InferencePipeline

    out, _, tune_csv, _ = _train_model(tmp_path)
    _run(
        tune_cli,
        [
            "tune",
            "--model",
            out,
            "--input",
            tune_csv,
            "--target-precision",
            "0.95",
            "--per-class-min-support",
            "3",
        ],
    )

    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)

    pipeline = InferencePipeline.from_directory(out)
    assert pipeline.artifacts.abstention.global_threshold == pytest.approx(
        meta["abstention"]["global_threshold"]
    )


def test_dry_run_leaves_directory_byte_identical(tmp_path):
    out, _, tune_csv, _ = _train_model(tmp_path)
    before = _file_hashes(out)

    _run(
        tune_cli,
        [
            "tune",
            "--model",
            out,
            "--input",
            tune_csv,
            "--target-precision",
            "0.95",
            "--dry-run",
        ],
    )

    after = _file_hashes(out)
    assert before == after


def test_overlap_warning_fires_when_tune_set_is_the_training_set(tmp_path, caplog):
    import logging

    out, items_csv, _, _ = _train_model(tmp_path)
    with caplog.at_level(logging.WARNING):
        _run(
            tune_cli,
            ["tune", "--model", out, "--input", items_csv, "--target-precision", "0.9"],
        )
    assert any("already in the deployed index" in r.message for r in caplog.records)


def test_no_overlap_warning_for_a_genuinely_fresh_set(tmp_path, caplog):
    import logging

    out, _, tune_csv, _ = _train_model(tmp_path)
    with caplog.at_level(logging.WARNING):
        _run(
            tune_cli,
            ["tune", "--model", out, "--input", tune_csv, "--target-precision", "0.9"],
        )
    assert not any("already in the deployed index" in r.message for r in caplog.records)


def test_unknown_label_is_a_clear_error(tmp_path):
    out, _, _, _ = _train_model(tmp_path)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["some words here"], "label": ["NOT_A_REAL_CLASS"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(SystemExit) as exc:
        _run(
            tune_cli,
            ["tune", "--model", out, "--input", str(bad_csv), "--target-precision", "0.9"],
        )
    assert "not in the model's label space" in str(exc.value)
