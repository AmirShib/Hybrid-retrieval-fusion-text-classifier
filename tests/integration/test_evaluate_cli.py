"""Integration tests for evaluation persistence and the evaluate CLI.

Runs fully offline via the torch-free tfidf encoder: train writes the model plus
`evaluation.json` and `model_card.md`, and the evaluate CLI scores a labeled CSV
and emits a JSON report.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.train as train_cli
from text_classifier.cli import evaluate as evaluate_cli
from text_classifier.datasets import make_synthetic


def _write_csvs(tmp_path) -> tuple[str, str]:
    label_space, items = make_synthetic(n_classes=4, per_class=9, seed=23)
    items_csv = tmp_path / "items.csv"
    classes_csv = tmp_path / "classes.csv"
    pd.DataFrame({"text": [it.text for it in items], "label": [it.label for it in items]}).to_csv(
        items_csv, index=False
    )
    pd.DataFrame({"key": label_space.keys, "description": label_space.descriptions}).to_csv(
        classes_csv, index=False
    )
    return str(items_csv), str(classes_csv)


def _run(module, argv) -> None:
    with patch.object(sys, "argv", argv):
        module.main()


def _train_model(tmp_path) -> tuple[str, str]:
    items_csv, classes_csv = _write_csvs(tmp_path)
    out = str(tmp_path / "model")
    _run(
        train_cli,
        [
            "train",
            "--items",
            items_csv,
            "--classes",
            classes_csv,
            "--out",
            out,
            "--encoder-kind",
            "tfidf",
            "--folds",
            "3",
            "--target-precision",
            "0.5",
            "--candidate-top-n",
            "8",
            "--k-neighbors",
            "10",
        ],
    )
    return out, items_csv


def test_training_writes_evaluation_and_model_card(tmp_path):
    out, _ = _train_model(tmp_path)
    assert os.path.isfile(os.path.join(out, "evaluation.json"))
    assert os.path.isfile(os.path.join(out, "model_card.md"))

    with open(os.path.join(out, "evaluation.json")) as fh:
        report = json.load(fh)
    assert set(report) >= {
        "manifest",
        "overall",
        "calibration",
        "risk_coverage_curve",
        "per_class",
        "abstention",
    }
    assert report["manifest"]["package_version"]
    assert report["manifest"]["n_classes"] == 4
    assert report["overall"]["n_items"] > 0
    # calibration block is populated
    assert "expected_calibration_error" in report["calibration"]

    card = open(os.path.join(out, "model_card.md")).read()
    assert "Model card" in card and "Coverage" in card


def test_meta_records_package_version(tmp_path):
    out, _ = _train_model(tmp_path)
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta.get("package_version")


def test_evaluate_cli_scores_and_writes_report(tmp_path, capsys):
    out, items_csv = _train_model(tmp_path)
    report_path = str(tmp_path / "report.json")
    _run(evaluate_cli, ["eval", "--model", out, "--input", items_csv, "--output", report_path])

    printed = capsys.readouterr().out
    assert "coverage" in printed and "brier score" in printed

    assert os.path.isfile(report_path)
    with open(report_path) as fh:
        report = json.load(fh)
    assert set(report) >= {"manifest", "overall", "calibration", "per_class"}
    assert report["overall"]["n_items"] > 0


def test_evaluate_cli_rejects_unknown_labels(tmp_path):
    out, _ = _train_model(tmp_path)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["some words here"], "label": ["NOT_A_REAL_CLASS"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(SystemExit) as exc:
        _run(evaluate_cli, ["eval", "--model", out, "--input", str(bad_csv)])
    assert "not in the model" in str(exc.value)


def test_evaluate_cli_missing_column_is_friendly(tmp_path):
    out, _ = _train_model(tmp_path)
    bad_csv = tmp_path / "nolabel.csv"
    pd.DataFrame({"text": ["some words here"]}).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit) as exc:
        _run(evaluate_cli, ["eval", "--model", out, "--input", str(bad_csv)])
    assert "missing required column" in str(exc.value)
