"""Integration tests for InferencePipeline.importance_report and the
text-classifier-importance CLI. Runs fully offline via the torch-free tfidf
encoder.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.train as train_cli
from text_classifier import PipelineConfig, TrainingPipeline
from text_classifier.application.inference import InferencePipeline
from text_classifier.cli import importance as importance_cli
from text_classifier.datasets import make_synthetic
from text_classifier.infrastructure import HashingEncoder


@pytest.fixture(scope="module")
def pipeline_and_items():
    label_space, items = make_synthetic(n_classes=8, per_class=20, seed=11)
    cfg = PipelineConfig(candidate_top_n=6)
    cfg.encoder.kind = "hashing"
    cfg.training.n_folds = 3
    cfg.training.target_precision = 0.5
    cfg.training.per_class_min_support = 1
    cfg.retrieval.k_neighbors = 10
    artifacts, _ = TrainingPipeline(cfg, shared_encoder=HashingEncoder(dim=64)).run(
        items, label_space
    )
    return InferencePipeline(artifacts), items[:12]


@pytest.fixture
def pipeline(pipeline_and_items) -> InferencePipeline:
    return pipeline_and_items[0]


@pytest.fixture
def items(pipeline_and_items):
    return pipeline_and_items[1]


def test_importance_report_shape(pipeline, items):
    texts = [it.text for it in items]
    labels = [it.label for it in items]
    report = pipeline.importance_report(texts, labels)

    assert set(report) == {"importance", "ablation"}
    baseline = report["ablation"]["baseline"]
    assert baseline["n_items"] > 0
    assert 0.0 <= baseline["coverage"] <= 1.0

    # XGBoost (the default fusion backend) supports predict_contribs.
    assert report["importance"] is not None
    shares = [r["share"] for r in report["importance"]]
    assert pytest.approx(sum(shares), abs=1e-6) == 1.0
    assert all(r["mean_abs_contribution"] >= 0 for r in report["importance"])
    # sorted descending
    contribs = [r["mean_abs_contribution"] for r in report["importance"]]
    assert contribs == sorted(contribs, reverse=True)

    ablations = report["ablation"]["ablations"]
    assert len(ablations) > 0
    for row in ablations:
        assert row["n_rows_masked"] > 0
        assert -1.0 <= row["delta_accuracy_if_no_abstain"] <= 1.0
    # sorted ascending by damage (most negative first)
    deltas = [r["delta_accuracy_if_no_abstain"] for r in ablations]
    assert deltas == sorted(deltas)


def test_importance_report_empty_input(pipeline):
    report = pipeline.importance_report([], [])
    assert report["importance"] is None
    assert report["ablation"]["baseline"]["n_items"] == 0
    assert report["ablation"]["ablations"] == []


def test_importance_report_rejects_unknown_label(pipeline, items):
    texts = [items[0].text]
    with pytest.raises(KeyError):
        pipeline.importance_report(texts, ["NOT_A_REAL_CLASS"])


def _write_csvs(tmp_path) -> tuple[str, str]:
    label_space, items = make_synthetic(n_classes=4, per_class=9, seed=29)
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


def test_importance_cli_writes_report(tmp_path, capsys):
    out, items_csv = _train_model(tmp_path)
    report_path = str(tmp_path / "importance.json")
    _run(
        importance_cli,
        ["importance", "--model", out, "--input", items_csv, "--output", report_path],
    )

    printed = capsys.readouterr().out
    assert "baseline" in printed
    assert "ablation" in printed

    with open(report_path) as fh:
        report = json.load(fh)
    assert set(report) == {"importance", "ablation"}
    assert report["ablation"]["baseline"]["n_items"] > 0


def test_importance_cli_rejects_unknown_labels(tmp_path):
    out, _ = _train_model(tmp_path)
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["some words here"], "label": ["NOT_A_REAL_CLASS"]}).to_csv(
        bad_csv, index=False
    )
    with pytest.raises(SystemExit) as exc:
        _run(importance_cli, ["importance", "--model", out, "--input", str(bad_csv)])
    assert "not in model's label space" in str(exc.value)
