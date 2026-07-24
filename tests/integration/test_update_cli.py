"""Integration tests for the update CLI (T68): add classes and/or labeled
examples to a deployed model without retraining the fusion model or
calibrator.

Runs fully offline via the torch-free tfidf encoder, following the same
train-via-CLI fixture pattern as ``test_tune_cli.py`` / ``test_evaluate_cli.py``.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import scripts.train as train_cli
from text_classifier import InferencePipeline
from text_classifier.cli import update as update_cli
from text_classifier.datasets import make_synthetic


def _split_items(items, holdout_per_class=3):
    """Per-class split: all but the last `holdout_per_class` items train the
    model; the held-out ones are genuinely new (never seen by the index)."""
    by_label = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)
    train, held_out = [], []
    for group in by_label.values():
        cut = max(1, len(group) - holdout_per_class)
        train.extend(group[:cut])
        held_out.extend(group[cut:])
    return train, held_out


def _write_items_csv(path, items) -> str:
    pd.DataFrame({"text": [it.text for it in items], "label": [it.label for it in items]}).to_csv(
        path, index=False
    )
    return str(path)


def _run(module, argv) -> None:
    with patch.object(sys, "argv", argv):
        module.main()


def _train_model(tmp_path, store_corpus=True):
    """Train a model on one split of a synthetic generation; return
    (model_dir, train_items_csv, held_out_items, classes_csv, label_space)."""
    label_space, items = make_synthetic(n_classes=4, per_class=24, seed=23)
    train_items, held_out = _split_items(items, holdout_per_class=3)

    items_csv = _write_items_csv(tmp_path / "items.csv", train_items)
    classes_csv = tmp_path / "classes.csv"
    pd.DataFrame({"key": label_space.keys, "description": label_space.descriptions}).to_csv(
        classes_csv, index=False
    )
    out = str(tmp_path / "model")
    argv = [
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
        "0.5",
        "--candidate-top-n",
        "8",
        "--k-neighbors",
        "10",
    ]
    if not store_corpus:
        argv.append("--no-store-corpus")
    _run(train_cli, argv)
    return out, items_csv, held_out, str(classes_csv), label_space


def _file_hashes(directory: str) -> dict:
    hashes = {}
    for path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                hashes[os.path.relpath(path, directory)] = hashlib.sha256(fh.read()).hexdigest()
    return hashes


def test_store_corpus_default_on_and_opt_out(tmp_path):
    out, items_csv, _, _, _ = _train_model(tmp_path, store_corpus=True)
    assert os.path.isfile(os.path.join(out, "corpus.jsonl.gz"))
    with gzip.open(os.path.join(out, "corpus.jsonl.gz"), "rt") as fh:
        rows = [json.loads(line) for line in fh]
    train_df = pd.read_csv(items_csv)
    assert len(rows) == len(train_df)

    nocorpus_root = tmp_path / "nocorpus"
    nocorpus_root.mkdir()
    out2, _, _, _, _ = _train_model(nocorpus_root, store_corpus=False)
    assert not os.path.isfile(os.path.join(out2, "corpus.jsonl.gz"))


def test_add_new_class_with_examples_predicts_correctly(tmp_path):
    out, _, held_out, _, label_space = _train_model(tmp_path)

    # tfidf's vocabulary is frozen to whatever tokens appeared in the training
    # corpus (synthetic "wNNN" tokens); real English words would be entirely
    # out-of-vocabulary and encode to an all-zero vector, so build the new
    # class's texts from tokens the fitted vectorizer actually knows.
    vectorizer = InferencePipeline.from_directory(out).artifacts.encoder._vectorizer
    vocab_tokens = sorted(vectorizer.vocabulary_.keys())
    assert len(vocab_tokens) >= 6
    new_tokens = vocab_tokens[:6]

    new_classes_csv = tmp_path / "new_classes.csv"
    pd.DataFrame(
        {
            "key": label_space.keys + ["WIDGETS"],
            "description": label_space.descriptions
            + [f"a brand new category about {' '.join(new_tokens)}"],
        }
    ).to_csv(new_classes_csv, index=False)

    new_items_csv = tmp_path / "widgets_items.csv"
    widget_texts = [
        " ".join(new_tokens),
        " ".join(new_tokens[::-1]),
        " ".join(new_tokens[:4]),
        " ".join(new_tokens[2:]),
    ]
    pd.DataFrame({"text": widget_texts, "label": ["WIDGETS"] * len(widget_texts)}).to_csv(
        new_items_csv, index=False
    )

    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        [
            "update",
            "--model",
            out,
            "--out",
            out_updated,
            "--classes",
            str(new_classes_csv),
            "--items",
            str(new_items_csv),
        ],
    )

    pipeline = InferencePipeline.from_directory(out_updated)
    preds = pipeline.predict([" ".join(new_tokens)])
    assert preds[0].top_key == "WIDGETS"
    assert preds[0].confidence > 0.3  # real example support, not description-only


def test_index_stability_untouched_classes_unchanged(tmp_path):
    out, items_csv, held_out, _, _ = _train_model(tmp_path)

    only_first_class = [it for it in held_out if it.label == held_out[0].label]
    target_label = only_first_class[0].label
    new_items_csv = _write_items_csv(tmp_path / "new_items_one_class.csv", only_first_class)

    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        ["update", "--model", out, "--out", out_updated, "--items", new_items_csv],
    )

    train_df = pd.read_csv(items_csv)
    untouched_texts = train_df[train_df["label"] != target_label]["text"].tolist()
    before = InferencePipeline.from_directory(out).predict(untouched_texts)
    after = InferencePipeline.from_directory(out_updated).predict(untouched_texts)
    for b, a in zip(before, after):
        assert b.top_key == a.top_key
        assert b.confidence == pytest.approx(a.confidence, abs=1e-9)


def test_add_examples_to_existing_class_changes_freq_keeps_fusion_identical(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    only_first_class = [it for it in held_out if it.label == held_out[0].label]
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", only_first_class)

    before = _file_hashes(out)
    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        ["update", "--model", out, "--out", out_updated, "--items", new_items_csv],
    )
    after = _file_hashes(out_updated)

    # fusion file is byte-identical to the source model (no refit).
    assert before["fusion.json"] == after["fusion.json"]
    # The calibrator itself is reused verbatim (never refit here), but its pickle
    # is not guaranteed byte-stable across an unrelated reload+re-save round trip
    # (a scipy interp1d inside IsotonicCalibrator isn't); check functional
    # equivalence instead.
    old_cal = InferencePipeline.from_directory(out).artifacts.calibrator
    new_cal = InferencePipeline.from_directory(out_updated).artifacts.calibrator
    probe = np.linspace(0.0, 1.0, 21)
    assert np.allclose(old_cal.transform(probe), new_cal.transform(probe))

    with open(os.path.join(out, "meta.json")) as fh:
        old_meta = json.load(fh)
    with open(os.path.join(out_updated, "meta.json")) as fh:
        new_meta = json.load(fh)
    assert old_meta["abstention"] == new_meta["abstention"]  # unchanged (no --tune-with)

    with gzip.open(os.path.join(out_updated, "corpus.jsonl.gz"), "rt") as fh:
        merged_rows = [json.loads(line) for line in fh]
    with gzip.open(os.path.join(out, "corpus.jsonl.gz"), "rt") as fh:
        old_rows = [json.loads(line) for line in fh]
    assert len(merged_rows) == len(old_rows) + len(only_first_class)


def test_reorder_or_remove_attempt_is_rejected(tmp_path):
    out, _, _, _, label_space = _train_model(tmp_path)
    dropped_csv = tmp_path / "dropped.csv"
    # omit the first class -- an implicit "remove" attempt
    pd.DataFrame(
        {"key": label_space.keys[1:], "description": label_space.descriptions[1:]}
    ).to_csv(dropped_csv, index=False)

    with pytest.raises(SystemExit) as exc:
        _run(
            update_cli,
            [
                "update",
                "--model",
                out,
                "--out",
                str(tmp_path / "model_bad"),
                "--classes",
                str(dropped_csv),
            ],
        )
    assert "retrain" in str(exc.value)


def test_missing_corpus_and_no_base_items_is_actionable(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path, store_corpus=False)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])

    with pytest.raises(SystemExit) as exc:
        _run(
            update_cli,
            [
                "update",
                "--model",
                out,
                "--out",
                str(tmp_path / "model_bad"),
                "--items",
                new_items_csv,
            ],
        )
    msg = str(exc.value)
    assert "--base-items" in msg
    assert "--store-corpus" in msg


def test_base_items_fallback_works_without_persisted_corpus(tmp_path):
    out, items_csv, held_out, _, _ = _train_model(tmp_path, store_corpus=False)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])

    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        [
            "update",
            "--model",
            out,
            "--out",
            out_updated,
            "--items",
            new_items_csv,
            "--base-items",
            items_csv,
        ],
    )
    assert os.path.isfile(os.path.join(out_updated, "corpus.jsonl.gz"))
    # sanity: the updated model still loads and predicts
    InferencePipeline.from_directory(out_updated).predict(["some text"])


def test_updated_dir_loads_via_inference_pipeline(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])
    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        ["update", "--model", out, "--out", out_updated, "--items", new_items_csv],
    )
    pipeline = InferencePipeline.from_directory(out_updated)
    assert pipeline.label_space.size == 4


def test_evaluation_marked_stale_without_tune_with(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])
    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        ["update", "--model", out, "--out", out_updated, "--items", new_items_csv],
    )
    with open(os.path.join(out_updated, "evaluation.json")) as fh:
        evaluation = json.load(fh)
    assert evaluation["stale"] is True
    card = open(os.path.join(out_updated, "model_card.md")).read()
    assert "Stale after update" in card


def test_tune_with_refreshes_evaluation_and_thresholds(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out)
    out_updated = str(tmp_path / "model_updated")
    _run(
        update_cli,
        [
            "update",
            "--model",
            out,
            "--out",
            out_updated,
            "--items",
            new_items_csv,
            "--tune-with",
            new_items_csv,
            "--target-precision",
            "0.6",
            "--per-class-min-support",
            "3",
        ],
    )
    with open(os.path.join(out_updated, "evaluation.json")) as fh:
        evaluation = json.load(fh)
    assert "stale" not in evaluation or not evaluation["stale"]
    assert evaluation["overall"]["n_items"] == len(held_out)


def test_update_provenance_recorded_and_carried_forward(tmp_path):
    out, _, held_out, _, label_space = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])

    out1 = str(tmp_path / "model_v1")
    _run(update_cli, ["update", "--model", out, "--out", out1, "--items", new_items_csv])
    with open(os.path.join(out1, "meta.json")) as fh:
        meta1 = json.load(fh)
    assert len(meta1["updates"]) == 1

    out2 = str(tmp_path / "model_v2")
    new_classes_csv = tmp_path / "extra_class.csv"
    pd.DataFrame(
        {"key": label_space.keys + ["EXTRA"], "description": label_space.descriptions + ["extra class"]}
    ).to_csv(new_classes_csv, index=False)
    _run(update_cli, ["update", "--model", out1, "--out", out2, "--classes", str(new_classes_csv)])
    with open(os.path.join(out2, "meta.json")) as fh:
        meta2 = json.load(fh)
    assert len(meta2["updates"]) == 2  # carried forward + this update's entry


def test_in_place_overwrites_model_dir(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])
    _run(
        update_cli,
        ["update", "--model", out, "--in-place", "--items", new_items_csv],
    )
    with gzip.open(os.path.join(out, "corpus.jsonl.gz"), "rt") as fh:
        rows = [json.loads(line) for line in fh]
    train_count = len(pd.read_csv(os.path.join(tmp_path, "items.csv")))
    assert len(rows) == train_count + 2


def test_requires_out_xor_in_place(tmp_path):
    out, _, held_out, _, _ = _train_model(tmp_path)
    new_items_csv = _write_items_csv(tmp_path / "new_items.csv", held_out[:2])
    with pytest.raises(SystemExit):
        _run(update_cli, ["update", "--model", out, "--items", new_items_csv])
    with pytest.raises(SystemExit):
        _run(
            update_cli,
            [
                "update",
                "--model",
                out,
                "--out",
                str(tmp_path / "x"),
                "--in-place",
                "--items",
                new_items_csv,
            ],
        )


def test_requires_classes_or_items(tmp_path):
    out, _, _, _, _ = _train_model(tmp_path)
    with pytest.raises(SystemExit):
        _run(update_cli, ["update", "--model", out, "--out", str(tmp_path / "x")])
