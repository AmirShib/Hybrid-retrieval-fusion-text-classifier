"""T25 — CLI-level tests for scripts/train.py encoder-kind selection.

Runs fully offline: the tfidf path is torch-free, so the train CLI trains and
saves a model directory without sentence-transformers installed.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.train as train_cli
from tests._doubles import make_synthetic


def _write_csvs(tmp_path) -> tuple[str, str]:
    label_space, items = make_synthetic(n_classes=4, per_class=9, seed=23)
    items_csv = tmp_path / "items.csv"
    classes_csv = tmp_path / "classes.csv"
    pd.DataFrame(
        {"text": [it.text for it in items], "label": [it.label for it in items]}
    ).to_csv(items_csv, index=False)
    pd.DataFrame(
        {"key": label_space.keys, "description": label_space.descriptions}
    ).to_csv(classes_csv, index=False)
    return str(items_csv), str(classes_csv)


def _run(argv) -> None:
    with patch.object(sys, "argv", argv):
        train_cli.main()


def test_train_cli_tfidf_trains_offline(tmp_path):
    """--encoder-kind tfidf trains end-to-end and writes a model dir (no torch)."""
    items_csv, classes_csv = _write_csvs(tmp_path)
    out = str(tmp_path / "model")
    _run(["train", "--items", items_csv, "--classes", classes_csv, "--out", out,
          "--encoder-kind", "tfidf", "--folds", "3",
          "--target-precision", "0.5", "--candidate-top-n", "8"])

    assert os.path.isfile(os.path.join(out, "meta.json"))
    with open(os.path.join(out, "meta.json")) as fh:
        assert json.load(fh)["components"]["encoder"] == "tfidf"


def test_train_cli_unknown_encoder_kind_exits(tmp_path, capsys):
    """An unknown --encoder-kind fails fast (non-zero) and lists registered kinds."""
    items_csv, classes_csv = _write_csvs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(["train", "--items", items_csv, "--classes", classes_csv,
              "--out", str(tmp_path / "model"), "--encoder-kind", "bogus"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "tfidf" in err and "sentence-transformers" in err


def test_train_cli_default_encoder_kind_is_sentence_transformers(tmp_path):
    """Omitting --encoder-kind keeps the default, so existing invocations are
    unchanged. (Validation only; we don't build the ST encoder here — that needs
    torch — but argparse must carry the default through.)"""
    parser_argv = ["train", "--items", "i.csv", "--classes", "c.csv", "--out", "o"]
    with patch.object(sys, "argv", parser_argv):
        # parse just the args by short-circuiting before any file IO would run.
        import argparse

        captured = {}
        real_parse = argparse.ArgumentParser.parse_args

        def _capture(self, *a, **k):
            ns = real_parse(self, *a, **k)
            captured["kind"] = ns.encoder_kind
            raise SystemExit(0)  # stop before reading CSVs

        with patch.object(argparse.ArgumentParser, "parse_args", _capture):
            with pytest.raises(SystemExit):
                train_cli.main()
    assert captured["kind"] == "sentence-transformers"
