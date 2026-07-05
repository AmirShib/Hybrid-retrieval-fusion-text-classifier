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
    pd.DataFrame({"text": [it.text for it in items], "label": [it.label for it in items]}).to_csv(
        items_csv, index=False
    )
    pd.DataFrame({"key": label_space.keys, "description": label_space.descriptions}).to_csv(
        classes_csv, index=False
    )
    return str(items_csv), str(classes_csv)


def _run(argv) -> None:
    with patch.object(sys, "argv", argv):
        train_cli.main()


def test_train_cli_tfidf_trains_offline(tmp_path):
    """--encoder-kind tfidf trains end-to-end and writes a model dir (no torch)."""
    items_csv, classes_csv = _write_csvs(tmp_path)
    out = str(tmp_path / "model")
    _run(
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
        ]
    )

    assert os.path.isfile(os.path.join(out, "meta.json"))
    with open(os.path.join(out, "meta.json")) as fh:
        assert json.load(fh)["components"]["encoder"] == "tfidf"


def test_train_cli_unknown_encoder_kind_exits(tmp_path, capsys):
    """An unknown --encoder-kind fails fast (non-zero) and lists registered kinds."""
    items_csv, classes_csv = _write_csvs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "train",
                "--items",
                items_csv,
                "--classes",
                classes_csv,
                "--out",
                str(tmp_path / "model"),
                "--encoder-kind",
                "bogus",
            ]
        )
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "tfidf" in err and "sentence-transformers" in err


def test_train_cli_default_encoder_kind_is_sentence_transformers(capsys):
    """Omitting --encoder-kind keeps the default, so existing invocations are
    unchanged. --dump-config lets us inspect the resolved PipelineConfig
    without needing --items/--classes/--out or touching torch."""
    _run(["train", "--dump-config"])
    dumped = json.loads(capsys.readouterr().out)
    assert dumped["encoder"]["kind"] == "sentence-transformers"


class TestConfigFileFlag:
    def test_dump_config_reflects_flag_overrides(self, capsys):
        _run(["train", "--dump-config", "--folds", "7", "--candidate-top-n", "3"])
        dumped = json.loads(capsys.readouterr().out)
        assert dumped["training"]["n_folds"] == 7
        assert dumped["candidate_top_n"] == 3

    def test_config_file_sets_a_field_flags_dont_touch(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"fusion": {"kind": "lightgbm"}}))
        _run(["train", "--config", str(cfg_path), "--dump-config"])
        dumped = json.loads(capsys.readouterr().out)
        assert dumped["fusion"]["kind"] == "lightgbm"
        # untouched sections keep their built-in defaults
        assert dumped["encoder"]["kind"] == "sentence-transformers"

    def test_explicit_flag_overrides_config_file(self, tmp_path, capsys):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"training": {"n_folds": 7}}))
        _run(["train", "--config", str(cfg_path), "--dump-config", "--folds", "4"])
        dumped = json.loads(capsys.readouterr().out)
        assert dumped["training"]["n_folds"] == 4

    def test_meta_json_config_block_is_a_valid_config_file(self, tmp_path):
        """A previous run's meta.json 'config' block round-trips as --config input."""
        items_csv, classes_csv = _write_csvs(tmp_path)
        out = str(tmp_path / "model")
        _run(
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
            ]
        )
        with open(os.path.join(out, "meta.json")) as fh:
            meta = json.load(fh)
        cfg_path = tmp_path / "reused_cfg.json"
        cfg_path.write_text(json.dumps(meta["config"]))

        out2 = str(tmp_path / "model2")
        _run(
            [
                "train",
                "--config",
                str(cfg_path),
                "--items",
                items_csv,
                "--classes",
                classes_csv,
                "--out",
                out2,
            ]
        )
        assert os.path.isfile(os.path.join(out2, "meta.json"))

    def test_unknown_key_in_config_file_errors_naming_key_and_section(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"fusion": {"xgb_parms": {}}}))
        with pytest.raises(SystemExit) as exc:
            _run(["train", "--config", str(cfg_path), "--dump-config"])
        assert "xgb_parms" in str(exc.value)
        assert "fusion" in str(exc.value)

    def test_config_file_not_found_errors_clearly(self):
        with pytest.raises(SystemExit) as exc:
            _run(["train", "--config", "/nonexistent/cfg.json", "--dump-config"])
        assert "/nonexistent/cfg.json" in str(exc.value)
