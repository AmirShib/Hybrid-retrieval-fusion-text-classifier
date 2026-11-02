"""T33 — every post-training CLI path on a model that carries a cross-encoder.

Training and inference with the cross-encoder are covered in
``tests/unit/test_cross_encoder_backend.py``; this checks the operations people
run on a *deployed* model -- evaluate, tune, importance, update (new examples and
a brand-new class) -- still work when the model's signals include the second-stage
reranker, and that its saved weights survive each rewrite. Offline: a tiny local
BERT cross-encoder. Skipped where the ``sentence-transformers`` extra is absent.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from text_classifier.application.inference import InferencePipeline  # noqa: E402
from text_classifier.application.training import TrainingPipeline  # noqa: E402
from text_classifier.cli import evaluate as evaluate_cli  # noqa: E402
from text_classifier.cli import importance as importance_cli  # noqa: E402
from text_classifier.cli import tune as tune_cli  # noqa: E402
from text_classifier.cli import update as update_cli  # noqa: E402
from tests.unit.test_cross_encoder_backend import (  # noqa: E402
    _dataset,
    _pipeline_config,
    _write_tiny_cross_encoder,
)

CE_WEIGHTS = os.path.join("signals", "cross-encoder", "model.safetensors")


def _run(module, argv) -> None:
    with patch.object(sys, "argv", argv):
        module.main()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A model trained with a fine-tuned cross-encoder, plus a labeled CSV."""
    root = tmp_path_factory.mktemp("ce-cli")
    ls, items = _dataset()
    words = {w for it in items for w in it.text.split()} | {
        w for d in ls.descriptions for w in d.split()
    }
    words |= {"brand", "new", "class", "zzz1", "zzz2"}
    ce_path = _write_tiny_cross_encoder(str(root / "tiny-ce"), words)
    out = str(root / "model")
    TrainingPipeline(_pipeline_config(ce_path, train=True)).run(items, ls, output_dir=out)
    labeled = str(root / "labeled.csv")
    pd.DataFrame({"text": [i.text for i in items], "label": [i.label for i in items]}).to_csv(
        labeled, index=False
    )
    return out, labeled, root


def _ce_columns(model: str):
    with open(os.path.join(model, "meta.json")) as fh:
        return [c for c in json.load(fh)["feature_names"] if c.startswith("ce_")]


def test_evaluate(model_dir):
    out, labeled, root = model_dir
    report = str(root / "eval.json")
    _run(evaluate_cli, ["evaluate", "--model", out, "--input", labeled, "--output", report])
    with open(report) as fh:
        assert json.load(fh)["overall"]["n_items"] > 0


def test_tune_keeps_the_cross_encoder(model_dir):
    """tune rewrites only the decision layer, in place; the cross-encoder's
    weights must come through untouched and the model must still load."""
    import shutil

    out, labeled, root = model_dir
    tuned = str(root / "tuned")
    shutil.copytree(out, tuned)
    before = open(os.path.join(tuned, CE_WEIGHTS), "rb").read()
    _run(tune_cli, ["tune", "--model", tuned, "--input", labeled, "--target-precision", "0.6"])
    assert open(os.path.join(tuned, CE_WEIGHTS), "rb").read() == before
    assert _ce_columns(tuned) == _ce_columns(out)
    assert InferencePipeline.from_directory(tuned).predict(["w76 w21 w412"])


def test_importance_covers_the_cross_encoder_columns(model_dir):
    out, labeled, root = model_dir
    report = str(root / "importance.json")
    _run(importance_cli, ["importance", "--model", out, "--input", labeled, "--output", report])
    with open(report) as fh:
        data = json.load(fh)
    ce_cols = set(_ce_columns(out))
    assert ce_cols <= {row["feature"] for row in data["importance"]}
    # Ablation masks a column to NaN, so it skips columns that are already all NaN.
    # Here those are the "neg" document's: these synthetic classes carry no
    # exclusions or sibling notes. Every cross-encoder column with data is ablated.
    explain = InferencePipeline.from_directory(out).explain(pd.read_csv(labeled)["text"].tolist())
    with_data = {c for c in ce_cols if explain[c].notna().any()}
    assert "ce_pos" in with_data
    assert with_data <= {row["feature"] for row in data["ablation"]["ablations"]}


def test_update_with_new_examples_keeps_the_cross_encoder(model_dir):
    out, labeled, root = model_dir
    new_items = str(root / "new_items.csv")
    df = pd.read_csv(labeled).head(4)
    df.assign(text=df["text"] + " zzz1").to_csv(new_items, index=False)
    updated = str(root / "updated-items")
    _run(update_cli, ["update", "--model", out, "--out", updated, "--items", new_items])
    assert os.path.exists(os.path.join(updated, CE_WEIGHTS))
    assert InferencePipeline.from_directory(updated).predict(["w76 w21 zzz1"])


def test_update_with_a_new_class_is_scored_by_the_cross_encoder(model_dir):
    """The provider reads class text per chunk, so a class added after training is
    reranked like any other -- its ce_* columns are computed, not left NaN."""
    out, _, root = model_dir
    space = InferencePipeline.from_directory(out).artifacts.label_space
    classes = str(root / "new_classes.csv")  # update takes the full taxonomy: old + new
    pd.DataFrame(
        {
            "key": list(space.keys) + ["NEW"],
            "description": list(space.descriptions) + ["brand new class zzz2"],
        }
    ).to_csv(classes, index=False)
    updated = str(root / "updated-class")
    _run(update_cli, ["update", "--model", out, "--out", updated, "--classes", classes])
    pipe = InferencePipeline.from_directory(updated)
    assert "NEW" in pipe.artifacts.label_space.keys
    frame = pipe.explain(["brand new class zzz2"])
    new_rows = frame[frame["candidate_key"] == "NEW"]
    assert not new_rows.empty and new_rows["ce_pos"].notna().all()
