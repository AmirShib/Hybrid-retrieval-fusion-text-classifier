"""evaluation.json / model_card.md describe *every* configured signal.

Regression for: per-signal diagnostics and "which signals matter" covered only the
built-in dense + BM25 signals; the manifest named the base cross-encoder, not
what was deployed."""

from __future__ import annotations

import json
import os

import pytest

from tests._doubles import make_synthetic
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, TrainingConfig

ALL_SIGNALS = ["dense", "lexical", "splade", "cross-encoder"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    cfg = PipelineConfig()
    cfg.signals = list(ALL_SIGNALS)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    ls, items = make_synthetic(n_classes=5, per_class=14, seed=3)
    out = str(tmp_path_factory.mktemp("m") / "model")
    TrainingPipeline(cfg).run(items, ls, output_dir=out)
    return out, items


def _load(model_dir):
    out, _ = model_dir
    with open(os.path.join(out, "evaluation.json")) as fh:
        ev = json.load(fh)
    with open(os.path.join(out, "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(out, "model_card.md")) as fh:
        card = fh.read()
    return ev, meta, card


def test_signal_report_covers_optional_and_reranker_signals(model_dir):
    ev, _, _ = _load(model_dir)
    rep = ev["signal_report"]
    names = {e["signal"]: e for e in rep["per_signal"]}
    for s in [
        "dense_knn",
        "bm25_knn",
        "splade_description",
        "splade_knn",
        "cross_encoder_pos",
        "cross_encoder_neg",
    ]:
        assert s in names, s
    assert names["cross_encoder_pos"]["round"] == 2 and names["splade_knn"]["round"] == 1
    assert rep["skipped_signals"] == []
    assert rep["agreement"]["n_voting_signals"] == 7  # 5 built-in + 2 SPLADE


def test_signal_contributions_cover_every_feature_and_signal(model_dir):
    ev, meta, _ = _load(model_dir)
    c = ev["signal_contributions"]
    assert {g["signal"] for g in c["by_signal"]} >= set(ALL_SIGNALS)
    assert sum(g["n_features"] for g in c["by_signal"]) == len(meta["feature_names"])
    assert sum(g["share"] for g in c["by_signal"]) == pytest.approx(1.0)
    assert {f["feature"] for f in c["all_features"]} == set(meta["feature_names"])
    assert c["least_used_features"] and c["top_features"]


def test_manifest_and_card_say_what_was_deployed(model_dir):
    ev, _, card = _load(model_dir)
    models = ev["manifest"]["models"]
    assert set(models["signals"]) == set(ALL_SIGNALS)
    ce = models["signals"]["cross-encoder"]
    assert ce["round"] == 2 and ce["deployed_weights"] == "signals/cross-encoder/"
    assert ce["trained_on_this_data"] is False
    for heading in [
        "## Models",
        "## Per-signal diagnostics",
        "## Which signals drive the fusion model's decisions",
    ]:
        assert heading in card
    assert "splade_knn" in card and "(reranker)" in card
    assert "up to 7 = all disagree" in card


def test_internal_raw_score_never_leaks_into_outputs(model_dir):
    out, items = model_dir
    pipe = InferencePipeline.from_directory(out)
    texts = [it.text for it in items[:6]]
    assert "raw_score" not in pipe.explain(texts).columns
    recs = pipe.explain_records(texts)
    assert "raw_score" not in json.dumps(recs, default=str)
