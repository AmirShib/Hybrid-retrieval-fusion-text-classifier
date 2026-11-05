"""T96 training-data cleanup: detection rules, actions, safety guards, the
review file, and the pipeline wiring."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from tests._doubles import make_synthetic
from text_classifier.application.cleanup import (
    REVIEW_COLUMNS,
    find_junk,
    find_label_conflicts,
    run_cleanup,
    write_review_file,
)
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import CleanupConfig, PipelineConfig, TrainingConfig
from text_classifier.domain import ClassDefinition, LabeledItem, LabelSpace

LS = LabelSpace([ClassDefinition(k, k) for k in ("a", "b", "c")])


def _items(rows):
    return [LabeledItem(text, label) for text, label in rows]


def test_junk_detection():
    items = _items([("Rejected", "a"), ("12345 ***", "a"), ("שמפו", "a"), ("ok text", "b")])
    assert [i for i, _ in find_junk(items)] == [0, 1]


def test_conflicts_majority_and_tie():
    items = _items([("x y", "a"), ("x  y", "a"), ("x y", "b"), ("z", "a"), ("z", "b")])
    c = find_label_conflicts(items)
    assert c[2][0] == "a" and c[0][0] == "a"  # whitespace-normalized match
    assert c[3][0] is None and c[4][0] is None  # tie -> no majority


def test_actions_and_class_floor():
    rows = [(f"a{i}", "a") for i in range(6)] + [(f"b{i}", "b") for i in range(3)]
    rows += [("Rejected", "a"), ("Rejected", "b"), ("dup", "a"), ("dup", "a"), ("dup", "b")]
    # 'b' has 5 rows (3 + placeholder + dup); with a floor of 5 nothing may leave it.
    res = run_cleanup(_items(rows), LS, CleanupConfig(enabled=True), min_per_class=5)
    rev = res.review.set_index("row")
    assert rev.loc[9, "action"] == "dropped"  # the 'a' placeholder: 'a' stays above 5
    for row in (10, 13):  # dropping / relabelling these would leave 'b' with 4 < 5
        assert rev.loc[row, "action"] == "kept" and "would fall below" in rev.loc[row, "reason"]
    assert [it.label for it in res.items].count("b") == 5
    assert list(res.review.columns) == REVIEW_COLUMNS
    assert res.review["issue"].tolist()[:2] == ["junk", "junk"]  # most actionable first


def test_conflict_minority_takes_the_majority_label():
    rows = [("dup", "a"), ("dup", "a"), ("dup", "b"), ("x", "b"), ("y", "b")]
    res = run_cleanup(_items(rows), LS, CleanupConfig(enabled=True), min_per_class=1)
    rev = res.review.set_index("row")
    assert rev.loc[2, "action"] == "relabeled" and rev.loc[2, "suggested_label"] == "a"
    assert [it.label for it in res.items] == ["a", "a", "a", "b", "b"]
    keep = run_cleanup(
        _items(rows), LS, CleanupConfig(enabled=True, conflict_action="keep"), min_per_class=1
    )
    assert [it.label for it in keep.items] == ["a", "a", "b", "b", "b"]


def _oof(rows):
    """Hand-built out-of-fold frame: per item, (candidate, is_true, signal, knn flag)."""
    out = []
    for item, cands in rows.items():
        for cand, true, sig, knn in cands:
            out.append(
                {
                    "item_id": item,
                    "candidate": cand,
                    "is_true": true,
                    "f": sig,
                    "is_d_knn_top1": knn,
                    "is_b_knn_top1": 0.0,
                }
            )
    return pd.DataFrame(out)


def test_likely_mislabel_is_flagged_but_kept_by_default(monkeypatch):
    import text_classifier.application.cleanup as cu

    items = _items([(f"t{i}", "a") for i in range(4)])
    oof = _oof({i: [(0, 1, 0.0, 0.0), (1, 0, 0.0, 1.0)] for i in range(4)})
    shares = oof[["item_id", "candidate", "is_true", "is_d_knn_top1", "is_b_knn_top1"]].copy()
    shares["share"] = np.where(shares["is_true"] == 1, 0.02, 0.98)  # confident in class 1
    monkeypatch.setattr(cu, "honest_shares", lambda *a, **k: shares)
    res = run_cleanup(
        items, LS, CleanupConfig(enabled=True), min_per_class=1, oof=oof, feature_names=["f"]
    )
    assert set(res.review["issue"]) == {"likely_mislabel"} and set(res.review["action"]) == {"kept"}
    assert [it.label for it in res.items] == ["a"] * 4
    relabel = run_cleanup(
        items,
        LS,
        CleanupConfig(enabled=True, mislabel_action="relabel"),
        min_per_class=1,
        oof=oof,
        feature_names=["f"],
    )
    # Three move to 'b'; the fourth would empty class 'a', so the floor keeps it.
    assert [it.label for it in relabel.items] == ["b", "b", "b", "a"]


def test_review_file_is_utf8_sig(tmp_path):
    df = pd.DataFrame([{c: "" for c in REVIEW_COLUMNS} | {"text": "שמפו", "row": 0}])
    path = write_review_file(df, str(tmp_path), "s.csv")
    assert open(path, "rb").read(3) == b"\xef\xbb\xbf"
    assert pd.read_csv(path, encoding="utf-8-sig")["text"].iloc[0] == "שמפו"


def test_pipeline_cleanup_round(tmp_path):
    ls, items = make_synthetic(n_classes=4, per_class=15, seed=4)
    items = list(items) + [LabeledItem("Rejected", items[0].label)]
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.cleanup.enabled = True
    out = str(tmp_path / "m")
    pipe = TrainingPipeline(cfg)
    pipe.run(items, ls, output_dir=out)
    rev = pd.read_csv(os.path.join(out, "suspicious_rows.csv"), encoding="utf-8-sig")
    assert (rev["issue"] == "junk").sum() == 1
    ev = json.load(open(os.path.join(out, "evaluation.json")))
    assert ev["cleanup"]["rows_out"] == len(items) - 1 and ev["cleanup"]["model_checks"] is True
    assert "## Training-data cleanup" in open(os.path.join(out, "model_card.md")).read()
    assert pipe.cleanup_result is not None and len(pipe.cleanup_result.items) == len(items) - 1


def test_cleanup_off_by_default(tmp_path):
    assert PipelineConfig().cleanup.enabled is False
