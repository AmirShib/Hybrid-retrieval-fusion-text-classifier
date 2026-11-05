"""T96 training-data cleanup: every detection rule, action and safety guard, the
review file round trip, the cleanup-pass overrides, rounds, and quarantine."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import text_classifier.application.cleanup as cu
from tests._doubles import make_synthetic
from text_classifier.application.cleanup import (
    REVIEW_COLUMNS,
    find_junk,
    find_label_conflicts,
    read_review_decisions,
    run_cleanup,
    write_review_file,
)
from text_classifier.application.training import TrainingPipeline, _positions
from text_classifier.config import CleanupConfig, PipelineConfig, TrainingConfig
from text_classifier.domain import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.infrastructure import ArtifactRepository

LS = LabelSpace([ClassDefinition(k, k) for k in ("a", "b", "c")])


def _items(rows):
    return [LabeledItem(text, label) for text, label in rows]


# ----------------------------------------------------------- rule-only checks
def test_junk_detection():
    items = _items([("Rejected", "a"), ("12345 ***", "a"), ("שמפו", "a"), ("ok text", "b")])
    assert [i for i, _ in find_junk(items)] == [0, 1]


def test_conflicts_majority_and_tie():
    items = _items([("x y", "a"), ("x  y", "a"), ("x y", "b"), ("z", "a"), ("z", "b")])
    c = find_label_conflicts(items)
    assert c[2][0] == "a" and c[0][0] == "a"  # whitespace-normalized match
    assert c[3][0] is None and c[4][0] is None  # tie -> no majority


def test_class_floor_blocks_automatic_changes():
    rows = [(f"a{i}", "a") for i in range(6)] + [(f"b{i}", "b") for i in range(3)]
    rows += [("Rejected", "a"), ("Rejected", "b"), ("dup", "a"), ("dup", "a"), ("dup", "b")]
    res = run_cleanup(_items(rows), LS, CleanupConfig(enabled=True), min_per_class=5)
    rev = res.review.set_index("row")
    assert rev.loc[9, "action"] == "dropped"
    for row in (10, 13):  # would leave 'b' with 4 < 5 rows
        assert rev.loc[row, "action"] == "kept" and "would fall below" in rev.loc[row, "reason"]
    assert list(res.review.columns) == REVIEW_COLUMNS
    assert res.review["issue"].tolist()[:2] == ["junk", "junk"]


def test_conflict_minority_takes_the_majority_label():
    rows = [("dup", "a"), ("dup", "a"), ("dup", "b"), ("x", "b"), ("y", "b")]
    res = run_cleanup(_items(rows), LS, CleanupConfig(enabled=True), min_per_class=1)
    assert [it.label for it in res.items] == ["a", "a", "a", "b", "b"]
    keep = run_cleanup(
        _items(rows), LS, CleanupConfig(enabled=True, conflict_action="keep"), min_per_class=1
    )
    assert [it.label for it in keep.items] == ["a", "a", "b", "b", "b"]


# ---------------------------------------------------------- model-based checks
def _fake_scores(monkeypatch, per_item):
    """``per_item[i] = (own share, rival share, dense kNN agrees, BM25 kNN agrees)``
    for item i (own label 'a', rival 'b'); None = own label not shortlisted."""

    def fake(oof, names, config, seed=0):
        rows = []
        # Rows not mentioned look clean: own label clearly ahead.
        full = {i: (0.9, 0.1, 0, 0) for i in range(6)} | dict(per_item)
        for i, spec in full.items():
            if spec is None:
                rows.append(
                    dict(
                        item_id=i,
                        candidate=1,
                        is_true=0,
                        share=1.0,
                        is_d_knn_top1=0.0,
                        is_b_knn_top1=0.0,
                    )
                )
                continue
            own, rival, d, b = spec
            rows.append(
                dict(
                    item_id=i,
                    candidate=0,
                    is_true=1,
                    share=own,
                    is_d_knn_top1=0.0,
                    is_b_knn_top1=0.0,
                )
            )
            rows.append(
                dict(
                    item_id=i,
                    candidate=1,
                    is_true=0,
                    share=rival,
                    is_d_knn_top1=float(d),
                    is_b_knn_top1=float(b),
                )
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(cu, "honest_shares", fake)


def _oof_fn(calls=None):
    def fn(current):
        if calls is not None:
            calls.append(len(current))
        return pd.DataFrame({"item_id": [0], "candidate": [0], "is_true": [1]}), ["f"]

    return fn


def _run(items, monkeypatch, per_item, **cfg):
    _fake_scores(monkeypatch, per_item)
    return run_cleanup(
        items, LS, CleanupConfig(enabled=True, **cfg), min_per_class=1, oof_fn=_oof_fn()
    )


A6 = [(f"t{i}", "a") for i in range(6)]


def test_likely_mislabel_kept_by_default_and_each_action(monkeypatch):
    spec = {0: (0.02, 0.98, 1, 0), 1: (0.6, 0.4, 0, 0)}
    res = _run(_items(A6), monkeypatch, spec)
    assert res.review.set_index("row").loc[0, "issue"] == "likely_mislabel"
    assert res.review.set_index("row").loc[0, "action"] == "kept"
    assert res.review.set_index("row").loc[1, "issue"] == "ambiguous"
    assert [
        it.label for it in _run(_items(A6), monkeypatch, spec, mislabel_action="relabel").items
    ][0] == "b"
    assert len(_run(_items(A6), monkeypatch, spec, mislabel_action="drop").items) == 5
    q = _run(_items(A6), monkeypatch, spec, mislabel_action="quarantine")
    assert len(q.items) == 6 and q.quarantined.tolist() == [True] + [False] * 5


def test_knn_agreement_count_and_signal_choice(monkeypatch):
    spec = {0: (0.02, 0.98, 1, 0), 1: (0.02, 0.98, 1, 1)}
    two = _run(_items(A6), monkeypatch, spec, min_knn_agreeing=2)
    assert set(two.review[two.review.issue == "likely_mislabel"]["row"]) == {1}
    bm25_only = _run(_items(A6), monkeypatch, spec, knn_agreement_signals=["lexical"])
    assert set(bm25_only.review[bm25_only.review.issue == "likely_mislabel"]["row"]) == {1}
    off = _run(_items(A6), monkeypatch, {0: (0.02, 0.98, 0, 0)}, require_knn_agreement=False)
    assert off.review["issue"].tolist() == ["likely_mislabel"]


def test_thresholds(monkeypatch):
    spec = {0: (0.08, 0.92, 1, 1)}
    loose = _run(_items(A6), monkeypatch, spec, max_label_share=0.1)
    assert loose.review["issue"].tolist() == ["likely_mislabel"]
    strict = _run(_items(A6), monkeypatch, spec)  # default max_label_share 0.05
    assert "likely_mislabel" not in strict.review["issue"].tolist()


def test_max_changed_fraction_keeps_the_most_confident(monkeypatch):
    spec = {0: (0.01, 0.91, 1, 1), 1: (0.01, 0.99, 1, 1), 2: (0.01, 0.95, 1, 1)}
    res = _run(_items(A6), monkeypatch, spec, mislabel_action="drop", max_changed_fraction=0.34)
    rev = res.review.set_index("row")  # budget = floor(0.34 * 6) = 2
    assert rev.loc[1, "action"] == "dropped" and rev.loc[2, "action"] == "dropped"
    assert rev.loc[0, "action"] == "kept" and "max_changed_fraction" in rev.loc[0, "reason"]


def test_not_shortlisted_and_ambiguous_actions(monkeypatch):
    spec = {0: None, 1: (0.5, 0.5, 0, 0)}
    res = _run(
        _items(A6), monkeypatch, spec, not_shortlisted_action="drop", ambiguous_action="quarantine"
    )
    rev = res.review.set_index("row")
    assert rev.loc[0, "action"] == "dropped" and rev.loc[1, "action"] == "quarantined"


def test_rounds_repeat_until_nothing_changes(monkeypatch):
    calls = []
    _fake_scores(monkeypatch, {0: (0.01, 0.99, 1, 1)})
    res = run_cleanup(
        _items(A6),
        LS,
        CleanupConfig(enabled=True, mislabel_action="drop", rounds=5),
        min_per_class=1,
        oof_fn=_oof_fn(calls),
    )
    # Round 1 drops item 0; round 2's item 0 is a different (locked-out) row set and
    # the fake flags position 0 again -> dropped; it stops once a round changes nothing
    # or the rounds run out. Each round runs exactly one out-of-fold pass.
    assert len(calls) == res.summary["rounds_run"] and res.summary["rounds_run"] >= 2
    assert calls[0] == 6 and calls[1] == 5


# ---------------------------------------------------------------- human review
def test_review_file_round_trip_applies_decisions_first(tmp_path, monkeypatch):
    items = _items(A6 + [("Rejected", "a")])
    _fake_scores(monkeypatch, {0: (0.01, 0.99, 1, 1)})
    first = run_cleanup(items, LS, CleanupConfig(enabled=True), min_per_class=1, oof_fn=_oof_fn())
    path = write_review_file(first.review, str(tmp_path), "s.csv")
    assert open(path, "rb").read(3) == b"\xef\xbb\xbf"
    reviewed = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert "decision" in reviewed.columns
    reviewed.loc[reviewed.text == "t0", "decision"] = "relabel:c"
    reviewed.loc[reviewed.text == "Rejected", "decision"] = "keep"  # a person overrides junk
    reviewed.to_csv(path, index=False, encoding="utf-8-sig")
    decisions = read_review_decisions(path, LS)
    again = run_cleanup(
        items,
        LS,
        CleanupConfig(enabled=True, mislabel_action="drop"),
        min_per_class=1,
        oof_fn=_oof_fn(),
        decisions=decisions,
    )
    labels = {it.text: it.label for it in again.items}
    assert labels["t0"] == "c"  # human relabel won over the automatic drop
    assert "Rejected" in labels  # human keep won over the junk rule
    assert set(again.review[again.review.issue == "reviewed"]["row"]) == {0, 6}


def test_invalid_review_decisions_are_rejected(tmp_path):
    p = str(tmp_path / "r.csv")
    pd.DataFrame(
        {"text": ["x", "y"], "label": ["a", "a"], "decision": ["relabel:zzz", "maybe"]}
    ).to_csv(p, index=False)
    with pytest.raises(ValueError, match="relabel target 'zzz'"):
        read_review_decisions(p, LS)


# --------------------------------------------------------- pipeline integration
def _cfg(**cleanup):
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.cleanup.enabled = True
    for k, v in cleanup.items():
        setattr(cfg.cleanup, k, v)
    return cfg


def test_cleanup_pass_overrides():
    cfg = _cfg(signals=["dense", "lexical"], n_folds=4, train_epochs=1, fine_tune="off")
    cfg.signals = ["dense", "lexical", "splade", "cross-encoder"]
    cfg.retrieval.cross_encoder.train = True
    cfg.encoder.train_epochs = 3
    cfg.training.use_per_fold_encoder = True
    p = TrainingPipeline(cfg)._cleanup_pass_config()
    assert p.signals == ["dense", "lexical"] and p.training.n_folds == 4
    assert p.encoder.train_epochs == 1 and p.training.use_per_fold_encoder is False
    assert p.retrieval.cross_encoder.train is False and p.cleanup.enabled is False
    assert cfg.signals == ["dense", "lexical", "splade", "cross-encoder"]  # original untouched


def test_cleanup_pass_uses_its_own_signals_and_folds(tmp_path, monkeypatch):
    seen = []
    orig = TrainingPipeline._build_oof

    def spy(self, texts, y, label_space):
        seen.append((list(self.cfg.signals), self.cfg.training.n_folds))
        return orig(self, texts, y, label_space)

    monkeypatch.setattr(TrainingPipeline, "_build_oof", spy)
    ls, items = make_synthetic(n_classes=4, per_class=15, seed=4)
    cfg = _cfg(signals=["dense"], n_folds=2)
    TrainingPipeline(cfg).run(items, ls, output_dir=str(tmp_path / "m"))
    assert seen[0] == (["dense"], 2)  # the cleanup pass
    assert seen[1] == (["dense", "lexical"], 3)  # the real training pass


def _with_quarantine_review(tmp_path, items, n):
    p = str(tmp_path / "decisions.csv")
    pd.DataFrame(
        {
            "text": [it.text for it in items[:n]],
            "label": [it.label for it in items[:n]],
            "decision": ["quarantine"] * n,
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")
    return p


@pytest.mark.parametrize(
    "targets, pool_excludes", [(["knn_pool", "fine_tune"], True), (["fine_tune"], False)]
)
def test_quarantine_keeps_rows_out_of_the_pool_only_when_asked(tmp_path, targets, pool_excludes):
    ls, items = make_synthetic(n_classes=4, per_class=15, seed=4)
    items = list(items)
    review = _with_quarantine_review(tmp_path, items, 5)
    out = str(tmp_path / "m")
    cfg = _cfg(review_decisions=review, quarantine_from=targets, detect_mislabels=False)
    art, _ = TrainingPipeline(cfg).run(items, ls, output_dir=out)
    pool = len(art.dense._s.example_labels)
    corpus = ArtifactRepository.load_corpus(out)
    if pool_excludes:
        assert pool == len(items) - 5 and len(corpus) == len(items) - 5
    else:
        assert pool == len(items) and len(corpus) == len(items)
    ev = json.load(open(os.path.join(out, "evaluation.json")))
    assert ev["cleanup"]["quarantined"] == 5 and ev["cleanup"]["rows_out"] == len(items)


def test_positions_helper():
    assert _positions(np.array([0, 2, 5]), np.array([5, 0, 3])).tolist() == [1, -1, 0]


def test_cleanup_off_by_default():
    assert PipelineConfig().cleanup.enabled is False
    PipelineConfig.from_dict({"cleanup": {"mislabel_action": "quarantine"}}).validate()
    with pytest.raises(ValueError, match="cleanup.signals"):
        PipelineConfig.from_dict({"cleanup": {"signals": ["cross-encoder"]}}).validate()
    with pytest.raises(ValueError, match="cleanup.min_knn_agreeing"):
        PipelineConfig.from_dict(
            {"cleanup": {"knn_agreement_signals": ["dense"], "min_knn_agreeing": 2}}
        ).validate()
