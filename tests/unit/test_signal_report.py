"""Per-signal diagnostics: standalone top-1 accuracy, firing rate, agreement.

`signal_report` reads an (item, candidate) feature table with `item_id`, `is_true`,
and the core signal columns — no encoder/model/index needed — so these assert exact
values from hand-built frames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from text_classifier.application.signal_report import SIGNALS, signal_report

FLAG_COLS = [c["top1"] for c in SIGNALS.values()]


def _row(item_id, is_true, flags, agree, missing=None):
    """One candidate row: `flags` is the set of top1-flag columns set to 1 here."""
    r = {"item_id": item_id, "is_true": is_true, "n_signal_agreement": agree}
    for col in FLAG_COLS:
        r[col] = 1.0 if col in flags else 0.0
    for col in ("d_knn_missing", "b_desc_missing", "b_knn_missing"):
        r[col] = 0.0
    for col, val in (missing or {}).items():
        r[col] = val
    return r


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two items with known per-signal top picks.

    item 0: true class A. dense signals (desc/proto/knn) + bm25_description all
            pick A (correct); bm25_knn picks B (wrong). distinct top picks = 2.
    item 1: true class D. every dense signal + bm25_description pick D (correct);
            bm25_knn fires for no candidate. distinct top picks = 1.
    """
    dense_and_bdesc = {
        "is_d_desc_top1",
        "is_d_proto_top1",
        "is_d_knn_top1",
        "is_b_desc_top1",
    }
    rows = [
        _row(0, 1, dense_and_bdesc, agree=3.0),  # A (true): distinct=2 -> agree=3
        _row(0, 0, {"is_b_knn_top1"}, agree=3.0),  # B: bm25_knn's wrong pick
        _row(0, 0, set(), agree=3.0),  # C
        _row(1, 1, dense_and_bdesc, agree=4.0),  # D (true): distinct=1 -> agree=4
        _row(1, 0, set(), agree=4.0),  # E
    ]
    return pd.DataFrame(rows)


def _by_name(report):
    return {e["signal"]: e for e in report["per_signal"]}


class TestPerSignal:
    def test_dense_and_bm25_description_are_perfect(self, frame):
        by = _by_name(signal_report(frame))
        for name in ("dense_description", "dense_prototype", "dense_knn", "bm25_description"):
            assert by[name]["top1_accuracy"] == pytest.approx(1.0)
            assert by[name]["fired_rate"] == pytest.approx(1.0)
            assert by[name]["top1_precision_when_fired"] == pytest.approx(1.0)

    def test_bm25_knn_fires_once_and_is_wrong(self, frame):
        e = _by_name(signal_report(frame))["bm25_knn"]
        assert e["top1_accuracy"] == pytest.approx(0.0)  # 0 correct over 2 items
        assert e["fired_rate"] == pytest.approx(0.5)  # fired for item 0 only
        assert e["top1_precision_when_fired"] == pytest.approx(0.0)  # 0 of 1 fired

    def test_sorted_best_first(self, frame):
        accs = [e["top1_accuracy"] for e in signal_report(frame)["per_signal"]]
        assert accs == sorted(accs, reverse=True)
        assert signal_report(frame)["per_signal"][-1]["signal"] == "bm25_knn"

    def test_missing_rate_reported_for_missable_signals(self, frame):
        by = _by_name(signal_report(frame))
        assert "candidate_missing_rate" in by["bm25_knn"]
        # No 'missing' column for the always-scored dense description/prototype.
        assert "candidate_missing_rate" not in by["dense_description"]
        assert "candidate_missing_rate" not in by["dense_prototype"]


class TestAgreementAndShape:
    def test_agreement(self, frame):
        ag = signal_report(frame)["agreement"]
        # distinct top picks: item0 -> 2, item1 -> 1; mean 1.5, half at full consensus.
        assert ag["mean_distinct_top_classes"] == pytest.approx(1.5)
        assert ag["consensus_rate"] == pytest.approx(0.5)

    def test_counts(self, frame):
        report = signal_report(frame)
        assert report["n_items"] == 2
        assert report["n_candidate_rows"] == 5

    def test_empty_frame_is_zero_filled_not_error(self):
        report = signal_report(pd.DataFrame(columns=["item_id", "is_true", *FLAG_COLS]))
        assert report == {
            "n_items": 0,
            "n_candidate_rows": 0,
            "per_signal": [],
            "agreement": {},
            "skipped_signals": [],
        }

    def test_missing_is_true_raises(self, frame):
        with pytest.raises(ValueError, match="is_true"):
            signal_report(frame.drop(columns=["is_true"]))

    def test_no_signal_absent_from_a_full_frame(self, frame):
        assert signal_report(frame)["skipped_signals"] == []

    def test_a_pruned_frame_names_its_skipped_signals_instead_of_raising(self, frame):
        """T87: a model trained with drop_features narrows the assembled frame,
        so its diagnostics narrow with it — named, not KeyError'd."""
        pruned = frame.drop(columns=["is_b_knn_top1", "b_knn_missing"])
        report = signal_report(pruned)
        assert report["skipped_signals"] == ["bm25_knn"]
        assert "bm25_knn" not in {e["signal"] for e in report["per_signal"]}
        assert len(report["per_signal"]) == len(SIGNALS) - 1
