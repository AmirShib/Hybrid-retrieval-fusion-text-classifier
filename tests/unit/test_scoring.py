"""T65 — scoring helpers: runner-up identity + top-k ranking.

`top_per_item` and `top_k_per_item` operate on an already-scored feature table
(item_id, candidate, conf columns); no encoder/model/index needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from text_classifier.application.scoring import rank_candidates, top_k_per_item, top_per_item


def _scored(rows):
    """rows: list of (item_id, candidate, conf)."""
    return pd.DataFrame(rows, columns=["item_id", "candidate", "conf"])


class TestTopPerItem:
    def test_runner_up_candidate_matches_second_highest_conf(self):
        scored = _scored(
            [
                (0, 5, 0.9),
                (0, 2, 0.4),
                (0, 7, 0.6),
            ]
        )
        top = top_per_item(scored)
        row = top[top["item_id"] == 0].iloc[0]
        assert row["candidate"] == 5
        assert row["second_candidate"] == 7
        assert row["second_conf"] == pytest.approx(0.6)
        assert row["margin"] == pytest.approx(0.3)

    def test_single_candidate_item_has_nan_second_and_margin_equals_conf(self):
        scored = _scored([(1, 3, 0.8)])
        top = top_per_item(scored)
        row = top[top["item_id"] == 1].iloc[0]
        assert np.isnan(row["second_candidate"])
        assert row["margin"] == pytest.approx(0.8)

    def test_multiple_items_independent(self):
        scored = _scored(
            [
                (0, 1, 0.7),
                (0, 2, 0.3),
                (1, 9, 0.95),
            ]
        )
        top = top_per_item(scored).set_index("item_id")
        assert top.loc[0, "candidate"] == 1
        assert top.loc[0, "second_candidate"] == 2
        assert np.isnan(top.loc[1, "second_candidate"])


class TestTopKPerItem:
    def test_ranks_strictly_by_conf_descending(self):
        scored = _scored(
            [
                (0, 1, 0.2),
                (0, 2, 0.9),
                (0, 3, 0.5),
            ]
        )
        ranked = top_k_per_item(scored, k=3)
        item0 = ranked[ranked["item_id"] == 0].sort_values("rank")
        assert item0["candidate"].tolist() == [2, 3, 1]
        assert item0["rank"].tolist() == [1, 2, 3]

    def test_k_greater_than_n_candidates_truncates(self):
        scored = _scored([(0, 1, 0.5), (0, 2, 0.9)])
        ranked = top_k_per_item(scored, k=5)
        assert len(ranked[ranked["item_id"] == 0]) == 2

    def test_k_equals_one_agrees_with_top_candidate(self):
        scored = _scored([(0, 1, 0.5), (0, 2, 0.9), (0, 3, 0.1)])
        ranked = top_k_per_item(scored, k=1)
        assert ranked["candidate"].tolist() == [2]
        assert ranked["rank"].tolist() == [1]

    def test_independent_per_item_caps(self):
        scored = _scored(
            [
                (0, 1, 0.5),
                (0, 2, 0.9),
                (1, 5, 0.3),
            ]
        )
        ranked = top_k_per_item(scored, k=2)
        assert len(ranked[ranked["item_id"] == 0]) == 2
        assert len(ranked[ranked["item_id"] == 1]) == 1


class TestRankCandidates:
    """`rank_candidates` is the one ranking rule behind `top_k_per_item`,
    `InferencePipeline.explain`, and `explain_records` — all three used to spell
    it out separately, so these pin the shared contract they now depend on."""

    def test_ranks_are_per_item_and_one_based(self):
        ranked = rank_candidates(_scored([(0, 1, 0.2), (0, 2, 0.9), (1, 3, 0.5), (1, 4, 0.7)]))
        assert ranked["rank"].tolist() == [1, 2, 1, 2]
        assert ranked["candidate"].tolist() == [2, 1, 4, 3]

    def test_extra_columns_are_carried_through(self):
        """`explain` reads the feature columns off the ranked frame, so ranking
        must not project them away — only reorder."""
        scored = _scored([(0, 1, 0.2), (0, 2, 0.9)])
        scored["d_desc_sim"] = [0.11, 0.22]
        ranked = rank_candidates(scored)
        assert ranked["d_desc_sim"].tolist() == [0.22, 0.11]

    def test_k_truncates_per_item_independently(self):
        ranked = rank_candidates(_scored([(0, 1, 0.9), (0, 2, 0.8), (0, 3, 0.7), (1, 4, 0.5)]), k=2)
        assert ranked["item_id"].tolist() == [0, 0, 1]
        assert ranked["rank"].tolist() == [1, 2, 1]

    def test_index_is_positional_after_truncation(self):
        """`explain_records` indexes a contributions matrix by row position, so
        the returned index must be a clean 0..n-1 range, not the pre-sort one."""
        ranked = rank_candidates(_scored([(0, 1, 0.1), (0, 2, 0.9), (1, 3, 0.5)]), k=1)
        assert ranked.index.tolist() == list(range(len(ranked)))

    def test_k_none_keeps_every_candidate(self):
        ranked = rank_candidates(_scored([(0, 1, 0.1), (0, 2, 0.9), (0, 3, 0.5)]), k=None)
        assert len(ranked) == 3

    def test_agrees_with_top_k_per_item(self):
        scored = _scored([(0, 1, 0.4), (0, 2, 0.9), (1, 3, 0.2), (1, 4, 0.8)])
        ranked = rank_candidates(scored, k=1)[["item_id", "rank", "candidate", "conf"]]
        pd.testing.assert_frame_equal(ranked, top_k_per_item(scored, 1))
