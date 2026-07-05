"""T65 — scoring helpers: runner-up identity + top-k ranking.

`top_per_item` and `top_k_per_item` operate on an already-scored feature table
(item_id, candidate, conf columns); no encoder/model/index needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from text_classifier.application.scoring import top_k_per_item, top_per_item


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
