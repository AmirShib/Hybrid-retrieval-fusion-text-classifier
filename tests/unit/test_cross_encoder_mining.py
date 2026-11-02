"""T33 phase 3 — mining cross-encoder training lists from round-one shortlists.

Pure pandas/numpy: no torch, no sentence-transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from text_classifier.config import CrossEncoderDocument, EvidenceSpec, PipelineConfig
from text_classifier.domain import ClassDefinition
from text_classifier.infrastructure.reranker import (
    mine_training_lists,
    prescore_column,
    training_examples,
)


def _shortlist(rows):
    """rows: (item_id, candidate, d_desc_sim)."""
    return pd.DataFrame(rows, columns=["item_id", "candidate", "d_desc_sim"])


class TestMineTrainingLists:
    def test_positive_is_the_true_class_and_negatives_are_hardest_first(self):
        sl = _shortlist([(0, 0, 0.9), (0, 1, 0.2), (0, 2, 0.7), (0, 3, 0.5)])
        lists = mine_training_lists(sl, np.array([0]), n_negatives=2, order_column="d_desc_sim")
        assert lists == [(0, 0, [2, 3])]  # 0.7 then 0.5; the 0.2 one is cut

    def test_true_class_missing_from_the_shortlist_is_still_the_positive(self):
        """The reranker should learn what the right answer looks like even when
        round one missed it."""
        sl = _shortlist([(0, 1, 0.8), (0, 2, 0.6)])
        assert mine_training_lists(sl, np.array([5]), 9, "d_desc_sim") == [(0, 5, [1, 2])]

    def test_item_with_no_negatives_is_skipped(self):
        sl = _shortlist([(0, 3, 0.9), (1, 1, 0.9), (1, 2, 0.1)])
        assert mine_training_lists(sl, np.array([3, 1]), 9, "d_desc_sim") == [(1, 1, [2])]

    def test_nan_scores_rank_last(self):
        sl = _shortlist([(0, 1, np.nan), (0, 2, 0.1), (0, 3, 0.3)])
        assert mine_training_lists(sl, np.array([0]), 3, "d_desc_sim") == [(0, 0, [3, 2, 1])]

    def test_without_an_order_column_sampling_is_seeded(self):
        sl = _shortlist([(0, c, 0.0) for c in range(1, 20)])
        a = mine_training_lists(sl, np.array([0]), 5, None, seed=3)
        b = mine_training_lists(sl, np.array([0]), 5, None, seed=3)
        assert a == b and len(a[0][2]) == 5 and 0 not in a[0][2]

    def test_empty_shortlist(self):
        assert mine_training_lists(_shortlist([]), np.array([]), 3) == []

    def test_builtin_prescore_nodes_map_to_their_raw_column(self):
        assert prescore_column("dense.desc") == "d_desc_sim"
        assert prescore_column("bm25.knn") == "b_knn_sum"
        assert prescore_column("some.custom.node") is None


class TestTrainingExamples:
    DEFS = [
        ClassDefinition("A", "fresh fruit", examples=("apples", "bananas"), exclusions=("juice",)),
        ClassDefinition("B", "dairy products", examples=("milk",), exclusions=("butter",)),
        ClassDefinition("C", "frozen food"),  # no examples, no exclusions
    ]
    POS = CrossEncoderDocument(
        name="pos",
        evidence=[EvidenceSpec(view="core", select="first"), EvidenceSpec(view="examples")],
    )
    # A document drawn only from a view some classes leave empty -- like "neg".
    EXCL = CrossEncoderDocument(name="neg", evidence=[EvidenceSpec(view="exclusions")])

    def test_renders_with_query_adaptive_evidence(self):
        ex = training_examples([(0, 0, [1])], ["green apples"], self.DEFS, self.POS)
        (query, pos, negs) = ex[0]
        assert query == "green apples"
        assert "fresh fruit" in pos and "apples" in pos  # the example most like the query
        assert len(negs) == 1 and "dairy" in negs[0]

    def test_classes_with_no_text_for_the_document_are_dropped_as_negatives(self):
        ex = training_examples([(0, 0, [2, 1])], ["apples"], self.DEFS, self.EXCL)
        assert ex[0][2] == ["butter"]  # class C has no exclusions -> dropped

    def test_item_whose_positive_renders_empty_is_skipped(self):
        assert training_examples([(0, 2, [0, 1])], ["x"], self.DEFS, self.EXCL) == []

    def test_item_left_with_no_negatives_is_skipped(self):
        assert training_examples([(0, 0, [2])], ["apples"], self.DEFS, self.EXCL) == []


class TestTrainingConfig:
    def _cfg(self, **ce):
        cfg = PipelineConfig()
        cfg.signals = ["dense", "lexical", "cross-encoder"]
        cfg.retrieval.cross_encoder.kind = "cross-encoder"
        cfg.retrieval.cross_encoder.model_name = "some/model"
        for k, v in ce.items():
            setattr(cfg.retrieval.cross_encoder, k, v)
        return cfg

    def test_defaults_are_off_and_valid(self):
        PipelineConfig().validate()
        assert PipelineConfig().retrieval.cross_encoder.train is False
        self._cfg(train=True).validate()

    def test_cross_encoder_kind_needs_a_model_name(self):
        with pytest.raises(ValueError, match="cross_encoder.model_name"):
            self._cfg(model_name="").validate()

    def test_training_is_refused_on_the_leave_one_out_path(self):
        cfg = self._cfg(train=True)
        cfg.training.n_folds = 1
        with pytest.raises(ValueError, match="leave-one-out"):
            cfg.validate(external_val=True, external_test=True)

    def test_training_without_the_signal_is_refused_rather_than_ignored(self):
        cfg = self._cfg(train=True)
        cfg.signals = ["dense", "lexical"]
        with pytest.raises(ValueError, match="'cross-encoder' in signals"):
            cfg.validate()

    def test_the_token_overlap_stand_in_cannot_be_trained(self):
        with pytest.raises(ValueError, match="no weights to train"):
            self._cfg(train=True, kind="token-overlap").validate()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("train_document", "nope"),
            ("train_negatives", 0),
            ("train_loss", "hinge"),
            ("train_epochs", 0),
            ("train_batch_size", 0),
            ("learning_rate", 0.0),
            ("weight_decay", -1.0),
            ("warmup_ratio", 1.0),
            ("train_holdout_ratio", 0.9),
            ("train_select_metric", "accuracy"),
            ("train_early_stopping_patience", -1),
            ("train_select_min_delta", -0.1),
        ],
    )
    def test_invalid_training_values_are_rejected_by_name(self, field, value):
        with pytest.raises(ValueError, match=f"cross_encoder.{field}"):
            self._cfg(**{field: value}).validate()

    def test_training_fields_round_trip_through_json(self):
        cfg = self._cfg(train=True, train_loss="lambda", train_negatives=4, learning_rate=3e-5)
        back = PipelineConfig.from_dict(cfg.to_dict())
        assert back.retrieval.cross_encoder == cfg.retrieval.cross_encoder
