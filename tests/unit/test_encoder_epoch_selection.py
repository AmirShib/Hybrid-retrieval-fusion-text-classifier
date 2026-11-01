"""T80 — best-epoch selection for encoder fine-tuning: the framework-free parts.

A multi-epoch fine-tune used to return whatever the last epoch produced; it now
returns the best one. The scoring and the epoch choice are done by
sentence-transformers itself (``InformationRetrievalEvaluator`` +
``SentenceTransformerTrainer``'s ``load_best_model_at_end``) and are covered
end-to-end, against a real tiny model, in ``test_train_encoder_trainer.py``. What
stays here needs no torch and no sentence-transformers:

- ``_stratified_holdout`` — the fine-tune/holdout split;
- config validation + the EncoderConfig fields' round-trip.

All offline: no torch, no sentence-transformers, no download.
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier.config import EncoderConfig, PipelineConfig
from text_classifier.domain import ENCODER_POOLING_MODES, ENCODER_SELECTION_METRICS
from text_classifier.infrastructure.encoder import MIN_EPOCH_HOLDOUT, _stratified_holdout


# --------------------------------------------------------------------------- #
# Part A — the fine-tune/holdout split
# --------------------------------------------------------------------------- #
class TestStratifiedHoldout:
    def test_ratio_zero_holds_out_nothing(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        fit, holdout = _stratified_holdout(labels, 0.0, seed=0)
        assert holdout.size == 0
        np.testing.assert_array_equal(fit, np.arange(6))

    def test_split_is_a_partition(self):
        labels = np.repeat(np.arange(5), 8)
        fit, holdout = _stratified_holdout(labels, 0.25, seed=0)
        np.testing.assert_array_equal(np.sort(np.concatenate([fit, holdout])), np.arange(40))
        assert holdout.size == 10

    def test_holdout_is_stratified_across_classes(self):
        labels = np.repeat(np.arange(4), 10)
        _, holdout = _stratified_holdout(labels, 0.2, seed=0)
        counts = np.bincount(labels[holdout], minlength=4)
        np.testing.assert_array_equal(counts, [2, 2, 2, 2])

    def test_a_class_never_loses_its_last_training_example(self):
        """A singleton class must keep its (item, description) pair — otherwise the
        loss has nothing pulling that class together."""
        labels = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        fit, _ = _stratified_holdout(labels, 0.9, seed=0)
        assert 0 in set(fit.tolist())

    def test_same_seed_same_split_different_seed_may_differ(self):
        labels = np.repeat(np.arange(4), 10)
        a = _stratified_holdout(labels, 0.3, seed=0)
        b = _stratified_holdout(labels, 0.3, seed=0)
        c = _stratified_holdout(labels, 0.3, seed=1)
        np.testing.assert_array_equal(a[1], b[1])
        assert not np.array_equal(a[1], c[1])

    def test_ascending_indices(self):
        labels = np.repeat(np.arange(3), 7)
        fit, holdout = _stratified_holdout(labels, 0.3, seed=2)
        np.testing.assert_array_equal(fit, np.sort(fit))
        np.testing.assert_array_equal(holdout, np.sort(holdout))


# --------------------------------------------------------------------------- #
# Part B — config surface
# --------------------------------------------------------------------------- #
class TestEncoderEpochConfig:
    def test_default_config_is_single_epoch_so_selection_is_inert(self):
        """The package default trains one epoch: there is nothing to select
        between, so the new fields cannot change a default run."""
        assert EncoderConfig().train_epochs == 1
        assert EncoderConfig().train_select_metric in ENCODER_SELECTION_METRICS

    def test_selection_fields_round_trip(self):
        cfg = PipelineConfig()
        cfg.encoder = EncoderConfig(
            train_epochs=20,
            train_holdout_ratio=0.15,
            train_select_metric="desc_mrr",
            train_select_min_delta=0.002,
            train_early_stopping_patience=4,
            train_holdout_seed=7,
        )
        back = PipelineConfig.from_dict(cfg.to_dict())
        assert back.encoder == cfg.encoder

    @pytest.mark.parametrize(
        "field,value",
        [
            ("train_epochs", 0),
            ("train_holdout_ratio", -0.1),
            ("train_holdout_ratio", 0.9),
            ("train_select_metric", "not-a-metric"),
            ("train_select_min_delta", -0.01),
            ("train_early_stopping_patience", -1),
            ("train_batch_size", 0),
        ],
    )
    def test_invalid_values_are_rejected_by_name(self, field, value):
        cfg = PipelineConfig()
        setattr(cfg.encoder, field, value)
        with pytest.raises(ValueError, match=f"encoder.{field}"):
            cfg.validate()

    def test_selection_metrics_are_the_ir_evaluator_backed_ones(self):
        assert ENCODER_SELECTION_METRICS == ("desc_acc@1", "desc_mrr", "knn_acc@1")

    def test_valid_multi_epoch_config_passes(self):
        cfg = PipelineConfig()
        cfg.encoder.train_epochs = 20
        cfg.encoder.train_holdout_ratio = 0.5
        cfg.encoder.train_early_stopping_patience = 3
        cfg.validate()

    def test_min_holdout_guard_is_a_positive_threshold(self):
        assert MIN_EPOCH_HOLDOUT >= 2


class TestEncoderTrainingKnobsConfig:
    """pooling_mode / max_seq_length / learning_rate / weight_decay."""

    def test_defaults_leave_the_models_own_behaviour_alone(self):
        enc = EncoderConfig()
        assert enc.pooling_mode is None
        assert enc.max_seq_length is None
        # the values the library's own `model.fit` used before these were configurable
        assert enc.learning_rate == 2e-5
        assert enc.weight_decay == 0.01

    def test_new_fields_round_trip_through_json(self):
        cfg = PipelineConfig()
        cfg.encoder = EncoderConfig(
            pooling_mode="cls", max_seq_length=256, learning_rate=5e-5, weight_decay=0.1
        )
        back = PipelineConfig.from_dict(cfg.to_dict())
        assert back.encoder == cfg.encoder

    def test_a_config_saved_before_these_fields_still_loads_with_defaults(self):
        data = PipelineConfig().to_dict()
        for key in ("pooling_mode", "max_seq_length", "learning_rate", "weight_decay"):
            data["encoder"].pop(key)
        back = PipelineConfig.from_dict(data)
        assert back.encoder == EncoderConfig()

    @pytest.mark.parametrize("mode", list(ENCODER_POOLING_MODES))
    def test_every_declared_pooling_mode_validates(self, mode):
        cfg = PipelineConfig()
        cfg.encoder.pooling_mode = mode
        cfg.validate()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("pooling_mode", "median"),
            ("pooling_mode", ""),
            ("max_seq_length", 0),
            ("max_seq_length", -5),
            ("max_seq_length", 12.5),
            ("max_seq_length", True),
            ("learning_rate", 0.0),
            ("learning_rate", -1e-5),
            ("weight_decay", -0.01),
        ],
    )
    def test_invalid_values_are_rejected_by_name(self, field, value):
        cfg = PipelineConfig()
        setattr(cfg.encoder, field, value)
        with pytest.raises(ValueError, match=f"encoder.{field}"):
            cfg.validate()

    def test_zero_weight_decay_is_allowed(self):
        cfg = PipelineConfig()
        cfg.encoder.weight_decay = 0.0
        cfg.validate()
