"""T80 — best-epoch selection for encoder fine-tuning.

A multi-epoch fine-tune used to return whatever the last epoch produced. These
tests cover the three pieces that let it return the *best* epoch instead:

Part A: ``encoder_retrieval_metrics`` — the held-out retrieval metrics.
Part B: ``EpochSelectionPolicy`` — the pure "which epoch / stop now" rule.
Part C: ``EncoderEpochTracker`` — the per-epoch loop, driven by a stub encoder
        whose embeddings improve (or degrade) on command: history, snapshotting,
        duplicate-call tolerance, early stopping.
Part D: ``_stratified_holdout`` — the fine-tune/holdout split.
Part E: config validation + the new EncoderConfig fields' round-trip.

All offline: no torch, no sentence-transformers, no download.
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier.config import EncoderConfig, PipelineConfig
from text_classifier.domain import (
    ENCODER_SELECTION_METRICS,
    EpochSelectionPolicy,
    TextEncoder,
    encoder_retrieval_metrics,
)
from text_classifier.infrastructure.encoder import (
    MIN_EPOCH_HOLDOUT,
    EncoderEpochTracker,
    _stratified_holdout,
)


def _unit(rows) -> np.ndarray:
    """L2-normalize a list of vectors (the package-wide embedding invariant)."""
    arr = np.asarray(rows, dtype=np.float32)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)


# --------------------------------------------------------------------------- #
# Part A — held-out retrieval metrics
# --------------------------------------------------------------------------- #
class TestRetrievalMetrics:
    def test_perfect_encoder_scores_one(self):
        """Every item sitting exactly on its own description: acc == mrr == 1."""
        desc = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        queries = desc[[0, 1, 2, 0]]
        metrics = encoder_retrieval_metrics(queries, desc, np.array([0, 1, 2, 0]))
        assert metrics["desc_acc@1"] == pytest.approx(1.0)
        assert metrics["desc_mrr"] == pytest.approx(1.0)
        assert metrics["desc_pos_sim"] == pytest.approx(1.0, abs=1e-6)

    def test_metrics_rank_a_partly_wrong_encoder_between_the_extremes(self):
        desc = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # item 0 correct; item 1 is closest to class 2, with class 1 second.
        queries = _unit([[1, 0, 0], [0, 0.4, 1.0]])
        metrics = encoder_retrieval_metrics(queries, desc, np.array([0, 1]))
        assert metrics["desc_acc@1"] == pytest.approx(0.5)
        # ranks are 1 and 2 -> (1 + 0.5) / 2
        assert metrics["desc_mrr"] == pytest.approx(0.75)

    def test_worst_case_ranks_last(self):
        desc = _unit([[1, 0], [-1, 0]])
        queries = _unit([[1, 0]])
        metrics = encoder_retrieval_metrics(queries, desc, np.array([1]))
        assert metrics["desc_acc@1"] == pytest.approx(0.0)
        assert metrics["desc_mrr"] == pytest.approx(0.5)  # rank 2
        assert metrics["desc_pos_sim"] < 0

    def test_knn_metric_only_present_with_a_pool(self):
        desc = _unit([[1, 0], [0, 1]])
        queries = _unit([[1, 0.2], [0.2, 1]])
        without = encoder_retrieval_metrics(queries, desc, np.array([0, 1]))
        assert "knn_acc@1" not in without

        pool = _unit([[1, 0.1], [0, 1]])
        with_pool = encoder_retrieval_metrics(
            queries, desc, np.array([0, 1]), pool_emb=pool, pool_labels=np.array([0, 1])
        )
        assert with_pool["knn_acc@1"] == pytest.approx(1.0)
        # a pool labeled the other way round is wrong for every query
        swapped = encoder_retrieval_metrics(
            queries, desc, np.array([0, 1]), pool_emb=pool, pool_labels=np.array([1, 0])
        )
        assert swapped["knn_acc@1"] == pytest.approx(0.0)

    def test_all_declared_metrics_are_produced(self):
        """Guards the config<->measurement contract: every selectable metric is
        actually measured when a pool is supplied."""
        desc = _unit([[1, 0], [0, 1]])
        queries = _unit([[1, 0.2]])
        metrics = encoder_retrieval_metrics(
            queries, desc, np.array([0]), pool_emb=desc, pool_labels=np.array([0, 1])
        )
        assert set(metrics) == set(ENCODER_SELECTION_METRICS)

    def test_empty_holdout_is_nan_not_an_exception(self):
        metrics = encoder_retrieval_metrics(
            np.zeros((0, 2), dtype=np.float32), _unit([[1, 0]]), np.array([], dtype=np.intp)
        )
        assert all(np.isnan(v) for v in metrics.values())


# --------------------------------------------------------------------------- #
# Part B — the selection policy
# --------------------------------------------------------------------------- #
class TestEpochSelectionPolicy:
    def _history(self, *scores) -> list:
        return [{"desc_acc@1": s} for s in scores]

    def test_picks_the_argmax_epoch_one_based(self):
        policy = EpochSelectionPolicy()
        assert policy.best_epoch(self._history(0.1, 0.9, 0.5)) == 2

    def test_last_epoch_wins_when_it_is_the_best(self):
        policy = EpochSelectionPolicy()
        assert policy.best_epoch(self._history(0.1, 0.2, 0.3)) == 3

    def test_ties_go_to_the_earlier_epoch(self):
        """Equal quality from less training is the cheaper, less-overfit model."""
        policy = EpochSelectionPolicy()
        assert policy.best_epoch(self._history(0.5, 0.5, 0.5)) == 1

    def test_min_delta_requires_a_meaningful_improvement(self):
        history = self._history(0.50, 0.51)
        assert EpochSelectionPolicy(min_delta=0.0).best_epoch(history) == 2
        assert EpochSelectionPolicy(min_delta=0.05).best_epoch(history) == 1

    def test_empty_history_selects_nothing(self):
        assert EpochSelectionPolicy().best_epoch([]) == 0

    def test_all_nan_history_selects_nothing(self):
        assert EpochSelectionPolicy().best_epoch(self._history(float("nan"), float("nan"))) == 0

    def test_nan_epochs_are_skipped_not_selected(self):
        assert EpochSelectionPolicy().best_epoch(self._history(float("nan"), 0.3)) == 2

    def test_patience_zero_never_stops(self):
        policy = EpochSelectionPolicy(patience=0)
        assert not policy.should_stop(self._history(0.9, 0.1, 0.1, 0.1))

    def test_patience_stops_after_n_non_improving_epochs(self):
        policy = EpochSelectionPolicy(patience=2)
        assert not policy.should_stop(self._history(0.9))
        assert not policy.should_stop(self._history(0.9, 0.1))
        assert policy.should_stop(self._history(0.9, 0.1, 0.1))

    def test_improvement_resets_patience(self):
        policy = EpochSelectionPolicy(patience=2)
        assert not policy.should_stop(self._history(0.5, 0.4, 0.9))

    def test_unmeasured_metric_raises_naming_what_was_available(self):
        policy = EpochSelectionPolicy(metric="knn_acc@1")
        with pytest.raises(KeyError, match="desc_acc@1"):
            policy.score({"desc_acc@1": 0.5})


# --------------------------------------------------------------------------- #
# Part C — the per-epoch tracker
# --------------------------------------------------------------------------- #
class _ScriptedEncoder(TextEncoder):
    """A stub encoder whose "training" is scripted: ``desc_acc@1`` on the two
    holdout items below follows ``self.scores`` (0.0/0.5/1.0), advancing one step
    per ``advance()``. Stands in for a model whose weights change between epochs
    without needing torch."""

    LAYOUTS = {
        0.0: [[0, 1], [1, 0]],  # both items nearest the wrong description
        0.5: [[1, 0], [1, 0]],  # item 0 right, item 1 wrong
        1.0: [[1, 0], [0, 1]],  # both right
    }

    def __init__(self, scores) -> None:
        self.scores = list(scores)
        self.step = 0
        self.encode_calls = 0

    def advance(self) -> None:
        self.step += 1

    @property
    def _score(self) -> float:
        return self.scores[min(self.step, len(self.scores) - 1)]

    def encode(self, texts):
        self.encode_calls += 1
        if list(texts) == ["desc-a", "desc-b"]:  # the class descriptions: fixed
            return _unit([[1, 0], [0, 1]])
        # A step-dependent nudge too small to change any ranking: consecutive
        # epochs of *equal* accuracy still produce different weights, as real
        # training would (the tracker's duplicate guard must not swallow them).
        rows = np.asarray(self.LAYOUTS[self._score], dtype=np.float32)
        rows[:, 1] += 1e-3 * self.step
        return _unit(rows)

    def save(self, directory: str) -> None:  # pragma: no cover - not exercised
        raise AssertionError("the tracker must not save through the encoder")


def _tracker(scores, **policy_kwargs):
    encoder = _ScriptedEncoder(scores)
    snapshots: list[int] = []
    tracker = EncoderEpochTracker(
        encoder,
        holdout_texts=["item-a", "item-b"],
        holdout_labels=np.array([0, 1]),
        descriptions=["desc-a", "desc-b"],
        policy=EpochSelectionPolicy(**policy_kwargs),
        snapshot=lambda: snapshots.append(len(tracker.history)),
    )
    return tracker, encoder, snapshots


def _run(tracker, encoder, n_epochs) -> int:
    """Drive ``n_epochs`` of scripted training; returns the epochs actually run."""
    for _ in range(n_epochs):
        if tracker.observe() is not None and tracker.stop_requested:
            break
        encoder.advance()
    return len(tracker.history)


class TestEncoderEpochTracker:
    def test_records_one_history_entry_per_epoch(self):
        tracker, encoder, _ = _tracker([0.0, 0.5, 1.0])
        _run(tracker, encoder, 3)
        assert [m["desc_acc@1"] for m in tracker.history] == [0.0, 0.5, 1.0]

    def test_snapshots_only_on_improving_epochs(self):
        """The mid epoch is the best; the snapshot must be taken there and not
        overwritten by the worse epochs that follow."""
        tracker, encoder, snapshots = _tracker([0.0, 1.0, 0.5, 0.0])
        _run(tracker, encoder, 4)
        assert snapshots == [1, 2]  # epoch 1 (first score), then the improvement
        assert tracker.best_epoch == 2

    def test_best_epoch_can_be_the_last(self):
        tracker, encoder, snapshots = _tracker([0.0, 0.5, 1.0])
        _run(tracker, encoder, 3)
        assert tracker.best_epoch == 3
        assert snapshots == [1, 2, 3]

    def test_repeated_call_on_unchanged_weights_is_ignored(self):
        """Some sentence-transformers versions can invoke an evaluator twice at an
        epoch boundary; counting it would shift the epoch numbering."""
        tracker, encoder, snapshots = _tracker([0.5, 1.0])
        assert tracker.observe() is not None
        assert tracker.observe() is None  # no advance() in between
        encoder.advance()
        assert tracker.observe() is not None
        assert len(tracker.history) == 2
        assert snapshots == [1, 2]

    def test_early_stopping_after_patience_epochs(self):
        tracker, encoder, _ = _tracker([1.0, 0.5, 0.5, 0.5, 0.5], patience=2)
        epochs_run = _run(tracker, encoder, 5)
        assert epochs_run == 3  # best at 1, two non-improving epochs -> stop
        assert tracker.stop_requested
        assert tracker.best_epoch == 1

    def test_no_early_stopping_without_patience(self):
        tracker, encoder, _ = _tracker([1.0, 0.5, 0.5, 0.5, 0.5])
        assert _run(tracker, encoder, 5) == 5
        assert not tracker.stop_requested

    def test_report_is_json_clean_and_carries_the_table(self):
        import json

        tracker, encoder, _ = _tracker([0.0, 1.0, 0.5], min_delta=0.01, patience=3)
        _run(tracker, encoder, 3)
        report = tracker.report()
        assert report["select_metric"] == "desc_acc@1"
        assert report["select_min_delta"] == 0.01
        assert report["early_stopping_patience"] == 3
        assert report["n_holdout_items"] == 2
        assert report["best_epoch"] == 2
        assert [e["epoch"] for e in report["epochs"]] == [1, 2, 3]
        json.loads(json.dumps(report))  # must survive persistence as-is

    def test_knn_pool_is_scored_when_supplied(self):
        encoder = _ScriptedEncoder([1.0])
        tracker = EncoderEpochTracker(
            encoder,
            holdout_texts=["item-a", "item-b"],
            holdout_labels=np.array([0, 1]),
            descriptions=["desc-a", "desc-b"],
            policy=EpochSelectionPolicy(metric="knn_acc@1"),
            snapshot=lambda: None,
            pool_texts=["pool-a", "pool-b"],
            pool_labels=np.array([0, 1]),
        )
        assert tracker.observe() is not None
        assert "knn_acc@1" in tracker.history[0]


# --------------------------------------------------------------------------- #
# Part D — the fine-tune/holdout split
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
# Part E — config surface
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

    def test_valid_multi_epoch_config_passes(self):
        cfg = PipelineConfig()
        cfg.encoder.train_epochs = 20
        cfg.encoder.train_holdout_ratio = 0.5
        cfg.encoder.train_early_stopping_patience = 3
        cfg.validate()

    def test_min_holdout_guard_is_a_positive_threshold(self):
        assert MIN_EPOCH_HOLDOUT >= 2
