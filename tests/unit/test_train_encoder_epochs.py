"""T80 — ``train_encoder`` orchestration, against a fake sentence-transformers.

The real fine-tune needs torch and a model download, which CI must not do. This
module installs a fake ``sentence_transformers`` that mirrors the parts of the
contract ``train_encoder`` depends on — verified against sentence-transformers
5.6's ``FitMixin``/``EvaluatorCallback``:

* ``fit`` runs the evaluator exactly once at the end of every epoch, passing
  ``(model, output_path=..., epoch=..., steps=...)``;
* a bare float return from the evaluator is accepted;
* with an evaluator present, ``fit`` does *not* itself write ``output_path``
  (``SaveModelCallback.on_train_end`` only saves when there is no evaluator);
* ``SentenceTransformer(<dir>)`` reloads what ``model.save(<dir>)`` wrote.

The fake model's embeddings are scripted per epoch, so a test can say "epoch 2
was the good one" and assert that the returned encoder is epoch 2's — the whole
point of best-epoch selection.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types

import numpy as np
import pytest

from text_classifier import ClassDefinition, LabeledItem, LabelSpace
from text_classifier.config import EncoderConfig


N_CLASSES = 3
PER_CLASS = 8
STATE_FILE = "fake_state.json"


def _label_space() -> LabelSpace:
    return LabelSpace(
        [ClassDefinition(f"C{c}", f"description of class {c}") for c in range(N_CLASSES)]
    )


def _items() -> list:
    return [
        LabeledItem(f"item {c} number {i}", f"C{c}")
        for c in range(N_CLASSES)
        for i in range(PER_CLASS)
    ]


# --------------------------------------------------------------------------- #
# the fake sentence-transformers
# --------------------------------------------------------------------------- #
class _Script:
    """Test-controlled state shared with the fake module: which epochs are 'good'
    (items embed onto their own class description) and what happened during fit."""

    def __init__(self) -> None:
        self.good_epochs: set = set()
        self.descriptions: dict = {}  # description text -> class index
        self.fit_kwargs: list = []
        self.saves: list = []  # (directory, epoch) in call order
        self.loads: list = []  # directories reloaded through SentenceTransformer
        self.built_losses: list = []  # (name, kwargs) for each loss constructed


@pytest.fixture
def script(monkeypatch) -> _Script:
    """Install a fake ``sentence_transformers`` package for the duration of a test."""
    state = _Script()

    class InputExample:
        def __init__(self, texts=None, label=0):
            self.texts = list(texts or [])
            self.label = label

    class NoDuplicatesDataLoader:
        """Only ``len`` matters here — it drives warmup/steps-per-epoch."""

        def __init__(self, examples, batch_size):
            self.examples = list(examples)
            self.batch_size = batch_size

        def __len__(self):
            return max(1, len(self.examples) // self.batch_size)

    class SentenceEvaluator:
        def __init__(self):
            self.greater_is_better = True
            self.primary_metric = None

    class MultipleNegativesSymmetricRankingLoss:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            state.built_losses.append(("multiple_negatives_symmetric_ranking", kwargs))

    class MultipleNegativesRankingLoss:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            state.built_losses.append(("multiple_negatives_ranking", kwargs))

    class CachedMultipleNegativesRankingLoss:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            state.built_losses.append(("cached_multiple_negatives_ranking", kwargs))

    class CosineSimilarityLoss:
        """A loss with no friendly alias -- reached only via the dynamic
        sentence_transformers.losses.<ClassName> fallback."""

        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            state.built_losses.append(("CosineSimilarityLoss", kwargs))

    class SentenceTransformer:
        """Embeds a text as a near-one-hot over classes.

        A *description* always lands on its own class. An *item* lands on its own
        class in a "good" epoch and on the next class in a bad one, so
        ``desc_acc@1`` is 1.0 or 0.0 by epoch. Every vector carries an
        epoch-dependent nudge (too small to change any ranking) so consecutive
        equal-scoring epochs still differ, exactly as real weights would.
        """

        def __init__(self, model_name_or_path, device=None, **kwargs):
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.kwargs = kwargs
            self.epoch = 0
            state_path = os.path.join(str(model_name_or_path), STATE_FILE)
            if os.path.exists(state_path):  # reloading a snapshot
                with open(state_path) as fh:
                    self.epoch = json.load(fh)["epoch"]
                state.loads.append(str(model_name_or_path))

        def _vector(self, text: str) -> np.ndarray:
            # Matched rather than looked up, so an asymmetric prompt prefix
            # ("query: ...") still resolves to the right class.
            described = re.search(r"description of class (\d+)", text)
            if described:
                target = int(described.group(1))
            else:
                cls = int(re.search(r"item (\d+) number", text).group(1))
                target = cls if self.epoch in state.good_epochs else (cls + 1) % N_CLASSES
            vec = np.full(N_CLASSES, 1e-4 * (self.epoch + 1), dtype=np.float32)
            vec[target] = 1.0
            return vec

        def encode(self, texts, **kwargs):
            out = np.stack([self._vector(t) for t in texts])
            if kwargs.get("normalize_embeddings"):
                out = out / np.linalg.norm(out, axis=1, keepdims=True)
            return out

        def fit(
            self,
            train_objectives,
            epochs=1,
            warmup_steps=0,
            evaluator=None,
            evaluation_steps=0,
            output_path=None,
            save_best_model=True,
            show_progress_bar=True,
        ):
            state.fit_kwargs.append(
                {
                    "n_examples": len(train_objectives[0][0].examples),
                    "epochs": epochs,
                    "warmup_steps": warmup_steps,
                    "evaluation_steps": evaluation_steps,
                    "save_best_model": save_best_model,
                    "output_path": output_path,
                    "has_evaluator": evaluator is not None,
                }
            )
            for _ in range(epochs):
                self.epoch += 1
                if evaluator is not None:
                    # EvaluatorCallback.on_epoch_end, once per epoch.
                    evaluator(self, output_path=None, epoch=float(self.epoch), steps=self.epoch)
            if output_path is not None and evaluator is None:
                self.save(output_path)  # SaveModelCallback.on_train_end

        def save(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, STATE_FILE), "w") as fh:
                json.dump({"epoch": self.epoch}, fh)
            state.saves.append((directory, self.epoch))

    root = types.ModuleType("sentence_transformers")
    root.InputExample = InputExample
    root.SentenceTransformer = SentenceTransformer
    losses = types.ModuleType("sentence_transformers.losses")
    losses.MultipleNegativesSymmetricRankingLoss = MultipleNegativesSymmetricRankingLoss
    losses.MultipleNegativesRankingLoss = MultipleNegativesRankingLoss
    losses.CachedMultipleNegativesRankingLoss = CachedMultipleNegativesRankingLoss
    losses.CosineSimilarityLoss = CosineSimilarityLoss
    datasets = types.ModuleType("sentence_transformers.datasets")
    datasets.NoDuplicatesDataLoader = NoDuplicatesDataLoader
    evaluation = types.ModuleType("sentence_transformers.evaluation")
    evaluation.SentenceEvaluator = SentenceEvaluator
    root.losses = losses
    root.datasets = datasets
    root.evaluation = evaluation
    for name, module in [
        ("sentence_transformers", root),
        ("sentence_transformers.losses", losses),
        ("sentence_transformers.datasets", datasets),
        ("sentence_transformers.evaluation", evaluation),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    ls = _label_space()
    state.descriptions = {desc: i for i, desc in enumerate(ls.descriptions)}
    return state


def _config(**kwargs) -> EncoderConfig:
    base = dict(model_name_or_path="fake/model", train_batch_size=4, encode_batch_size=8)
    base.update(kwargs)
    return EncoderConfig(**base)


def _train(config, output_path=None):
    from text_classifier.infrastructure.encoder import train_encoder

    return train_encoder(_items(), _label_space(), config, output_path=output_path)


# --------------------------------------------------------------------------- #
# selection off: the pre-existing behaviour, unchanged
# --------------------------------------------------------------------------- #
class TestWithoutSelection:
    def test_single_epoch_trains_on_every_item_and_scores_nothing(self, script):
        script.good_epochs = {1}
        encoder = _train(_config(train_epochs=1))
        (call,) = script.fit_kwargs
        assert call["n_examples"] == N_CLASSES * PER_CLASS  # nothing held out
        assert call["has_evaluator"] is False
        assert encoder.training_history == {}

    def test_zero_holdout_ratio_disables_selection_for_multi_epoch(self, script):
        script.good_epochs = {2}
        encoder = _train(_config(train_epochs=5, train_holdout_ratio=0.0))
        (call,) = script.fit_kwargs
        assert call["n_examples"] == N_CLASSES * PER_CLASS
        assert call["has_evaluator"] is False
        assert call["epochs"] == 5
        assert encoder.training_history == {}

    def test_output_path_is_still_written_without_selection(self, script, tmp_path):
        out = tmp_path / "enc"
        _train(_config(train_epochs=1), output_path=str(out))
        assert (out / STATE_FILE).exists()

    def test_too_small_a_holdout_falls_back_to_last_epoch(self, script, caplog):
        """3 classes × 4 items at 25% yields a 3-item holdout — below the floor for
        ranking epochs, so selection switches itself off loudly and trains on all."""
        from text_classifier.infrastructure.encoder import train_encoder

        tiny = [
            LabeledItem(f"item {c} number {i}", f"C{c}") for c in range(N_CLASSES) for i in range(4)
        ]
        with caplog.at_level("WARNING"):
            encoder = train_encoder(
                tiny, _label_space(), _config(train_epochs=4, train_holdout_ratio=0.25), None
            )
        assert "best-epoch selection disabled" in caplog.text
        assert script.fit_kwargs[0]["has_evaluator"] is False
        assert script.fit_kwargs[0]["n_examples"] == len(tiny)
        assert encoder.training_history == {}

    def test_a_ratio_that_rounds_to_nothing_leaves_every_item_training(self, script):
        """Small classes: floor(ratio * count) == 0 per class, so no holdout at all —
        selection is off, and no item is lost to it."""
        encoder = _train(_config(train_epochs=4, train_holdout_ratio=0.05))
        assert script.fit_kwargs[0]["n_examples"] == N_CLASSES * PER_CLASS
        assert script.fit_kwargs[0]["has_evaluator"] is False
        assert encoder.training_history == {}


# --------------------------------------------------------------------------- #
# selection on
# --------------------------------------------------------------------------- #
class TestTrainLoss:
    """`EncoderConfig.train_loss`: a config choice, not a hardcoded default."""

    def test_default_builds_the_previous_hardcoded_loss(self, script):
        _train(_config(train_epochs=1))
        assert [name for name, _ in script.built_losses] == ["multiple_negatives_symmetric_ranking"]

    def test_alternate_loss_is_selected(self, script):
        _train(_config(train_epochs=1, train_loss="multiple_negatives_ranking"))
        assert [name for name, _ in script.built_losses] == ["multiple_negatives_ranking"]

    def test_cached_variant_defaults_mini_batch_size(self, script):
        _train(_config(train_epochs=1, train_loss="cached_multiple_negatives_ranking"))
        (name, kwargs) = script.built_losses[0]
        assert name == "cached_multiple_negatives_ranking"
        assert kwargs["mini_batch_size"] == 32

    def test_train_loss_params_pass_through_to_the_loss_constructor(self, script):
        _train(
            _config(
                train_epochs=1,
                train_loss="multiple_negatives_ranking",
                train_loss_params={"scale": 10.0},
            )
        )
        (name, kwargs) = script.built_losses[0]
        assert kwargs == {"scale": 10.0}

    def test_train_loss_params_override_the_cached_default(self, script):
        _train(
            _config(
                train_epochs=1,
                train_loss="cached_multiple_negatives_ranking",
                train_loss_params={"mini_batch_size": 8},
            )
        )
        (_, kwargs) = script.built_losses[0]
        assert kwargs["mini_batch_size"] == 8

    def test_arbitrary_sentence_transformers_class_name_resolves(self, script):
        """Not one of the three friendly aliases -- reached only via the
        dynamic sentence_transformers.losses.<ClassName> fallback."""
        _train(
            _config(
                train_epochs=1,
                train_loss="CosineSimilarityLoss",
                train_loss_params={"loss_fct": "mse"},
            )
        )
        (name, kwargs) = script.built_losses[0]
        assert name == "CosineSimilarityLoss"
        assert kwargs == {"loss_fct": "mse"}

    def test_unknown_loss_raises_a_clear_error(self, script):
        with pytest.raises(ValueError, match="unknown encoder.train_loss 'NotARealLoss'"):
            _train(_config(train_epochs=1, train_loss="NotARealLoss"))


class TestBestEpochSelection:
    def test_holds_out_a_stratified_slice_and_wires_fit_for_per_epoch_eval(self, script):
        script.good_epochs = {3}
        _train(_config(train_epochs=3, train_holdout_ratio=0.25))
        (call,) = script.fit_kwargs
        # 25% of each class (2 of 8) withheld from the gradient updates
        assert call["n_examples"] == N_CLASSES * (PER_CLASS - 2)
        assert call["has_evaluator"] is True
        assert call["evaluation_steps"] == 0  # exactly one evaluator call per epoch
        assert call["save_best_model"] is False  # we select the epoch ourselves
        assert call["output_path"] is None  # written only after the best is restored

    def test_middle_epoch_wins_and_is_the_encoder_returned(self, script):
        """Epoch 2 is the good one; epochs 3-5 degrade. The returned encoder must
        be epoch 2's reloaded snapshot, not the final weights."""
        script.good_epochs = {2}
        encoder = _train(_config(train_epochs=5, train_holdout_ratio=0.25))
        history = encoder.training_history
        assert history["best_epoch"] == 2
        assert history["epochs_run"] == 5
        assert [e["desc_acc@1"] for e in history["epochs"]] == [0.0, 1.0, 0.0, 0.0, 0.0]
        assert encoder.model.epoch == 2  # the snapshot was reloaded
        assert script.loads, "expected the best epoch to be reloaded from its snapshot"

    def test_last_epoch_best_skips_the_reload(self, script):
        script.good_epochs = {4, 5}
        encoder = _train(_config(train_epochs=5, train_holdout_ratio=0.25))
        # ties go to the earlier epoch, and epoch 4 improved on 1-3
        assert encoder.training_history["best_epoch"] == 4
        assert encoder.model.epoch == 4
        assert script.loads

    def test_still_improving_at_the_end_keeps_the_live_model(self, script):
        script.good_epochs = {3}
        encoder = _train(_config(train_epochs=3, train_holdout_ratio=0.25))
        assert encoder.training_history["best_epoch"] == 3
        assert encoder.model.epoch == 3
        assert not script.loads  # nothing to restore: fit left us on the best epoch

    def test_early_stopping_cuts_the_run_short(self, script):
        script.good_epochs = {1}
        encoder = _train(
            _config(
                train_epochs=20,
                train_holdout_ratio=0.25,
                train_early_stopping_patience=2,
            )
        )
        history = encoder.training_history
        assert history["epochs_run"] == 3  # best at 1, then two flat epochs
        assert history["early_stopped"] is True
        assert history["epochs_requested"] == 20
        assert history["best_epoch"] == 1
        assert encoder.model.epoch == 1

    def test_history_is_persisted_next_to_the_selected_weights(self, script, tmp_path):
        from text_classifier.infrastructure.encoder import SentenceTransformerEncoder

        script.good_epochs = {2}
        out = tmp_path / "enc"
        encoder = _train(_config(train_epochs=4, train_holdout_ratio=0.25), output_path=str(out))
        assert (out / STATE_FILE).exists()
        with open(out / SentenceTransformerEncoder.TRAINING_HISTORY_NAME) as fh:
            persisted = json.load(fh)
        assert persisted == encoder.training_history
        assert persisted["select_metric"] == "desc_acc@1"
        assert persisted["n_fit_items"] == N_CLASSES * (PER_CLASS - 2)
        assert persisted["n_holdout_items"] == N_CLASSES * 2

    def test_snapshot_directory_is_cleaned_up(self, script):
        script.good_epochs = {2}
        _train(_config(train_epochs=4, train_holdout_ratio=0.25))
        snapshot_dirs = {d for d, _ in script.saves}
        assert snapshot_dirs, "expected at least one snapshot"
        assert not any(os.path.exists(d) for d in snapshot_dirs)

    def test_selecting_on_knn_metric_scores_the_example_pool(self, script):
        script.good_epochs = {2}
        encoder = _train(
            _config(
                train_epochs=3,
                train_holdout_ratio=0.25,
                train_select_metric="knn_acc@1",
            )
        )
        assert all("knn_acc@1" in e for e in encoder.training_history["epochs"])
        assert encoder.training_history["select_metric"] == "knn_acc@1"

    def test_desc_metrics_do_not_pay_for_the_pool(self, script):
        script.good_epochs = {2}
        encoder = _train(_config(train_epochs=2, train_holdout_ratio=0.25))
        assert all("knn_acc@1" not in e for e in encoder.training_history["epochs"])

    def test_holdout_split_is_reproducible_across_runs(self, script):
        script.good_epochs = {2}
        first = _train(_config(train_epochs=3, train_holdout_ratio=0.25))
        second = _train(_config(train_epochs=3, train_holdout_ratio=0.25))
        assert first.training_history["epochs"] == second.training_history["epochs"]

    def test_returned_encoder_honors_encode_config(self, script):
        """The selected (possibly reloaded) encoder keeps the encode-time settings —
        a reload must not silently drop the asymmetric prompts."""
        script.good_epochs = {2}
        encoder = _train(
            _config(
                train_epochs=4,
                train_holdout_ratio=0.25,
                query_prompt="query: ",
                document_prompt="passage: ",
            )
        )
        assert encoder.model.epoch == 2  # reloaded
        out = encoder.encode_queries(["item 0 number 0"])
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-5)
        assert encoder._query_prompt == "query: "
        assert encoder._document_prompt == "passage: "
