"""``train_encoder`` and the model-build knobs, against a *real* sentence-transformers.

Fine-tuning goes through ``SentenceTransformerTrainer`` with the ``NO_DUPLICATES``
batch sampler and ``InformationRetrievalEvaluator``-scored best-epoch selection.
Faking that API would test the fake, so these tests drive the real library -- on a
tiny randomly-initialised BERT written to a temp dir, which keeps them offline (no
model download) and fast (a 1-layer, 16-wide model trains in milliseconds on CPU).

The module is skipped, not failed, where the ``sentence-transformers`` extra is not
installed; ``test_encoder_missing_extra.py`` covers that absence.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from text_classifier import ClassDefinition, LabeledItem, LabelSpace  # noqa: E402
from text_classifier.config import EncoderConfig  # noqa: E402
from text_classifier.infrastructure import encoder as enc  # noqa: E402
from text_classifier.infrastructure.encoder import (  # noqa: E402
    SentenceTransformerEncoder,
    _stratified_holdout,
    train_encoder,
)

N_CLASSES = 3
PER_CLASS = 12
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "item",
    "number",
    "description",
    "of",
    "class",
]
VOCAB += [str(i) for i in range(PER_CLASS + N_CLASSES)]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> str:
    """A 1-layer random BERT + WordPiece vocab on disk: a loadable HF checkpoint
    with no pooling module, so SentenceTransformer gives it mean pooling."""
    from transformers import BertConfig, BertModel, BertTokenizer

    path = str(tmp_path_factory.mktemp("tiny-bert"))
    with open(os.path.join(path, "vocab.txt"), "w") as fh:
        fh.write("\n".join(VOCAB))
    BertTokenizer(os.path.join(path, "vocab.txt")).save_pretrained(path)
    BertModel(
        BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    ).save_pretrained(path)
    return path


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


def _config(model_path: str, **overrides) -> EncoderConfig:
    base = dict(
        model_name_or_path=model_path,
        device="cpu",
        encode_batch_size=8,
        train_batch_size=3,  # == N_CLASSES: the largest batch NO_DUPLICATES can fill
        learning_rate=2e-3,  # large enough that a 1-layer toy model actually moves
    )
    base.update(overrides)
    return EncoderConfig(**base)


def _holdout_desc_acc(encoder, items, label_space, config) -> float:
    """Recompute desc_acc@1 on the same stratified holdout ``train_encoder`` used."""
    labels = np.asarray(label_space.encode_labels([it.label for it in items]), dtype=np.int64)
    _, holdout = _stratified_holdout(labels, config.train_holdout_ratio, config.train_holdout_seed)
    q = encoder.encode_queries([items[i].text for i in holdout])
    d = encoder.encode_documents(label_space.descriptions)
    return float((np.argmax(q @ d.T, axis=1) == labels[holdout]).mean())


# --------------------------------------------------------------------------- #
# no selection: every item trains, last epoch wins
# --------------------------------------------------------------------------- #
class TestWithoutSelection:
    def test_single_epoch_returns_an_encoder_and_scores_nothing(self, tiny_model):
        encoder = train_encoder(_items(), _label_space(), _config(tiny_model))
        assert isinstance(encoder, SentenceTransformerEncoder)
        assert encoder.training_history == {}
        emb = encoder.encode(["item 1 number 2"])
        assert emb.shape == (1, 16)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)  # L2-normalized invariant

    def test_zero_holdout_ratio_disables_selection_for_multi_epoch(self, tiny_model):
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.0)
        assert train_encoder(_items(), _label_space(), cfg).training_history == {}

    def test_too_small_a_holdout_falls_back_to_the_last_epoch(self, tiny_model, caplog):
        # 12 items/class * 0.1 -> 1 per class = 3 held out, below the 4-item floor
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.1)
        items = [it for it in _items() if int(it.text.split()[-1]) < 5]  # 5/class -> 0 held out
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.3)  # 5*.3 -> 1/class = 3
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            encoder = train_encoder(items, _label_space(), cfg)
        assert encoder.training_history == {}
        assert "best-epoch selection disabled" in caplog.text

    def test_output_path_is_written_and_reloadable(self, tiny_model, tmp_path):
        out = str(tmp_path / "enc")
        encoder = train_encoder(_items(), _label_space(), _config(tiny_model), out)
        reloaded = SentenceTransformerEncoder.load(out, 8, "cpu")
        texts = ["item 0 number 3", "item 2 number 7"]
        np.testing.assert_allclose(encoder.encode(texts), reloaded.encode(texts), atol=1e-5)


# --------------------------------------------------------------------------- #
# best-epoch selection (Trainer + InformationRetrievalEvaluator)
# --------------------------------------------------------------------------- #
class TestBestEpochSelection:
    def test_history_has_one_scored_row_per_epoch(self, tiny_model):
        cfg = _config(tiny_model, train_epochs=3, train_holdout_ratio=0.25)
        h = train_encoder(_items(), _label_space(), cfg).training_history
        assert h["select_metric"] == "desc_acc@1"
        assert h["epochs_requested"] == 3 and h["epochs_run"] == 3
        assert [e["epoch"] for e in h["epochs"]] == [1, 2, 3]
        for row in h["epochs"]:
            assert 0.0 <= row["desc_acc@1"] <= 1.0
            assert 0.0 < row["desc_mrr"] <= 1.0
            assert "knn_acc@1" not in row  # not selected on, so the pool is not paid for
        assert 1 <= h["best_epoch"] <= 3
        assert h["n_holdout_items"] == 9 and h["n_fit_items"] == 27
        assert h["early_stopped"] is False
        json.dumps(h)  # JSON-clean: it is written next to the model

    def test_returned_encoder_scores_its_best_epoch(self, tiny_model):
        """The Trainer reloads the best checkpoint into the live model, so what comes
        back must reproduce the best epoch's recorded score -- not the last epoch's."""
        cfg = _config(tiny_model, train_epochs=4, train_holdout_ratio=0.25, learning_rate=5e-2)
        items, ls = _items(), _label_space()
        encoder = train_encoder(items, ls, cfg)
        h = encoder.training_history
        best = h["epochs"][h["best_epoch"] - 1]["desc_acc@1"]
        assert _holdout_desc_acc(encoder, items, ls, cfg) == pytest.approx(best)
        assert best == max(e["desc_acc@1"] for e in h["epochs"])

    def test_ties_go_to_the_earlier_epoch(self, tiny_model):
        cfg = _config(tiny_model, train_epochs=3, train_holdout_ratio=0.25, learning_rate=1e-9)
        h = train_encoder(_items(), _label_space(), cfg).training_history
        scores = [e["desc_acc@1"] for e in h["epochs"]]
        assert h["best_epoch"] == scores.index(max(scores)) + 1

    def test_early_stopping_cuts_the_run_short(self, tiny_model):
        # min_delta 1.0 is unbeatable, so after one stale epoch patience is spent
        cfg = _config(
            tiny_model,
            train_epochs=6,
            train_holdout_ratio=0.25,
            train_early_stopping_patience=1,
            train_select_min_delta=1.0,
        )
        h = train_encoder(_items(), _label_space(), cfg).training_history
        assert h["early_stopped"] is True
        assert h["epochs_run"] < 6
        assert len(h["epochs"]) == h["epochs_run"]

    def test_no_early_stopping_without_patience(self, tiny_model):
        cfg = _config(tiny_model, train_epochs=3, train_holdout_ratio=0.25)
        assert train_encoder(_items(), _label_space(), cfg).training_history["epochs_run"] == 3

    def test_selecting_on_knn_scores_the_example_pool(self, tiny_model):
        cfg = _config(
            tiny_model, train_epochs=2, train_holdout_ratio=0.25, train_select_metric="knn_acc@1"
        )
        h = train_encoder(_items(), _label_space(), cfg).training_history
        assert h["select_metric"] == "knn_acc@1"
        for row in h["epochs"]:
            assert 0.0 <= row["knn_acc@1"] <= 1.0
            assert "desc_acc@1" in row  # the desc evaluator always runs alongside

    def test_selecting_on_mrr(self, tiny_model):
        cfg = _config(
            tiny_model, train_epochs=2, train_holdout_ratio=0.25, train_select_metric="desc_mrr"
        )
        h = train_encoder(_items(), _label_space(), cfg).training_history
        best = h["epochs"][h["best_epoch"] - 1]["desc_mrr"]
        assert best == max(e["desc_mrr"] for e in h["epochs"])

    def test_history_is_persisted_next_to_the_selected_weights(self, tiny_model, tmp_path):
        out = str(tmp_path / "enc")
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.25)
        encoder = train_encoder(_items(), _label_space(), cfg, out)
        with open(os.path.join(out, SentenceTransformerEncoder.TRAINING_HISTORY_NAME)) as fh:
            assert json.load(fh) == encoder.training_history

    def test_working_directory_is_removed(self, tiny_model, monkeypatch):
        made: list = []
        real = tempfile.mkdtemp

        def spy(*a, **k):
            path = real(*a, **k)
            made.append(path)
            return path

        monkeypatch.setattr(enc.tempfile, "mkdtemp", spy)
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.25)
        train_encoder(_items(), _label_space(), cfg)
        trainer_dirs = [p for p in made if "tc-encoder-train-" in p]
        assert trainer_dirs and not any(os.path.exists(p) for p in trainer_dirs)

    def test_holdout_split_and_training_are_reproducible(self, tiny_model):
        cfg = _config(tiny_model, train_epochs=2, train_holdout_ratio=0.25)
        a = train_encoder(_items(), _label_space(), cfg)
        b = train_encoder(_items(), _label_space(), cfg)
        assert a.training_history["epochs"] == b.training_history["epochs"]
        np.testing.assert_allclose(
            a.encode(["item 0 number 1"]), b.encode(["item 0 number 1"]), atol=1e-5
        )


# --------------------------------------------------------------------------- #
# Trainer arguments: sampler, learning rate, weight decay, device
# --------------------------------------------------------------------------- #
class TestTrainingArguments:
    def test_uses_the_no_duplicates_sampler(self, tmp_path):
        args = enc._training_arguments(_config("x"), str(tmp_path), None)
        assert getattr(args.batch_sampler, "value", args.batch_sampler) == "no_duplicates"

    def test_learning_rate_and_weight_decay_come_from_config(self, tmp_path):
        cfg = _config("x", learning_rate=7e-5, weight_decay=0.2)
        args = enc._training_arguments(cfg, str(tmp_path), None)
        assert args.learning_rate == 7e-5 and args.weight_decay == 0.2

    def test_defaults_match_what_model_fit_used(self, tmp_path):
        args = enc._training_arguments(EncoderConfig(), str(tmp_path), None)
        assert args.learning_rate == 2e-5 and args.weight_decay == 0.01

    def test_warmup_fraction_is_carried(self, tmp_path):
        args = enc._training_arguments(_config("x", warmup_ratio=0.2), str(tmp_path), None)
        assert 0.2 in (getattr(args, "warmup_ratio", None), getattr(args, "warmup_steps", None))

    def test_nothing_is_reported_externally(self, tmp_path):
        args = enc._training_arguments(_config("x"), str(tmp_path), None)
        assert list(args.report_to) == []  # air-gapped hosts: no tracker, no network

    def test_without_selection_nothing_is_evaluated_or_saved(self, tmp_path):
        args = enc._training_arguments(_config("x"), str(tmp_path), None)
        assert getattr(args.save_strategy, "value", args.save_strategy) == "no"
        assert not args.load_best_model_at_end

    def test_with_selection_the_best_checkpoint_is_reloaded(self, tmp_path):
        args = enc._training_arguments(_config("x"), str(tmp_path), "eval_desc_cosine_accuracy@1")
        assert args.load_best_model_at_end and args.greater_is_better
        assert args.metric_for_best_model == "eval_desc_cosine_accuracy@1"
        assert args.save_only_model  # no optimizer-state pickles
        assert args.save_total_limit == 1

    def test_cpu_device_pins_the_trainer_to_cpu(self, tmp_path):
        assert enc._training_arguments(_config("x", device="cpu"), str(tmp_path), None).use_cpu

    def test_indexed_cuda_device_warns_that_the_trainer_ignores_the_index(self, caplog):
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            enc._device_notes(_config("x", device="cuda:1"))
        assert "CUDA_VISIBLE_DEVICES" in caplog.text


# --------------------------------------------------------------------------- #
# loss choice and scale
# --------------------------------------------------------------------------- #
class TestTrainLoss:
    def test_scale_reaches_the_loss(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        losses = enc._losses_module()
        model = SentenceTransformer(tiny_model, device="cpu")
        loss = enc._build_train_loss("multiple_negatives_ranking", model, losses, {"scale": 5.0})
        assert isinstance(loss, losses.MultipleNegativesRankingLoss)
        assert loss.scale == 5.0

    def test_cached_variant_defaults_its_mini_batch_size(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        losses = enc._losses_module()
        model = SentenceTransformer(tiny_model, device="cpu")
        loss = enc._build_train_loss("cached_multiple_negatives_ranking", model, losses, {})
        assert loss.mini_batch_size == 32

    def test_unknown_loss_raises_a_clear_error(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        losses = enc._losses_module()
        model = SentenceTransformer(tiny_model, device="cpu")
        with pytest.raises(ValueError, match="unknown encoder.train_loss"):
            enc._build_train_loss("NoSuchLoss", model, losses, {})

    @pytest.mark.parametrize(
        "name",
        [
            "multiple_negatives_symmetric_ranking",
            "multiple_negatives_ranking",
            "cached_multiple_negatives_ranking",
        ],
    )
    def test_each_alias_actually_trains(self, tiny_model, name):
        cfg = _config(tiny_model, train_loss=name, train_loss_params={"scale": 10.0})
        assert train_encoder(_items(), _label_space(), cfg).encode(["item 0 number 1"]).shape == (
            1,
            16,
        )


# --------------------------------------------------------------------------- #
# pooling_mode
# --------------------------------------------------------------------------- #
def _pooling_mode_str(model) -> str:
    models = enc._models_module()
    pooling = next(m for m in model if isinstance(m, models.Pooling))
    # older sentence-transformers: get_pooling_mode_str(); 6.x: a `pooling_mode` attribute
    if hasattr(pooling, "get_pooling_mode_str"):
        return pooling.get_pooling_mode_str()
    return pooling.pooling_mode


class TestPoolingMode:
    def test_none_leaves_the_model_untouched(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(tiny_model, device="cpu")
        assert enc._apply_model_overrides(model, _config(tiny_model)) is model
        assert _pooling_mode_str(model) == "mean"  # a plain HF checkpoint defaults to mean

    @pytest.mark.parametrize("mode", ["mean", "cls", "max", "weightedmean", "mean_sqrt_len_tokens"])
    def test_each_mode_builds_a_working_encoder(self, tiny_model, mode):
        encoder = SentenceTransformerEncoder.load(
            tiny_model,
            8,
            "cpu",
            config=_config(tiny_model, pooling_mode=mode),
            apply_model_overrides=True,
        )
        assert _pooling_mode_str(encoder.model) == mode
        emb = encoder.encode(["item 1 number 2", "item 2 number 9"])
        assert emb.shape == (2, 16) and np.isfinite(emb).all()

    def test_lasttoken_is_supported_or_fails_with_a_clear_error(self, tiny_model):
        cfg = _config(tiny_model, pooling_mode="lasttoken")
        try:
            encoder = SentenceTransformerEncoder.load(
                tiny_model, 8, "cpu", config=cfg, apply_model_overrides=True
            )
        except ValueError as exc:  # an older sentence-transformers without lasttoken
            assert "not supported by the installed sentence-transformers" in str(exc)
        else:
            assert _pooling_mode_str(encoder.model) == "lasttoken"

    def test_different_modes_embed_differently(self, tiny_model):
        def embed(mode):
            return SentenceTransformerEncoder.load(
                tiny_model,
                8,
                "cpu",
                config=_config(tiny_model, pooling_mode=mode),
                apply_model_overrides=True,
            ).encode(["item 1 number 2 description of class 0"])

        assert not np.allclose(embed("cls"), embed("max"), atol=1e-4)

    def test_pooling_survives_fine_tune_and_reload(self, tiny_model, tmp_path):
        out = str(tmp_path / "enc")
        cfg = _config(tiny_model, pooling_mode="cls")
        trained = train_encoder(_items(), _label_space(), cfg, out)
        assert _pooling_mode_str(trained.model) == "cls"
        # A saved dir carries its pooling: reloading with *no* override keeps it.
        reloaded = SentenceTransformerEncoder.load(out, 8, "cpu", config=cfg)
        assert _pooling_mode_str(reloaded.model) == "cls"
        np.testing.assert_allclose(
            trained.encode(["item 2 number 4"]), reloaded.encode(["item 2 number 4"]), atol=1e-5
        )

    def test_overrides_are_not_reapplied_on_reload(self, tiny_model, tmp_path):
        """`load` also reloads saved model dirs; without the flag the config's
        pooling_mode must not rebuild what the dir already carries."""
        out = str(tmp_path / "enc")
        train_encoder(_items(), _label_space(), _config(tiny_model, pooling_mode="max"), out)
        reloaded = SentenceTransformerEncoder.load(
            out, 8, "cpu", config=_config(tiny_model, pooling_mode="cls")
        )
        assert _pooling_mode_str(reloaded.model) == "max"

    def test_replacing_the_models_own_pooling_warns(self, tiny_model, caplog):
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            SentenceTransformerEncoder.load(
                tiny_model,
                8,
                "cpu",
                config=_config(tiny_model, pooling_mode="cls"),
                apply_model_overrides=True,
            )
        assert "replaces the model's own pooling" in caplog.text

    def test_setting_the_models_own_mode_does_not_warn_about_replacement(self, tiny_model, caplog):
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            SentenceTransformerEncoder.load(
                tiny_model,
                8,
                "cpu",
                config=_config(tiny_model, pooling_mode="mean"),
                apply_model_overrides=True,
            )
        assert "replaces the model's own pooling" not in caplog.text

    def test_post_pooling_modules_are_dropped_with_a_warning(self, tiny_model, caplog):
        from sentence_transformers import SentenceTransformer

        models = enc._models_module()
        base = SentenceTransformer(tiny_model, device="cpu")
        with_norm = SentenceTransformer(modules=[*list(base), models.Normalize()], device="cpu")
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            rebuilt = enc._with_pooling(with_norm, "cls")
        assert [type(m).__name__ for m in rebuilt] == ["Transformer", "Pooling"]
        assert "Normalize" in caplog.text

    def test_the_models_prompts_are_carried_over(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        base = SentenceTransformer(tiny_model, device="cpu")
        base.prompts = {"query": "query: ", "document": "passage: "}
        rebuilt = enc._with_pooling(base, "cls")
        assert rebuilt.prompts == {"query": "query: ", "document": "passage: "}

    def test_unsupported_mode_raises_a_clear_error(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        with pytest.raises(ValueError, match="encoder.pooling_mode='median'"):
            enc._with_pooling(SentenceTransformer(tiny_model, device="cpu"), "median")


# --------------------------------------------------------------------------- #
# max_seq_length
# --------------------------------------------------------------------------- #
class TestMaxSeqLength:
    LONG = " ".join(["item 1 number 2"] * 6)  # 24 tokens

    def _loaded(self, tiny_model, **overrides):
        return SentenceTransformerEncoder.load(
            tiny_model,
            8,
            "cpu",
            config=_config(tiny_model, **overrides),
            apply_model_overrides=True,
        )

    def test_none_keeps_the_models_default(self, tiny_model):
        from sentence_transformers import SentenceTransformer

        default = SentenceTransformer(tiny_model, device="cpu").max_seq_length
        assert self._loaded(tiny_model).model.max_seq_length == default

    def test_setting_it_truncates_what_gets_encoded(self, tiny_model):
        short = self._loaded(tiny_model, max_seq_length=4)
        long = self._loaded(tiny_model, max_seq_length=60)
        assert short.model.max_seq_length == 4
        # Truncation changes the embedding of a text longer than the cap...
        assert not np.allclose(short.encode([self.LONG]), long.encode([self.LONG]), atol=1e-4)
        # ...and cannot change one that already fits: [CLS] item 1 [SEP] is 4 tokens.
        np.testing.assert_allclose(short.encode(["item 1"]), long.encode(["item 1"]), atol=1e-5)

    def test_it_applies_to_fine_tuning_and_is_persisted_with_the_model(self, tiny_model, tmp_path):
        out = str(tmp_path / "enc")
        cfg = _config(tiny_model, max_seq_length=7)
        trained = train_encoder(_items(), _label_space(), cfg, out)
        assert trained.model.max_seq_length == 7
        # Reloaded with the plain config (no override flag): the saved dir keeps it.
        assert (
            SentenceTransformerEncoder.load(
                out, 8, "cpu", config=EncoderConfig()
            ).model.max_seq_length
            == 7
        )

    def test_a_length_beyond_the_position_embeddings_warns(self, tiny_model, caplog):
        with caplog.at_level(logging.WARNING, logger=enc.logger.name):
            self._loaded(tiny_model, max_seq_length=500)  # tiny model addresses 64
        assert "exceeds the model's max_position_embeddings" in caplog.text

    def test_it_combines_with_a_pooling_mode(self, tiny_model):
        loaded = self._loaded(tiny_model, pooling_mode="cls", max_seq_length=5)
        assert _pooling_mode_str(loaded.model) == "cls"
        assert loaded.model.max_seq_length == 5


# --------------------------------------------------------------------------- #
# the epoch evaluator
# --------------------------------------------------------------------------- #
class TestEpochEvaluator:
    def _setup(self, tiny_model, **cfg_overrides):
        from sentence_transformers import SentenceTransformer

        items, ls = _items(), _label_space()
        cfg = _config(tiny_model, train_holdout_ratio=0.25, **cfg_overrides)
        labels = np.asarray(ls.encode_labels([it.label for it in items]), dtype=np.int64)
        fit, hold = _stratified_holdout(labels, 0.25, 0)
        texts = [it.text for it in items]
        model = SentenceTransformer(tiny_model, device="cpu")
        return model, cfg, texts, labels, fit, hold, ls.descriptions

    def test_desc_accuracy_matches_a_direct_computation(self, tiny_model):
        model, cfg, texts, labels, fit, hold, descs = self._setup(tiny_model)
        evaluator = enc._build_epoch_evaluator(cfg, texts, labels, fit, hold, descs, with_knn=False)
        metrics = evaluator(model)
        encoder = SentenceTransformerEncoder(model, 8)
        q = encoder.encode_queries([texts[i] for i in hold])
        d = encoder.encode_documents(descs)
        expected = float((np.argmax(q @ d.T, axis=1) == labels[hold]).mean())
        key = enc._selection_key("desc_acc@1", len(descs)).removeprefix("eval_")
        assert metrics[key] == pytest.approx(expected)

    def test_knn_accuracy_matches_a_direct_computation(self, tiny_model):
        model, cfg, texts, labels, fit, hold, descs = self._setup(tiny_model)
        evaluator = enc._build_epoch_evaluator(cfg, texts, labels, fit, hold, descs, with_knn=True)
        metrics = evaluator(model)
        encoder = SentenceTransformerEncoder(model, 8)
        q = encoder.encode_queries([texts[i] for i in hold])
        pool = encoder.encode_documents([texts[i] for i in fit])
        nearest = np.argmax(q @ pool.T, axis=1)
        expected = float((labels[fit][nearest] == labels[hold]).mean())
        key = enc._selection_key("knn_acc@1", len(descs)).removeprefix("eval_")
        assert metrics[key] == pytest.approx(expected)

    def test_the_pool_never_contains_a_holdout_item(self, tiny_model):
        _, _, _, _, fit, hold, _ = self._setup(tiny_model)
        assert not set(fit.tolist()) & set(hold.tolist())

    def test_desc_mrr_is_a_full_ranking_not_truncated_at_ten(self, tiny_model):
        model, cfg, texts, labels, fit, hold, descs = self._setup(tiny_model)
        metrics = enc._build_epoch_evaluator(cfg, texts, labels, fit, hold, descs, with_knn=False)(
            model
        )
        assert f"desc_cosine_mrr@{len(descs)}" in metrics

    def test_prompts_follow_the_encoders_roles(self, tiny_model, monkeypatch):
        seen: dict = {}
        real = enc._import_first

        def spy(candidates, name):
            cls = real(candidates, name)
            if name != "InformationRetrievalEvaluator":
                return cls

            def wrapped(*a, **k):
                seen.update(k)
                return cls(*a, **k)

            return wrapped

        monkeypatch.setattr(enc, "_import_first", spy)
        model, cfg, texts, labels, fit, hold, descs = self._setup(
            tiny_model, query_prompt="query: ", document_prompt="passage: "
        )
        enc._build_epoch_evaluator(cfg, texts, labels, fit, hold, descs, with_knn=False)
        assert seen["query_prompt"] == "query: " and seen["corpus_prompt"] == "passage: "

    def test_no_prompt_args_are_passed_when_none_are_configured(self, tiny_model, monkeypatch):
        seen: dict = {}
        real = enc._import_first

        def spy(candidates, name):
            cls = real(candidates, name)
            if name != "InformationRetrievalEvaluator":
                return cls

            def wrapped(*a, **k):
                seen.update(k)
                return cls(*a, **k)

            return wrapped

        monkeypatch.setattr(enc, "_import_first", spy)
        model, cfg, texts, labels, fit, hold, descs = self._setup(tiny_model)
        enc._build_epoch_evaluator(cfg, texts, labels, fit, hold, descs, with_knn=False)
        assert not {"query_prompt", "corpus_prompt", "truncate_dim"} & set(seen)
