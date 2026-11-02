"""T33 phase 3 — the real cross-encoder backend and its fine-tuning, end to end.

Drives the real sentence-transformers ``CrossEncoder`` on a tiny randomly
initialised BERT written to a temp dir: offline (no download) and fast. Skipped,
not failed, where the ``sentence-transformers`` extra is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from text_classifier.application import training as training_mod  # noqa: E402
from text_classifier.application.inference import InferencePipeline  # noqa: E402
from text_classifier.application.training import TrainingPipeline  # noqa: E402
from text_classifier.config import (  # noqa: E402
    CrossEncoderConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from text_classifier.infrastructure import ArtifactRepository  # noqa: E402
from text_classifier.infrastructure.registry import build_reranker  # noqa: E402
from text_classifier.infrastructure.reranker import (  # noqa: E402
    SentenceTransformersCrossEncoder,
    train_cross_encoder,
)
from tests._doubles import make_synthetic  # noqa: E402

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _write_tiny_cross_encoder(path: str, words, *, learnable: bool = False, seed: int = 0) -> str:
    """A random BERT cross-encoder on disk, seeded so every session gets the same
    weights (unseeded, a test's outcome would depend on the draw).

    The default is the smallest model that exercises the plumbing. It cannot learn
    query/document interaction: verified with a plain PyTorch loop, it collapses
    to a constant score. ``learnable=True`` is 64-wide, 2 layers, no dropout --
    the smallest configuration that reliably learns the toy ranking task, for the
    tests that assert fine-tuning actually improves ranking."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    with open(os.path.join(path, "vocab.txt"), "w") as fh:
        fh.write("\n".join(SPECIAL + sorted(set(words))))
    tok = BertTokenizer(os.path.join(path, "vocab.txt"))
    tok.save_pretrained(path)
    size = (
        dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
        if learnable
        else dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    )
    dropout = dict(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0) if learnable else {}
    torch.manual_seed(seed)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=tok.vocab_size,
            max_position_embeddings=128,
            num_labels=1,
            **size,
            **dropout,
        )
    ).save_pretrained(path)
    return path


def _dataset():
    return make_synthetic(n_classes=4, per_class=12, seed=23)


@pytest.fixture(scope="module")
def tiny_ce(tmp_path_factory) -> str:
    ls, items = _dataset()
    words = {w for it in items for w in it.text.split()}
    words |= {w for d in ls.descriptions for w in d.split()}
    return _write_tiny_cross_encoder(str(tmp_path_factory.mktemp("tiny-ce")), words)


def _ce_config(path: str, **kw) -> CrossEncoderConfig:
    base = dict(kind="cross-encoder", model_name=path, device="cpu", batch_size=8)
    base.update(kw)
    return CrossEncoderConfig(**base)


def _hf_logits(path: str, pairs):
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    model = BertForSequenceClassification.from_pretrained(path).eval()
    tok = BertTokenizer.from_pretrained(path)
    batch = tok([q for q, _ in pairs], [d for _, d in pairs], return_tensors="pt", padding=True)
    with torch.no_grad():
        return model(**batch).logits[:, 0].numpy()


PAIRS = [
    ("w76 w21 w412", "class about w76 w251 w68"),
    ("w130 w496", "class about w362 w382"),
    ("w157 w21", "class about w157 w21"),
]


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #
class TestBackend:
    def test_registered_under_cross_encoder(self, tiny_ce):
        assert isinstance(build_reranker(_ce_config(tiny_ce)), SentenceTransformersCrossEncoder)

    def test_scores_are_the_models_raw_logits_not_squashed(self, tiny_ce):
        """The port requires raw logits. CrossEncoder.predict's default differs by
        version (5.6+ applies a sigmoid, 5.2 does not), so it is forced."""
        got = build_reranker(_ce_config(tiny_ce)).score(PAIRS)
        assert got.dtype == np.float32 and got.shape == (3,)
        np.testing.assert_allclose(got, _hf_logits(tiny_ce, PAIRS), atol=1e-5)

    def test_order_is_preserved_and_batching_does_not_change_scores(self, tiny_ce):
        small = build_reranker(_ce_config(tiny_ce, batch_size=1)).score(PAIRS)
        big = build_reranker(_ce_config(tiny_ce, batch_size=64)).score(PAIRS)
        np.testing.assert_allclose(small, big, atol=1e-5)
        reversed_scores = build_reranker(_ce_config(tiny_ce)).score(PAIRS[::-1])
        np.testing.assert_allclose(reversed_scores[::-1], big, atol=1e-5)

    def test_no_pairs_scores_nothing(self, tiny_ce):
        assert build_reranker(_ce_config(tiny_ce)).score([]).shape == (0,)

    def test_save_load_round_trip_is_portable_and_identical(self, tiny_ce, tmp_path):
        ce = build_reranker(_ce_config(tiny_ce))
        out = str(tmp_path / "ce")
        ce.save(out)
        files = os.listdir(out)
        assert "model.safetensors" in files
        assert not any(f.endswith((".bin", ".pkl", ".pt")) for f in files)  # no pickle
        with open(os.path.join(out, "reranker.json")) as fh:
            assert json.load(fh) == {"kind": "cross-encoder"}
        back = SentenceTransformersCrossEncoder.load(out, config=_ce_config(tiny_ce))
        np.testing.assert_allclose(back.score(PAIRS), ce.score(PAIRS), atol=1e-6)

    def test_multi_label_models_are_rejected(self, tmp_path):
        from transformers import BertConfig, BertForSequenceClassification

        path = _write_tiny_cross_encoder(str(tmp_path), ["a", "b"])
        cfg = BertConfig.from_pretrained(path)
        cfg.num_labels = 3
        BertForSequenceClassification(cfg).save_pretrained(path)
        with pytest.raises(ValueError, match="num_labels=3"):
            build_reranker(_ce_config(path))


# --------------------------------------------------------------------------- #
# fine-tuning
# --------------------------------------------------------------------------- #
EXAMPLES = [
    ("w76 w21", "class about w76 w21", ["class about w362 w382", "class about w130 w496"]),
    ("w362 w382", "class about w362 w382", ["class about w76 w21", "class about w130 w496"]),
    ("w130 w496", "class about w130 w496", ["class about w76 w21", "class about w362 w382"]),
] * 4


class TestFineTuning:
    @pytest.mark.parametrize("loss", ["binary_cross_entropy", "lambda", "listnet"])
    def test_each_loss_trains_and_moves_the_weights(self, tiny_ce, loss):
        ce = build_reranker(_ce_config(tiny_ce))
        before = ce.score(PAIRS)
        cfg = _ce_config(tiny_ce, train=True, train_loss=loss, learning_rate=5e-3, train_epochs=2)
        train_cross_encoder(ce, EXAMPLES, cfg)
        assert not np.allclose(ce.score(PAIRS), before, atol=1e-6)

    @pytest.mark.parametrize(
        "loss,batch", [("binary_cross_entropy", 9), ("listnet", 3), ("lambda", 3)]
    )
    def test_fine_tuning_learns_to_rank_each_positive_first(self, tmp_path, loss, batch):
        """Each query's positive must outrank its own negatives after training.
        Untrained, the scores are near-identical (random init), so this measures
        learning, not noise. Verified to hold for 5/5 seeds per loss."""
        ls, items = _dataset()
        words = {w for it in items for w in it.text.split()} | {
            w for d in ls.descriptions for w in d.split()
        }
        path = _write_tiny_cross_encoder(str(tmp_path), words, learnable=True)
        ce = build_reranker(_ce_config(path))
        cfg = _ce_config(
            path,
            train=True,
            train_loss=loss,
            learning_rate=1e-3,
            train_epochs=300,
            train_batch_size=batch,
            warmup_ratio=0.0,
        )
        train_cross_encoder(ce, EXAMPLES[:3], cfg)
        for query, positive, negatives in EXAMPLES[:3]:
            scores = ce.score([(query, positive)] + [(query, n) for n in negatives])
            assert scores[0] > scores[1:].max(), (query, scores)

    def test_no_examples_is_a_no_op(self, tiny_ce):
        ce = build_reranker(_ce_config(tiny_ce))
        before = ce.score(PAIRS)
        train_cross_encoder(ce, [], _ce_config(tiny_ce, train=True))
        np.testing.assert_array_equal(ce.score(PAIRS), before)


# --------------------------------------------------------------------------- #
# the pipeline: per-fold fine-tuning, leakage, persistence
# --------------------------------------------------------------------------- #
def _pipeline_config(tiny_ce: str, train: bool) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.signals = ["dense", "lexical", "cross-encoder"]
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=5)
    cfg.retrieval.cross_encoder = _ce_config(
        tiny_ce, train=train, train_negatives=3, learning_rate=5e-3
    )
    cfg.validate()
    return cfg


class TestPipeline:
    def test_pretrained_cross_encoder_runs_end_to_end(self, tiny_ce, tmp_path):
        ls, items = _dataset()
        artifacts, report = TrainingPipeline(_pipeline_config(tiny_ce, train=False)).run(
            items, ls, output_dir=str(tmp_path / "m")
        )
        assert report.n_items > 0
        meta = json.load(open(tmp_path / "m" / "meta.json"))
        assert any(c.startswith("ce_pos") for c in meta["feature_names"])

    def test_each_fold_fine_tunes_a_fresh_model_on_its_training_rows_only(
        self, tiny_ce, monkeypatch
    ):
        """The leakage rule: fold f's cross-encoder trains on fold f's training
        rows, never its held-out rows, and starts from model_name -- not from the
        previous fold's fine-tuned weights."""
        ls, items = _dataset()
        cfg = _pipeline_config(tiny_ce, train=True)
        base = build_reranker(cfg.retrieval.cross_encoder).score(PAIRS)

        seen_rows, starts = [], []
        real_fit = TrainingPipeline._fit_cross_encoder
        real_train = training_mod.train_cross_encoder

        def spy_fit(self, signal_providers, rows, *args, **kwargs):
            seen_rows.append(np.asarray(rows).copy())
            return real_fit(self, signal_providers, rows, *args, **kwargs)

        def spy_train(reranker, examples, config):
            starts.append(reranker.score(PAIRS))  # weights at the start of training
            return real_train(reranker, examples, config)

        monkeypatch.setattr(TrainingPipeline, "_fit_cross_encoder", spy_fit)
        monkeypatch.setattr(training_mod, "train_cross_encoder", spy_train)
        TrainingPipeline(cfg).run(items, ls)

        y = np.array(ls.encode_labels([it.label for it in items]))
        skf = StratifiedKFold(3, shuffle=True, random_state=0)
        folds = list(skf.split([it.text for it in items], y))
        assert len(seen_rows) == 3 + 1  # n_folds per-fold fits + the deployment fit
        for rows, (tr, va) in zip(seen_rows[:3], folds):
            np.testing.assert_array_equal(np.sort(rows), np.sort(tr))
            assert not set(rows.tolist()) & set(va.tolist())
        np.testing.assert_array_equal(np.sort(seen_rows[3]), np.arange(len(items)))
        assert len(starts) == 4
        for s in starts:  # every fine-tune began from the pretrained weights
            np.testing.assert_allclose(s, base, atol=1e-6)

    def test_fine_tuned_model_ships_and_reloads_as_trained(self, tiny_ce, tmp_path):
        ls, items = _dataset()
        out = str(tmp_path / "m")
        cfg = _pipeline_config(tiny_ce, train=True)
        artifacts, _ = TrainingPipeline(cfg).run(items, ls, output_dir=out)
        ce_dir = os.path.join(out, "signals", "cross-encoder")
        assert "model.safetensors" in os.listdir(ce_dir)

        shipped = next(p for p in artifacts.signal_providers if hasattr(p, "reranker"))
        reloaded_artifacts = ArtifactRepository().load(out)
        reloaded = next(p for p in reloaded_artifacts.signal_providers if hasattr(p, "reranker"))
        trained = shipped.reranker.score(PAIRS)
        np.testing.assert_allclose(reloaded.reranker.score(PAIRS), trained, atol=1e-5)
        # ...and it is the fine-tuned model, not the pretrained one
        pretrained = build_reranker(cfg.retrieval.cross_encoder).score(PAIRS)
        assert not np.allclose(trained, pretrained, atol=1e-6)

        texts = [it.text for it in items[:5]]
        a = InferencePipeline(artifacts).predict(texts)
        b = InferencePipeline(reloaded_artifacts).predict(texts)
        assert [p.predicted_key for p in a] == [p.predicted_key for p in b]
        np.testing.assert_allclose([p.confidence for p in a], [p.confidence for p in b], atol=1e-5)
