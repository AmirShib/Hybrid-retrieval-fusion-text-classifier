"""T94: the learned-sparse (SPLADE) signal.

Offline: the ``hashing`` backend exercises every piece of plumbing (per-fold
fit, schema, candidate union, persistence, LOO self-masking, no-retrain update).
The real ``splade`` backend is exercised against a tiny randomly initialised
BERT built on the fly — no download — and skipped when torch is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests._doubles import make_synthetic
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.application.updating import update
from text_classifier.config import PipelineConfig, RetrievalConfig, SpladeConfig, TrainingConfig
from text_classifier.domain import LabeledItem, LabelSpace, SignalContext
from text_classifier.infrastructure import ArtifactRepository
from text_classifier.infrastructure.splade import HashingSparseEncoder, SpladeSignalProvider

SPLADE_COLS = SpladeSignalProvider().column_names()


def _train(cfg, n_folds=3):
    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=5)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=n_folds, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    return TrainingPipeline(cfg).run(items, label_space) + (items, label_space)


def _ls():
    from text_classifier.domain import ClassDefinition

    return LabelSpace(
        [
            ClassDefinition("bank", "bank account loan deposit"),
            ClassDefinition("food", "restaurant pizza dinner menu"),
            ClassDefinition("car", "engine wheel garage repair"),
        ]
    )


# ---------------------------------------------------------------- encoder/provider
def test_hashing_encoder_is_deterministic_and_sparse():
    enc = HashingSparseEncoder(1024)
    a = enc.encode(["Loan loan bank", ""])
    b = enc.encode(["Loan loan bank", ""])
    assert a.shape == (2, 1024)
    assert (a != b).nnz == 0
    assert a[1].nnz == 0
    assert a[0].nnz == 2 and np.isclose(a[0].data.max(), 1 + np.log(2))


def test_provider_scores_overlap_and_marks_no_overlap_missing():
    ls = _ls()
    p = SpladeSignalProvider(SpladeConfig(hashing_dim=4096))
    p.fit(["my loan at the bank", "pizza for dinner", "engine repair"], np.array([0, 1, 2]), ls)
    ctx = SignalContext(
        texts=["bank loan please", "zzz unrelated"],
        q_emb=np.zeros((2, 1)),
        k=3,
        n_classes=3,
        label_space=ls,
    )
    desc, knn = p.build(ctx)
    assert desc.node == "splade.desc" and knn.node == "splade.knn"
    assert np.nanargmax(desc.value[0]) == 0
    assert np.isnan(desc.value[0, 1]) and np.isnan(desc.value[1]).all()  # NaN, never 0
    assert knn.value[0, 0] > 0 and np.isnan(knn.value[0, 1])
    assert np.isnan(knn.value[1]).all()
    assert knn.top1_idx.tolist() == [0, -1]


def test_self_ids_mask_the_query_out_of_its_own_neighbours():
    ls = _ls()
    texts = ["my loan at the bank", "pizza for dinner", "engine repair"]
    p = SpladeSignalProvider(SpladeConfig(hashing_dim=4096))
    p.fit(texts, np.array([0, 1, 2]), ls)
    ctx = SignalContext(
        texts=texts,
        q_emb=np.zeros((3, 1)),
        k=3,
        n_classes=3,
        label_space=ls,
        self_ids=np.arange(3),
    )
    knn = p.build(ctx)[1]
    assert np.isnan(knn.value).all()  # each item's only match was itself


def test_descriptions_follow_the_label_space():
    from text_classifier.domain import ClassDefinition

    ls = _ls()
    p = SpladeSignalProvider(SpladeConfig(hashing_dim=4096))
    p.fit(["bank loan"], np.array([0]), ls)
    bigger = LabelSpace(list(ls.definitions) + [ClassDefinition("pet", "dog cat vet")])
    ctx = SignalContext(
        texts=["vet for my dog"], q_emb=np.zeros((1, 1)), k=2, n_classes=4, label_space=bigger
    )
    desc = p.build(ctx)[0].value
    assert desc.shape == (1, 4) and np.nanargmax(desc[0]) == 3


def test_unfitted_provider_raises_clearly():
    p = SpladeSignalProvider()
    ctx = SignalContext(texts=["x"], q_emb=np.zeros((1, 1)), k=1, n_classes=3, label_space=_ls())
    with pytest.raises(ValueError, match="before it was fitted"):
        p.build(ctx)


# ---------------------------------------------------------------- config
def test_config_round_trips_and_validates():
    cfg = PipelineConfig.from_dict(
        {"signals": ["dense", "lexical", "splade"], "retrieval": {"splade": {"top_terms": 64}}}
    )
    assert cfg.retrieval.splade.top_terms == 64
    cfg.validate()
    bad = PipelineConfig.from_dict({"retrieval": {"splade": {"kind": "splade"}}})
    with pytest.raises(ValueError, match="retrieval.splade.model_name"):
        bad.validate()
    bad = PipelineConfig.from_dict({"retrieval": {"splade": {"train": True}}})
    with pytest.raises(ValueError, match="'splade' in signals"):
        bad.validate()


def test_default_config_is_untouched():
    assert "splade" not in PipelineConfig().signals


# ---------------------------------------------------------------- end to end
def test_end_to_end_train_save_load_predict(tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "splade"]
    artifacts, report, items, _ = _train(cfg)
    assert report.n_items > 0
    assert isinstance(artifacts.signal_providers[-1], SpladeSignalProvider)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)
    with open(os.path.join(model_dir, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["components"]["signals"] == ["dense", "lexical", "splade"]
    assert set(SPLADE_COLS) <= set(meta["feature_names"])
    for f in ("pool.npz", "splade.json", os.path.join("encoder", "sparse_encoder.json")):
        assert os.path.isfile(os.path.join(model_dir, "signals", "splade", f))

    texts = [it.text for it in items[:12]]
    before = InferencePipeline(artifacts).predict(texts)
    after = InferencePipeline(ArtifactRepository().load(model_dir)).predict(texts)
    assert [p.top_key for p in before] == [p.top_key for p in after]
    assert np.allclose([p.confidence for p in before], [p.confidence for p in after])


def test_splade_only_retrieval_still_trains():
    """The signal stands on its own in candidate selection (round one)."""
    cfg = PipelineConfig()
    cfg.signals = ["dense", "splade"]
    _, report, _, _ = _train(cfg)
    assert report.n_items > 0


def test_signal_is_fitted_per_fold_without_leakage(monkeypatch):
    """Every fit sees a strict subset of the corpus except the final deployment fit."""
    sizes = []
    orig = SpladeSignalProvider.fit

    def spy(self, texts, labels, label_space):
        sizes.append(len(texts))
        return orig(self, texts, labels, label_space)

    monkeypatch.setattr(SpladeSignalProvider, "fit", spy)
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "splade"]
    _, _, items, _ = _train(cfg)
    assert len(sizes) == 3 + 1
    assert all(s < len(items) for s in sizes[:3]) and sizes[3] == len(items)


def test_update_indexes_new_examples(tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "splade"]
    artifacts, _, items, ls = _train(cfg)
    before = artifacts.signal_providers[-1]._pool.shape[0]
    new = [LabeledItem("brand new example text", ls.key_at(0))]
    updated, _ = update(artifacts, items, new_items=new)
    assert updated.signal_providers[-1]._pool.shape[0] == before + 1


# ---------------------------------------------------------------- real backend
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_mlm(tmp_path_factory):
    """A tiny random BERT + word-level tokenizer saved to disk: no network."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

    words = [
        "bank",
        "loan",
        "account",
        "deposit",
        "pizza",
        "dinner",
        "menu",
        "restaurant",
        "engine",
        "wheel",
        "garage",
        "repair",
        "class",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "item",
    ]
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    torch.manual_seed(0)
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    )
    path = str(tmp_path_factory.mktemp("tiny_mlm"))
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


def _splade_cfg(path, **kw):
    return SpladeConfig(
        kind="splade",
        model_name=path,
        max_length=32,
        batch_size=4,
        device="cpu",
        top_terms=16,
        **kw,
    )


def test_real_encoder_is_sparse_nonnegative_and_skips_special_tokens(tiny_mlm):
    from text_classifier.infrastructure.splade import SpladeEncoder

    enc = SpladeEncoder.from_pretrained(tiny_mlm, _splade_cfg(tiny_mlm))
    X = enc.encode(["bank loan", "pizza dinner menu", "garage"])
    assert X.shape == (3, enc.vocab_size)
    assert (X.data > 0).all() and max(X[i].nnz for i in range(3)) <= 16
    special = set(enc.tokenizer.all_special_ids)
    assert not special & set(X.indices.tolist())


def test_training_reduces_loss_and_ranks_own_class(tiny_mlm):
    from text_classifier.infrastructure.splade import SpladeEncoder, train_splade

    ls = _ls()
    texts = [
        "bank loan",
        "account deposit",
        "pizza dinner",
        "menu restaurant",
        "engine repair",
        "wheel garage",
    ] * 4
    labels = np.array([0, 0, 1, 1, 2, 2] * 4)
    enc = SpladeEncoder.from_pretrained(tiny_mlm, _splade_cfg(tiny_mlm))
    first = train_splade(
        enc,
        texts,
        labels,
        ls.descriptions,
        _splade_cfg(tiny_mlm, train_epochs=1, learning_rate=1e-3, train_batch_size=6),
    )["loss"]
    last = train_splade(
        enc,
        texts,
        labels,
        ls.descriptions,
        _splade_cfg(tiny_mlm, train_epochs=15, learning_rate=1e-3, train_batch_size=6),
    )["loss"]
    assert last < first
    S = (enc.encode(texts[:6]) @ enc.encode(ls.descriptions).T).toarray()
    assert (S.argmax(1) == labels[:6]).mean() >= 0.5


def test_real_backend_end_to_end_with_training(tiny_mlm, tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "splade"]
    cfg.retrieval = RetrievalConfig(splade=_splade_cfg(tiny_mlm, train=True))
    artifacts, report, items, _ = _train(cfg)
    assert report.n_items > 0
    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)
    assert os.path.isfile(
        os.path.join(model_dir, "signals", "splade", "encoder", "model.safetensors")
    )
    texts = [it.text for it in items[:8]]
    before = InferencePipeline(artifacts).predict(texts)
    after = InferencePipeline(ArtifactRepository().load(model_dir)).predict(texts)
    assert np.allclose([p.confidence for p in before], [p.confidence for p in after], atol=1e-5)


# ---------------------------------------------------------------- agreement
def _oof_frame(signals):
    cfg = PipelineConfig()
    cfg.signals = signals
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=5)
    pipe = TrainingPipeline(cfg)
    artifacts, _ = pipe.run(items, label_space)
    texts = [it.text for it in items]
    from text_classifier.application.features import FeatureAssembler
    from text_classifier.domain import CandidatePolicy

    asm = FeatureAssembler(label_space, CandidatePolicy(artifacts.config.candidate_top_n))
    q = artifacts.encoder.encode(texts)
    frame = asm.assemble(
        texts,
        q,
        artifacts.dense,
        artifacts.lexical,
        10,
        query_ids=np.arange(len(texts)),
        signal_providers=artifacts.signal_providers,
    )
    return artifacts, frame


@pytest.mark.parametrize(
    "signals, voters",
    [(["dense", "lexical"], 5), (["dense", "splade"], 5), (["dense", "lexical", "splade"], 7)],
)
def test_signal_agreement_counts_every_round_one_signal(signals, voters):
    from text_classifier.domain.services import fusion_feature_names

    artifacts, frame = _oof_frame(signals)
    names = fusion_feature_names(artifacts.feature_providers, (), artifacts.signal_providers)
    assert "n_signal_agreement" in names
    assert frame.attrs["agreement_voters"] == voters
    agree = frame["n_signal_agreement"].to_numpy()
    assert agree.min() >= 0 and agree.max() <= voters - 1


def test_signal_agreement_ignores_the_reranker_and_needs_two_voters():
    from text_classifier.domain.services import fusion_feature_names

    artifacts, frame = _oof_frame(["dense", "splade", "cross-encoder"])
    assert frame.attrs["agreement_voters"] == 5  # ce.* picks do not vote
    artifacts, _ = _oof_frame(["dense", "cross-encoder"])
    names = fusion_feature_names(artifacts.feature_providers, (), artifacts.signal_providers)
    assert "n_signal_agreement" not in names  # one round-one signal: nothing to agree


def test_signal_report_uses_the_recorded_voter_count():
    import pandas as pd

    from text_classifier.application.signal_report import signal_report

    df = pd.DataFrame(
        {"item_id": [0, 0, 1], "is_true": [1, 0, 1], "n_signal_agreement": [6.0, 6.0, 3.0]}
    )
    df.attrs["agreement_voters"] = 7
    rep = signal_report(df)["agreement"]
    assert rep["mean_distinct_top_classes"] == pytest.approx((1 + 4) / 2)
    assert rep["consensus_rate"] == pytest.approx(0.5)
