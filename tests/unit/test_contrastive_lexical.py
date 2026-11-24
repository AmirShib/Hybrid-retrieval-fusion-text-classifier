"""T97: the round-two contrastive lexical signal.

Every value asserted here is worked out by hand from the ticket's formulas:
``w(t) = log(K / df(t)) / log(K)`` over the item's shortlist of size ``K``."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tests._doubles import make_synthetic
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.application.updating import update
from text_classifier.config import ContrastiveLexicalConfig, PipelineConfig, TrainingConfig
from text_classifier.domain import (
    CandidateView,
    ClassDefinition,
    LabeledItem,
    LabelSpace,
    SignalContext,
)
from text_classifier.infrastructure import ArtifactRepository
from text_classifier.infrastructure.contrastive import ContrastiveLexicalSignalProvider

# The tokenizer T95 recommends for Hebrew item names: keeps 1-char tokens,
# decimals, percents and gershayim words (מ"ל) whole.
HEBREW_TOKENS = {"token_pattern": r"""(?u)\d+(?:\.\d+)?%?|[^\W\d_]+(?:['"][^\W\d_]+)*['"]?"""}

COLA = LabelSpace(
    [
        ClassDefinition("A", 'קוקה קולה 500 מ"ל'),
        ClassDefinition("B", "קוקה קולה זירו 1.5 ליטר"),
        ClassDefinition("C", 'פפסי מקס 500 מ"ל'),
        ClassDefinition("D", "מים מינרליים"),  # never shortlisted below
    ]
)


def _w(k: int, df: int) -> float:
    return math.log(k / df) / math.log(k)


def _ctx(texts, space, mask, self_ids=None):
    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    return SignalContext(
        texts=list(texts),
        q_emb=np.zeros((len(texts), 4), dtype=np.float32),
        k=5,
        n_classes=space.size,
        label_space=space,
        self_ids=self_ids,
        candidates=CandidateView(mask=mask, rows=rows, cols=cols, signals={}),
    )


def _columns(provider, ctx):
    """Every column the assembler would gather, as a dict of (b, C) / (b,) arrays."""
    (sm,) = provider.build(ctx)
    out = {"lex_local_overlap": sm.value}
    out.update(sm.extra_columns)
    out.update(sm.extra_scalars)
    return sm, out


def _provider(**cfg):
    return ContrastiveLexicalSignalProvider(ContrastiveLexicalConfig(**cfg), HEBREW_TOKENS)


# ------------------------------------------------------------------ the worked example
def test_worked_example_every_column():
    query = 'קוקה קולה זירו 500 מ"ל'
    _, col = _columns(_provider(), _ctx([query], COLA, [[1, 1, 1, 0]]))

    # K = 3. df: קוקה 2, קולה 2, זירו 1 (B), 500 2, מ"ל 2.
    w2, w1 = _w(3, 2), _w(3, 1)
    W = 4 * w2 + w1
    local = col["lex_local_overlap"][0]
    assert local[0] == pytest.approx(4 * w2)  # A: קוקה קולה 500 מ"ל
    assert local[1] == pytest.approx(2 * w2 + w1)  # B: קוקה קולה זירו
    assert local[2] == pytest.approx(2 * w2)  # C: 500 מ"ל
    assert np.isnan(local[3])  # D was not shortlisted

    unique = col["lex_unique_hits"][0]
    assert unique[:3] == pytest.approx([0.0, w1 / W, 0.0])

    # Strongest single competitor's evidence that c lacks:
    #   A lacks זירו (B has it)               -> w1
    #   B lacks 500, מ"ל (A and C have them)   -> 2*w2
    #   C lacks קוקה, קולה, זירו (B has all three) -> 2*w2 + w1
    counter = col["lex_counter_hits"][0]
    assert counter[:3] == pytest.approx([w1 / W, 2 * w2 / W, (2 * w2 + w1) / W])
    assert col["lex_shortlist_size"][0] == 3


def test_counter_is_the_strongest_single_competitor_not_the_total():
    space = LabelSpace(
        [
            ClassDefinition("a", "base"),
            ClassDefinition("b", "base xx"),
            ClassDefinition("c", "base yy"),
        ]
    )
    _, col = _columns(_provider(), _ctx(["base xx yy"], space, [[1, 1, 1]]))
    w1 = _w(3, 1)
    W = 2 * w1  # "base" is in all three: weight 0
    # a lacks xx (b) and yy (c), but no single competitor has both.
    assert col["lex_counter_hits"][0, 0] == pytest.approx(w1 / W)


def test_word_every_candidate_has_carries_no_weight():
    space = LabelSpace([ClassDefinition("a", "milk"), ClassDefinition("b", "milk")])
    _, col = _columns(_provider(), _ctx(["milk"], space, [[1, 1]]))
    assert col["lex_local_overlap"][0] == pytest.approx([0.0, 0.0])
    # Nothing distinguishes them, so there is no evidence to share out: NaN.
    assert np.isnan(col["lex_unique_hits"][0]).all()
    assert np.isnan(col["lex_counter_hits"][0]).all()


def test_a_word_only_one_candidate_has_weighs_one_at_any_shortlist_size():
    defs = [ClassDefinition("hit", "zero")] + [
        ClassDefinition(f"o{i}", f"other{i}") for i in range(5)
    ]
    space = LabelSpace(defs)
    for k in (2, 3, 6):
        mask = [[1] * k + [0] * (6 - k)]
        _, col = _columns(_provider(), _ctx(["zero"], space, mask))
        assert col["lex_local_overlap"][0, 0] == pytest.approx(1.0)
        assert col["lex_shortlist_size"][0] == k


# ------------------------------------------------------------------ NaN rules
def test_no_overlap_is_a_true_zero_inside_the_shortlist():
    _, col = _columns(_provider(), _ctx(["זירו"], COLA, [[1, 1, 1, 0]]))
    assert col["lex_local_overlap"][0, :3] == pytest.approx([0.0, 1.0, 0.0])


def test_query_matching_no_candidate_has_no_evidence_to_share():
    _, col = _columns(_provider(), _ctx(["שוקולד"], COLA, [[1, 1, 1, 0]]))
    assert col["lex_local_overlap"][0, :3] == pytest.approx([0.0, 0.0, 0.0])
    assert np.isnan(col["lex_unique_hits"][0, :3]).all()
    assert np.isnan(col["lex_counter_hits"][0, :3]).all()


def test_lone_candidate_is_nan_but_still_reports_its_size():
    _, col = _columns(_provider(), _ctx(["קוקה"], COLA, [[1, 0, 0, 0]]))
    for name in ("lex_local_overlap", "lex_unique_hits", "lex_counter_hits"):
        assert np.isnan(col[name]).all()
    assert col["lex_shortlist_size"][0] == 1


def test_queries_are_independent_within_a_chunk():
    p = _provider()
    one = _columns(p, _ctx(['קוקה קולה זירו 500 מ"ל'], COLA, [[1, 1, 1, 0]]))[1]
    both = _columns(
        p,
        _ctx(['קוקה קולה זירו 500 מ"ל', "מים"], COLA, [[1, 1, 1, 0], [0, 1, 0, 1]]),
    )[1]
    np.testing.assert_allclose(both["lex_local_overlap"][0], one["lex_local_overlap"][0])
    np.testing.assert_allclose(both["lex_counter_hits"][0], one["lex_counter_hits"][0])
    assert both["lex_local_overlap"][1, 3] == pytest.approx(1.0)  # מים: only D has it


def test_small_chunks_give_the_same_result():
    texts = ['קוקה קולה זירו 500 מ"ל', "מים", "פפסי 500", "קולה"]
    mask = [[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 0]]
    whole = _columns(_provider(), _ctx(texts, COLA, mask))[1]
    small = ContrastiveLexicalSignalProvider(ContrastiveLexicalConfig(), HEBREW_TOKENS, chunk=1)
    split = _columns(small, _ctx(texts, COLA, mask))[1]
    for name, arr in whole.items():
        np.testing.assert_allclose(split[name], arr, equal_nan=True)


# ------------------------------------------------------------------ tokenization
def test_tokenization_follows_the_bm25_token_kwargs():
    query = 'קוקה קולה 500 מ"ל'
    mask = [[1, 1, 1, 0]]
    hebrew = _columns(_provider(), _ctx([query], COLA, mask))[1]
    # sklearn's default pattern drops 1-char tokens: מ"ל vanishes.
    default = ContrastiveLexicalSignalProvider(ContrastiveLexicalConfig(), {})
    plain = _columns(default, _ctx([query], COLA, mask))[1]
    assert hebrew["lex_local_overlap"][0, 2] == pytest.approx(2 * _w(3, 2))  # 500, מ"ל
    assert plain["lex_local_overlap"][0, 2] == pytest.approx(_w(3, 2))  # 500 only


def test_codebook_views_are_opt_in():
    space = LabelSpace(
        [
            ClassDefinition("a", "soft drink", examples=("קולה",)),
            ClassDefinition("b", "water"),
        ]
    )
    off = _columns(_provider(), _ctx(["קולה"], space, [[1, 1]]))[1]
    on = _columns(_provider(views=["description", "examples"]), _ctx(["קולה"], space, [[1, 1]]))[1]
    assert off["lex_local_overlap"][0, 0] == pytest.approx(0.0)
    assert on["lex_local_overlap"][0, 0] == pytest.approx(1.0)


def test_class_with_no_text_in_the_chosen_views_is_not_scored():
    space = LabelSpace(
        [
            ClassDefinition("a", "x", examples=("קולה",)),
            ClassDefinition("b", "y", examples=("מים",)),
            ClassDefinition("c", "z"),  # no examples
        ]
    )
    _, col = _columns(_provider(views=["examples"]), _ctx(["קולה"], space, [[1, 1, 1]]))
    assert np.isnan(col["lex_local_overlap"][0, 2])
    assert col["lex_shortlist_size"][0] == 2  # only scored candidates count
    assert col["lex_local_overlap"][0, 0] == pytest.approx(1.0)


def test_round_two_contract():
    p = _provider()
    assert p.needs_candidates is True
    assert tuple(p.candidate_features()) == ()
    ctx = SignalContext(
        texts=["x"], q_emb=np.zeros((1, 4)), k=5, n_classes=COLA.size, label_space=COLA
    )
    with pytest.raises(ValueError, match="needs_candidates"):
        p.build(ctx)


def test_columns_match_what_build_emits():
    for cfg in ({}, {"neighbors": 2}):
        p = _provider(**cfg)
        if cfg:
            p.fit(["קוקה"], np.array([0]), COLA)
        sm, col = _columns(p, _ctx(["קוקה"], COLA, [[1, 1, 0, 0]]))
        emitted = set(sm.columns.values()) | {sm.gap_column} | set(col) - {"lex_local_overlap"}
        emitted |= {"lex_local_overlap"}
        assert emitted == set(p.column_names())


# ------------------------------------------------------------------ neighbours
POOL_TEXTS = ["קולה זירו פחית", "קולה בקבוק", "מים מינרליים", "מים בטעם לימון"]
POOL_LABELS = np.array([0, 0, 3, 3])


def test_neighbours_are_off_by_default():
    p = _provider()
    p.fit(POOL_TEXTS, POOL_LABELS, COLA)
    assert "lex_evidence_docs" not in p.column_names()
    _, col = _columns(p, _ctx(["פחית"], COLA, [[1, 1, 0, 0]]))
    assert col["lex_local_overlap"][0, 0] == pytest.approx(0.0)  # "פחית" only in the pool


def test_neighbours_add_the_most_overlapping_examples_of_that_class():
    p = _provider(neighbors=1)
    p.fit(POOL_TEXTS, POOL_LABELS, COLA)
    # "קולה זירו פחית" shares 2 words with the query, "קולה בקבוק" 1: only the first joins A.
    _, col = _columns(p, _ctx(["זירו פחית בקבוק"], COLA, [[1, 0, 0, 1]]))
    assert col["lex_local_overlap"][0, 0] == pytest.approx(2.0)  # זירו, פחית (K=2, df=1)
    assert col["lex_evidence_docs"][0, 0] == 2  # description + 1 neighbour
    assert col["lex_evidence_docs"][0, 3] == 1  # D: no neighbour shares a word


def test_neighbours_exclude_the_item_itself_in_leave_one_out_mode():
    p = _provider(neighbors=2)
    p.fit(POOL_TEXTS, POOL_LABELS, COLA)
    ctx = _ctx(["קולה זירו פחית"], COLA, [[1, 1, 0, 0]], self_ids=np.array([0]))
    _, col = _columns(p, ctx)
    # Without its own row, A's only neighbour is "קולה בקבוק": זירו comes from B alone.
    assert col["lex_unique_hits"][0, 1] > 0
    assert col["lex_evidence_docs"][0, 0] == 2


def test_with_added_examples_extends_the_pool():
    p = _provider(neighbors=2)
    p.fit(POOL_TEXTS, POOL_LABELS, COLA)
    q = _ctx(["אנרגיה"], COLA, [[1, 1, 0, 0]])
    assert _columns(p, q)[1]["lex_local_overlap"][0, 1] == pytest.approx(0.0)
    bigger = p.with_added_examples(["משקה אנרגיה"], np.array([1]))
    assert _columns(bigger, q)[1]["lex_local_overlap"][0, 1] == pytest.approx(1.0)
    assert _columns(p, q)[1]["lex_local_overlap"][0, 1] == pytest.approx(0.0)  # original untouched
    assert p.pool_size == len(POOL_TEXTS) and p._pool.matrix.shape[1] < bigger._pool.matrix.shape[1]


def test_save_load_round_trip(tmp_path):
    p = _provider(neighbors=2)
    p.fit(POOL_TEXTS, POOL_LABELS, COLA)
    p.save(str(tmp_path))
    again = ContrastiveLexicalSignalProvider.load_from(
        str(tmp_path), ContrastiveLexicalConfig(neighbors=2), HEBREW_TOKENS
    )
    ctx = _ctx(["זירו פחית בקבוק", "מים לימון"], COLA, [[1, 1, 0, 1], [1, 0, 0, 1]])
    for name, arr in _columns(p, ctx)[1].items():
        np.testing.assert_allclose(_columns(again, ctx)[1][name], arr, equal_nan=True)
    assert not any(f.endswith((".pkl", ".pickle")) for f in os.listdir(tmp_path))


# ------------------------------------------------------------------ config
def test_config_defaults_round_trip_and_validation():
    assert "contrastive-lexical" not in PipelineConfig().signals
    cfg = PipelineConfig.from_dict(
        {
            "retrieval": {
                "contrastive_lexical": {"views": ["description", "examples"], "neighbors": 2}
            }
        }
    )
    assert cfg.retrieval.contrastive_lexical.neighbors == 2
    assert PipelineConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    cfg.validate()
    for bad in ({"neighbors": -1}, {"neighbors": 6}, {"views": []}, {"views": ["exclusions"]}):
        with pytest.raises(ValueError, match="contrastive_lexical"):
            PipelineConfig.from_dict({"retrieval": {"contrastive_lexical": bad}}).validate()


# ------------------------------------------------------------------ pipeline
def _train(cfg, n_folds=3):
    label_space, items = make_synthetic(n_classes=6, per_class=15, seed=5)
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=n_folds, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    return TrainingPipeline(cfg).run(items, label_space) + (items, label_space)


def test_end_to_end_train_save_load_predict(tmp_path):
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "contrastive-lexical"]
    cfg.retrieval.contrastive_lexical.neighbors = 2
    artifacts, report, items, _ = _train(cfg)
    assert report.n_items > 0
    provider = artifacts.signal_providers[-1]
    assert isinstance(provider, ContrastiveLexicalSignalProvider)

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)
    with open(os.path.join(model_dir, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["components"]["signals"] == cfg.signals
    assert set(provider.column_names()) <= set(meta["feature_names"])

    texts = [it.text for it in items[:12]]
    before = InferencePipeline(artifacts).predict(texts)
    after = InferencePipeline(ArtifactRepository().load(model_dir)).predict(texts)
    assert [p.top_key for p in before] == [p.top_key for p in after]
    assert np.allclose([p.confidence for p in before], [p.confidence for p in after])


def test_pool_is_fitted_per_fold_without_leakage(monkeypatch):
    sizes = []
    orig = ContrastiveLexicalSignalProvider.fit

    def spy(self, texts, labels, label_space):
        sizes.append(len(texts))
        return orig(self, texts, labels, label_space)

    monkeypatch.setattr(ContrastiveLexicalSignalProvider, "fit", spy)
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "contrastive-lexical"]
    cfg.retrieval.contrastive_lexical.neighbors = 2
    _, _, items, _ = _train(cfg)
    assert len(sizes) == 3 + 1
    assert all(s < len(items) for s in sizes[:3]) and sizes[3] == len(items)


def test_update_indexes_new_examples_when_neighbours_are_on():
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "contrastive-lexical"]
    cfg.retrieval.contrastive_lexical.neighbors = 2
    artifacts, _, items, ls = _train(cfg)
    before = artifacts.signal_providers[-1].pool_size
    new = [LabeledItem("brand new example text", ls.key_at(0))]
    updated, _ = update(artifacts, items, new_items=new)
    assert updated.signal_providers[-1].pool_size == before + 1


def test_runs_alongside_the_cross_encoder(tmp_path):
    """Two round-two signals: both score the same shortlist, independently."""
    cfg = PipelineConfig()
    cfg.signals = ["dense", "lexical", "cross-encoder", "contrastive-lexical"]
    cfg.retrieval.cross_encoder.kind = "token-overlap"
    cfg.retrieval.contrastive_lexical.neighbors = 2
    artifacts, _, items, _ = _train(cfg)
    texts = [it.text for it in items[:12]]
    frame = InferencePipeline(artifacts).explain(texts)
    assert frame["ce_pos"].notna().any() and frame["lex_local_overlap"].notna().any()

    model_dir = str(tmp_path / "model")
    ArtifactRepository().save(artifacts, model_dir)
    before = InferencePipeline(artifacts).predict(texts)
    after = InferencePipeline(ArtifactRepository().load(model_dir)).predict(texts)
    assert [p.top_key for p in before] == [p.top_key for p in after]
