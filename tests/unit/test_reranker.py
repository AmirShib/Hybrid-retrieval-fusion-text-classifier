"""T33 — the cross-encoder rerank signal and the assembler's second stage.

Everything here runs offline: the ``token-overlap`` reranker needs no model and
no network, exactly like the ``hashing`` encoder the rest of the suite uses. The
tests are grouped by what they protect:

1. the second stage itself (ordering, the no-candidates rule, demand gating),
2. the signal's semantics (NaN vs 0, top_k, the gap column),
3. end-to-end schema/persistence parity,
4. the default path being byte-for-byte unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from text_classifier.application.features import FeatureAssembler
from text_classifier.application.inference import InferencePipeline
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    CrossEncoderConfig,
    CrossEncoderDocument,
    EvidenceSpec,
    PipelineConfig,
)
from text_classifier.domain import (
    CandidatePolicy,
    CandidateView,
    ClassDefinition,
    LabeledItem,
    LabelSpace,
    SignalContext,
    SignalMatrix,
    SignalProvider,
)
from text_classifier.infrastructure import (
    ArtifactRepository,
    CrossEncoderSignalProvider,
    TokenOverlapReranker,
)
from text_classifier.infrastructure.reranker import _render


# --------------------------------------------------------------------------- #
# Fixtures: a taxonomy where the *exclusions* actually matter. `food_mfg`
# excludes exactly what `retail_food` is (and vice versa), which is the real
# confusion these classes have in every industrial taxonomy — and the only
# thing in the system that can represent it is the negative document.
# --------------------------------------------------------------------------- #
CLASSES = [
    ClassDefinition(
        "retail_food",
        "retail sale of food",
        title="Retail sale of food",
        definition="Resale of food products to the general public",
        examples=("supermarket", "grocery store", "mini-market"),
        exclusions=("manufacture of food products",),
        sibling_distinctions=("wholesale of food",),
    ),
    ClassDefinition(
        "food_mfg",
        "manufacture of food products",
        title="Manufacture of food",
        definition="Processing of raw materials into food products",
        examples=("bakery plant", "dairy processing"),
        exclusions=("retail sale of food",),
    ),
    ClassDefinition(
        "repair",
        "repair of motor vehicles",
        title="Vehicle repair",
        definition="Maintenance and repair of motor vehicles",
        examples=("garage", "car workshop"),
        # Deliberately no exclusions/siblings: the "absent view" case.
    ),
]

VOCAB = {
    "retail_food": "supermarket grocery store food sale retail shop",
    "food_mfg": "bakery plant dairy processing manufacture food factory",
    "repair": "garage car workshop repair vehicle motor service",
}


def _items(n_per_class: int = 24) -> list:
    rng = np.random.default_rng(0)
    return [
        LabeledItem(" ".join(rng.choice(v.split(), size=5)), k)
        for k, v in VOCAB.items()
        for _ in range(n_per_class)
    ]


def _config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training.n_folds = 3
    cfg.signals = ["dense", "lexical", "cross-encoder"]
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _train(cfg: PipelineConfig, tmp_path):
    space = LabelSpace(CLASSES)
    out = str(tmp_path / "model")
    artifacts, report = TrainingPipeline(cfg).run(_items(), space, output_dir=out)
    return artifacts, report, out


class _CountingReranker(TokenOverlapReranker):
    """Token-overlap scores, plus a tally of how many pairs were scored.

    Call *count* is the assertion for every "this must not run" claim in T33 —
    the whole point of demand gating is that the expensive stage is not entered,
    which cannot be observed from the output frame alone."""

    def __init__(self):
        self.calls = 0
        self.pairs = []

    def score(self, pairs):
        self.calls += 1
        self.pairs.extend(pairs)
        return super().score(pairs)


# =========================================================================== #
# 1. The second stage
# =========================================================================== #
class _RoundOneSpy(SignalProvider):
    """An ordinary signal that records whether it saw a candidate view."""

    name = "spy"

    def __init__(self):
        self.saw_candidates = []

    def candidate_features(self):
        return ()

    def column_names(self):
        return ["spy_col"]

    def build(self, ctx):
        self.saw_candidates.append(ctx.candidates)
        M = np.full((len(ctx.texts), ctx.n_classes), 0.5, dtype=np.float64)
        return [
            SignalMatrix(
                node="spy.x", value=M, derive=frozenset({"raw"}), columns={"raw": "spy_col"}
            )
        ]

    def save(self, path):
        return None

    @classmethod
    def load(cls, path):
        return cls()


class _RoundTwoSpy(_RoundOneSpy):
    name = "spy2"
    needs_candidates = True

    def column_names(self):
        return ["spy2_col"]

    def build(self, ctx):
        self.saw_candidates.append(ctx.candidates)
        M = np.full((len(ctx.texts), ctx.n_classes), 0.5, dtype=np.float64)
        return [
            SignalMatrix(
                node="spy2.x", value=M, derive=frozenset({"raw"}), columns={"raw": "spy2_col"}
            )
        ]


class _BadRoundTwo(_RoundTwoSpy):
    name = "bad"

    def candidate_features(self):
        return ("spy2.x",)


def _assemble_with(providers, requested=None, texts=None):
    """Run the assembler over the built-in signals plus ``providers``."""
    space = LabelSpace(CLASSES)
    cfg = _config()
    pipe = TrainingPipeline(cfg)
    items = _items()
    artifacts, _ = pipe.run(items, space)
    texts = texts or [it.text for it in items[:4]]
    a = artifacts
    assembler = FeatureAssembler(space, CandidatePolicy(cfg.candidate_top_n))
    return assembler.assemble(
        texts,
        a.encoder.encode_queries(texts),
        a.dense,
        a.lexical,
        cfg.retrieval.k_neighbors,
        query_ids=list(range(len(texts))),
        requested=requested,
        signal_providers=list(a.signal_providers) + list(providers),
    )


def test_round_one_provider_never_sees_the_candidate_view():
    spy = _RoundOneSpy()
    _assemble_with([spy])
    assert spy.saw_candidates and all(c is None for c in spy.saw_candidates)


def test_round_two_provider_receives_the_shortlist():
    spy = _RoundTwoSpy()
    _assemble_with([spy])
    assert spy.saw_candidates
    for view in spy.saw_candidates:
        assert view is not None
        assert view.mask.dtype == bool
        assert view.rows.shape == view.cols.shape
        assert view.rows.shape[0] == int(view.mask.sum())
        # Round one's matrices are handed over so a reranker can pick what is
        # worth its cost without recomputing anything.
        assert "dense.desc" in view.signals


def test_second_stage_provider_declaring_candidate_features_is_rejected():
    with pytest.raises(ValueError, match="needs_candidates=True but declares candidate_features"):
        _assemble_with([_BadRoundTwo()])


def test_second_stage_is_skipped_when_all_its_columns_are_pruned():
    """T87 demand gating at the signal layer: with none of its columns
    requested, a round-two provider is not called at all. Safe *only* because
    it contributes no candidates, which is why the rule above is enforced."""
    spy = _RoundTwoSpy()
    df = _assemble_with([spy], requested=["d_desc_sim", "b_desc_sim"])
    assert spy.saw_candidates == []
    assert "spy2_col" not in df.columns


def test_second_stage_runs_when_one_of_its_columns_is_requested():
    spy = _RoundTwoSpy()
    df = _assemble_with([spy], requested=["d_desc_sim", "spy2_col"])
    assert spy.saw_candidates
    assert "spy2_col" in df.columns


# =========================================================================== #
# 2. Signal semantics
# =========================================================================== #
def _featurize(cfg, tmp_path, text):
    _, _, out = _train(cfg, tmp_path)
    inf = InferencePipeline(ArtifactRepository().load(out))
    _, feats = inf._featurize([text], inf._assembled_names)
    return feats.sort_values("candidate").reset_index(drop=True)


def test_negative_evidence_fires_on_the_class_that_excludes_the_query(tmp_path):
    """The load-bearing example. For a retail query, `food_mfg`'s exclusion text
    ("retail sale of food") matches better than its own definition — so its
    negative score outruns its positive one and the gap goes sharply negative.
    Nothing else in the system can represent this."""
    feats = _featurize(_config(), tmp_path, "retail sale of food supermarket")
    retail, mfg = feats.loc[0], feats.loc[1]

    assert mfg["ce_neg"] > mfg["ce_pos"]  # excluded-by matches better than is-a
    assert mfg["ce_pos_neg_gap"] < 0
    assert retail["ce_pos"] > retail["ce_neg"]
    assert retail["ce_pos_neg_gap"] > 0
    assert retail["ce_pos_neg_gap"] > mfg["ce_pos_neg_gap"]


def test_a_class_with_no_exclusions_is_nan_not_zero(tmp_path):
    """CLAUDE.md's NaN invariant at the text layer: an absent view is *absent*,
    not a document that scored badly. A 0.0 here would read to the fusion model
    as "we checked and it doesn't resemble the exclusions", which is a claim we
    have no evidence for."""
    feats = _featurize(_config(), tmp_path, "retail sale of food supermarket")
    repair = feats.loc[2]
    assert np.isnan(repair["ce_neg"])
    assert repair["ce_neg_missing"] == 1.0
    assert np.isnan(repair["ce_pos_neg_gap"])  # not ce_pos - 0
    assert not np.isnan(repair["ce_pos"])  # it does have positive text
    assert repair["ce_pos_missing"] == 0.0


def test_top_k_below_the_shortlist_leaves_the_rest_missing(tmp_path):
    """`ce_missing` distinguishes "shortlisted but not reranked" from "never
    retrieved" — the latter never reaches the frame at all."""
    cfg = _config()
    cfg.retrieval.cross_encoder = CrossEncoderConfig(
        documents=[CrossEncoderDocument(name="pos", top_k=1)], gaps=[]
    )
    feats = _featurize(cfg, tmp_path, "retail sale of food supermarket")
    assert len(feats) == 3  # all three classes shortlisted
    assert feats["ce_pos"].notna().sum() == 1  # only one reranked
    assert feats["ce_pos_missing"].sum() == 2


def test_reranking_never_changes_the_candidate_set(tmp_path):
    """A second-stage signal reorders the shortlist and can never extend it, so
    candidate recall — the ceiling on system accuracy — is untouched."""
    space = LabelSpace(CLASSES)
    items = _items()
    with_ce = TrainingPipeline(_config()).run(items, space)[1]
    without = TrainingPipeline(_config(signals=["dense", "lexical"])).run(items, space)[1]
    assert with_ce.candidate_recall == without.candidate_recall


def test_reranker_is_not_invoked_when_its_columns_are_dropped(tmp_path):
    """The efficiency claim, asserted as a call count rather than inferred from
    the output: dropping the columns must cost zero cross-encoder work."""
    cfg = _config()
    ce_cols = CrossEncoderSignalProvider(TokenOverlapReranker()).column_names()
    cfg.fusion.drop_features = ce_cols
    counter = _CountingReranker()

    space = LabelSpace(CLASSES)
    pipe = TrainingPipeline(cfg)
    provider = CrossEncoderSignalProvider(counter, cfg.retrieval.cross_encoder)
    # Substitute the counting provider for the registered one on every fold.
    import text_classifier.application.training as training_mod

    real_build = training_mod.build_signal_providers

    def _patched(retrieval_cfg, kinds, dense, lexical, ops=None):
        built = real_build(
            retrieval_cfg, [k for k in kinds if k != "cross-encoder"], dense, lexical, ops
        )
        return built + [provider]

    training_mod.build_signal_providers = _patched
    try:
        pipe.run(_items(), space)
    finally:
        training_mod.build_signal_providers = real_build
    assert counter.calls == 0


def test_pairs_are_built_from_the_configured_views(tmp_path):
    """The document handed to the model is asserted directly, not just its
    score — a silently mis-rendered document would still produce plausible
    numbers."""
    definition = CLASSES[0]
    doc = CrossEncoderDocument(
        name="pos",
        evidence=[
            EvidenceSpec(view="core", select="first"),
            EvidenceSpec(view="examples", select="best", label="For example: "),
        ],
    )
    rendered = _render(definition, doc, frozenset({"mini-market", "shop"}))
    assert "Retail sale of food" in rendered
    assert "For example: mini-market" in rendered  # query-adaptive selection
    assert "supermarket" not in rendered  # the other examples are not pasted in


def test_evidence_selection_picks_the_query_closest_entry():
    doc = CrossEncoderDocument(name="d", evidence=[EvidenceSpec(view="examples")])
    assert "grocery store" in _render(CLASSES[0], doc, frozenset({"grocery", "store"}))
    assert "mini-market" in _render(CLASSES[0], doc, frozenset({"mini-market"}))


def test_absent_views_render_to_empty_not_to_a_label():
    """A class with no exclusions must render nothing — not a bare "Excludes: "
    prefix, which would be a document, and would be scored."""
    doc = CrossEncoderDocument(
        name="neg", evidence=[EvidenceSpec(view="exclusions", label="Excludes: ")]
    )
    assert _render(CLASSES[2], doc, frozenset({"garage"})) == ""


def test_per_slot_truncation_keeps_every_slot_represented():
    """Budgeting per slot rather than per document is what stops a long
    definition from truncating away the evidence that follows it."""
    doc = CrossEncoderDocument(
        name="pos",
        evidence=[
            EvidenceSpec(view="core", select="first", max_chars=10),
            EvidenceSpec(view="examples", select="first", label="ex: ", max_chars=6),
        ],
    )
    first, second = _render(CLASSES[0], doc, frozenset()).split("\n")
    assert first == "Retail sal"  # core view truncated to its own budget
    assert second == "ex: superm"  # and the slot after it survives intact


def test_instruction_is_prepended_when_configured():
    """The LLM-judge affordance: the same code path, told what the parts mean."""
    doc = CrossEncoderDocument(
        name="pos",
        instruction="Does the item belong to this class?",
        evidence=[EvidenceSpec(view="description")],
    )
    assert _render(CLASSES[0], doc, frozenset()).startswith("Does the item belong")


def test_reranker_returning_the_wrong_number_of_scores_is_rejected():
    class _Broken(TokenOverlapReranker):
        def score(self, pairs):
            return np.zeros(len(pairs) + 1, dtype=np.float32)

    provider = CrossEncoderSignalProvider(_Broken())
    space = LabelSpace(CLASSES)
    mask = np.ones((1, 3), dtype=bool)
    rows, cols = np.nonzero(mask)
    ctx = SignalContext(
        texts=["supermarket food"],
        q_emb=np.zeros((1, 4), dtype=np.float32),
        k=5,
        n_classes=3,
        label_space=space,
        candidates=CandidateView(mask=mask, rows=rows, cols=cols, signals={}),
    )
    with pytest.raises(ValueError, match="expected exactly one score per pair"):
        provider.build(ctx)


def test_building_without_the_candidate_view_is_rejected():
    provider = CrossEncoderSignalProvider(TokenOverlapReranker())
    ctx = SignalContext(
        texts=["x"],
        q_emb=np.zeros((1, 4), dtype=np.float32),
        k=5,
        n_classes=3,
        label_space=LabelSpace(CLASSES),
    )
    with pytest.raises(ValueError, match="must run in the assembler's second round"):
        provider.build(ctx)


# =========================================================================== #
# 3. Schema, persistence, end to end
# =========================================================================== #
def test_end_to_end_train_save_load_predict(tmp_path):
    cfg = _config()
    _, report, out = _train(cfg, tmp_path)
    assert report.candidate_recall > 0

    import json
    import os

    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["components"]["signals"] == ["dense", "lexical", "cross-encoder"]
    for name in ("ce_pos", "ce_neg", "ce_pos_neg_gap", "rank_ce_pos", "is_ce_neg_top1"):
        assert name in meta["feature_names"]
    assert os.path.isdir(os.path.join(out, "signals", "cross-encoder"))

    inf = InferencePipeline(ArtifactRepository().load(out))
    preds = inf.predict(["supermarket grocery food store", "bakery dairy processing plant"])
    assert [p.top_key for p in preds] == ["retail_food", "food_mfg"]


def test_loaded_schema_matches_the_trained_schema_exactly(tmp_path):
    """Train/infer parity: the column order the model was fitted on has to be
    the order inference rebuilds, or every column silently means something else."""
    from text_classifier.domain import fusion_feature_names

    cfg = _config()
    artifacts, _, out = _train(cfg, tmp_path)
    reloaded = ArtifactRepository().load(out)
    schema = lambda a: fusion_feature_names(  # noqa: E731
        a.feature_providers, a.config.fusion.drop_features, a.signal_providers
    )
    assert schema(artifacts) == schema(reloaded)
    assert "ce_pos" in schema(reloaded)


def test_leave_one_out_mode_works():
    """n_folds=1 (leave-one-out). The cross-encoder has no self-match surface —
    it only ever scores class-side taxonomy text, never the example pool — but
    the path still has to run."""
    cfg = _config()
    cfg.training.n_folds = 1
    external = [LabeledItem(f"{key} sample {i} distinct", key) for key in VOCAB for i in range(4)]
    _, report = TrainingPipeline(cfg).run(
        _items(), LabelSpace(CLASSES), val_items=external, test_items=external[:6]
    )
    assert report.candidate_recall > 0


def test_provider_columns_match_what_build_produces(tmp_path):
    """The `column_names()` contract every schema mechanism relies on."""
    cfg = _config()
    _, _, out = _train(cfg, tmp_path)
    inf = InferencePipeline(ArtifactRepository().load(out))
    _, feats = inf._featurize(["supermarket food"], inf._assembled_names)
    provider = [p for p in inf.artifacts.signal_providers if p.name == "cross-encoder"][0]
    for name in provider.column_names():
        assert name in feats.columns


def test_reranker_score_shape_and_ordering():
    reranker = TokenOverlapReranker()
    scores = reranker.score([("a b c", "a b c"), ("a b c", "x y z")])
    assert scores.shape == (2,)
    assert scores.dtype == np.float32
    assert scores[0] > scores[1]  # more relevant pair scores higher


def test_reranker_round_trips(tmp_path):
    path = str(tmp_path / "rr")
    TokenOverlapReranker().save(path)
    pairs = [("food shop", "retail sale of food")]
    assert TokenOverlapReranker.load(path).score(pairs) == pytest.approx(
        TokenOverlapReranker().score(pairs)
    )


# =========================================================================== #
# 4. The default path is untouched
# =========================================================================== #
def test_default_config_has_no_cross_encoder_columns(tmp_path):
    import json
    import os

    cfg = _config(signals=["dense", "lexical"])
    _train(cfg, tmp_path)
    meta = json.load(open(os.path.join(str(tmp_path / "model"), "meta.json")))
    assert not [n for n in meta["feature_names"] if "ce_" in n]
    # No signal provider wrote anything: the on-disk layout is unchanged too.
    assert not os.path.exists(os.path.join(str(tmp_path / "model"), "signals"))


def test_enabling_the_signal_leaves_every_core_column_identical(tmp_path):
    """The narrow claim T33 makes about not disturbing anything: the core
    columns are computed from round-one signals only, so adding a round-two
    signal cannot move them."""
    space = LabelSpace(CLASSES)
    items = _items()
    texts = [it.text for it in items[:6]]

    frames = {}
    for signals in (["dense", "lexical"], ["dense", "lexical", "cross-encoder"]):
        cfg = _config(signals=signals)
        artifacts, _ = TrainingPipeline(cfg).run(items, space)
        inf = InferencePipeline(artifacts)
        _, frames[tuple(signals)] = inf._featurize(texts, inf._assembled_names)

    base = frames[("dense", "lexical")]
    withce = frames[("dense", "lexical", "cross-encoder")]
    assert len(base) == len(withce)
    for col in base.columns:
        assert np.array_equal(
            base[col].to_numpy(), withce[col].to_numpy(), equal_nan=col != "item_id"
        ), f"core column {col} changed when the cross-encoder was enabled"


def test_array_backend_parity():
    """The provider computes through `ArrayOps`, so it produces identical
    matrices under either backend.

    The feature-assembly layer is pinned to numpy today by deliberate decision
    (`TrainingPipeline._assembler_ops`, T85 — T86 is what makes it
    backend-polymorphic), so this does not exercise a GPU path that exists yet.
    It exists so the provider *moves with* T86 rather than becoming another site
    that has to be rewritten: the two numeric steps (top-n selection, scattering
    scores into the matrix) go through the port, and the only host transfers are
    the two explicit `to_host` calls around the text rendering, which cannot run
    on a device at all."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from text_classifier.infrastructure.array_ops import NumpyArrayOps
    from text_classifier.infrastructure.array_ops_torch import TorchArrayOps

    space = LabelSpace(CLASSES)
    texts = ["supermarket grocery food store", "bakery plant manufacture food", "garage repair"]
    mask = np.ones((len(texts), len(CLASSES)), dtype=bool)
    rows, cols = np.nonzero(mask)
    cand = CandidateView(
        mask=mask,
        rows=rows,
        cols=cols,
        signals={"dense.desc": np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.1], [0.1, 0.2, 0.7]])},
    )

    def _matrices(ops):
        ctx = SignalContext(
            texts=texts,
            q_emb=np.zeros((len(texts), 4), dtype=np.float32),
            k=5,
            n_classes=len(CLASSES),
            label_space=space,
            candidates=cand,
        )
        provider = CrossEncoderSignalProvider(TokenOverlapReranker(), ops=ops)
        return {m.node: m.value for m in provider.build(ctx)}

    on_numpy = _matrices(NumpyArrayOps())
    on_torch = _matrices(TorchArrayOps(device="cpu"))
    assert set(on_numpy) == set(on_torch)
    for node, value in on_numpy.items():
        # Handed to the assembler on the host either way, matching `_scatter_knn`.
        assert isinstance(on_torch[node], np.ndarray)
        assert np.array_equal(value, on_torch[node], equal_nan=True), node


def test_config_round_trip_with_and_without_the_block():
    default = PipelineConfig()
    assert PipelineConfig.from_dict(default.to_dict()) == default

    cfg = _config()
    cfg.retrieval.cross_encoder = CrossEncoderConfig(
        documents=[CrossEncoderDocument(name="only", evidence=[EvidenceSpec(view="core")])],
        gaps=[],
    )
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg


def test_legacy_config_without_the_block_loads():
    """A pre-T33 meta.json has no `cross_encoder` key at all."""
    data = PipelineConfig().to_dict()
    del data["retrieval"]["cross_encoder"]
    cfg = PipelineConfig.from_dict(data)
    assert cfg.retrieval.cross_encoder == CrossEncoderConfig()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda ce: setattr(ce, "batch_size", 0), "batch_size"),
        (lambda ce: setattr(ce, "documents", []), "documents"),
        (lambda ce: ce.documents.append(CrossEncoderDocument(name="pos")), "duplicate"),
        (lambda ce: setattr(ce.documents[0], "top_k", 0), "top_k"),
        (lambda ce: setattr(ce.documents[0].evidence[0], "view", "nope"), "view"),
        (lambda ce: setattr(ce.documents[0].evidence[0], "select", "nope"), "select"),
        (lambda ce: setattr(ce, "gaps", [["pos", "ghost"]]), "gaps"),
        (lambda ce: setattr(ce, "gaps", [["pos"]]), "gaps"),
    ],
)
def test_invalid_cross_encoder_config_is_rejected(mutate, match):
    cfg = PipelineConfig()
    mutate(cfg.retrieval.cross_encoder)
    with pytest.raises(ValueError, match=match):
        cfg.validate()
