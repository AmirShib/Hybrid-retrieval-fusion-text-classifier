"""Component registry + factory DI for the three swappable ports.

The encoder, fusion model, and calibrator are selected by a ``kind`` string in
``PipelineConfig`` rather than hardcoded. This module maps each ``kind`` to a
small spec describing how to *build* the component from config, what *filename*
(or directory) it persists to, and how to *load* it back. The training pipeline
and the artifact repository go through these factories, so adding a new backend
is purely additive:

    1. implement the matching port in ``infrastructure/`` (e.g. a ``FusionModel``);
    2. ``register_fusion("my-backend", FusionSpec(...))``;
    3. set ``config.fusion.kind = "my-backend"``.

No edits to ``TrainingPipeline`` or ``ArtifactRepository`` are required.

The persistence spec deliberately lives next to the factory: a backend with a
different on-disk format (LightGBM ``.txt``, a Platt ``.json``) only has to name
its filename and loader here, and round-trips without the repository knowing
anything about it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

import numpy as np

from ..config import (
    CalibrationConfig,
    EncoderConfig,
    FeatureProviderConfig,
    FeaturesConfig,
    FusionConfig,
    RetrievalConfig,
)
from ..domain import (
    ArrayOps,
    ConfidenceCalibrator,
    DenseRetriever,
    FeatureProvider,
    FusionModel,
    LabeledItem,
    LabelSpace,
    LexicalRetriever,
    SignalProvider,
    TextEncoder,
)
from .array_ops import NumpyArrayOps
from .encoder import (
    HashingEncoder,
    SentenceTransformerEncoder,
    TfidfEncoder,
    fit_tfidf_encoder,
    train_encoder,
)
from .feature_providers import ClassKeywordOverlapProvider
from .fusion import (
    BetaCalibrator,
    IsotonicCalibrator,
    LightGBMFusionModel,
    PerClassCalibrator,
    PlattCalibrator,
    XGBoostFusionModel,
    XGBRankerFusionModel,
)
from .retrieval import DenseRetrieverAdapter, LexicalRetrieverAdapter
from .signals import DenseSignalProvider, LexicalSignalProvider


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class EncoderSpec:
    """How to build/persist a ``TextEncoder``. Encoders persist to a *directory*
    (SentenceTransformer writes several files), so ``load`` also receives the
    ``EncoderConfig`` for batch-size/device wiring.

    ``corpus_dependent`` marks encoders whose parameters depend on the training
    data (e.g. TF-IDF vocabulary). For those the pipeline must fit per fold on
    training rows only — never reuse one encoder across folds — to stay
    leakage-free. ``fit`` builds such an encoder from a corpus; for pretrained
    encoders it performs optional fine-tuning (used only when
    ``use_per_fold_encoder`` is set)."""

    build: Callable[[EncoderConfig], TextEncoder]
    dirname: str
    load: Callable[[str, EncoderConfig], TextEncoder]
    corpus_dependent: bool = False
    fit: Optional[Callable[[Sequence[LabeledItem], LabelSpace, EncoderConfig], TextEncoder]] = None


@dataclass(frozen=True)
class FusionSpec:
    build: Callable[[FusionConfig], FusionModel]
    filename: str
    load: Callable[[str], FusionModel]


@dataclass(frozen=True)
class CalibratorSpec:
    build: Callable[[CalibrationConfig], ConfidenceCalibrator]
    filename: str
    load: Callable[[str], ConfidenceCalibrator]


@dataclass(frozen=True)
class FeatureProviderSpec:
    """How to build/persist a ``FeatureProvider``. Providers persist to a
    *directory* (like encoders) so a backend can write several files; ``load``
    receives the provider's ``FeatureProviderConfig`` for symmetry with the other
    specs, even though the sample provider reconstructs entirely from disk."""

    build: Callable[[FeatureProviderConfig], FeatureProvider]
    load: Callable[[str, FeatureProviderConfig], FeatureProvider]


@dataclass(frozen=True)
class DenseRetrieverSpec:
    """How to build/persist a ``DenseRetriever`` (T34 phase 1). ``build``'s
    signature matches ``DenseRetrieverAdapter.build`` exactly (encoder, texts,
    labels, label_space, cfg, array_ops); ``load`` receives the *model
    directory* plus the ``RetrievalConfig`` (for e.g. the persisted chunk size),
    mirroring ``EncoderSpec``'s directory-based load so a future backend that
    needs several files has somewhere to put them."""

    build: Callable[
        [TextEncoder, Sequence[str], np.ndarray, LabelSpace, RetrievalConfig, Optional[ArrayOps]],
        DenseRetriever,
    ]
    filename: str
    load: Callable[[str, RetrievalConfig], DenseRetriever]


@dataclass(frozen=True)
class LexicalRetrieverSpec:
    """How to build/persist a ``LexicalRetriever``. ``build``'s signature matches
    ``LexicalRetrieverAdapter.build`` (texts, labels, label_space, cfg). ``load``
    receives the *model directory* (not a single file path) since the built-in
    BM25 backend already manages two files (``lexical.npz`` + ``lexical.json``);
    a directory-based load keeps that multi-file layout un-special-cased."""

    build: Callable[[Sequence[str], np.ndarray, LabelSpace, RetrievalConfig], LexicalRetriever]
    filename: str
    load: Callable[[str], LexicalRetriever]


@dataclass(frozen=True)
class SignalProviderSpec:
    """How to build/persist a ``SignalProvider`` (T34 phase 2). ``build``
    receives the ``RetrievalConfig`` plus the fold's already-built ``dense``/
    ``lexical`` retrievers, mirroring ``DenseRetrieverSpec``/
    ``LexicalRetrieverSpec``'s ``build`` signature -- most third-party
    providers ignore both and build their own state; the two built-ins
    (registered below) wrap one of them directly rather than re-implementing
    retrieval (``SignalProvider``'s own contract).

    ``load`` receives the *model directory* + ``RetrievalConfig``, mirroring
    ``DenseRetrieverSpec.load``'s directory-based load. Persisting is the
    provider's own ``save(path)`` (the ``SignalProvider`` ABC's uniform
    contract) — no separate spec-level save callable is needed."""

    build: Callable[
        [RetrievalConfig, Optional[DenseRetriever], Optional[LexicalRetriever], Optional[ArrayOps]],
        SignalProvider,
    ]
    load: Callable[[str, RetrievalConfig], SignalProvider]


@dataclass(frozen=True)
class ArrayOpsSpec:
    """How to build an ``ArrayOps`` backend. Unlike the other specs there is
    nothing to persist: the backend is a pure execution choice (T84), never a
    property of a saved model directory, so there is no ``filename``/``load``."""

    build: Callable[[], ArrayOps]


# --------------------------------------------------------------------------- maps
_ENCODERS: Dict[str, EncoderSpec] = {}
_FUSIONS: Dict[str, FusionSpec] = {}
_CALIBRATORS: Dict[str, CalibratorSpec] = {}
_FEATURE_PROVIDERS: Dict[str, FeatureProviderSpec] = {}
_ARRAY_OPS: Dict[str, ArrayOpsSpec] = {}
_DENSE_RETRIEVERS: Dict[str, DenseRetrieverSpec] = {}
_LEXICAL_RETRIEVERS: Dict[str, LexicalRetrieverSpec] = {}
_SIGNAL_PROVIDERS: Dict[str, SignalProviderSpec] = {}

_T = TypeVar("_T")


def register_encoder(name: str, spec: EncoderSpec) -> None:
    _ENCODERS[name] = spec


def register_fusion(name: str, spec: FusionSpec) -> None:
    _FUSIONS[name] = spec


def register_calibrator(name: str, spec: CalibratorSpec) -> None:
    _CALIBRATORS[name] = spec


def register_feature_provider(name: str, spec: FeatureProviderSpec) -> None:
    _FEATURE_PROVIDERS[name] = spec


def register_array_ops(name: str, spec: ArrayOpsSpec) -> None:
    _ARRAY_OPS[name] = spec


def register_dense_retriever(name: str, spec: DenseRetrieverSpec) -> None:
    _DENSE_RETRIEVERS[name] = spec


def register_lexical_retriever(name: str, spec: LexicalRetrieverSpec) -> None:
    _LEXICAL_RETRIEVERS[name] = spec


def register_signal_provider(name: str, spec: SignalProviderSpec) -> None:
    _SIGNAL_PROVIDERS[name] = spec


def _lookup(registry: Mapping[str, _T], name: str, what: str) -> _T:
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"unknown {what} kind {name!r}; registered {what} kinds: {sorted(registry)}"
        ) from None


# --------------------------------------------------------------- spec accessors
def encoder_spec(kind: str) -> EncoderSpec:
    return _lookup(_ENCODERS, kind, "encoder")


def fusion_spec(kind: str) -> FusionSpec:
    return _lookup(_FUSIONS, kind, "fusion")


def calibrator_spec(kind: str) -> CalibratorSpec:
    return _lookup(_CALIBRATORS, kind, "calibrator")


def feature_provider_spec(kind: str) -> FeatureProviderSpec:
    return _lookup(_FEATURE_PROVIDERS, kind, "feature provider")


def array_ops_spec(kind: str) -> ArrayOpsSpec:
    return _lookup(_ARRAY_OPS, kind, "array ops")


def registered_array_ops_kinds() -> List[str]:
    return sorted(_ARRAY_OPS)


def dense_retriever_spec(kind: str) -> DenseRetrieverSpec:
    return _lookup(_DENSE_RETRIEVERS, kind, "dense retriever")


def lexical_retriever_spec(kind: str) -> LexicalRetrieverSpec:
    return _lookup(_LEXICAL_RETRIEVERS, kind, "lexical retriever")


def signal_provider_spec(kind: str) -> SignalProviderSpec:
    return _lookup(_SIGNAL_PROVIDERS, kind, "signal provider")


# ------------------------------------------------------------------- factories
def build_encoder(config: EncoderConfig) -> TextEncoder:
    return encoder_spec(config.kind).build(config)


def encoder_is_corpus_dependent(config: EncoderConfig) -> bool:
    """Whether this encoder must be fit on a corpus (and therefore per fold)."""
    return encoder_spec(config.kind).corpus_dependent


def fit_encoder(
    config: EncoderConfig, items: Sequence[LabeledItem], label_space: LabelSpace
) -> TextEncoder:
    """Fit/train a corpus-dependent (or fine-tunable) encoder on ``items``."""
    spec = encoder_spec(config.kind)
    if spec.fit is None:
        raise ValueError(f"encoder kind {config.kind!r} is not corpus-fittable")
    return spec.fit(items, label_space, config)


def build_fusion(config: FusionConfig) -> FusionModel:
    return fusion_spec(config.kind).build(config)


def build_calibrator(config: CalibrationConfig) -> ConfidenceCalibrator:
    return calibrator_spec(config.kind).build(config)


def build_feature_providers(config: FeaturesConfig) -> List[FeatureProvider]:
    """Build the (unfitted) custom feature providers named in ``config``, in order.

    Returns ``[]`` when none are configured — the byte-for-byte-identical default.
    The caller fits each provider (per fold for the OOF loop; on all data for the
    deployment index)."""
    return [feature_provider_spec(pc.kind).build(pc) for pc in config.providers]


def build_array_ops(kind: str) -> ArrayOps:
    return array_ops_spec(kind).build()


def build_dense_retriever(
    cfg: RetrievalConfig,
    encoder: TextEncoder,
    texts: Sequence[str],
    labels: np.ndarray,
    label_space: LabelSpace,
    array_ops: Optional[ArrayOps] = None,
) -> DenseRetriever:
    """Build the dense retriever named by ``cfg.dense_kind``. Signature matches
    ``DenseRetrieverAdapter.build`` exactly, so this is a drop-in for any
    ordinary (non-T88-optimized) dense-retriever build site."""
    return dense_retriever_spec(cfg.dense_kind).build(
        encoder, texts, labels, label_space, cfg, array_ops
    )


def build_lexical_retriever(
    cfg: RetrievalConfig,
    texts: Sequence[str],
    labels: np.ndarray,
    label_space: LabelSpace,
) -> LexicalRetriever:
    """Build the lexical retriever named by ``cfg.lexical_kind``. Signature
    matches ``LexicalRetrieverAdapter.build`` exactly."""
    return lexical_retriever_spec(cfg.lexical_kind).build(texts, labels, label_space, cfg)


def build_signal_providers(
    cfg: RetrievalConfig,
    signal_kinds: Sequence[str],
    dense: Optional[DenseRetriever],
    lexical: Optional[LexicalRetriever],
    array_ops: Optional[ArrayOps] = None,
) -> List[SignalProvider]:
    """Build the ``SignalProvider``s named by ``signal_kinds``, in order (T34
    phase 2). The built-in ``"dense"``/``"lexical"`` kinds wrap the given,
    already-built ``dense``/``lexical`` retrievers directly -- a
    ``SignalProvider`` does not re-implement retrieval (see its docstring) --
    so the caller's existing per-fold retriever-build discipline (the
    leakage-free OOF loop) is what a wrapping provider is built against, with
    no separate fit step of its own. Any other kind dispatches through the
    registry, which decides for itself whether/how it needs ``dense``/
    ``lexical``. Raises the same "unknown kind" error as
    ``dense_retriever_spec``/``lexical_retriever_spec`` for an unregistered
    kind."""
    providers: List[SignalProvider] = []
    for kind in signal_kinds:
        if kind == "dense":
            providers.append(DenseSignalProvider(dense, array_ops))
        elif kind == "lexical":
            providers.append(LexicalSignalProvider(lexical, array_ops))
        else:
            providers.append(signal_provider_spec(kind).build(cfg, dense, lexical, array_ops))
    return providers


def load_signal_providers(
    directory: str,
    cfg: RetrievalConfig,
    signal_kinds: Sequence[str],
    dense: DenseRetriever,
    lexical: LexicalRetriever,
    array_ops: Optional[ArrayOps] = None,
) -> List[SignalProvider]:
    """Reload the ``SignalProvider``s named by ``signal_kinds`` from a saved
    model directory. The built-in ``"dense"``/``"lexical"`` kinds wrap the
    already-loaded ``dense``/``lexical`` retrievers (their numeric state lives
    in ``dense.npz``/``lexical.npz``, loaded once by ``ArtifactRepository`` —
    nothing to re-read); any other kind's ``SignalProviderSpec.load`` manages
    its own files under ``directory``. Raises the "unknown signal provider
    kind" error naming the registered kinds when ``meta.json`` names a kind
    this code does not have registered -- the schema-drift contract shared with
    encoder/fusion/calibrator/dense/lexical kind mismatches."""
    providers: List[SignalProvider] = []
    for kind in signal_kinds:
        if kind == "dense":
            providers.append(DenseSignalProvider(dense, array_ops))
        elif kind == "lexical":
            providers.append(LexicalSignalProvider(lexical, array_ops))
        else:
            providers.append(signal_provider_spec(kind).load(directory, cfg))
    return providers


# ----------------------------------------------------------------- built-ins
register_encoder(
    "sentence-transformers",
    EncoderSpec(
        build=lambda cfg: SentenceTransformerEncoder.load(
            cfg.model_name_or_path, cfg.encode_batch_size, cfg.device, config=cfg, **cfg.params
        ),
        dirname="encoder",
        load=lambda path, cfg: SentenceTransformerEncoder.load(
            path, cfg.encode_batch_size, cfg.device, config=cfg, **cfg.params
        ),
        corpus_dependent=False,  # pretrained weights are data-independent
        fit=lambda items, ls, cfg: train_encoder(items, ls, cfg),  # optional fine-tune
    ),
)

register_encoder(
    "tfidf",
    EncoderSpec(
        build=lambda cfg: TfidfEncoder.from_config(cfg),  # unfitted; must be fit before use
        dirname="encoder",
        load=lambda path, cfg: TfidfEncoder.load(path),
        corpus_dependent=True,  # vocabulary/IDF depend on the corpus -> fit per fold
        fit=lambda items, ls, cfg: fit_tfidf_encoder(items, ls, cfg),
    ),
)

register_encoder(
    "hashing",
    EncoderSpec(
        # Dependency-free, deterministic, stateless: builds without config and
        # round-trips through the artifact repository like any other encoder. The
        # offline demo and CI select it via cfg.encoder.kind = "hashing".
        build=lambda cfg: HashingEncoder(),
        dirname="encoder",
        load=lambda path, cfg: HashingEncoder.load(path, cfg.encode_batch_size, cfg.device),
        corpus_dependent=False,  # no learned/data-dependent state
    ),
)


def _with_objective(params: Dict[str, Any], objective: Optional[str]) -> Dict[str, Any]:
    """Merge ``FusionConfig.objective`` into a backend's params dict as its
    "objective" key. An "objective" already present in ``params`` (a
    power-user override typed directly into xgb_params/params) wins."""
    if objective is None:
        return params
    return {"objective": objective, **params}


register_fusion(
    "xgboost",
    FusionSpec(
        build=lambda cfg: XGBoostFusionModel(
            _with_objective(cfg.xgb_params, cfg.objective), cfg.auto_scale_pos_weight
        ),
        filename="fusion.json",
        load=XGBoostFusionModel.load,
    ),
)

register_fusion(
    "lightgbm",
    FusionSpec(
        build=lambda cfg: LightGBMFusionModel(
            _with_objective(cfg.params, cfg.objective), cfg.auto_scale_pos_weight
        ),
        filename="fusion.txt",  # LightGBM native text format
        load=LightGBMFusionModel.load,
    ),
)

register_fusion(
    "xgbranker",
    FusionSpec(
        build=lambda cfg: XGBRankerFusionModel(_with_objective(cfg.params, cfg.objective)),
        filename="fusion_ranker",  # a directory: native model + isotonic head
        load=XGBRankerFusionModel.load,
    ),
)

register_calibrator(
    "isotonic",
    CalibratorSpec(
        build=lambda cfg: IsotonicCalibrator(),
        filename="calibrator.npz",
        load=IsotonicCalibrator.load,
    ),
)

register_calibrator(
    "platt",
    CalibratorSpec(
        build=lambda cfg: PlattCalibrator(),
        filename="calibrator.json",
        load=PlattCalibrator.load,
    ),
)

register_calibrator(
    "beta",
    CalibratorSpec(
        build=lambda cfg: BetaCalibrator(),
        filename="calibrator.json",
        load=BetaCalibrator.load,
    ),
)

register_calibrator(
    "per-class",
    CalibratorSpec(
        # `params` (see CalibrationConfig): "inner" (isotonic|platt|beta,
        # default "beta") selects the per-class inner calibrator kind;
        # "min_support" (default 50) is the minimum row count a class needs
        # before it gets its own curve rather than falling back to global.
        build=lambda cfg: PerClassCalibrator(
            inner=cfg.params.get("inner", "beta"),
            min_support=cfg.params.get("min_support", 50),
        ),
        filename="calibrator_per_class",  # a directory: manifest + one inner calibrator per class
        load=PerClassCalibrator.load,
    ),
)

register_array_ops("numpy", ArrayOpsSpec(build=lambda: NumpyArrayOps()))


def _build_torch_array_ops() -> ArrayOps:
    """Deferred import (T85): this only runs when ``build_array_ops("torch")``
    is actually called -- registration itself (below) is metadata only, so
    listing "torch" as a registered kind never imports torch (see
    ``array_ops.py::resolve_array_backend``, which checks ``torch_installed()``
    -- a non-importing probe -- rather than registry membership)."""
    from .array_ops_torch import TorchArrayOps

    return TorchArrayOps()


register_array_ops("torch", ArrayOpsSpec(build=_build_torch_array_ops))


def _load_dense_exact(directory: str, cfg: RetrievalConfig) -> DenseRetriever:
    arrays: Dict[str, Any] = dict(np.load(os.path.join(directory, "dense.npz")))
    return DenseRetrieverAdapter.from_state(arrays, chunk=cfg.dense_chunk)


def _load_lexical_bm25(directory: str) -> LexicalRetriever:
    arrays = dict(np.load(os.path.join(directory, "lexical.npz")))
    with open(os.path.join(directory, "lexical.json")) as fh:
        meta = json.load(fh)
    return LexicalRetrieverAdapter.from_state(arrays, meta)


register_dense_retriever(
    "exact",
    DenseRetrieverSpec(
        build=DenseRetrieverAdapter.build,
        filename="dense.npz",
        load=_load_dense_exact,
    ),
)

register_lexical_retriever(
    "bm25",
    LexicalRetrieverSpec(
        build=LexicalRetrieverAdapter.build,
        filename="lexical.npz",
        load=_load_lexical_bm25,
    ),
)

register_signal_provider(
    "dense",
    SignalProviderSpec(
        build=lambda cfg, dense, lexical, ops: DenseSignalProvider(dense, ops),
        load=lambda directory, cfg: DenseSignalProvider(
            dense_retriever_spec(cfg.dense_kind).load(directory, cfg)
        ),
    ),
)

register_signal_provider(
    "lexical",
    SignalProviderSpec(
        build=lambda cfg, dense, lexical, ops: LexicalSignalProvider(lexical, ops),
        load=lambda directory, cfg: LexicalSignalProvider(
            lexical_retriever_spec(cfg.lexical_kind).load(directory)
        ),
    ),
)

register_feature_provider(
    "class-keyword",
    FeatureProviderSpec(
        # The sample provider. `params` pass straight to the provider (and on
        # to sklearn's CountVectorizer): e.g. {"ngram_range": [1, 2]} or a custom
        # {"column": "..."}. Fit per fold by the pipeline (its lexicon is
        # train-set-derived), so leakage-free like any corpus-dependent state.
        build=lambda pc: ClassKeywordOverlapProvider(**pc.params),
        load=lambda path, pc: ClassKeywordOverlapProvider.load(path),
    ),
)
