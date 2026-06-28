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

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, TypeVar

from ..config import CalibrationConfig, EncoderConfig, FusionConfig
from ..domain import ConfidenceCalibrator, FusionModel, LabeledItem, LabelSpace, TextEncoder
from .encoder import (
    SentenceTransformerEncoder,
    TfidfEncoder,
    fit_tfidf_encoder,
    train_encoder,
)
from .fusion import (
    IsotonicCalibrator,
    LightGBMFusionModel,
    XGBoostFusionModel,
    XGBRankerFusionModel,
)


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


# --------------------------------------------------------------------------- maps
_ENCODERS: Dict[str, EncoderSpec] = {}
_FUSIONS: Dict[str, FusionSpec] = {}
_CALIBRATORS: Dict[str, CalibratorSpec] = {}

_T = TypeVar("_T")


def register_encoder(name: str, spec: EncoderSpec) -> None:
    _ENCODERS[name] = spec


def register_fusion(name: str, spec: FusionSpec) -> None:
    _FUSIONS[name] = spec


def register_calibrator(name: str, spec: CalibratorSpec) -> None:
    _CALIBRATORS[name] = spec


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


# ------------------------------------------------------------------- factories
def build_encoder(config: EncoderConfig) -> TextEncoder:
    return encoder_spec(config.kind).build(config)


def encoder_is_corpus_dependent(config: EncoderConfig) -> bool:
    """Whether this encoder must be fit on a corpus (and therefore per fold)."""
    return encoder_spec(config.kind).corpus_dependent


def fit_encoder(config: EncoderConfig, items: Sequence[LabeledItem],
                label_space: LabelSpace) -> TextEncoder:
    """Fit/train a corpus-dependent (or fine-tunable) encoder on ``items``."""
    spec = encoder_spec(config.kind)
    if spec.fit is None:
        raise ValueError(f"encoder kind {config.kind!r} is not corpus-fittable")
    return spec.fit(items, label_space, config)


def build_fusion(config: FusionConfig) -> FusionModel:
    return fusion_spec(config.kind).build(config)


def build_calibrator(config: CalibrationConfig) -> ConfidenceCalibrator:
    return calibrator_spec(config.kind).build(config)


# ----------------------------------------------------------------- built-ins
register_encoder(
    "sentence-transformers",
    EncoderSpec(
        build=lambda cfg: SentenceTransformerEncoder.load(
            cfg.model_name_or_path, cfg.encode_batch_size, cfg.device
        ),
        dirname="encoder",
        load=lambda path, cfg: SentenceTransformerEncoder.load(
            path, cfg.encode_batch_size, cfg.device
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

register_fusion(
    "xgboost",
    FusionSpec(
        build=lambda cfg: XGBoostFusionModel(cfg.xgb_params, cfg.auto_scale_pos_weight),
        filename="fusion.json",
        load=XGBoostFusionModel.load,
    ),
)

register_fusion(
    "lightgbm",
    FusionSpec(
        build=lambda cfg: LightGBMFusionModel(cfg.params, cfg.auto_scale_pos_weight),
        filename="fusion.txt",  # LightGBM native text format
        load=LightGBMFusionModel.load,
    ),
)

register_fusion(
    "xgbranker",
    FusionSpec(
        build=lambda cfg: XGBRankerFusionModel(cfg.params),
        filename="fusion_ranker",  # a directory: native model + isotonic head
        load=XGBRankerFusionModel.load,
    ),
)

register_calibrator(
    "isotonic",
    CalibratorSpec(
        build=lambda cfg: IsotonicCalibrator(),
        filename="calibrator.pkl",
        load=IsotonicCalibrator.load,
    ),
)
