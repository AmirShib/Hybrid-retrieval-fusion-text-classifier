"""Domain layer: the framework-free core (models, ports, policies)."""

from .models import (
    ClassDefinition,
    CoverageReport,
    LabeledItem,
    LabelSpace,
    Prediction,
)
from .ports import (
    ArrayOps,
    ConfidenceCalibrator,
    DenseRetriever,
    FeatureContext,
    FeatureProvider,
    FusionModel,
    LexicalRetriever,
    TextEncoder,
)
from .services import (
    ENCODER_SELECTION_METRICS,
    FEATURE_DEPS,
    FEATURE_NAMES,
    AbstentionPolicy,
    CandidatePolicy,
    EpochSelectionPolicy,
    ThresholdTuner,
    composed_feature_names,
    encoder_retrieval_metrics,
    feature_closure,
    fusion_feature_names,
)

__all__ = [
    "ClassDefinition",
    "LabeledItem",
    "LabelSpace",
    "Prediction",
    "CoverageReport",
    "ArrayOps",
    "TextEncoder",
    "DenseRetriever",
    "LexicalRetriever",
    "FusionModel",
    "ConfidenceCalibrator",
    "FeatureProvider",
    "FeatureContext",
    "FEATURE_NAMES",
    "FEATURE_DEPS",
    "composed_feature_names",
    "fusion_feature_names",
    "feature_closure",
    "CandidatePolicy",
    "AbstentionPolicy",
    "ThresholdTuner",
    "ENCODER_SELECTION_METRICS",
    "EpochSelectionPolicy",
    "encoder_retrieval_metrics",
]
