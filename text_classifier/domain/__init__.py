"""Domain layer: the framework-free core (models, ports, policies)."""

from .models import (
    ClassDefinition,
    CoverageReport,
    LabeledItem,
    LabelSpace,
    Prediction,
)
from .ports import (
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
    FEATURE_NAMES,
    AbstentionPolicy,
    CandidatePolicy,
    EpochSelectionPolicy,
    ThresholdTuner,
    composed_feature_names,
    encoder_retrieval_metrics,
    fusion_feature_names,
)

__all__ = [
    "ClassDefinition",
    "LabeledItem",
    "LabelSpace",
    "Prediction",
    "CoverageReport",
    "TextEncoder",
    "DenseRetriever",
    "LexicalRetriever",
    "FusionModel",
    "ConfidenceCalibrator",
    "FeatureProvider",
    "FeatureContext",
    "FEATURE_NAMES",
    "composed_feature_names",
    "fusion_feature_names",
    "CandidatePolicy",
    "AbstentionPolicy",
    "ThresholdTuner",
    "ENCODER_SELECTION_METRICS",
    "EpochSelectionPolicy",
    "encoder_retrieval_metrics",
]
