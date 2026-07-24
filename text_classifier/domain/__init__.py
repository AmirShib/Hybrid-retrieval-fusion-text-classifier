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
    FEATURE_NAMES,
    AbstentionPolicy,
    CandidatePolicy,
    ThresholdTuner,
    composed_feature_names,
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
    "CandidatePolicy",
    "AbstentionPolicy",
    "ThresholdTuner",
]
