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
    FusionModel,
    LexicalRetriever,
    TextEncoder,
)
from .services import (
    FEATURE_NAMES,
    AbstentionPolicy,
    CandidatePolicy,
    ThresholdTuner,
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
    "FEATURE_NAMES",
    "CandidatePolicy",
    "AbstentionPolicy",
    "ThresholdTuner",
]
