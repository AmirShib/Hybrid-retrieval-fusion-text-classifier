"""Application layer: use-case orchestration (training and inference pipelines)."""

from .evaluation import (
    build_manifest,
    evaluate_decisions,
    render_model_card,
    write_evaluation_artifacts,
)
from .features import FeatureAssembler
from .inference import InferencePipeline
from .scoring import add_confidence, top_per_item
from .training import TrainingPipeline

__all__ = [
    "FeatureAssembler",
    "TrainingPipeline",
    "InferencePipeline",
    "add_confidence",
    "top_per_item",
    "evaluate_decisions",
    "build_manifest",
    "render_model_card",
    "write_evaluation_artifacts",
]
