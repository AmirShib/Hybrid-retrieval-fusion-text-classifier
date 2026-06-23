"""Hybrid retrieval-fusion text classifier with abstention.

A domain-driven package:
    domain/          framework-free models, ports, policies
    infrastructure/  adapters (sentence-transformers, BM25, XGBoost, persistence)
    application/     training and inference pipelines
"""
from .config import (
    EncoderConfig,
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from .domain import ClassDefinition, LabeledItem, LabelSpace, Prediction
from .application import InferencePipeline, TrainingPipeline

__all__ = [
    "PipelineConfig", "EncoderConfig", "RetrievalConfig", "FusionConfig", "TrainingConfig",
    "ClassDefinition", "LabeledItem", "LabelSpace", "Prediction",
    "TrainingPipeline", "InferencePipeline",
]
