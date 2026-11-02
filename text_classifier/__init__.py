"""Hybrid retrieval-fusion text classifier with abstention.

A domain-driven package:
    domain/          framework-free models, ports, policies
    infrastructure/  adapters (sentence-transformers, BM25, XGBoost, persistence)
    application/     training and inference pipelines
"""

# Import xgboost before anything here can import torch. xgboost and torch each
# bundle an OpenMP runtime; on macOS with Homebrew's libomp, initialising torch's
# first makes xgboost's first fit() segfault (OMP: Error #179). The fusion model
# imports xgboost lazily, while the sentence-transformers encoder pulls in torch
# earlier in a training run -- exactly the crashing order. Loading xgboost first
# is sufficient on the affected host (see T85 / tests/conftest.py, which applies
# the same fix to the test suite). Not reproducible on Linux, where it is a no-op
# beyond the import itself; xgboost is a hard dependency, so this always succeeds.
# It cannot help if torch was already imported before this package.
import xgboost as _xgboost  # noqa: F401  (isort: keep first)

from ._version import __version__
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
    "__version__",
    "PipelineConfig",
    "EncoderConfig",
    "RetrievalConfig",
    "FusionConfig",
    "TrainingConfig",
    "ClassDefinition",
    "LabeledItem",
    "LabelSpace",
    "Prediction",
    "TrainingPipeline",
    "InferencePipeline",
]
