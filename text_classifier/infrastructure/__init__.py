"""Infrastructure layer: concrete adapters implementing the domain ports."""
from .encoder import SentenceTransformerEncoder, train_encoder
from .fusion import IsotonicCalibrator, XGBoostFusionModel
from .persistence import ArtifactRepository, DeployedArtifacts
from .retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    DenseState,
    LexicalRetrieverAdapter,
)

__all__ = [
    "SentenceTransformerEncoder", "train_encoder",
    "BM25Index", "DenseRetrieverAdapter", "DenseState", "LexicalRetrieverAdapter",
    "XGBoostFusionModel", "IsotonicCalibrator",
    "ArtifactRepository", "DeployedArtifacts",
]
