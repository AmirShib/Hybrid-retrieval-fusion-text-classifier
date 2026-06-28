"""Infrastructure layer: concrete adapters implementing the domain ports."""
from .encoder import (
    SentenceTransformerEncoder,
    TfidfEncoder,
    fit_tfidf_encoder,
    train_encoder,
)
from .fusion import IsotonicCalibrator, XGBoostFusionModel
from .persistence import ArtifactRepository, DeployedArtifacts
from .registry import (
    CalibratorSpec,
    EncoderSpec,
    FusionSpec,
    build_calibrator,
    build_encoder,
    build_fusion,
    encoder_is_corpus_dependent,
    fit_encoder,
    register_calibrator,
    register_encoder,
    register_fusion,
)
from .retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    DenseState,
    LexicalRetrieverAdapter,
)

__all__ = [
    "SentenceTransformerEncoder", "TfidfEncoder", "train_encoder", "fit_tfidf_encoder",
    "BM25Index", "DenseRetrieverAdapter", "DenseState", "LexicalRetrieverAdapter",
    "XGBoostFusionModel", "IsotonicCalibrator",
    "ArtifactRepository", "DeployedArtifacts",
    "EncoderSpec", "FusionSpec", "CalibratorSpec",
    "build_encoder", "build_fusion", "build_calibrator",
    "encoder_is_corpus_dependent", "fit_encoder",
    "register_encoder", "register_fusion", "register_calibrator",
]
