"""Infrastructure layer: concrete adapters implementing the domain ports."""

from .encoder import (
    HashingEncoder,
    SentenceTransformerEncoder,
    TfidfEncoder,
    fit_tfidf_encoder,
    train_encoder,
)
from .feature_providers import ClassKeywordOverlapProvider
from .fusion import (
    BetaCalibrator,
    IsotonicCalibrator,
    LightGBMFusionModel,
    PlattCalibrator,
    XGBoostFusionModel,
    XGBRankerFusionModel,
)
from .persistence import ArtifactRepository, DeployedArtifacts
from .registry import (
    CalibratorSpec,
    EncoderSpec,
    FeatureProviderSpec,
    FusionSpec,
    build_calibrator,
    build_encoder,
    build_feature_providers,
    build_fusion,
    encoder_is_corpus_dependent,
    fit_encoder,
    register_calibrator,
    register_encoder,
    register_feature_provider,
    register_fusion,
)
from .retrieval import (
    BM25Index,
    DenseRetrieverAdapter,
    DenseState,
    LexicalRetrieverAdapter,
)

__all__ = [
    "SentenceTransformerEncoder",
    "TfidfEncoder",
    "HashingEncoder",
    "train_encoder",
    "fit_tfidf_encoder",
    "BM25Index",
    "DenseRetrieverAdapter",
    "DenseState",
    "LexicalRetrieverAdapter",
    "XGBoostFusionModel",
    "LightGBMFusionModel",
    "XGBRankerFusionModel",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "BetaCalibrator",
    "ArtifactRepository",
    "DeployedArtifacts",
    "ClassKeywordOverlapProvider",
    "EncoderSpec",
    "FusionSpec",
    "CalibratorSpec",
    "FeatureProviderSpec",
    "build_encoder",
    "build_fusion",
    "build_calibrator",
    "build_feature_providers",
    "encoder_is_corpus_dependent",
    "fit_encoder",
    "register_encoder",
    "register_fusion",
    "register_calibrator",
    "register_feature_provider",
]
