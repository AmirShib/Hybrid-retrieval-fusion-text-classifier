"""Configuration objects. Plain dataclasses so they serialize cleanly to JSON
and can be version-controlled alongside a trained model directory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EncoderConfig:
    kind: str = "sentence-transformers"   # registry key (see infrastructure/registry.py)
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    encode_batch_size: int = 64
    device: Optional[str] = None
    # fine-tuning (MultipleNegativesSymmetricRankingLoss on item<->description pairs)
    train_epochs: int = 1
    train_batch_size: int = 64
    warmup_ratio: float = 0.1
    # Backend-specific kwargs read by non-ST encoders. For kind="tfidf" these are
    # passed straight to sklearn's TfidfVectorizer (e.g. {"ngram_range": [1, 2],
    # "max_features": 50000, "stop_words": "english"}).
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    k_neighbors: int = 20
    k1: float = 1.5
    b: float = 0.75
    bm25_token_kwargs: Dict[str, Any] = field(default_factory=lambda: {"stop_words": "english"})
    dense_chunk: int = 256       # query chunking for kNN matmuls
    feature_chunk: int = 4096    # query chunking for feature assembly


@dataclass
class FusionConfig:
    kind: str = "xgboost"   # registry key (see infrastructure/registry.py)
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    })
    auto_scale_pos_weight: bool = True   # set scale_pos_weight = n_neg / n_pos at fit time
    # Generic params block read by non-xgboost backends (e.g. LightGBM in T41).
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    kind: str = "isotonic"   # registry key (see infrastructure/registry.py)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    n_folds: int = 5
    target_precision: float = 0.95
    per_class_min_support: int = 100
    use_per_fold_encoder: bool = False   # True = rigorous (refit encoder per fold), expensive
    random_state: int = 0

    def fold_roles(self) -> Dict[str, List[int]]:
        """Last fold = test, second-last = calibration, rest = fusion training."""
        folds = list(range(self.n_folds))
        return {"train": folds[:-2], "calibration": [folds[-2]], "test": [folds[-1]]}


@dataclass
class PipelineConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    candidate_top_n: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        # `.get` with defaults keeps older serialized configs (written before a
        # field existed) loadable — new fields fall back to their defaults.
        return cls(
            encoder=EncoderConfig(**data["encoder"]),
            retrieval=RetrievalConfig(**data["retrieval"]),
            fusion=FusionConfig(**data["fusion"]),
            calibration=CalibrationConfig(**data.get("calibration", {})),
            training=TrainingConfig(**data["training"]),
            candidate_top_n=data["candidate_top_n"],
        )
