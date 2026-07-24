"""Configuration objects. Plain dataclasses so they serialize cleanly to JSON
and can be version-controlled alongside a trained model directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

_T = TypeVar("_T")


@dataclass
class EncoderConfig:
    kind: str = "sentence-transformers"  # registry key (see infrastructure/registry.py)
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    encode_batch_size: int = 64
    device: Optional[str] = None
    # fine-tuning (MultipleNegativesSymmetricRankingLoss on item<->description pairs)
    train_epochs: int = 1
    train_batch_size: int = 64
    warmup_ratio: float = 0.1
    # Backend-specific kwargs. For kind="tfidf" these pass straight to sklearn's
    # TfidfVectorizer (e.g. {"ngram_range": [1, 2], "max_features": 50000}). For
    # kind="sentence-transformers" these pass straight to the SentenceTransformer
    # constructor (e.g. {"trust_remote_code": True, "revision": ..., "token": ...}) --
    # anything sentence_transformers.SentenceTransformer.__init__ accepts.
    params: Dict[str, Any] = field(default_factory=dict)
    # Encode-time kwargs merged into every SentenceTransformer.encode(...) call
    # (e.g. {"truncate_dim": 256, "precision": "float32"}). User keys win over
    # our defaults EXCEPT normalize_embeddings/convert_to_numpy, which are forced
    # (the L2-norm invariant: dot product == cosine) -- overriding them is
    # ignored with a warning.
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Asymmetric encoding for instruction-tuned models (E5/BGE/GTE...). A
    # *_prompt is a literal prefix prepended to each text ("query: " /
    # "passage: "); a *_prompt_name selects a named prompt from the model's own
    # config (newer sentence-transformers). An explicit prompt wins over its
    # prompt_name. All None (the default) == symmetric encoding, byte-identical
    # to previous behaviour. Queries = the items being classified; documents =
    # the example pool + class descriptions they are matched against.
    query_prompt: Optional[str] = None
    document_prompt: Optional[str] = None
    query_prompt_name: Optional[str] = None
    document_prompt_name: Optional[str] = None


@dataclass
class RetrievalConfig:
    k_neighbors: int = 20
    k1: float = 1.5
    b: float = 0.75
    # No stopword removal by default: a language-specific filter is an opt-in
    # (train CLI: --bm25-stop-words english), not a hidden assumption that
    # degrades BM25 on non-English corpora. Any sklearn CountVectorizer kwarg
    # is accepted here (stop_words, token_pattern, ...).
    bm25_token_kwargs: Dict[str, Any] = field(default_factory=dict)
    dense_chunk: int = 256  # query chunking for kNN matmuls
    feature_chunk: int = 4096  # query chunking for feature assembly


@dataclass
class FusionConfig:
    kind: str = "xgboost"  # registry key (see infrastructure/registry.py)
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
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
            # Row/column subsampling draws from an RNG; an explicit seed keeps two
            # identical training runs identical (scores, thresholds, coverage).
            "random_state": 0,
        }
    )
    auto_scale_pos_weight: bool = True  # set scale_pos_weight = n_neg / n_pos at fit time
    # Generic params block read by non-xgboost backends (e.g. LightGBM in T41).
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    kind: str = "isotonic"  # registry key: "isotonic" | "platt" | "beta"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureProviderConfig:
    """One custom feature provider (T70): a registry ``kind`` plus its params.
    ``params`` is forwarded to the provider's factory (see
    ``infrastructure/registry.py``)."""

    kind: str  # registry key (see infrastructure/registry.py)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    """Custom fusion features (T70). ``providers`` is an *ordered* list — the
    provider columns are appended to the core ~28 in this order, and that composed
    order is persisted into ``meta.json``. Empty (the default) means the feature
    schema and outputs are byte-for-byte identical to a build without T70."""

    providers: List[FeatureProviderConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    n_folds: int = 5
    target_precision: float = 0.95
    per_class_min_support: int = 100
    use_per_fold_encoder: bool = False  # True = rigorous (refit encoder per fold), expensive
    random_state: int = 0
    # Persist the raw training corpus (text + label, gzip-compressed JSONL) into
    # the model dir as corpus.jsonl.gz (T68). `text-classifier-update` needs it
    # to add labeled examples later without retraining (BM25's IDF is
    # corpus-global, so appending examples requires the full corpus to refit
    # against). Opt out via --no-store-corpus for privacy/size; an update on a
    # dir with no persisted corpus still works via --base-items.
    store_corpus: bool = True

    def fold_roles(
        self, *, external_val: bool = False, external_test: bool = False
    ) -> Dict[str, List[int]]:
        """Assign each cross-validation fold a role.

        Default (no external splits): last fold = test, second-to-last =
        calibration, the rest train the fusion model.

        When the caller supplies an external validation and/or test set, the
        corresponding fold role is *retired* and its fold joins the fusion
        training folds — the calibrator/thresholds are then fit on the external
        val set, and the held-out evaluation runs on the external test set. A
        retired role returns an empty list (not a new type). With both external
        sets, every fold trains the fusion model; the k-fold machinery still
        runs because the fusion model's own training rows must stay leakage-free
        (out-of-fold feature generation), so ``n_folds >= 2`` is enough.

        ``n_folds == 1`` is the leave-one-out (LOO) mode, valid only when both
        external roles are supplied (there is no fold left to carve a calibration
        or test set from). It has a single synthetic training "fold" ``[0]``: the
        pipeline featurizes every training item against the deployment index with
        that item masked out, rather than running the k-fold loop.
        """
        folds = list(range(self.n_folds))
        # Reserve folds from the end so the no-external assignment is byte-for-byte
        # what it was before (test = last, calibration = second-to-last).
        test = [] if external_test else [folds.pop()]
        calibration = [] if external_val else [folds.pop()]
        return {"train": folds, "calibration": calibration, "test": test}


@dataclass
class PipelineConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    candidate_top_n: int = 10

    def validate(self, *, external_val: bool = False, external_test: bool = False) -> None:
        """Reject config values that produce silently broken runs or deep
        framework tracebacks. Raises ``ValueError`` naming the field, the
        received value, and the constraint.

        ``external_val``/``external_test`` relax the fold floor: each external
        split retires a fold role (calibration or test), so one train fold plus
        the two folds required for out-of-fold feature generation is no longer
        the floor. With either external split present the floor drops to
        ``n_folds >= 2`` (still two folds for OOF); with neither it stays
        ``>= 3`` (one train + one calibration + one test). With *both* external
        splits the floor drops to ``n_folds >= 1``: ``1`` selects leave-one-out
        featurization (each training item scored against every other, itself
        masked out), which is the leakage-free way to give every item the
        maximum-size index without a k-fold split.

        Registry-key existence (encoder/fusion/calibrator ``kind``) is *not*
        checked here: the registry already raises a good error at build time,
        and this module must stay import-free of infrastructure.
        """
        both_external = external_val and external_test
        if both_external:
            n_folds_ok = self.training.n_folds >= 1
            n_folds_constraint = (
                ">= 1 (both external roles are supplied; n_folds=1 selects leave-one-out "
                "featurization, n_folds>=2 uses a k-fold out-of-fold split)"
            )
        elif external_val or external_test:
            n_folds_ok = self.training.n_folds >= 2
            n_folds_constraint = (
                ">= 2 (an external split retires a fold role; two folds are still "
                "required for out-of-fold feature generation)"
            )
        else:
            n_folds_ok = self.training.n_folds >= 3
            n_folds_constraint = ">= 3 (one train fold + one calibration fold + one test fold)"
        checks = [
            # fold_roles() needs >=1 train fold + 1 calibration + 1 test; with
            # n_folds=2 the fusion training set is silently empty. External
            # splits retire roles, so the floor relaxes to 2 (see above).
            (
                "training.n_folds",
                self.training.n_folds,
                n_folds_ok,
                n_folds_constraint,
            ),
            (
                "training.target_precision",
                self.training.target_precision,
                0.0 < self.training.target_precision <= 1.0,
                "in (0, 1]",
            ),
            (
                "training.per_class_min_support",
                self.training.per_class_min_support,
                self.training.per_class_min_support >= 1,
                ">= 1",
            ),
            ("candidate_top_n", self.candidate_top_n, self.candidate_top_n >= 1, ">= 1"),
            (
                "retrieval.k_neighbors",
                self.retrieval.k_neighbors,
                self.retrieval.k_neighbors >= 1,
                ">= 1",
            ),
            (
                "retrieval.dense_chunk",
                self.retrieval.dense_chunk,
                self.retrieval.dense_chunk >= 1,
                ">= 1 (a zero chunk never advances)",
            ),
            (
                "retrieval.feature_chunk",
                self.retrieval.feature_chunk,
                self.retrieval.feature_chunk >= 1,
                ">= 1 (a zero chunk never advances)",
            ),
            (
                "encoder.encode_batch_size",
                self.encoder.encode_batch_size,
                self.encoder.encode_batch_size >= 1,
                ">= 1",
            ),
        ]
        problems = [
            f"{name} must be {constraint}; got {value!r}"
            for name, value, ok, constraint in checks
            if not ok
        ]
        # Leave-one-out featurization (n_folds=1) has no per-item fit hook, so a
        # custom feature provider fit on all training rows would see the very item
        # it later scores — the exact leakage LOO's self-masking otherwise removes.
        # Reject the combination rather than leak silently.
        if both_external and self.training.n_folds == 1 and self.features.providers:
            problems.append(
                "training.n_folds=1 (leave-one-out featurization) does not support custom "
                "feature providers (features.providers); a provider fit on all training rows "
                "would see the item it scores. Use n_folds>=2 (k-fold out-of-fold) with providers."
            )
        if problems:
            raise ValueError("invalid PipelineConfig: " + "; ".join(problems))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        # Every section is optional (`.get` with a default of `{}`), so a
        # *partial* config file — e.g. `{"fusion": {"kind": "lightgbm"}}` —
        # loads fine and everything else falls back to its dataclass default.
        # Unknown keys are a typo, not a feature: they raise rather than
        # silently vanish, naming the offending key, the section, and the
        # valid keys for that section.
        valid_top = {f.name for f in fields(cls)}
        unknown_top = sorted(set(data) - valid_top)
        if unknown_top:
            raise ValueError(
                f"invalid PipelineConfig: unknown key(s) {unknown_top} in section "
                f"'PipelineConfig'; valid keys: {sorted(valid_top)}"
            )
        return cls(
            encoder=_build_section(EncoderConfig, data, "encoder"),
            retrieval=_build_section(RetrievalConfig, data, "retrieval"),
            fusion=_build_section(FusionConfig, data, "fusion"),
            calibration=_build_section(CalibrationConfig, data, "calibration"),
            training=_build_section(TrainingConfig, data, "training"),
            features=_build_features_section(data),
            candidate_top_n=data.get("candidate_top_n", cls().candidate_top_n),
        )


def _build_features_section(data: Dict[str, Any]) -> FeaturesConfig:
    """Build the ``features`` section: a list of ``FeatureProviderConfig``.

    Its shape (a list of provider objects) differs from the other single-object
    sections, so it gets its own builder. Unknown keys — at the section level, or
    inside any provider entry — are rejected by name, and a provider entry missing
    its required ``kind`` raises a clear error rather than a bare ``TypeError``."""
    sub = data.get("features") or {}
    if not isinstance(sub, dict):
        raise ValueError(
            f"invalid PipelineConfig: section 'features' must be an object; got {sub!r}"
        )
    valid = {f.name for f in fields(FeaturesConfig)}
    unknown = sorted(set(sub) - valid)
    if unknown:
        raise ValueError(
            f"invalid PipelineConfig: unknown key(s) {unknown} in section 'features'; "
            f"valid keys: {sorted(valid)}"
        )
    raw = sub.get("providers") or []
    if not isinstance(raw, list):
        raise ValueError(
            f"invalid PipelineConfig: 'features.providers' must be a list; got {raw!r}"
        )
    provider_keys = {f.name for f in fields(FeatureProviderConfig)}
    providers: List[FeatureProviderConfig] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"invalid PipelineConfig: 'features.providers[{i}]' must be an object; "
                f"got {entry!r}"
            )
        entry_unknown = sorted(set(entry) - provider_keys)
        if entry_unknown:
            raise ValueError(
                f"invalid PipelineConfig: unknown key(s) {entry_unknown} in "
                f"'features.providers[{i}]'; valid keys: {sorted(provider_keys)}"
            )
        if "kind" not in entry:
            raise ValueError(
                f"invalid PipelineConfig: 'features.providers[{i}]' is missing required key 'kind'"
            )
        providers.append(
            FeatureProviderConfig(kind=entry["kind"], params=entry.get("params") or {})
        )
    return FeaturesConfig(providers=providers)


def _build_section(dc_cls: Type[_T], data: Dict[str, Any], section_name: str) -> _T:
    """Build one nested config dataclass from ``data[section_name]``, tolerating
    a missing section (defaults apply) but rejecting unknown keys by name."""
    sub = data.get(section_name) or {}
    if not isinstance(sub, dict):
        raise ValueError(
            f"invalid PipelineConfig: section {section_name!r} must be an object; got {sub!r}"
        )
    valid = {f.name for f in fields(dc_cls)}  # type: ignore[arg-type]
    unknown = sorted(set(sub) - valid)
    if unknown:
        raise ValueError(
            f"invalid PipelineConfig: unknown key(s) {unknown} in section {section_name!r}; "
            f"valid keys: {sorted(valid)}"
        )
    return dc_cls(**sub)
