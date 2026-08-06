"""Configuration objects. Plain dataclasses so they serialize cleanly to JSON
and can be version-controlled alongside a trained model directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar

# The domain layer owns the set of encoder-epoch metrics; import it rather than
# restate it, so a new metric is valid in config the moment it can be measured.
# (Acyclic: `domain` imports neither config nor infrastructure.)
from .domain.services import ENCODER_SELECTION_METRICS

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
    # Best-epoch selection. With train_epochs > 1, `train_holdout_ratio` of the
    # fine-tuning items are held out from the gradient updates and re-scored after
    # every epoch; the epoch scoring best on `train_select_metric` is the one
    # returned and saved, instead of blindly the last. 0.0 turns selection off
    # (last epoch wins, and every item trains). Ignored when train_epochs == 1,
    # where there is nothing to choose between -- so the package default is
    # unchanged by these fields.
    train_holdout_ratio: float = 0.1
    # One of domain.services.ENCODER_SELECTION_METRICS. "knn_acc@1" additionally
    # re-encodes the fine-tuning pool each epoch (slower, closer to the d_knn_*
    # signals); the desc_* metrics only re-encode the holdout + descriptions.
    train_select_metric: str = "desc_acc@1"
    # How much an epoch must beat the incumbent by to count as an improvement --
    # noise suppression on small holdouts.
    train_select_min_delta: float = 0.0
    # Stop after this many consecutive non-improving epochs (0 = train them all).
    train_early_stopping_patience: int = 0
    # Seed for the stratified fine-tune/holdout split, so a rerun holds out the
    # same items (the determinism invariant).
    train_holdout_seed: int = 0
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
    # T34 phase 1: which retriever *builder* backs each signal. Registry keys
    # (see infrastructure/registry.py's register_dense_retriever/
    # register_lexical_retriever); the built-ins ("exact" wraps
    # DenseRetrieverAdapter, "bm25" wraps LexicalRetrieverAdapter) are the
    # byte-for-byte-identical defaults. This is what T31 (FAISS) and other
    # retriever backends plug into instead of forking the concrete adapter.
    dense_kind: str = "exact"
    lexical_kind: str = "bm25"
    # No stopword removal by default: a language-specific filter is an opt-in
    # (train CLI: --bm25-stop-words english), not a hidden assumption that
    # degrades BM25 on non-English corpora. Any sklearn CountVectorizer kwarg
    # is accepted here (stop_words, token_pattern, ...).
    bm25_token_kwargs: Dict[str, Any] = field(default_factory=dict)
    dense_chunk: int = 256  # query chunking for kNN matmuls
    feature_chunk: int = 4096  # query chunking for feature assembly
    # T32 A4: drop BM25 terms whose document frequency exceeds this fraction of
    # the corpus before building the weight matrix. The Lucene IDF already
    # trends to 0 as df -> n_docs, so a high-df term contributes almost nothing
    # to ranking while owning the longest postings list; pruning it shrinks the
    # weight matrix for near-zero ranking cost. `None` (the default) is off and
    # byte-for-byte today's behaviour — this is the one BM25 knob here that can
    # change scores, so it stays opt-in.
    bm25_max_df_ratio: Optional[float] = None
    # T32 B: reject (rather than silently allocate) a `BM25Index.score_matrix`
    # call that would densify a block larger than this many elements.
    # `score_matrix` is for the small class-description set; the example pool
    # must go through the chunked, sparse `top_k` path instead. `None` (the
    # default) is unbounded — today's behaviour.
    bm25_max_block_elems: Optional[int] = None
    # Threads used to parallelize BM25.top_k's per-chunk sparse mat-mul
    # (Qbin_chunk @ Wt). scipy's sparse @ releases the GIL during the C-level
    # multiply, so plain threads (no pickling of the (vocab, n_docs) Wt matrix
    # across processes) already parallelize this across cores. 1 (default) is
    # the original single-threaded behaviour, byte-for-byte; -1 uses
    # os.cpu_count().
    bm25_n_jobs: int = 1


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
    # Generic params block read by non-xgboost backends (e.g. LightGBM).
    params: Dict[str, Any] = field(default_factory=dict)
    # Feature columns withheld from the fusion model — this is what makes a
    # retrain-based ablation possible: "is the model better without this
    # column?", as opposed to the masking ablation's "what if this signal fails
    # at inference?". Persisted here inside meta.json's config block and
    # re-applied at load, so inference rebuilds the exact column list the model
    # was fitted on. Empty (the default) is the ordinary path and leaves the
    # schema byte-for-byte unchanged.
    #
    # T87: the assembler no longer computes the full schema unconditionally —
    # it computes what training/scoring *requests* (this narrowed list), and
    # skips the leaf-column work (ranks, margins, a fully-dropped custom
    # provider's `compute`) that nothing downstream reads. The trade this
    # accepts: `signal_report` on the training out-of-fold frame narrows with
    # it, reporting only the signals whose columns survived. `explain` /
    # `explain_records` / `importance_report` at inference time are unaffected
    # — they always request the full composed schema (`composed_feature_names`)
    # in a separate assembly pass, because they read core columns by name.
    drop_features: List[str] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    # registry key: "isotonic" | "platt" | "beta" | "per-class"
    kind: str = "isotonic"
    # "per-class" reads params["inner"] (isotonic|platt|beta, default "beta" --
    # a per-class slice of the calibration fold is small, which is where
    # isotonic overfits worst) and params["min_support"] (default 50): classes
    # with fewer than that many calibration rows fall back to a global curve
    # fit over all rows, mirroring AbstentionPolicy's per-class-with-global-
    # fallback thresholds.
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureProviderConfig:
    """One custom feature provider: a registry ``kind`` plus its params.
    ``params`` is forwarded to the provider's factory (see
    ``infrastructure/registry.py``)."""

    kind: str  # registry key (see infrastructure/registry.py)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    """Custom fusion features. ``providers`` is an *ordered* list — the
    provider columns are appended to the core ~28 in this order, and that composed
    order is persisted into ``meta.json``. Empty (the default) means the feature
    schema and outputs are byte-for-byte identical to a build with no custom
    providers configured."""

    providers: List[FeatureProviderConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    n_folds: int = 5
    target_precision: float = 0.95
    per_class_min_support: int = 100
    use_per_fold_encoder: bool = False  # True = rigorous (refit encoder per fold), expensive
    random_state: int = 0
    # Persist the raw training corpus (text + label, gzip-compressed JSONL) into
    # the model dir as corpus.jsonl.gz. `text-classifier-update` needs it
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
    # T34 phase 2: which SignalProvider(s) compute the retrieval signals that
    # feed candidate selection + the core feature columns. Registry keys (see
    # infrastructure/registry.py's register_signal_provider), mirroring how
    # retrieval.dense_kind/lexical_kind select a retriever *backend*: this
    # selects which *signals* run at all. The default ["dense", "lexical"] is
    # the byte-for-byte-identical five-signal schema; a third "kind" here adds
    # a whole new signal (its own matrices, joining candidate selection via its
    # own `candidate_features()`) without touching FeatureAssembler/pipelines.
    signals: List[str] = field(default_factory=lambda: ["dense", "lexical"])
    # T84: which ArrayOps backend runs the numeric kernels in feature assembly
    # and dense retrieval. Registry key (see infrastructure/registry.py),
    # "auto" (the default) picks numpy vs. torch from T83's measured crossover
    # thresholds using quantities known before assembly runs, and always falls
    # back to numpy when torch is absent, no device is visible, or the corpus
    # is below the crossover. An explicit "numpy"/"torch" always wins. This is
    # an execution choice recorded here for provenance, never load-bearing: a
    # model trained under any backend loads and scores on a numpy-only host.
    array_backend: str = "auto"

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
                "retrieval.bm25_max_df_ratio",
                self.retrieval.bm25_max_df_ratio,
                self.retrieval.bm25_max_df_ratio is None
                or 0.0 < self.retrieval.bm25_max_df_ratio <= 1.0,
                "None or in (0, 1]",
            ),
            (
                "retrieval.bm25_max_block_elems",
                self.retrieval.bm25_max_block_elems,
                self.retrieval.bm25_max_block_elems is None
                or self.retrieval.bm25_max_block_elems >= 1,
                "None or >= 1",
            ),
            (
                "encoder.encode_batch_size",
                self.encoder.encode_batch_size,
                self.encoder.encode_batch_size >= 1,
                ">= 1",
            ),
            (
                "encoder.train_epochs",
                self.encoder.train_epochs,
                self.encoder.train_epochs >= 1,
                ">= 1",
            ),
            (
                "encoder.train_batch_size",
                self.encoder.train_batch_size,
                self.encoder.train_batch_size >= 1,
                ">= 1",
            ),
            (
                "encoder.train_holdout_ratio",
                self.encoder.train_holdout_ratio,
                0.0 <= self.encoder.train_holdout_ratio <= 0.5,
                "in [0, 0.5] (0 disables best-epoch selection; a holdout larger "
                "than half the data starves the fine-tune)",
            ),
            (
                "encoder.train_select_metric",
                self.encoder.train_select_metric,
                self.encoder.train_select_metric in ENCODER_SELECTION_METRICS,
                f"one of {list(ENCODER_SELECTION_METRICS)}",
            ),
            (
                "encoder.train_select_min_delta",
                self.encoder.train_select_min_delta,
                self.encoder.train_select_min_delta >= 0.0,
                ">= 0",
            ),
            (
                "encoder.train_early_stopping_patience",
                self.encoder.train_early_stopping_patience,
                self.encoder.train_early_stopping_patience >= 0,
                ">= 0 (0 trains every epoch)",
            ),
            # Membership in the schema is checked later, by fusion_feature_names,
            # which is the only place the *composed* schema (core + providers) is
            # known. Here we only reject shapes that are wrong on their face.
            (
                "fusion.drop_features",
                self.fusion.drop_features,
                all(isinstance(n, str) and n.strip() for n in self.fusion.drop_features),
                "a list of non-empty feature-column names",
            ),
            (
                "fusion.drop_features",
                self.fusion.drop_features,
                len(set(self.fusion.drop_features)) == len(self.fusion.drop_features),
                "free of duplicates",
            ),
            (
                "signals",
                self.signals,
                bool(self.signals) and all(isinstance(s, str) and s.strip() for s in self.signals),
                "a non-empty list of non-empty registry-key strings",
            ),
            (
                "signals",
                self.signals,
                len(set(self.signals)) == len(self.signals),
                "free of duplicates",
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
            array_backend=data.get("array_backend", cls().array_backend),
            signals=list(data.get("signals", cls().signals)),
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
