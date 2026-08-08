"""Training pipeline (application service).

Orchestrates the full training use case:
  1. out-of-fold feature generation (leakage-free: each item is scored against an
     index built from other folds, optionally with a per-fold-trained encoder);
  2. fusion model training on the training folds;
  3. isotonic calibration + threshold tuning on a held-out calibration fold;
  4. coverage/accuracy evaluation on an untouched test fold;
  5. final encoder + indices on all data, assembled into DeployedArtifacts.

A caller who already holds a validation and/or test split can pass it to
``run`` (``val_items`` / ``test_items``) to replace step 3's calibration fold
and/or step 4's test fold with the external set — the drift-realistic temporal
split. External sets are featurized against the deployment index (step 5's
indices, built once up front and reused), never through the out-of-fold loop.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .._messages import format_preview
from ..config import CalibrationConfig, PipelineConfig
from ..domain import (
    AbstentionPolicy,
    ArrayOps,
    CandidatePolicy,
    ConfidenceCalibrator,
    CoverageReport,
    DenseRetriever,
    FeatureProvider,
    FusionModel,
    LabeledItem,
    LabelSpace,
    LexicalRetriever,
    SignalProvider,
    TextEncoder,
    ThresholdTuner,
    fusion_feature_names,
)
from ..infrastructure import (
    ArtifactRepository,
    DeployedArtifacts,
    build_array_ops,
    build_calibrator,
    build_encoder,
    build_feature_providers,
    build_fusion,
    build_signal_providers,
    encoder_is_corpus_dependent,
    fit_encoder,
    resolve_array_backend,
)
from ..infrastructure.array_ops import NumpyArrayOps
from .evaluation import build_manifest, evaluate_decisions, write_evaluation_artifacts
from .features import FeatureAssembler
from .indexing import RetrievalIndexBuilder
from .scoring import add_confidence, select_feature_columns, top_per_item
from .signal_report import signal_report

log = logging.getLogger(__name__)


def fit_calibration_and_abstention(
    ca: pd.DataFrame,
    fusion: FusionModel,
    calibration_cfg: CalibrationConfig,
    feature_names: Sequence[str],
    target_precision: float,
    per_class_min_support: int,
) -> Tuple[ConfidenceCalibrator, AbstentionPolicy]:
    """Fit a calibrator on ``fusion``'s raw scores for the calibration rows ``ca``,
    then tune the global + per-class abstention thresholds for ``target_precision``.

    ``ca`` must carry the raw feature columns (``feature_names``) plus ``is_true``
    (whether that candidate is the item's true class) and ``candidate`` (its class
    index) — the same shape as an out-of-fold or featurized-external frame.

    This is the decision-layer half of ``TrainingPipeline._fit_fusion`` (the fusion
    model itself is fit separately, before this is called), extracted so the
    re-tune use case can reuse the identical threshold logic against a fresh
    labeled set without duplicating it.
    """
    names = list(feature_names)
    raw = fusion.predict_proba(
        select_feature_columns(ca, names, context="fit_calibration_and_abstention").to_numpy(
            np.float32
        )
    )
    calibrator = build_calibrator(calibration_cfg)
    calibrator.fit(raw, ca["is_true"].to_numpy(), classes=ca["candidate"].to_numpy())

    decided = top_per_item(add_confidence(ca, fusion, calibrator, names))
    global_thr = ThresholdTuner.threshold_for_precision(
        decided["conf"].to_numpy(), decided["is_true"].to_numpy(), target_precision
    )
    per_class: Dict[int, float] = {}
    for cls, grp in decided.groupby("candidate"):
        if len(grp) >= per_class_min_support:
            per_class[int(cls)] = ThresholdTuner.threshold_for_precision(
                grp["conf"].to_numpy(), grp["is_true"].to_numpy(), target_precision
            )
    return calibrator, AbstentionPolicy(global_thr, per_class)


class TrainingPipeline:
    def __init__(self, config: PipelineConfig, shared_encoder: Optional[TextEncoder] = None):
        self.cfg = config
        self.assembler: Optional[FeatureAssembler] = None
        # Retrieval/encoder-facing backend, replaced in run() with this run's
        # resolved backend (numpy or, T85, torch).
        self._ops: ArrayOps = NumpyArrayOps()
        # FeatureAssembler/build_signal_providers-facing backend -- always
        # numpy (T85 deliberately does not touch application/features.py;
        # its kernels mix raw numpy indexing with the array port in ways that
        # are only safe against host arrays. T86 is what makes this backend-
        # polymorphic, at which point this can become `self._ops` again).
        self._assembler_ops: ArrayOps = NumpyArrayOps()
        # Optional injected encoder for the shared-encoder path (DI / offline tests).
        self._shared_override = shared_encoder
        # Custom feature providers fitted on all training data, and the
        # composed feature schema (core + provider columns). Populated when the
        # deployment index is built; the fusion/eval steps select X by this list.
        self._providers: List[FeatureProvider] = []
        # Deliberately *not* computed here. The composed schema is only knowable
        # once the FeatureProviders and SignalProviders exist, and this used to
        # be seeded with `fusion_feature_names(drop=...)` — the core-only
        # schema — which rejected a `drop_features` entry naming any provider
        # column before the run could start (a custom FeatureProvider's, or a
        # signal's; T33's `ce_*` columns are what surfaced it). Building
        # providers early just to read `column_names()` is not the fix either:
        # a reranker or a corpus-fitted encoder would be constructed, and a real
        # cross-encoder backend would load a model, purely to validate a string.
        # `_build_oof` (per fold) and `_build_deployment_index` both compute and
        # validate the real composed list, and every reader below runs after
        # them, so the deferral costs no safety.
        self._feature_names: List[str] = []
        # T34 phase 2: the SignalProviders that ship in the deployed model,
        # built once (on all training data) by `_build_deployment_index` and
        # reused by `_featurize_external`/the returned `DeployedArtifacts`,
        # mirroring `self._providers`'s lifecycle above.
        self._signal_providers: List[SignalProvider] = []
        # The frozen encoder shared across folds, and the index builder that
        # owns every once-per-run cache (document embeddings, BM25 tokenization
        # — see `application/indexing.py`). Both are per-run state, keyed to
        # that run's corpus and array backend: `run()` clears them and the
        # first index-construction site builds them (`_index_builder`).
        self._shared_encoder: Optional[TextEncoder] = None
        self._indexes: Optional[RetrievalIndexBuilder] = None

    def _index_builder(
        self, texts: Sequence[str], y: np.ndarray, label_space: LabelSpace
    ) -> RetrievalIndexBuilder:
        """The one ``RetrievalIndexBuilder`` for this run, built on first use.

        Both index-construction sites — the out-of-fold fold loop and the
        deployment build — go through here, and whichever runs first creates it
        (the k-fold path starts with the folds; the leave-one-out path starts
        with the deployment index). Sharing the instance is what shares the
        once-per-run encode/tokenize caches between them, so this is memoized
        rather than constructed per call. ``run()`` clears it, so a pipeline
        instance can be reused for a second corpus.

        ``shared_encoder`` is ``None`` on the per-fold-encoder path, which
        disables the embedding cache: a fold's vectors are stale the moment the
        next fold refits the encoder.
        """
        if self._indexes is None:
            self._indexes = RetrievalIndexBuilder(
                self.cfg,
                label_space,
                texts,
                y,
                self._ops,
                shared_encoder=(
                    None if self._use_per_fold_encoder() else self._load_shared_encoder()
                ),
            )
        return self._indexes

    def _use_per_fold_encoder(self) -> bool:
        """Refit the encoder per fold when explicitly requested, or whenever the
        encoder is corpus-dependent (e.g. TF-IDF) and no encoder was injected —
        a shared corpus-dependent encoder fit on all data would leak vocabulary
        from the validation rows into their own features."""
        if self.cfg.training.use_per_fold_encoder:
            return True
        return self._shared_override is None and encoder_is_corpus_dependent(self.cfg.encoder)

    def _load_shared_encoder(self) -> TextEncoder:
        """Build (or take the injected) frozen encoder shared across every fold.

        Called once per ``run()`` and memoized into ``self._shared_encoder``:
        the out-of-fold loop and the deployment build both need it, and for a
        pretrained backend a second call would re-load the same weights from
        disk for no benefit.
        """
        if self._shared_encoder is not None:
            return self._shared_encoder
        encoder = (
            self._shared_override
            if self._shared_override is not None
            else build_encoder(self.cfg.encoder)
        )
        # T85: upgrade a SentenceTransformerEncoder to hand back device-
        # resident tensors when this run resolved a torch backend, so the
        # embeddings that build DenseState never take a forced D2H on the way
        # out of the encoder. duck-typed (hasattr, not isinstance) so
        # TfidfEncoder/HashingEncoder -- always numpy -- and any injected
        # test double are untouched; a no-op when self._ops is numpy (every
        # existing caller, byte-for-byte unchanged).
        if self._ops.name != "numpy" and hasattr(encoder, "set_array_backend"):
            encoder.set_array_backend(self._ops.name)
        self._shared_encoder = encoder
        return encoder

    # ---------------------------------------------------------------- public API
    def run(
        self,
        items: Sequence[LabeledItem],
        label_space: LabelSpace,
        output_dir: Optional[str] = None,
        *,
        val_items: Optional[Sequence[LabeledItem]] = None,
        test_items: Optional[Sequence[LabeledItem]] = None,
    ) -> Tuple[DeployedArtifacts, CoverageReport]:
        """Train the pipeline on ``items`` and return the deployed artifacts.

        By default the calibration and test sets are carved out of ``items`` via
        the internal k-fold split. A caller who already holds a split can pass it
        in instead:

        - ``val_items`` — an external validation set. The calibration fold role
          is retired (that fold joins fusion training) and the calibrator +
          abstention thresholds are fit on ``val_items``.
        - ``test_items`` — an external test set. The test fold role is retired
          and the held-out evaluation runs on ``test_items``.

        Either is optional and independent; pass both to reserve every internal
        fold for fusion training (``n_folds >= 2`` then suffices for out-of-fold
        feature generation). External sets are encoded and featurized against the
        index built from *all* training items — the same index that ships in the
        deployed model — so they are scored under the production condition and
        never enter the out-of-fold loop. This is a deliberate, documented
        asymmetry: the fusion model is fit on per-fold-index features while the
        calibrator sees full-train-index features, which anchors confidence (and
        therefore the risk-coverage numbers in ``evaluation.json``) at the
        deployed operating point. It is exactly what makes calibrating on a
        temporally-later validation slice meaningful.

        With neither ``val_items`` nor ``test_items`` the behaviour is identical
        to before this parameter existed.
        """
        items = list(items)
        val_items = None if val_items is None else list(val_items)
        test_items = None if test_items is None else list(test_items)
        self._validate_inputs(items, label_space, val_items, test_items)
        # T84: resolved once per run from this run's scale (auto picks numpy vs.
        # a device-resident backend from T83's crossover; explicit always wins).
        # The same instance is threaded through every FeatureAssembler/
        # DenseRetrieverAdapter this run builds, so a run never mixes backends.
        backend = resolve_array_backend(
            None if self.cfg.array_backend == "auto" else self.cfg.array_backend,
            n_items=len(items),
            n_classes=label_space.size,
            k_neighbors=self.cfg.retrieval.k_neighbors,
        )
        self._ops = build_array_ops(backend)
        self.assembler = FeatureAssembler(
            label_space, CandidatePolicy(self.cfg.candidate_top_n), self._assembler_ops
        )
        texts = [it.text for it in items]
        y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)

        # Per-run state, reset here so a pipeline instance can be run twice: the
        # index builder's caches and the shared encoder belong to *this* corpus
        # and this run's array backend.
        self._shared_encoder = None
        self._indexes = None

        roles = self.cfg.training.fold_roles(
            external_val=val_items is not None, external_test=test_items is not None
        )

        # Build the deployment index (encoder + dense/lexical over *all* training
        # items) once, up front: external val/test items are featurized against
        # it, and the finished DeployedArtifacts reuse the very same objects.
        if self.cfg.training.n_folds == 1:
            # Leave-one-out: build the deployment index first, then featurize every
            # training item against it with itself masked out — no k-fold loop.
            encoder, dense, lexical = self._build_deployment_index(texts, y, label_space)
            oof = self._build_loo(texts, y, label_space, encoder, dense, lexical)
        else:
            oof = self._build_oof(texts, y, label_space)
            encoder, dense, lexical = self._build_deployment_index(texts, y, label_space)
        val_feats = None
        if val_items is not None:
            val_feats, _ = self._featurize_external(val_items, label_space, encoder, dense, lexical)
        test_feats = None
        test_y: Optional[np.ndarray] = None
        if test_items is not None:
            test_feats, test_y = self._featurize_external(
                test_items, label_space, encoder, dense, lexical
            )

        fusion, calibrator, abstention = self._fit_fusion(oof, roles, val_feats)
        report, evaluation = self._evaluate(
            oof, roles, y, label_space, fusion, calibrator, abstention, test_feats, test_y
        )

        artifacts = DeployedArtifacts(
            self.cfg,
            label_space,
            encoder,
            dense,
            lexical,
            fusion,
            calibrator,
            abstention,
            feature_providers=self._providers,
            signal_providers=self._signal_providers,
        )
        if output_dir:
            repo = ArtifactRepository()
            repo.save(artifacts, output_dir)
            if self.cfg.training.store_corpus:
                repo.save_corpus(output_dir, items)
            # Per-signal diagnostics from the leakage-free out-of-fold rows: how each
            # retrieval technique performs *alone*, before fusion combines them. This
            # is the "which signals carry my data" evidence a data scientist reads
            # alongside the headline metrics.
            evaluation = {**evaluation, "signal_report": signal_report(oof)}
            # Persist the held-out evaluation + a provenance manifest next to the
            # model so a trained directory carries its own evidence: how it scored,
            # on what, and with which version/config. This is what makes a deployed
            # model auditable after the fact, not just at training time.
            manifest = build_manifest(
                n_training_items=len(items),
                n_classes=label_space.size,
                config=self.cfg,
                n_evaluated=report.n_items,
                splits=self._split_provenance(val_items, test_items),
            )
            write_evaluation_artifacts(output_dir, evaluation, manifest)
            log.info("saved trained pipeline + evaluation to %s", output_dir)
        return artifacts, report

    @staticmethod
    def _split_provenance(
        val_items: Optional[Sequence[LabeledItem]],
        test_items: Optional[Sequence[LabeledItem]],
    ) -> dict:
        """Record, for the manifest, whether each split came from an external set
        or an internal fold — so a persisted model dir is auditable after the fact."""
        return {
            "val": f"external:n={len(val_items)}" if val_items is not None else "internal-fold",
            "test": f"external:n={len(test_items)}" if test_items is not None else "internal-fold",
        }

    # ---------------------------------------------------------------- validation
    def _validate_inputs(
        self,
        items: Sequence[LabeledItem],
        label_space: LabelSpace,
        val_items: Optional[Sequence[LabeledItem]] = None,
        test_items: Optional[Sequence[LabeledItem]] = None,
    ) -> None:
        """Fail fast, before any encoder/index work, on inputs that would otherwise
        surface as a cryptic numpy/pandas/sklearn traceback deep in the pipeline.

        Checks, in order:
          0. the config itself is coherent (``PipelineConfig.validate``) — before
             any data work, so a bad ``n_folds`` never reaches StratifiedKFold;
             external splits relax the fold floor to ``>= 2``;
          1. the item list is non-empty;
          2. the label space has at least two classes (fusion needs negatives);
          3. every item label is defined in ``label_space``;
          4. every class present has at least ``n_folds`` examples — the minimum
             ``StratifiedKFold`` requires per class;
          5. each external split (val/test), if supplied, has only known labels
             and does not overlap the training text (see ``_validate_external``).

        A class that is declared in the label space but has *zero* training
        examples is acceptable (it simply never appears in the folds and gets an
        all-NaN prototype); only classes with 1..n_folds-1 examples are rejected,
        because StratifiedKFold cannot split them.

        Note on the minimum-support invariant: a dataset where every item belongs
        to a *single* class is acceptable (StratifiedKFold simply splits within
        that class), provided that class clears the per-class minimum. What
        StratifiedKFold cannot do is split a class with fewer than ``n_folds``
        members, so that is the boundary we guard.
        """
        self.cfg.validate(external_val=val_items is not None, external_test=test_items is not None)

        if not items:
            raise ValueError("TrainingPipeline.run requires a non-empty list of items")

        if label_space.size < 2:
            raise ValueError(
                "TrainingPipeline needs at least 2 classes to train a fusion model "
                f"and calibrate it; LabelSpace has {label_space.size}. "
                "(A single-class problem has no negatives to learn from.)"
            )

        unknown = label_space.unknown_keys(it.label for it in items)
        if unknown:
            raise ValueError(
                f"{len(unknown)} item label(s) are not defined in the LabelSpace: "
                f"{format_preview(unknown)}"
            )

        n_folds = self.cfg.training.n_folds
        underpopulated = sorted(
            ((k, c) for k, c in Counter(it.label for it in items).items() if c < n_folds),
            key=lambda kc: (kc[1], kc[0]),
        )
        if underpopulated:
            raise ValueError(
                f"StratifiedKFold(n_folds={n_folds}) needs at least {n_folds} examples "
                f"per class; these class(es) have too few (key, count): "
                f"{format_preview(underpopulated)}"
            )

        # External splits: check labels + leakage before any encoding happens.
        train_texts = {it.text for it in items}
        self._validate_external("validation", val_items, label_space, train_texts)
        self._validate_external("test", test_items, label_space, train_texts)
        if val_items is not None and test_items is not None:
            val_texts = {it.text for it in val_items}
            n_shared = sum(1 for it in test_items if it.text in val_texts)
            if n_shared:
                log.warning(
                    "%d item(s) appear in both the external validation and test sets; "
                    "calibration and evaluation are no longer independent",
                    n_shared,
                )

    def _validate_external(
        self,
        name: str,
        ext_items: Optional[Sequence[LabeledItem]],
        label_space: LabelSpace,
        train_texts: set,
    ) -> None:
        """Validate one external split (val or test) against the same fail-fast
        contract as the training inputs, before any encoding.

        - The split, if supplied, must be non-empty.
        - Every label must be defined in ``label_space`` (checked before encoding
          so an unknown label surfaces here, not deep in ``encode_labels``).
        - No exact-text overlap with the training items. An overlapping item sits
          in the deployed index the external set is scored against, self-retrieves
          with a perfect match, and silently inflates calibration/evaluation — the
          same leakage trap the retune CLI warns about. This is a hard error, not a warning.
        """
        if ext_items is None:
            return
        if not ext_items:
            raise ValueError(f"external {name} set was provided but is empty")

        unknown = label_space.unknown_keys(it.label for it in ext_items)
        if unknown:
            raise ValueError(
                f"{len(unknown)} external {name}-set label(s) are not defined in the "
                f"LabelSpace: {format_preview(unknown)}"
            )

        n_overlap = sum(1 for it in ext_items if it.text in train_texts)
        if n_overlap:
            raise ValueError(
                f"{n_overlap} item(s) in the external {name} set have text identical to a "
                f"training item. Such an item sits in the deployed index, self-retrieves a "
                f"perfect match, and silently inflates calibration/evaluation. The external "
                f"{name} set must be disjoint from the training items."
            )

    # ---------------------------------------------------------------- (1) OOF
    def _encoder_for_split(
        self, items_idx: np.ndarray, texts, y, label_space, shared: Optional[TextEncoder]
    ) -> TextEncoder:
        if not self._use_per_fold_encoder():
            assert shared is not None
            return shared
        fold_items = [LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in items_idx]
        return fit_encoder(self.cfg.encoder, fold_items, label_space)

    def _fit_providers(self, items_idx: np.ndarray, texts, y, label_space) -> List[FeatureProvider]:
        """Build and fit the custom feature providers on the rows in
        ``items_idx`` only. Called per fold on that fold's *training* rows, so a
        provider's training-derived state (e.g. a class lexicon) never includes the
        held-out items it will score — the same out-of-fold discipline as
        prototypes and indices. Returns ``[]`` when none are configured."""
        providers = build_feature_providers(self.cfg.features)
        if not providers:
            return providers
        fold_items = [LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in items_idx]
        for provider in providers:
            provider.fit(fold_items, label_space)
        return providers

    def _build_oof(self, texts: List[str], y: np.ndarray, label_space: LabelSpace) -> pd.DataFrame:
        assert self.assembler is not None  # set in run() before this is called
        indexes = self._index_builder(texts, y, label_space)
        shared = None if self._use_per_fold_encoder() else self._load_shared_encoder()
        skf = StratifiedKFold(
            self.cfg.training.n_folds, shuffle=True, random_state=self.cfg.training.random_state
        )
        frames, recall_hits, total = [], 0, 0
        for fold, (tr, va) in enumerate(skf.split(texts, y)):
            enc = self._encoder_for_split(tr, texts, y, label_space, shared)
            # This fold's retrieval state, over its *training* rows only — the
            # leakage rule. Whether that reuses the run-wide caches (T88/T32) or
            # rebuilds from scratch is the index builder's decision, made once
            # and applied identically here and in `_build_deployment_index`.
            dense, lexical = indexes.build(enc, tr)
            # Providers are fit on this fold's training rows only (leakage-free).
            providers = self._fit_providers(tr, texts, y, label_space)
            # T34 phase 2: the SignalProviders for this fold, wrapping this
            # fold's `dense`/`lexical` retrievers (same leakage-free discipline
            # -- built fresh per fold, exactly like `dense`/`lexical` above).
            signal_providers = build_signal_providers(
                self.cfg.retrieval, self.cfg.signals, dense, lexical, self._assembler_ops
            )
            # T87: request only the columns the fusion model will actually be
            # fitted on. This is the accepted diagnostic-narrowing trade —
            # `signal_report(oof)` downstream only sees what survives here — and
            # is fold-invariant (same drop list, provider names are stable
            # before/after fit), so every fold's frame carries the same columns.
            requested = fusion_feature_names(
                providers, self.cfg.fusion.drop_features, signal_providers
            )

            va_texts = [texts[i] for i in va]
            # T89: the held-out rows' query embeddings, taken from the run-wide
            # document cache when that is legitimate rather than encoded again.
            # Across the folds the `va` sets partition the item list, so this is
            # a whole extra pass over the corpus saved per run.
            q_emb = indexes.query_embeddings(enc, va)
            feats = self.assembler.assemble(
                va_texts,
                q_emb,
                dense,
                lexical,
                self.cfg.retrieval.k_neighbors,
                query_ids=va,
                query_labels=y[va],
                chunk=self.cfg.retrieval.feature_chunk,
                providers=providers,
                requested=requested,
                signal_providers=signal_providers,
            )
            feats["fold"] = fold
            frames.append(feats)

            recall_hits += int(feats.groupby("item_id")["is_true"].max().sum())
            total += len(va)
            log.info("fold %d: %d items, %d feature rows", fold, len(va), len(feats))

        recall = recall_hits / max(total, 1)
        log.info("candidate recall = %.4f (the ceiling on system accuracy)", recall)
        oof = pd.concat(frames, ignore_index=True)
        oof.attrs["candidate_recall"] = recall
        return oof

    def _build_loo(self, texts, y, label_space, encoder, dense, lexical) -> pd.DataFrame:
        """Leave-one-out featurization of the training items (``n_folds == 1``).

        Every training item is scored against the *deployment* index — the same
        dense/BM25 indices and prototypes that ship in the model, built over all
        training items — with that item masked out of its own neighbors and its own
        class prototype (see ``FeatureAssembler.assemble``'s ``self_ids``). This
        gives each item the maximum-size, deployment-matching index while keeping
        the out-of-fold leakage rule (an item never sees itself in its own index),
        and reuses the single deployment index instead of rebuilding one per fold.

        Valid only with both external splits (enforced in ``PipelineConfig.validate``
        and reachable only via ``fold_roles(n_folds=1)``), so every row here trains
        the fusion model; the frame carries a single synthetic ``fold = 0`` and no
        custom providers (rejected upstream — there is no per-item fit hook). It is
        the drop-in replacement for ``_build_oof``'s output on the LOO path.
        """
        assert self.assembler is not None  # set in run() before this is called
        indexes = self._index_builder(texts, y, label_space)
        self_ids = np.arange(len(texts))
        # T89: `_build_deployment_index` ran first on this path and, on the
        # shared-encoder path, already encoded exactly this corpus into the
        # cache — the builder owns that corpus, so reusing it whole needs no
        # "same list" guard here.
        q_emb = indexes.query_embeddings(encoder)
        feats = self.assembler.assemble(
            texts,
            q_emb,
            dense,
            lexical,
            self.cfg.retrieval.k_neighbors,
            query_ids=self_ids,
            query_labels=y,
            chunk=self.cfg.retrieval.feature_chunk,
            providers=self._providers,
            self_ids=self_ids,
            requested=self._feature_names,
            signal_providers=self._signal_providers,
        )
        feats["fold"] = 0
        recall = float(feats.groupby("item_id")["is_true"].max().mean()) if len(feats) else 0.0
        log.info(
            "leave-one-out: %d items, %d feature rows; candidate recall = %.4f",
            len(texts),
            len(feats),
            recall,
        )
        feats.attrs["candidate_recall"] = recall
        return feats

    def _featurize_external(
        self,
        ext_items: Sequence[LabeledItem],
        label_space: LabelSpace,
        encoder: TextEncoder,
        dense: DenseRetriever,
        lexical: Optional[LexicalRetriever],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Featurize an external val/test set against the *deployment* index.

        The dense/lexical indices here are the ones built from all training items
        (the same objects that ship in the model), so external items are scored
        under the production condition and never touch the out-of-fold loop. The
        returned frame carries the same columns as an OOF fold (feature columns +
        ``item_id`` + ``candidate`` + ``is_true``) but no ``fold`` column: external
        rows are consumed directly, not filtered by fold role.

        Returns ``(features, y)`` where ``y`` is the external labels encoded to
        class indices, aligned to ``item_id`` (which is a positional index into
        ``ext_items``) — the evaluation path needs it to recover the true class of
        items whose true label missed the candidate set.
        """
        assert self.assembler is not None  # set in run() before this is called
        texts = [it.text for it in ext_items]
        y = np.array(label_space.encode_labels([it.label for it in ext_items]), dtype=np.int64)
        q_emb = encoder.encode_queries(texts)
        feats = self.assembler.assemble(
            texts,
            q_emb,
            dense,
            lexical,
            self.cfg.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=y,
            chunk=self.cfg.retrieval.feature_chunk,
            # The deployment providers (fit on all training data) — the same ones
            # that ship in the model — so external items are scored under the
            # production condition, exactly like the dense/lexical indices here.
            providers=self._providers,
            requested=self._feature_names,
            signal_providers=self._signal_providers,
        )
        return feats, y

    # ----------------------------------------------------- (2-3) fusion + thresholds
    def _fit_fusion(self, oof: pd.DataFrame, roles: dict, val_feats: Optional[pd.DataFrame] = None):
        """Fit the fusion model on the training folds, then the calibrator and
        abstention thresholds on the calibration data.

        The calibration data is the external validation set (``val_feats``) when
        one was supplied, otherwise the internal calibration fold. When external,
        ``roles["calibration"]`` is empty and that fold has already joined
        ``roles["train"]``, so no training rows are lost.
        """
        tr = oof[oof["fold"].isin(roles["train"])]
        ca = val_feats if val_feats is not None else oof[oof["fold"].isin(roles["calibration"])]

        names = self._feature_names
        fusion = build_fusion(self.cfg.fusion)
        if getattr(fusion, "NEEDS_GROUPS", False):
            # Learning-to-rank: each item's candidate rows form one query group.
            # Sort so groups are contiguous, then pass run-length group sizes.
            tr = tr.sort_values("item_id", kind="stable")
            groups = tr.groupby("item_id", sort=False).size().to_numpy()
            X_tr = select_feature_columns(tr, names, context="_fit_fusion (training rows)")
            fusion.fit(X_tr.to_numpy(np.float32), tr["is_true"].to_numpy(), groups=groups)
        else:
            X_tr = select_feature_columns(tr, names, context="_fit_fusion (training rows)")
            fusion.fit(X_tr.to_numpy(np.float32), tr["is_true"].to_numpy())

        calibrator, abstention = fit_calibration_and_abstention(
            ca,
            fusion,
            self.cfg.calibration,
            names,
            self.cfg.training.target_precision,
            self.cfg.training.per_class_min_support,
        )
        log.info(
            "global threshold=%.4f, %d per-class thresholds",
            abstention.global_threshold,
            len(abstention.per_class),
        )
        return fusion, calibrator, abstention

    # ---------------------------------------------------------------- (4) evaluate
    def _evaluate(
        self,
        oof,
        roles,
        y,
        label_space,
        fusion,
        calibrator,
        abstention,
        test_feats: Optional[pd.DataFrame] = None,
        test_y: Optional[np.ndarray] = None,
    ):
        """Score the held-out test set and assemble the full evaluation.

        The test set is the external test set (``test_feats``) when one was
        supplied, otherwise the untouched internal test fold. ``true_y`` is the
        matching label array: the external test's own labels, or the training
        labels for the internal fold (``item_id`` indexes into whichever set the
        features came from).

        Returns ``(CoverageReport, evaluation_dict)``: the report is the compact
        headline (kept for the public API), and the dict is the rich, persistable
        report (per-class breakdown, calibration, risk-coverage) built from the
        same per-item decisions.
        """
        if test_feats is not None:
            # test_feats and test_y are produced together by _featurize_external.
            assert test_y is not None
            te = test_feats
            true_y = test_y
        else:
            te = oof[oof["fold"].isin(roles["test"])]
            true_y = y
        decided = top_per_item(add_confidence(te, fusion, calibrator, self._feature_names))

        item_ids = decided["item_id"].to_numpy(dtype=np.intp)
        pred_idx = decided["candidate"].to_numpy(dtype=np.intp)
        conf = decided["conf"].to_numpy(dtype=np.float64)
        correct = decided["is_true"].to_numpy().astype(bool)
        accept = abstention.accept(conf, pred_idx)
        # item_id is the positional index into the scored set (query_ids), so
        # true_y[item_id] is the true class even for items whose true class missed
        # the candidate set.
        true_idx = true_y[item_ids]

        n = len(decided)
        coverage = float(accept.mean()) if n else 0.0
        acc_acc = float(correct[accept].mean()) if accept.any() else float("nan")
        acc_all = float(correct.mean()) if n else float("nan")
        recall = float(te.groupby("item_id")["is_true"].max().mean()) if n else 0.0
        report = CoverageReport(coverage, acc_acc, acc_all, recall, n)
        log.info(
            "eval: coverage=%.3f acc_on_accepted=%.3f acc_no_abstain=%.3f",
            coverage,
            acc_acc,
            acc_all,
        )

        evaluation = evaluate_decisions(
            confidence=conf,
            correct=correct,
            accepted=accept,
            pred_idx=pred_idx,
            true_idx=true_idx,
            keys=label_space.keys,
            candidate_recall=recall,
        )
        evaluation["abstention"] = {
            "global_threshold": abstention.global_threshold,
            "n_per_class_thresholds": len(abstention.per_class),
            "per_class": {label_space.key_at(c): thr for c, thr in abstention.per_class.items()},
        }
        return report, evaluation

    # ---------------------------------------------------------------- (5) deploy
    def _build_deployment_index(
        self, texts, y, label_space
    ) -> Tuple[TextEncoder, DenseRetriever, Optional[LexicalRetriever]]:
        """Fit the final encoder and build the dense + lexical indices over *all*
        training items.

        Built once and returned so a single set of index objects serves two
        purposes: featurizing any external val/test set (production-condition
        scoring) and shipping inside the ``DeployedArtifacts``. Splitting this out
        of deployment assembly is what lets the external sets be scored against the
        exact index the model will use in production.
        """
        indexes = self._index_builder(texts, y, label_space)
        if self._use_per_fold_encoder():
            items = [
                LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in range(len(texts))
            ]
            encoder = fit_encoder(self.cfg.encoder, items, label_space)
        else:
            encoder = self._load_shared_encoder()
        # The same construction the fold loop uses, over *all* rows. On the
        # k-fold path this reuses the caches `_build_oof` already populated; on
        # the leave-one-out path this runs first and populates them for
        # `_build_loo`. Either way there is exactly one encode/tokenize pass
        # over the corpus per run.
        dense, lexical = indexes.build(encoder, None)
        # Custom feature providers fit on *all* training rows — the version
        # that ships in the model and scores external val/test sets. The composed
        # schema (core + provider columns) is what the fusion/eval steps select by.
        self._providers = self._fit_providers(np.arange(len(texts)), texts, y, label_space)
        # T34 phase 2: the SignalProviders that ship in the deployed model,
        # wrapping the deployment `dense`/`lexical` indices built above (the
        # same objects `_featurize_external`/the returned `DeployedArtifacts` use).
        self._signal_providers = build_signal_providers(
            self.cfg.retrieval, self.cfg.signals, dense, lexical, self._assembler_ops
        )
        self._feature_names = fusion_feature_names(
            self._providers, self.cfg.fusion.drop_features, self._signal_providers
        )
        return encoder, dense, lexical
