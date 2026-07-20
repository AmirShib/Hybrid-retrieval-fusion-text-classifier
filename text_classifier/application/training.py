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
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import PipelineConfig
from ..domain import (
    AbstentionPolicy,
    CandidatePolicy,
    CoverageReport,
    FeatureProvider,
    LabeledItem,
    LabelSpace,
    TextEncoder,
    ThresholdTuner,
    composed_feature_names,
)
from ..infrastructure import (
    ArtifactRepository,
    DenseRetrieverAdapter,
    DeployedArtifacts,
    LexicalRetrieverAdapter,
    build_calibrator,
    build_encoder,
    build_feature_providers,
    build_fusion,
    encoder_is_corpus_dependent,
    fit_encoder,
)
from .evaluation import build_manifest, evaluate_decisions, write_evaluation_artifacts
from .features import FeatureAssembler
from .scoring import add_confidence, top_per_item

log = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: PipelineConfig, shared_encoder: Optional[TextEncoder] = None):
        self.cfg = config
        self.assembler: Optional[FeatureAssembler] = None
        # Optional injected encoder for the shared-encoder path (DI / offline tests).
        self._shared_override = shared_encoder
        # Custom feature providers (T70) fitted on all training data, and the
        # composed feature schema (core + provider columns). Populated when the
        # deployment index is built; the fusion/eval steps select X by this list.
        self._providers: List[FeatureProvider] = []
        self._feature_names: List[str] = composed_feature_names()

    def _use_per_fold_encoder(self) -> bool:
        """Refit the encoder per fold when explicitly requested, or whenever the
        encoder is corpus-dependent (e.g. TF-IDF) and no encoder was injected —
        a shared corpus-dependent encoder fit on all data would leak vocabulary
        from the validation rows into their own features."""
        if self.cfg.training.use_per_fold_encoder:
            return True
        return self._shared_override is None and encoder_is_corpus_dependent(self.cfg.encoder)

    def _load_shared_encoder(self) -> TextEncoder:
        if self._shared_override is not None:
            return self._shared_override
        return build_encoder(self.cfg.encoder)

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
        self.assembler = FeatureAssembler(label_space, CandidatePolicy(self.cfg.candidate_top_n))
        texts = [it.text for it in items]
        y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)

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
        )
        if output_dir:
            ArtifactRepository().save(artifacts, output_dir)
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

        known = set(label_space.keys)
        unknown = sorted({it.label for it in items if it.label not in known})
        if unknown:
            shown = unknown[:10]
            suffix = " ..." if len(unknown) > 10 else ""
            raise ValueError(
                f"{len(unknown)} item label(s) are not defined in the LabelSpace: {shown}{suffix}"
            )

        n_folds = self.cfg.training.n_folds
        underpopulated = sorted(
            ((k, c) for k, c in Counter(it.label for it in items).items() if c < n_folds),
            key=lambda kc: (kc[1], kc[0]),
        )
        if underpopulated:
            shown_counts = underpopulated[:10]
            suffix = " ..." if len(underpopulated) > 10 else ""
            raise ValueError(
                f"StratifiedKFold(n_folds={n_folds}) needs at least {n_folds} examples "
                f"per class; these class(es) have too few (key, count): {shown_counts}{suffix}"
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
          leakage trap T66 warns about. This is a hard error, not a warning.
        """
        if ext_items is None:
            return
        if not ext_items:
            raise ValueError(f"external {name} set was provided but is empty")

        known = set(label_space.keys)
        unknown = sorted({it.label for it in ext_items if it.label not in known})
        if unknown:
            shown = unknown[:10]
            suffix = " ..." if len(unknown) > 10 else ""
            raise ValueError(
                f"{len(unknown)} external {name}-set label(s) are not defined in the "
                f"LabelSpace: {shown}{suffix}"
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

    def _fit_providers(
        self, items_idx: np.ndarray, texts, y, label_space
    ) -> List[FeatureProvider]:
        """Build and fit the custom feature providers (T70) on the rows in
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
        shared = None
        if not self._use_per_fold_encoder():
            shared = self._load_shared_encoder()
        skf = StratifiedKFold(
            self.cfg.training.n_folds, shuffle=True, random_state=self.cfg.training.random_state
        )
        frames, recall_hits, total = [], 0, 0
        for fold, (tr, va) in enumerate(skf.split(texts, y)):
            enc = self._encoder_for_split(tr, texts, y, label_space, shared)
            tr_texts = [texts[i] for i in tr]
            dense = DenseRetrieverAdapter.build(
                enc, tr_texts, y[tr], label_space, self.cfg.retrieval
            )
            lexical = LexicalRetrieverAdapter.build(
                tr_texts, y[tr], label_space, self.cfg.retrieval
            )
            # Providers are fit on this fold's training rows only (leakage-free).
            providers = self._fit_providers(tr, texts, y, label_space)

            va_texts = [texts[i] for i in va]
            q_emb = enc.encode_queries(va_texts)
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

    def _build_loo(
        self, texts, y, label_space, encoder, dense, lexical
    ) -> pd.DataFrame:
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
        self_ids = np.arange(len(texts))
        q_emb = encoder.encode_queries(texts)
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
        dense: DenseRetrieverAdapter,
        lexical: LexicalRetrieverAdapter,
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
            fusion.fit(tr[names].to_numpy(np.float32), tr["is_true"].to_numpy(), groups=groups)
        else:
            fusion.fit(tr[names].to_numpy(np.float32), tr["is_true"].to_numpy())

        raw = fusion.predict_proba(ca[names].to_numpy(np.float32))
        calibrator = build_calibrator(self.cfg.calibration)
        calibrator.fit(raw, ca["is_true"].to_numpy())

        decided = top_per_item(add_confidence(ca, fusion, calibrator, names))
        target = self.cfg.training.target_precision
        global_thr = ThresholdTuner.threshold_for_precision(
            decided["conf"].to_numpy(), decided["is_true"].to_numpy(), target
        )
        per_class = {}
        for cls, grp in decided.groupby("candidate"):
            if len(grp) >= self.cfg.training.per_class_min_support:
                per_class[int(cls)] = ThresholdTuner.threshold_for_precision(
                    grp["conf"].to_numpy(), grp["is_true"].to_numpy(), target
                )
        log.info("global threshold=%.4f, %d per-class thresholds", global_thr, len(per_class))
        return fusion, calibrator, AbstentionPolicy(global_thr, per_class)

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
    ) -> Tuple[TextEncoder, DenseRetrieverAdapter, LexicalRetrieverAdapter]:
        """Fit the final encoder and build the dense + lexical indices over *all*
        training items.

        Built once and returned so a single set of index objects serves two
        purposes: featurizing any external val/test set (production-condition
        scoring) and shipping inside the ``DeployedArtifacts``. Splitting this out
        of deployment assembly is what lets the external sets be scored against the
        exact index the model will use in production.
        """
        if self._use_per_fold_encoder():
            items = [
                LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in range(len(texts))
            ]
            encoder = fit_encoder(self.cfg.encoder, items, label_space)
        else:
            encoder = self._load_shared_encoder()
        dense = DenseRetrieverAdapter.build(encoder, texts, y, label_space, self.cfg.retrieval)
        lexical = LexicalRetrieverAdapter.build(texts, y, label_space, self.cfg.retrieval)
        # Custom feature providers (T70) fit on *all* training rows — the version
        # that ships in the model and scores external val/test sets. The composed
        # schema (core + provider columns) is what the fusion/eval steps select by.
        self._providers = self._fit_providers(
            np.arange(len(texts)), texts, y, label_space
        )
        self._feature_names = composed_feature_names(self._providers)
        return encoder, dense, lexical
