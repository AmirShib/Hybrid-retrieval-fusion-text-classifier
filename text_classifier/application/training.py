"""Training pipeline (application service).

Orchestrates the full training use case:
  1. out-of-fold feature generation (leakage-free: each item is scored against an
     index built from other folds, optionally with a per-fold-trained encoder);
  2. fusion model training on the training folds;
  3. isotonic calibration + threshold tuning on a held-out calibration fold;
  4. coverage/accuracy evaluation on an untouched test fold;
  5. final encoder + indices on all data, assembled into DeployedArtifacts.
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
    FEATURE_NAMES,
    LabeledItem,
    LabelSpace,
    ThresholdTuner,
)
from ..infrastructure import (
    ArtifactRepository,
    DenseRetrieverAdapter,
    DeployedArtifacts,
    IsotonicCalibrator,
    LexicalRetrieverAdapter,
    SentenceTransformerEncoder,
    XGBoostFusionModel,
    train_encoder,
)
from .features import FeatureAssembler
from .scoring import add_confidence, top_per_item

log = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: PipelineConfig, shared_encoder: Optional[SentenceTransformerEncoder] = None):
        self.cfg = config
        self.assembler: Optional[FeatureAssembler] = None
        # Optional injected encoder for the shared-encoder path (DI / offline tests).
        self._shared_override = shared_encoder

    def _load_shared_encoder(self) -> SentenceTransformerEncoder:
        if self._shared_override is not None:
            return self._shared_override
        return SentenceTransformerEncoder.load(
            self.cfg.encoder.model_name_or_path,
            self.cfg.encoder.encode_batch_size, self.cfg.encoder.device,
        )

    # ---------------------------------------------------------------- public API
    def run(self, items: Sequence[LabeledItem], label_space: LabelSpace,
            output_dir: Optional[str] = None) -> Tuple[DeployedArtifacts, CoverageReport]:
        items = list(items)
        self._validate_inputs(items, label_space)
        self.assembler = FeatureAssembler(label_space, CandidatePolicy(self.cfg.candidate_top_n))
        texts = [it.text for it in items]
        y = np.array(label_space.encode_labels([it.label for it in items]), dtype=np.int64)

        oof = self._build_oof(texts, y, label_space)
        fusion, calibrator, abstention = self._fit_fusion(oof)
        report = self._evaluate(oof, fusion, calibrator, abstention)

        artifacts = self._build_deployment(texts, y, label_space, fusion, calibrator, abstention)
        if output_dir:
            ArtifactRepository().save(artifacts, output_dir)
            log.info("saved trained pipeline to %s", output_dir)
        return artifacts, report

    # ---------------------------------------------------------------- validation
    def _validate_inputs(self, items: Sequence[LabeledItem], label_space: LabelSpace) -> None:
        """Fail fast, before any encoder/index work, on inputs that would otherwise
        surface as a cryptic numpy/pandas/sklearn traceback deep in the pipeline.

        Checks, in order:
          1. the item list is non-empty;
          2. the label space has at least two classes (fusion needs negatives);
          3. every item label is defined in ``label_space``;
          4. every class present has at least ``n_folds`` examples — the minimum
             ``StratifiedKFold`` requires per class.

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
                f"{len(unknown)} item label(s) are not defined in the LabelSpace: "
                f"{shown}{suffix}"
            )

        n_folds = self.cfg.training.n_folds
        underpopulated = sorted(
            ((k, c) for k, c in Counter(it.label for it in items).items() if c < n_folds),
            key=lambda kc: (kc[1], kc[0]),
        )
        if underpopulated:
            shown = underpopulated[:10]
            suffix = " ..." if len(underpopulated) > 10 else ""
            raise ValueError(
                f"StratifiedKFold(n_folds={n_folds}) needs at least {n_folds} examples "
                f"per class; these class(es) have too few (key, count): {shown}{suffix}"
            )

    # ---------------------------------------------------------------- (1) OOF
    def _encoder_for_split(self, items_idx: np.ndarray, texts, y, label_space,
                           shared: Optional[SentenceTransformerEncoder]) -> SentenceTransformerEncoder:
        if not self.cfg.training.use_per_fold_encoder:
            assert shared is not None
            return shared
        fold_items = [LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in items_idx]
        return train_encoder(fold_items, label_space, self.cfg.encoder)

    def _build_oof(self, texts: List[str], y: np.ndarray, label_space: LabelSpace) -> pd.DataFrame:
        shared = None
        if not self.cfg.training.use_per_fold_encoder:
            shared = self._load_shared_encoder()
        skf = StratifiedKFold(self.cfg.training.n_folds, shuffle=True,
                              random_state=self.cfg.training.random_state)
        frames, recall_hits, total = [], 0, 0
        for fold, (tr, va) in enumerate(skf.split(texts, y)):
            enc = self._encoder_for_split(tr, texts, y, label_space, shared)
            tr_texts = [texts[i] for i in tr]
            dense = DenseRetrieverAdapter.build(enc, tr_texts, y[tr], label_space, self.cfg.retrieval)
            lexical = LexicalRetrieverAdapter.build(tr_texts, y[tr], label_space, self.cfg.retrieval)

            va_texts = [texts[i] for i in va]
            q_emb = enc.encode(va_texts)
            feats = self.assembler.assemble(
                va_texts, q_emb, dense, lexical, self.cfg.retrieval.k_neighbors,
                query_ids=va, query_labels=y[va], chunk=self.cfg.retrieval.feature_chunk,
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

    # ----------------------------------------------------- (2-3) fusion + thresholds
    def _fit_fusion(self, oof: pd.DataFrame):
        roles = self.cfg.training.fold_roles()
        tr = oof[oof["fold"].isin(roles["train"])]
        ca = oof[oof["fold"].isin(roles["calibration"])]

        fusion = XGBoostFusionModel(self.cfg.fusion.xgb_params, self.cfg.fusion.auto_scale_pos_weight)
        fusion.fit(tr[FEATURE_NAMES].to_numpy(np.float32), tr["is_true"].to_numpy())

        raw = fusion.predict_proba(ca[FEATURE_NAMES].to_numpy(np.float32))
        calibrator = IsotonicCalibrator()
        calibrator.fit(raw, ca["is_true"].to_numpy())

        decided = top_per_item(add_confidence(ca, fusion, calibrator))
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
    def _evaluate(self, oof, fusion, calibrator, abstention) -> CoverageReport:
        roles = self.cfg.training.fold_roles()
        te = oof[oof["fold"].isin(roles["test"])]
        decided = top_per_item(add_confidence(te, fusion, calibrator))
        accept = abstention.accept(decided["conf"].to_numpy(), decided["candidate"].to_numpy())

        n = len(decided)
        coverage = float(accept.mean()) if n else 0.0
        acc_acc = float(decided.loc[accept, "is_true"].mean()) if accept.any() else float("nan")
        acc_all = float(decided["is_true"].mean()) if n else float("nan")
        recall = float(te.groupby("item_id")["is_true"].max().mean()) if n else 0.0
        report = CoverageReport(coverage, acc_acc, acc_all, recall, n)
        log.info("eval: coverage=%.3f acc_on_accepted=%.3f acc_no_abstain=%.3f",
                 coverage, acc_acc, acc_all)
        return report

    # ---------------------------------------------------------------- (5) deploy
    def _build_deployment(self, texts, y, label_space, fusion, calibrator, abstention) -> DeployedArtifacts:
        if self.cfg.training.use_per_fold_encoder:
            items = [LabeledItem(texts[i], label_space.key_at(int(y[i]))) for i in range(len(texts))]
            encoder = train_encoder(items, label_space, self.cfg.encoder)
        else:
            encoder = self._load_shared_encoder()
        dense = DenseRetrieverAdapter.build(encoder, texts, y, label_space, self.cfg.retrieval)
        lexical = LexicalRetrieverAdapter.build(texts, y, label_space, self.cfg.retrieval)
        return DeployedArtifacts(self.cfg, label_space, encoder, dense, lexical,
                                 fusion, calibrator, abstention)
