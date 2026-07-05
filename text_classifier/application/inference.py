"""Inference pipeline (application service).

Loads a trained model directory and classifies new items. Leakage-free by
construction: a genuinely new item has no self-match in the index.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..config import PipelineConfig
from ..domain import CandidatePolicy, LabelSpace, Prediction
from ..infrastructure import ArtifactRepository, DeployedArtifacts
from .features import FeatureAssembler
from .scoring import add_confidence, top_k_per_item, top_per_item


class InferencePipeline:
    def __init__(self, artifacts: DeployedArtifacts):
        self._a = artifacts
        self._assembler = FeatureAssembler(
            artifacts.label_space, CandidatePolicy(artifacts.config.candidate_top_n)
        )

    @classmethod
    def from_directory(cls, directory: str) -> "InferencePipeline":
        return cls(ArtifactRepository().load(directory))

    @property
    def label_space(self) -> LabelSpace:
        return self._a.label_space

    @property
    def config(self) -> PipelineConfig:
        return self._a.config

    def predict(self, texts: Sequence[str]) -> List[Prediction]:
        """Classify each input string.

        Empty strings are accepted: they encode to a degenerate vector that
        retrieves nothing, so the corresponding item abstains (``top_key=""``,
        ``abstained=True``) rather than raising. Non-string inputs (including
        ``None``) are a programming error and raise ``TypeError`` before any
        encoding work begins, pointing at the offending index.
        """
        texts = list(texts)
        self._validate_texts(texts)
        a = self._a
        q_emb = a.encoder.encode(texts)
        feats = self._assembler.assemble(
            texts,
            q_emb,
            a.dense,
            a.lexical,
            a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
        )

        # Every item defaults to abstaining; this also covers items whose features
        # surfaced no candidate at all (and are therefore absent from `decided`).
        results: List[Prediction] = [
            Prediction(top_key="", confidence=0.0, abstained=True) for _ in texts
        ]
        if not len(feats):
            return results

        # Collapse to one decision per item, then score the whole batch at once:
        # pulling the columns into arrays lets us call `accept` a single time and
        # map class indices to keys vectorized — no per-row pandas loop on the hot
        # path (the trailing loop only packages the immutable Predictions).
        decided = top_per_item(add_confidence(feats, a.fusion, a.calibrator))
        item_ids = decided["item_id"].to_numpy(dtype=np.intp)
        candidates = decided["candidate"].to_numpy(dtype=np.intp)
        confidences = decided["conf"].to_numpy(dtype=np.float64)
        margins = decided["margin"].to_numpy(dtype=np.float64)
        second_candidates = decided["second_candidate"].to_numpy(dtype=np.float64)
        accepted = a.abstention.accept(confidences, candidates)
        top_keys = np.asarray(a.label_space.keys)[candidates]

        for item_id, top_key, conf, margin, ok, second in zip(
            item_ids, top_keys, confidences, margins, accepted, second_candidates
        ):
            key = str(top_key)
            runner_up_key = None if np.isnan(second) else str(a.label_space.keys[int(second)])
            results[item_id] = Prediction(
                top_key=key,
                confidence=float(conf),
                abstained=not ok,
                predicted_key=key if ok else None,
                runner_up_key=runner_up_key,
                margin=float(margin),
            )
        return results

    def predict_topk(self, texts: Sequence[str], k: int) -> List[List[Tuple[str, float]]]:
        """Per input, up to ``k`` ``(class_key, confidence)`` pairs ranked by
        confidence descending. Shorter than ``k`` when fewer candidates
        surfaced; empty for an item that retrieved nothing. Reuses a single
        feature-assembly + fusion pass — no extra scoring work per k."""
        texts = list(texts)
        self._validate_texts(texts)
        a = self._a
        q_emb = a.encoder.encode(texts)
        feats = self._assembler.assemble(
            texts,
            q_emb,
            a.dense,
            a.lexical,
            a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
        )
        results: List[List[Tuple[str, float]]] = [[] for _ in texts]
        if not len(feats):
            return results

        ranked = top_k_per_item(add_confidence(feats, a.fusion, a.calibrator), k)
        item_ids = ranked["item_id"].to_numpy(dtype=np.intp)
        candidates = ranked["candidate"].to_numpy(dtype=np.intp)
        confidences = ranked["conf"].to_numpy(dtype=np.float64)
        keys = np.asarray(a.label_space.keys)[candidates]

        for item_id, key, conf in zip(item_ids, keys, confidences):
            results[item_id].append((str(key), float(conf)))
        return results

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"InferencePipeline.predict expects str inputs; item at index {i} "
                    f"is {type(t).__name__}: {t!r}"
                )
