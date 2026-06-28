"""Inference pipeline (application service).

Loads a trained model directory and classifies new items. Leakage-free by
construction: a genuinely new item has no self-match in the index.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config import PipelineConfig
from ..domain import CandidatePolicy, LabelSpace, Prediction
from ..infrastructure import ArtifactRepository, DeployedArtifacts
from .features import FeatureAssembler
from .scoring import add_confidence, top_per_item


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
            texts, q_emb, a.dense, a.lexical, a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))), query_labels=None,
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
        accepted = a.abstention.accept(confidences, candidates)
        top_keys = np.asarray(a.label_space.keys)[candidates]

        for item_id, top_key, conf, margin, ok in zip(
            item_ids, top_keys, confidences, margins, accepted
        ):
            key = str(top_key)
            results[item_id] = Prediction(
                top_key=key,
                confidence=float(conf),
                abstained=not ok,
                predicted_key=key if ok else None,
                margin=float(margin),
            )
        return results

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"InferencePipeline.predict expects str inputs; item at index {i} "
                    f"is {type(t).__name__}: {t!r}"
                )
