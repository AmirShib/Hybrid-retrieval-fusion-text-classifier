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
        texts = list(texts)
        a = self._a
        q_emb = a.encoder.encode(texts)
        feats = self._assembler.assemble(
            texts, q_emb, a.dense, a.lexical, a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))), query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
        )

        results: List[Prediction] = [None] * len(texts)  # type: ignore[list-item]
        if len(feats):
            decided = top_per_item(add_confidence(feats, a.fusion, a.calibrator))
            for _, row in decided.iterrows():
                i = int(row["item_id"])
                cls = int(row["candidate"])
                conf = float(row["conf"])
                abstain = not bool(a.abstention.accept(np.array([conf]), np.array([cls]))[0])
                results[i] = Prediction(
                    top_key=a.label_space.key_at(cls),
                    confidence=conf,
                    abstained=abstain,
                    predicted_key=None if abstain else a.label_space.key_at(cls),
                    margin=float(row["margin"]) if "margin" in row else None,
                )

        # items where no candidate surfaced at all -> abstain
        for i, r in enumerate(results):
            if r is None:
                results[i] = Prediction(top_key="", confidence=0.0, abstained=True)
        return results
