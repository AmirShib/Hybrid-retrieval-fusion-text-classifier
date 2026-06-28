"""Persistence: writes/reads a self-contained model directory. Uses only stdlib
pickle + numpy + json so there is no extra dependency and the directory is
portable to the air-gapped host.

Layout:
    <dir>/encoder/         SentenceTransformer.save() output
    <dir>/dense.npz        dense retriever numeric state
    <dir>/lexical.pkl      pickled LexicalRetrieverAdapter (vectorizers + BM25 weights)
    <dir>/fusion.json      XGBoost model
    <dir>/calibrator.pkl   isotonic regressor
    <dir>/meta.json        label space, thresholds, config, feature schema
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..config import PipelineConfig
from ..domain import (
    AbstentionPolicy,
    ClassDefinition,
    FEATURE_NAMES,
    LabelSpace,
)
from .encoder import SentenceTransformerEncoder
from .fusion import IsotonicCalibrator, XGBoostFusionModel
from .retrieval import DenseRetrieverAdapter, DenseState, LexicalRetrieverAdapter


@dataclass
class DeployedArtifacts:
    """Everything the inference pipeline needs, in memory."""
    config: PipelineConfig
    label_space: LabelSpace
    encoder: SentenceTransformerEncoder
    dense: DenseRetrieverAdapter
    lexical: LexicalRetrieverAdapter
    fusion: XGBoostFusionModel
    calibrator: IsotonicCalibrator
    abstention: AbstentionPolicy


class ArtifactRepository:
    """Reads/writes DeployedArtifacts to a directory."""

    def save(self, artifacts: DeployedArtifacts, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        artifacts.encoder.save(os.path.join(directory, "encoder"))

        s = artifacts.dense.state
        np.savez_compressed(
            os.path.join(directory, "dense.npz"),
            example_emb=s.example_emb, example_labels=s.example_labels,
            prototypes=s.prototypes, description_emb=s.description_emb, class_freq=s.class_freq,
        )
        with open(os.path.join(directory, "lexical.pkl"), "wb") as fh:
            pickle.dump(artifacts.lexical, fh)

        artifacts.fusion.save(os.path.join(directory, "fusion.json"))
        artifacts.calibrator.save(os.path.join(directory, "calibrator.pkl"))

        meta = {
            "feature_names": FEATURE_NAMES,
            "config": artifacts.config.to_dict(),
            "classes": [
                {"key": k, "description": d}
                for k, d in zip(artifacts.label_space.keys, artifacts.label_space.descriptions)
            ],
            "abstention": {
                "global_threshold": artifacts.abstention.global_threshold,
                "per_class": {str(k): v for k, v in artifacts.abstention.per_class.items()},
            },
        }
        with open(os.path.join(directory, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    def load(self, directory: str) -> DeployedArtifacts:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"model directory not found: {directory!r}")
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in model directory {directory!r} "
                f"(expected at {meta_path!r})"
            )
        with open(meta_path) as fh:
            meta = json.load(fh)

        self._check_feature_schema(meta.get("feature_names"))

        config = PipelineConfig.from_dict(meta["config"])
        label_space = LabelSpace([ClassDefinition(c["key"], c["description"]) for c in meta["classes"]])

        encoder = SentenceTransformerEncoder.load(
            os.path.join(directory, "encoder"),
            batch_size=config.encoder.encode_batch_size,
            device=config.encoder.device,
        )

        npz = np.load(os.path.join(directory, "dense.npz"))
        dense = DenseRetrieverAdapter(
            DenseState(npz["example_emb"], npz["example_labels"], npz["prototypes"],
                       npz["description_emb"], npz["class_freq"]),
            chunk=config.retrieval.dense_chunk,
        )
        with open(os.path.join(directory, "lexical.pkl"), "rb") as fh:
            lexical: LexicalRetrieverAdapter = pickle.load(fh)

        fusion = XGBoostFusionModel.load(os.path.join(directory, "fusion.json"))
        calibrator = IsotonicCalibrator.load(os.path.join(directory, "calibrator.pkl"))

        abstention = AbstentionPolicy(
            global_threshold=float(meta["abstention"]["global_threshold"]),
            per_class={int(k): float(v) for k, v in meta["abstention"]["per_class"].items()},
        )
        return DeployedArtifacts(config, label_space, encoder, dense, lexical, fusion, calibrator, abstention)

    @staticmethod
    def _check_feature_schema(saved_names) -> None:
        """Guard against schema drift between a persisted model and the running code.

        ``FEATURE_NAMES`` is the single source of truth for column order; a model
        trained against a different version of it would feed XGBoost mislabelled
        columns and produce silently wrong scores. Detecting the mismatch at load
        time turns that into a clear, actionable error.
        """
        if saved_names == FEATURE_NAMES:
            return
        saved = list(saved_names or [])
        missing = [n for n in FEATURE_NAMES if n not in saved]
        extra = [n for n in saved if n not in FEATURE_NAMES]
        if not missing and not extra:
            detail = "feature names match but column order differs"
        else:
            detail = f"missing from model: {missing or 'none'}; unknown to code: {extra or 'none'}"
        raise ValueError(
            "feature schema drift between the saved model and the current code "
            f"(meta.json has {len(saved)} feature(s), code expects {len(FEATURE_NAMES)}): "
            f"{detail}. Retrain the model against this version of the package."
        )
