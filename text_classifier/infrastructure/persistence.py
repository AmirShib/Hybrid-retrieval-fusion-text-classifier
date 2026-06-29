"""Persistence: writes/reads a self-contained model directory. Uses only stdlib
pickle + numpy + json so there is no extra dependency and the directory is
portable to the air-gapped host.

Layout:
    <dir>/encoder/         SentenceTransformer.save() output
    <dir>/dense.npz        dense retriever numeric state
    <dir>/lexical.pkl      pickled LexicalRetrieverAdapter (vectorizers + BM25 weights)
    <dir>/fusion.json      XGBoost model
    <dir>/calibrator.pkl   calibrator (isotonic | platt | beta)
    <dir>/meta.json        label space, thresholds, config, feature schema
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .._version import __version__
from ..config import PipelineConfig
from ..domain import (
    AbstentionPolicy,
    ClassDefinition,
    ConfidenceCalibrator,
    FEATURE_NAMES,
    FusionModel,
    LabelSpace,
    TextEncoder,
)
from .registry import calibrator_spec, encoder_spec, fusion_spec
from .retrieval import DenseRetrieverAdapter, DenseState, LexicalRetrieverAdapter

log = logging.getLogger(__name__)


# Defaults for model dirs written before component kinds were recorded (T23).
_LEGACY_COMPONENTS = {
    "encoder": "sentence-transformers",
    "fusion": "xgboost",
    "calibrator": "isotonic",
}


@dataclass
class DeployedArtifacts:
    """Everything the inference pipeline needs, in memory. Component fields are
    typed against the ports, not concrete classes, so any registered backend
    fits."""
    config: PipelineConfig
    label_space: LabelSpace
    encoder: TextEncoder
    dense: DenseRetrieverAdapter
    lexical: LexicalRetrieverAdapter
    fusion: FusionModel
    calibrator: ConfidenceCalibrator
    abstention: AbstentionPolicy


class ArtifactRepository:
    """Reads/writes DeployedArtifacts to a directory."""

    def save(self, artifacts: DeployedArtifacts, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        cfg = artifacts.config
        # Component filenames/dirnames come from the registry, so a backend with
        # its own on-disk format round-trips without editing this method.
        enc_spec = encoder_spec(cfg.encoder.kind)
        fus_spec = fusion_spec(cfg.fusion.kind)
        cal_spec = calibrator_spec(cfg.calibration.kind)

        artifacts.encoder.save(os.path.join(directory, enc_spec.dirname))

        s = artifacts.dense.state
        np.savez_compressed(
            os.path.join(directory, "dense.npz"),
            example_emb=s.example_emb, example_labels=s.example_labels,
            prototypes=s.prototypes, description_emb=s.description_emb, class_freq=s.class_freq,
        )
        with open(os.path.join(directory, "lexical.pkl"), "wb") as fh:
            pickle.dump(artifacts.lexical, fh)

        artifacts.fusion.save(os.path.join(directory, fus_spec.filename))
        artifacts.calibrator.save(os.path.join(directory, cal_spec.filename))

        meta = {
            "feature_names": FEATURE_NAMES,
            "package_version": __version__,
            "config": cfg.to_dict(),
            "components": {
                "encoder": cfg.encoder.kind,
                "fusion": cfg.fusion.kind,
                "calibrator": cfg.calibration.kind,
            },
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
        self._check_package_version(meta.get("package_version"))

        config = PipelineConfig.from_dict(meta["config"])
        label_space = LabelSpace([ClassDefinition(c["key"], c["description"]) for c in meta["classes"]])

        # Dispatch each swappable component through the registry by its recorded
        # kind (defaulting for legacy dirs that predate the `components` block).
        components = self._components_from_meta(meta)
        enc_spec = encoder_spec(components["encoder"])
        fus_spec = fusion_spec(components["fusion"])
        cal_spec = calibrator_spec(components["calibrator"])

        encoder = enc_spec.load(os.path.join(directory, enc_spec.dirname), config.encoder)

        npz = np.load(os.path.join(directory, "dense.npz"))
        dense = DenseRetrieverAdapter(
            DenseState(npz["example_emb"], npz["example_labels"], npz["prototypes"],
                       npz["description_emb"], npz["class_freq"]),
            chunk=config.retrieval.dense_chunk,
        )
        with open(os.path.join(directory, "lexical.pkl"), "rb") as fh:
            lexical: LexicalRetrieverAdapter = pickle.load(fh)

        fusion = fus_spec.load(os.path.join(directory, fus_spec.filename))
        calibrator = cal_spec.load(os.path.join(directory, cal_spec.filename))

        abstention = AbstentionPolicy(
            global_threshold=float(meta["abstention"]["global_threshold"]),
            per_class={int(k): float(v) for k, v in meta["abstention"]["per_class"].items()},
        )
        return DeployedArtifacts(config, label_space, encoder, dense, lexical, fusion, calibrator, abstention)

    @staticmethod
    def _components_from_meta(meta: Dict) -> Dict[str, str]:
        """Resolve each component's ``kind`` for load dispatch.

        Prefers the explicit ``components`` block (written since T23); falls back
        to the kinds embedded in ``config``; finally to the built-in defaults so
        a model directory written before this change still loads.
        """
        comp = meta.get("components") or {}
        cfg = meta.get("config") or {}
        return {
            "encoder": comp.get("encoder")
            or cfg.get("encoder", {}).get("kind")
            or _LEGACY_COMPONENTS["encoder"],
            "fusion": comp.get("fusion")
            or cfg.get("fusion", {}).get("kind")
            or _LEGACY_COMPONENTS["fusion"],
            "calibrator": comp.get("calibrator")
            or cfg.get("calibration", {}).get("kind")
            or _LEGACY_COMPONENTS["calibrator"],
        }

    @staticmethod
    def _check_package_version(saved_version) -> None:
        """Warn (do not fail) when a model was trained on a different version.

        Compatibility of the on-disk format is governed by the feature-schema
        check, which raises on a real mismatch. The package version is recorded
        for provenance and surfaced as a soft warning so an operator can notice a
        version skew without it blocking a load that is otherwise valid.
        """
        if saved_version and saved_version != __version__:
            log.warning(
                "model was trained with text-classifier %s but the current "
                "version is %s; behavior should be unchanged (the feature schema "
                "is checked separately), but verify if results look off.",
                saved_version, __version__,
            )

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
