"""Inference pipeline (application service).

Loads a trained model directory and classifies new items. Leakage-free by
construction: a genuinely new item has no self-match in the index.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..domain import CandidatePolicy, LabelSpace, Prediction, composed_feature_names
from ..infrastructure import ArtifactRepository, DeployedArtifacts
from ..infrastructure.persistence import NewClass
from .features import FeatureAssembler
from .scoring import add_confidence, top_k_per_item, top_per_item


class InferencePipeline:
    def __init__(self, artifacts: DeployedArtifacts):
        self._a = artifacts
        self._assembler = FeatureAssembler(
            artifacts.label_space, CandidatePolicy(artifacts.config.candidate_top_n)
        )
        # Custom feature providers (T70) shipped with the model, plus the composed
        # schema (core + provider columns) the fusion model was trained on. Empty /
        # core-only for a model with no custom features.
        self._providers = artifacts.feature_providers
        self._feature_names = composed_feature_names(self._providers)

    @classmethod
    def from_directory(cls, directory: str) -> "InferencePipeline":
        return cls(ArtifactRepository().load(directory))

    @property
    def label_space(self) -> LabelSpace:
        return self._a.label_space

    @property
    def config(self) -> PipelineConfig:
        return self._a.config

    @property
    def artifacts(self) -> DeployedArtifacts:
        return self._a

    def with_added_classes(self, new_classes: Sequence[NewClass]) -> "InferencePipeline":
        """Return a new pipeline whose label space is widened with ``new_classes``,
        **without retraining** (see ``DeployedArtifacts.with_added_classes``).

        Each ``new_classes`` entry is a ``ClassDefinition`` or a
        ``(key, description)`` pair. Added classes are description-only: they are
        retrievable from their description but, lacking example support, draw a low
        calibrated confidence and typically abstain under a precision-tuned
        threshold until they are seeded and the model is retrained. The original
        pipeline is left unchanged."""
        return InferencePipeline(self._a.with_added_classes(new_classes))

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
        q_emb = a.encoder.encode_queries(texts)
        feats = self._assembler.assemble(
            texts,
            q_emb,
            a.dense,
            a.lexical,
            a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
            providers=self._providers,
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
        decided = top_per_item(add_confidence(feats, a.fusion, a.calibrator, self._feature_names))
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
        q_emb = a.encoder.encode_queries(texts)
        feats = self._assembler.assemble(
            texts,
            q_emb,
            a.dense,
            a.lexical,
            a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
            providers=self._providers,
        )
        results: List[List[Tuple[str, float]]] = [[] for _ in texts]
        if not len(feats):
            return results

        ranked = top_k_per_item(
            add_confidence(feats, a.fusion, a.calibrator, self._feature_names), k
        )
        item_ids = ranked["item_id"].to_numpy(dtype=np.intp)
        candidates = ranked["candidate"].to_numpy(dtype=np.intp)
        confidences = ranked["conf"].to_numpy(dtype=np.float64)
        keys = np.asarray(a.label_space.keys)[candidates]

        for item_id, key, conf in zip(item_ids, keys, confidences):
            results[item_id].append((str(key), float(conf)))
        return results

    def explain(self, texts: Sequence[str], top_k: Optional[int] = None) -> pd.DataFrame:
        """Return the full per-(item, candidate) signal table behind ``predict``.

        One row per (input item, candidate class) that survived candidate
        selection, carrying every raw retrieval-signal feature *plus* the
        calibrated ``conf`` the fusion model assigns — the numbers ``predict``
        computes and then collapses to a single decision. This is the
        data-scientist view: *why* each candidate scored the way it did, signal by
        signal, before fusion picked a winner.

        Columns, in order: ``item_id`` (row index into ``texts``), ``text``,
        ``rank`` (1 = the item's most-confident candidate), ``candidate_key``
        (class key), ``conf`` (calibrated P(correct)), then the effective feature
        schema (the core ~28 signal columns plus any custom-provider columns).
        A ``NaN`` in a signal column means that signal did not retrieve that class
        — distinct from a true 0, exactly as the fusion model consumes it.

        ``top_k`` keeps only each item's ``k`` most-confident candidates; ``None``
        (default) returns every candidate. Reuses a single encode → assemble →
        calibrate pass, so the values match ``predict``/``predict_topk`` exactly.
        Items that retrieved no candidate contribute no rows.
        """
        texts = list(texts)
        self._validate_texts(texts)
        a = self._a
        columns = ["item_id", "text", "rank", "candidate_key", "conf", *self._feature_names]
        q_emb = a.encoder.encode_queries(texts)
        feats = self._assembler.assemble(
            texts,
            q_emb,
            a.dense,
            a.lexical,
            a.config.retrieval.k_neighbors,
            query_ids=list(range(len(texts))),
            query_labels=None,
            chunk=a.config.retrieval.feature_chunk,
            providers=self._providers,
        )
        if not len(feats):
            return pd.DataFrame(columns=columns)

        scored = add_confidence(feats, a.fusion, a.calibrator, self._feature_names)
        scored = scored.sort_values(["item_id", "conf"], ascending=[True, False]).reset_index(
            drop=True
        )
        scored["rank"] = scored.groupby("item_id", sort=False).cumcount() + 1
        if top_k is not None:
            scored = scored[scored["rank"] <= top_k].reset_index(drop=True)

        item_ids = scored["item_id"].to_numpy(dtype=np.intp)
        keys = np.asarray(a.label_space.keys)
        out = pd.DataFrame(
            {
                "item_id": item_ids,
                "text": np.asarray(texts, dtype=object)[item_ids],
                "rank": scored["rank"].to_numpy(dtype=np.int64),
                "candidate_key": keys[scored["candidate"].to_numpy(dtype=np.intp)],
                "conf": scored["conf"].to_numpy(dtype=np.float64),
            }
        )
        for name in self._feature_names:
            out[name] = scored[name].to_numpy()
        return out

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"InferencePipeline.predict expects str inputs; item at index {i} "
                    f"is {type(t).__name__}: {t!r}"
                )
