"""Inference pipeline (application service).

Loads a trained model directory and classifies new items. Leakage-free by
construction: a genuinely new item has no self-match in the index.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._messages import format_preview
from ..config import PipelineConfig
from ..domain import (
    CandidatePolicy,
    LabelSpace,
    Prediction,
    composed_feature_names,
    fusion_feature_names,
)
from ..infrastructure import ArtifactRepository, DeployedArtifacts
from ..infrastructure.persistence import NewClass
from .evaluation import json_safe
from .features import FeatureAssembler
from .importance import ablation_report, global_feature_importance
from .scoring import (
    add_confidence,
    rank_candidates,
    select_feature_columns,
    top_k_per_item,
    top_per_item,
)
from .signal_report import SIGNALS

# Map each signal's ``is_*_top1`` feature flag back to the human-readable signal
# name, so an explanation can say *which* signals ranked a candidate first.
_TOP1_FLAG_TO_SIGNAL: Dict[str, str] = {cols["top1"]: name for name, cols in SIGNALS.items()}


class InferencePipeline:
    def __init__(self, artifacts: DeployedArtifacts):
        self._a = artifacts
        self._assembler = FeatureAssembler(
            artifacts.label_space, CandidatePolicy(artifacts.config.candidate_top_n)
        )
        # Custom feature providers shipped with the model, plus the composed
        # schema (core + provider columns) the fusion model was trained on. Empty /
        # core-only for a model with no custom features.
        self._providers = artifacts.feature_providers
        # Two schemas, deliberately: `_feature_names` is what the fusion model was
        # fitted on (the composed schema minus `fusion.drop_features`) and drives
        # scoring + contribution alignment; `_assembled_names` is every column the
        # assembler produces. They differ only when features were dropped, and the
        # diagnostic surface (`explain`, and `signal_report` downstream of it)
        # follows the *assembled* list — a dropped column is still measured, it
        # just did not reach the model, and hiding it would break a report that
        # reads core signal columns by name.
        self._feature_names = fusion_feature_names(
            self._providers, artifacts.config.fusion.drop_features, artifacts.signal_providers
        )
        self._assembled_names = composed_feature_names(self._providers, artifacts.signal_providers)

    @classmethod
    def from_directory(cls, directory: str, device: Optional[str] = None) -> "InferencePipeline":
        """Load a trained model directory for inference.

        ``device`` (e.g. ``"cuda"``, ``"cpu"``) pins both the encoder and the
        fusion model to that device, overriding auto-detection. ``None`` (the
        default) auto-detects: GPU if one is visible on this host, else CPU.
        """
        return cls(ArtifactRepository().load(directory, device=device))

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

    def _featurize(
        self, texts: List[str], requested: Sequence[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Encode ``texts`` and assemble their (item, candidate) feature table.

        The single encode → assemble pass every public method on this class
        starts from; they differ only in ``requested``, the schema they need
        (``_feature_names`` to score, ``_assembled_names`` for the diagnostics
        that read core columns by name — see ``__init__``). Returns the query
        embeddings alongside the frame because ``explain_records`` needs them
        again for its neighbor evidence, and re-encoding to get them back would
        be the one thing this method exists to prevent.

        Callers must have validated ``texts`` first (``_validate_texts``): this
        is where encoding actually begins, and a non-string input must fail
        before it, not inside the encoder.
        """
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
            requested=requested,
            signal_providers=a.signal_providers,
        )
        return q_emb, feats

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
        _, feats = self._featurize(texts, self._feature_names)

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
        _, feats = self._featurize(texts, self._feature_names)
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
        columns = ["item_id", "text", "rank", "candidate_key", "conf", *self._assembled_names]
        _, feats = self._featurize(texts, self._assembled_names)
        if not len(feats):
            return pd.DataFrame(columns=columns)

        scored = rank_candidates(
            add_confidence(feats, a.fusion, a.calibrator, self._feature_names), top_k
        )

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
        for name in self._assembled_names:
            out[name] = scored[name].to_numpy()
        return out

    def importance_report(self, texts: Sequence[str], true_keys: Sequence[str]) -> Dict[str, Any]:
        """Feature importance + per-feature ablation against a freshly labeled set.

        Combines ``application.importance.global_feature_importance`` (mean
        additive contribution per column, aggregated from the same attribution
        ``explain_records(..., include_contributions=True)`` exposes per row) with
        ``ablation_report`` (mask each column to ``NaN`` — the domain's own
        "signal missing" encoding — and re-score with this unchanged model, to
        measure the actual accuracy/coverage cost of losing it). Neither retrains;
        both reuse a single encode -> assemble pass.

        ``true_keys`` must align 1:1 with ``texts``; every key must be in
        ``label_space.keys`` or this raises ``KeyError`` naming the unknown keys,
        matching the ``evaluate`` CLI's fail-fast validation.
        """
        texts = list(texts)
        self._validate_texts(texts)
        a = self._a
        unknown = a.label_space.unknown_keys(true_keys)
        if unknown:
            raise KeyError(f"label(s) not in model's label space: {format_preview(unknown)}")
        true_idx_by_item = np.array(a.label_space.encode_labels(true_keys), dtype=np.intp)

        _, feats = self._featurize(texts, self._assembled_names)
        if not len(feats):
            empty = {
                "n_items": 0,
                "coverage": None,
                "accuracy_on_accepted": None,
                "accuracy_if_no_abstain": None,
            }
            return {"importance": None, "ablation": {"baseline": empty, "ablations": []}}

        X = select_feature_columns(
            feats, self._feature_names, context="importance_report"
        ).to_numpy(dtype=np.float32)
        importance = global_feature_importance(a.fusion, X, self._feature_names)
        ablation = ablation_report(
            feats, a.fusion, a.calibrator, a.abstention, self._feature_names, true_idx_by_item
        )
        return {"importance": importance, "ablation": ablation}

    def explain_records(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 3,
        include_contributions: bool = False,
        n_neighbors: int = 5,
    ) -> List[Dict[str, Any]]:
        """Per-item explanation payloads for review UIs and debugging.

        One JSON-clean dict per input, assembled from a single feature pass (the
        plain ``predict`` path is untouched). Each payload carries:

        - ``text`` and ``decision`` — ``top_key``, ``confidence``, ``abstained``,
          plus ``threshold_applied`` and ``threshold_scope`` (``"per_class"`` or
          ``"global"``): reviewers keep asking "how close to the threshold was it".
        - ``candidates`` — the top-``top_k`` classes by calibrated confidence, each
          with its assembled ``features`` (``NaN`` → ``null``, so a reviewer sees
          which signals nominated it), ``signals_top1`` (which signals ranked it
          first), the matched class ``description``, and — when
          ``include_contributions`` and the backend supports it — per-feature
          ``contributions`` toward the raw margin (``contributions_space`` =
          ``"raw_margin"``, pre-calibration; a ``bias`` term completes the sum).
        - ``neighbors`` — up to ``n_neighbors`` nearest dense and lexical example
          neighbors as ``{label_key, score}``. Neighbor *texts* need a persisted
          corpus (a later capability), so ``texts_available`` is ``False`` and only
          class keys + scores are surfaced.

        An item that retrieved no candidate gets an abstaining decision with empty
        ``candidates`` (its neighbors are still reported).
        """
        texts = list(texts)
        self._validate_texts(texts)
        a = self._a
        keys = a.label_space.keys
        descriptions = a.label_space.descriptions

        q_emb, feats = self._featurize(texts, self._assembled_names)
        neighbors = self._neighbor_evidence(texts, q_emb, keys, n_neighbors)

        # Default: every item abstains with no candidates (covers items whose
        # features surfaced nothing, which are absent from the scored frame).
        records: List[Dict[str, Any]] = [
            {
                "text": texts[i],
                "decision": {
                    "top_key": "",
                    "confidence": 0.0,
                    "abstained": True,
                    "threshold_applied": None,
                    "threshold_scope": None,
                },
                "candidates": [],
                "neighbors": neighbors[i],
            }
            for i in range(len(texts))
        ]
        if len(feats):
            topk = rank_candidates(
                add_confidence(feats, a.fusion, a.calibrator, self._feature_names), top_k
            )

            contribs = None
            if include_contributions:
                X_topk = select_feature_columns(
                    topk, self._feature_names, context="explain_records (contributions)"
                )
                contribs = a.fusion.predict_contribs(X_topk.to_numpy(np.float32))

            for item_id, group in topk.groupby("item_id", sort=False):
                item_id = int(item_id)
                cand_dicts = [
                    # `idx` is the reset (0..n-1) row position, aligned to `contribs`.
                    self._candidate_dict(
                        row, keys, descriptions, None if contribs is None else contribs[idx]
                    )
                    for idx, row in group.iterrows()
                ]
                records[item_id]["candidates"] = cand_dicts
                records[item_id]["decision"] = self._decision(cand_dicts[0], a)

        return [json_safe(rec) for rec in records]

    def _candidate_dict(
        self,
        row: pd.Series,
        keys: Sequence[str],
        descriptions: Sequence[str],
        contrib_row: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """One candidate entry: its class key, confidence, per-signal feature
        values (NaN preserved — ``json_safe`` turns it into null), which signals
        ranked it first, the class description, and optional SHAP contributions."""
        cand_idx = int(row["candidate"])
        features = {name: row[name] for name in self._feature_names}
        signals_top1 = [
            signal
            for flag, signal in _TOP1_FLAG_TO_SIGNAL.items()
            if flag in row.index and float(row[flag]) == 1.0
        ]
        entry: Dict[str, Any] = {
            "key": keys[cand_idx],
            "conf": float(row["conf"]),
            "features": features,
            "signals_top1": signals_top1,
            "description": descriptions[cand_idx],
        }
        if contrib_row is not None:
            # (n_features + 1,): per-feature contributions then the trailing bias.
            entry["contributions"] = {
                name: float(contrib_row[j]) for j, name in enumerate(self._feature_names)
            }
            entry["contributions"]["bias"] = float(contrib_row[-1])
            entry["contributions_space"] = "raw_margin"
        return entry

    def _decision(self, top_candidate: Dict[str, Any], a: DeployedArtifacts) -> Dict[str, Any]:
        """The abstention decision for an item, from its best candidate — the same
        threshold logic ``predict`` applies, made transparent for a reviewer."""
        cand_idx = a.label_space.index_of(top_candidate["key"])
        conf = float(top_candidate["conf"])
        threshold = a.abstention.threshold_for(cand_idx)
        scope = "per_class" if cand_idx in a.abstention.per_class else "global"
        accepted = conf >= threshold
        return {
            "top_key": top_candidate["key"],
            "confidence": conf,
            "abstained": not accepted,
            "threshold_applied": float(threshold),
            "threshold_scope": scope,
        }

    def _neighbor_evidence(
        self,
        texts: Sequence[str],
        q_emb: np.ndarray,
        keys: Sequence[str],
        n_neighbors: int,
    ) -> List[Dict[str, Any]]:
        """Per-item nearest dense + lexical example neighbors as ``{label_key,
        score}``. Padding (label ``< 0`` / NaN score) is dropped. Neighbor texts
        require a persisted corpus (a later capability), so ``texts_available`` is
        ``False`` here — only class keys and scores are available.

        ``a.lexical`` is ``None`` for a model trained with "lexical" excluded
        from ``config.signals`` (no BM25 index was ever built); every item's
        ``"lexical"`` neighbor list is then empty rather than an error."""
        a = self._a
        k = a.config.retrieval.k_neighbors
        d_lab, d_sim = a.dense.knn_example_labels(q_emb, k)
        if a.lexical is not None:
            b_lab, b_sco = a.lexical.knn_example_labels(list(texts), k)
        else:
            b_lab = b_sco = None

        def rows(labels: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for lab, sc in zip(labels.tolist(), scores.tolist()):
                if lab is None or lab < 0 or sc is None or (isinstance(sc, float) and np.isnan(sc)):
                    continue
                out.append({"label_key": keys[int(lab)], "score": float(sc)})
                if len(out) >= n_neighbors:
                    break
            return out

        return [
            {
                "dense": rows(d_lab[i], d_sim[i]),
                "lexical": [] if b_lab is None else rows(b_lab[i], b_sco[i]),
                "texts_available": False,
            }
            for i in range(len(texts))
        ]

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"InferencePipeline.predict expects str inputs; item at index {i} "
                    f"is {type(t).__name__}: {t!r}"
                )
