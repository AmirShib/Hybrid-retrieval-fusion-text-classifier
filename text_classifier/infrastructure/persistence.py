"""Persistence: writes/reads a self-contained model directory. Uses only numpy +
json + native model formats (no pickle) so a directory loaded on the air-gapped
host is inert data, never attacker-controlled code. Directories written before
this change fall back to a legacy pickle loader (with a warning) for the one or
two files that used to be pickled.

Layout:
    <dir>/encoder/         SentenceTransformer.save() output (+ encoder_training.json:
                           the per-epoch table when a fine-tune selected its best epoch)
    <dir>/dense.npz        dense retriever numeric state
    <dir>/lexical.npz      BM25 weight matrices + example labels (see lexical.json)
    <dir>/lexical.json     BM25 vocab/analyzer config + scalars, pairs with lexical.npz
    <dir>/fusion.json      XGBoost model
    <dir>/calibrator.npz   isotonic calibrator breakpoints (kind == "isotonic")
    <dir>/calibrator.json  parametric calibrator coefficients (kind in platt|beta)
    <dir>/meta.json        label space, thresholds, config, feature schema
    <dir>/corpus.jsonl.gz  optional: raw training corpus (text+label), see TrainingConfig.store_corpus
"""

from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .._version import __version__
from ..config import FeatureProviderConfig, PipelineConfig
from ..domain import (
    AbstentionPolicy,
    ClassDefinition,
    ConfidenceCalibrator,
    FeatureProvider,
    FusionModel,
    LabeledItem,
    LabelSpace,
    TextEncoder,
    fusion_feature_names,
)
from .registry import (
    calibrator_spec,
    dense_retriever_spec,
    encoder_spec,
    feature_provider_spec,
    fusion_spec,
    lexical_retriever_spec,
)
from .retrieval import DenseRetrieverAdapter, LexicalRetrieverAdapter

log = logging.getLogger(__name__)


# Defaults for model dirs written before component kinds were recorded.
_LEGACY_COMPONENTS = {
    "encoder": "sentence-transformers",
    "fusion": "xgboost",
    "calibrator": "isotonic",
    "dense": "exact",
    "lexical": "bm25",
}


# A new class may be given as a ClassDefinition or a plain (key, description) pair.
NewClass = Union[ClassDefinition, Sequence[str]]


def _lexical_json_path(npz_filename: str) -> str:
    """The JSON sidecar path that pairs with a lexical retriever's ``.npz``
    filename (e.g. ``lexical.npz`` -> ``lexical.json``) -- the built-in BM25
    backend manages two files, so its ``LexicalRetrieverSpec.filename`` names
    only the array half and this derives the other, deterministically, rather
    than the spec needing a second field only one backend uses today."""
    root, _ = os.path.splitext(npz_filename)
    return root + ".json"


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
    # Custom fusion-feature providers, fitted on all training data. Empty
    # for a model with no custom features — the byte-for-byte-identical default.
    # A trailing field with a default keeps every positional construction valid.
    feature_providers: List[FeatureProvider] = field(default_factory=list)

    def with_added_classes(self, new_classes: Sequence[NewClass]) -> "DeployedArtifacts":
        """Widen this model's label space with new classes, **without retraining**.

        Returns a new ``DeployedArtifacts`` whose label space is extended with
        ``new_classes`` (``ClassDefinition``s or ``(key, description)`` pairs).
        The fusion model, calibrator, and abstention policy are reused verbatim —
        every feature is a per-candidate retrieval signal, so adding classes does
        not change their input or output shape. Only the class-indexed retrieval
        state grows: the dense description embeddings (encoded with this model's
        frozen encoder) and the description BM25 (refit — its IDF is corpus-global).

        New classes are appended after the existing ones, so every existing class
        index is preserved and the trained model applies unchanged.

        Each added class is **description-only**: it has no training examples, so
        it gets a ``NaN`` prototype, no kNN support, and ``class_freq = 0``. It is
        retrievable and can win a query on description similarity alone, but the
        fusion model — trained when every candidate had example support — assigns
        it a *low calibrated confidence*, so under a precision-tuned abstention
        threshold it will typically route to human review rather than auto-accept.
        That is the honest signal for a class with no example evidence; seed
        examples and retrain (the encoder is frozen, so retraining is cheap) once
        ``>= n_folds`` examples exist to lift it to full confidence.

        Raises ``ValueError`` if ``new_classes`` is empty, if any new key already
        exists in the label space, or if the new keys collide with each other.
        """
        defs = [
            c if isinstance(c, ClassDefinition) else ClassDefinition(c[0], c[1])
            for c in new_classes
        ]
        if not defs:
            raise ValueError("with_added_classes requires at least one new class")

        existing = set(self.label_space.keys)
        collisions = sorted({d.key for d in defs if d.key in existing})
        if collisions:
            raise ValueError(
                f"cannot add class(es) already in the label space: {collisions}. "
                "To change an existing class's description, retrain the model."
            )

        current = [
            ClassDefinition(k, d)
            for k, d in zip(self.label_space.keys, self.label_space.descriptions)
        ]
        # LabelSpace raises on new-vs-new duplicate keys; existing collisions are
        # already reported above with a clearer, more actionable message.
        extended_space = LabelSpace(current + defs)

        dense = self.dense.with_added_classes(self.encoder, [d.description for d in defs])
        lexical = self.lexical.with_added_descriptions(extended_space.descriptions)
        return replace(self, label_space=extended_space, dense=dense, lexical=lexical)


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
        dense_spec = dense_retriever_spec(cfg.retrieval.dense_kind)
        lex_spec = lexical_retriever_spec(cfg.retrieval.lexical_kind)

        artifacts.encoder.save(os.path.join(directory, enc_spec.dirname))

        dense_arrays = artifacts.dense.to_state()
        np.savez_compressed(os.path.join(directory, dense_spec.filename), **dense_arrays)

        arrays, lex_meta = artifacts.lexical.to_state()
        np.savez_compressed(os.path.join(directory, lex_spec.filename), **arrays)
        with open(os.path.join(directory, _lexical_json_path(lex_spec.filename)), "w") as fh:
            json.dump(lex_meta, fh)

        artifacts.fusion.save(os.path.join(directory, fus_spec.filename))
        artifacts.calibrator.save(os.path.join(directory, cal_spec.filename))

        # Custom feature providers: each persists to its own subdirectory,
        # indexed so two providers of the same kind can't collide. The manifest
        # records kind + relative path + declared names so load rebuilds them in
        # order; the composed feature-name list below is the authoritative schema.
        provider_manifest = self._save_providers(directory, cfg, artifacts.feature_providers)
        feature_names = fusion_feature_names(artifacts.feature_providers, cfg.fusion.drop_features)

        meta = {
            "feature_names": feature_names,
            "package_version": __version__,
            "config": cfg.to_dict(),
            "components": {
                "encoder": cfg.encoder.kind,
                "fusion": cfg.fusion.kind,
                "calibrator": cfg.calibration.kind,
                "dense": cfg.retrieval.dense_kind,
                "lexical": cfg.retrieval.lexical_kind,
            },
            "feature_providers": provider_manifest,
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

    def update_decision_layer(
        self,
        directory: str,
        calibrator: ConfidenceCalibrator,
        abstention: AbstentionPolicy,
        target_precision: float,
        n_items: int,
    ) -> None:
        """Persist a re-tuned calibrator + abstention policy into an existing
        model directory, touching only the decision layer.

        Unlike ``save`` (a full rewrite), this writes just the calibrator file
        (via its recorded registry kind — retuning never changes the calibrator
        *kind*, only refits it) and updates ``meta.json``'s ``abstention`` block
        and ``config.training.target_precision``, appending a ``retunes``
        provenance entry. The encoder, dense/lexical indices, fusion model, and
        feature-provider files are left byte-identical.
        """
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)

        cal_kind = self._components_from_meta(meta)["calibrator"]
        cal_spec = calibrator_spec(cal_kind)
        calibrator.save(os.path.join(directory, cal_spec.filename))

        meta.setdefault("config", {}).setdefault("training", {})["target_precision"] = (
            target_precision
        )
        meta["abstention"] = {
            "global_threshold": abstention.global_threshold,
            "per_class": {str(k): v for k, v in abstention.per_class.items()},
        }
        retunes = meta.setdefault("retunes", [])
        retunes.append(
            {
                "retuned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "n_items": int(n_items),
                "target_precision": target_precision,
                "package_version": __version__,
            }
        )

        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    @staticmethod
    def save_corpus(directory: str, items: Sequence[LabeledItem]) -> None:
        """Persist the raw training corpus as gzip-compressed JSONL (one
        ``{"text": ..., "label": ...}`` object per line) so a later
        ``update`` can add labeled examples without needing
        ``--base-items``: appending examples to BM25 requires refitting on the
        *full* corpus (its IDF is corpus-global), and the model dir otherwise
        keeps no raw text at all (``dense.npz`` is embeddings, ``lexical.npz``/
        ``.json`` a fitted vectorizer's numeric state). Opt out via
        ``TrainingConfig.store_corpus=False``."""
        path = os.path.join(directory, "corpus.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps({"text": it.text, "label": it.label}) + "\n")

    @staticmethod
    def load_corpus(directory: str) -> Optional[List[LabeledItem]]:
        """Load the persisted training corpus, or ``None`` if this model dir
        predates ``--store-corpus`` or opted out of it (``update`` then needs
        ``--base-items`` to supply the original items instead)."""
        path = os.path.join(directory, "corpus.jsonl.gz")
        if not os.path.isfile(path):
            return None
        items: List[LabeledItem] = []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                items.append(LabeledItem(row["text"], row["label"]))
        return items

    @staticmethod
    def read_meta(directory: str) -> Dict:
        """Read a model dir's raw ``meta.json`` dict — used by ``update``
        to carry forward provenance (``updates``/``retunes``) that a fresh
        ``save()`` would otherwise drop, read *before* that rewrite happens
        (which matters for ``--in-place``, where source and target are the
        same directory)."""
        with open(os.path.join(directory, "meta.json")) as fh:
            return json.load(fh)

    @staticmethod
    def apply_update_provenance(target_dir: str, prior_meta: Dict, entry: Dict) -> None:
        """After ``save()`` rewrites ``target_dir``'s ``meta.json`` from
        scratch (part of ``update``), carry forward ``prior_meta``'s
        ``updates``/``retunes`` provenance history — dropped by the fresh
        rewrite, since ``save()`` builds ``meta.json`` from nothing — and
        append ``entry`` to ``updates``. Call with ``prior_meta`` read via
        ``read_meta`` *before* ``save()`` runs."""
        target_meta_path = os.path.join(target_dir, "meta.json")
        with open(target_meta_path) as fh:
            meta = json.load(fh)
        meta["updates"] = list(prior_meta.get("updates", [])) + [entry]
        if prior_meta.get("retunes"):
            meta["retunes"] = list(prior_meta["retunes"]) + list(meta.get("retunes", []))
        with open(target_meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    @staticmethod
    def _load_lexical(directory: str, kind: str = "bm25"):
        """Load the lexical retriever named by ``kind`` (the registry key
        recorded in ``meta.json``'s ``components`` block), falling back to a
        legacy ``lexical.pkl`` (with a warning) for a model directory saved
        before the npz+json format existed. The legacy fallback only applies to
        the built-in ``"bm25"`` kind -- a directory that predates ``kind`` being
        recorded always defaults to ``"bm25"`` (see ``_components_from_meta``),
        so this is exactly the pre-T34 lookup path, unchanged."""
        spec = lexical_retriever_spec(kind)
        npz_path = os.path.join(directory, spec.filename)
        json_path = os.path.join(directory, _lexical_json_path(spec.filename))
        if os.path.isfile(npz_path) and os.path.isfile(json_path):
            return spec.load(directory)
        pkl_path = os.path.join(directory, "lexical.pkl")
        if os.path.isfile(pkl_path):
            log.warning(
                "loading legacy pickle artifact %r; re-save this model directory "
                "to upgrade to the pickle-free format",
                pkl_path,
            )
            with open(pkl_path, "rb") as fh:
                return pickle.load(fh)
        raise FileNotFoundError(
            f"no lexical index found in {directory!r} "
            f"(expected {spec.filename}+{_lexical_json_path(spec.filename)}, or legacy lexical.pkl)"
        )

    @staticmethod
    def _save_providers(
        directory: str, cfg: PipelineConfig, providers: Sequence[FeatureProvider]
    ) -> List[Dict]:
        """Persist each feature provider to ``features/NN_<kind>/`` and return the
        manifest (kind + relative path + declared names, in order). Empty in, empty
        out — a model with no providers writes no ``features/`` directory, so its
        on-disk layout is byte-for-byte the one with no custom features."""
        if not providers:
            return []
        provider_cfgs = cfg.features.providers
        if len(provider_cfgs) != len(providers):
            raise ValueError(
                f"feature-provider mismatch: config lists {len(provider_cfgs)} provider(s) "
                f"but {len(providers)} were fitted; they must correspond one-to-one"
            )
        os.makedirs(os.path.join(directory, "features"), exist_ok=True)
        manifest: List[Dict] = []
        for i, (pc, provider) in enumerate(zip(provider_cfgs, providers)):
            rel = os.path.join("features", f"{i:02d}_{pc.kind}")
            provider.save(os.path.join(directory, rel))
            manifest.append({"kind": pc.kind, "path": rel, "names": provider.names()})
        return manifest

    @staticmethod
    def _load_providers(
        directory: str, meta: Dict, config: PipelineConfig
    ) -> List[FeatureProvider]:
        """Rebuild the feature providers from the manifest, in order, dispatching
        each through the registry by its recorded kind. Returns ``[]`` for a model
        dir with no ``feature_providers`` block (any model saved with no custom
        feature providers configured)."""
        manifest = meta.get("feature_providers") or []
        provider_cfgs = config.features.providers
        providers: List[FeatureProvider] = []
        for i, entry in enumerate(manifest):
            spec = feature_provider_spec(entry["kind"])
            # Pass the matching config entry when present so a provider's load can
            # honour its params; fall back to a bare config for the entry's kind.
            pc = (
                provider_cfgs[i] if i < len(provider_cfgs) else FeatureProviderConfig(entry["kind"])
            )
            providers.append(spec.load(os.path.join(directory, entry["path"]), pc))
        return providers

    def load(self, directory: str, device: Optional[str] = None) -> DeployedArtifacts:
        """Load a trained model directory.

        ``device`` (e.g. ``"cuda"``, ``"cpu"``) overrides auto-detection for both
        the encoder and the fusion model, on top of whatever the model was
        *trained* on -- useful to pin inference to a device explicitly rather
        than letting each component probe ``torch.cuda.is_available()`` for
        itself. ``None`` (the default) keeps auto-detection: the persisted
        ``encoder.device`` (usually unset) and per-call fusion auto-detection.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"model directory not found: {directory!r}")
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"meta.json not found in model directory {directory!r} (expected at {meta_path!r})"
            )
        with open(meta_path) as fh:
            meta = json.load(fh)

        self._check_package_version(meta.get("package_version"))

        config = PipelineConfig.from_dict(meta["config"])
        if device is not None:
            config.encoder.device = device
        # Rebuild feature providers before the schema check: the effective schema
        # is core + provider columns, so the providers must exist to compute it.
        feature_providers = self._load_providers(directory, meta, config)
        self._check_feature_schema(
            meta.get("feature_names"),
            fusion_feature_names(feature_providers, config.fusion.drop_features),
        )

        label_space = LabelSpace(
            [ClassDefinition(c["key"], c["description"]) for c in meta["classes"]]
        )

        # Dispatch each swappable component through the registry by its recorded
        # kind (defaulting for legacy dirs that predate the `components` block).
        components = self._components_from_meta(meta)
        enc_spec = encoder_spec(components["encoder"])
        fus_spec = fusion_spec(components["fusion"])
        cal_spec = calibrator_spec(components["calibrator"])
        dense_spec = dense_retriever_spec(components["dense"])

        encoder = enc_spec.load(os.path.join(directory, enc_spec.dirname), config.encoder)

        dense = dense_spec.load(directory, config.retrieval)
        lexical = self._load_lexical(directory, components["lexical"])

        fusion = fus_spec.load(os.path.join(directory, fus_spec.filename))
        fusion.set_device(device)
        calibrator = cal_spec.load(os.path.join(directory, cal_spec.filename))

        abstention = AbstentionPolicy(
            global_threshold=float(meta["abstention"]["global_threshold"]),
            per_class={int(k): float(v) for k, v in meta["abstention"]["per_class"].items()},
        )
        return DeployedArtifacts(
            config,
            label_space,
            encoder,
            dense,
            lexical,
            fusion,
            calibrator,
            abstention,
            feature_providers=feature_providers,
        )

    @staticmethod
    def _components_from_meta(meta: Dict) -> Dict[str, str]:
        """Resolve each component's ``kind`` for load dispatch.

        Prefers the explicit ``components`` block; falls back
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
            "dense": comp.get("dense")
            or cfg.get("retrieval", {}).get("dense_kind")
            or _LEGACY_COMPONENTS["dense"],
            "lexical": comp.get("lexical")
            or cfg.get("retrieval", {}).get("lexical_kind")
            or _LEGACY_COMPONENTS["lexical"],
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
                saved_version,
                __version__,
            )

    @staticmethod
    def _check_feature_schema(saved_names, expected_names) -> None:
        """Guard against schema drift between a persisted model and the running code.

        The *effective* schema — the core columns plus every active provider's
        columns, in order — is the single source of truth for column order; a model
        trained against a different version of it would feed XGBoost mislabelled
        columns and produce silently wrong scores. ``expected_names`` is that schema
        recomputed from the code + the model's rebuilt providers; ``saved_names`` is
        what was persisted. Detecting the mismatch at load time turns a silent
        wrong-answer bug into a clear, actionable error.
        """
        if saved_names == expected_names:
            return
        saved = list(saved_names or [])
        missing = [n for n in expected_names if n not in saved]
        extra = [n for n in saved if n not in expected_names]
        if not missing and not extra:
            detail = "feature names match but column order differs"
        else:
            detail = f"missing from model: {missing or 'none'}; unknown to code: {extra or 'none'}"
        raise ValueError(
            "feature schema drift between the saved model and the current code "
            f"(meta.json has {len(saved)} feature(s), code expects {len(expected_names)}): "
            f"{detail}. Retrain the model against this version of the package."
        )
