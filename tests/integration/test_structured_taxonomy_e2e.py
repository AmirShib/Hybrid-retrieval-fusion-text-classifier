"""End-to-end: a structured taxonomy survives train → save → load → add-classes.

The unit tests pin the serialization shape; this pins the *wiring* — that the
structured fields actually reach ``meta.json`` through a real training run, come
back intact, and are not silently stripped by the paths that rebuild a label
space from keys and descriptions.
"""

from __future__ import annotations

import json
import os

import pytest

from text_classifier.application.training import TrainingPipeline
from text_classifier.config import (
    FusionConfig,
    PipelineConfig,
    RetrievalConfig,
    TrainingConfig,
)
from text_classifier.domain import ClassDefinition, LabelSpace
from text_classifier.infrastructure.persistence import ArtifactRepository
from tests._doubles import HashingEncoder, make_synthetic


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3,
        random_state=0,
        use_per_fold_encoder=False,
        target_precision=0.5,
        per_class_min_support=1,
    )
    cfg.fusion = FusionConfig(
        xgb_params={"n_estimators": 20, "max_depth": 3, "random_state": 0, "n_jobs": 1}
    )
    cfg.retrieval = RetrievalConfig(k_neighbors=10)
    return cfg


def _enrich(space: LabelSpace) -> LabelSpace:
    """Attach structured fields to a synthetic label space, leaving `description`
    (and therefore every retrieval signal) exactly as it was."""
    return LabelSpace(
        [
            ClassDefinition(
                d.key,
                d.description,
                title=f"title for {d.key}",
                definition=f"definition of {d.key}",
                examples=(f"{d.key} example one", f"{d.key} example two"),
                exclusions=(f"not {d.key}",),
                parent_path=("root", "branch"),
            )
            for d in space.definitions
        ]
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    enc = HashingEncoder(dim=64)
    space, items = make_synthetic(n_classes=6, per_class=15, seed=3)
    space = _enrich(space)
    artifacts, _ = TrainingPipeline(_cfg(), shared_encoder=enc).run(items, space)
    d = str(tmp_path_factory.mktemp("structured_model"))
    ArtifactRepository().save(artifacts, d)
    return d, space


def test_meta_json_carries_the_structured_fields(saved):
    directory, _ = saved
    with open(os.path.join(directory, "meta.json")) as fh:
        meta = json.load(fh)
    entry = meta["classes"][0]
    assert entry["title"].startswith("title for ")
    assert entry["examples"] == [
        f"{entry['key']} example one",
        f"{entry['key']} example two",
    ]
    assert entry["parent_path"] == ["root", "branch"]
    # Absent fields stay absent rather than serializing as empty.
    assert "sibling_distinctions" not in entry


def test_round_trip_preserves_every_definition(saved):
    directory, original = saved
    loaded = ArtifactRepository().load(directory).label_space
    assert loaded.definitions == original.definitions


def test_descriptions_are_unchanged_by_the_structured_fields(saved):
    directory, original = saved
    loaded = ArtifactRepository().load(directory).label_space
    assert loaded.descriptions == original.descriptions


def test_adding_a_class_preserves_incumbent_structured_fields(saved):
    """`with_added_classes` used to rebuild incumbents from keys+descriptions,
    which would strip their structured fields."""
    directory, original = saved
    artifacts = ArtifactRepository().load(directory)
    extended = artifacts.with_added_classes([ClassDefinition("brand_new", "a new class")])

    assert extended.label_space.definitions[: original.size] == original.definitions
    added = extended.label_space.definition_at(original.size)
    assert added.key == "brand_new"
    assert not added.has_structured_fields
