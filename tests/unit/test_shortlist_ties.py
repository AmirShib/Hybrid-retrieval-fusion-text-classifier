"""Shortlist tie cap and the sparse prototype computation."""

from __future__ import annotations

import json
import os

import numpy as np

from tests._doubles import make_synthetic
from text_classifier.application.training import TrainingPipeline
from text_classifier.config import PipelineConfig, TrainingConfig
from text_classifier.infrastructure import ArtifactRepository
from text_classifier.infrastructure.array_ops import NumpyArrayOps
from text_classifier.infrastructure.retrieval import _prototypes_and_freq
from text_classifier.infrastructure.signals import _topn_mask


def test_small_ties_are_kept_large_ties_are_cut_to_exactly_n():
    M = np.array(
        [
            [0.9, 0.5, 0.5, 0.1, 0.0, 0.0],  # tie at the cutoff: 3 admitted for n=2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # every class tied: would admit all 6
        ]
    )
    uncapped = _topn_mask(M, 2)
    assert uncapped.sum(axis=1).tolist() == [3, 6]
    capped = _topn_mask(M, 2, tie_cap=1.5)  # limit = 3 classes
    assert capped.sum(axis=1).tolist() == [3, 2]
    assert capped[1].tolist() == [True, True, False, False, False, False]  # class order


def test_tie_cap_never_admits_missing_values():
    M = np.array([[np.nan, 0.0, 0.0, 0.0, 0.0]])
    assert _topn_mask(M, 1, tie_cap=1.0).tolist() == [[False, True, False, False, False]]


def test_models_saved_without_the_setting_keep_uncapped_shortlists(tmp_path):
    cfg = PipelineConfig()
    cfg.encoder.kind = "hashing"
    cfg.training = TrainingConfig(
        n_folds=3, random_state=0, target_precision=0.5, per_class_min_support=1
    )
    ls, items = make_synthetic(n_classes=4, per_class=12, seed=2)
    art, _ = TrainingPipeline(cfg).run(items, ls)
    d = str(tmp_path / "m")
    ArtifactRepository().save(art, d)
    assert ArtifactRepository().load(d).config.candidate_tie_cap == 2.0
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path))
    del meta["config"]["candidate_tie_cap"]  # what a pre-cap model dir looks like
    json.dump(meta, open(meta_path, "w"))
    assert ArtifactRepository().load(d).config.candidate_tie_cap is None


def test_sparse_prototypes_match_the_scatter_path():
    class ScatterOps(NumpyArrayOps):
        name = "scatter-reference"  # any non-numpy name takes the scatter branch

    rng = np.random.default_rng(0)
    emb = rng.random((300, 40)).astype(np.float32)
    labels = rng.integers(0, 6, 300)
    labels[labels == 3] = 4  # class 3 has no examples -> NaN prototype
    new_p, new_f = _prototypes_and_freq(emb, labels, 6)
    old_p, old_f = _prototypes_and_freq(emb, labels, 6, ScatterOps())
    np.testing.assert_allclose(new_p, old_p, atol=1e-7, equal_nan=True)
    assert np.isnan(new_p[3]).all() and np.array_equal(new_f, old_f)
