"""T34 phase 2 — golden-frame regression test.

Captures the *actual values* ``FeatureAssembler.assemble`` produced on a fixed
synthetic corpus under the pre-refactor code (see
``tests/unit/t34_phase2_golden*.npz``, generated once from the unmodified
``application/features.py`` / ``infrastructure/retrieval.py`` and checked in).
The Phase 2 signal-provider refactor must reproduce these values exactly —
same columns, same order, same floats, `rtol=0`/`atol=0` — for the default
``dense``/``lexical`` provider pair. This is the safety net the T34 ticket's
"hard requirement" asks for: a refactor with a golden-output guarantee, not a
behaviour change.

Three fixtures cover the paths that matter:
  - ``t34_phase2_golden.npz``          -- ordinary assemble(), with labels
  - ``t34_phase2_golden_nolabels.npz`` -- ordinary assemble(), no labels
  - ``t34_phase2_golden_requested.npz``-- T87 ``requested=`` narrowing
"""

from __future__ import annotations

import os

import numpy as np
import numpy.testing as npt
import pytest

from text_classifier.application.features import FeatureAssembler
from text_classifier.config import RetrievalConfig
from text_classifier.domain import CandidatePolicy, fusion_feature_names
from text_classifier.infrastructure.retrieval import DenseRetrieverAdapter, LexicalRetrieverAdapter

FIXTURES = os.path.dirname(__file__)


@pytest.fixture
def env(hashing_encoder):
    from tests._doubles import make_synthetic

    label_space, items = make_synthetic(n_classes=5, per_class=10, seed=7)
    texts = [it.text for it in items]
    label_idx = np.array(label_space.encode_labels([it.label for it in items]))
    cfg = RetrievalConfig()
    dense = DenseRetrieverAdapter.build(hashing_encoder, texts, label_idx, label_space, cfg)
    lexical = LexicalRetrieverAdapter.build(texts, label_idx, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=3))
    q_items = items[:8]
    q_texts = [it.text for it in q_items]
    q_emb = hashing_encoder.encode(q_texts)
    q_labels = np.array(label_space.encode_labels([it.label for it in q_items]))
    return dict(
        assembler=assembler,
        dense=dense,
        lexical=lexical,
        q_texts=q_texts,
        q_emb=q_emb,
        q_labels=q_labels,
    )


def _assert_matches_golden(df, fixture_name):
    golden = dict(np.load(os.path.join(FIXTURES, fixture_name)))
    assert list(df.columns) == list(golden.keys()), (
        f"column set/order changed vs. {fixture_name}: "
        f"got {list(df.columns)}, expected {list(golden.keys())}"
    )
    for col in golden:
        got = df[col].to_numpy()
        want = golden[col]
        npt.assert_array_equal(got, want, err_msg=f"column {col!r} differs from golden fixture")


def test_assemble_matches_golden_with_labels(env):
    e = env
    df = e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        query_labels=e["q_labels"],
        chunk=4096,
    )
    _assert_matches_golden(df, "t34_phase2_golden.npz")


def test_assemble_matches_golden_without_labels(env):
    e = env
    df = e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        query_labels=None,
        chunk=4096,
    )
    _assert_matches_golden(df, "t34_phase2_golden_nolabels.npz")


def test_assemble_matches_golden_with_requested_narrowing(env):
    e = env
    req = fusion_feature_names()[:10]
    df = e["assembler"].assemble(
        e["q_texts"],
        e["q_emb"],
        e["dense"],
        e["lexical"],
        k_neighbors=3,
        query_ids=list(range(len(e["q_texts"]))),
        query_labels=e["q_labels"],
        chunk=4096,
        requested=req,
    )
    _assert_matches_golden(df, "t34_phase2_golden_requested.npz")
