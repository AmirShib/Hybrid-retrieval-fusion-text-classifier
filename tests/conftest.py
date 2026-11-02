"""Shared pytest fixtures for the text-classifier test suite.

Determinism contract
--------------------
Unit tests (tests/unit/) assert exact values where the math is deterministic:
array shapes, specific float values from a seeded RNG, exact key lookups.
Integration tests (tests/integration/) assert bounds and invariants only —
never exact floats — because pipeline outputs depend on XGBoost internals
that may vary across platforms and library versions.

Determinism does not depend on PYTHONHASHSEED: HashingEncoder hashes tokens
with hashlib.sha256 (T21), so embeddings are identical across processes, Python
versions, and platforms without any env-var pinning.
"""

from __future__ import annotations

# T85: import xgboost before anything else in the session gets a chance to
# import torch. On at least one dev host (macOS + Homebrew libomp), xgboost
# and torch each bundle their own OpenMP runtime, and initializing torch's
# *first* -- which pytest's single-process collection makes easy to do by
# accident, since the T85 torch-backend tests import torch at module level --
# reliably segfaults xgboost's first `fit()` afterward with
# `OMP: Error #179: Function pthread_mutex_init failed`. Importing xgboost
# here, before pytest collects any test module, sidesteps it: verified this
# ordering (xgboost first) is sufficient on the affected host, independent of
# which library is actually *used* first. A torch-free run is unaffected --
# xgboost is a hard dependency already, so this import always succeeds.
import xgboost  # noqa: F401

import pytest

from text_classifier import ClassDefinition, LabelSpace

from tests._doubles import HashingEncoder, make_synthetic


@pytest.fixture
def hashing_encoder() -> HashingEncoder:
    """Offline TextEncoder double; no network or torch required."""
    return HashingEncoder(dim=128)


@pytest.fixture
def synthetic_dataset() -> tuple:
    """Return (LabelSpace, list[LabeledItem]) — 40 classes, imbalanced, seeded."""
    return make_synthetic(n_classes=40, per_class=60, seed=0)


@pytest.fixture
def tiny_label_space() -> LabelSpace:
    """Hand-built 3-class LabelSpace for exact-value assertions."""
    return LabelSpace(
        [
            ClassDefinition(key="alpha", description="the alpha class"),
            ClassDefinition(key="beta", description="the beta class"),
            ClassDefinition(key="gamma", description="the gamma class"),
        ]
    )
