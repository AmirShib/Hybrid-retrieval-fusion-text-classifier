"""Shared pytest fixtures for the text-classifier test suite.

Determinism contract
--------------------
Unit tests (tests/unit/) assert exact values where the math is deterministic:
array shapes, specific float values from a seeded RNG, exact key lookups.
Integration tests (tests/integration/) assert bounds and invariants only —
never exact floats — because pipeline outputs depend on XGBoost internals
that may vary across platforms and library versions.

PYTHONHASHSEED is pinned to "0" in CI (.github/workflows/ci.yml) so that
builtin hash() is stable within a run.  T21 will replace hash() with
hashlib.sha256, removing the PYTHONHASHSEED dependency entirely.
"""
from __future__ import annotations

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
    return LabelSpace([
        ClassDefinition(key="alpha", description="the alpha class"),
        ClassDefinition(key="beta", description="the beta class"),
        ClassDefinition(key="gamma", description="the gamma class"),
    ])
