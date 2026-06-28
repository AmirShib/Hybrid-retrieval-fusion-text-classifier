"""T21 — Determinism of the HashingEncoder test double.

HashingEncoder is the offline stand-in for a real bi-encoder used throughout
the suite. It must be byte-for-byte reproducible so that seeded tests assert
exact values; switching from the builtin ``hash()`` to ``hashlib.sha256`` makes
that reproducibility independent of ``PYTHONHASHSEED``, Python version, and
platform. These tests pin that contract.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from tests._doubles import HashingEncoder


def test_known_value_regression():
    """Golden value: ``hello world`` in an 8-dim space maps to two opposite buckets.

    'hello' -> bucket 4 (+1), 'world' -> bucket 0 (-1); after L2 normalization
    each non-zero entry is ±1/sqrt(2). Pinning this catches any accidental change
    to the hashing scheme.
    """
    vec = HashingEncoder(dim=8).encode(["hello world"])[0]
    expected = np.array(
        [-0.70710677, 0.0, 0.0, 0.0, 0.70710677, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vec, expected, rtol=0, atol=1e-6)


def test_l2_normalization_preserved():
    """Every non-empty row has unit norm (the cosine==dot invariant)."""
    out = HashingEncoder(dim=64).encode(["alpha beta gamma", "single", "a b c d e"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-6)


def test_empty_text_yields_zero_vector():
    """A token-less string has no buckets to fill; norm clipping keeps it finite."""
    out = HashingEncoder(dim=16).encode([""])
    assert out.shape == (1, 16)
    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


def test_repeated_token_accumulates():
    """The same token twice doubles its bucket before normalization."""
    enc = HashingEncoder(dim=8)
    one = enc.encode(["hello"])[0]
    two = enc.encode(["hello hello"])[0]
    # Direction is identical (both unit vectors pointing the same way).
    np.testing.assert_allclose(one, two, atol=1e-6)


def test_cross_process_stability(tmp_path):
    """A fresh interpreter with a *randomized* PYTHONHASHSEED produces the same
    embedding as this process — proving the encoder no longer depends on it."""
    script = (
        "import numpy as np;"
        "from tests._doubles import HashingEncoder;"
        "v = HashingEncoder(dim=32).encode(['quick brown fox'])[0];"
        "print(';'.join(repr(float(x)) for x in v))"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "12345"  # deliberately not 0
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )
    assert proc.returncode == 0, proc.stderr
    child = np.array([float(x) for x in proc.stdout.strip().split(";")], dtype=np.float32)
    parent = HashingEncoder(dim=32).encode(["quick brown fox"])[0]
    np.testing.assert_allclose(child, parent, atol=1e-6)
