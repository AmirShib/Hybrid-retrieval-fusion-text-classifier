"""T85 -- TorchArrayOps: parity with NumpyArrayOps on every ArrayOps method,
on CPU (GPU-free, so this runs in ordinary CI). This is the offline half of
the ticket's "tolerance-based parity" testing strategy -- CPU is the
reference implementation everywhere else in this repo, and torch-CPU here is
what proves the port abstraction itself is correct without needing a GPU.

Skipped entirely when torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from text_classifier.infrastructure.array_ops import NumpyArrayOps  # noqa: E402
from text_classifier.infrastructure.array_ops_torch import TorchArrayOps  # noqa: E402
from text_classifier.infrastructure.registry import build_array_ops  # noqa: E402


@pytest.fixture
def tops() -> TorchArrayOps:
    return TorchArrayOps(device="cpu")


@pytest.fixture
def nops() -> NumpyArrayOps:
    return NumpyArrayOps()


def test_registry_builds_torch_backend():
    ops = build_array_ops("torch")
    assert isinstance(ops, TorchArrayOps)
    assert ops.name == "torch"


def test_to_host_returns_numpy_from_tensor_and_array(tops):
    t = tops.asarray(np.array([1.0, 2.0, 3.0]))
    assert isinstance(t, torch.Tensor)
    host = tops.to_host(t)
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, [1.0, 2.0, 3.0])
    # Also accepts an already-numpy input (defensive, matches NumpyArrayOps).
    np.testing.assert_array_equal(tops.to_host(np.array([4.0])), [4.0])


def test_zeros_full_dtype_and_device(tops):
    z = tops.zeros((2, 3), dtype=np.float64)
    assert z.dtype == torch.float64
    assert z.device.type == "cpu"
    f = tops.full((2, 3), -np.inf, dtype=np.float32)
    assert torch.isinf(f).all()


@pytest.mark.parametrize(
    "rows,cols,values,shape",
    [
        (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), (3, 3)),
        (
            np.array([0, 0, 0, 1]),
            np.array([1, 1, 1, 2]),
            np.array([1.5, -2.0, 0.5, 4.0]),
            (2, 3),
        ),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]), (4, 4)),
        (
            np.array([0, 1, 2, 3, 0, 1]),
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (4, 5),
        ),
    ],
)
def test_scatter_add_matches_numpy_backend(tops, nops, rows, cols, values, shape):
    got = tops.to_host(tops.scatter_add(tops.zeros(shape, dtype=np.float64), rows, cols, values))
    want = nops.to_host(nops.scatter_add(nops.zeros(shape, dtype=np.float64), rows, cols, values))
    np.testing.assert_allclose(got, want)


@pytest.mark.parametrize(
    "rows,cols,values,shape",
    [
        (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), (3, 3)),
        (
            np.array([0, 0, 0, 1]),
            np.array([1, 1, 1, 2]),
            np.array([1.5, -2.0, 0.5, 4.0]),
            (2, 3),
        ),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]), (4, 4)),
    ],
)
def test_scatter_max_matches_numpy_backend(tops, nops, rows, cols, values, shape):
    got = tops.to_host(
        tops.scatter_max(tops.full(shape, -np.inf, dtype=np.float64), rows, cols, values)
    )
    want = nops.to_host(
        nops.scatter_max(nops.full(shape, -np.inf, dtype=np.float64), rows, cols, values)
    )
    np.testing.assert_array_equal(got, want)


def test_scatter_randomized_duplicates_match_numpy(tops, nops):
    rng = np.random.default_rng(0)
    shape = (5, 7)
    n = 200
    rows = rng.integers(0, shape[0], n)
    cols = rng.integers(0, shape[1], n)
    values = rng.standard_normal(n)

    got_sum = tops.to_host(
        tops.scatter_add(tops.zeros(shape, dtype=np.float64), rows, cols, values)
    )
    want_sum = nops.to_host(
        nops.scatter_add(nops.zeros(shape, dtype=np.float64), rows, cols, values)
    )
    np.testing.assert_allclose(got_sum, want_sum, rtol=1e-10)

    got_max = tops.to_host(
        tops.scatter_max(tops.full(shape, -np.inf, dtype=np.float64), rows, cols, values)
    )
    want_max = nops.to_host(
        nops.scatter_max(nops.full(shape, -np.inf, dtype=np.float64), rows, cols, values)
    )
    np.testing.assert_array_equal(got_max, want_max)


def test_gather_matches_fancy_indexing_broadcast(tops):
    M = np.arange(20).reshape(4, 5)
    rows = np.arange(4)[:, None]
    cols = np.array([[0, 1], [2, 3], [4, 0], [1, 2]]) % 5
    got = tops.to_host(tops.gather(M, rows, cols))
    np.testing.assert_array_equal(got, M[rows, cols])


def test_topk_matches_numpy_backend(tops, nops):
    x = np.array([[3.0, 1.0, 4.0, 1.0, 5.0]])
    vals_n, idx_n = nops.topk(x, 3, axis=1)
    vals_t, idx_t = tops.topk(x, 3, axis=1)
    np.testing.assert_array_equal(tops.to_host(vals_t), vals_n)
    np.testing.assert_array_equal(x[0, tops.to_host(idx_t)[0]], vals_n[0])


def test_argpartition_top_k_set_matches_numpy_backend(tops, nops):
    """TorchArrayOps.argpartition is a full sort (see its docstring) rather
    than a true partition; the values it exposes at the requested tail slice
    must still match the reference top-k set, which is the only contract
    `_dense_topk` (retrieval.py) actually relies on."""
    rng = np.random.default_rng(1)
    sims = rng.standard_normal((3, 10))
    k_eff = 4
    part_n = nops.argpartition(sims, -k_eff, axis=1)[:, -k_eff:]
    part_t = tops.to_host(tops.argpartition(sims, -k_eff, axis=1))[:, -k_eff:]
    rows = np.arange(3)[:, None]
    np.testing.assert_allclose(
        np.sort(sims[rows, part_n], axis=1), np.sort(sims[rows, part_t], axis=1)
    )


def test_nanmin_nanmax_match_numpy_backend(tops, nops):
    M = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]])
    got_min = tops.to_host(tops.nanmin(M, axis=1))
    got_max = tops.to_host(tops.nanmax(M, axis=1))
    want_min = nops.nanmin(M, axis=1)
    want_max = nops.nanmax(M, axis=1)
    np.testing.assert_allclose(got_min, want_min, equal_nan=True)
    np.testing.assert_allclose(got_max, want_max, equal_nan=True)


def test_matmul_where_isnan_isfinite_maximum_log1p_match_numpy_backend(tops, nops):
    a = np.array([[1.0, 2.0]])
    b = np.array([[1.0], [1.0]])
    np.testing.assert_allclose(tops.to_host(tops.matmul(a, b)), nops.matmul(a, b))

    x = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(tops.to_host(tops.isnan(x)), nops.isnan(x))
    np.testing.assert_array_equal(tops.to_host(tops.isfinite(x)), nops.isfinite(x))
    np.testing.assert_allclose(
        tops.to_host(tops.where(x > 1, x, 0.0)), nops.where(x > 1, x, 0.0), equal_nan=True
    )

    a2, b2 = np.array([1.0, 5.0, 2.0]), np.array([4.0, 1.0, 2.0])
    np.testing.assert_allclose(tops.to_host(tops.maximum(a2, b2)), nops.maximum(a2, b2))

    pos = np.abs(np.array([0.1, 2.0, 5.0]))
    np.testing.assert_allclose(tops.to_host(tops.log1p(pos)), nops.log1p(pos))


def test_asarray_accepts_numpy_and_existing_tensor(tops):
    arr = np.array([1, 2, 3], dtype=np.int64)
    t1 = tops.asarray(arr)
    assert isinstance(t1, torch.Tensor)
    t2 = tops.asarray(t1)  # already a tensor: pass-through, not a re-wrap error
    assert isinstance(t2, torch.Tensor)
    np.testing.assert_array_equal(tops.to_host(t2), arr)
