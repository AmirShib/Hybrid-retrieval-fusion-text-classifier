"""T84/T85 -- ArrayOps port: the numpy backend's scatter kernels vs. the
``np.add.at``/``np.maximum.at`` they replace, the auto-selection rule, the
torch-optional boundary the port must not cross for an ordinary numpy run, and
(T85) the torch backend method-by-method against the numpy one it has to be
interchangeable with."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from text_classifier.domain import ArrayOps
from text_classifier.infrastructure.array_ops import (
    CROSSOVER_MIN_CLASSES,
    CROSSOVER_MIN_ITEMS,
    NumpyArrayOps,
    resolve_array_backend,
)
from text_classifier.infrastructure import registry
from text_classifier.infrastructure.registry import (
    array_ops_spec,
    build_array_ops,
    registered_array_ops_kinds,
)


def _reference_scatter(rows, cols, values, shape):
    ksum = np.zeros(shape, dtype=np.float64)
    kmax = np.full(shape, -np.inf, dtype=np.float64)
    np.add.at(ksum, (rows, cols), values)
    np.maximum.at(kmax, (rows, cols), values)
    return ksum, kmax


@pytest.mark.parametrize(
    "rows,cols,values,shape",
    [
        # ordinary, no duplicates
        (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), (3, 3)),
        # duplicate (row, col) pairs within one target cell
        (
            np.array([0, 0, 0, 1]),
            np.array([1, 1, 1, 2]),
            np.array([1.5, -2.0, 0.5, 4.0]),
            (2, 3),
        ),
        # empty update
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]), (4, 4)),
        # every row hit, some columns never touched (all-NaN-equivalent columns)
        (
            np.array([0, 1, 2, 3, 0, 1]),
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (4, 5),
        ),
    ],
)
def test_scatter_add_matches_add_at(rows, cols, values, shape):
    ops = NumpyArrayOps()
    target = ops.zeros(shape, dtype=np.float64)
    got = ops.to_host(ops.scatter_add(target, rows, cols, values))
    want, _ = _reference_scatter(rows, cols, values, shape)
    np.testing.assert_array_equal(got, want)


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
def test_scatter_max_matches_maximum_at(rows, cols, values, shape):
    ops = NumpyArrayOps()
    target = ops.full(shape, -np.inf, dtype=np.float64)
    got = ops.to_host(ops.scatter_max(target, rows, cols, values))
    _, want = _reference_scatter(rows, cols, values, shape)
    np.testing.assert_array_equal(got, want)


def test_scatter_add_scatter_max_randomized_duplicates():
    rng = np.random.default_rng(0)
    shape = (5, 7)
    n = 200
    rows = rng.integers(0, shape[0], n)
    cols = rng.integers(0, shape[1], n)
    values = rng.standard_normal(n)
    ops = NumpyArrayOps()

    got_sum = ops.to_host(ops.scatter_add(ops.zeros(shape, dtype=np.float64), rows, cols, values))
    got_max = ops.to_host(
        ops.scatter_max(ops.full(shape, -np.inf, dtype=np.float64), rows, cols, values)
    )
    want_sum, want_max = _reference_scatter(rows, cols, values, shape)
    np.testing.assert_allclose(got_sum, want_sum, rtol=0, atol=0)
    np.testing.assert_array_equal(got_max, want_max)


def test_gather_matches_fancy_indexing():
    ops = NumpyArrayOps()
    M = np.arange(20).reshape(4, 5)
    rows = np.array([0, 1, 3, 2])
    cols = np.array([4, 0, 2, 1])
    np.testing.assert_array_equal(ops.gather(M, rows, cols), M[rows, cols])


def test_topk_returns_best_first_values_and_indices():
    ops = NumpyArrayOps()
    x = np.array([[3.0, 1.0, 4.0, 1.0, 5.0]])
    vals, idx = ops.topk(x, 3, axis=1)
    np.testing.assert_array_equal(vals, np.array([[5.0, 4.0, 3.0]]))
    np.testing.assert_array_equal(x[0, idx[0]], vals[0])


def test_matmul_where_isnan_pass_through():
    ops = NumpyArrayOps()
    a = np.array([[1.0, 2.0]])
    b = np.array([[1.0], [1.0]])
    np.testing.assert_array_equal(ops.matmul(a, b), a @ b)
    x = np.array([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(ops.isnan(x), np.isnan(x))
    np.testing.assert_array_equal(ops.where(x > 1, x, 0.0), np.where(x > 1, x, 0.0))


class TestAutoSelection:
    def test_explicit_kind_always_wins(self):
        assert resolve_array_backend("numpy", n_items=1, n_classes=1) == "numpy"
        # Even a made-up kind wins -- resolution doesn't validate registry
        # membership (build_array_ops raises for that; resolve_array_backend
        # is only responsible for picking a name).
        assert resolve_array_backend("bogus", n_items=1, n_classes=1) == "bogus"

    def test_auto_is_numpy_when_no_torch_backend_registered(self, monkeypatch):
        """The registry gate still short-circuits. T85 registers a torch
        backend, so this pokes the gate directly rather than relying on the
        registry being empty (which it no longer is)."""
        monkeypatch.setattr(registry, "registered_array_ops_kinds", lambda: ["numpy"])
        got = resolve_array_backend(
            None, n_items=10_000_000, n_classes=100_000, device_visible=True
        )
        assert got == "numpy"

    def test_torch_dense_kind_forces_the_torch_backend(self):
        """An explicitly device-resident retriever outranks the scale
        heuristic: a torch dense index feeding a numpy assembler would
        transfer around every kernel."""
        got = resolve_array_backend(None, n_items=1, n_classes=1, dense_kind="torch")
        assert got == "torch"

    def test_auto_is_numpy_when_no_device_visible(self):
        """With T85's torch backend registered, the device probe is what
        decides at scale -- and CI hosts have no CUDA device."""
        assert "torch" in registered_array_ops_kinds()
        got = resolve_array_backend(
            None,
            n_items=CROSSOVER_MIN_ITEMS + 1,
            n_classes=CROSSOVER_MIN_CLASSES + 1,
            device_visible=False,
        )
        assert got == "numpy"

    def test_auto_short_circuits_before_probing_cuda(self):
        """Every import-free check runs before any CUDA probe --
        device_visible=None would otherwise trigger ``cuda_available()``, which
        imports torch. A below-crossover run must never pay for that, on any
        host (see the no-torch-import test below)."""
        got = resolve_array_backend(None, n_items=1, n_classes=1, device_visible=None)
        assert got == "numpy"


class TestRegistry:
    def test_numpy_registered_by_default(self):
        assert "numpy" in registered_array_ops_kinds()
        ops = build_array_ops("numpy")
        assert isinstance(ops, ArrayOps)
        assert ops.name == "numpy"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown array ops kind"):
            array_ops_spec("bogus")


def test_no_torch_import_reachable_from_a_numpy_backend_run(monkeypatch):
    """T84's stated guard: with torch blocked at import time, resolving 'auto'
    and running feature assembly (the array-ops call sites) must not need it.

    This targets the array-ops seam specifically -- not the whole training
    pipeline, which already has an unrelated, pre-existing torch import in
    ``infrastructure/fusion.py`` (XGBoost's own ``device=None`` resolution) that
    is out of T84's scope."""
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`

    backend = resolve_array_backend(None, n_items=500, n_classes=10)
    assert backend == "numpy"

    from text_classifier.application.features import FeatureAssembler
    from text_classifier.domain import CandidatePolicy, LabelSpace
    from text_classifier.domain.models import ClassDefinition
    from text_classifier.infrastructure import (
        DenseRetrieverAdapter,
        LexicalRetrieverAdapter,
    )
    from text_classifier.config import PipelineConfig

    ops = build_array_ops(backend)
    classes = [ClassDefinition(f"c{i}", f"desc {i}") for i in range(4)]
    label_space = LabelSpace(classes)
    texts = [f"item {i} about c{i % 4}" for i in range(12)]
    labels = np.array([i % 4 for i in range(12)], dtype=np.int64)
    cfg = PipelineConfig().retrieval
    emb = np.eye(4, dtype=np.float32)[labels]
    desc_emb = np.eye(4, dtype=np.float32)
    dense = DenseRetrieverAdapter.build_from_embeddings(
        emb, labels, desc_emb, label_space, cfg, ops
    )
    lexical = LexicalRetrieverAdapter.build(texts, labels, label_space, cfg)
    assembler = FeatureAssembler(label_space, CandidatePolicy(top_n_per_signal=2), ops)
    frame = assembler.assemble(
        texts, emb, dense, lexical, k_neighbors=3, query_ids=list(range(len(texts)))
    )
    assert len(frame) > 0
    assert "torch" not in sys.modules or sys.modules["torch"] is None


# --------------------------------------------------------------- torch backend
# T85. Skipped wholesale when torch is absent -- that is the supported
# configuration for a core (torch-free) install, not a broken one. Every case
# below asserts the torch backend against the numpy backend it must be
# interchangeable with, method by method; the frame-level parity that actually
# matters lives in tests/integration/test_device_parity.py.
pytest.importorskip("torch", reason="the torch array backend needs the 'gpu' extra")

from text_classifier.infrastructure.array_ops import TorchArrayOps  # noqa: E402


@pytest.fixture
def ops_pair():
    return NumpyArrayOps(), TorchArrayOps("cpu")


def _same(np_ops, t_ops, got_t, want_np, exact=True):
    got = t_ops.to_host(got_t)
    want = np.asarray(want_np)
    assert got.shape == want.shape
    if exact:
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


class TestTorchBackendMatchesNumpy:
    def test_registered_and_named(self):
        assert "torch" in registered_array_ops_kinds()
        ops = build_array_ops("torch")
        assert isinstance(ops, ArrayOps) and ops.name == "torch"

    def test_asarray_is_a_no_op_for_resident_arrays(self, ops_pair):
        _, t = ops_pair
        resident = t.asarray(np.arange(6, dtype=np.float32))
        assert t.asarray(resident) is resident, "re-adopting must not copy or transfer"

    def test_scatter_add_and_max(self, ops_pair):
        n, t = ops_pair
        rows = np.array([0, 0, 1, 1, 1])
        cols = np.array([1, 1, 0, 2, 2])
        vals = np.array([1.5, -2.0, 3.0, 0.5, 4.0])
        for op in ("scatter_add", "scatter_max"):
            init = (
                (lambda o: o.zeros((2, 3), dtype=np.float64))
                if op == "scatter_add"
                else (lambda o: o.full((2, 3), -np.inf, dtype=np.float64))
            )
            _same(n, t, getattr(t, op)(init(t), rows, cols, vals),
                  getattr(n, op)(init(n), rows, cols, vals))

    def test_scatter_set(self, ops_pair):
        n, t = ops_pair
        base_np = np.arange(12, dtype=np.float64).reshape(3, 4)
        rows, cols, vals = np.array([0, 2]), np.array([3, 1]), np.array([-1.0, -2.0])
        _same(n, t, t.scatter_set(t.asarray(base_np), rows, cols, vals),
              n.scatter_set(base_np, rows, cols, vals))
        # the source array is not mutated by either backend
        np.testing.assert_array_equal(base_np, np.arange(12).reshape(3, 4))

    def test_topk_argsort_argmax(self, ops_pair):
        n, t = ops_pair
        x = np.array([[3.0, 1.0, 4.0, 1.0, 5.0], [-1.0, 0.0, 2.0, 2.5, -3.0]])
        v_t, i_t = t.topk(t.asarray(x), 3, axis=1)
        v_n, i_n = n.topk(x, 3, axis=1)
        _same(n, t, v_t, v_n)
        _same(n, t, i_t, i_n)
        _same(n, t, t.argmax(t.asarray(x), axis=1), n.argmax(x, axis=1))
        _same(n, t, t.argsort(t.asarray(x), axis=1), n.argsort(x, axis=1))

    def test_argsort_stable_keeps_equal_keys_in_order(self, ops_pair):
        n, t = ops_pair
        # `_exclude_self` sorts by a boolean: every False ties, and their
        # original (best-first) order must survive.
        flags = np.array([[False, True, False, True, False]])
        _same(n, t, t.argsort(t.asarray(flags), axis=1, stable=True),
              n.argsort(flags, axis=1, stable=True))

    def test_nan_reductions_keep_the_all_nan_row_nan(self, ops_pair):
        n, t = ops_pair
        x = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
        for name in ("nanmin", "nanmax"):
            got = t.to_host(getattr(t, name)(t.asarray(x), axis=1))
            want = np.array([1.0 if name == "nanmin" else 3.0, np.nan])
            np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
            np.testing.assert_allclose(got[:1], want[:1])

    def test_where_keeps_the_arrays_dtype_for_a_scalar_operand(self, ops_pair):
        _, t = ops_pair
        M = t.asarray(np.arange(4, dtype=np.float32).reshape(2, 2))
        out = t.where(M > 1, M, np.nan)
        assert out.dtype == M.dtype, "a weak scalar must not promote the array"
        np.testing.assert_array_equal(
            t.to_host(out), np.where(np.arange(4).reshape(2, 2) > 1,
                                     np.arange(4, dtype=np.float32).reshape(2, 2), np.nan)
        )

    def test_shape_ops(self, ops_pair):
        n, t = ops_pair
        x = np.arange(6, dtype=np.float64).reshape(2, 3)
        _same(n, t, t.transpose(t.asarray(x)), n.transpose(x))
        _same(n, t, t.reshape(t.asarray(x), -1), n.reshape(x, -1))
        _same(n, t, t.concatenate([t.asarray(x), t.asarray(x)], axis=0), n.concatenate([x, x], 0))
        _same(n, t, t.stack([t.asarray(x), t.asarray(x)], axis=1), n.stack([x, x], axis=1))
        _same(n, t, t.repeat(t.arange(3), 2), n.repeat(n.arange(3), 2))
        _same(n, t, t.tile(t.arange(3), 2), n.tile(n.arange(3), 2))
        _same(n, t, t.take(t.asarray(x), np.array([1, 0])), n.take(x, np.array([1, 0])))
        idx = np.array([[2, 0], [1, 1]])
        _same(n, t, t.take_along_axis(t.asarray(x), idx, 1), n.take_along_axis(x, idx, 1))

    def test_nonzero_is_row_major(self, ops_pair):
        n, t = ops_pair
        mask = np.array([[False, True, True], [True, False, False]])
        r_t, c_t = t.nonzero(t.asarray(mask))
        r_n, c_n = n.nonzero(mask)
        _same(n, t, r_t, r_n)
        _same(n, t, c_t, c_n)

    def test_reductions_and_elementwise(self, ops_pair):
        n, t = ops_pair
        x = np.array([[1.0, 4.0], [9.0, 16.0]])
        _same(n, t, t.sum(t.asarray(x), axis=1), n.sum(x, axis=1), exact=False)
        _same(n, t, t.sqrt(t.asarray(x)), n.sqrt(x), exact=False)
        _same(n, t, t.log1p(t.asarray(x)), n.log1p(x), exact=False)
        _same(n, t, t.maximum(t.asarray(x), 5.0), n.maximum(x, 5.0), exact=False)
        _same(n, t, t.gather(t.asarray(x), np.array([0, 1]), np.array([1, 0])),
              n.gather(x, np.array([0, 1]), np.array([1, 0])))
