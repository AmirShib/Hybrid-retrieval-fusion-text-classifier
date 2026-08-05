"""T84 -- ArrayOps port: the numpy backend's scatter kernels vs. the
``np.add.at``/``np.maximum.at`` they replace, the auto-selection rule, and the
torch-optional boundary the port must not cross for an ordinary numpy run."""

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

    def test_auto_is_numpy_when_no_torch_backend_registered(self):
        assert "torch" not in registered_array_ops_kinds()
        # device_visible=True would matter only if a torch backend existed.
        got = resolve_array_backend(
            None, n_items=10_000_000, n_classes=100_000, device_visible=True
        )
        assert got == "numpy"

    def test_auto_is_numpy_when_no_device_visible(self):
        got = resolve_array_backend(
            None,
            n_items=CROSSOVER_MIN_ITEMS + 1,
            n_classes=CROSSOVER_MIN_CLASSES + 1,
            device_visible=False,
        )
        assert got == "numpy"

    def test_auto_short_circuits_before_probing_cuda(self):
        """The registry check (no torch backend registered) must run before any
        CUDA probe -- device_visible=None normally triggers ``cuda_available()``,
        which imports torch; that must never happen while numpy is the only
        registered backend (see the no-torch-import test below)."""
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
