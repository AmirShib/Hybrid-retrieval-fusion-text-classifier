"""Array-backend port implementations (T84, T85).

``NumpyArrayOps`` is a thin pass-through numpy backend -- the default. Every
method except ``scatter_add``/``scatter_max`` is a direct 1:1 call onto
``numpy``, so routing a kernel through the port is a zero-behaviour-change
swap.

``scatter_add``/``scatter_max`` are the one place that backend does more than
delegate: they replace ``np.add.at``/``np.maximum.at`` (unbuffered, the
slowest scatter numpy offers) with vectorized equivalents -- ``np.bincount``
for the sum, a sort + ``np.maximum.reduceat`` for the max, since ``bincount``
itself has no max-reduction mode. Both are correct for arbitrary duplicate
indices, including rows that receive no updates at all.

``TorchArrayOps`` (T85) is the device-resident backend: the same kernels, the
same call sequence, running on whatever device it was constructed for. It
lives behind the ``gpu`` extra -- torch is imported inside its constructor,
never at module import, so a numpy-backend run on a torch-free host never
touches it (T63's boundary, guarded by a test).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..domain.ports import ArrayOps
# `device` is import-cheap and torch-free at module level (its probes wrap the
# import in try/except), so this costs nothing on a numpy-only host.
from .device import cuda_available, torch_installed

logger = logging.getLogger(__name__)


class NumpyArrayOps(ArrayOps):
    name = "numpy"

    def asarray(self, x: Any, dtype: Optional[Any] = None) -> np.ndarray:
        return np.asarray(x, dtype=dtype)

    def to_host(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def zeros(self, shape: Any, dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def full(self, shape: Any, value: Any, dtype: Any) -> np.ndarray:
        return np.full(shape, value, dtype=dtype)

    def where(self, cond: Any, a: Any, b: Any) -> np.ndarray:
        return np.where(cond, a, b)

    def isnan(self, x: Any) -> np.ndarray:
        return np.isnan(x)

    def isfinite(self, x: Any) -> np.ndarray:
        return np.isfinite(x)

    def maximum(self, a: Any, b: Any) -> np.ndarray:
        return np.maximum(a, b)

    def log1p(self, x: Any) -> np.ndarray:
        return np.log1p(x)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return a @ b

    def topk(self, x: Any, k: int, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        n = x.shape[axis]
        k = min(k, n)
        idx = np.argpartition(x, n - k, axis=axis)
        idx = np.take(idx, range(n - k, n), axis=axis)
        vals = np.take_along_axis(x, idx, axis=axis)
        order = np.argsort(-vals, axis=axis)
        idx = np.take_along_axis(idx, order, axis=axis)
        vals = np.take_along_axis(vals, order, axis=axis)
        return vals, idx

    def argsort(self, x: Any, axis: int = -1, stable: bool = False) -> np.ndarray:
        return np.argsort(x, axis=axis, kind="stable" if stable else None)

    def argmax(self, x: Any, axis: int = -1) -> np.ndarray:
        return np.argmax(x, axis=axis)

    def argpartition(self, x: Any, k: int, axis: int = -1) -> np.ndarray:
        return np.argpartition(x, k, axis=axis)

    def nanmin(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.nanmin(x, axis=axis)

    def nanmax(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.nanmax(x, axis=axis)

    def sum(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    def sqrt(self, x: Any) -> np.ndarray:
        return np.sqrt(x)

    def arange(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.int64)

    def astype(self, x: Any, dtype: Any) -> np.ndarray:
        return np.asarray(x).astype(dtype, copy=False)

    def reshape(self, x: Any, shape: Any) -> np.ndarray:
        return np.reshape(x, shape)

    def transpose(self, x: Any) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(x).T)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> np.ndarray:
        return np.concatenate(list(arrays), axis=axis)

    def stack(self, arrays: Sequence[Any], axis: int = 0) -> np.ndarray:
        return np.stack(list(arrays), axis=axis)

    def repeat(self, x: Any, n: int) -> np.ndarray:
        return np.repeat(x, n)

    def tile(self, x: Any, n: int) -> np.ndarray:
        return np.tile(x, n)

    def take(self, x: Any, idx: Any) -> np.ndarray:
        return np.asarray(x)[idx]

    def take_along_axis(self, x: Any, idx: Any, axis: int) -> np.ndarray:
        return np.take_along_axis(x, idx, axis=axis)

    def nonzero(self, mask: Any) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.nonzero(mask)
        return rows, cols

    def scatter_add(self, target: Any, rows: Any, cols: Any, values: Any) -> np.ndarray:
        target = np.asarray(target)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        values = np.asarray(values)
        if rows.size == 0:
            return target
        R, C = target.shape
        flat = rows.astype(np.int64) * C + cols.astype(np.int64)
        add = np.bincount(flat, weights=values, minlength=R * C).reshape(R, C)
        return target + add.astype(target.dtype, copy=False)

    def scatter_max(self, target: Any, rows: Any, cols: Any, values: Any) -> np.ndarray:
        target = np.array(target, copy=True)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        values = np.asarray(values)
        if rows.size == 0:
            return target
        R, C = target.shape
        flat = rows.astype(np.int64) * C + cols.astype(np.int64)
        order = np.argsort(flat, kind="stable")
        flat_s = flat[order]
        val_s = values[order]
        uniq, first_idx = np.unique(flat_s, return_index=True)
        grouped_max = np.maximum.reduceat(val_s, first_idx)
        flat_target = target.reshape(-1)
        flat_target[uniq] = np.maximum(flat_target[uniq], grouped_max)
        return flat_target.reshape(R, C)

    def scatter_set(self, target: Any, rows: Any, cols: Any, values: Any) -> np.ndarray:
        target = np.array(target, copy=True)
        target[rows, cols] = values
        return target

    def gather(self, M: Any, rows: Any, cols: Any) -> np.ndarray:
        return np.asarray(M)[rows, cols]


# ------------------------------------------------------------------ torch backend
def _require_torch() -> Any:
    """Import ``torch``, or raise a clear, actionable error.

    ``torch`` lives behind the ``gpu`` extra, not core ``dependencies`` (T63
    established that boundary; T85 slots this backend into it) — an air-gapped
    or lightweight install may not have it, and a bare ``ImportError`` from the
    middle of a training run is unhelpful."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "array backend 'torch' requires the 'gpu' extra, which is not installed. "
            "Install it with:\n"
            "    pip install text-classifier[gpu]\n"
            "Or keep the default numpy backend (array_backend='numpy'), which needs "
            "no torch at all."
        ) from exc
    return torch


class TorchArrayOps(ArrayOps):
    """Device-resident array backend (T85).

    Same kernels, same call order as ``NumpyArrayOps`` — only the arrays' home
    changes. Cross-backend results are equal within float tolerance, never
    bit-for-bit: float32 reduction order differs on a device, so continuous
    columns move in the last ulps and a near-tie can flip an ordinal column
    (``rank_*``, ``is_*_top1``) and therefore the candidate set. CPU stays the
    reference implementation; see ``docs/device-policy.md``.

    Dtypes cross the port as numpy dtypes (the port's stated convention) and
    are mapped onto torch here, so no call site has to know which backend it is
    talking to.

    Determinism: the two scatter kernels accumulate with ``index_put_``/
    ``scatter_reduce_``, whose CUDA implementations use atomics and therefore
    do not fix a summation order. Runs on one host + one GPU + one seed agree
    to float tolerance; for bitwise repeatability set
    ``torch.use_deterministic_algorithms(True)``, which selects torch's
    deterministic implementations of exactly these ops."""

    name = "torch"

    def __init__(self, device: Optional[str] = None):
        torch = _require_torch()
        self._torch = torch
        if device is None:
            from .device import resolve_device

            device = resolve_device(None)
        self.device = torch.device(device)
        self._dtypes = {
            np.dtype(np.float64): torch.float64,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32,
            np.dtype(bool): torch.bool,
        }
        logger.info("array backend: torch on device=%s", self.device)

    # ---------------------------------------------------------------- helpers
    def _dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return None
        if isinstance(dtype, self._torch.dtype):
            return dtype
        return self._dtypes[np.dtype(dtype)]

    def _t(self, x: Any, dtype: Optional[Any] = None) -> Any:
        """Adopt ``x`` as a tensor on this backend's device.

        A tensor already here is returned untouched (no copy, no transfer) —
        the property that lets a kernel call this defensively, and the reason a
        device-resident encoder's output crosses the boundary for free."""
        torch = self._torch
        want = self._dtype(dtype)
        if isinstance(x, torch.Tensor):
            out = x if x.device == self.device else x.to(self.device)
            return out if want is None or out.dtype == want else out.to(want)
        arr = np.asarray(x)
        if arr.dtype == np.float16:  # not in the map; widen rather than fail
            arr = arr.astype(np.float32)
        tensor = torch.as_tensor(arr, device=self.device)
        return tensor if want is None else tensor.to(want)

    # ---------------------------------------------------------------- port
    def asarray(self, x: Any, dtype: Optional[Any] = None) -> Any:
        return self._t(x, dtype)

    def to_host(self, x: Any) -> np.ndarray:
        if isinstance(x, self._torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _shape(shape: Any) -> Tuple[int, ...]:
        return tuple(int(s) for s in np.atleast_1d(shape))

    def _pair(self, a: Any, b: Any) -> Tuple[Any, Any]:
        """Two operands as tensors of one dtype, following numpy's weak-scalar
        rule: a bare Python/numpy scalar adopts the array's dtype rather than
        promoting it (so ``where(mask, float32_matrix, np.nan)`` stays float32
        on the device, exactly as it does on the host)."""
        torch = self._torch
        a_scalar, b_scalar = np.isscalar(a), np.isscalar(b)
        if a_scalar and not b_scalar:
            bt = self._t(b)
            return torch.as_tensor(a, dtype=bt.dtype, device=self.device), bt
        if b_scalar and not a_scalar:
            at = self._t(a)
            return at, torch.as_tensor(b, dtype=at.dtype, device=self.device)
        at, bt = self._t(a), self._t(b)
        if at.dtype != bt.dtype:
            promoted = torch.promote_types(at.dtype, bt.dtype)
            at, bt = at.to(promoted), bt.to(promoted)
        return at, bt

    def zeros(self, shape: Any, dtype: Any) -> Any:
        return self._torch.zeros(self._shape(shape), dtype=self._dtype(dtype), device=self.device)

    def full(self, shape: Any, value: Any, dtype: Any) -> Any:
        torch_dtype = self._dtype(dtype)
        if torch_dtype in (self._torch.int64, self._torch.int32):
            value = int(value)
        return self._torch.full(
            self._shape(shape), value, dtype=torch_dtype, device=self.device
        )

    def where(self, cond: Any, a: Any, b: Any) -> Any:
        at, bt = self._pair(a, b)
        return self._torch.where(self._t(cond, bool), at, bt)

    def isnan(self, x: Any) -> Any:
        return self._torch.isnan(self._t(x))

    def isfinite(self, x: Any) -> Any:
        return self._torch.isfinite(self._t(x))

    def maximum(self, a: Any, b: Any) -> Any:
        at, bt = self._pair(a, b)
        return self._torch.maximum(at, bt)

    def log1p(self, x: Any) -> Any:
        return self._torch.log1p(self._t(x))

    def matmul(self, a: Any, b: Any) -> Any:
        return self._t(a) @ self._t(b)

    def topk(self, x: Any, k: int, axis: int = -1) -> Tuple[Any, Any]:
        x = self._t(x)
        k = min(k, x.shape[axis])
        vals, idx = self._torch.topk(x, k, dim=axis, largest=True, sorted=True)
        return vals, idx.to(self._torch.int64)

    def argsort(self, x: Any, axis: int = -1, stable: bool = False) -> Any:
        return self._torch.argsort(self._t(x), dim=axis, stable=stable).to(self._torch.int64)

    def argmax(self, x: Any, axis: int = -1) -> Any:
        return self._torch.argmax(self._t(x), dim=axis).to(self._torch.int64)

    def argpartition(self, x: Any, k: int, axis: int = -1) -> Any:
        # No partition primitive in torch; a full sort satisfies the port's
        # stated contract (position k lands in its sorted place). The two call
        # sites both slice a fixed handful of columns off the result, so the
        # extra ordering is discarded, not relied on.
        return self.argsort(x, axis=axis)

    def nanmin(self, x: Any, axis: Optional[int] = None) -> Any:
        return self._nan_reduce(x, axis, minimum=True)

    def nanmax(self, x: Any, axis: Optional[int] = None) -> Any:
        return self._nan_reduce(x, axis, minimum=False)

    def _nan_reduce(self, x: Any, axis: Optional[int], minimum: bool) -> Any:
        """NaN-skipping min/max that keeps numpy's all-NaN answer.

        torch has no ``nanmin``/``nanmax`` over an axis, and the usual
        replace-with-inf trick reports the sentinel for an all-NaN row where
        numpy reports NaN (with a RuntimeWarning the call sites already
        suppress). ``_row_minmax`` depends on that NaN — it is how an
        all-missing row stays missing — so it is restored explicitly."""
        torch = self._torch
        t = self._t(x)
        fill = float("inf") if minimum else float("-inf")
        filled = torch.where(torch.isnan(t), torch.full_like(t, fill), t)
        if axis is None:
            out = filled.min() if minimum else filled.max()
            return torch.where(torch.isnan(t).all(), torch.full_like(out, float("nan")), out)
        out = filled.amin(dim=axis) if minimum else filled.amax(dim=axis)
        all_nan = torch.isnan(t).all(dim=axis)
        return torch.where(all_nan, torch.full_like(out, float("nan")), out)

    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        t = self._t(x)
        return t.sum() if axis is None else t.sum(dim=axis)

    def sqrt(self, x: Any) -> Any:
        return self._torch.sqrt(self._t(x))

    def arange(self, n: int) -> Any:
        return self._torch.arange(int(n), dtype=self._torch.int64, device=self.device)

    def astype(self, x: Any, dtype: Any) -> Any:
        return self._t(x).to(self._dtype(dtype))

    def reshape(self, x: Any, shape: Any) -> Any:
        return self._t(x).reshape(tuple(np.atleast_1d(shape)))

    def transpose(self, x: Any) -> Any:
        return self._t(x).T.contiguous()

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        return self._torch.cat([self._t(a) for a in arrays], dim=axis)

    def stack(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        tensors = [self._t(a) for a in arrays]
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            dtype = self._torch.promote_types(dtype, t.dtype)
        return self._torch.stack([t.to(dtype) for t in tensors], dim=axis)

    def repeat(self, x: Any, n: int) -> Any:
        return self._torch.repeat_interleave(self._t(x), int(n))

    def tile(self, x: Any, n: int) -> Any:
        return self._t(x).repeat(int(n))

    def take(self, x: Any, idx: Any) -> Any:
        return self._t(x)[self._index(idx)]

    def take_along_axis(self, x: Any, idx: Any, axis: int) -> Any:
        return self._torch.take_along_dim(self._t(x), self._t(idx, np.int64), dim=axis)

    def nonzero(self, mask: Any) -> Tuple[Any, Any]:
        rows, cols = self._torch.nonzero(self._t(mask, bool), as_tuple=True)
        return rows, cols

    def _index(self, idx: Any) -> Any:
        """Coerce an index argument to a tensor, preserving boolean masks."""
        arr = self._t(idx)
        return arr if arr.dtype == self._torch.bool else arr.to(self._torch.int64)

    def scatter_add(self, target: Any, rows: Any, cols: Any, values: Any) -> Any:
        target = self._t(target)
        rows, cols = self._index(rows), self._index(cols)
        if rows.numel() == 0:
            return target
        R, C = target.shape
        flat = rows * C + cols
        out = target.reshape(-1).clone()
        out.index_put_((flat,), self._t(values, target.dtype), accumulate=True)
        return out.reshape(R, C)

    def scatter_max(self, target: Any, rows: Any, cols: Any, values: Any) -> Any:
        target = self._t(target)
        rows, cols = self._index(rows), self._index(cols)
        if rows.numel() == 0:
            return target
        R, C = target.shape
        flat = rows * C + cols
        out = target.reshape(-1).clone()
        out.scatter_reduce_(0, flat, self._t(values, target.dtype), reduce="amax")
        return out.reshape(R, C)

    def scatter_set(self, target: Any, rows: Any, cols: Any, values: Any) -> Any:
        target = self._t(target).clone()
        target[self._index(rows), self._index(cols)] = self._t(values, target.dtype)
        return target

    def gather(self, M: Any, rows: Any, cols: Any) -> Any:
        return self._t(M)[self._index(rows), self._index(cols)]

    def free_memory(self) -> None:
        # Freed blocks stay in torch's caching allocator, so an OOM retry at a
        # smaller chunk can fail on memory that is already ours. Only meaningful
        # on CUDA; a no-op elsewhere.
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()


# --------------------------------------------------------------- auto-selection
# T83's measured crossover: below this region a device round trip's overhead
# (kernel launch + transfer latency) is not repaid by the compute it replaces;
# at or above it, the CPU cost of the dense-side stages (prototypes, top-k,
# scatter, rank/margin leaves) is large enough that a device-resident port
# should win. See docs/device-policy.md "Crossover thresholds". Module
# constants (not a heuristic sprinkled through the pipeline) so T85 can cite
# this exact rule.
CROSSOVER_MIN_ITEMS = 100_000
CROSSOVER_MIN_CLASSES = 500


def available_array_backend(kind: str) -> str:
    """Downgrade a requested backend to numpy when this host cannot provide it.

    The array backend is an execution choice, never a property of a trained
    model (T84) — so a model directory that records ``"torch"`` must still load
    and score on a torch-free, CPU-only host, which is the portability
    invariant in CLAUDE.md. ``resolve_array_backend`` never picks a backend the
    host lacks; this is its load-side counterpart, where the kind comes from a
    file written on another machine."""
    if kind == "torch" and not torch_installed():
        logger.warning(
            "this model was trained with the torch array backend, which is not installed "
            "here; running on numpy instead. Predictions are equal within float tolerance, "
            "not bit-for-bit (see docs/device-policy.md)."
        )
        return "numpy"
    return kind


def resolve_array_backend(
    explicit: Optional[str],
    *,
    n_items: int,
    n_classes: int,
    k_neighbors: Optional[int] = None,
    device_visible: Optional[bool] = None,
    dense_kind: Optional[str] = None,
) -> str:
    """Resolve the array backend for a run of this scale.

    An explicit ``kind`` (anything other than ``None``/``"auto"``) always
    wins -- the same explicit-beats-detected rule ``resolve_device`` already
    follows, and so does an explicit device-resident ``dense_kind``. Otherwise
    ``"torch"`` requires all four of: a registered torch backend, a run at or
    above ``CROSSOVER_MIN_ITEMS``/``CROSSOVER_MIN_CLASSES``, torch actually
    installed, and a visible device. The choice and the reason are logged --
    this is meant to be traceable, not silent.

    The check order is load-bearing, not cosmetic. Everything cheap and
    import-free runs first, so an ordinary numpy-scale run never imports torch
    even on a host that has it (T63's boundary; also the OpenMP-runtime clash
    between torch's bundled libomp and xgboost's when both land in one
    process). Only a run that is *already* over the crossover pays for
    ``find_spec`` and then the CUDA probe."""
    if explicit is not None and explicit != "auto":
        logger.info("array backend: %r (explicit)", explicit)
        return explicit

    if dense_kind == "torch":
        # An explicitly device-resident retriever is itself a device request,
        # and outranks the scale heuristic: `dense_kind="torch"` with a numpy
        # assembler would upload and download around every kernel. The
        # contradictory *explicit* pairing (`array_backend="numpy"`) never gets
        # here -- `PipelineConfig.validate` rejects it.
        logger.info("array backend: torch (auto: retrieval.dense_kind='torch')")
        return "torch"

    from . import registry

    if "torch" not in registry.registered_array_ops_kinds():
        logger.info("array backend: numpy (auto: no torch backend registered)")
        return "numpy"

    if not (n_items >= CROSSOVER_MIN_ITEMS or n_classes >= CROSSOVER_MIN_CLASSES):
        logger.info(
            "array backend: numpy (auto: n_items=%d n_classes=%d below crossover "
            "n_items>=%d or n_classes>=%d)",
            n_items,
            n_classes,
            CROSSOVER_MIN_ITEMS,
            CROSSOVER_MIN_CLASSES,
        )
        return "numpy"

    if not torch_installed():
        logger.info("array backend: numpy (auto: torch is not installed)")
        return "numpy"

    if device_visible is None:
        device_visible = cuda_available()
    if not device_visible:
        logger.info("array backend: numpy (auto: no device visible)")
        return "numpy"

    logger.info(
        "array backend: torch (auto: n_items=%d n_classes=%d meets crossover "
        "n_items>=%d or n_classes>=%d)",
        n_items,
        n_classes,
        CROSSOVER_MIN_ITEMS,
        CROSSOVER_MIN_CLASSES,
    )
    return "torch"
