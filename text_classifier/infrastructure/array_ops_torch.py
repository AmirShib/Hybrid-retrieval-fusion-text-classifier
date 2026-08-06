"""Torch ``ArrayOps`` backend (T85).

Kept in its own module, separate from ``array_ops.py``, so that importing
``infrastructure.array_ops`` (which ``infrastructure/__init__.py`` imports
unconditionally) never imports torch. This module itself imports torch at the
top level -- that is fine, because nothing imports *this* module until
``registry.py``'s ``_build_torch_array_ops`` actually calls it (see there),
which only happens when a caller explicitly asks for the "torch" array-ops
kind. T63's boundary ("an ordinary numpy-backend run never imports torch") is
enforced by that lazy indirection, not by avoiding the import here.

Every method transparently accepts either a numpy array/python scalar or an
existing torch tensor, uploading the former to this backend's device on
first touch. This is what lets ``DenseState``'s arrays -- uploaded once at
build time -- stay resident across every query chunk: an already-device-
resident tensor just gets used in place; only host-side inputs (index
bookkeeping like `rows`/`cols`, or a caller that never went through the torch
encoder) pay a small per-call upload.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from ..domain.ports import ArrayOps
from .device import resolve_device


def _torch_dtype(dtype: Any) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    np_dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.float64): torch.float64,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.bool_): torch.bool,
    }
    try:
        return mapping[np_dtype]
    except KeyError:
        raise TypeError(f"no torch dtype mapping for numpy dtype {np_dtype}") from None


class TorchArrayOps(ArrayOps):
    """Device-resident backend. ``device`` defaults to the same auto-detection
    ``resolve_device`` gives the encoder (cuda, else mps, else cpu) -- the
    common single-GPU CUDA deployment then has the encoder and the array
    backend agree on a device with no extra config. An explicit, differently-
    pinned ``encoder.device`` is a known limitation (see T85's ticket notes):
    the two resolve independently, so a multi-GPU pin would need matching
    config on both today.

    Auto-detection deliberately excludes MPS (``mps_ok=False``, unlike the
    encoder's own ``resolve_device`` call): several kernels here (notably
    ``_prototypes_and_freq``'s class-sum accumulation) use float64
    intermediates for precision, and PyTorch's MPS backend does not support
    float64 at all -- it raises rather than silently downcasting. CUDA and
    CPU both support float64, so this only affects Apple Silicon hosts, and
    only the array backend's device choice; the encoder is unaffected and may
    still pick MPS for itself. An explicit ``device="mps"`` still works if a
    caller asks for it directly -- only the auto-probe skips it."""

    name = "torch"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = resolve_device(device, mps_ok=False)

    # ------------------------------------------------------------ conversion
    def _as_tensor(self, x: Any, dtype: Any = None) -> torch.Tensor:
        torch_dtype = _torch_dtype(dtype)
        if isinstance(x, torch.Tensor):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(np.asarray(x), device=self.device)
        if torch_dtype is not None and t.dtype != torch_dtype:
            t = t.to(torch_dtype)
        return t

    def _match_dtype(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.dtype != b.dtype:
            dt = torch.result_type(a, b)
            a, b = a.to(dt), b.to(dt)
        return a, b

    def asarray(self, x: Any, dtype: Optional[Any] = None) -> torch.Tensor:
        return self._as_tensor(x, dtype)

    def to_host(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def zeros(self, shape: Any, dtype: Any) -> torch.Tensor:
        return torch.zeros(shape, dtype=_torch_dtype(dtype), device=self.device)

    def full(self, shape: Any, value: Any, dtype: Any) -> torch.Tensor:
        size = shape if isinstance(shape, (tuple, list, torch.Size)) else (shape,)
        return torch.full(size, value, dtype=_torch_dtype(dtype), device=self.device)

    # ------------------------------------------------------------ elementwise
    def where(self, cond: Any, a: Any, b: Any) -> torch.Tensor:
        cond_t = self._as_tensor(cond).bool()
        a_t, b_t = self._match_dtype(self._as_tensor(a), self._as_tensor(b))
        return torch.where(cond_t, a_t, b_t)

    def isnan(self, x: Any) -> torch.Tensor:
        return torch.isnan(self._as_tensor(x))

    def isfinite(self, x: Any) -> torch.Tensor:
        return torch.isfinite(self._as_tensor(x))

    def maximum(self, a: Any, b: Any) -> torch.Tensor:
        a_t, b_t = self._match_dtype(self._as_tensor(a), self._as_tensor(b))
        return torch.maximum(a_t, b_t)

    def log1p(self, x: Any) -> torch.Tensor:
        t = self._as_tensor(x)
        if not t.is_floating_point():
            t = t.to(torch.float64)
        return torch.log1p(t)

    # ------------------------------------------------------------ reduction / ordering
    def matmul(self, a: Any, b: Any) -> torch.Tensor:
        a_t, b_t = self._match_dtype(self._as_tensor(a), self._as_tensor(b))
        return a_t @ b_t

    def topk(self, x: Any, k: int, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._as_tensor(x)
        k = min(k, t.shape[axis])
        vals, idx = torch.topk(t, k, dim=axis, largest=True, sorted=True)
        return vals, idx

    def argsort(self, x: Any, axis: int = -1) -> torch.Tensor:
        return torch.argsort(self._as_tensor(x), dim=axis)

    def argpartition(self, x: Any, k: int, axis: int = -1) -> torch.Tensor:
        """Torch has no native partition kernel; a full ``argsort`` trivially
        satisfies ``argpartition``'s contract (every element before position
        ``k`` is <= the element at ``k``, every element after is >=) for any
        valid ``k``, positive or negative. O(n log n) instead of O(n) -- a
        correctness-first choice, matching this port's stated scope: not a
        performance-tuned array-API reimplementation, only what the existing
        kernels need."""
        return torch.argsort(self._as_tensor(x), dim=axis)

    def _nanreduce(self, x: Any, axis: Optional[int], *, fill: float, reduce: str) -> torch.Tensor:
        """``torch.nanmin``/``nanmax`` are absent from some torch builds (this
        dev host's included), so implement both by hand: replace NaN with a
        sentinel that can't win the reduction, reduce, then restore NaN for
        any row/column that was all-NaN (a real reduction over an empty set,
        not a value)."""
        t = self._as_tensor(x)
        nan_mask = torch.isnan(t)
        filled = torch.where(nan_mask, torch.full_like(t, fill), t)
        reducer = torch.amin if reduce == "min" else torch.amax
        if axis is None:
            result = reducer(filled)
            all_nan = torch.all(nan_mask)
        else:
            result = reducer(filled, dim=axis)
            all_nan = torch.all(nan_mask, dim=axis)
        return torch.where(all_nan, torch.full_like(result, float("nan")), result)

    def nanmin(self, x: Any, axis: Optional[int] = None) -> torch.Tensor:
        return self._nanreduce(x, axis, fill=float("inf"), reduce="min")

    def nanmax(self, x: Any, axis: Optional[int] = None) -> torch.Tensor:
        return self._nanreduce(x, axis, fill=float("-inf"), reduce="max")

    # ------------------------------------------------------------ scatter / gather
    def scatter_add(self, target: Any, rows: Any, cols: Any, values: Any) -> torch.Tensor:
        target_t = self._as_tensor(target).clone()
        rows_t = self._as_tensor(rows, dtype=np.int64)
        if rows_t.numel() == 0:
            return target_t
        cols_t = self._as_tensor(cols, dtype=np.int64)
        values_t = self._as_tensor(values).to(target_t.dtype)
        target_t.index_put_((rows_t, cols_t), values_t, accumulate=True)
        return target_t

    def scatter_max(self, target: Any, rows: Any, cols: Any, values: Any) -> torch.Tensor:
        target_t = self._as_tensor(target).clone()
        rows_t = self._as_tensor(rows, dtype=np.int64)
        if rows_t.numel() == 0:
            return target_t
        cols_t = self._as_tensor(cols, dtype=np.int64)
        values_t = self._as_tensor(values).to(target_t.dtype)
        R, C = target_t.shape
        flat_idx = rows_t * C + cols_t
        flat_target = target_t.reshape(-1)
        flat_target.scatter_reduce_(0, flat_idx, values_t, reduce="amax", include_self=True)
        return flat_target.reshape(R, C)

    def gather(self, M: Any, rows: Any, cols: Any) -> torch.Tensor:
        M_t = self._as_tensor(M)
        rows_t = self._as_tensor(rows, dtype=np.int64)
        cols_t = self._as_tensor(cols, dtype=np.int64)
        rows_b, cols_b = torch.broadcast_tensors(rows_t, cols_t)
        return M_t[rows_b, cols_b]
