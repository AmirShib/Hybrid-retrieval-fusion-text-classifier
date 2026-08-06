"""Array-backend port implementations (T84).

``NumpyArrayOps`` is a thin pass-through numpy backend -- the default, always
registered. A torch backend (T85, ``infrastructure/array_ops_torch.py``) is
also always *registered* here (metadata only, no import), but its
``ArrayOpsSpec.build`` defers ``import torch`` until it is actually called, so
an ordinary numpy-backend run never imports torch as a side effect of loading
this module. Every method except
``scatter_add``/``scatter_max`` is a direct 1:1 call onto ``numpy``, so routing
a kernel through the port is a zero-behaviour-change swap.

``scatter_add``/``scatter_max`` are the one place this backend does more than
delegate: they replace ``np.add.at``/``np.maximum.at`` (unbuffered, the
slowest scatter numpy offers) with vectorized equivalents -- ``np.bincount``
for the sum, a sort + ``np.maximum.reduceat`` for the max, since ``bincount``
itself has no max-reduction mode. Both are correct for arbitrary duplicate
indices, including rows that receive no updates at all.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from ..domain.ports import ArrayOps

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

    def argsort(self, x: Any, axis: int = -1) -> np.ndarray:
        return np.argsort(x, axis=axis)

    def argpartition(self, x: Any, k: int, axis: int = -1) -> np.ndarray:
        return np.argpartition(x, k, axis=axis)

    def nanmin(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.nanmin(x, axis=axis)

    def nanmax(self, x: Any, axis: Optional[int] = None) -> np.ndarray:
        return np.nanmax(x, axis=axis)

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

    def gather(self, M: Any, rows: Any, cols: Any) -> np.ndarray:
        return np.asarray(M)[rows, cols]


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


def resolve_array_backend(
    explicit: Optional[str],
    *,
    n_items: int,
    n_classes: int,
    k_neighbors: Optional[int] = None,
    device_visible: Optional[bool] = None,
) -> str:
    """Resolve the array backend for a run of this scale.

    An explicit ``kind`` (anything other than ``None``/``"auto"``) always
    wins -- the same explicit-beats-detected rule ``resolve_device`` already
    follows. Otherwise: numpy unless the scale meets or exceeds
    ``CROSSOVER_MIN_ITEMS``/``CROSSOVER_MIN_CLASSES``, torch is actually
    installed, and a device is visible. The choice and the reason are logged
    -- this is meant to be traceable, not silent.

    **Order matters, and is deliberate.** The crossover check runs *before*
    ``torch_installed()``/``cuda_available()`` -- not after -- so a small run
    never even asks whether torch is installed or a device is visible. This
    is not just an optimization: ``cuda_available()`` does a real
    ``import torch``, and on a host where torch happens to be installed for
    an unrelated reason (e.g. this repo's dev environment, where the
    ``sentence-transformers`` extra pulls it in), skipping straight to
    ``torch_installed()`` first would make *every* numpy-backend run -- even
    a handful of items -- pay that import, reintroducing exactly the T63
    boundary violation T84 already hit once (colliding with xgboost's bundled
    OpenMP runtime badly enough to segfault). Gating on scale first means an
    ordinary small run stays torch-free regardless of what else happens to be
    installed alongside this package."""
    if explicit is not None and explicit != "auto":
        logger.info("array backend: %r (explicit)", explicit)
        return explicit

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

    # Cheap (no import) before any real torch probe -- see the docstring.
    from .device import torch_installed

    if not torch_installed():
        logger.info("array backend: numpy (auto: torch not installed)")
        return "numpy"

    if device_visible is None:
        from .device import cuda_available

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
