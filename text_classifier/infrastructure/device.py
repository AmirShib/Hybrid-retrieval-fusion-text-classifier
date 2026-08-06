"""GPU/CPU device resolution, shared by the encoder and fusion adapters.

Detection is best-effort and torch-free at import time: ``sentence-transformers``
already pulls in torch, but the torch-free encoders (``TfidfEncoder``,
``HashingEncoder``) and CPU-only hosts must keep working with no GPU library
installed at all, so the probe is wrapped in ``try/except`` rather than assumed
to succeed.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def torch_installed() -> bool:
    """Whether torch is importable, without actually importing it.

    ``importlib.util.find_spec`` only does import-system bookkeeping (locates
    the module) — it does not execute torch's ``__init__.py``, so it never
    loads torch's C extensions or its bundled OpenMP runtime. That matters:
    ``cuda_available()``/``mps_available()`` do a real ``import torch``, and
    T84 found that colliding with xgboost's own OpenMP runtime can segfault
    (see ``infrastructure/array_ops.py::resolve_array_backend``). This probe
    is the cheap, safe check callers use *before* deciding whether a real
    import (and its side effects) is warranted at all. ``False`` (not raise)
    if the check itself fails for any reason -- best-effort, like the other
    probes in this module."""
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def cuda_available() -> bool:
    """Whether torch can see a CUDA device. ``False`` (not raise) if torch is
    absent or the check itself fails for any reason -- this is a best-effort
    probe, never a hard requirement."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    """Whether torch can see an Apple Silicon (Metal/MPS) GPU. ``False`` (not
    raise) if torch is absent, this isn't an Apple Silicon host, or the check
    itself fails for any reason -- same best-effort contract as
    ``cuda_available``."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(explicit: Optional[str] = None, *, mps_ok: bool = False) -> str:
    """Return ``explicit`` if given (an explicit user choice always wins),
    otherwise ``"cuda"`` if a CUDA GPU is visible, else ``"mps"`` if
    ``mps_ok`` and an Apple Silicon GPU is visible, else ``"cpu"``.

    ``mps_ok`` defaults to ``False``: most of this package's GPU-consuming
    device params (XGBoost's and LightGBM's ``device=`` -- see
    ``infrastructure/fusion.py``) only understand ``"cuda"``/``"cpu"`` and
    would error on ``"mps"`` rather than fall back, so those call sites must
    keep the pre-MPS-aware CUDA-or-CPU behaviour unchanged. Pass
    ``mps_ok=True`` only from a consumer that actually accepts ``"mps"``
    (torch/``SentenceTransformer``)."""
    if explicit is not None:
        return explicit
    if cuda_available():
        return "cuda"
    if mps_ok and mps_available():
        return "mps"
    return "cpu"
