"""GPU/CPU device resolution, shared by the encoder and fusion adapters.

Detection is best-effort and torch-free at import time: ``sentence-transformers``
already pulls in torch, but the torch-free encoders (``TfidfEncoder``,
``HashingEncoder``) and CPU-only hosts must keep working with no GPU library
installed at all, so the probe is wrapped in ``try/except`` rather than assumed
to succeed.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cuda_available() -> bool:
    """Whether torch can see a CUDA device. ``False`` (not raise) if torch is
    absent or the check itself fails for any reason -- this is a best-effort
    probe, never a hard requirement."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(explicit: Optional[str] = None) -> str:
    """Return ``explicit`` if given (an explicit user choice always wins),
    otherwise ``"cuda"`` if a GPU is visible, else ``"cpu"``."""
    if explicit is not None:
        return explicit
    return "cuda" if cuda_available() else "cpu"
