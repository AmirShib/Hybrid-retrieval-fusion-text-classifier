"""``infrastructure/device.py`` -- device resolution, including the
Apple Silicon (MPS) vs. CUDA distinction. ``mps_ok`` must default to ``False``
so XGBoost/LightGBM's ``device=`` (which only understands ``"cuda"``/``"cpu"``)
never receives ``"mps"`` from ``resolve_device``'s auto-detect path."""

from __future__ import annotations

from text_classifier.infrastructure.device import mps_available, resolve_device


def test_explicit_device_always_wins(monkeypatch):
    monkeypatch.setattr("text_classifier.infrastructure.device.cuda_available", lambda: True)
    monkeypatch.setattr("text_classifier.infrastructure.device.mps_available", lambda: True)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cpu", mps_ok=True) == "cpu"


def test_auto_detect_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr("text_classifier.infrastructure.device.cuda_available", lambda: True)
    monkeypatch.setattr("text_classifier.infrastructure.device.mps_available", lambda: True)
    assert resolve_device(mps_ok=True) == "cuda"


def test_mps_only_returned_when_mps_ok_and_no_cuda(monkeypatch):
    monkeypatch.setattr("text_classifier.infrastructure.device.cuda_available", lambda: False)
    monkeypatch.setattr("text_classifier.infrastructure.device.mps_available", lambda: True)
    assert resolve_device(mps_ok=True) == "mps"
    # mps_ok defaults False -- the fusion (XGBoost/LightGBM) call sites' contract.
    assert resolve_device() == "cpu"


def test_no_device_visible_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("text_classifier.infrastructure.device.cuda_available", lambda: False)
    monkeypatch.setattr("text_classifier.infrastructure.device.mps_available", lambda: False)
    assert resolve_device(mps_ok=True) == "cpu"
    assert resolve_device() == "cpu"


def test_mps_available_is_best_effort(monkeypatch):
    """Absent/broken torch must yield False, not raise (same contract as
    cuda_available)."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`
    assert mps_available() is False
