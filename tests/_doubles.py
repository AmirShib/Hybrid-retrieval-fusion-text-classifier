"""Shared test doubles for the text-classifier test suite.

``HashingEncoder`` and ``make_synthetic`` now live inside the package itself
(``text_classifier.infrastructure.encoder`` and ``text_classifier.datasets``),
so the offline demo and the installed package can use them without depending on
the test tree. This module re-exports them under their historical names so the
existing test imports keep working unchanged. The ``"hashing"`` encoder kind is
registered as a built-in by the package registry, so no registration happens
here.
"""

from __future__ import annotations

from text_classifier.datasets import make_synthetic
from text_classifier.infrastructure import HashingEncoder

__all__ = ["HashingEncoder", "make_synthetic"]
