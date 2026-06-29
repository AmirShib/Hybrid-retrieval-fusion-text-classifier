"""Single source of truth for the installed package version.

Read from the installed distribution metadata so the version is never
duplicated between ``pyproject.toml`` and the source. Falls back to the
in-tree default when the package is imported from a source checkout that was
never installed (e.g. ``PYTHONPATH=.`` during development).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("text-classifier")
except PackageNotFoundError:  # running from a non-installed source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
