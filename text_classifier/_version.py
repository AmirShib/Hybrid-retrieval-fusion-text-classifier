"""Single source of truth for the package version.

``pyproject.toml`` declares ``version`` as dynamic and reads this literal via
``[tool.setuptools.dynamic]`` (``attr = "text_classifier._version.__version__"``),
so a release is a one-line bump here — nothing else needs to change or can
drift out of sync.
"""

from __future__ import annotations

__version__ = "0.1.1"

__all__ = ["__version__"]
