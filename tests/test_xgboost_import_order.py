"""T85: the package must load xgboost before anything in it can import torch.

On macOS + Homebrew libomp, torch initialising its OpenMP runtime first makes
xgboost's first fit() segfault. A subprocess is used so the check sees a fresh
interpreter, not modules other tests already imported.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_package_loads_xgboost_first():
    code = (
        "import sys, text_classifier; "
        "mods = list(sys.modules); "
        "assert 'xgboost' in mods, 'xgboost not imported'; "
        "assert 'torch' not in mods or mods.index('xgboost') < mods.index('torch'), "
        "'torch was imported before xgboost'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_the_package_still_does_not_import_torch():
    """The guard must not undo T63: a torch-free install imports no torch."""
    code = "import sys, text_classifier; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
