#!/usr/bin/env python
"""Thin wrapper kept for `python -m scripts.train` from a source checkout.

The real implementation lives in ``text_classifier.cli.train`` and is installed
as the ``text-classifier-train`` console script when the package is installed.
"""
from text_classifier.cli.train import main

if __name__ == "__main__":
    main()
