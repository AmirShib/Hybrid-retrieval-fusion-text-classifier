#!/usr/bin/env python
"""Thin wrapper kept for `python -m scripts.infer` from a source checkout.

The real implementation lives in ``text_classifier.cli.infer`` and is installed
as the ``text-classifier-infer`` console script when the package is installed.
"""
from text_classifier.cli.infer import main

if __name__ == "__main__":
    main()
