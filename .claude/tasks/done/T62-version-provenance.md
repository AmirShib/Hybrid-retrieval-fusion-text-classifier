# T62 — Package version provenance

status: done
tier: 6
depends_on: T07

## Goal
Make a trained model self-describing about the code version that produced it.

## Why
A model directory is shipped to an air-gapped host and may outlive the checkout
that produced it. The feature schema is already version-guarded (persistence
raises on drift), but there was no record of the *package* version, so an
operator could not tell which code trained a given model.

## What was done
- `text_classifier/_version.py` resolves `__version__` from installed
  distribution metadata, falling back to the in-tree default for a non-installed
  source tree; re-exported as `text_classifier.__version__`.
- `persistence.save` records `package_version` in `meta.json`; `load` emits a
  soft warning (does not raise) on a version skew, since on-disk compatibility is
  governed by the separate feature-schema check.
- The evaluation manifest (T61) also records the package version.

## Acceptance criteria
- [x] `text_classifier.__version__` is importable.
- [x] `meta.json` records `package_version`.
- [x] Loading a model trained on a different version warns rather than failing.
