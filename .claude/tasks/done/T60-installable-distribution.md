# T60 — Installable distribution + CLI ergonomics

status: done
tier: 6
depends_on: T23, T24, T25

## Goal
Make the package usable after `pip install` with no source checkout, and make the
CLIs pleasant and safe to drive.

## Why
Previously the CLIs lived only in `scripts/`, which the wheel did not ship, and
`scripts/demo.py` imported from `tests/`, which the wheel also did not ship. So
`pip install` produced a package whose documented commands did not exist. A
domain expert could not install it and use it — the core promise of the package.

## What was done
- **Console entry points** in `pyproject.toml` (`[project.scripts]`):
  `text-classifier-train`, `text-classifier-infer`, `text-classifier-eval`.
  CLI logic moved into `text_classifier/cli/` (`train.py`, `infer.py`,
  `evaluate.py`, `_common.py`); `scripts/*.py` are thin dev wrappers that import
  the package entry points so `python -m scripts.train` still works.
- **Offline pieces moved into the package**: `HashingEncoder` lives in
  `infrastructure/encoder.py` and is registered as the built-in `"hashing"`
  encoder kind; `make_synthetic` lives in `text_classifier/datasets.py`. The
  demo, CI, and air-gapped smoke tests no longer depend on the test tree.
  `tests/_doubles.py` re-exports both for backwards compatibility.
- **Friendly CSV ingestion** (`cli/_common.py`): clear, single-line errors for a
  missing/renamed column or a malformed value, plus `--text-col`, `--label-col`,
  `--key-col`, `--desc-col` flags on train and `--text-col`/`--label-col` on the
  others. `--log-level` on every CLI.
- **PEP 561**: ship `text_classifier/py.typed` so consumers get the type hints.
- **Dependency version floors** in `pyproject.toml` (hash-pinned lockfile is
  T50).
- **CI `package-smoke` job**: build the wheel, install it into a clean
  environment, and drive all three console scripts end-to-end via the torch-free
  `hashing` encoder. This is the regression guard that would have caught the
  original "CLIs not shipped" gap.

## Acceptance criteria
- [x] `pip install .` exposes `text-classifier-{train,infer,eval}`.
- [x] The built wheel contains `text_classifier/cli/*`, `datasets.py`, and
      `py.typed`, and excludes `scripts/` and `tests/`.
- [x] `python -m scripts.demo` runs without importing from `tests/`.
- [x] A missing CSV column produces a clear error, not a deep traceback.
- [x] CI builds the wheel and runs the console scripts end-to-end.
