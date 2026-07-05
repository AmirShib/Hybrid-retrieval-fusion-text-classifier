# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/): the version
lives in one place, `text_classifier/_version.py` (see `RELEASING.md`).

## [Unreleased]

### Added
- `--config`/`--dump-config` on the train CLI: every `PipelineConfig` field
  (fusion kind + hyperparameters, calibration kind, BM25/encoder kwargs, ...)
  is now reachable from the command line via a JSON file, with precedence
  `defaults < --config < explicit flags`. A trained model's `meta.json`
  `config` block is directly reusable as `--config` input.
- `--top-k` on the infer CLI and `InferencePipeline.predict_topk`:
  `Prediction.runner_up_key` is populated, and up to `k` ranked
  `(class_key, confidence)` suggestions are available per item for a human
  review queue, at no extra scoring cost over the existing top-1 pass.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and GitHub issue/PR
  templates.
- Single-sourced package version (`text_classifier/_version.py`); `pyproject.toml`
  reads it via `[tool.setuptools.dynamic]` instead of duplicating the string.

## [0.1.0] - 2026-07

Initial release.

### Added
- Hybrid retrieval-fusion architecture: five retrieval signals (dense kNN,
  BM25, description similarity, ...) assembled into per-(item, candidate)
  features, fused by a pointwise XGBoost model, isotonic-calibrated, with a
  tuned abstention threshold for target-precision serving.
- Hexagonal/DDD layering (`domain` / `infrastructure` / `application`) with a
  pluggable component registry: swappable encoder (sentence-transformers,
  TF-IDF, hashing), fusion model (XGBoost, LightGBM, XGBRanker), and
  calibrator (isotonic, Platt, beta) behind ports, selected purely by config.
- Leakage-free training and calibration: out-of-fold scoring throughout, with
  a regression test pinning the invariant.
- Console scripts `text-classifier-train`, `text-classifier-infer`,
  `text-classifier-eval`, each installed via `pip install .`.
- Persisted `evaluation.json` and `model_card.md` per trained model directory;
  package version recorded in `meta.json` and checked on load.
- Deterministic training (seeded fusion backends, config validation at
  pipeline entry) and an offline quality-regression benchmark with metric
  floors, run in CI.
- Hash-locked dependency pins (`requirements.lock`) for air-gapped,
  reproducible installs.
- ruff + mypy + pre-commit gates; a type-clean, `py.typed` package.
