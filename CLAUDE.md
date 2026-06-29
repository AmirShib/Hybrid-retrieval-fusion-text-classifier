# CLAUDE.md

Guidance for agents working in this repo. Keep it short; edit when an invariant changes.

## What this is
Hybrid retrieval-fusion text classifier with calibrated abstention. Five retrieval
signals → ~28 features per (item, candidate) → pointwise XGBoost fusion → isotonic
calibration → tuned abstention threshold. Built for imbalanced data and air-gapped hosts.

## Architecture (hexagonal / DDD)
```
text_classifier/
  domain/         framework-free: models, ports (ABCs), services (policies, schema)
  infrastructure/ adapters: encoder (sentence-transformers), retrieval (BM25+dense), fusion (XGBoost), persistence
  application/    use cases: features, scoring, TrainingPipeline, InferencePipeline
  config.py       dataclasses (serialize to JSON alongside a model dir)
scripts/          train.py, infer.py (CLIs), demo.py (offline smoke test)
```
Dependency rule: `domain` imports no ML framework. `infrastructure` depends on `domain`.
`application` orchestrates through the ports. The two pipelines are the public entry points.

## Invariants — do not break these
- **`NaN` means "signal did not retrieve this class."** It is distinct from a true 0.
  XGBoost consumes NaN natively as "missing"; never impute it away.
- **`domain/services.py::FEATURE_NAMES` is the single source of truth for column order.**
  Every producer/consumer references it. Adding/removing/reordering a feature touches it
  *and* `application/features.py` *and* the persisted `meta.json` schema.
- **Out-of-fold leakage rule:** an item's features must be scored against indices/prototypes
  built from *other* folds only. Calibration and the coverage report come from folds the
  fusion model never trained on. Never let an item see itself in its own index.
- **Embeddings are L2-normalized** so dot product == cosine. Encoders must preserve this.
- **`LabelSpace` owns the canonical key↔index map.** Column `c` always means `key_at(c)`.
- A trained **model directory must be portable** (stdlib pickle + numpy + json + native
  XGBoost/SentenceTransformer formats only) — it ships to an air-gapped host.

## Commands
- Offline smoke test (no network, no torch): `python -m scripts.demo`
- Train: `python -m scripts.train --items items.csv --classes classes.csv --out model_dir/`
- Infer: `python -m scripts.infer --model model_dir/ --input new.csv --output preds.csv`
- Evaluate on a labeled set: `python -m text_classifier.cli.evaluate --model model_dir/ --input labeled.csv`
- Tests: `pytest -q`
- After `pip install .`: console scripts `text-classifier-train` / `-infer` / `-eval`. CLI logic lives in
  `text_classifier/cli/`; `scripts/*.py` are thin dev wrappers. Training writes `evaluation.json` +
  `model_card.md` into the model dir.

## Conventions
- Keep numerics vectorized: signals are `(batch, n_classes)` matrices; gather candidate rows
  with fancy indexing. No per-row Python loops on the hot path.
- New retrieval backend, fusion model, or calibrator → implement the matching port in
  `domain/ports.py` under `infrastructure/`; don't reach around the port.
- Tests must run **offline** — use the `HashingEncoder` double, never download a model in CI.

## Working on tasks
Open tickets live in `.claude/tasks/`; see `.claude/tasks/INDEX.md` for the backlog and
status. Pick one up by reading its file top-to-bottom — each is self-contained.
