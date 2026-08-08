# CLAUDE.md

Guidance for agents working in this repo. Keep it short; edit when an invariant changes.

## What this is
Hybrid retrieval-fusion text classifier with calibrated abstention. Five retrieval
signals → ~28 features per (item, candidate) → pointwise XGBoost fusion → isotonic
calibration → tuned abstention threshold. Built for imbalanced data and air-gapped hosts.

Signals are computed in **two rounds** (T33). Round one is the five built-ins:
they score every class and their top-n form the candidate shortlist. Round two is
providers declaring `needs_candidates` — a cross-encoder reranker today — which
receive that shortlist and may only *reorder* it, never extend it. Candidate
recall therefore stays a property of round one alone. Round two is off by default.

## Architecture (hexagonal / DDD)
```
text_classifier/
  domain/         framework-free: models, ports (ABCs), services (policies, schema)
  infrastructure/ adapters: encoder (sentence-transformers), retrieval (BM25+dense), fusion (XGBoost), persistence
  application/    use cases: features, scoring, indexing, TrainingPipeline, InferencePipeline
  config.py       dataclasses (serialize to JSON alongside a model dir)
  _messages.py    layer-neutral error-message formatting (`format_preview`)
scripts/          train.py, infer.py (CLIs), demo.py (offline smoke test)
```
Dependency rule: `domain` imports no ML framework. `infrastructure` depends on `domain`.
`application` orchestrates through the ports. The two pipelines are the public entry points.

`application/indexing.py::RetrievalIndexBuilder` owns every once-per-run cache
(document embeddings, BM25 tokenization) and is the *only* place dense/lexical indices
are constructed during training — the fold loop and the deployment build both call
`build(encoder, rows)`. Add a retrieval optimization there, not at a call site.

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
  The container is not part of the invariant: it's numpy by default and for every encoder
  kind except `SentenceTransformerEncoder` with `array_backend="torch"` (T85), which
  returns a resident torch tensor instead — normalization survives either way.
- **`LabelSpace` owns the canonical key↔index map.** Column `c` always means `key_at(c)`.
- A trained **model directory must be portable** (numpy + json + native
  XGBoost/SentenceTransformer formats only; no pickle) — it ships to an air-gapped
  host, so loading it must never execute embedded code. Directories saved before
  this invariant still load via a legacy pickle fallback, with a warning.

## Commands
- Offline smoke test (no network, no torch): `python -m scripts.demo`
- Train: `python -m scripts.train --items items.csv --classes classes.csv --out model_dir/`
- Infer: `python -m scripts.infer --model model_dir/ --input new.csv --output preds.csv`
- Evaluate on a labeled set: `python -m text_classifier.cli.evaluate --model model_dir/ --input labeled.csv`
- Re-tune the operating point (no retrain): `python -m text_classifier.cli.tune --model model_dir/ --input fresh_labeled.csv --target-precision 0.97`
- Add classes/examples to a deployed model (no retrain): `python -m text_classifier.cli.update --model model_dir/ --out updated_dir/ --classes classes.csv --items new_items.csv`
- Feature importance + per-feature ablation report (no retrain): `python -m text_classifier.cli.importance --model model_dir/ --input labeled.csv`
- Tests: `pytest -q`
- After `pip install .`: console scripts `text-classifier-train` / `-infer` / `-eval` / `-tune` / `-update` / `-importance`. CLI logic lives in
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
