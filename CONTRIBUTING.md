# Contributing

## Dev setup

```bash
git clone <repo-url>
cd Hybrid-retrieval-fusion-text-classifier
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
pre-commit install
```

## Running the checks locally

```bash
pytest -q                                              # tests
ruff check text_classifier/ scripts/ tests/            # lint
ruff format --check text_classifier/ scripts/ tests/   # formatting
mypy text_classifier                                   # types
```

`pre-commit install` wires ruff + mypy into your local `git commit`; CI
(`.github/workflows/ci.yml`) runs the same checks plus the quality-regression
benchmark (`pytest -m quality`) and a wheel-install smoke test. All of this
must run **offline** — tests use the `HashingEncoder`/`tfidf` backends rather
than downloading a real sentence-transformers model; don't add a test that
needs network access.

## Code conventions

This is a hexagonal/DDD codebase: `text_classifier/domain/` is framework-free
(models, ports, services), `text_classifier/infrastructure/` holds the
concrete adapters (encoder, retrieval, fusion, persistence), and
`text_classifier/application/` orchestrates use cases through the domain
ports. A few invariants are load-bearing and any change touching them needs
to preserve them explicitly, not incidentally:

- `NaN` means "signal did not retrieve this class" and is distinct from a
  true `0` — never impute it away.
- `domain/services.py::FEATURE_NAMES` is the single source of truth for
  feature-column order; adding/removing/reordering a feature touches it,
  `application/features.py`, and the persisted `meta.json` schema together.
- An item's features must be scored against indices/prototypes built from
  *other* folds only (no leakage), and embeddings are L2-normalized so dot
  product equals cosine similarity.
- A trained model directory must stay portable (stdlib pickle + numpy + json
  + native XGBoost/SentenceTransformer formats only) so it can ship to an
  air-gapped host.

## Commits and pull requests

- Keep commits focused; write commit messages that explain *why*, not just
  *what* (the diff already shows what changed).
- A PR should leave `pytest -q`, `ruff check`, `ruff format --check`, and
  `mypy` green.
- Add a `CHANGELOG.md` entry under `[Unreleased]` for user-facing changes
  (new CLI flags, new config fields, behavior changes) — see `RELEASING.md`
  for how that turns into a release.

## Reporting bugs / requesting features

Use the GitHub issue templates (`.github/ISSUE_TEMPLATE/`). For security
issues, see `SECURITY.md` instead of opening a public issue.
