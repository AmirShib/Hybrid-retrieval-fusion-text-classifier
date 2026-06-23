# T01 — Test harness, fixtures, CI, determinism

status: todo
tier: 1
depends_on: none

## Goal
Stand up the test infrastructure everything else builds on: pytest layout, shared
fixtures (a deterministic offline encoder + synthetic dataset), a CI workflow that runs
offline, and a determinism contract so numeric tests are reproducible.

## Why
There is currently no test framework, no fixtures, and `scripts/demo.py` asserts nothing.
Every other Tier 1 ticket needs a synthetic dataset and an offline encoder double. CI must
never download a model (air-gapped ethos + reliability).

## Scope / deliverables
1. **Dependencies**
   - Add a dev/test extra to `pyproject.toml`:
     `[project.optional-dependencies] test = ["pytest", "pytest-cov"]`.
   - Do **not** add a hard dep on `sentence-transformers`/`torch` to the test path.
2. **Layout**
   ```
   tests/
     conftest.py          # shared fixtures
     unit/                # T02–T05 land here
     integration/         # T06–T07 land here
   ```
3. **`tests/conftest.py` fixtures**
   - `hashing_encoder` — return a `HashingEncoder` test double. **Lift the class out of
     `scripts/demo.py` into `tests/conftest.py` (or `tests/_doubles.py`) and import it in
     both places** so there is one definition. It implements the `TextEncoder` port and
     returns L2-normalized float32 embeddings. (Note: it currently uses builtin `hash()`,
     which is salted per-process — T21 will replace it with `hashlib`. For T01, pin
     `PYTHONHASHSEED=0` in CI as a stopgap and leave a `# TODO(T21)` comment.)
   - `synthetic_dataset` — parametrizable factory returning `(LabelSpace, list[LabeledItem])`.
     Reuse `make_synthetic` from `scripts/demo.py` (also lift it into the test helpers).
     Default: ~40 classes, deliberate class imbalance, fixed `seed`.
   - `tiny_label_space` — a hand-built 3-class `LabelSpace` for exact-value assertions.
4. **Determinism contract**
   - All fixtures take an explicit `seed`; default seeded.
   - Document in `tests/conftest.py` the rule: *unit tests assert exact values where the
     math is deterministic; integration tests assert bounds/invariants, not exact floats.*
5. **CI** — `.github/workflows/ci.yml`
   - Triggers: push + pull_request.
   - Matrix: Python 3.10 and 3.11.
   - Steps: checkout → setup-python → `pip install -e .[test]` → `pytest -q --cov=text_classifier`.
   - Set `env: PYTHONHASHSEED: "0"` until T21.
   - Must pass with **no network access** (no model download).
6. **Smoke gate** — one trivial test `tests/test_imports.py` that imports the public API
   from `text_classifier` and asserts `__all__` is importable. Proves CI wiring works.

## Acceptance criteria
- [ ] `pip install -e .[test] && pytest -q` runs green locally with no network.
- [ ] `HashingEncoder` and `make_synthetic` have a single definition, imported by both
      `scripts/demo.py` and the tests (no copy-paste divergence).
- [ ] CI workflow runs on push/PR, offline, on 3.10 + 3.11, and reports coverage.
- [ ] `tests/conftest.py` exposes `hashing_encoder`, `synthetic_dataset`, `tiny_label_space`.

## Out of scope
Actual unit assertions for domain/features/retrieval (T02–T07). Replacing `hash()` (T21).

## Notes for the implementer
- Keep CI offline-safe: if any test imports `sentence_transformers`, it must be marked
  `@pytest.mark.skipif` on import failure — but Tier 1 should not need it at all.
