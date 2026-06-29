# T51 — ruff + mypy + pre-commit; type-clean the package

status: todo
tier: 5
depends_on: —

## Goal
A statically-checked, consistently-formatted codebase that stays that way.

## Why
The package already ships `py.typed` (T60), promising consumers usable type
hints — but nothing verifies those hints are correct or that style stays
consistent. For a package meant to be maintained by others, lint + type gates are
table stakes.

## Scope
- Add `ruff` (lint + format) config to `pyproject.toml`; fix findings.
- Add `mypy` config and make the package type-clean (it is largely annotated
  already). Treat the lazy ML imports (xgboost/lightgbm/sentence-transformers)
  pragmatically (per-module ignores where stubs are missing).
- Add a `.pre-commit-config.yaml` running ruff + mypy.
- Add a CI `lint` job (ruff check + mypy) alongside the existing test jobs.

## Acceptance criteria
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] `mypy text_classifier` passes (with documented, minimal ignores).
- [ ] pre-commit config present; CI runs the lint/type gate.
- [ ] No runtime behavior change.
