# T64 — Release process + project hygiene

status: todo
tier: 6
depends_on: T60

## Goal
The repository scaffolding a professional consumer expects before depending on
the package: a changelog, contribution guidance, a security policy, and a
repeatable release.

## Why
The package is now installable (T60), but there is no record of what changes
between versions, no contribution/issue guidance, and no defined way to cut a
release. These are what let an outside team adopt, upgrade, and report against
the package with confidence.

## Scope
- `CHANGELOG.md` (Keep a Changelog style); seed it with the current state.
- `CONTRIBUTING.md` (dev setup, `pytest -q`, lint/type gate once T51 lands,
  ticket workflow in `.claude/tasks/`).
- `SECURITY.md` and GitHub issue/PR templates.
- Single-source the version: derive `pyproject` `version` and
  `text_classifier.__version__` from one place (e.g. `setuptools-scm` or a
  `__about__`), and document the tag → build → publish flow.
- Optional: a release workflow that builds the wheel/sdist and attaches them to a
  GitHub release (publishing to an index is a separate decision).

## Acceptance criteria
- [ ] CHANGELOG, CONTRIBUTING, SECURITY, and issue/PR templates exist.
- [ ] Version is single-sourced; bumping it is documented.
- [ ] A documented (ideally automated) build-and-release path produces the
      wheel + sdist.
