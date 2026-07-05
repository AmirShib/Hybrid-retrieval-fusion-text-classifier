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

## Working on a ticket

Open work lives in `.claude/tasks/`; `.claude/tasks/INDEX.md` is the backlog
and priority order. To pick one up:

1. Read the ticket file top to bottom — each is self-contained (goal, design,
   files to change, tests, acceptance criteria).
2. Set its `status:` field to `in-progress`.
3. Implement, keeping to the ticket's stated scope — the "Out of scope"
   section is there to stop drift into adjacent, unrelated work.
4. Add/update tests per the ticket's "Tests" section; all suites (`pytest -q`)
   and lint/type gates must stay green.
5. When done: set `status: done`, move the file into `.claude/tasks/done/`,
   and update its row in `INDEX.md`'s table (the row stays for history).

## Code conventions

See `CLAUDE.md` for the architectural invariants (hexagonal layering, the
NaN-means-"no retrieval" contract, out-of-fold leakage rules, feature-column
ordering, model-directory portability) — these are load-bearing and any
change touching them needs to preserve them explicitly, not incidentally.

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
