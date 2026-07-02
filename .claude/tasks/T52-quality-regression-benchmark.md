# T52 — Offline quality-regression benchmark in CI (metric floors)

status: todo
tier: 5
depends_on: T01, T61

## Goal
Add a small, fully offline benchmark that trains on a fixed synthetic dataset and
asserts the headline metrics stay above known floors, wired into CI as its own job.
It catches the class of regression the unit suite cannot: a change that is *correct
code* but *worse model* (a feature computed subtly differently, a retrieval change
that dents candidate recall, a calibration change that craters coverage).

## Why
The test suite proves behaviour (shapes, round-trips, leakage, edge cases) but
nothing anywhere asserts the system still *classifies well*. The ingredients
already exist: `text_classifier.datasets.make_synthetic` (deterministic, seeded),
the torch-free `hashing` and `tfidf` encoders, and `TrainingPipeline.run` returning
a `CoverageReport` with `candidate_recall`, `coverage`, and
`accuracy_on_accepted`. T30's vectorization, T31's FAISS backend, and T33's
cross-encoder signal are all upcoming changes with exactly this regression risk —
land the net before them.

## Design
- New test module `tests/quality/test_benchmark.py`, marked
  `@pytest.mark.quality` (register the marker in the pytest config). Excluded from
  the default `pytest -q` run via `addopts`-free approach: the CI test job runs
  `pytest -q -m "not quality"`, the benchmark job runs `pytest -q -m quality`.
  Locally, `pytest -q` keeps running everything — cheap enough (seconds, no torch).
- One parametrized case per encoder kind (`hashing`, `tfidf`): generate
  `make_synthetic(n_classes=20, per_class=30, seed=7)` (size chosen so the run
  stays under ~30s), train with a fixed `PipelineConfig` (small `n_estimators`,
  `n_folds=4`), and assert on the returned `CoverageReport`:
  - `candidate_recall >= FLOOR_RECALL`
  - `accuracy_on_accepted >= FLOOR_ACC`
  - `coverage >= FLOOR_COV`
- Floors are constants at the top of the module with a comment stating the value
  observed when the floor was set. Set each floor with a real margin below the
  observed value (e.g. observed 0.97 → floor 0.90): the job must page on
  regressions, not on noise. Requires T26 (seeded fusion) to have landed or be
  included, otherwise the margin has to absorb run-to-run variance too.
- CI (`.github/workflows/ci.yml`): a `quality` job, single Python version, installs
  the same torch-free subset as `package-smoke` and runs `pytest -q -m quality`.
- Also assert the training run *writes* `evaluation.json` whose
  `overall.candidate_recall` matches the report (cheap cross-check tying T61's
  persisted numbers to the in-memory report).

## Files to add/change
- `tests/quality/__init__.py`, `tests/quality/test_benchmark.py` — the benchmark.
- `pyproject.toml` — register the `quality` pytest marker.
- `.github/workflows/ci.yml` — add the job; scope the existing test job to
  `-m "not quality"`.

## Tests
The ticket *is* tests; meta-checks:
- [ ] Benchmark passes on current `dev` for both encoder kinds.
- [ ] Artificially breaking a feature (e.g. locally zeroing `d_desc_sim` in the
      assembler) makes at least one floor assertion fail — evidence the net can
      actually catch a real regression. Record which floor tripped in the PR
      description; do not commit the sabotage.
- [ ] Full run (`pytest -q`) still passes and stays fast enough for local use.

## Acceptance criteria
- [ ] CI has a distinct `quality` job whose failure message shows metric vs floor.
- [ ] Runs offline: no network, no torch, no model download.
- [ ] Floors documented in-module with the observed values they were derived from.

## Out of scope
Benchmarking on real datasets (CLINC150 needs a download — CI must stay offline;
a manually-run `examples/` benchmark script can be a follow-up); performance/speed
benchmarks (T32 covers memory profiling); tracking metrics over time (just floors).
