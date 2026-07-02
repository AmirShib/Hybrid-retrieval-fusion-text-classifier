# T27 — Validate the config at pipeline entry (n_folds ≥ 3 and friends)

status: todo
tier: 2
depends_on: T01

## Goal
Fail fast, with an actionable message, on config values that today produce silently
broken training runs or deep framework tracebacks. Centralize the checks in a
`PipelineConfig.validate()` called at the top of `TrainingPipeline.run`.

## Why
`TrainingConfig.fold_roles()` assigns the last fold to test, the second-last to
calibration, and *the rest* to fusion training (`config.py:70-73`). With
`n_folds=2` the training set is **empty** — the fusion model fits on zero rows or
dies inside xgboost with an unrelated-looking error; with `n_folds=1`,
`folds[-2]` silently reuses the test fold for calibration. Nothing guards this:
`TrainingPipeline._validate_inputs` (`application/training.py`) checks the *data*
against `n_folds` but never `n_folds` itself, and the train CLI passes `--folds`
straight through. T20 hardened the data boundary; this ticket hardens the config
boundary the same way.

## Design
Add `PipelineConfig.validate() -> None` in `config.py` (keeps the rule next to the
fields it constrains; dataclasses stay plain, no framework). Raise `ValueError`
with the offending field name, the received value, and the constraint. Checks:

- `training.n_folds >= 3` — fold roles need at least one train fold plus one
  calibration fold plus one test fold. Say exactly that in the message.
- `0 < training.target_precision <= 1`.
- `training.per_class_min_support >= 1`.
- `candidate_top_n >= 1`.
- `retrieval.k_neighbors >= 1`.
- `retrieval.dense_chunk >= 1` and `retrieval.feature_chunk >= 1` (a zero chunk
  makes the assembly loop spin forever).
- `encoder.encode_batch_size >= 1`.
- Registry-key existence for `encoder.kind` / `fusion.kind` / `calibration.kind` is
  already handled at build time with a good message — do NOT duplicate it here
  (validate() must not import the registry; config stays dependency-free).

Call sites:
- `TrainingPipeline.run` — first line of `_validate_inputs` (before any data work).
- `InferencePipeline` does not need it: a persisted model dir was written by a run
  that validated, and re-validating on load would break loading older dirs whose
  persisted config predates a new rule. Load-time compatibility stays governed by
  the feature-schema check (`persistence.py`).

## Files to change
- `text_classifier/config.py` — `PipelineConfig.validate()`.
- `text_classifier/application/training.py` — call it in `_validate_inputs`.
- `tests/unit/test_validation.py` — extend with config-validation cases.

## Tests
- [ ] `n_folds=2` and `n_folds=1` raise `ValueError` mentioning `n_folds` and the
      three fold roles; `n_folds=3` passes.
- [ ] Each numeric bound above: one failing value (raises, message names the field)
      and the boundary value (passes).
- [ ] `TrainingPipeline.run` with a bad config raises before touching the encoder
      (use a doubles-free assertion: e.g. an encoder stub whose `encode` fails the
      test if called).
- [ ] Default `PipelineConfig()` validates clean.

## Acceptance criteria
- [ ] Every check above is enforced with a message naming the field, the received
      value, and the constraint.
- [ ] No behaviour change for valid configs; the CLI surfaces the `ValueError`
      text as-is (existing `main()` behaviour is fine).
- [ ] `config.py` still imports nothing beyond stdlib/dataclasses.

## Out of scope
Cross-field data/config checks (per-class example counts vs `n_folds` already live
in `_validate_inputs`); a JSON-schema for config files (belongs to a future
`--config` CLI ticket); warnings for merely *suspicious* values (e.g. huge
`k_neighbors`).
