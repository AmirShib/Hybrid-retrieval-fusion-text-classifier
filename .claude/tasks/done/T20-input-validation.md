# T20 — Input validation: labels ∈ classes, empty/dup keys, empty text, clear errors

status: done
tier: 2
depends_on: T01, T07

## Goal
Add and test explicit validation at every public entry point so that bad inputs
produce a clear, actionable `ValueError` (or `KeyError`) rather than a cryptic
numpy/pandas traceback deep inside the pipeline.

## Why
Right now a typo in a label key, a duplicate class definition, or an empty text
silently corrupts features or raises an opaque internal error.  Adding boundary
validation makes the package usable without reading source code.

## Files to add/change
- `text_classifier/domain/models.py` — `LabelSpace.__init__`, `LabeledItem`
- `text_classifier/application/training.py` — `TrainingPipeline.run` entry point
- `text_classifier/application/inference.py` — `InferencePipeline.predict`
- `tests/unit/test_validation.py` — new test file

## Part A — `LabelSpace` construction

- [ ] **Duplicate keys:** `LabelSpace([ClassDefinition("A","d"), ClassDefinition("A","d2")])`
      → `ValueError` mentioning the duplicate key.
- [ ] **Empty definitions list:** `LabelSpace([])` → `ValueError` (already raises; assert
      the message is human-readable).
- [ ] **Empty key string:** `ClassDefinition(key="", description="x")` used in `LabelSpace`
      → `ValueError` mentioning that keys must be non-empty.
- [ ] **Empty description string:** decide and document policy (warn or raise); add a test
      that pins the chosen behaviour.

## Part B — `LabeledItem` / training inputs

- [ ] **Label not in LabelSpace:** passing `LabeledItem(text="x", label="UNKNOWN")` to
      `TrainingPipeline.run` → `KeyError` or `ValueError` before any GPU/index work begins.
- [ ] **Empty items list:** `run([], label_space)` → `ValueError`.
- [ ] **Empty text:** `LabeledItem(text="", label="A")` in the training set → either a
      clear `ValueError` (recommended) or documented pass-through; either way tested.
- [ ] **All items belong to one class:** acceptable input (not an error), but must not
      crash `StratifiedKFold`. Document and test the minimum-items-per-class invariant.

## Part C — `InferencePipeline.predict`

- [ ] **Empty string in input:** `predict([""])` → returns a `Prediction` (not a crash);
      document whether it abstains or not.
- [ ] **Non-string input:** `predict([42])` → `ValueError` or `TypeError` before encoding.
- [ ] **None in input:** `predict([None])` → same as above.

## Part D — `ArtifactRepository`

- [ ] **Missing file:** `ArtifactRepository().load("/nonexistent/path")` → clear
      `FileNotFoundError` with the path in the message (not a KeyError from `json.load`).
- [ ] **Corrupt meta.json:** if `feature_names` in meta differs from current `FEATURE_NAMES`,
      raise `ValueError` listing the mismatch (schema drift guard).

## Acceptance criteria
- [ ] Every public entry point has at least one validation test for each failure mode above.
- [ ] All raised errors include the offending value in the message (not just "invalid input").
- [ ] `tests/unit/test_validation.py` is self-contained and runs offline.
- [ ] No regression in the existing 147-test suite.

## Out of scope
Validating the contents of `PipelineConfig` (that's T51 territory). Per-field
type annotations / mypy (T51).
