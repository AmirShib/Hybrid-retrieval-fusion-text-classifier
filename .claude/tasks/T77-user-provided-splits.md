# T77 — User-provided validation/test splits (bring your own val, test, or both)

status: todo
tier: 7
depends_on: T61

## Goal
Let a user who already has a train/val/test split hand those sets to the training
pipeline directly, instead of having `TrainingPipeline` re-derive everything from
one pooled `items` list via `StratifiedKFold`. Either external set is optional and
independent: supply just a validation set (used for calibration + threshold
tuning), just a test set (used for the held-out evaluation), or both. CLI surface:
`--val-items val.csv` and/or `--test-items test.csv` on the train CLI.

## Why
Today `fold_roles()` hardcodes "second-to-last fold = calibration, last fold =
test", and the CLI accepts a single `--items` file. Users arriving with a
predefined split (a frozen benchmark test set, a temporally-later validation set,
a split shared across model families for comparability) have no supported path:
their only options are pooling everything (destroys the split's meaning — e.g. a
time-based split becomes a random one) or holding the test set outside the tool
entirely and losing the persisted `evaluation.json`/`model_card.md` evidence
chain. Temporal splits are the important case: calibrating on a *later* slice is
exactly the drift-realistic operating point the internal random folds cannot
express.

## Design
**Semantics — external sets replace fold roles, they don't join the pool:**
- External **val** provided → the calibration fold role is retired; that fold
  joins the fusion-training folds. Calibrator + thresholds are fit on the
  external val set instead.
- External **test** provided → the test fold role is retired; that fold joins
  the fusion-training folds. `_evaluate` runs on the external test set instead.
- Both provided → all `n_folds` folds train the fusion model. OOF assembly is
  still required for the *training* items (the fusion model's own training rows
  must stay leakage-free), so the k-fold machinery stays; only the role
  assignment changes.
- With both external sets, relax the `n_folds >= 3` validation floor to
  `>= 2` (still need ≥2 folds for OOF feature generation); with exactly one
  external set the floor is also `>= 2` (one role retired). Encode this in
  `PipelineConfig.validate` or the pipeline entry, with a clear message.

**Featurization of external sets:** encode + assemble external val/test items
against indices built from **all training items** (the same construction as the
deployed index in `_build_deployment` — build it once, before deployment reuses
it). External items are absent from that index, so there is no self-match; and
scoring them against the full-train index matches the production condition better
than any within-fold index does. Do NOT route external items through the OOF
fold loop.

**Ordering note:** calibrating on full-train-index features while the fusion
model was fit on per-fold-index features is a deliberate, documented asymmetry —
it anchors confidence at the production operating point. Mention it in the
docstring; the risk-coverage numbers in `evaluation.json` then describe deployed
behaviour, which is the point.

**API** — extend the application layer, keep the old signature working:
`TrainingPipeline.run(items, label_space, output_dir=None, *,
val_items: Optional[Sequence[LabeledItem]] = None,
test_items: Optional[Sequence[LabeledItem]] = None)`.
Internally: `fold_roles()` grows the ability to express "no calibration fold" /
"no test fold" (return empty lists for retired roles rather than a new type).

**Validation (fail fast, same spirit as `_validate_inputs`):**
- Every external-set label must exist in the `LabelSpace`.
- Exact-text overlap between train and external sets → hard error naming the
  count (an overlapping item sits in the deployed index and self-retrieves,
  silently inflating calibration/eval — same trap T66 warns about).
- Overlap between val and test → warn (defensible in some workflows, but the
  user should know).
- External val must contain ≥ 2 distinct outcomes' worth of decisions for the
  parametric calibrators to fit (the existing single-class fallback covers the
  degenerate case; just document it).
- `per_class_min_support` applies to the external val when tuning per-class
  thresholds, unchanged.

**CLI** — `cli/train.py`: `--val-items PATH` and `--test-items PATH`, both
optional, reusing `--text-col/--label-col` for their schema. `read_items`
already does the parsing — reuse it. Record in the manifest
(`build_manifest`) that external splits were used and their sizes, so a model
dir is auditable: `"splits": {"val": "external:n=1234", "test": "internal-fold"}`
or similar.

## Files to add/change
- `text_classifier/config.py` — `fold_roles()` gains retired-role support;
  validation floor logic.
- `text_classifier/application/training.py` — accept + featurize external sets,
  split-overlap validation, reordered deployment-index build.
- `text_classifier/application/evaluation.py` — manifest records split
  provenance.
- `text_classifier/cli/train.py` — `--val-items` / `--test-items`.
- `tests/unit/test_validation.py` — overlap errors, label checks, fold-floor
  relaxation.
- `tests/integration/test_e2e.py` (or new `test_external_splits.py`) —
  end-to-end with external val, external test, and both.
- `README.md` — document the three modes and the temporal-split use case.

## Tests
- [ ] External val only: calibrator/thresholds derived from it (assert the
      calibration fold's rows went to fusion training — fold-role bookkeeping).
- [ ] External test only: `evaluation.json` counts match the external set size.
- [ ] Both: all folds train fusion; pipeline runs with `n_folds=2`.
- [ ] Train/val text overlap raises, naming the overlap count.
- [ ] External-set label unknown to `LabelSpace` raises before any encoding.
- [ ] Backward-compat: `run(items, label_space)` with no external sets is
      byte-identical to today (thresholds, evaluation, artifacts).
- [ ] All tests offline via `HashingEncoder` (house rule).

## Acceptance criteria
- [ ] No behaviour change whatsoever when neither flag is passed.
- [ ] OOF leakage rule intact for fusion-training rows; external sets never
      enter any index they are scored against.
- [ ] Model dir remains portable; manifest records split provenance.
- [ ] The leakage trap (pointing `--val-items` at training rows) is a hard
      error, not a footnote.

## Out of scope
User-controlled fold *assignment* for the internal CV (e.g. a group/fold column
on items.csv) — different feature. Multiple validation sets / cross-dataset
evaluation. Re-tuning an already-trained model on a fresh labeled set (that is
T66, and T66's `retune` path is the natural code-sharing partner for the
external-val featurization added here).
