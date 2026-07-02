# T66 — Re-tune calibration + abstention thresholds on a trained model (tune CLI)

status: todo
tier: 6
depends_on: T61

## Goal
A `text-classifier-tune` console script (and matching application-layer function)
that takes an existing model directory plus a *fresh* labeled CSV, refits the
calibrator and re-tunes the abstention thresholds against a chosen
`--target-precision`, and updates the model directory in place — without retraining
the encoder, indices, or fusion model.

## Why
The operating point (target precision → thresholds) is baked in at train time
(`TrainingPipeline._fit_fusion`). Moving the coverage/precision knob, or adapting
to drift that the eval CLI has surfaced, currently costs a full retrain — encoder
encode of the whole corpus, k-fold OOF assembly, fusion fit — even though
calibration + threshold tuning are seconds of work on arrays the model already
produces. The risk-coverage curve persisted in `evaluation.json` is proof the
information exists; this ticket makes it actionable. It is also the natural
first response to drift: re-anchor confidence on recent labeled data, retrain only
when candidate recall itself has degraded.

## Design
**Application layer** — `retune(artifacts, items, label_space, target_precision,
per_class_min_support) -> (AbstentionPolicy, ConfidenceCalibrator, evaluation_dict)`
in a new `text_classifier/application/tuning.py`:
1. Encode + assemble features for the labeled items against the *deployed* indices
   (same path as `cli/evaluate.py` uses today — reuse, don't duplicate).
2. Get **raw** fusion scores (`fusion.predict_proba`), fit a fresh calibrator of the
   configured kind (`build_calibrator(config.calibration)`) on (raw, is_true).
3. `top_per_item` → tune global + per-class thresholds exactly as
   `TrainingPipeline._fit_fusion` does (extract that block into a shared helper in
   `domain`/application rather than copying it).
4. Build the evaluation dict via `evaluate_decisions` so the caller can see the new
   operating point before committing to it.

**Persistence** — a focused update, not a rewrite: save the new calibrator file
(via its registry spec), rewrite the `abstention` block and `config.training.
target_precision` in `meta.json`, and write a fresh `evaluation.json` +
`model_card.md` via `write_evaluation_artifacts` with a manifest noting
`"retuned_at"` / item count. Add an `ArtifactRepository.update_decision_layer(...)`
method for this; do not hand-edit JSON in the CLI.

**CLI** — `text_classifier/cli/tune.py`, console script `text-classifier-tune`:
`--model dir/ --input labeled.csv --target-precision 0.97
[--dry-run] [--text-col/--label-col]`. `--dry-run` prints the would-be coverage /
accuracy-on-accepted / thresholds and writes nothing. Non-dry-run prints the same
plus what was updated.

**The leakage warning (document loudly, in `--help` and the README section):** the
labeled items must be *fresh* — items that were in the training set sit inside the
deployed indices and retrieve themselves, so their confidences are optimistically
biased and the tuned threshold will under-abstain in production. This tool must
never be pointed at `items.csv` from training. (Detecting overlap by exact text
match is cheap — do it and *warn*, listing the count.)

## Files to add/change
- `text_classifier/application/tuning.py` — the use case.
- `text_classifier/application/training.py` — extract the shared threshold-tuning
  helper; behaviour unchanged.
- `text_classifier/infrastructure/persistence.py` — `update_decision_layer`.
- `text_classifier/cli/tune.py` + `pyproject.toml` (`[project.scripts]`) — the CLI.
- `tests/integration/test_tune_cli.py` — end-to-end.
- `README.md` — usage + the freshness warning.

## Tests
- [ ] Train a model (hashing encoder), retune on a held-out labeled set with a
      higher target: new global threshold ≥ old, coverage on that set drops,
      accuracy-on-accepted meets the new target (up to the tuner's guarantee).
- [ ] `meta.json` abstention block and calibrator file change; encoder/dense/
      lexical/fusion files byte-identical (assert on mtimes-or-hashes).
- [ ] Reloading via `InferencePipeline.from_directory` uses the new thresholds.
- [ ] `--dry-run` leaves the directory byte-identical.
- [ ] Overlap warning fires when the tune set intersects training texts.

## Acceptance criteria
- [ ] No encoder/index/fusion retraining anywhere in the path; runtime is
      dominated by encoding the tune set.
- [ ] Threshold logic shared with `TrainingPipeline`, not duplicated.
- [ ] Model dir stays fully portable and loadable by unchanged inference code.

## Out of scope
Target-coverage mode (T43 adds the alternative objective; this CLI should grow the
flag when T43 lands); refitting the fusion model on new data (that is retraining);
automatic drift detection/scheduling.
