# T07 — End-to-end pipeline + persistence round-trip

status: todo
tier: 1
depends_on: T01, T02, T03, T04, T05

## Goal
One integration test that exercises the whole system offline — `TrainingPipeline.run` →
`ArtifactRepository.save` → `ArtifactRepository.load` → `InferencePipeline.predict` — and
asserts both **round-trip identity** and **sane behavioral bounds**. This is the capstone that
ties the unit-tested parts together and turns `scripts/demo.py` into a real, asserting test.

## Why
Units passing doesn't prove the wiring is correct: column-order drift, a persistence field
mismatch, or a config serialization bug only surfaces end-to-end. This is also the regression
net that makes Tier 3/4 refactors safe.

## Files under test
`application/training.py`, `application/inference.py`, `application/scoring.py`,
`infrastructure/persistence.py` — together.

## Test cases

### Train → report
- [ ] Run `TrainingPipeline(cfg, shared_encoder=hashing_encoder).run(items, label_space)` on
      the synthetic fixture with a fixed seed and small XGBoost (`n_estimators≈30`).
- [ ] `CoverageReport` fields are well-formed: `0 ≤ coverage ≤ 1`,
      `0 ≤ accuracy_on_accepted ≤ 1` (or NaN if nothing accepted),
      `candidate_recall` in `[0,1]`, `n_items == len(test fold)`.
- [ ] **Behavioral bound:** on the (easy, separable) synthetic data, candidate recall is high
      (e.g. `> 0.9`) and accuracy-on-accepted ≥ accuracy-if-no-abstain (abstention should not
      *hurt* accepted accuracy). Pick thresholds with margin so the test isn't flaky.
- [ ] **Determinism:** running `run(...)` twice with the same seed yields the same report
      (same coverage/recall). If XGBoost introduces nondeterminism, pin `random_state` and
      single-thread (`n_jobs=1`) in the test config.

### Persistence round-trip
- [ ] `save(artifacts, tmp_path)` writes the documented layout: `encoder/`, `dense.npz`,
      `lexical.pkl`, `fusion.json`, `calibrator.pkl`, `meta.json`. Assert each exists.
- [ ] `meta.json` contains `feature_names == FEATURE_NAMES`, the full config, the class list
      (keys+descriptions in order), and abstention thresholds (global + per-class).
- [ ] `load(tmp_path)` reconstructs a `DeployedArtifacts` whose `label_space`, `config`, and
      `abstention` equal the originals.
- [ ] **Prediction identity:** `InferencePipeline(artifacts).predict(sample)` ==
      `InferencePipeline(load(tmp_path)).predict(sample)` — same `predicted_key`, same
      `abstained`, and `confidence` equal within ~1e-6, for a batch of sample texts.

### Inference contract
- [ ] `predict` returns one `Prediction` per input, in input order.
- [ ] An item that surfaces **no candidate** (craft an out-of-distribution / empty-ish text)
      yields `Prediction(top_key="", confidence=0.0, abstained=True, predicted_key=None)`.
- [ ] Abstained predictions have `predicted_key is None`; accepted ones have
      `predicted_key == top_key`.
- [ ] Empty input list → empty output list (no crash).
- [ ] `confidence` is in `[0, 1]`; `margin` (when present) is `≥ 0`.

### Config serialization
- [ ] `PipelineConfig.to_dict()` → `from_dict()` round-trips to an equal config (covers the
      nested dataclasses). Guards the `meta.json` config path.

## Acceptance criteria
- [ ] Full train→save→load→predict path runs offline with the `HashingEncoder` double.
- [ ] Prediction-identity-after-reload test passes within tolerance.
- [ ] Behavioral bounds assert with comfortable margins (not brittle).
- [ ] `scripts/demo.py` is either converted to / mirrored by this test, or reduced to a thin
      wrapper that calls shared helpers (no asserting logic duplicated).

## Out of scope
Per-fold encoder fine-tuning path (`use_per_fold_encoder=True`) — needs real
sentence-transformers; mark `skipif` and leave for a later integration ticket.
