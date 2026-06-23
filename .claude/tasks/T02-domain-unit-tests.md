# T02 — Domain unit tests (LabelSpace, policies, threshold tuner)

status: todo
tier: 1
depends_on: T01

## Goal
Exhaustively test the framework-free domain layer: `LabelSpace`, `AbstentionPolicy`,
`CandidatePolicy`, and `ThresholdTuner`. These are pure and deterministic, so assert
**exact** values and **exact** exceptions.

## Why
This layer encodes the decision rules and the key↔index contract the whole system trusts.
It has no IO and no randomness — there is no excuse for it not to be at 100% covered.

## Files under test
- `text_classifier/domain/models.py` (`LabelSpace`, value objects)
- `text_classifier/domain/services.py` (`CandidatePolicy`, `AbstentionPolicy`, `ThresholdTuner`, `FEATURE_NAMES`)

## Test cases

### `LabelSpace`
- [ ] `from_pairs` and the `ClassDefinition` constructor produce equivalent spaces.
- [ ] `len`, `size`, `keys`, `descriptions` return the expected values and order.
- [ ] `index_of` / `key_at` are inverse for every class; round-trip over all indices.
- [ ] `encode_labels` maps a sequence of keys to the right indices, order preserved.
- [ ] **Raises** `ValueError` on empty definitions.
- [ ] **Raises** `ValueError` on duplicate keys.
- [ ] `index_of` of an unknown key raises `KeyError`; `key_at` out of range raises `IndexError`.
- [ ] Value objects (`ClassDefinition`, `LabeledItem`, `Prediction`, `CoverageReport`) are
      frozen — assigning an attribute raises.

### `CandidatePolicy`
- [ ] Default `top_n_per_signal == 10`; custom value is respected; frozen.

### `AbstentionPolicy`
- [ ] `threshold_for` returns the per-class override when present, else the global threshold.
- [ ] `threshold_for` coerces the class index to `int` (pass a `np.int64`, expect a hit).
- [ ] `accept` is elementwise: build a `confidence` vector and `class_index` vector mixing
      classes with and without overrides; assert the exact boolean mask (`>=`, inclusive).
- [ ] Boundary: `confidence == threshold` is accepted (inclusive).

### `ThresholdTuner.threshold_for_precision`
This is the highest-value target — assert the contract precisely:
- [ ] Empty input → returns `1.0`.
- [ ] All items correct → returns the **lowest** confidence (max coverage), so everything
      is accepted.
- [ ] Nothing can meet the target (e.g. target 0.99 but accuracy tops out lower) → returns
      `max(confidence) + 1e-6`, i.e. a threshold that **accepts nothing**. Verify by feeding
      the result back through a `>=` and confirming zero acceptances.
- [ ] Mixed case with a hand-constructed array where the deepest acceptable point is known:
      e.g. confidences `[0.9,0.8,0.7,0.6]`, correct `[1,1,0,1]`, target `0.66` → walking the
      sorted prefix, running accuracy is `[1.0, 1.0, 0.667, 0.75]`; deepest index meeting
      ≥0.66 is the last → returns `0.6`. Assert the exact value.
- [ ] Ordering independence: shuffle the input pairs; the returned threshold is identical
      (function sorts internally).
- [ ] Ties in confidence are handled without crashing and give a sane threshold.

## Acceptance criteria
- [ ] `domain/models.py` and `domain/services.py` at 100% line coverage.
- [ ] Every exception path above is asserted with `pytest.raises`.
- [ ] At least the two hand-computed `ThresholdTuner` cases assert exact floats.

## Out of scope
Feature assembly (T03). The tuner's *integration* with calibration (T07).
