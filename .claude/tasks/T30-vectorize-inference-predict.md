# T30 — Vectorize `InferencePipeline.predict` (drop the `.iterrows()` loop)

status: todo
tier: 3
depends_on: T07

## Goal
Replace the per-row `decided.iterrows()` loop in `InferencePipeline.predict`
(`application/inference.py`) with vectorized operations that produce identical
`Prediction` output. No behaviour change — purely the hot-path efficiency fix the
project's own conventions call for.

## Why
`CLAUDE.md` mandates "No per-row Python loops on the hot path," yet inference
currently iterates rows one at a time and, worse, calls
`abstention.accept(np.array([conf]), np.array([cls]))` **once per item** with
one-element arrays. For a large `predict` batch that per-row `accept` call is the
dominant Python overhead. `decided` already has exactly one row per item (from
`top_per_item`), so this is a handful of column-wise array ops plus a single
`accept` call.

## Current code (the loop to remove)
```python
decided = top_per_item(add_confidence(feats, a.fusion, a.calibrator))
for _, row in decided.iterrows():
    i = int(row["item_id"]); cls = int(row["candidate"]); conf = float(row["conf"])
    abstain = not bool(a.abstention.accept(np.array([conf]), np.array([cls]))[0])
    results[i] = Prediction(...)
```

## Approach
- Pull `item_id`, `candidate`, `conf`, and `margin` out of `decided` as numpy
  arrays in one shot (`decided[col].to_numpy()`).
- Call `a.abstention.accept(conf_arr, cand_arr)` **once** → `(m,)` bool array.
- Map candidate indices → keys via a vectorized lookup (e.g.
  `np.asarray(label_space.keys)[cand_arr]`). Packaging the per-item results into
  `Prediction` objects may be a list comprehension; the numeric work must not be
  per-row.
- Scatter into `results` by `item_id`; items with no surfaced candidate keep the
  existing abstain default (`top_key=""`, `confidence=0.0`).

## Files to change
- `text_classifier/application/inference.py` — replace the loop.
- (Optional) `text_classifier/application/scoring.py` if a small vectorized
  helper (e.g. `decisions_to_arrays`) reads cleanly there.

## Tests
- [ ] **Output identity:** on the e2e fixture, the new `predict` returns
      `Prediction`s identical to the current implementation (`top_key`,
      `predicted_key`, `abstained`, `confidence`, `margin`). Capture the current
      output as a golden reference before refactoring.
- [ ] No-candidate item still abstains (`top_key=""`, `confidence=0.0`) — the
      existing T07 path stays green.
- [ ] Empty input returns `[]`.
- [ ] Mixed batch (some accepted, some abstained) preserves per-item order by
      `item_id`.
- [ ] `accept` is called **once** per `predict` (assert via a spy on a multi-item
      batch).

## Acceptance criteria
- [ ] No `.iterrows()` (or equivalent per-row Python loop) remains on the predict
      hot path.
- [ ] `abstention.accept` is invoked once per batch, not once per item.
- [ ] Output is identical to the pre-refactor implementation; all existing tests
      pass unchanged.

## Out of scope
Vectorizing the encoder or feature assembler (already chunked/vectorized).
Changing the `Prediction` schema or the abstention policy.
