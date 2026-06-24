# T06 — Leakage regression test (the scientific claim)

status: done
tier: 1
depends_on: T01, T03

## Goal
Prove, with a test, the package's central correctness claim: **out-of-fold features are
leakage-free** — an item is never scored against an index/prototype that was built using that
same item. This is the guarantee that makes the coverage report trustworthy.

## Why
The whole value proposition (calibrated abstention with a believable accuracy estimate) rests
on no train/eval leakage. Today nothing verifies it; a future refactor of `_build_oof` could
silently start leaking and every reported metric would become optimistic with no failing test.
This ticket gets its own file because it is the most important test in the suite.

## Background (read before implementing)
`TrainingPipeline._build_oof` uses `StratifiedKFold`. For each fold it builds dense + lexical
indices from the **training** rows `tr` and assembles features for the **validation** rows
`va`. The invariant: for any validation item `i`, none of its retrieved neighbors,
prototypes, or description indices may have been derived from item `i` itself.

## Approach (pick the strongest you can implement; do at least 1 + 2)

### 1. Self-neighbor exclusion (direct, white-box)
- Build a dataset with **duplicated or near-duplicated** item texts within a class.
- Run `_build_oof` (or replicate its per-fold loop) and capture, for each validation item,
  its dense kNN neighbor source-indices.
- Assert the validation item's **own original training-row index is never among its own
  neighbors** — because the index for its fold was built from *other* folds only. A clean way:
  tag each item with a unique token and assert an item can't retrieve its own unique token
  from the example pool at validation time.

### 2. Fold-disjointness (structural)
- Instrument/replicate the fold loop and assert `set(tr) ∩ set(va) == ∅` for every fold, and
  that `DenseRetrieverAdapter.build` / `LexicalRetrieverAdapter.build` for a fold only ever
  receive `tr` rows (assert on the texts/labels passed in).

### 3. Leak-detection canary (behavioral, strongest)
- Train two pipelines on the same synthetic data:
  (a) the real OOF pipeline;
  (b) a deliberately **leaky** variant where each item is scored against an index that
      *includes itself* (you can construct this in-test by building the index over all rows
      and assembling features for those same rows).
- Assert the leaky variant reports a **substantially higher** candidate recall / accuracy
  on its own evaluation than the OOF variant. This proves the test is *sensitive* — it would
  actually catch leakage if it were introduced. (If the two are indistinguishable, the test
  is worthless; this canary guards against that.)

### 4. Held-out role separation
- Assert `TrainingConfig.fold_roles()` keeps train / calibration / test folds disjoint, and
  that `_fit_fusion` trains only on `roles["train"]`, calibrates only on `roles["calibration"]`,
  and `_evaluate` reads only `roles["test"]` (white-box check on the `fold` column membership).

## Acceptance criteria
- [ ] A test proves a validation item cannot retrieve itself from its own fold's index
      (approach 1).
- [ ] A test asserts per-fold train/val disjointness and that index builders only see train
      rows (approach 2).
- [ ] The canary (approach 3) demonstrates the leak detector is sensitive: leaky > OOF by a
      clear margin. Document the threshold and why it's safe against fixture noise.
- [ ] Role-separation (approach 4) asserted.
- [ ] Tests use the offline `hashing_encoder`; no model download; deterministic seed.

## Out of scope
End-to-end accuracy bounds and persistence (T07).

## Notes
- Use a small `n_folds` (e.g. 3) and a small synthetic set so the test is fast.
- If replicating internal loops is brittle, prefer adding a thin, test-only hook or asserting
  on `oof["fold"]` membership rather than monkeypatching deep internals.
