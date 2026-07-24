# T68 — Update a deployed model with new classes/examples without retraining fusion

status: in-review
tier: 6
depends_on: T61, T66

> **Implementation note (in-review).** Delivered: `text-classifier-update`
> console script + `application/updating.py::update`. Classes: `--classes` is
> the full taxonomy (new keys appended via `DenseRetrieverAdapter.with_added_classes`
> + `LexicalRetrieverAdapter.with_added_descriptions`, T78's existing methods;
> an existing key's changed description is re-embedded via a new
> `with_updated_descriptions`; a file missing an existing key is rejected —
> "retrain" in the message). Examples: `--items` requires the original corpus
> (`corpus.jsonl.gz`, written by `--store-corpus` — default on — or supplied via
> `--base-items`); dense side only encodes the delta (new
> `DenseRetrieverAdapter.with_added_examples`, prototypes/`class_freq`
> recomputed over the merged pool via a shared `_prototypes_and_freq` helper
> extracted from `build`); lexical side does a full `LexicalRetrieverAdapter.build`
> over the merged corpus (BM25 IDF is corpus-global, no incremental path).
> Fusion/calibrator are reused verbatim (`replace()`, never refit). Corpus-fitted
> encoders (tfidf) get a warning, not a rejection. `--tune-with` runs T66's
> `retune` inline and reports new-class candidate recall; without it, thresholds
> are untouched and the carried-forward `evaluation.json`/`model_card.md` are
> marked `stale`. `meta.json` gains an `updates` provenance list, carried
> forward across `--out`/`--in-place` by a new `ArtifactRepository.read_meta`/
> `apply_update_provenance` pair (needed because `save()` always rewrites
> `meta.json` from scratch). Tests: `tests/integration/test_update_cli.py` (14
> cases). One deviation from the ticket's literal test list: "fusion +
> calibrator files byte-identical" is checked as fusion-file-hash-identical +
> calibrator-functionally-identical (`transform()` on a probe), not
> calibrator-hash-identical — `IsotonicCalibrator`'s pickle (a scipy `interp1d`
> inside it) isn't byte-stable across an unrelated reload+re-save even with zero
> semantic change, confirmed by direct round-trip testing.

## Goal
A `text-classifier-update` CLI (+ application use case) that adds new classes
and/or new labeled examples to an existing model directory by rebuilding only the
cheap, data-dependent parts — indices, prototypes, description embeddings, label
space — while keeping the trained fusion model and calibrator. Turns "the
taxonomy changed → full retrain" into a minutes-long index refresh.

## Why
Label spaces are never static at work: classes get added, descriptions reworded,
labeled examples arrive weekly. Today any of these means the full pipeline —
encoder work over the whole corpus, k-fold OOF assembly, fusion fit.

The architecture already paid for a cheaper path: **the pointwise fusion model is
class-agnostic by construction.** All ~28 features are *relative* (similarities,
ranks, counts, agreement — `FEATURE_NAMES`, `domain/services.py`); no per-class
weight exists anywhere in the XGBoost model. So a new class is "just" a new
column in the signal matrices: rebuild the indices and the fusion model scores it
like any other candidate. Nothing exposes this today.

## Design

### The corpus problem (solve first)
The model dir does not persist raw training texts — `dense.npz` holds embeddings,
`lexical.pkl` a fitted vectorizer — but appending examples to BM25 requires
refitting on the *full* corpus (IDF changes). Two supported paths:
- **Preferred**: training gains `--store-corpus` (default **on**; `--no-store-corpus`
  to opt out for privacy/size) writing `corpus.jsonl.gz` (text + label key per
  line) into the model dir. `update` requires it.
- **Fallback**: `update --base-items original_items.csv` when the dir predates
  the flag or opted out.

### The update itself (`application/updating.py`)
1. Load artifacts. Extend `LabelSpace` by **appending** new `ClassDefinition`s —
   existing class indices are stable, so `AbstentionPolicy.per_class` (keyed by
   index) and every persisted structure stay valid. Reject edits that *reorder or
   remove* classes (that IS a retrain; description-text edits to an existing key
   are allowed — they only re-embed that description).
2. Merge corpora (old corpus + new items; validate labels against the extended
   space, reusing T20's checks). Re-encode what changed: new/changed
   descriptions, new example texts (old example embeddings are reused as-is —
   the encoder didn't change). Recompute prototypes and `class_freq` for
   affected classes only; rebuild BM25 over the merged corpus.
3. Fusion + calibrator: untouched, per the class-agnostic argument above.
   `class_log_freq` shifts for grown classes — that is *correct* (it is a live
   feature, not a fitted weight).
4. Thresholds: new classes fall back to the global threshold (existing
   `AbstentionPolicy` semantics — no code change). Print a loud recommendation to
   run `text-classifier-tune` (T66) on fresh labeled data afterwards, and run it
   inline when the user passes `--tune-with labeled.csv`.
5. Persist: dense state, lexical index, corpus, `meta.json` classes block +
   an `updates` provenance list (timestamp, n added classes/items, package
   version). Model card gets an "updated" line. Write to `--out new_dir`
   by default; `--in-place` opt-in.

### Guardrails
- Encoder is per-fold/corpus-fitted (`tfidf` with `use_per_fold_encoder`-style
  fitting)? A corpus-dependent encoder's vocabulary would drift from the fusion
  model's training distribution — allow but warn; recommend full retrain.
- Report candidate recall of the *new classes* on any `--tune-with` labeled rows
  so the operator can see whether the new descriptions/examples actually retrieve.

## Files to add/change
- `text_classifier/application/updating.py` — use case.
- `text_classifier/cli/update.py` + `pyproject.toml` scripts entry.
- `text_classifier/cli/train.py`, `application/training.py`,
  `infrastructure/persistence.py` — `--store-corpus`, corpus + provenance IO.
- `README.md` — "evolving your taxonomy" section (this is a selling point).
- `tests/integration/test_update_cli.py`.

## Tests
- [x] Add a class with examples (offline tfidf encoder, not hashing — same
      offline/deterministic contract, and words built from the fitted
      vocabulary since tfidf's is frozen/corpus-dependent): update, then infer
      an obvious member of the new class → predicted correctly (`conf > 0.3`,
      real example support, not description-only).
- [x] Index stability: predictions and confidences for items of untouched
      classes are identical before/after an update that only adds examples to
      one other class.
- [x] Add examples to an existing class: `meta.json` abstention block
      (global + per-class thresholds) unchanged, fusion file byte-identical,
      calibrator functionally identical (see implementation note above for why
      not byte-identical), corpus grows by exactly the added count.
- [x] Reorder/remove attempts (a `--classes` file missing an existing key)
      rejected with a message containing "retrain".
- [x] Dir without corpus + no `--base-items` → actionable error naming both
      `--base-items` and `--store-corpus`.
- [x] Updated dir loads via unchanged `InferencePipeline.from_directory`.

## Acceptance criteria
- [x] No fusion/calibrator refit anywhere in the path (verified: fusion file
      hash-identical, calibrator functionally identical) unless `--tune-with`
      is passed, which explicitly opts into T66's refit; runtime is dominated
      by encoding the delta + the BM25 refit.
- [x] `meta.json` records `updates` provenance (carried forward across
      `--out`/`--in-place`); evaluation artifacts are marked `stale` (headline
      metrics predate the update) until `--tune-with`/a re-eval runs.
- [x] Old model dirs (no corpus) still fully usable for prediction/eval/tune;
      `update --items` on one is rejected with the actionable error above
      (classes-only `update` still works with no corpus at all).

## Out of scope
Removing/merging classes (retrain); encoder refresh (retrain); automatic
retrain-vs-update decision (the candidate-recall report is the input to that
human call); online/streaming updates.
