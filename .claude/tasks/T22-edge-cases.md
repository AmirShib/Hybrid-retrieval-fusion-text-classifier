# T22 — Edge cases: single class, class with no examples, k > n_docs, empty batch

status: todo
tier: 2
depends_on: T01, T04, T05, T07

## Goal
Harden the pipeline against degenerate-but-legal inputs that are absent from the
synthetic fixture used in T01–T07: a label space with only one class, a class
that has zero training examples, a k-neighbors request larger than the corpus,
and a zero-length query batch.

## Why
These are boundary conditions that arise in real deployments (new class added
mid-lifecycle, rare class never seen in training split, small corpus, empty API
call).  The system should either handle them gracefully (return a sensible result)
or raise an explicit error — not silently produce NaN-filled DataFrames or crash
with an index-out-of-bounds.

## Files under test
`infrastructure/retrieval.py`, `application/features.py`,
`application/training.py`, `application/inference.py`

## Part A — Single-class label space

- [ ] `DenseRetrieverAdapter.build` with `n_classes=1` → `prototype_similarity`
      returns a `(b, 1)` matrix; `knn_example_labels` returns `(b, k)` labels all `== 0`.
- [ ] `LexicalRetrieverAdapter.build` with 1 class → `description_score` returns `(b, 1)`.
- [ ] `FeatureAssembler.assemble` with 1 class → a non-empty feature frame is returned
      (not a crash from single-column masks).
- [ ] `TrainingPipeline.run` with a single-class dataset either raises a clear error
      (can't calibrate without two classes) or completes without crash — document and
      test whichever behaviour is chosen.

## Part B — Class with zero training examples

- [ ] `DenseRetrieverAdapter.build` where one class is in `LabelSpace` but has no
      examples in the training set:
      - prototype row for that class is all-NaN.
      - `class_freq[c] == 0`.
      - `prototype_similarity` for that class column is NaN for all queries.
      - `knn_example_labels` never returns that class index.
- [ ] `FeatureAssembler.assemble` correctly marks `d_knn_missing == 1` and
      `d_proto_sim == NaN` for candidate rows belonging to the empty class.
- [ ] Round-trip: `TrainingPipeline.run` on a dataset where one class has fewer
      examples than `n_folds` (some folds will see zero examples of that class)
      completes without crash and produces a finite `CoverageReport`.

## Part C — `k > n_docs` (more neighbors than documents)

- [ ] `BM25Index.top_k(queries, k=1000)` when corpus has 5 documents returns exactly
      5 real entries and the remaining 995 are padded (`-1` / `NaN`).
- [ ] `DenseRetrieverAdapter.knn_example_labels` with `k > n_examples` clamps to
      `n_examples` real neighbors and pads the rest with `-1` / `NaN`.
- [ ] `FeatureAssembler.assemble` with `k_neighbors > n_training_examples` produces
      a valid (possibly sparse) feature frame without crash.

## Part D — Empty query batch

- [ ] `DenseRetrieverAdapter.knn_example_labels(np.zeros((0, d)), k)` returns
      arrays of shape `(0, k)`.
- [ ] `DenseRetrieverAdapter.prototype_similarity(np.zeros((0, d)))` returns `(0, C)`.
- [ ] `LexicalRetrieverAdapter.knn_example_labels([], k)` returns arrays of shape `(0, k)`.
- [ ] `FeatureAssembler.assemble([], ...)` returns an empty DataFrame with the correct
      columns (already tested in T03; re-confirm after any changes from this ticket).
- [ ] `InferencePipeline.predict([])` returns `[]` (already tested in T07; re-confirm).

## Acceptance criteria
- [ ] All edge cases either pass gracefully or raise a clear, documented error.
- [ ] No new uncovered branches added to `retrieval.py` or `features.py` without tests.
- [ ] Tests run offline (HashingEncoder only).
- [ ] Full suite stays green.

## Out of scope
Performance under degenerate inputs (T32).
Single-example fine-tuning of the encoder (needs sentence-transformers).
