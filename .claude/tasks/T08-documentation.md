# T08 — Code comments and professional documentation

status: todo
tier: 1
depends_on: T01

## Goal
Go through every source file and add clear, professional inline comments and docstrings
so the codebase is fully self-documenting at the level a new contributor needs to
understand *why* a decision was made, not just *what* the code does.

## Why
The architecture is non-trivial: the NaN-as-missing encoding, the vectorized candidate
set union, the OOF leakage control, the precomputed BM25 weight matrix trick, and the
isotonic calibration + threshold tuning pipeline all contain subtle design choices that
are not obvious to a reader. Today most of the reasoning lives only in the README and in
the original author's head. This ticket makes the codebase maintainable by anyone.

## Documentation philosophy
- **Explain the WHY, not the WHAT.** Well-named identifiers already show what code does.
  Comments must earn their place by capturing a hidden constraint, a non-obvious invariant,
  a mathematical derivation, or a deliberate trade-off.
- **Module docstrings** set the scene — what the module's responsibility is, what it
  depends on, and any gotchas a reader should know before diving in.
- **Class docstrings** describe the contract: what the class represents, its key invariants,
  thread safety, and any lifecycle constraints.
- **Method/function docstrings** follow the NumPy docstring convention:
  summary line, `Parameters`, `Returns`, `Raises`, and `Notes` (for math or algorithms).
- **Inline comments** only for genuinely non-obvious lines — typically math derivations,
  index-manipulation tricks, or a "this looks wrong but it's intentional because..." case.
- Do **not** paraphrase the code. Do **not** reference git history, issue numbers, or
  the current task. These rot.

## File-by-file scope

### `text_classifier/domain/models.py`
- Module docstring: this is the shared vocabulary — framework-free, immutable value
  objects; explain why it imports nothing from ML frameworks.
- `ClassDefinition`: docstring noting that `key` is the stable external identifier and
  `description` is the natural-language text used both as a retrieval target and as the
  encoder fine-tuning label.
- `LabeledItem`: explain the role of ground-truth label in training vs. inference.
- `Prediction`: document each field — particularly `top_key` (always present) vs.
  `predicted_key` (None on abstention), and the intended use of `margin` for routing
  confidence.
- `CoverageReport`: explain the human-in-the-loop framing — coverage is the fraction
  accepted, accuracy-on-accepted is what the downstream consumer experiences.
- `LabelSpace`: full class docstring covering the key↔index invariant and why it is the
  single owner of that mapping; per-method docstrings with `Parameters`/`Returns`.

### `text_classifier/domain/ports.py`
- Module docstring: explain the hexagonal pattern — these ABCs are the only things the
  application layer imports; all concrete ML code lives behind them.
- Each port (`TextEncoder`, `DenseRetriever`, `LexicalRetriever`, `FusionModel`,
  `ConfidenceCalibrator`): class docstring covering the contract, and method docstrings
  with full `Parameters` / `Returns` including array shapes and dtypes (use the existing
  shape notation `(b, C)`, `(n, d)`, etc., but expand it per method).
- Note on the L2-normalization contract for `TextEncoder.encode`.
- Note on the NaN contract for `DenseRetriever.prototype_similarity` (NaN column for
  absent classes) and for `LexicalRetriever.knn_example_labels` (-1 padding for
  unfilled slots vs. NaN for scores).

### `text_classifier/domain/services.py`
- `FEATURE_NAMES`: a comment block explaining it is the single source of truth for
  column order and that any change touches features.py + meta.json schema.
- `CandidatePolicy`: explain that candidate recall is the ceiling on accuracy.
- `AbstentionPolicy`: explain the fallback rule (per-class → global), and the `int`
  coercion note in `threshold_for`.
- `ThresholdTuner.threshold_for_precision`: a `Notes` section with the algorithm —
  sort by descending confidence, compute running accuracy, find the deepest point still
  meeting the target. Document the "accept nothing" sentinel (`conf[0] + 1e-6`) and why
  it's `+ 1e-6` (just above max confidence → no item clears it).

### `text_classifier/config.py`
- Module docstring: explain that these dataclasses serialize to JSON alongside a model
  directory (the air-gapped portability requirement).
- `EncoderConfig`: note on `MultipleNegativesSymmetricRankingLoss` — why this loss and
  not cross-entropy.
- `RetrievalConfig`: explain `dense_chunk` / `feature_chunk` as memory-bound tuning
  knobs, not correctness parameters.
- `FusionConfig`: explain `auto_scale_pos_weight` — compensates for the extreme class
  imbalance that arises in pointwise training (one positive vs. many negatives per item).
- `TrainingConfig.fold_roles`: a comment on why last fold = test, second-last =
  calibration (calibration must be unseen by fusion; test must be unseen by both).
- `PipelineConfig.from_dict`: note on nested dataclass reconstruction (why it's manual
  rather than using `dacite` or similar).

### `text_classifier/infrastructure/encoder.py`
- `SentenceTransformerEncoder`: note on the L2-normalization flag and why it's enforced
  here (so callers can rely on dot product == cosine).
- `train_encoder`: full docstring covering the training objective, why
  `NoDuplicatesDataLoader` matters (prevents same-class sibling from being a negative for
  itself), and what `warmup_ratio` does to the LR schedule.

### `text_classifier/infrastructure/retrieval.py`
- Module docstring: explain the two key optimizations — BM25 precomputed weight matrix
  (`Q_binary @ W.T`) and query-chunked cosine mat-mul — and why each was chosen.
- `BM25Index.fit`: a `Notes` section deriving the BM25 formula and explaining each
  variable (`k1`, `b`, `idf`, `len_norm`), specifically the Lucene IDF variant used.
  Explain the `W` matrix shape and why it's transposed to `(vocab, n_docs)`.
- `BM25Index._query_incidence`: one-line comment: why we binarize (`q.data[:] = 1`) —
  query-term frequency is ignored in this BM25 variant.
- `BM25Index.top_k`: document the `-1` / NaN padding convention for unfilled slots and
  the positive-only filter.
- `_dense_topk`: explain the chunked argpartition-then-argsort pattern and why it's
  used instead of a full sort (O(n) partial select vs O(n log n)).
- `DenseRetrieverAdapter.build`: explain why prototypes are the L2-normalized mean (not
  centroid in raw space), and why a class with zero examples gets `NaN`.
- `DenseState`: note that this is the serializable subset — the rest of the adapter
  (chunk size) is reconstructed from config at load time.

### `text_classifier/infrastructure/fusion.py`
- `XGBoostFusionModel.fit`: comment on why NaN is passed through (not imputed) — XGBoost
  treats NaN as "missing" natively, which is exactly the "signal did not retrieve this
  class" semantics.
- `XGBoostFusionModel.fit`: explain `scale_pos_weight` calculation and why
  `auto_scale_pos_weight` is important for the heavily imbalanced pointwise training set.
- `IsotonicCalibrator`: explain isotonic regression as a non-parametric monotone map, and
  why `out_of_bounds="clip"` (prevents extrapolation beyond the calibration range).

### `text_classifier/infrastructure/persistence.py`
- Module docstring: document the on-disk layout with a comment tree and explain the
  constraint: only stdlib pickle + numpy + json + native formats (air-gapped portability).
- `ArtifactRepository.save`: comment on why each file is in its format (XGBoost native
  JSON survives version bumps; numpy `.npz` is portable and compact; pickle is used only
  for sklearn objects).
- `ArtifactRepository.load`: note on the feature-schema check opportunity (currently
  absent — could assert `meta["feature_names"] == FEATURE_NAMES` as a forward-compat
  guard; leave a `# TODO` if not implementing here).

### `text_classifier/application/features.py`
- Module docstring: explain the vectorization strategy (all signals as `(b, C)` matrices,
  candidate mask via union, gather with fancy indexing) and the memory-bounding via
  chunking.
- `_scatter_knn`: full docstring + inline comments on the NaN-for-missing encoding —
  particularly why `ksum[empty] = np.nan` rather than leaving zeros.
- `_topn_mask`: explain the NaN-ranks-last trick (`np.where(isnan, -inf, M)`) and the
  tie-admission behavior.
- `_row_rank`: explain the dense-rank method and why it uses argsort on argsort.
- `_row_minmax`: explain why `rng = np.where(hi > lo, hi - lo, 1.0)` uses `1.0` as the
  degenerate denominator (constant row → maps to 0/1, not a crash).
- `FeatureAssembler._assemble_chunk`: a structured comment block at the top grouping the
  four phases: (1) signals as matrices, (2) candidate mask, (3) per-query scalars,
  (4) gather. Inline comments on the non-obvious derived features:
  - `bdesc = np.where(bdesc_raw > 0, bdesc_raw, np.nan)` — why `0` means "no overlap",
    not a score.
  - `n_agree` — what "agreement" means and why 5 signals.
  - `abs_top_bm25` fallback to `0.0` when all-NaN.

### `text_classifier/application/scoring.py`
- `add_confidence`: brief docstring explaining the two-step transform (raw fusion score
  → isotonic calibration → P(correct)).
- `top_per_item`: explain why `margin` is `conf - second_conf` (or `conf - 0` if only
  one candidate) and how it's used downstream for routing decisions.

### `text_classifier/application/training.py`
- Module docstring: lay out the five stages numbered to match the inline sections.
- `_build_oof`: explain *why* each fold builds its own index — the leakage invariant.
  Comment on the `recall_hits / max(total, 1)` guard and what candidate recall means for
  system accuracy.
- `_fit_fusion`: explain the role-separation (train folds → fusion, calibration fold →
  isotonic + threshold, test fold → untouched until `_evaluate`).
- `_evaluate`: explain why `accept.any()` is guarded before computing `acc_acc`.
- `_build_deployment`: note on why the per-fold-encoder path refits on *all* data (not
  a single fold's encoder) for deployment.

### `text_classifier/application/inference.py`
- Note on the `.iterrows()` usage: comment that it is a known performance limitation for
  large batches and reference T30 for the fix.
- Explain the "no-candidate" fallback (results[i] is None → Prediction abstain with
  `top_key=""`).

### `scripts/demo.py`
- Expand the module docstring to explain `HashingEncoder` — that it is a deliberate test
  double, not a model substitute, and why shared tokens produce higher cosine (bag-of-hashed-
  tokens construction).
- Comment on `make_synthetic`'s imbalance (`c % 7` pattern) and why it's intentional.

## Acceptance criteria
- [ ] Every public class and public method has a NumPy-style docstring.
- [ ] Every module has a docstring covering its responsibility and any inter-module
      contracts it relies on.
- [ ] All inline comments address the WHY; none paraphrase the code.
- [ ] The BM25 math derivation is explained inline in `BM25Index.fit`.
- [ ] The NaN/missing convention is documented at every point it is produced or consumed.
- [ ] The leakage invariant is documented in `_build_oof` (not just in CLAUDE.md).
- [ ] The `# TODO` for the feature-schema load check is in `persistence.py`.
- [ ] No comment references a ticket number, git blame, or "added for X" rationale
      (those belong in the commit message, not the code).
- [ ] Running `python -m scripts.demo` still passes after the documentation pass.

## Out of scope
Changing any logic. Adding type annotations (T51). Writing tests (T01–T07).
Changing docstring format to something other than NumPy style.
