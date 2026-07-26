# T80 — Track and choose the best epoch of an encoder fine-tune

status: in-review
tier: 4
depends_on: T24

> **Implementation note (in-review).** Delivered: per-epoch scoring + best-epoch
> selection inside `train_encoder`. The decision rule is the framework-free
> `EpochSelectionPolicy` (domain), the metrics are `encoder_retrieval_metrics`
> (domain, pure numpy), and the per-epoch loop is `EncoderEpochTracker`
> (infrastructure) — driven by one `SentenceEvaluator` call per epoch from
> `SentenceTransformer.fit` (`evaluation_steps=0`, verified against
> sentence-transformers 5.6's `EvaluatorCallback.on_epoch_end`). Config:
> `encoder.train_holdout_ratio` / `train_select_metric` /
> `train_select_min_delta` / `train_early_stopping_patience` /
> `train_holdout_seed`, all validated in `PipelineConfig.validate`. CLI:
> `--encoder-epochs`, `--encoder-epoch-holdout`, `--encoder-select-metric`,
> `--encoder-patience`. The per-epoch table is persisted to
> `<model_dir>/encoder/encoder_training.json`. Tests (offline, no torch):
> `tests/unit/test_encoder_epoch_selection.py` (metrics, policy, tracker, split,
> config) and `tests/unit/test_train_encoder_epochs.py` (the whole
> `train_encoder` orchestration against a fake `sentence_transformers` that
> mirrors the real `fit`/evaluator/save contract).

## Goal
Make a multi-epoch encoder fine-tune return its *best* epoch instead of its last,
and record how every epoch scored.

## Why
`EncoderConfig.train_epochs` existed but nothing observed it: `train_encoder`
called `model.fit(epochs=N)` and returned whatever the final epoch produced.
That makes "train for 20 epochs" a gamble — a bi-encoder fine-tuned with
`MultipleNegativesSymmetricRankingLoss` on a small, imbalanced corpus reliably
overfits the class descriptions somewhere in the middle of a long run, so the
last epoch is often measurably worse than epoch 5, and nothing in the model
directory said so. Downstream, that degrades every dense signal at once
(`d_desc_sim`, `d_proto_sim`, `d_knn_*`), which is expensive to diagnose from the
fusion metrics alone.

The fusion layer already has held-out evidence for its own decisions
(`evaluation.json`, the calibration/test folds). The encoder had none.

## Design
**Split.** `_stratified_holdout(labels, ratio, seed)` withholds
`floor(ratio * count)` items per class from the gradient updates, never a class's
last remaining example (a class with no (item, description) pair contributes
nothing to the loss). Stratified so rare classes are represented in the scored
slice; seeded so a rerun holds out the same items.

**Metrics** (`domain/services.py::encoder_retrieval_metrics`, pure numpy over
L2-normalized embeddings — the same dot-product-is-cosine invariant as the
retrievers):

| metric | meaning | analogue |
|---|---|---|
| `desc_acc@1` | nearest class description is the true class | `d_desc_sim` |
| `desc_mrr` | reciprocal rank of the true class | — (smoother) |
| `desc_pos_sim` | mean cosine to own description | diagnostic only |
| `knn_acc@1` | nearest labeled example shares the label | `d_knn_*` |

`knn_acc@1` needs the fine-tuning pool re-encoded each epoch, so it is only
computed when it is the selection target.

**Decision.** `EpochSelectionPolicy(metric, min_delta, patience)` — `best_epoch`
(earliest epoch achieving the max; 0 when nothing is scoreable) and `should_stop`
(patience epochs since the best). It is the single source of truth: the tracker
asks it about the whole history rather than keeping its own incumbent, so
selection during training and selection re-derived from a persisted history
cannot drift.

**Loop.** `EncoderEpochTracker.observe()` scores the holdout with the live
weights, appends to the history, and calls a `snapshot` callback on improving
epochs only. `train_encoder` snapshots to a temp dir (`model.save`) and reloads
the winner only when the best epoch is not the last one — the common
"still improving at the end" case pays nothing. Early stopping raises
`_StopFineTuning` out of `fit`; the best weights are already on disk.

**Version tolerance.** `observe()` ignores a repeated call on unchanged weights
(compares the holdout embeddings), so a sentence-transformers version that
evaluates at both a step boundary and the epoch end cannot shift the epoch
numbering or trip patience early.

## Invariants respected
- **Leakage:** the holdout is carved out of the caller's items, which in the OOF
  loop are already one fold's *training* rows. Withheld-from-the-gradient is
  still in-fold; the fusion model's rows are untouched, and no item is scored
  against an index built from its own fold.
- **Determinism:** the split is seeded (`train_holdout_seed`); ties in selection
  resolve to the earlier epoch, so two identical runs select identically.
- **Portability:** the epoch table is plain JSON in the encoder directory.
- **Defaults:** `train_epochs` stays `1`, where selection is inert — a default
  run is byte-for-byte what it was.

## Follow-ups (deliberately out of scope)
- Surface `best_epoch` in `model_card.md` / `evaluation.json` (today it lives in
  `encoder/encoder_training.json`); the pipeline would have to read it back off
  the encoder it fitted.
- Per-fold encoders each select their own epoch and each write their own history,
  but only the deployment encoder's is persisted (the per-fold encoders are
  discarded, as before). Logging is the only record for those.
- The same "select the best iteration" gap exists for the fusion model
  (XGBoost `n_estimators` with no early stopping against a validation fold).
  Separate ticket if wanted — it needs a fold role the fusion model can watch.
