# T92 — Encoder fine-tuning via SentenceTransformerTrainer + training knobs

status: in-review
tier: 4
depends_on: T80

## What landed (2026-11-01/02, on `dev`)

**Trainer migration** (`infrastructure/encoder.py::train_encoder`):
- `SentenceTransformerTrainer` replaces `model.fit` and the hand-written epoch
  loop. `BatchSamplers.NO_DUPLICATES` replaces `NoDuplicatesDataLoader` (same
  job: same-class items share the description text, so they never share a batch).
- Per-epoch scoring is `InformationRetrievalEvaluator` over the stratified
  holdout: `desc_acc@1` = its `accuracy@1`, `desc_mrr` = full-rank `mrr@C`,
  `knn_acc@1` = a second evaluator whose corpus is the fine-tuning pool (built
  only when selecting on it). Query/document prompts are passed to the evaluator.
- Best epoch = `load_best_model_at_end` (verified the best weights are reloaded
  into the live model); patience/min-delta map onto `EarlyStoppingCallback`;
  ties go to the earlier epoch. `encoder_training.json` keeps its schema.
- Checkpoints are weights-only (`save_only_model`) in a temp dir that is removed;
  `report_to="none"`; the model dir is still written only by `model.save()`.
- Removed: `EpochSelectionPolicy`, `encoder_retrieval_metrics`,
  `EncoderEpochTracker`, the `desc_pos_sim` metric.

**New `EncoderConfig` fields** (all default to the previous behaviour; CLI flags
in parentheses): `learning_rate` = 2e-5 (`--encoder-lr`), `weight_decay` = 0.01
(`--encoder-weight-decay`), `max_seq_length` = None (`--encoder-max-seq-length`,
persisted with the saved model), `pooling_mode` = None (`--encoder-pooling`;
rebuilds Transformer + `Pooling(mode)`, warns when replacing the model's pooling
or dropping a post-pooling module, carries the model's prompts over). Overrides
apply on build/fine-tune, never when reloading a saved model dir.

**Default loss** is now `multiple_negatives_ranking` (was the symmetric variant,
deprecated upstream from sentence-transformers 5.6). Behaviour change for
untouched configs; the symmetric loss stays selectable.

**Dependencies:** the extra is `sentence-transformers[train]>=3.0` (from 4.x,
`datasets`/`accelerate` sit behind that `train` extra, so locked installs lacked
them); `requirements.lock` regenerated, additions only.

**Compatibility, tested:** sentence-transformers 6.2.0, 5.6.0 (locked) and 5.2.0
(on transformers 4.57 and 5.22). 5.2.0 + transformers 5 needed
`args.save_safetensors` supplied (its Trainer reads an argument transformers 5
removed). Version-dependent import paths and APIs go through small compat helpers.

**Tests:** `tests/unit/test_train_encoder_trainer.py` drives the real library on a
tiny randomly-initialised BERT built locally (offline; skipped without the extra).

## Known gaps / follow-ups
- **Device handling differs from `SentenceTransformer(device=...)`:** the Trainer
  places the model itself. `"cpu"` is honoured; a CUDA index (`"cuda:1"`) is not,
  and several visible GPUs are all used (DataParallel). Both warn; use
  `CUDA_VISIBLE_DEVICES`. Not tested on a GPU.
- **Prompts are not applied to the training pairs**, only to scoring and inference
  (unchanged from before). With E5/BGE-style prompts the model trains without them.
  Candidate fix: `SentenceTransformerTrainingArguments.prompts`.
- **Mixed precision** (`bf16`/`fp16`) is now a one-argument addition; not exposed.
