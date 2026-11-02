# T94 — Learned-sparse (SPLADE) retrieval signal

status: in-review
tier: 3
depends_on: T34 (phase 2), T33 (two-round assembler)

## Goal
Let the system learn *which words to trust* instead of relying on BM25's IDF.
A SPLADE model reads a text with a masked-language model and outputs a sparse,
non-negative weight per vocabulary word: words in the text, plus related words
the model infers (for Hebrew, the bare form of a prefixed word). Scoring is a
sparse dot product, like BM25, but with learned weights.

Optional: off unless `"splade"` is in `PipelineConfig.signals`. The default
config is byte-for-byte unchanged.

## What landed
- `domain/ports.py`: `SparseTextEncoder` port; `SignalProvider.fit(texts,
  labels, label_space)` optional hook (default no-op). This is the "stateful
  signal" lifecycle T34 phase 2 listed as future work.
- `infrastructure/splade.py`:
  - `HashingSparseEncoder` — offline stand-in (log-tf over crc32-hashed tokens).
  - `SpladeEncoder` — `AutoModelForMaskedLM`, `max over tokens of
    log1p(relu(logits))`, special tokens zeroed, top-`top_terms` kept. Lazy
    torch/transformers import. Saved with `save_pretrained` (safetensors).
  - `train_splade` — softmax CE of item vs. {batch classes ∪ `train_negatives`
    sampled classes} over class descriptions + FLOPS regularizer on both sides;
    AdamW, linear warmup, grad clipping.
  - `SpladeSignalProvider` — round one. Nodes `splade.desc` (vs. class
    description, 0 → NaN) and `splade.knn` (top-k pool neighbours by sparse dot,
    zero-overlap neighbours dropped, scattered per class). Both join the
    candidate union. Columns: `s_desc_sim/_missing`, `rank_/norm_/margin_s_desc`,
    `q_gap_s_desc`, `is_s_desc_top1`, `s_knn_sum/_max/_count/_missing`,
    `rank_/margin_s_knn`, `q_gap_s_knn`, `is_s_knn_top1`, `abs_top_splade`.
    Descriptions are encoded lazily from `ctx.label_space` and cached by
    content, so added/edited classes work without wiring. `self_ids` masks the
    query out of its own neighbours (LOO path). `top_terms(text)` shows which
    words the signal relies on.
- `config.py`: `SpladeConfig` at `retrieval.splade`, validated (model_name
  required for `kind="splade"`; `train` needs `"splade"` in signals, kind
  `splade`, and `n_folds >= 2`).
- `application/training.py`: `_fit_signal_providers` called per fold on the
  fold's training rows and once on all rows for the deployed model.
- `application/updating.py`: added examples are indexed via
  `with_added_examples` (encoder reused, not retrained).
- `registry.py`: `"splade"` signal kind; persisted under `signals/splade/`
  (`encoder/`, `pool.npz`, `splade.json`). No pickle.
- `pyproject.toml`: `splade` extra (`torch`, `transformers`).
- `examples/splade_hebrew_config.json`.

## Tests
`tests/unit/test_splade_signal.py`: hashing encoder determinism, NaN-not-0,
self-masking, label-space-driven descriptions, unfitted error, config
validation, end-to-end train → save → load → identical predictions, splade
standing in for lexical, per-fold fit sizes (leakage), no-retrain update. Real
backend against a tiny random BERT built in the test (no download): sparsity /
special tokens, training lowers loss, end-to-end with `train=true` and
safetensors round-trip. Skipped without torch.

## Follow-ups (not done)
- **Measure it.** Run on the Hebrew data with `cli.importance` and compare
  accuracy at target precision with/without `"splade"`, and candidate recall.
  Also run the cheap baseline first: BM25 with Hebrew prefix stripping.
- Encoding cost: without `train`, each fold re-encodes its pool with the same
  pretrained model. A run-wide cache (as `RetrievalIndexBuilder` does for dense)
  would save `n_folds` passes.
- Training targets are class descriptions only. Adding same-class example pairs
  as positives would directly train the `splade.knn` signal too.
- No best-epoch selection / holdout yet (cross-encoder and encoder have it).
- Memory: the MLM head is `batch * max_length * vocab` floats; with a 128k
  vocabulary keep `batch_size`/`train_batch_size` small or use a GPU.
