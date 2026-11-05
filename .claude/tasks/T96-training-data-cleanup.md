# T96 — Training-data cleanup round

status: in-review
tier: 3

## What landed
`application/cleanup.py` (`run_cleanup`, `honest_shares`, `write_review_file`),
`CleanupConfig` (config section `cleanup`, off by default), CLI `--cleanup`,
wiring in `TrainingPipeline._run` (extra out-of-fold pass before training;
progress stage "cleanup"), summary in `evaluation.json["cleanup"]` + model card,
`suspicious_rows.csv` (UTF-8 BOM). Tests: `tests/unit/test_cleanup.py`.

## Measurement that set the defaults (2026-11-05)
20k Hebrew COICOP items (TF-IDF + BM25, cleaned text), untouched 3,977-item test:
baseline 90.57%; flag rule best>=0.8/own<=0.10 (2.2% flagged): drop 90.34%,
relabel 90.37%; rule 0.9/0.05 (1.4%): drop 90.50%, relabel 90.45%. Flagged rows
mixed real label errors (oranges labelled beef) with correct rows the model got
wrong (ice-cream bars labelled ice cream; model said sweets), and the suggestion
itself was often wrong. Hence `mislabel_action="keep"` by default.

## Controls added (2026-11-05)
Cleanup pass: `signals`, `n_folds`, `train_epochs`, `fine_tune`; `rounds`;
detector: `detector_params`, `knn_agreement_signals`, `min_knn_agreeing`;
actions: `not_shortlisted_action`, `ambiguous_action`, `quarantine` +
`quarantine_from` (`knn_pool`, `fine_tune`; implemented at every row-consuming
site: per-fold/final encoder fine-tune, index build, SPLADE fit, cross-encoder
mining + fine-tune, stored corpus, LOO self positions); `max_changed_fraction`;
`review_decisions` (human decisions first, locked against automation).

## Open
- Re-measure keep / quarantine / drop with a neural encoder, fine-tuning and
  real class descriptions, on an untouched test set.
- Per-row weights (down-weight instead of drop) need sample-weight support in
  the fusion port.
