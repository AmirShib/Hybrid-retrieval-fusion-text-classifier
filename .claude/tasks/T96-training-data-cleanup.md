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

## Open
- Re-measure with a neural encoder and real class descriptions (the detector
  is only as good as the honest model behind it).
- Per-row weights (down-weight instead of drop) need sample-weight support in
  the fusion port.
- Use reviewed `suspicious_rows.csv` decisions as an input on the next run.
