# T73 — Richer labeled-evaluation analytics (confusion, aggregate scores, abstention quality, bootstrap CIs)

status: todo
tier: 7
depends_on: T61

## Goal
Extend the labeled evaluation report with confusion analysis, standard aggregate
scores, abstention-quality metrics, and optional bootstrap confidence intervals — all
additive to `application/evaluation.py`, computed from arrays already passed into
`evaluate_decisions`. No new dependencies, no pipeline changes.

## Why
`evaluate_decisions` already gives coverage, accuracy-on-accepted, calibration
(Brier/ECE + reliability table), a risk-coverage curve, and a per-class table. What a
domain expert still asks for and we don't emit:
- *Which* classes get confused with which (per-class precision/recall hides pairwise
  confusion — the most actionable artifact on imbalanced multi-class).
- The standard one-number aggregates (macro/micro/weighted F1, balanced accuracy, MCC,
  Cohen's kappa) people expect to compare models.
- Whether abstention is well-targeted (are we abstaining on items we'd have gotten
  right = wasted coverage?). The system is built around abstention; we never score it.
- Error bars, so a small eval set's headline numbers aren't over-read.

Everything reuses arrays already in scope (`pred_idx`, `true_idx`, `accepted`,
`correct`, `confidence`); this is new pure functions assembled into the report dict.

## What to add (all in `application/evaluation.py`)

### 1. Confusion analysis -> `evaluation["confusion"]`
`confused_pairs(pred_idx, true_idx, accepted, keys, top_k=15)`:
- Most frequent off-diagonal `(true_key -> pred_key)` pairs among **accepted** decisions,
  with `count` and the pair's share of the true class's accepted volume.
- Top-k pairs (a dense C x C matrix is unreadable / not JSON-friendly at scale).
- Optional full matrix behind a flag for small C.

### 2. Aggregate scores -> `evaluation["aggregate"]`
`aggregate_scores(...)`: macro / micro / weighted **F1**, **balanced accuracy**, **MCC**,
Cohen's **kappa**.
- **Report BOTH framings** (decision from design review):
  - `accepted_only`: over accepted decisions — quality of what you act on.
  - `all_items`: abstention treated as a wrong / "no-prediction" outcome — the system
    including its abstentions.
  Each framing is its own sub-dict so a consumer can read either without ambiguity.
- Computed from the per-class true/pred counts already built for `per_class_table` —
  factor a shared `_class_counts()` so the two paths cannot disagree (see Refactor).

### 3. Abstention quality -> extend `overall` (or `evaluation["abstention_quality"]`)
- `accuracy_on_abstained`: of the items abstained on, the fraction that *would* have been
  correct. High -> over-abstaining (leaving easy wins); low -> abstention is catching the
  hard cases. Reads alongside the existing `accuracy_on_accepted`.

### 4. Bootstrap CIs (opt-in, seeded) -> `[lo, hi]` on headline scalars
`bootstrap_ci(metric_fn, *arrays, n=1000, alpha=0.05, seed)`:
- Wraps coverage, accuracy-on-accepted, Brier, ECE, macro-F1 with a percentile interval.
- **Seeded** (determinism is a tested invariant) and **opt-in** via an `n_bootstrap`
  argument (0 = off, the default) so the standard report stays fast and byte-identical.
- If this proves to widen scope too much, it is the one clean piece to split into a
  follow-up; items 1-3 stand alone.

## Refactor (the one place a bug could hide)
Extract a shared `_class_counts(pred_idx, true_idx, accepted, correct)` that both
`per_class_table`, `aggregate_scores`, and `confused_pairs` consume, so "accepted &
correct for class c" is defined once. Verify existing `per_class_table` output is
byte-identical after the refactor.

## Guardrails
- **Additive + versioned schema.** Add `report_schema_version` to the manifest; new blocks
  only *add* keys. Existing keys/shapes unchanged.
- **No new deps.** numpy + stdlib only (no `sklearn.metrics`); fits the air-gapped core.
  `_json_safe` already covers NaN/inf -> None.
- **No registry, no plotting.** Direct functions in `evaluation.py`; visualization/export
  is out of scope (separate ticket).
- **CLI + model card.** `evaluate` CLI prints a couple of new headline lines (macro-F1,
  top confusion pair, accuracy-on-abstained); full detail stays in `evaluation.json`;
  `render_model_card` gains a confusion + aggregate section.

## Files to change
- `text_classifier/application/evaluation.py` — new functions + `_class_counts` refactor +
  wire into `evaluate_decisions`; `render_model_card` additions; `report_schema_version`.
- `text_classifier/cli/evaluate.py` — print new headline lines; optional `--n-bootstrap`.
- `text_classifier/application/training.py` — passes through unchanged (gets the new blocks
  for free via `evaluate_decisions`); confirm `evaluation.json`/`model_card.md` still write.
- tests — unit per metric (known-answer cases incl. degenerate: all-accepted, all-abstained,
  single class, zero-support class); both aggregate framings; seeded bootstrap reproducible;
  `_class_counts` refactor leaves `per_class_table` byte-identical; JSON round-trip.

## Acceptance criteria
- [ ] `evaluation.json` gains `confusion`, `aggregate` (with `accepted_only` + `all_items`),
      `accuracy_on_abstained`, and `report_schema_version`; all existing keys unchanged.
- [ ] Default run adds no dependency and no measurable slowdown (bootstrap off by default).
- [ ] `--n-bootstrap N` produces seeded, reproducible CIs on the headline scalars.
- [ ] Existing evaluation tests pass untouched; `per_class_table` output unchanged.

## Out of scope
Inference-time / label-free monitoring + drift (separate ticket: the meatier Tier-B work).
AUROC/AUPRC of the correctness score (cheap follow-up). Plotting and CSV/Parquet export of
the metric tables (ties into T72; separate). A pluggable metrics registry.
