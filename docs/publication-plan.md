# Publication plan

An assessment of what in this package is publishable, where, and what work
stands between here and a submission. Written 2026-08-13.

## Verdict

**As a methods paper at an ML/IR venue: no.** Every component is established.
Feature-based fusion of retrieval signals is learning-to-rank; abstention is
selective classification (Chow 1970; El-Yaniv & Wiener 2010; Geifman &
El-Yaniv 2017); isotonic calibration is Zadrozny & Elkan 2002; hybrid BM25 +
dense retrieval is standard practice. A reviewer at NeurIPS/SIGIR/ACL will
correctly call this a competent engineering combination of known parts.

**As an applied paper in official statistics: yes.** Occupation and industry
coding (ISCO, ISIC, NACE, SOC, COICOP) is dominated by rule/dictionary systems —
CASCOT, Statistics Canada's ACTR, national census autocoders — and, more
recently, fastText and fine-tuned BERT. The field's headline metric is
"production rate at a target accuracy", which is our risk–coverage curve under a
different name. We are speaking that field's language natively, and it is
materially under-served.

The gap to publication is **not** the method. It is that the repository
currently reports absolute numbers with no baselines and no statistical
guarantee. That is what the work below fixes.

## Candidate contributions, ranked

### 1. Calibrated abstention with a finite-sample risk guarantee (the paper)

Existing autocoders threshold a raw score. We produce a calibrated `P(correct)`
with reliability diagnostics and per-class thresholds, and we can state the
operating point as a guarantee rather than an estimate. The proposed thesis:

> The hybrid's advantage is concentrated in low-support and newly-added classes —
> exactly where official taxonomies hurt — and calibrated abstention converts
> that advantage into an auditable coverage guarantee at a stated accuracy.

Falsifiable, and it targets the two operational pain points NSOs actually have:
the long tail of rarely-used codes, and taxonomy revisions.

**Required upgrade:** `ThresholdTuner.threshold_for_precision` picks the deepest
point where running accuracy clears the target *on the calibration set*. That is
an in-sample estimate with no coverage guarantee, and any reviewer who knows
conformal prediction will hit it. Reframing the decision layer as conformal risk
control (Angelopoulos et al.) or class-conditional Mondrian conformal prediction
yields a distribution-free finite-sample guarantee on accuracy-on-accepted. See
`.claude/tasks/T91-conformal-risk-control.md`. This is the single change that
moves the work from "we tuned a threshold and it worked" to a defensible
methodological claim.

### 2. Taxonomy revision without retraining

`text-classifier-update` adds classes and examples to a deployed model with no
retrain; a new class is reachable immediately through the description signal
alone. Revisions (ISIC Rev.4 → Rev.5, ISCO-88 → ISCO-08, national extensions)
are a chronic, expensive problem in official statistics and nobody in the
autocoding literature has a clean story for absorbing them.

The ISCO example ships the experiment: `prepare.py --target isco88` relabels the
same 7,000 items into the previous revision of the taxonomy.

### 3. Cross-lingual coding against English-language international classifications

`examples/coicop_hebrew/` already demonstrates Hebrew retail items coded against
English COICOP descriptions, zero-shot. Every non-anglophone NSO has this
problem. This can stand as its own paper, and ESCO (28 languages, all mapped to
ISCO-08, open licence) makes it reproducible.

### 4. Signal analysis as the framing device

The "treat retrieval signals like trading factors" framing is expository, not a
contribution, and should not be sold as one — the math is LTR fusion plus
selective prediction. But one idea transfers with real content: quant practice
does not merely combine signals, it *characterizes* them — marginal contribution
net of the others, cross-signal correlation, decay over time, regime breaks.

`application/signal_report.py` and `application/importance.py` already do the
first two. What is missing is **decay**: signals degrade at different rates as a
taxonomy ages and the item distribution drifts. A finding of the form "BM25-kNN
holds up under vocabulary drift while dense-prototype decays fastest, so monitor
this ratio to know when to recalibrate" is actionable for an NSO, and the framing
earns its place by having motivated the measurement. As metaphor alone it does
not survive review.

## What is missing

1. **Baselines on identical splits.** Nothing in the repo compares against
   anything. Minimum viable set: fine-tuned multilingual BERT, fastText, dense
   retrieval alone, an LLM-prompted classifier, and — if obtainable — the
   incumbent rule-based coder, which is what an NSO actually cares about beating.
   `--val-items` / `--test-items` already make splits shareable across model
   families; use that, and keep the split files in the repo.
2. **Public, reproducible data.** Done, see below.
3. **A real guarantee.** T91.
4. **Drift results.** The temporal split (`--val-items` on a later slice) is the
   experiment. Calibrate on period *t*, evaluate on *t+1*. Degradation is a
   better paper than no degradation: it motivates a recalibration protocol.
5. **A cost model.** Cost per manual code × items routed to review, against the
   cost of an error. Cheap to add and it is what the audience budgets against.

## Data

### Primary: ILO ISCO-08 (in the repo, working)

`examples/isco/` builds a 436-class, 7,002-item benchmark from two public ILO
workbooks. Naturally imbalanced (median 13 titles per class, tail to 1, head to
113), rich official class descriptions, and each item carries both its ISCO-08
and ISCO-88 code so the revision experiment comes free.

Measured with the TF-IDF encoder (the offline floor — no semantics at all):

| metric | value |
|---|---|
| candidate recall | 0.983 |
| accuracy, no abstention | 0.701 |
| coverage at target precision 0.95 | 0.334 |
| accuracy on accepted | 0.953 |
| expected calibration error | 0.036 |
| best single signal (BM25 ↔ description) | 0.558 |

Fusion beats the best single signal by 14 points, and the five signals agree on
a single top class only 19% of the time — there is genuinely something to fuse.
A real bi-encoder should move all of this up.

**Caveat to state plainly in any paper:** the ILO index is a *coding index* —
clean canonical job titles, not survey write-ins with their typos, abbreviations
and fragments. It is the right dataset for reproducibility and for comparing
methods, and it is not a substitute for evaluation on real write-ins. Pair it
with confidential NSO data for the production claim.

### Secondary, worth adding

- **ESCO** (European Commission) — occupations in 28 languages, every one mapped
  to an ISCO-08 code, CSV, open licence. This is the cross-lingual benchmark and
  it also supplies many alternative surface forms per occupation, which is a
  harder and more realistic item set than the ILO index.
- **WageIndicator occupation database** — ~4,200 ISCO-08-coded titles across many
  languages. Verify licensing.
- **Industry (ISIC/NACE) side is weaker.** No comparable public labeled set
  surfaced. Options: derive items from ISIC/NACE explanatory notes the way the
  ISCO example does, or use a company-description dataset with NACE labels and
  accept its noisier provenance. Treat industry coding as future work rather
  than blocking the occupation paper on it.

## Venues

**Primary**
- *Journal of Official Statistics* — open access, methodological, exactly this scope
- *Statistical Journal of the IAOS* — receptive to operational/quality-framework work
- *Journal of Survey Statistics and Methodology*

**Conferences**
- NTTS (Eurostat), UNECE HLG-MOS machine-learning work, ICES

**Software paper — do this regardless**
- **JOSS**. The repository would pass review as-is: tests, CI, docs, examples, a
  real API, a licence. It is a citable publication for a few days of work, and
  JOSS-plus-application-paper is the standard pattern. *SoftwareX* is the
  alternative.

## Order of work

1. **JOSS submission** — ready now, independent of everything else.
2. **Baselines on the ISCO benchmark** — without these there is no paper, and the
   results may change which claim is worth making.
3. **T91, conformal risk control** — the credibility upgrade.
4. **Temporal-drift experiment** — needs a dated corpus; NSO data or a proxy.
5. **Decide: one paper or two** (occupation coding; cross-lingual coding).

## Legal

Check licensing before any submission or data redistribution:

- ISCO-08 workbooks are ILO copyright; `prepare.py` downloads them to a
  git-ignored `build/` and they are not committed.
- `examples/coicop_hebrew/_data.csv` (~160k labeled Hebrew retail items) is
  committed — confirm the right to redistribute it and to publish derived
  results.
- NSO microdata will carry its own disclosure rules; expect to report aggregates
  only.
