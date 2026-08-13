# T91 — Conformal risk control for the abstention threshold

status: todo
tier: 5
depends_on: T66

## Goal

Give the abstention threshold a **finite-sample, distribution-free guarantee**:
"accuracy on accepted items is at least `target_precision`, with probability at
least `1 − δ` over the draw of the calibration set." Today's threshold is a point
estimate of that quantity with no guarantee attached.

Additive, opt-in, and off by default. The existing tuner stays the default path
and its output must not change by a single bit.

## Why

`ThresholdTuner.threshold_for_precision` (`domain/services.py`) sorts by
descending confidence, walks the running accuracy, and returns the confidence of
the **deepest** point still clearing the target:

```python
acceptable = np.where(running_acc >= target)[0]
return float(conf[acceptable[-1]])   # deepest acceptable point
```

Two problems, both consequences of that being an in-sample selection:

1. **It is a maximum, not a bound.** Scanning every prefix and keeping the
   deepest one that clears the target is selection over ~n hypotheses on the same
   data the estimate comes from. The resulting empirical accuracy is optimistically
   biased almost by construction, and the bias grows with n.
2. **The tail is where it bites.** For a class with 30 calibration items, the
   running accuracy at the chosen cut is estimated from a handful of decisions.
   `per_class_min_support` limits how often we do this, but it does not quantify
   the risk of the ones we do.

For a production official-statistics deployment, "we hit 95% on the calibration
set" and "we will hit 95% on next quarter's intake" are different claims, and
only the second is worth publishing or promising to a stakeholder. See
`docs/publication-plan.md`.

Conformal risk control gives the second claim: choose the threshold as the
smallest λ whose *upper confidence bound* on error still meets the target, rather
than the smallest λ whose *point estimate* does.

## Design

### Domain: a new selection rule beside the existing one

Add to `domain/services.py`, next to `ThresholdTuner` — pure numpy, no IO, which
is where this belongs:

```python
class ConformalThresholdTuner:
    """Threshold selection with a finite-sample guarantee on accuracy-on-accepted.

    Same signature and same return type as ThresholdTuner.threshold_for_precision,
    so it drops into the same call sites; the difference is which point on the
    risk-coverage curve is chosen.
    """

    @staticmethod
    def threshold_for_precision(
        confidence: np.ndarray,
        correct: np.ndarray,
        target: float,
        delta: float = 0.05,
    ) -> float:
```

The rule: over candidate thresholds λ (the observed confidences suffice — the
accept set only changes there), compute for each the number accepted `n(λ)` and
the errors among them `e(λ)`. Accept λ only if the upper confidence bound on the
error rate at level δ is still ≤ `1 − target`. Return the smallest such λ
(maximum coverage among the valid ones), and `max(conf) + eps` — accept nothing —
when none qualify, matching the existing tuner's degenerate case exactly.

For the bound, use the exact Clopper–Pearson upper limit on a binomial
proportion (`scipy.stats.beta.ppf`; scipy is already a core dependency). It is
the conservative, textbook choice and needs no asymptotic assumption — which
matters precisely in the small-`n(λ)` regime that motivates this ticket. A
Hoeffding bound is the fallback if a scipy-free path is ever needed; do not use a
normal approximation.

Multiplicity over the λ grid must be handled, not ignored — that is the exact
failure of the current tuner. Bonferroni over the distinct candidate thresholds
(`δ / n_candidates`) is acceptable and simple; a fixed-sequence test walking λ
from most to least conservative and stopping at the first failure is tighter and
costs the same. Pick one, and state which in the docstring, because a reviewer
will ask.

### Application: config plumbing

- `CalibrationConfig` (or a new `AbstentionConfig` in `config.py`, if that reads
  better against the existing config layout) gains:
  - `threshold_rule: str = "empirical"` — `"empirical"` (today's tuner) or
    `"conformal"`.
  - `risk_delta: float = 0.05` — only meaningful for `"conformal"`.
- `fit_calibration_and_abstention` (`application/training.py:70`) selects the
  rule. It is already the single shared path for both training and
  `application/tuning.py::retune`, so both inherit this for free — do not add a
  second selection site.
- Both fields persist into `meta.json` via the existing config block and are
  echoed in `model_card.md` next to the thresholds. A deployed model must state
  which rule produced its operating point; an unlabelled threshold is exactly the
  ambiguity this ticket exists to remove.

### Per-class thresholds

The per-class loop is where the guarantee matters most and where it will most
often decline to act: a class with 30 calibration items frequently admits *no* λ
whose bound clears a 0.95 target. That is the correct answer, not a bug. When a
class yields no valid λ, fall back to the global threshold — the same behaviour
`AbstentionPolicy` already has for a class with no override, so no change is
needed in the policy itself.

Expect the conformal rule to select **more conservative thresholds and lower
coverage** than the empirical one, and expect the gap to widen as calibration
support shrinks. Reporting that gap honestly is a large part of the point.

## Evaluation

Extend `application/evaluation.py` so a report can carry the realized risk beside
the promised one — the empirical accuracy-on-accepted is already computed, so this
is mostly a matter of recording the target and δ alongside it.

## Tests

Offline, `HashingEncoder` only, as everything here is.

**Domain (`tests/unit/test_domain.py` or a new `test_conformal.py`):**
- Degenerate cases match `ThresholdTuner` exactly: empty input → 1.0; nothing
  meets the target → accept nothing.
- Monotonicity in δ: a smaller δ never returns a *lower* (more permissive)
  threshold.
- Conservativeness: on the same input, the conformal threshold is ≥ the empirical
  one. This is the ticket's core claim and should be property-tested over random
  inputs, not pinned to one example.
- Bound correctness: on synthetic data with a known error rate, the selected
  threshold's true error rate exceeds `1 − target` in at most ~δ of repeated
  draws. This is the guarantee; simulate it (a few hundred trials at a fixed seed
  is enough to catch a broken bound, and it is fast because it runs on arrays,
  not models).
- Small-`n` behaviour: with 20 calibration items and target 0.99, the rule
  declines rather than returning an unsupported threshold.

**Application:**
- `threshold_rule="empirical"` reproduces current output bit-for-bit — pin this,
  it is the compatibility contract.
- Config round-trips through `meta.json` and is re-applied on load.
- `retune` honours the rule from the loaded config.

## Out of scope

- Conformal *prediction sets* (returning a set of classes with coverage
  guarantees) rather than a single top-1 decision with abstention. That is a
  bigger, genuinely interesting change to the output contract — the human-review
  queue would receive a candidate set rather than a suggestion — and it deserves
  its own ticket. `--top-k` (T65) already covers the practical need.
- Adaptive/online recalibration under drift. That is the temporal-split
  experiment in `docs/publication-plan.md`, and it needs this ticket first.

## References

- Angelopoulos, Bates et al., conformal risk control / "Learn then Test"
- Vovk, Mondrian (class-conditional) conformal prediction
- Clopper & Pearson (1934) for the interval

Verify current versions of these before citing in a paper; they are here to name
the right ideas, not as a bibliography.
