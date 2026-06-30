# T45 — Per-class calibration behind the `ConfidenceCalibrator` port

status: todo
tier: 4
depends_on: T42

## Goal
Add a `PerClassCalibrator` that fits a separate inner calibrator (isotonic, platt,
or beta) per class, with a global fallback for classes whose calibration support is
too small to trust. Selectable via `CalibrationConfig.kind="per-class"` (plus an
`inner` kind and a `min_support`), trainable and persistable end-to-end through the
registry.

## Why
Calibration is global today: one score→P(correct) curve for every class. But a raw
score of 0.8 can mean 90%-reliable for a common, well-represented class and
50%-reliable for a rare one. A global curve averages those and is wrong for both.
Thresholds already go per-class (`AbstentionPolicy`, `domain/services.py`); calibration
is the remaining global stage. T42 deliberately scoped this out ("thresholds already
go per-class; calibration stays global") — this ticket is that deliberate follow-up.

## The port change (this is NOT purely additive)
Unlike T42, this needs the candidate class index to flow through the calibrator. The
current contract is class-blind:

    ConfidenceCalibrator.fit(scores, correct)
    ConfidenceCalibrator.transform(scores)
    # application/scoring.py::add_confidence calls calibrator.transform(raw) — no class

Extend both signatures with an optional keyword so existing calibrators stay a
one-line, behaviour-preserving change:

    fit(self, scores, correct, *, classes=None)
    transform(self, scores, *, classes=None)

- `classes=None` → every existing calibrator (isotonic/platt/beta) ignores it and is
  byte-for-byte identical. Add `classes=None` to their signatures only.
- `application/scoring.py::add_confidence` passes `features["candidate"].to_numpy()`
  as `classes`. This is the one non-additive edit outside the new file.

## Design
- Construct from a factory `make_inner: () -> ConfidenceCalibrator` (so per-class and
  global share the inner kind) and `min_support: int`.
- `fit`: always fit a global inner calibrator on all rows. For each class with
  `count >= min_support`, fit a dedicated inner calibrator on that class's rows.
  Classes below `min_support` fall through to global — mirror
  `AbstentionPolicy.threshold_for`'s per-class-with-global-fallback shape.
- `transform`: default all rows to the global curve, then overwrite rows of
  well-supported classes with their own calibrator. When `classes is None`, behave
  exactly like the global inner calibrator (so it still works in any class-blind path).
- Prefer a **parametric** inner kind (platt/beta) as the recommended default: per-class
  slices of the single held-out calibration fold are small, which is where isotonic
  overfits worst. `_ParametricCalibrator`'s base-rate fallback (`fusion.py`) already
  covers an all-correct / all-incorrect class slice without raising.

## Persistence
Per-class calibrators persist to a **directory** (same pattern as
`XGBRankerFusionModel`): a small JSON manifest listing which class indices have their
own calibrator + the inner kind, plus one inner-calibrator file per entry and one for
the global. Register with its own `filename` (a dirname) and `load` in the registry —
no `ArtifactRepository` edits.

## Files to add/change
- `text_classifier/infrastructure/calibration.py` (or `fusion.py`) — `PerClassCalibrator`.
- `text_classifier/domain/ports.py` — add `*, classes=None` to `fit`/`transform`.
- `text_classifier/infrastructure/fusion.py` — thread `classes=None` through the three
  existing calibrators (ignored).
- `text_classifier/application/scoring.py` — pass candidate column as `classes`.
- `text_classifier/infrastructure/registry.py` — `register_calibrator("per-class", ...)`.
- `text_classifier/config.py` — document `kind="per-class"`, `inner`, `min_support`.
- `tests/unit/test_calibration.py` — extend.
- `tests/integration/test_e2e.py` — parametrize round-trip over `per-class`.

## Tests
- [ ] Existing three calibrators unchanged: passing no `classes` is identical to today.
- [ ] A class with `>= min_support` skewed reliability gets a better per-class Brier/ECE
      than the global curve; a class below `min_support` exactly equals the global curve.
- [ ] Degenerate: a class slice that is all-correct / all-incorrect → no NaN/inf (base-rate
      fallback exercised); a class unseen at fit time → global curve at transform.
- [ ] save → load round-trip identical within ~1e-6, including the per-class/global split.
- [ ] `build_calibrator(CalibrationConfig(kind="per-class", inner="beta"))` returns the
      right type; unknown `inner` raises the registry `ValueError`.

## Acceptance criteria
- [ ] `kind="per-class"` selects via config alone; no pipeline/`ArtifactRepository` edits
      beyond the registry entry + the one `scoring.py` line.
- [ ] Class-blind paths (`classes=None`) reproduce global behaviour exactly.
- [ ] Torch-free, package-free; round-trips within ~1e-6.

## Out of scope
Automatic `min_support` selection or hierarchical/grouped-by-frequency calibration
(a cheaper middle ground if fully per-class proves too data-starved — separate ticket).
Per-class temperature scaling.
