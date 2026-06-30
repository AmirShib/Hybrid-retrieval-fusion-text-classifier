# T70 — Pluggable custom features into the fusion layer (train + inference parity)

status: todo
tier: 7
depends_on: T23, T03

## Goal
Let users contribute additional fusion features through a `FeatureProvider` port,
selected/composed via config, so the fusion model can learn from features beyond the
built-in ~28. The exact same providers must run at inference from the persisted model
dir — no train-only features.

## Why
The five retrieval signals → ~28 features are fixed today. Domain users will want to
add their own (text length, metadata, a domain lexicon hit, an external score). The
registry seam (T23) already makes encoder/fusion/calibrator pluggable; features are the
remaining hardcoded stage. This is the natural next extensibility step.

## The hard part: `FEATURE_NAMES` stops being static
`domain/services.py::FEATURE_NAMES` is declared *the single source of truth for column
order*, hand-edited in three places (`services.py`, `application/features.py`,
`meta.json`). Custom features mean the effective column list is **composed at runtime**
from the active providers and **must be persisted into `meta.json`** so inference
rebuilds the identical order. Getting this wrong makes train and infer silently
disagree on what a column means — the worst possible failure.

Design constraint: the built-in 28 stay a "core" provider so that with **no** custom
providers configured, the schema and outputs are byte-for-byte identical to today.

## The port
    class FeatureProvider(ABC):
        def names(self) -> list[str]: ...                 # stable, unique column names
        def compute(self, ctx) -> dict[str, np.ndarray]:  # name -> (rows,) float, vectorized
        def save(self, path) / load(path)                 # portable artifacts only

`ctx` exposes the same per-(item, candidate) grid `FeatureAssembler` already builds
(query texts, embeddings, candidate `rows`/`cols`, label space) — providers gather over
that grid; **no per-row Python loops on the hot path** (CLAUDE.md convention).

The assembler concatenates core + provider columns; the final ordered name list is what
gets written to `meta.json` and re-read at load.

## Three non-negotiable constraints (each must be a test)
1. **Inference parity / air-gapped.** A provider must run from the persisted model dir
   with no labels and no network. Any train-set-derived state is persisted as a portable
   artifact (stdlib pickle + numpy + json + native formats only).
2. **NaN discipline.** "Provider did not fire for this (item, candidate)" emits `NaN`,
   never a true `0` — XGBoost consumes it as missing (CLAUDE.md invariant).
3. **Leakage.** A provider that consumes training data obeys the out-of-fold rule
   exactly like prototypes/indices: an item never sees itself in state built from its
   own fold. Add it to the leakage regression test (T06).

## Files to add/change
- `text_classifier/domain/ports.py` — `FeatureProvider`.
- `text_classifier/application/features.py` — compose core + providers; emit ordered names.
- `text_classifier/domain/services.py` — `FEATURE_NAMES` becomes the core provider's
  `names()`; document that the *effective* schema is composed and persisted.
- `text_classifier/infrastructure/registry.py` — `register_feature_provider(...)`.
- persistence/`meta.json` — persist the composed, ordered feature name list + provider
  manifest; load rebuilds the same assembler.
- `text_classifier/config.py` — list active provider kinds + params.
- tests — unit (a sample provider: names/compute/NaN/save-load), e2e parity
  (train→save→load→infer with a custom provider, identical column order), leakage (T06).

## Acceptance criteria
- [ ] Zero providers configured → schema + outputs byte-for-byte identical to today.
- [ ] A custom provider's columns reach the fusion model at train AND inference, in the
      identical order, sourced from `meta.json`.
- [ ] Provider artifacts are portable to an air-gapped host.
- [ ] NaN-as-missing and out-of-fold rules hold for provider features (tested).

## Out of scope
A library of built-in extra providers (this ticket is the seam + one sample). Feature
selection/ablation across providers (overlaps T40). Per-provider hyperparameter search.

## Note
This ticket's provider-dependency surface is the forcing function for T71 (DAG
orchestration): once providers can depend on signals and on each other, there is a real
dependency graph to model. Land this first; let it inform T71.
