# T23 — Pluggable component registry + factory DI (encoder / fusion / calibrator)

status: done
tier: 2
depends_on: T01, T07

## Goal
Make the **encoder, fusion model, and calibrator selectable by configuration**
instead of hardcoded. Introduce a small registry/factory so a new backend is
wired in by registering it under a name and setting that name in the config —
without editing `TrainingPipeline` or `ArtifactRepository`.

## Why
The architecture is already hexagonal: `TextEncoder`, `FusionModel`, and
`ConfidenceCalibrator` are ports (ABCs) in `domain/ports.py`. But three places
reach around the ports and bind the concrete classes, so you cannot actually run
a different model today:

- `application/training.py`
  - `__init__` annotates `shared_encoder: SentenceTransformerEncoder` (concrete).
  - `_fit_fusion` does `XGBoostFusionModel(...)` and `IsotonicCalibrator()` directly
    (lines ~125/129) — fusion/calibrator are not injected or config-driven at all.
- `infrastructure/persistence.py`
  - `DeployedArtifacts` field types are the concrete classes.
  - `ArtifactRepository.load` hardcodes `XGBoostFusionModel.load(...)`,
    `IsotonicCalibrator.load(...)`, `SentenceTransformerEncoder.load(...)` and the
    on-disk filenames (`fusion.json`, `calibrator.pkl`).

This ticket is the prerequisite that unblocks T24 (alt encoders), T41 (LightGBM
fusion), T42 (alt calibrators), and T31 (FAISS retrieval).

## Files to add/change
- `text_classifier/infrastructure/registry.py` — NEW: name→factory maps
- `text_classifier/config.py` — add a `kind` selector + per-backend params to
  `EncoderConfig`, `FusionConfig`, and a new `CalibrationConfig`
- `text_classifier/application/training.py` — build components via the registry;
  loosen annotations to the ports
- `text_classifier/infrastructure/persistence.py` — store component `kind` in
  `meta.json`; dispatch load through the registry; field types → ports
- `text_classifier/domain/ports.py` — add `kind` classmethod/attr to each port (optional)
- `tests/unit/test_registry.py` — NEW

## Part A — Registry

- [ ] A `register_fusion(name)` / `register_calibrator(name)` / `register_encoder(name)`
      decorator (or explicit `REGISTRY[name] = cls`) populating three dicts.
- [ ] `build_fusion(config) -> FusionModel`, `build_calibrator(config) -> ConfidenceCalibrator`,
      `build_encoder(config) -> TextEncoder` look up `config.<x>.kind` and instantiate
      with that backend's params.
- [ ] Unknown `kind` → `ValueError` listing the registered names (no silent fallback).
- [ ] Built-ins register on import: `"sentence-transformers"`, `"xgboost"`, `"isotonic"`.

## Part B — Config

- [ ] `FusionConfig.kind: str = "xgboost"`; keep `xgb_params` as the xgboost-specific block.
      Generalize so another backend reads its own params block (e.g. `params: Dict`).
- [ ] `EncoderConfig.kind: str = "sentence-transformers"`.
- [ ] New `CalibrationConfig(kind: str = "isotonic", params: Dict = {})`; add to `PipelineConfig`.
- [ ] `to_dict()` / `from_dict()` round-trip the new fields (extend the T07 config test).

## Part C — Training pipeline

- [ ] `TrainingPipeline.__init__(shared_encoder: Optional[TextEncoder] = None)` — port type.
- [ ] `_fit_fusion` obtains `fusion = build_fusion(self.cfg.fusion)` and
      `calibrator = build_calibrator(self.cfg.calibration)` — no concrete class names.
- [ ] Behaviour with the default config is **byte-for-byte unchanged** (xgboost + isotonic):
      the existing T07 determinism test still passes.

## Part D — Persistence

- [ ] `DeployedArtifacts` field types → `TextEncoder`, `FusionModel`, `ConfidenceCalibrator`.
- [ ] `meta.json` gains `components: {encoder, fusion, calibrator}` recording each `kind`.
- [ ] `save` asks each component for its artifact filename/format (or keeps a registry of
      `(kind) -> (filename, save_fn, load_fn)`), so a backend with a different on-disk format
      (e.g. LightGBM `.txt`, a Platt `.json`) round-trips without editing the repository.
- [ ] `load` reads `meta["components"]` and dispatches through the registry.
- [ ] **Back-compat:** a model dir written before this change (no `components` key) loads as
      xgboost/isotonic/sentence-transformers defaults. Add a test with a hand-written legacy
      `meta.json`.

## Tests (`tests/unit/test_registry.py` + extend `tests/integration/test_e2e.py`)

- [ ] Register a trivial in-test `FusionModel` double under a fake name; `build_fusion`
      returns it; an end-to-end train→save→load→predict with that double works offline.
- [ ] Unknown-kind `ValueError` lists registered names.
- [ ] Default-config round-trip is identical to current behaviour (regression guard).
- [ ] `meta.json` records the three component kinds; legacy meta (no `components`) still loads.

## Acceptance criteria
- [ ] Encoder, fusion, and calibrator are all chosen by `PipelineConfig`, end to end.
- [ ] Adding a new backend requires only: implement the port + `register_*` — **no edits to
      `TrainingPipeline` or `ArtifactRepository`.** Prove it with the in-test fusion double.
- [ ] Default behaviour and all 147 existing tests unchanged.
- [ ] Persistence is format-agnostic and back-compatible.

## Out of scope
The actual alternative backends (T24 encoders, T41 LightGBM, T42 calibrators) —
this ticket only builds the seams they plug into.
