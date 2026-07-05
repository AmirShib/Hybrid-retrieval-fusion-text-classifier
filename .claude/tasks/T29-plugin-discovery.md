# T29 — Plugin discovery: third-party backends must load on hosts that didn't train

status: todo
tier: 2
depends_on: T23

## Goal
Make externally-registered components (encoder/fusion/calibrator — and retrieval
signals once T34 lands) discoverable at load time via a setuptools entry-point
group, plus an explicit `--plugin` escape hatch on the CLIs.

## Why
T23's registry is process-local: `register_encoder("my-encoder", ...)` works only
if my module has already been imported. Training in a notebook, that's fine — I
imported it myself. But `ArtifactRepository.load` (`persistence.py`) dispatches by
the `kind` string recorded in `meta.json`, and *nothing on the inference host
imports the plugin module*. Net effect: a model directory trained with a custom
backend raises `unknown encoder kind 'my-encoder'` on any other host/process —
"pluggable" currently means "pluggable within one Python process". For
professional use (train on a workstation, infer on a server) this is a wall.

## Design
- **Entry-point group** `text_classifier.plugins`. A plugin package declares:

      [project.entry-points."text_classifier.plugins"]
      my_backend = "my_pkg.tc_plugin:register"

  where `register()` calls the existing `register_*` functions.
- **Lazy scan on miss**: in `registry._lookup`, on a `KeyError`, load all
  entry points in the group once (`importlib.metadata.entry_points(
  group="text_classifier.plugins")`, py≥3.10 API), then retry; only then raise —
  with a message listing registered kinds *and* discovered plugin names, so a
  half-installed plugin is diagnosable. Lazy (not import-time) keeps `import
  text_classifier` fast and avoids import-order surprises. Guard against a broken
  plugin: catch its import error, warn with the dist name, continue scanning.
- **CLI escape hatch**: `--plugin dotted.module` (repeatable) on train/infer/eval
  (`cli/_common.py`) → `importlib.import_module` before any registry use. Covers
  plugins not packaged with entry points (a single in-house `.py`).
- **Provenance**: when saving, if a component's kind is not a built-in, record the
  providing distribution name+version in `meta.json`'s `components` block (best
  effort via `importlib.metadata.packages_distributions()`); `load` uses it in the
  error message ("kind 'my-encoder' is provided by my-pkg 1.2 — is it installed?").

## Files to change
- `text_classifier/infrastructure/registry.py` — lazy entry-point scan in `_lookup`.
- `text_classifier/infrastructure/persistence.py` — provenance in `components`.
- `text_classifier/cli/_common.py` + the three CLIs — `--plugin`.
- `README.md` — a "writing a plugin" section with the pyproject snippet.
- `tests/unit/test_registry.py`, `tests/integration/test_cli.py` — extend.

## Tests
- [ ] Unit: monkeypatch `importlib.metadata.entry_points` to expose a fake plugin;
      a `kind` unknown before the scan resolves after; a genuinely unknown kind
      raises listing the discovered plugin names.
- [ ] Unit: a plugin whose `register()` raises → warning naming it, other plugins
      still load, original lookup error preserved.
- [ ] Integration: `--plugin` with a module written to `tmp_path` (on
      `sys.path`) registers a dummy encoder; train+infer round-trip through it.
- [ ] Scan happens at most once per process (idempotence guard).

## Acceptance criteria
- [ ] A pip-installed plugin package needs zero imports by the operator: train on
      host A, `text-classifier-infer` on host B just works.
- [ ] No import-time cost when every kind is built-in (scan only on miss).
- [ ] Error messages name what was searched and what was found.

## Out of scope
Sandboxing/trust policy for plugins (a plugin is code you chose to install);
plugin API versioning; auto-installing missing plugins.
