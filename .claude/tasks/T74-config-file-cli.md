# T74 — `--config`: reach the full PipelineConfig from the CLI

status: todo
tier: 7
depends_on: T60

## Goal
A `--config config.json` flag on the CLIs (train first; tune/update as they land)
that loads a full `PipelineConfig`, with precedence *defaults < file < explicit
flags*, plus `--dump-config` to print the effective config. One flag unlocks
every knob the library already has.

## Why
The library exposes everything (`PipelineConfig`: fusion kind + `xgb_params`,
calibration kind, `bm25_token_kwargs`, encoder `params`/`encode_kwargs`, chunk
sizes…); the CLI exposes eight flags (`cli/train.py`). Choosing LightGBM fusion,
beta calibration, TF-IDF n-grams, or (post-T28) E5 prompts is impossible from the
command line — the exact audience the console scripts target (a domain expert on
an air-gapped host who is not writing Python) is locked out of most of the
package. Growing one flag per field is a treadmill; the config object already
round-trips (`to_dict`/`from_dict`) and is persisted per model dir, so the file
format already exists — `meta.json`'s `config` block IS a valid config file.

## Design
- `--config path.json`: parse JSON → `PipelineConfig.from_dict` → then apply only
  the flags the user *explicitly set* on top (argparse: compare against
  sentinel/`parser.get_default`, or switch defaults to `None` and resolve).
  Precedence documented in `--help`.
- **Partial configs allowed**: a file containing only `{"fusion": {"kind":
  "lightgbm"}}` works. `from_dict` currently requires several top-level keys
  (`data["encoder"]` etc., `config.py:89-99`) — relax to `.get` with defaults
  uniformly (it already does this for `calibration`).
- **Unknown keys must error, not vanish**: dataclass `**kwargs` raises a bare
  `TypeError` today; wrap per-section to produce `unknown key 'xgb_parms' in
  section 'fusion'; valid keys: [...]` (typo-proofing is the whole point of
  config files). Run `PipelineConfig.validate()` (T27) after merging.
- `--dump-config`: print the effective merged config as JSON and exit 0 —
  reproducibility (`text-classifier-train --config base.json --folds 3
  --dump-config > run.json`) and debugging precedence.
- README: a complete annotated `config.json` example.

## Files to change
- `text_classifier/config.py` — tolerant/strict `from_dict` (partial sections,
  named unknown-key errors).
- `text_classifier/cli/train.py`, `cli/_common.py` — flag + precedence + dump.
- `README.md` — example file.
- `tests/unit/test_validation.py` (from_dict), `tests/integration/test_cli.py`.

## Tests
- [ ] File-only, flag-only, and file+flag (flag wins) each produce the expected
      `PipelineConfig`; `--dump-config` output re-loads to the same config
      (round-trip property).
- [ ] Partial file (one section) merges over defaults.
- [ ] Unknown key → error naming the key, the section, and valid keys.
- [ ] `meta.json`'s `config` block from a trained dir is directly usable as
      `--config` input (train-from-a-previous-run's-config workflow).
- [ ] e2e: `--config` selecting `fusion.kind=lightgbm` (skip if absent) trains
      and round-trips.

## Acceptance criteria
- [ ] Every `PipelineConfig` field is reachable without writing Python.
- [ ] Precedence is deterministic, documented, and covered by tests.
- [ ] Existing flag-only invocations behave byte-identically.

## Out of scope
YAML/TOML support (JSON only — it's what we persist); config schema files;
environment-variable interpolation; multi-run sweep configs (a T40 concern).
