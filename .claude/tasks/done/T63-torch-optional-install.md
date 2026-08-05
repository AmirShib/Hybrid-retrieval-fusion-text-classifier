# T63 — Torch-optional install via extras

status: done
tier: 6
depends_on: T24, T60

## Goal
Let a user install a fully functional, torch-free package and only pull in
sentence-transformers/torch when they actually want the semantic encoder.

## Why
The package already has a torch-free path (the `tfidf` and `hashing` encoder
kinds), but `sentence-transformers` is a *hard* dependency, so an air-gapped or
lightweight user is still forced to source torch wheels they will never use.

## Scope
- Move `sentence-transformers` out of core `dependencies` into an extra, e.g.
  `pip install text-classifier[sentence-transformers]`. Core stays torch-free
  (numpy/pandas/scipy/scikit-learn/xgboost).
- Make the default encoder kind degrade gracefully: if the default is
  `sentence-transformers` and it is not installed, raise a clear, actionable
  error pointing at the extra (and at the `tfidf`/`hashing` alternatives) instead
  of an ImportError deep in the pipeline.
- Decide the default-experience trade-off: either keep `sentence-transformers`
  as the default kind (and require the extra) or switch the out-of-the-box
  default to `tfidf`. Document the decision in the README.
- Update CI so both the torch-free core and the `[sentence-transformers]` extra
  are exercised.

## Acceptance criteria
- [ ] `pip install text-classifier` installs without torch and can train/infer
      with `--encoder-kind tfidf` (or `hashing`).
- [ ] Selecting the sentence-transformers encoder without the extra gives a clear
      install hint, not a raw ImportError.
- [ ] README documents the core vs. extra install paths.

## Note
This deliberately changes the default install footprint, so it was kept separate
from T60 rather than bundled into the packaging pass.

## Resolution (2026-08-05)
Kept `sentence-transformers` as the default encoder *kind* (best out-of-the-box
quality) but moved the package to an opt-in extra. `EncoderConfig.kind` still
defaults to `"sentence-transformers"`; selecting it without the extra now raises
a clear `ImportError` via `encoder.py::_require_sentence_transformers`, used at
`SentenceTransformerEncoder.load` and `train_encoder` (the two public entry
points sentence_transformers imports are reachable from). `requirements.lock`
regenerated with `--extra sentence-transformers --python-platform
x86_64-unknown-linux-gnu` (the platform flag now pinned explicitly so a refresh
from a non-Linux dev machine can't silently drift the reference platform — see
README). CI: added `core-install-torch-free` and `sentence-transformers-extra`
jobs. Tests: `tests/unit/test_encoder_missing_extra.py`.
