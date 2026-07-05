# T25 — Expose `--encoder-kind` in the train CLI

status: done
tier: 2
depends_on: T23, T24

## Goal
Let `scripts/train.py` select the encoder backend by registry key from the
command line, so the torch-free `tfidf` encoder (and any future backend) is
usable without writing Python. Today `train.py` hardcodes the default
`EncoderConfig.kind="sentence-transformers"`; the only torch-free path (T24's
`TfidfEncoder`) is reachable solely through the library API.

## Why
The project advertises an air-gapped, torch-free option, but a CLI operator
can't reach it. `--encoder` only sets `model_name_or_path`, which is meaningless
for TF-IDF. Adding `--encoder-kind` closes the gap between "the backend exists
and is tested" (T24, done) and "an operator can actually run it." The training
pipeline already auto-fits corpus-dependent encoders per fold
(`_use_per_fold_encoder` consults `encoder_is_corpus_dependent`), so **no
pipeline changes are needed** — this is pure CLI plumbing.

## Scope
- Add `--encoder-kind` to `scripts/train.py`, default `"sentence-transformers"`
  (no behavior change for existing invocations).
- Validate the value against the registry and fail fast on an unknown kind,
  listing the registered kinds (reuse the registry's existing `ValueError` from
  `encoder_spec`, or surface `sorted(_ENCODERS)`).
- Wire it to `cfg.encoder.kind`.
- When `--encoder-kind tfidf`, `--encoder` (model path) is irrelevant; document
  that it is ignored for corpus-fitted encoders (don't error — just note it).

## Optional (nice to have; can be a follow-up)
- `--encoder-param key=value` (repeatable) → parsed into `cfg.encoder.params`
  (e.g. `--encoder-param ngram_range=1,2 --encoder-param max_features=50000`), so
  `TfidfVectorizer` can be tuned from the CLI. If skipped, backend defaults apply.
- Sibling flags `--fusion-kind` / `--calibration-kind` for symmetry — the same
  CLI gap exists for those two registries. Out of scope unless trivial; mention
  in the PR if added.

## Files to change
- `scripts/train.py` — add the argument, validate, set `cfg.encoder.kind`.
- `README.md` — show the torch-free CLI example, e.g.
  `python -m scripts.train --items items.csv --classes classes.csv --out m/ --encoder-kind tfidf`.
- `tests/` — a CLI-level test that `--encoder-kind tfidf` trains end-to-end
  offline and writes a model dir; an unknown kind exits non-zero with a helpful
  message listing the registered kinds.

## Acceptance criteria
- [ ] `python -m scripts.train ... --encoder-kind tfidf` trains and saves a model
      directory with no torch installed.
- [ ] Default invocation (no `--encoder-kind`) is byte-for-byte unchanged.
- [ ] Unknown `--encoder-kind` exits non-zero, listing the registered kinds.
- [ ] README documents the torch-free CLI path.

## Out of scope
New encoder backends (T24 / future tickets). Changing pipeline per-fold logic
(it already handles corpus-dependent encoders). Exposing reranker config in the
CLI (T33).
