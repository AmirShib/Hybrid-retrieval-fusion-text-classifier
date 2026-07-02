# T35 — Language-neutral BM25 defaults (stop treating English as the default language)

status: todo
tier: 3
depends_on: —

## Goal
Change `RetrievalConfig.bm25_token_kwargs` default from
`{"stop_words": "english"}` to `{}` (no stopword removal), surface the option in
the train CLI, and document language configuration in the README.

## Why
The default silently applies **English** stopword removal to every corpus
(`config.py:31`). The repo's own Hebrew example (`examples/coicop_hebrew/`) runs
with English stopwords — harmless there by luck (no overlap), but a French or
German user gets BM25 quietly degraded by a wrong-language filter, with no
warning and no CLI knob to fix it (`bm25_token_kwargs` is unreachable from the
CLI today). A language-specific filter should be an explicit opt-in, not a
hidden default. `CountVectorizer`'s default `token_pattern` is already
unicode-aware, so tokenization itself is fine.

## Design
- Default `bm25_token_kwargs` → `{}`.
- Train CLI: `--bm25-stop-words english` (any sklearn-accepted value; `none`
  sentinel maps to omitting the key) so English users can restore the old
  behaviour in one flag until `--config` (T74) lands.
- README: a short "non-English / multilingual corpora" note — stopwords, the
  multilingual encoder choice, and that TF-IDF `params` accept the same kwargs.
- **This is a behaviour change for English corpora**: BM25 scores shift slightly.
  Call it out in the changelog (T64). Retraining is NOT required for existing
  model dirs (the fitted vectorizer is persisted; the default only affects new
  training runs). Re-baseline T52's floors if it has landed.

## Files to change
- `text_classifier/config.py` — the default.
- `text_classifier/cli/train.py` — the flag.
- `README.md` — language note.
- `tests/unit/test_retrieval.py` — default-behaviour assertions.
- `examples/coicop_hebrew/` prep/notebook — drop any now-redundant override.

## Tests
- [ ] Default `BM25Index` keeps stopwords ("the" scores > 0 against a doc
      containing it) and old behaviour is recoverable via the kwarg.
- [ ] CLI flag round-trips into `meta.json`'s config block.
- [ ] A Hebrew/Unicode micro-corpus retrieves correctly under the default.

## Acceptance criteria
- [ ] No hidden language assumptions left in defaults (grep for "english").
- [ ] Existing persisted models unaffected; change is train-time only.

## Out of scope
Language auto-detection; per-language analyzers/stemming (users pass their own
`analyzer`/`tokenizer` via `bm25_token_kwargs` — note the T67 persistence caveat
about callables); encoder multilingualism (that's model choice).
