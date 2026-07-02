# T75 — DataFrame-native API + streaming inference for large files

status: todo
tier: 7
depends_on: T30

## Goal
Meet users where their data lives: `predict_df` / `train_df` convenience APIs
over pandas DataFrames, and a chunked streaming mode for the infer CLI so a
multi-million-row backfill runs in bounded memory.

## Why
Everything at a workplace is a DataFrame; the library wants `Sequence[
LabeledItem]` and `Sequence[str]`, so every integration starts with the same
boilerplate loop (and its edge cases: NaN text cells, dtype=object surprises,
index alignment of results). Meanwhile `cli/infer.py` reads the whole CSV,
predicts the whole batch, writes at the end — `predict` chunks feature assembly
internally (`feature_chunk`), but the CLI's *input* is all-in-memory, so a 10M-row
file needs 10M rows of text + predictions resident. Both gaps are friction, not
capability — exactly the kind of thing that stalls adoption.

## Design
**Library** (new `text_classifier/application/frames.py`, exported at top level):
- `predict_df(pipeline, df, text_col="text", prefix="") -> pd.DataFrame`:
  returns a **copy** of `df` with `predicted_key`, `top_key`, `confidence`,
  `abstained`, `margin`, `runner_up_key` (T65) columns appended, original index
  preserved. NaN/non-str cells: raise by default with the offending row labels
  (consistent with `_validate_texts`), or `on_bad_text="abstain"` to coerce to
  the abstain result — the boundary decides, the pipeline stays strict.
- `train_df(items_df, classes_df, config, text_col/label_col/key_col/desc_col,
  output_dir) -> (artifacts, report)`: thin wrapper building `LabeledItem`s /
  `LabelSpace` with the same validation the CLI readers do (`cli/_common.py` —
  extract, don't duplicate; the CLI readers become callers of the same
  functions).
- No new pandas coupling in `domain/` — these live in `application/`, pandas is
  already a core dependency.

**CLI streaming** (`cli/infer.py`):
- `--chunksize N` (default 0 = current behaviour): `pd.read_csv(...,
  chunksize=N)` → predict per chunk → append to the output CSV (header once).
  Peak memory becomes O(chunksize + model). Progress line per chunk to stderr
  (rows done, accept rate so far).
- Failure semantics: a bad row inside chunk k must not silently truncate output —
  either fail fast (default) or `--on-bad-text abstain` mirrors the library knob.
  Document that the encoder dominates runtime and bigger chunks amortize it.

## Files to change
- `text_classifier/application/frames.py` + `text_classifier/__init__.py`.
- `text_classifier/cli/_common.py` — extract shared readers/validators.
- `text_classifier/cli/infer.py` — `--chunksize`, `--on-bad-text`.
- `README.md` — DataFrame quickstart (this becomes the first snippet most users
  copy).
- `tests/unit/test_frames.py`, `tests/integration/test_cli.py`.

## Tests
- [ ] `predict_df`: result aligns on a non-default index (e.g. shuffled string
      index); input df not mutated; column set exact.
- [ ] Bad-text handling: raise names row labels; `"abstain"` yields abstained
      rows in place.
- [ ] `train_df` == CSV path: same items/label space as `read_items`/`read_label_
      space` on the equivalent file (golden comparison).
- [ ] Streaming: output of `--chunksize 7` on a 25-row file is byte-identical to
      the non-streaming run (predictions are per-row independent — assert it);
      header written once; memory bounded (no full-file list retained — assert
      via reader mock).
- [ ] `--chunksize` + `--top-k` (T65) compose.

## Acceptance criteria
- [ ] A pandas user goes from `df` to predictions in one call with no manual
      loop; a backfill of arbitrary size runs in bounded memory.
- [ ] Default CLI behaviour byte-identical when flags are absent.
- [ ] No logic duplicated between CLI readers and the df API.

## Out of scope
Parquet/JSONL/SQL sources (T72 owns formats — the df API is the in-memory
counterpart and T72's natural substrate); dask/polars; a server (serving recipe
is a docs task); parallel multi-process inference.
