# T75 — Frame-native API (interchange protocols, iterables) + streaming inference

status: todo
tier: 7
depends_on: T30

## Goal
Meet users where their data lives, **without coupling to their framework**:
`predict_df` / `train_df` convenience APIs that accept pandas *or any dataframe
speaking the interchange protocols* (polars, pyarrow, cuDF, …), entry points that
accept any *iterable* (torch DataLoaders, HF datasets, generators), and a chunked
streaming mode for the infer CLI so a multi-million-row backfill runs in bounded
memory.

## Why
Everything at a workplace is a DataFrame — but not necessarily a *pandas* one;
polars, Arrow tables, HF `datasets`, and torch datasets are all first-class
citizens of modern stacks. The library wants `Sequence[LabeledItem]` and
`Sequence[str]`, so every integration starts with the same boilerplate loop (and
its edge cases: NaN text cells, dtype=object surprises, index alignment of
results). The wrong fix is per-framework integrations (`import polars`,
`import datasets`, `import torch` branches): a treadmill of optional deps and
version skew that would undercut T63's torch-optional core. The right fix is
**protocols**: ship zero new dependencies and accept anything that speaks
(a) `Iterable[str]` / iterable of records, or (b) the dataframe interchange
protocol (`__dataframe__`, pandas ≥ 1.5 consumes it via
`pd.api.interchange.from_dataframe` — already our floor) or the Arrow C stream
(`__arrow_c_stream__`, consumed when pyarrow is present). The framework names
never appear in the codebase; they work anyway.

Meanwhile `cli/infer.py` reads the whole CSV, predicts the whole batch, writes at
the end — `predict` chunks feature assembly internally (`feature_chunk`), but the
CLI's *input* is all-in-memory, so a 10M-row file needs 10M rows of text +
predictions resident. Both gaps are friction, not capability — exactly the kind
of thing that stalls adoption.

## Design
**Input normalization** (one private helper, used by everything below):
`_to_frame(obj)` accepts, in order of preference:
1. a pandas DataFrame — used as-is;
2. anything with `__arrow_c_stream__` (polars ≥ 1.x, pyarrow Table) — via
   pyarrow when installed;
3. anything with `__dataframe__` — via `pd.api.interchange.from_dataframe`;
4. otherwise a clear TypeError naming what was received and what is accepted.
No `import polars` / `import datasets` / `import torch` anywhere; add them only
to the *test* extras to prove the protocols work.

**Iterables at the entry points**: widen `InferencePipeline.predict` (and
`predict_topk`, T65) from `Sequence[str]` to `Iterable[str]` — it already does
`texts = list(texts)`, so this is a type-hint + docs change that makes torch
DataLoaders, HF datasets, and generators work as-is. Same for training item
intake. Document the three-line recipes (polars / HF / torch) in the README.

**Library** (new `text_classifier/application/frames.py`, exported at top level):
- `predict_df(pipeline, df, text_col="text", prefix="") -> pd.DataFrame`:
  `df` is anything `_to_frame` accepts; returns a **pandas copy** with
  `predicted_key`, `top_key`, `confidence`, `abstained`, `margin`,
  `runner_up_key` (T65) columns appended, original index preserved (interchange
  inputs get a fresh range index — document it). Returning pandas keeps the
  output contract single-typed; polars users call `.pipe(pl.from_pandas)` if
  they want back. NaN/non-str cells: raise by default with the offending row
  labels (consistent with `_validate_texts`), or `on_bad_text="abstain"` to
  coerce to the abstain result — the boundary decides, the pipeline stays strict.
- `train_df(items_df, classes_df, config, text_col/label_col/key_col/desc_col,
  output_dir) -> (artifacts, report)`: both frames go through `_to_frame`; thin
  wrapper building `LabeledItem`s / `LabelSpace` with the same validation the
  CLI readers do (`cli/_common.py` — extract, don't duplicate; the CLI readers
  become callers of the same functions).
- No new pandas coupling in `domain/` — these live in `application/`, pandas is
  already a core dependency. File formats (parquet/JSONL/SQL/cloud) stay T72's
  job: this ticket owns objects already in memory, T72 owns storage.

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
- [ ] Protocol inputs: a polars DataFrame (skip if polars absent from the test
      extras) and a pyarrow Table round-trip through `predict_df`/`train_df`
      with results identical to the equivalent pandas frame; an unsupported
      object raises the naming TypeError.
- [ ] Iterables: a generator of strings into `predict` yields the same results
      as the equivalent list; consumed exactly once.
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
counterpart and T72's natural substrate); *native* framework integrations
(polars/HF/torch imports in the package — the protocols make them unnecessary);
returning non-pandas frames; a server (serving recipe is a docs task); parallel
multi-process inference; removing pandas from the internal hot path (T76).
