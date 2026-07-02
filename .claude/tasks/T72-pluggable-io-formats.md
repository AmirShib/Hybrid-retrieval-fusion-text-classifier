# T72 — Pluggable input/output formats behind a `RecordSource` / `RecordSink` port

status: todo
tier: 7
depends_on: T23

## Goal
Let operators read training/inference data from formats other than CSV — Parquet,
JSONL, a SQL database, cloud object storage — and write predictions to the same set,
selected by the URI/extension (with an explicit `--format` override). CSV and JSONL stay
in the torch-free, dependency-light core; everything heavier is an optional extra.

## Why
Ingestion is hardcoded to `pd.read_csv` today (`cli/_common.py::_read_csv`) and output to
`out.to_csv` (`cli/infer.py`). Not everyone exports CSV: Parquet is the analytics default,
many users want to pull straight from a warehouse/DB, and data may live in S3/GCS. Forcing
a CSV round-trip is friction and a data-fidelity risk (dtypes, quoting, encoding).

The seam already exists: **format is the only hardcoded part, and it is isolated to one
file.** Everything past the CLI boundary speaks domain objects (`LabeledItem`,
`LabelSpace`, `List[str]`) — the pipeline has no concept of CSV. So this is a
boundary-generalization, not a core change.

## Design: separate "get a table" from "interpret the table"
`cli/_common.py` currently fuses two jobs. Split them:
1. **bytes -> DataFrame** (`pd.read_csv`) — the *only* format-specific part. Extract behind
   a port.
2. **DataFrame -> domain objects** + friendly column validation (`read_items`,
   `read_label_space`, `read_texts`) — already format-agnostic. Leave **unchanged**; it
   keeps consuming a DataFrame.

### Port
    class RecordSource(ABC):
        def read(self, uri: str, **opts) -> pd.DataFrame: ...
    class RecordSink(ABC):
        def write(self, df: pd.DataFrame, uri: str, **opts) -> None: ...

Register by scheme/extension via the T23 registry, exactly like encoder/fusion/calibrator.
`_read_csv` becomes `_read_frame(uri, kind)` that dispatches; the three `read_*` helpers
and all of their error messages are untouched. `cli/infer.py` writes through a `RecordSink`
keyed off the output extension.

### Backends + dependency tiers
| URI / extension                         | Backend            | Dependency (extra)        |
|-----------------------------------------|--------------------|---------------------------|
| `*.csv`                                 | `pd.read_csv`      | **core** (pandas)         |
| `*.jsonl` / `*.ndjson`                  | `pd.read_json(lines=True)` | **core**          |
| `*.parquet` / `*.feather`               | `pd.read_parquet`  | `pyarrow`  -> `[parquet]`  |
| `sqlite:///...`, `postgresql://...` (+ `--query`/`--table`) | `pd.read_sql` | `SQLAlchemy` + driver -> `[sql]` |
| `s3://...`, `gs://...`                   | any reader via `fsspec` | `s3fs`/`gcsfs` -> `[cloud]` |

## Constraints (each is a test / acceptance item)
1. **Air-gapped core unchanged.** With no extras installed, only CSV/JSONL are available
   and behaviour is byte-for-byte identical to today. A missing extra yields a clear
   "install `text-classifier[parquet]` to read `.parquet`" message — never a raw
   `ImportError` traceback. Mirrors `lightgbm`/torch optionality (T41, T63).
2. **Domain stays framework-free.** DB drivers / pyarrow / fsspec live in
   `infrastructure` behind the port; `domain` imports none of them.
3. **Errors stay at the boundary.** Each backend wraps failures the way `_read_csv` does
   ("could not read parquet file X", "query returned no `text` column"), so the friendly
   single-line error UX is identical across formats.
4. **User-extensible.** A custom `RecordSource` (internal API, proprietary store) registers
   under its own scheme and is reachable via `--input my-scheme://...` with no fork — same
   philosophy as the T70 `FeatureProvider`.
5. **SQL safety.** Go through parameterized `read_sql`; never string-format operator-
   supplied values into a query. Connection strings are trusted operator input.

## Files to add/change
- `text_classifier/infrastructure/io_formats.py` (new) — `RecordSource`/`RecordSink`
  implementations (csv, jsonl, parquet, sql, fsspec).
- `text_classifier/infrastructure/registry.py` — `register_source` / `register_sink`,
  dispatch by scheme then extension.
- `text_classifier/cli/_common.py` — `_read_csv` -> `_read_frame(uri, kind)`; `read_items`
  / `read_label_space` / `read_texts` unchanged (consume the frame).
- `text_classifier/cli/{train,infer,evaluate}.py` — accept any URI; add `--format` /
  `--input-format` / `--output-format` overrides; route output through a sink.
- `pyproject.toml` — `[project.optional-dependencies]` `parquet`, `sql`, `cloud`.
- tests — unit per backend (offline: csv/jsonl/parquet round-trip with a temp file; a
  sqlite round-trip covers the SQL path with no external service); missing-extra message;
  e2e train/infer driven from a non-CSV source.

## Acceptance criteria
- [ ] `--items data.parquet` / `--input rows.jsonl` / `--input "sqlite:///x.db" --table t`
      train and infer end to end.
- [ ] Output format follows the `--output` extension (`preds.parquet`, `preds.jsonl`).
- [ ] Core install (no extras) supports CSV+JSONL and is byte-for-byte unchanged; a
      missing extra gives an install hint, not a traceback.
- [ ] DataFrame->domain mapping and its error messages are untouched.

## Out of scope
A generic connector framework (lazy datasets, schema registry). Streaming/chunked reads —
training loads all items in memory to build indices anyway; chunked *inference* is
**T75** (`--chunksize`). In-memory framework interop (polars / HF datasets / torch
datasets / Arrow interchange objects) is also **T75** — this ticket owns *files and
storage*, T75 owns *objects already in memory*; together they cover "not everyone
hands us a CSV". Writing back to a DB table (read-from-DB, write-to-file is the v1
asymmetry; add a SQL sink later if asked).
