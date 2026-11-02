"""Shared CLI helpers: logging setup and friendly CSV ingestion.

The point of these helpers is to turn the two most common operator mistakes —
a wrong/missing column and a malformed value — into a clear, single-line error
at the boundary, instead of a deep pandas/numpy traceback from inside the
pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..application.evaluation import json_safe
from ..config import PipelineConfig
from ..domain import ClassDefinition, LabeledItem, LabelSpace


def configure_logging(level: str = "INFO") -> None:
    """Initialize root logging once, with a timestamped format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def add_logging_arg(parser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )


def add_config_args(parser) -> None:
    """``--config``/``--dump-config``, shared by every CLI that builds a
    ``PipelineConfig``. Precedence is defaults < ``--config`` file < explicit
    flags (flags are applied on top by the caller after ``load_pipeline_config``)."""
    parser.add_argument(
        "--config",
        default=None,
        help="JSON file with a (partial) PipelineConfig; same shape as meta.json's "
        "'config' block. Precedence: built-in defaults < --config < explicit flags.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="print the effective PipelineConfig as JSON and exit (no training/inference)",
    )


def load_pipeline_config(config_path: Optional[str]) -> PipelineConfig:
    """Load a ``PipelineConfig`` from an optional ``--config`` JSON file,
    falling back to all-defaults when none is given. Errors are reported as a
    single-line ``SystemExit`` naming the file and the problem, not a raw
    traceback from deep inside ``PipelineConfig.from_dict``."""
    if config_path is None:
        return PipelineConfig()
    try:
        with open(config_path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"error: config file not found: {config_path!r}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: could not parse config file {config_path!r}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"error: config file {config_path!r} must contain a JSON object")
    try:
        return PipelineConfig.from_dict(data)
    except ValueError as exc:
        raise SystemExit(f"error: invalid config file {config_path!r}: {exc}")


# --------------------------------------------------------------- report output
def write_json_report(path: Optional[str], payload, label: str = "full report") -> None:
    """Write ``payload`` to ``path`` as JSON and print a confirmation; a ``None``
    path writes nothing.

    Every reporting CLI ends with the same three lines (coerce through
    ``json_safe`` because reports carry numpy scalars and NaN, dump indented,
    tell the operator where it went). Centralizing it keeps ``--output``
    behaving identically across ``eval``/``tune``/``importance``/
    ``retrain-ablate`` — including the ``json_safe`` step, which is easy to
    forget and produces a report that is not valid JSON when it is.
    """
    if not path:
        return
    with open(path, "w") as fh:
        json.dump(json_safe(payload), fh, indent=2)
    print(f"\nwrote {label} to {path}")


# ------------------------------------------------------------ number formatting
# Reports carry `None` wherever a metric is undefined (no accepted items, an
# empty class): these render that as "n/a" rather than crashing on a format
# spec, and keep one house style across every CLI's console output.
def pct(x) -> str:
    """A ratio as a 1-decimal percentage (``"94.2%"``), or ``"n/a"``."""
    return "n/a" if x is None else f"{100 * x:.1f}%"


def signed_pct(x) -> str:
    """Like ``pct`` but always signed (``"+1.3%"``) — for deltas, where the
    direction is the whole point."""
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def num(x) -> str:
    """A raw metric at 4 decimals (Brier, ECE), or ``"n/a"``."""
    return "n/a" if x is None else f"{x:.4f}"


def signed_num(x) -> str:
    """A signed raw metric at 4 decimals, ``"n/a"`` for ``None`` *or* NaN —
    an all-NaN ablation arm is an undefined effect, not a zero one."""
    if x is None or x != x:  # NaN != NaN
        return "n/a"
    return f"{x:+.4f}"


def _read_csv(path: str, kind: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"error: {kind} file not found: {path!r}")
    except Exception as exc:  # malformed CSV, encoding, etc.
        raise SystemExit(f"error: could not read {kind} file {path!r}: {exc}")


def _require_columns(df: pd.DataFrame, columns: Sequence[str], path: str, kind: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"error: {kind} file {path!r} is missing required column(s) {missing}; "
            f"found columns: {list(df.columns)}"
        )


# Optional structured taxonomy columns. Scalars map straight through; multi-value
# fields need a separator in CSV (JSONL carries real arrays and needs none).
_CLASS_SCALAR_COLS = ("title", "definition")
_CLASS_TUPLE_COLS = (
    "examples",
    "inclusions",
    "exclusions",
    "parent_path",
    "sibling_distinctions",
)
# Pipe, not semicolon: taxonomy prose is full of semicolons *inside* a single
# inclusion note ("cornflakes; oatmeal; muesli"), so splitting on one would
# shred entries. Pipes essentially never occur in this text.
CLASS_LIST_SEP = "|"


def _cell(value: object) -> str:
    """A CSV cell as clean text; pandas' NaN for a blank cell becomes ""."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _split_cell(value: object) -> Tuple[str, ...]:
    """Split a multi-value CSV cell on ``CLASS_LIST_SEP``, dropping blanks."""
    text = _cell(value)
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(CLASS_LIST_SEP) if part.strip())


def _label_space_from_jsonl(path: str) -> LabelSpace:
    """Read a classes JSONL file: one JSON object per line, arrays as arrays.

    This is the format for a real structured taxonomy — nested lists survive a
    round-trip intact, with no in-cell separator convention to get wrong."""
    defs = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"error: could not parse classes file {path!r} line {lineno}: {exc}"
                    )
                if not isinstance(entry, dict):
                    raise SystemExit(
                        f"error: classes file {path!r} line {lineno} must be a JSON "
                        f"object, got {type(entry).__name__}"
                    )
                if "key" not in entry or "description" not in entry:
                    raise SystemExit(
                        f"error: classes file {path!r} line {lineno} is missing "
                        f"'key' and/or 'description'"
                    )
                kwargs = {c: entry[c] for c in _CLASS_SCALAR_COLS if entry.get(c)}
                kwargs.update({c: tuple(entry[c]) for c in _CLASS_TUPLE_COLS if entry.get(c)})
                try:
                    defs.append(
                        ClassDefinition(str(entry["key"]), str(entry["description"]), **kwargs)
                    )
                except ValueError as exc:
                    raise SystemExit(f"error: invalid class in {path!r} line {lineno}: {exc}")
    except FileNotFoundError:
        raise SystemExit(f"error: classes file not found: {path!r}")
    try:
        return LabelSpace(defs)
    except ValueError as exc:  # empty file, duplicate keys
        raise SystemExit(f"error: invalid classes in {path!r}: {exc}")


def read_label_space(path: str, key_col: str = "key", desc_col: str = "description") -> LabelSpace:
    """Read a classes file into a LabelSpace, with clear column/format errors.

    Two formats, picked by extension: ``.jsonl``/``.ndjson`` for a structured
    taxonomy (real arrays), anything else as CSV.

    A CSV needs only ``key`` and ``description`` — exactly as before. It *may*
    also carry the optional structured columns (``title``, ``definition``,
    ``examples``, ``inclusions``, ``exclusions``, ``parent_path``,
    ``sibling_distinctions``); multi-value ones are ``|``-separated. Absent
    columns simply leave those fields empty, so every existing classes CSV reads
    identically to before."""
    if path.lower().endswith((".jsonl", ".ndjson")):
        return _label_space_from_jsonl(path)

    df = _read_csv(path, "classes")
    _require_columns(df, [key_col, desc_col], path, "classes")
    present_scalars = [c for c in _CLASS_SCALAR_COLS if c in df.columns]
    present_tuples = [c for c in _CLASS_TUPLE_COLS if c in df.columns]
    records = df.to_dict("records")

    def _optional(row: Mapping[str, Any]) -> Dict[str, Any]:
        """Structured fields present in this CSV; absent columns stay unset so
        ClassDefinition applies its own defaults."""
        fields: Dict[str, Any] = {c: _cell(row[c]) for c in present_scalars}
        fields.update({c: _split_cell(row[c]) for c in present_tuples})
        return fields

    try:
        return LabelSpace(
            [
                ClassDefinition(str(row[key_col]), str(row[desc_col]), **_optional(row))
                for row in records
            ]
        )
    except ValueError as exc:  # empty/duplicate keys, empty descriptions
        raise SystemExit(f"error: invalid classes in {path!r}: {exc}")


def read_items(path: str, text_col: str = "text", label_col: str = "label") -> List[LabeledItem]:
    """Read a labeled items CSV into LabeledItems, with clear errors."""
    df = _read_csv(path, "items")
    _require_columns(df, [text_col, label_col], path, "items")
    try:
        return [
            LabeledItem(str(text), str(label))
            for text, label in zip(df[text_col].tolist(), df[label_col].tolist())
        ]
    except ValueError as exc:  # empty text/label
        raise SystemExit(f"error: invalid items in {path!r}: {exc}")


def read_texts(path: str, text_col: str = "text") -> Tuple[pd.DataFrame, List[str]]:
    """Read an inputs CSV and return (full_frame, texts) for inference."""
    df = _read_csv(path, "input")
    _require_columns(df, [text_col], path, "input")
    return df, df[text_col].astype(str).tolist()
