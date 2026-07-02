"""Shared CLI helpers: logging setup and friendly CSV ingestion.

The point of these helpers is to turn the two most common operator mistakes —
a wrong/missing column and a malformed value — into a clear, single-line error
at the boundary, instead of a deep pandas/numpy traceback from inside the
pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import pandas as pd

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


def read_label_space(path: str, key_col: str = "key", desc_col: str = "description") -> LabelSpace:
    """Read a classes CSV into a LabelSpace, with clear column/format errors."""
    df = _read_csv(path, "classes")
    _require_columns(df, [key_col, desc_col], path, "classes")
    try:
        return LabelSpace(
            [
                ClassDefinition(str(k), str(d))
                for k, d in zip(df[key_col].tolist(), df[desc_col].tolist())
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
