"""Shared CLI helpers: logging setup and friendly CSV ingestion.

The point of these helpers is to turn the two most common operator mistakes —
a wrong/missing column and a malformed value — into a clear, single-line error
at the boundary, instead of a deep pandas/numpy traceback from inside the
pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence, Tuple

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
