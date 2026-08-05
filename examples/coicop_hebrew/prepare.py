#!/usr/bin/env python
"""Prepare COICOP 2018 classes for the Hebrew item-classification demo.

This downloads the UN Statistics Division's **COICOP 2018 hierarchies & mappings**
workbook (once) and turns one level of the hierarchy into the package's
``classes.csv`` input shape:

    <out>/classes.csv     key,description        (plain, unchanged)
    <out>/classes.jsonl   structured taxonomy    (one JSON object per class)

``key`` is the COICOP code (e.g. ``01.1.1.4``) and ``description`` is a short,
natural-language gloss built from the official *title* plus the *intro* and
*includes* notes. Richer descriptions sharpen the item<->description retrieval
signal — the only signal available zero-shot — so we fold the includes list in.

That fold is lossy, which is why ``classes.jsonl`` is written alongside it. The
workbook ships *title*, *intro*, *includes* and *excludes* as separate fields;
gluing them into one string caps the includes list at 240 characters and dilutes
whatever survives — a four-word product list inside a 200-word definition is a
couple of percent of the embedded text, so a query echoing those four words
matches only weakly. The JSONL file keeps the fields apart (plus ``parent_path``,
derived from the code hierarchy) so each can later be indexed as its own short
retrieval document. Both files are written every run; ``classes.csv`` is
byte-identical to what this script produced before, so nothing downstream of it
changes.

The source workbook is the COICOP 2018 / COICOP 1999 correspondence table, which
ships a ``COICOP 2018`` worksheet carrying the full structure (code, title,
intro, includes, excludes). COICOP has four nested levels — division (``01``),
group (``01.1``), class (``01.1.1``) and subclass (``01.1.1.1``). The subclass
level (the default here) is the granularity grocery items live at.

Because every item in this demo is a grocery product, we keep **division 01**
(food and non-alcoholic beverages) by default. Pass ``--all`` for the whole
classification, or ``--divisions 01,02`` to pick specific divisions.

Usage:
    python prepare.py [--out build] [--level 4] [--divisions 01] [--all]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request

# UN SD COICOP 2018 hierarchies & mappings workbook (the 2018<->1999
# correspondence table; its "COICOP 2018" sheet holds the full structure).
DATA_URL = (
    "https://unstats.un.org/unsd/classifications/Econ/Download/"
    "COICOP2018_COICOP1999_correspondence_table_final.xlsx"
)
SHEET = "COICOP 2018"

# Trailing durability tags on titles, e.g. "Cereals (ND)" -> "Cereals". They mark
# Non-Durable / Semi-Durable / Durable / Services and are noise for embeddings.
_DURABILITY = re.compile(r"\s*\((ND|SD|D|S)\)\s*$")


def _download(dest: str) -> None:
    if os.path.isfile(dest):
        return
    print(f"downloading COICOP 2018 workbook -> {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    # Read the whole body first (urlretrieve's Content-Length check trips behind
    # chunking proxies), then write only once we have the full file.
    with urllib.request.urlopen(DATA_URL, timeout=180) as resp:
        raw = resp.read()
    with open(dest, "wb") as fh:
        fh.write(raw)


def _clean(text: object) -> str:
    """Normalize a cell: drop Excel's literal CRs and bullet markers, collapse WS."""
    s = "" if text is None else str(text)
    if s.lower() == "nan":
        return ""
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    s = s.replace("*", " ")
    return re.sub(r"\s+", " ", s).strip()


def _describe(title: str, intro: str, includes: str) -> str:
    """Build a compact class description from the title and its notes.

    Title carries the label; intro/includes add the concrete products that belong
    to the class (e.g. "cornflakes, oatmeal, muesli"), which is exactly the
    vocabulary an item name is likely to echo. We cap the includes list so a long
    note doesn't drown the title.

    This is the *single-view* rendering, kept for ``classes.csv`` so the existing
    notebooks and the plain ``key,description`` path are unaffected. It is also
    the reason ``classes.jsonl`` exists alongside it: capping and gluing is
    lossy — the cap discards the tail of the includes list outright, and what
    survives is diluted by the definition it is glued to. The structured file
    keeps the fields apart so each can be indexed as its own short document.
    """
    title = _DURABILITY.sub("", _clean(title))
    parts = [title]
    intro = _clean(intro)
    if intro:
        parts.append(intro)
    includes = _clean(includes)
    if includes:
        parts.append("Includes: " + includes[:240])
    return " — ".join(parts)


def _split_note(text: str) -> list:
    """Split a COICOP note into its individual entries.

    These notes are semicolon-delimited lists in the source ("cornflakes;
    oatmeal; muesli"), so the semicolon is the real record separator *here* —
    unrelated to the pipe convention the package's CSV reader uses, since JSONL
    carries these as a real array and needs no in-cell separator at all.
    """
    cleaned = _clean(text)
    if not cleaned:
        return []
    return [part.strip(" .,") for part in cleaned.split(";") if part.strip(" .,")]


def _parent_path(code: str, titles: dict) -> list:
    """Titles of a code's ancestors, root-first: '01.1.1.4' -> 01, 01.1, 01.1.1."""
    segments = code.split(".")
    path = []
    for depth in range(1, len(segments)):
        ancestor = ".".join(segments[:depth])
        title = titles.get(ancestor)
        if title:
            path.append(title)
    return path


def _write_csv(path: str, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {len(rows):>5} rows -> {path}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=os.path.join(here, "build"), help="output directory")
    p.add_argument(
        "--xlsx",
        default=os.path.join(here, "build", "coicop_2018.xlsx"),
        help="path to the workbook (downloaded if absent)",
    )
    p.add_argument(
        "--level",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="hierarchy depth to emit: 1=division ... 4=subclass (default)",
    )
    p.add_argument(
        "--divisions", default="01", help="comma-separated division codes to keep (e.g. '01,02')"
    )
    p.add_argument("--all", action="store_true", help="keep every division")
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - guidance only
        raise SystemExit("prepare.py needs pandas + openpyxl: pip install pandas openpyxl") from exc

    os.makedirs(args.out, exist_ok=True)
    _download(args.xlsx)

    df = pd.read_excel(args.xlsx, sheet_name=SHEET, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    codes = df["code"].astype(str).str.strip()
    depth = codes.str.count(r"\.") + 1  # "01"->1, "01.1.1.4"->4

    keep = depth == args.level
    if not args.all:
        wanted = {d.strip() for d in args.divisions.split(",") if d.strip()}
        keep &= codes.str.split(".").str[0].isin(wanted)

    # Ancestor titles for parent_path: every row, at every level, not just the
    # ones we emit.
    all_titles = {
        str(c).strip(): _DURABILITY.sub("", _clean(t)) for c, t in zip(codes, df["title"])
    }
    has_excludes = "excludes" in df.columns
    excludes_col = df.loc[keep, "excludes"] if has_excludes else [""] * int(keep.sum())

    rows = []
    records = []
    for code, title, intro, includes, excludes in zip(
        codes[keep],
        df.loc[keep, "title"],
        df.loc[keep, "intro"],
        df.loc[keep, "includes"],
        excludes_col,
    ):
        rows.append([code, _describe(title, intro, includes)])
        record = {
            "key": code,
            # Same text as classes.csv: `description` keeps its original meaning
            # as the one text the description signals index today, so a model
            # built from either file retrieves identically until the structured
            # views are actually wired up.
            "description": _describe(title, intro, includes),
            "title": _DURABILITY.sub("", _clean(title)),
            "definition": _clean(intro),
            "inclusions": _split_note(includes),
            "exclusions": _split_note(excludes),
            "parent_path": _parent_path(code, all_titles),
        }
        records.append({k: v for k, v in record.items() if v})

    if not rows:
        raise SystemExit("no classes selected — check --level / --divisions")

    _write_csv(os.path.join(args.out, "classes.csv"), ["key", "description"], rows)

    jsonl_path = os.path.join(args.out, "classes.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>5} rows -> {jsonl_path}")

    scope = "all divisions" if args.all else f"division(s) {args.divisions}"
    print(f"\n{len(rows)} COICOP level-{args.level} classes ({scope}).")

    # Field coverage: how many classes actually carry each structured field. This
    # is what says whether a per-field retrieval view is worth building — a field
    # populated for 10% of classes is a sparse signal, not a broken one, but you
    # should expect a correspondingly small effect from it.
    print("\nstructured field coverage:")
    n = len(records)
    for field in ("title", "definition", "inclusions", "exclusions", "parent_path"):
        filled = sum(1 for r in records if r.get(field))
        print(f"  {field:<14} {filled:>5}/{n}  ({filled / n:5.1%})")
    if not has_excludes:
        print("  (no 'excludes' column in this workbook sheet)")

    print("\nNext: open coicop_hebrew_classification.ipynb, or feed classes.csv")
    print("(plain key,description) / classes.jsonl (structured) to the library.")


if __name__ == "__main__":
    main()
