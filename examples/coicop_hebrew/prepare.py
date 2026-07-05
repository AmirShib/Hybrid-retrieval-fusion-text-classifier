#!/usr/bin/env python
"""Prepare COICOP 2018 classes for the Hebrew item-classification demo.

This downloads the UN Statistics Division's **COICOP 2018 hierarchies & mappings**
workbook (once) and turns one level of the hierarchy into the package's
``classes.csv`` input shape:

    <out>/classes.csv   key,description

``key`` is the COICOP code (e.g. ``01.1.1.4``) and ``description`` is a short,
natural-language gloss built from the official *title* plus the *intro* and
*includes* notes. Richer descriptions sharpen the item<->description retrieval
signal — the only signal available zero-shot — so we fold the includes list in.

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
    p.add_argument("--xlsx", default=os.path.join(here, "build", "coicop_2018.xlsx"),
                   help="path to the workbook (downloaded if absent)")
    p.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4],
                   help="hierarchy depth to emit: 1=division ... 4=subclass (default)")
    p.add_argument("--divisions", default="01",
                   help="comma-separated division codes to keep (e.g. '01,02')")
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
    depth = codes.str.count(r"\.") + 1   # "01"->1, "01.1.1.4"->4

    keep = depth == args.level
    if not args.all:
        wanted = {d.strip() for d in args.divisions.split(",") if d.strip()}
        keep &= codes.str.split(".").str[0].isin(wanted)

    rows = []
    for code, title, intro, includes in zip(
        codes[keep], df.loc[keep, "title"], df.loc[keep, "intro"], df.loc[keep, "includes"]
    ):
        rows.append([code, _describe(title, intro, includes)])

    if not rows:
        raise SystemExit("no classes selected — check --level / --divisions")

    _write_csv(os.path.join(args.out, "classes.csv"), ["key", "description"], rows)
    scope = "all divisions" if args.all else f"division(s) {args.divisions}"
    print(f"\n{len(rows)} COICOP level-{args.level} classes ({scope}).")
    print("Next: open coicop_hebrew_classification.ipynb, or feed classes.csv to the library.")


if __name__ == "__main__":
    main()
